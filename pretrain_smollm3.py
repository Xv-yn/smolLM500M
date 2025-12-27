"""
Usage (Parquet packed shards):

accelerate launch --num_processes 2 pretrain_smollm3.py \
  --model_dir . \
  --packed_dir ./packed/seq4096 \
  --output_dir ./runs/poc_packed \
  --seq_len 4096 \
  --micro_batch_size 1 \
  --grad_accum_steps 8 \
  --learning_rate 2e-4 \
  --num_train_steps 2000 \
  --warmup_steps 200 \
  --save_every 500 \
  --log_every 10

Notes:
- build_packed_shards.py writes shard-*.parquet (NOT datasets.save_to_disk()).
- This script reads those parquet shards directly.
"""

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ParquetPackedDataset(Dataset):
    """
    Efficient reader for shard-*.parquet produced by build_packed_shards.py.

    Strategy:
      - index row groups across all shards
      - on __getitem__, locate (file, row_group, row_in_group)
      - cache the decoded row group's input_ids in memory
    """

    def __init__(self, packed_dir: str, seq_len: int, validate_first_n: int = 8):
        self.packed_dir = packed_dir
        self.seq_len = seq_len

        self.files = sorted(glob.glob(os.path.join(packed_dir, "shard-*.parquet")))
        if not self.files:
            raise FileNotFoundError(
                f"No shard-*.parquet found in {packed_dir}. "
                "Point --packed_dir at the folder created by build_packed_shards.py."
            )

        # Build an index over ALL row groups across ALL files.
        # Each entry: (file_idx, row_group_idx, start_row_global, num_rows_in_group)
        self._rg_index: List[Tuple[int, int, int, int]] = []
        self._rg_starts: List[int] = []  # start_row_global for bisect
        total_rows = 0

        self._pfiles = []  # ParquetFile objects (one per shard)
        for fi, path in enumerate(self.files):
            pf = pq.ParquetFile(path)
            self._pfiles.append(pf)

            for rgi in range(pf.num_row_groups):
                rg_rows = pf.metadata.row_group(rgi).num_rows
                self._rg_index.append((fi, rgi, total_rows, rg_rows))
                self._rg_starts.append(total_rows)
                total_rows += rg_rows

        self._total_rows = total_rows

        # Per-worker cache (each worker has its own Dataset instance)
        self._cache_key = None  # (file_idx, row_group_idx)
        self._cache_ids = None  # list[list[int]] or np array

        # Quick sanity check: validate a few rows from the first row group
        if validate_first_n > 0:
            fi0, rg0, _, rg_rows0 = self._rg_index[0]
            table = self._pfiles[fi0].read_row_group(rg0, columns=["input_ids"])
            col = table["input_ids"]
            ncheck = min(validate_first_n, rg_rows0)
            for i in range(ncheck):
                ids = col[i].as_py()
                if len(ids) != self.seq_len:
                    raise ValueError(
                        f"Packed shard seq_len mismatch in {self.files[fi0]} row_group={rg0} row={i}. "
                        f"Expected {self.seq_len}, got {len(ids)}. "
                        "Launch with matching --seq_len or rebuild shards."
                    )

    def __len__(self):
        return self._total_rows

    def _locate_row_group(self, idx: int):
        # Find the row-group containing global row idx
        rg_pos = bisect.bisect_right(self._rg_starts, idx) - 1
        if rg_pos < 0:
            rg_pos = 0
        fi, rgi, start, nrows = self._rg_index[rg_pos]
        row_in_group = idx - start
        return rg_pos, fi, rgi, row_in_group

    def _load_row_group_into_cache(self, fi: int, rgi: int):
        table = self._pfiles[fi].read_row_group(rgi, columns=["input_ids"])
        col = table["input_ids"]

        # Materialize once. This is the expensive step, so we cache it.
        # Keep as Python lists (fast enough) or convert to numpy object array.
        # We'll keep Python lists for simplicity.
        ids_list = [
            col[i].as_py() for i in range(col.num_chunks and len(col) or table.num_rows)
        ]
        # NOTE: col may be ChunkedArray; easiest is iterate by table rows:
        if len(ids_list) == 0:
            # fallback safe path
            ids_list = [table["input_ids"][i].as_py() for i in range(table.num_rows)]

        self._cache_key = (fi, rgi)
        self._cache_ids = ids_list

    def __getitem__(self, idx: int):
        rg_pos, fi, rgi, row_in_group = self._locate_row_group(idx)

        if self._cache_key != (fi, rgi):
            self._load_row_group_into_cache(fi, rgi)

        ids = self._cache_ids[row_in_group]

        # Safety check (can comment out once stable)
        if len(ids) != self.seq_len:
            raise ValueError(
                f"Packed shard seq_len mismatch in {self.files[fi]} row_group={rgi} row={row_in_group}. "
                f"Expected {self.seq_len}, got {len(ids)}."
            )

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def collate_packed(examples):
    # Examples already contain tensors of shape [seq_len]; just stack.
    input_ids = torch.stack([ex["input_ids"] for ex in examples], dim=0)
    labels = torch.stack([ex["labels"] for ex in examples], dim=0)
    attention_mask = torch.stack([ex["attention_mask"] for ex in examples], dim=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument(
        "--packed_dir",
        type=str,
        required=True,
        help="Folder containing shard-*.parquet produced by build_packed_shards.py",
    )
    ap.add_argument("--output_dir", type=str, default="./outputs")
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--micro_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--num_train_steps", type=int, default=2000)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle packed blocks.")
    args = ap.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    torch.manual_seed(args.seed)

    # --- tokenizer (only needed for saving + generation utils)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        print("\n=== Token IDs ===")
        print("tokenizer.bos_token_id:", tokenizer.bos_token_id)
        print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
        print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
        print("=================\n")

    accelerator.wait_for_everyone()

    # --- model
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # --- load packed parquet shards
    packed = ParquetPackedDataset(args.packed_dir, args.seq_len)

    train_loader = DataLoader(
        packed,
        batch_size=args.micro_batch_size,
        shuffle=args.shuffle,
        collate_fn=collate_packed,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # --- optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # --- scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_train_steps,
    )

    # --- accelerate prepare
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    model.train()
    os.makedirs(args.output_dir, exist_ok=True)

    data_iter = iter(train_loader)

    for step in range(1, args.num_train_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            loss = model(**batch).loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.is_main_process and step % args.log_every == 0:
            lr = lr_scheduler.get_last_lr()[0]
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.6e}")

        if accelerator.is_main_process and step % args.save_every == 0:
            ckpt = os.path.join(args.output_dir, f"ckpt-step-{step}")
            os.makedirs(ckpt, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(ckpt, save_function=accelerator.save)
            tokenizer.save_pretrained(ckpt)
            print(f"Saved checkpoint: {ckpt}")

    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_dir)
        print(f"Saved final model: {final_dir}")


if __name__ == "__main__":
    main()
