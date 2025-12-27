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
"""

import argparse
import glob
import os
import random

import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ParquetPackedIterable(IterableDataset):
    def __init__(
        self,
        packed_dir: str,
        seq_len: int,
        shuffle_shards: bool,
        seed: int,
        pad_id: int,
    ):
        super().__init__()
        self.packed_dir = packed_dir
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.pad_id = pad_id

        self.files = sorted(glob.glob(os.path.join(packed_dir, "shard-*.parquet")))
        if not self.files:
            raise FileNotFoundError(f"No shard-*.parquet in {packed_dir}")

    def __iter__(self):
        # DDP info (works with Accelerate too)
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world = dist.get_world_size()
            else:
                rank, world = 0, 1
        except Exception:
            rank, world = 0, 1

        files = self.files[:]

        # Cheap shuffle: shuffle shard order, not rows
        if self.shuffle_shards:
            rng = random.Random(self.seed + rank)
            rng.shuffle(files)

        # Split shards across ranks
        files = files[rank::world]

        for path in files:
            pf = pq.ParquetFile(path)

            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=["input_ids"])
                col = table["input_ids"]

                for i in range(table.num_rows):
                    ids = col[i].as_py()
                    if len(ids) != self.seq_len:
                        continue

                    input_ids = torch.tensor(ids, dtype=torch.long)

                    # Build attention_mask from pad_id (right or full padding OK)
                    attention_mask = (input_ids != self.pad_id).long()

                    # Labels: ignore padding positions ONLY
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100

                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": attention_mask,
                    }


def collate_packed(examples):
    input_ids = torch.stack([ex["input_ids"] for ex in examples], dim=0)
    attention_mask = torch.stack([ex["attention_mask"] for ex in examples], dim=0)
    labels = torch.stack([ex["labels"] for ex in examples], dim=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--packed_dir", type=str, required=True)
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
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"]
    )
    args = ap.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
    )
    torch.manual_seed(args.seed)

    # --- tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # IMPORTANT: ensure PAD is distinct from EOS to avoid "ignore all eos => empty loss"
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    if accelerator.is_main_process:
        print("\n=== Token IDs ===")
        print("tokenizer.bos_token_id:", tokenizer.bos_token_id)
        print("tokenizer.eos_token_id:", eos_id)
        print("tokenizer.pad_token_id:", pad_id)
        print("=================\n")

        if eos_id is not None and pad_id == eos_id:
            print(
                "WARNING: pad_token_id == eos_token_id. This can cause NaN if you mask PAD in labels."
            )
            print(
                "We attempted to add a real PAD token; if this still matches, your tokenizer may forbid it.\n"
            )

    accelerator.wait_for_everyone()

    # --- model
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Make sure embeddings match tokenizer if we added PAD
    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    if accelerator.is_main_process:
        print("model vocab_size:", model.config.vocab_size)
        print("tokenizer vocab_size:", len(tokenizer))

    # --- data
    packed = ParquetPackedIterable(
        args.packed_dir, args.seq_len, args.shuffle, args.seed, pad_id=pad_id
    )

    train_loader = DataLoader(
        packed,
        batch_size=args.micro_batch_size,
        shuffle=False,
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
        eps=1e-6,
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

        # Guard 1: token id range sanity
        vocab = model.config.vocab_size
        ids = batch["input_ids"]
        if (ids < 0).any() or (ids >= vocab).any():
            if accelerator.is_main_process:
                print("vocab_size:", vocab)
                print("min token:", ids.min().item())
                print("max token:", ids.max().item())
                bad = ids[(ids < 0) | (ids >= vocab)]
                print("some bad ids:", bad[:20].tolist())
            raise RuntimeError("Found out-of-range token id")

        # Guard 2: skip batches that have zero supervised tokens (all labels == -100)
        if (batch["labels"] != -100).sum().item() == 0:
            if accelerator.is_main_process and step % args.log_every == 0:
                print(f"step {step:6d} | skipped batch (all labels ignored)")
            continue

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss

            # Guard 3: catch NaN early and print batch diagnostics
            if torch.isnan(loss) or torch.isinf(loss):
                if accelerator.is_main_process:
                    am = batch["attention_mask"]
                    valid = (batch["labels"] != -100).sum().item()
                    print(f"\nNaN/Inf loss at step {step}")
                    print("valid label tokens:", valid)
                    print("attention_mask sum:", am.sum().item(), "of", am.numel())
                    print("input_ids min/max:", ids.min().item(), ids.max().item())
                raise RuntimeError("NaN/Inf loss encountered")

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

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
