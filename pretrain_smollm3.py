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

import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ParquetPackedIterable(IterableDataset):
    def __init__(self, packed_dir: str, seq_len: int, shuffle_shards: bool, seed: int):
        super().__init__()
        self.packed_dir = packed_dir
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards
        self.seed = seed

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

        # Split shards across ranks (rank 0 gets files 0, world, 2world...)
        files = files[rank::world]

        for path in files:
            pf = pq.ParquetFile(path)

            # Iterate row groups sequentially
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=["input_ids"])
                col = table["input_ids"]

                # Iterate rows sequentially
                for i in range(table.num_rows):
                    ids = col[i].as_py()
                    if len(ids) != self.seq_len:
                        continue

                    input_ids = torch.tensor(ids, dtype=torch.long)

                    labels = input_ids.clone()
                    labels[labels == tokenizer.pad_token_id] = (
                        -100
                    )  # ignore pad/eos in loss

                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": torch.ones_like(input_ids),
                    }


def collate_packed(examples):
    input_ids = torch.stack([ex["input_ids"] for ex in examples], dim=0)
    attention_mask = torch.stack([ex["attention_mask"] for ex in examples], dim=0)

    labels = input_ids.clone()
    # We'll set tokenizer.pad_token_id later by closure or global; see below
    labels[labels == PAD_ID] = -100

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

    global PAD_ID
    PAD_ID = tokenizer.pad_token_id

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

    if accelerator.is_main_process:
        print("model vocab_size:", model.config.vocab_size)
        print("tokenizer vocab_size:", len(tokenizer))
    # --- load packed parquet shards
    packed = ParquetPackedIterable(
        args.packed_dir, args.seq_len, args.shuffle, args.seed
    )

    train_loader = DataLoader(
        packed,
        batch_size=args.micro_batch_size,
        shuffle=False,  # MUST be False for IterableDataset
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
        # DEBUG: check token id range before forward
        vocab = model.config.vocab_size
        ids = batch["input_ids"]
        if (ids < 0).any() or (ids >= vocab).any():
            print("vocab_size:", vocab)
            print("min token:", ids.min().item())
            print("max token:", ids.max().item())
            # optionally print a few offending values
            bad = ids[(ids < 0) | (ids >= vocab)]
            print("some bad ids:", bad[:20].tolist())
            raise RuntimeError("Found out-of-range token id")

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if accelerator.is_main_process:
                    bad_params = []
                    for n, p in model.named_parameters():
                        if (
                            p is not None
                            and p.data is not None
                            and (torch.isnan(p.data).any() or torch.isinf(p.data).any())
                        ):
                            bad_params.append(n)
                            break
                    if bad_params:
                        raise RuntimeError(
                            f"NaN/Inf appeared in params after step {step}: {bad_params[0]}"
                        )

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
