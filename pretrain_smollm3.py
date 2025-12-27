import argparse
import glob
import os
import random
import time

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
                    attention_mask = (input_ids != self.pad_id).long()

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

    # IMPORTANT: now this means OPTIMIZER steps (weight updates)
    ap.add_argument("--num_train_steps", type=int, default=2000)

    # warmup in OPTIMIZER steps too
    ap.add_argument("--warmup_steps", type=int, default=200)

    ap.add_argument("--log_every", type=int, default=10)  # in OPT steps
    ap.add_argument("--save_every", type=int, default=500)  # in OPT steps
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--shuffle", action="store_true")

    ap.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"]
    )
    ap.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["auto", "sdpa", "flash_attention_2"],
    )
    ap.add_argument(
        "--compile", action="store_true", help="torch.compile the model (PyTorch 2.x)"
    )

    args = ap.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
    )
    torch.manual_seed(args.seed)

    # Speed knobs (safe)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Ensure PAD is distinct from EOS (prevents empty-loss batches if you mask pad)
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

    accelerator.wait_for_everyone()

    # --- model
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)

    model_kwargs = dict(trust_remote_code=True)
    if args.attn_impl != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl

    model = AutoModelForCausalLM.from_config(config, **model_kwargs)

    # if we added PAD, resize embeddings
    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # important for training speed/memory
    model.config.use_cache = False

    if accelerator.is_main_process:
        print("model vocab_size:", model.config.vocab_size)
        print("tokenizer vocab_size:", len(tokenizer))
        print("attn_impl:", args.attn_impl)
        print("mixed_precision:", args.mixed_precision)

    # Optional: torch.compile (can help, sometimes hurts)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

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

    # --- optimizer (fused if available)
    adamw_kwargs = dict(
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )
    try:
        optimizer = torch.optim.AdamW(model.parameters(), fused=True, **adamw_kwargs)
        if accelerator.is_main_process:
            print("Using fused AdamW")
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
        if accelerator.is_main_process:
            print("Using standard AdamW (fused not available)")

    # --- scheduler (OPTIMIZER steps)
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

    # IMPORTANT: after prepare(), model may be DDP-wrapped and won't expose .config
    base_model = accelerator.unwrap_model(model)
    vocab_size = base_model.config.vocab_size

    # ---- training loop
    data_iter = iter(train_loader)

    micro_step = 0
    opt_step = 0

    # logging for throughput
    t0 = time.time()
    last_log_t = t0
    last_log_micro = 0

    tokens_per_micro = args.micro_batch_size * args.seq_len
    tokens_per_opt = tokens_per_micro * args.grad_accum_steps

    while opt_step < args.num_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        micro_step += 1

        # token id sanity (DDP-safe)
        ids = batch["input_ids"]
        if ids.min().item() < 0 or ids.max().item() >= vocab_size:
            if accelerator.is_main_process:
                print("vocab_size:", vocab_size)
                print("min token:", ids.min().item())
                print("max token:", ids.max().item())
            raise RuntimeError("Found out-of-range token id")

        # skip empty-supervision batches (all -100)
        if (batch["labels"] != -100).sum().item() == 0:
            continue

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                if accelerator.is_main_process:
                    am = batch["attention_mask"]
                    valid = (batch["labels"] != -100).sum().item()
                    print(
                        f"\nNaN/Inf loss at micro_step {micro_step} opt_step {opt_step}"
                    )
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
                opt_step += 1

                # log on OPT steps
                if accelerator.is_main_process and (opt_step % args.log_every == 0):
                    now = time.time()
                    dt = now - last_log_t
                    dmicro = micro_step - last_log_micro
                    tok = dmicro * tokens_per_micro
                    tps = tok / max(dt, 1e-9)

                    lr = lr_scheduler.get_last_lr()[0]
                    remaining = args.num_train_steps - opt_step
                    eta_sec = remaining * (dt / max(args.log_every, 1e-9))
                    eta_hr = eta_sec / 3600.0

                    print(
                        f"opt {opt_step:7d}/{args.num_train_steps} | "
                        f"micro {micro_step:9d} | "
                        f"loss {loss.item():.4f} | lr {lr:.3e} | "
                        f"{tps:,.0f} tok/s | "
                        f"ETA {eta_hr:.2f} h"
                    )

                    last_log_t = now
                    last_log_micro = micro_step

                # save on OPT steps
                if accelerator.is_main_process and (opt_step % args.save_every == 0):
                    ckpt = os.path.join(args.output_dir, f"ckpt-opt-{opt_step}")
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
