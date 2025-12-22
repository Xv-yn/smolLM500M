"""
Usage:

accelerate launch --num_processes 2 pretrain_smollm3.py \
  --model_dir . \
  --packed_dir ./data/packed_seq2048 \
  --output_dir ./runs/poc_packed \
  --seq_len 2048 \
  --micro_batch_size 1 \
  --grad_accum_steps 8 \
  --learning_rate 2e-4 \
  --num_train_steps 2000 \
  --warmup_steps 200 \
  --save_every 500 \
  --log_every 10
"""

import argparse
import os

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def collate_packed(examples):
    """
    Packed dataset is already fixed-length (seq_len). Just stack into tensors.
    Supports datasets that either have attention_mask or not.
    """
    input_ids = torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long)
    labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)

    if "attention_mask" in examples[0]:
        attention_mask = torch.tensor(
            [ex["attention_mask"] for ex in examples], dtype=torch.long
        )
    else:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument(
        "--packed_dir",
        type=str,
        required=True,
        help="Path created by datasets.save_to_disk()",
    )
    ap.add_argument("--output_dir", type=str, default="./outputs")
    ap.add_argument("--seq_len", type=int, default=2048)
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

    # --- load packed dataset
    packed = load_from_disk(args.packed_dir)

    # basic sanity checks (fail early with clear errors)
    needed = {"input_ids", "labels"}
    missing = needed - set(packed.column_names)
    if missing:
        raise ValueError(
            f"Packed dataset missing columns: {missing}. Found: {packed.column_names}\n"
            "Your packed builder should save fixed-length 'input_ids' and 'labels'."
        )

    # Optional: verify length of first example matches seq_len
    ex0 = packed[0]
    if len(ex0["input_ids"]) != args.seq_len or len(ex0["labels"]) != args.seq_len:
        raise ValueError(
            f"Packed dataset seq_len mismatch. "
            f"Expected {args.seq_len}, got input_ids={len(ex0['input_ids'])}, labels={len(ex0['labels'])}.\n"
            "Rebuild packed dataset with matching --seq_len."
        )

    train_loader = DataLoader(
        packed,
        batch_size=args.micro_batch_size,
        shuffle=True,
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
