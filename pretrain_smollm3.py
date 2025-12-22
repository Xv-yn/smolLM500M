import argparse
import os
from itertools import chain

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler


def pack_ids(list_of_id_lists, seq_len: int, eos_id: int | None):
    """
    Concatenate token lists and chunk into fixed seq_len blocks.
    Adds EOS between docs if eos_id is provided.
    """
    flat = []
    for ids in list_of_id_lists:
        flat.extend(ids)
        if eos_id is not None:
            flat.append(eos_id)

    total = len(flat)
    if total < seq_len:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    n = total // seq_len
    blocks = [flat[i * seq_len : (i + 1) * seq_len] for i in range(n)]
    return {
        "input_ids": blocks,
        "labels": [b[:] for b in blocks],
        "attention_mask": [[1] * seq_len for _ in range(n)],
    }


def collate(examples):
    input_ids = torch.tensor([ex["input_ids"] for ex in examples], dtype=torch.long)
    labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
    attention_mask = torch.tensor(
        [ex["attention_mask"] for ex in examples], dtype=torch.long
    )
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--parquet_path", type=str, required=True)
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
    ap.add_argument("--tokenize_batch_size", type=int, default=256)
    args = ap.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    print("\n=== Token IDs ===")
    print("tokenizer.bos_token_id:", tokenizer.bos_token_id)
    print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
    print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
    print("=================\n")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    ds = load_dataset("parquet", data_files=args.parquet_path, split="train")
    if "text" not in ds.column_names:
        raise ValueError(
            f"Expected a 'text' column in parquet. Found: {ds.column_names}"
        )

    # Tokenize from text (IGNORE any existing input_ids)
    def tok(batch):
        out = tokenizer(
            batch["text"],
            add_special_tokens=False,  # important for pretraining
            truncation=False,
        )
        return {"input_ids": out["input_ids"]}

    ds_tok = ds.map(
        tok,
        batched=True,
        batch_size=args.tokenize_batch_size,
        remove_columns=[c for c in ds.column_names if c != "text"],
        desc="Tokenizing from text with SmolLM3 tokenizer",
    )

    # Pack to fixed seq_len
    def pack_map(batch):
        return pack_ids(batch["input_ids"], args.seq_len, tokenizer.eos_token_id)

    packed = ds_tok.map(
        pack_map,
        batched=True,
        remove_columns=ds_tok.column_names,
        desc=f"Packing to seq_len={args.seq_len}",
    )
    packed = packed.filter(lambda x: len(x["input_ids"]) == args.seq_len)

    train_loader = DataLoader(
        packed,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_train_steps,
    )

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
