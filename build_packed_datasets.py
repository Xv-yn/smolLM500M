"""
Usage:

python build_packed_dataset.py \
  --model_dir . \
  --input_parquet ./data/test_pretraining.parquet \
  --out_dir ./data/packed_seq2048 \
  --seq_len 2048 \
  --num_proc 4

"""

import argparse
import os
from itertools import chain

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--input_parquet", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--num_proc", type=int, default=4)
    ap.add_argument("--tokenize_batch_size", type=int, default=512)
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    eos_id = tok.eos_token_id

    print(
        "Tokenizer IDs:",
        {"bos": tok.bos_token_id, "eos": tok.eos_token_id, "pad": tok.pad_token_id},
    )

    ds = load_dataset("parquet", data_files=args.input_parquet, split="train")
    if "text" not in ds.column_names:
        raise ValueError(f"Expected 'text' column. Found: {ds.column_names}")

    def tokenize_fn(batch):
        out = tok(batch["text"], add_special_tokens=False, truncation=False)
        return {"input_ids": out["input_ids"]}

    ds_tok = ds.map(
        tokenize_fn,
        batched=True,
        batch_size=args.tokenize_batch_size,
        num_proc=args.num_proc,
        remove_columns=[c for c in ds.column_names if c != "text"],
        desc="Tokenizing",
    )

    def pack_fn(batch):
        # batch["input_ids"] is list[list[int]]
        flat = []
        for ids in batch["input_ids"]:
            flat.extend(ids)
            if eos_id is not None:
                flat.append(eos_id)

        total = len(flat)
        if total < args.seq_len:
            return {"input_ids": [], "labels": []}

        n = total // args.seq_len
        blocks = [flat[i * args.seq_len : (i + 1) * args.seq_len] for i in range(n)]
        return {"input_ids": blocks, "labels": [b[:] for b in blocks]}

    packed = ds_tok.map(
        pack_fn,
        batched=True,
        batch_size=64,
        num_proc=args.num_proc,
        remove_columns=ds_tok.column_names,
        desc=f"Packing to {args.seq_len}",
    )

    packed = packed.filter(
        lambda x: len(x["input_ids"]) == args.seq_len, num_proc=args.num_proc
    )

    os.makedirs(args.out_dir, exist_ok=True)
    packed.save_to_disk(args.out_dir)
    print(f"Saved packed dataset to: {args.out_dir}")
    print("Columns:", packed.column_names, "| Examples:", len(packed))


if __name__ == "__main__":
    main()
