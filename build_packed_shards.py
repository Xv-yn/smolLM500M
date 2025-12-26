"""
export SOURCES='[
  {"name":"fineweb_edu","dataset":"HuggingFaceFW/fineweb-edu","config":null,"text_col":"text","weight":0.88,"token_cap":17600000000},
  {"name":"dclm","dataset":"mlfoundations/dclm-baseline-1.0","config":null,"text_col":"text","weight":0.07,"token_cap":1400000000},
  {"name":"stackexchange","dataset":"allenai/dolmino-mix-1124","config":"stackexchange","text_col":"text","weight":0.03,"token_cap":600000000},
  {"name":"wiki","dataset":"allenai/dolmino-mix-1124","config":"wiki","text_col":"text","weight":0.01,"token_cap":200000000},
  {"name":"code","dataset":"bigcode/the-stack-v2","config":"Python","text_col":"content","weight":0.01,"token_cap":200000000}
]'


python build_packed_shards.py \
  --model_dir . \
  --out_dir ./packed/seq4096 \
  --seq_len 4096 \
  --total_tokens 20000000000 \
  --blocks_per_shard 8192 \
  --shuffle_buffer 100000 \
  --tokenize_batch_size 256 \
  --seed 1234 \
  --sources_json "$SOURCES"

"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class SourceSpec:
    name: str
    # HF dataset id, e.g. "HuggingFaceFW/fineweb-edu"
    dataset: str
    # Optional config name if dataset has configs
    config: Optional[str]
    # column containing text
    text_col: str
    # sampling weight (relative)
    weight: float
    # token cap for this source (None means no cap)
    token_cap: Optional[int]


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def weighted_choice(rng, items: List[SourceSpec]) -> SourceSpec:
    total = sum(s.weight for s in items)
    r = rng.random() * total
    acc = 0.0
    for s in items:
        acc += s.weight
        if r <= acc:
            return s
    return items[-1]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument(
        "--total_tokens",
        type=int,
        required=True,
        help="Stop after writing this many tokens total",
    )
    ap.add_argument("--blocks_per_shard", type=int, default=8192)

    ap.add_argument("--shuffle_buffer", type=int, default=100_000)
    ap.add_argument("--tokenize_batch_size", type=int, default=256)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action="store_true")

    # You can keep this JSON in a separate file if you prefer, but inline is easiest initially.
    ap.add_argument(
        "--sources_json",
        type=str,
        required=True,
        help="JSON list of sources: [{name,dataset,config,text_col,weight,token_cap}, ...]",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    state_path = os.path.join(args.out_dir, "state.json")
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")

    tok = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, fix_mistral_regex=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id

    # ---- load sources
    src_dicts = json.loads(args.sources_json)
    sources = [SourceSpec(**d) for d in src_dicts]

    # token caps sanity
    for s in sources:
        if s.token_cap is not None and s.token_cap <= 0:
            raise ValueError(f"Bad token_cap for {s.name}: {s.token_cap}")

    # ---- resume state
    import random

    rng = random.Random(args.seed)

    if args.resume and os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        shard_idx = state["shard_idx"]
        total_tokens_written = state["total_tokens_written"]
        per_source_tokens = state["per_source_tokens"]
        carry = state["carry"]  # leftover tokens buffer
        rng.setstate(tuple(state["rng_state"]))
        print(f"[resume] shard_idx={shard_idx} total_tokens={total_tokens_written}")
    else:
        shard_idx = 0
        total_tokens_written = 0
        per_source_tokens = {s.name: 0 for s in sources}
        carry = []
        # store rng state in a JSON-safe way
        # random.getstate() contains non-JSON types, so we convert to list recursively
        # easiest: store repr and eval is unsafe; so we store a tuple via list conversion
        # We'll store as list and reconstruct tuple on resume.
        print("[start] fresh build")

    # ---- create iterators per source (streaming + shuffle)
    iters = {}
    for s in sources:
        ds = load_dataset(
            s.dataset,
            s.config,
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        iters[s.name] = iter(ds)

    # shard buffers
    shard_input_ids = []
    shard_sources = []

    def flush_shard():
        nonlocal shard_idx, shard_input_ids, shard_sources

        if not shard_input_ids:
            return

        filename = f"shard-{shard_idx:06d}.parquet"
        out_path = os.path.join(args.out_dir, filename)

        table = pa.table(
            {
                "input_ids": pa.array(shard_input_ids, type=pa.list_(pa.int32())),
                "source": pa.array(shard_sources, type=pa.string()),
            }
        )

        tmp_path = out_path + ".tmp"
        pq.write_table(
            table,
            tmp_path,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            data_page_size=1024 * 1024,
        )
        os.replace(tmp_path, out_path)

        rows = len(shard_input_ids)
        tokens = rows * args.seq_len
        digest = sha256_file(out_path)

        # append manifest line
        rec = {
            "file": filename,
            "rows": rows,
            "tokens": tokens,
            "sha256": digest,
            "time": time.time(),
        }
        with open(manifest_path, "a") as mf:
            mf.write(json.dumps(rec) + "\n")

        print(f"[write] {filename} rows={rows} tokens={tokens} sha256={digest[:12]}...")

        shard_idx += 1
        shard_input_ids = []
        shard_sources = []

    def save_state():
        # random.getstate() is tuple of tuples; convert recursively to lists for JSON
        import random as _random

        st = _random.getstate()

        def to_list(x):
            if isinstance(x, tuple):
                return [to_list(i) for i in x]
            return x

        rng_state_list = to_list(st)
        atomic_write_json(
            state_path,
            {
                "shard_idx": shard_idx,
                "total_tokens_written": total_tokens_written,
                "per_source_tokens": per_source_tokens,
                "carry": carry,
                "rng_state": rng_state_list,
                "seq_len": args.seq_len,
                "blocks_per_shard": args.blocks_per_shard,
            },
        )

    def source_available(s: SourceSpec) -> bool:
        cap = s.token_cap
        if cap is None:
            return True
        return per_source_tokens[s.name] < cap

    # main loop
    while total_tokens_written < args.total_tokens:
        # pick a source that still has budget
        alive = [s for s in sources if source_available(s)]
        if not alive:
            print("[done] all sources hit token caps before reaching total_tokens")
            break

        s = weighted_choice(rng, alive)
        it = iters[s.name]

        # get a record; if iterator ends, recreate with a new shuffle seed
        try:
            rec = next(it)
        except StopIteration:
            # refresh iterator (rare but safe)
            ds = load_dataset(s.dataset, s.config, split="train", streaming=True)
            ds = ds.shuffle(
                buffer_size=args.shuffle_buffer, seed=rng.randint(0, 2**31 - 1)
            )
            iters[s.name] = iter(ds)
            it = iters[s.name]
            rec = next(it)

        text = rec.get(s.text_col, None)
        if not text or not isinstance(text, str):
            continue

        # tokenize (no special tokens)
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            continue

        # append EOS between docs
        if eos_id is not None:
            ids = ids + [eos_id]

        # extend carry buffer, and emit full blocks
        carry.extend(ids)

        while len(carry) >= args.seq_len:
            block = carry[: args.seq_len]
            carry = carry[args.seq_len :]

            shard_input_ids.append(block)
            shard_sources.append(s.name)

            per_source_tokens[s.name] += args.seq_len
            total_tokens_written += args.seq_len

            # shard full?
            if len(shard_input_ids) >= args.blocks_per_shard:
                flush_shard()
                save_state()

            # stop exactly at token budget
            if total_tokens_written >= args.total_tokens:
                break

    # flush leftovers
    flush_shard()
    save_state()
    print(f"[done] total_tokens_written={total_tokens_written} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
