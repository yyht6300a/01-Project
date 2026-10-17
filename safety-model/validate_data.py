#!/usr/bin/env python3
"""Validate canonical safety-model datasets (schema + split integrity + stats).

Usage (from safety-model/):

    py -3.10 validate_data.py fixtures/tiny.jsonl
    py -3.10 validate_data.py data/train.jsonl data/val.jsonl --tokenizer Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from safety_data import dataset_stats, group_key, load_jsonl, validate_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--tokenizer", default=None, help="optional HF tokenizer for length stats")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    args = parser.parse_args()

    errors: list[str] = []
    per_file: dict[str, list[dict]] = {}
    ids_seen: dict[str, str] = {}
    group_splits: dict[str, dict[str, str]] = defaultdict(dict)

    for fname in args.files:
        rows = load_jsonl(fname)
        per_file[fname] = rows
        for i, sample in enumerate(rows):
            for e in validate_sample(sample):
                errors.append(f"{fname}[{i}]: {e}")
            sid = sample.get("id")
            if isinstance(sid, str):
                if sid in ids_seen:
                    errors.append(f"{fname}[{i}]: duplicate id {sid!r} (also at {ids_seen[sid]})")
                ids_seen[sid] = f"{fname}[{i}]"
            group_splits[group_key(sample)][fname] = sample.get("meta", {}).get("split")

    for fname, rows in per_file.items():
        print(f"--- {fname}: {len(rows)} samples ---")
        for key, value in dataset_stats(rows).items():
            print(f"    {key}: {value}")

    for group, files in group_splits.items():
        splits = {s for s in files.values() if s is not None}
        if len(files) > 1 and len(splits) > 1:
            errors.append(f"group {group} appears with multiple splits {sorted(splits)} across files {sorted(files)}")

    if args.tokenizer:
        import numpy as np
        from transformers import AutoTokenizer

        from inference_utils import encode_sample

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        lengths = []
        for rows in per_file.values():
            for sample in rows:
                prompt_ids, answer_ids = encode_sample(sample, tok, 10**9)
                lengths.append(len(prompt_ids) + len(answer_ids))
        arr = np.array(lengths)
        print(f"\ntoken lengths (prompt+answer): p50={np.percentile(arr, 50):.0f} "
              f"p95={np.percentile(arr, 95):.0f} max={arr.max()} | max_seq_len={args.max_seq_len}")
        if np.percentile(arr, 95) > args.max_seq_len * 0.9:
            print(f"WARN: p95 close to or above max_seq_len={args.max_seq_len}; consider raising it")

    if errors:
        print(f"\nFAIL: {len(errors)} problem(s)")
        for e in errors[:50]:
            print("  " + e)
        return 1
    print("\nOK: schema + split integrity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

