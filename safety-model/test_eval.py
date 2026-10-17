"""Held-out evaluation with test loss: answer-only CE loss + ranking metrics.

Usage (from safety-model/):
  py -3.10 test_eval.py --data data_v3/test.jsonl \
      --adapter runs/qwen25-1.5b-risk-lora-v3-tc2/best
  py -3.10 test_eval.py --data data_v3/test.jsonl \
      --adapter runs/qwen25-1.5b-risk-lora-v3b-tc2/best --target-format 2field
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch

import prompts
import train_sft
from inference_utils import load_model_and_tokenizer
from safety_data import load_jsonl, validate_sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--target-format", default="4field", choices=("4field", "2field"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    samples = load_jsonl(args.data)
    if args.limit:
        samples = samples[: args.limit]
    for i, s in enumerate(samples):
        errs = validate_sample(s)
        if errs:
            print(f"FATAL: sample[{i}] schema errors: {errs}")
            return 2

    model, tokenizer, device = load_model_and_tokenizer(args.model, adapter=args.adapter,
                                                        device=args.device)
    use_bf16 = device.type == "cuda"
    autocast = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
                if use_bf16 else (lambda: contextlib.nullcontext()))
    answer_fn = prompts.answer_json_2field if args.target_format == "2field" else prompts.answer_json
    gen_limit = len(samples)
    stats = train_sft.evaluate(model, tokenizer, samples, max_seq_len=1024,
                               gen_limit=gen_limit, batch_size=args.batch_size,
                               device=device, pad_id=tokenizer.pad_token_id,
                               autocast=autocast, answer_fn=answer_fn)
    out = {"data": args.data, "adapter": args.adapter,
           "target_format": args.target_format, **stats}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
