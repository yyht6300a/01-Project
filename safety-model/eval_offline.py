#!/usr/bin/env python3
"""Offline evaluation of a (base | LoRA) v2 risk model on canonical samples.

Usage (from safety-model/):

    py -3.10 eval_offline.py --data fixtures/tiny_v1.jsonl
    py -3.10 eval_offline.py --data data_v1/test.jsonl --adapter runs/qwen25-1.5b-risk-lora-v1/best

Outputs (under --out, default runs/eval_offline/<timestamp>):
    metrics.json       -- AUROC (ordinal authorization score), accuracy, parse failures
    confusion.csv      -- teacher authorization x predicted authorization counts
    predictions.jsonl  -- per-sample raw and parsed model answers
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import prompts
from inference_utils import chat_prompt_ids, generate_batch_text, load_model_and_tokenizer
from metrics import average_precision, auroc
from safety_data import load_jsonl, validate_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    samples = load_jsonl(args.data)
    if args.limit:
        samples = samples[: args.limit]
    for i, sample in enumerate(samples):
        errs = validate_sample(sample)
        if errs:
            print(f"FATAL: sample[{i}] schema errors: {errs}")
            return 2

    model, tokenizer, device = load_model_and_tokenizer(args.model, adapter=args.adapter,
                                                        device=args.device)
    out_dir = Path(args.out) if args.out else Path("runs") / "eval_offline" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    prompt_ids = [chat_prompt_ids(tokenizer, s["input"]) for s in samples]
    texts = generate_batch_text(model, tokenizer, prompt_ids, device,
                                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                                progress=lambda done, total: print(f"  ... {done}/{total}", flush=True))

    ys, scores, rows, parse_fail = [], [], [], 0
    auth_correct, rl_correct, rl_total = 0, 0, 0
    confusion: Counter = Counter()
    for sample, text in zip(samples, texts):
        label = sample["label"]
        y = 0 if label["authorized"] else 1
        parsed = prompts.parse_answer(text)
        teacher_auth = label.get("authorization")
        ys.append(y)
        if parsed is None:
            parse_fail += 1
            scores.append(None)
            confusion[(teacher_auth, "UNPARSED")] += 1
        else:
            scores.append(prompts.AUTH_SCORE[parsed["authorization"]])
            auth_correct += int(parsed["authorization"] == teacher_auth)
            confusion[(teacher_auth, parsed["authorization"])] += 1
            if parsed["risk_level"] is not None and label.get("risk_level") in ("LOW", "HIGH"):
                rl_total += 1
                rl_correct += int(parsed["risk_level"] == label["risk_level"])
        rows.append({"id": sample.get("id"), "y_unsafe": y, "raw": text, "parsed": parsed,
                     "teacher": {"authorization": teacher_auth,
                                 "risk_level": label.get("risk_level")}})

    pairs = [(y, s) for y, s in zip(ys, scores) if s is not None]
    ys_p = [p[0] for p in pairs]
    rs_p = [p[1] for p in pairs]
    n_parsed = len(pairs)
    metrics = {
        "n": len(samples), "n_parsed": n_parsed, "parse_fail": parse_fail,
        "auroc": auroc(ys_p, rs_p) if pairs else None,
        "average_precision": average_precision(ys_p, rs_p) if pairs else None,
        "auth_acc": round(auth_correct / n_parsed, 4) if n_parsed else None,
        "risk_level_acc": round(rl_correct / rl_total, 4) if rl_total else None,
        "duration_s": round(time.time() - t0, 1),
        "data": args.data, "adapter": args.adapter,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    with (out_dir / "confusion.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["teacher_authorization", "predicted", "count"])
        for (t, p), c in sorted(confusion.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            writer.writerow([t, p, c])

    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
