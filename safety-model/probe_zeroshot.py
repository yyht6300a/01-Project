#!/usr/bin/env python3
"""Zero-shot probe: how well does the BASE model already follow the frozen prompt?

Usage (from safety-model/):

    py -3.10 probe_zeroshot.py --data fixtures/tiny.jsonl --limit 4

Writes runs/zeroshot_probe.jsonl. Evidence for the "pretrained model vs. task
format" question: parse rate and direction of risk on a few labeled samples.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import prompts
from inference_utils import chat_prompt_ids, generate_text, load_model_and_tokenizer
from safety_data import load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="fixtures/tiny.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/zeroshot_probe.jsonl")
    args = parser.parse_args()

    samples = load_jsonl(args.data)[: args.limit]
    model, tokenizer, device = load_model_and_tokenizer(args.model, device=args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    rows = []
    t0 = time.time()
    for sample in samples:
        prompt_ids = chat_prompt_ids(tokenizer, sample["input"])
        text = generate_text(model, tokenizer, prompt_ids, device, max_new_tokens=args.max_new_tokens)
        risk = prompts.parse_risk(text)
        n_ok += risk is not None
        y = 0 if sample["label"]["authorized"] else 1
        rows.append({"id": sample.get("id"), "y_unsafe": y,
                     "target_risk": prompts.target_risk(sample["label"]),
                     "parsed_risk": risk, "raw": text})
        print(f"{str(sample.get('id')):>24}  y_unsafe={y}  target={prompts.target_risk(sample['label']):.2f}  "
              f"parsed={risk}  {'OK' if risk is not None else 'PARSE-FAIL'}")
        print(f"    raw: {text.strip()[:160]}")
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nparse rate: {n_ok}/{len(samples)} | elapsed {time.time() - t0:.0f}s | outputs -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

