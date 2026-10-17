#!/usr/bin/env python3
"""LoRA SFT for the safety risk model (Qwen2.5-1.5B-Instruct base).

The model learns to answer the frozen system prompt with
{"risk": <0..1>, "reason": "<one short sentence>"} for one tool call.

Run from safety-model/:

    py -3.10 validate_data.py fixtures/tiny.jsonl
    py -3.10 train_sft.py --config configs/sft_lora.yaml --train fixtures/tiny.jsonl --val fixtures/tiny.jsonl --smoke
    py -3.10 train_sft.py --config configs/sft_lora.yaml

Design choices (see README for rationale):
- custom loop (no Trainer): small API surface, robust to transformers 5.x drift;
- bf16 autocast on CUDA, fp32 on CPU (debug only);
- warmup + cosine schedule; early stop on val AUROC (fallback: -val_loss);
- best/last checkpoints keep the LoRA adapter + optimizer/scheduler state,
  so preempted school-GPU jobs can resume with --resume.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import prompts
from inference_utils import encode_sample, generate_batch_text, generate_text, pick_device, pick_dtype
from metrics import auroc
from safety_data import load_jsonl, validate_sample

BASE_DIR = Path(__file__).resolve().parent
EVAL_MAX_NEW_TOKENS = 384


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ChatDataset(Dataset):
    def __init__(self, samples, tokenizer, max_seq_len, answer_fn=None):
        self.rows = []
        for idx, s in enumerate(samples):
            prompt_ids, answer_ids = encode_sample(s, tokenizer, max_seq_len, answer_fn)
            self.rows.append((prompt_ids, answer_ids, s.get("id", str(idx))))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        prompt_ids, answer_ids, sid = self.rows[i]
        return {"prompt_ids": prompt_ids, "answer_ids": answer_ids, "id": sid}


def make_collate(pad_id: int):
    def collate(batch):
        max_len = max(len(b["prompt_ids"]) + len(b["answer_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            seq = b["prompt_ids"] + b["answer_ids"]
            n_prompt = len(b["prompt_ids"])
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            labels[i, n_prompt:len(seq)] = torch.tensor(b["answer_ids"], dtype=torch.long)
            attn[i, :len(seq)] = 1
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
    return collate


def lr_lambda(step: int, warmup_steps: int, total_steps: int, eta_min_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, tokenizer, samples, max_seq_len, gen_limit, batch_size, device, pad_id, autocast,
             answer_fn=None):
    dataset = ChatDataset(samples, tokenizer, max_seq_len, answer_fn)
    # The loss phase materializes [bs, seq, vocab] logits (~0.5GB per sample at
    # seq=800/vocab=152k in fp32), so cap it; generation below keeps `batch_size`
    # (its per-step logits are tiny). Protects 8GB GPUs with long v2 targets.
    loss_bs = max(1, min(2, batch_size))
    loader = DataLoader(dataset, batch_size=loss_bs, shuffle=False,
                        collate_fn=make_collate(pad_id))
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():
            out = model(**batch)
        losses.append(float(out.loss.detach()))
    val_loss = float(np.mean(losses)) if losses else float("nan")

    n_gen = min(gen_limit, len(dataset))
    gen_prompts = [dataset.rows[i][0] for i in range(n_gen)]
    texts = generate_batch_text(model, tokenizer, gen_prompts, device,
                                max_new_tokens=EVAL_MAX_NEW_TOKENS, batch_size=batch_size,
                                progress=lambda done, total: log(f"eval gen {done}/{total}"))
    auth_scores, ys = [], []
    rl_pred, rl_true = [], []
    parse_fail, auth_correct = 0, 0
    for i in range(n_gen):
        y = 0 if samples[i]["label"]["authorized"] else 1
        ys.append(y)
        parsed = prompts.parse_answer(texts[i])
        if parsed is None:
            parse_fail += 1
            auth_scores.append(None)
            continue
        auth_scores.append(prompts.AUTH_SCORE[parsed["authorization"]])
        auth_correct += int(parsed["authorization"] == samples[i]["label"].get("authorization"))
        if parsed["risk_level"] is not None:
            rl_pred.append(1 if parsed["risk_level"] == "HIGH" else 0)
            rl_true.append(0 if samples[i]["label"].get("risk_level") == "LOW" else 1)
    pairs = [(y, s) for y, s in zip(ys, auth_scores) if s is not None]
    val_auroc = auroc([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
    n_parsed = len(pairs)
    auth_acc = round(auth_correct / n_parsed, 4) if n_parsed else None
    rl_acc = round(sum(1 for a, b in zip(rl_true, rl_pred) if a == b) / len(rl_true), 4) if rl_true else None
    return {"val_loss": round(val_loss, 4), "val_auroc": val_auroc,
            "auth_acc": auth_acc, "risk_level_acc": rl_acc,
            "parse_fail": parse_fail, "n_gen": n_gen, "n_parsed": n_parsed}


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (BASE_DIR / p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_lora.yaml")
    parser.add_argument("--train", default=None, help="override data.train")
    parser.add_argument("--val", default=None, help="override data.val")
    parser.add_argument("--model", default=None, help="override base_model")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU run to validate the pipeline")
    parser.add_argument("--max-steps", type=int, default=None, help="override train.max_steps (keep small for bench)")
    parser.add_argument("--eval-steps", type=int, default=None, help="override train.eval_steps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve(args.config))
    train_path = resolve(args.train or cfg["data"]["train"])
    val_path = resolve(args.val or cfg["data"]["val"])
    base_model = args.model or cfg["base_model"]
    out_dir = resolve(args.output_dir or cfg["train"]["output_dir"])
    seed = int(args.seed if args.seed is not None else cfg["train"].get("seed", 42))
    set_seed(seed)

    device = pick_device(args.device)
    dtype = pick_dtype(device)

    micro_batch = int(cfg["train"]["micro_batch_size"])
    grad_accum = int(cfg["train"]["grad_accum"])
    max_epochs = int(cfg["train"]["max_epochs"])
    max_steps = int(cfg["train"].get("max_steps", -1))
    eval_steps = int(cfg["train"].get("eval_steps", 0))
    eval_gen_limit = int(cfg["train"].get("eval_gen_limit", 96))
    eval_batch_size = int(cfg["train"].get("eval_batch_size", 8))
    max_seq_len = int(cfg["model"]["max_seq_len"])
    target_format = str(cfg["model"].get("target_format", "4field"))
    answer_fn = prompts.answer_json_2field if target_format == "2field" else prompts.answer_json
    log(f"target format: {target_format}")
    if args.smoke:
        micro_batch, grad_accum = 1, 1
        max_epochs, max_steps = 1, 2
        eval_steps = 1
        eval_gen_limit = 2
        max_seq_len = min(max_seq_len, 384)
        log("smoke mode: max_steps=2, seq<=384, batch=1, gen_limit=2")
    if args.max_steps is not None and not args.smoke:
        max_steps = int(args.max_steps)
    if args.eval_steps is not None and not args.smoke:
        eval_steps = int(args.eval_steps)
    if args.max_steps is not None or args.eval_steps is not None:
        log(f"CLI override: max_steps={max_steps} eval_steps={eval_steps}")

    train_samples = load_jsonl(train_path)
    val_samples = load_jsonl(val_path)
    for name, samples in (("train", train_samples), ("val", val_samples)):
        for i, s in enumerate(samples):
            errs = validate_sample(s)
            if errs:
                log(f"FATAL: {name}[{i}] schema errors: {errs}")
                return 2
    log(f"data: train={len(train_samples)} val={len(val_samples)} | device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    base.config.use_cache = False

    last_dir = out_dir / "last"
    resume_wanted = args.resume or bool(cfg["train"].get("resume", False))
    resume_from = last_dir if resume_wanted and (last_dir / "adapter_config.json").exists() else None
    if resume_from:
        model = PeftModel.from_pretrained(base, str(resume_from), is_trainable=True)
        log(f"resumed LoRA adapter from {resume_from}")
    else:
        lora_cfg = cfg["model"]["lora"]
        peft_cfg = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["lora_alpha"]),
            lora_dropout=float(lora_cfg["lora_dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, peft_cfg)
    if cfg["train"].get("grad_checkpoint"):
        model.gradient_checkpointing_enable()
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    log(f"trainable params: {n_trainable:,}")

    no_decay, decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith("bias") else decay).append(p)
    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(optim_cfg["weight_decay"])},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(optim_cfg["lr"]),
        betas=tuple(float(b) for b in optim_cfg["betas"]),
        eps=1e-8,
    )

    steps_per_epoch = max(1, math.ceil(len(train_samples) / (micro_batch * grad_accum)))
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * max_epochs
    warmup_steps = max(int(optim_cfg["min_warmup_steps"]),
                       math.ceil(float(optim_cfg["warmup_ratio"]) * total_steps))
    warmup_steps = min(warmup_steps, max(total_steps - 1, 1))
    eta_min_ratio = float(optim_cfg["eta_min_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: lr_lambda(s, warmup_steps, total_steps, eta_min_ratio),
    )
    log(f"total optimizer steps={total_steps} (steps/epoch={steps_per_epoch}) warmup={warmup_steps}")

    start_step, best_metric, best_step, bad_evals = 0, None, -1, 0
    state_file = (resume_from / "trainer_state.pt") if resume_from else None
    if state_file and state_file.exists():
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state.get("step", 0))
        best_metric = state.get("best_metric")
        best_step = int(state.get("best_step", -1))
        bad_evals = int(state.get("bad_evals", 0))
        if cfg["train"].get("resume_reset_metrics"):
            best_metric, best_step, bad_evals = None, -1, 0
            log("resumed trainer state; eval baseline reset (resume_reset_metrics)")
        log(f"resumed trainer state at step={start_step} best_metric={best_metric}")

    es_cfg = cfg["train"].get("early_stop") or {}
    patience = int(es_cfg.get("patience", 3))
    min_delta = float(es_cfg.get("min_delta", 0.002))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps({"config": cfg, "smoke": args.smoke, "device": str(device), "base_model": base_model},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics_path = out_dir / "metrics.jsonl"
    (out_dir / "train_info.json").write_text(
        json.dumps({
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "base_model": str(base_model),
            "device": str(device),
            "dtype": str(dtype),
            "target_format": target_format,
            "trainable_params": n_trainable,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "micro_batch_size": micro_batch,
            "grad_accum": grad_accum,
            "effective_batch": micro_batch * grad_accum,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "max_seq_len": max_seq_len,
            "max_epochs": max_epochs,
            "seed": cfg["train"].get("seed"),
            "resumed": bool(resume_from),
            "resume_from": str(resume_from) if resume_from else None,
            "start_step": start_step,
            "optim": cfg.get("optim", {}),
            "lora": cfg.get("model", {}).get("lora", {}),
            "early_stop": cfg["train"].get("early_stop"),
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    steps_fh = (out_dir / "train_steps.jsonl").open("a", encoding="utf-8")

    def save_checkpoint(path: Path, step: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        torch.save({
            "step": step, "best_metric": best_metric, "best_step": best_step,
            "bad_evals": bad_evals,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        }, path / "trainer_state.pt")

    train_ds = ChatDataset(train_samples, tokenizer, max_seq_len, answer_fn)
    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True,
                              collate_fn=make_collate(tokenizer.pad_token_id), num_workers=0)

    use_bf16 = device.type == "cuda"
    autocast = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
                if use_bf16 else (lambda: contextlib.nullcontext()))

    stop = False
    global_step = start_step
    micro = 0
    running = 0.0
    t0 = time.time()
    model.train()
    while not stop and global_step < total_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                out = model(**batch)
                loss = out.loss / grad_accum
            loss.backward()
            running += float(out.loss.detach())
            micro += 1
            if micro % grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, float(optim_cfg["grad_clip"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % 10 == 0 or global_step == total_steps:
                step_loss = running / micro
                step_lr = scheduler.get_last_lr()[0]
                step_elapsed = time.time() - t0
                log(f"step {global_step}/{total_steps} loss={step_loss:.4f} "
                    f"lr={step_lr:.2e} elapsed={step_elapsed:.0f}s")
                steps_fh.write(json.dumps({
                    "step": global_step, "loss": round(float(step_loss), 6),
                    "lr": float(step_lr), "elapsed_s": round(step_elapsed, 1),
                    "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                }, ensure_ascii=False) + "\n")
                steps_fh.flush()
            running, micro = 0.0, 0

            if global_step % 100 == 0 and global_step % max(eval_steps, 1) != 0:
                save_checkpoint(last_dir, global_step)
                log(f"periodic checkpoint saved at step {global_step} -> {last_dir}")

            do_eval = (eval_steps > 0 and global_step % eval_steps == 0) or global_step == total_steps
            if do_eval:
                # Persist progress before evaluating: if the eval dies (e.g. CUDA
                # OOM on an 8GB card), the run can still resume from the full step
                # count instead of the previous periodic checkpoint.
                save_checkpoint(last_dir, global_step)
                log(f"eval@{global_step} start: {len(val_samples)} val samples, "
                    f"gen batch={eval_batch_size}, up to {EVAL_MAX_NEW_TOKENS} new tokens")
                try:
                    stats = evaluate(model, tokenizer, val_samples, max_seq_len, eval_gen_limit,
                                     eval_batch_size, device, tokenizer.pad_token_id, autocast,
                                     answer_fn)
                except Exception as exc:
                    # Never let a failed evaluation kill an otherwise healthy run.
                    log(f"eval@{global_step} FAILED: {type(exc).__name__}: {exc}")
                    torch.cuda.empty_cache()
                    with metrics_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"step": global_step, "eval_error": str(exc)[:500]},
                                            ensure_ascii=False) + "\n")
                    model.train()
                    if global_step >= total_steps:
                        break
                    continue
                metric = stats["val_auroc"] if stats["val_auroc"] is not None else -stats["val_loss"]
                improved = best_metric is None or (metric - best_metric) > min_delta
                record = {"step": global_step, **stats, "metric": metric,
                          "elapsed_s": round(time.time() - t0, 1)}
                with metrics_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                log("eval@%d: %s" % (global_step, json.dumps(record, ensure_ascii=False)))
                if improved:
                    best_metric, best_step, bad_evals = metric, global_step, 0
                    save_checkpoint(out_dir / "best", global_step)
                    log(f"saved new best (metric={metric:.4f}) -> {out_dir / 'best'}")
                else:
                    bad_evals += 1
                    log(f"no improvement ({bad_evals}/{patience})")
                    if bad_evals >= patience:
                        stop = True
                save_checkpoint(last_dir, global_step)
                model.train()
                if stop:
                    break
            if global_step >= total_steps:
                break
        if stop:
            break

    if best_step < 0:
        save_checkpoint(out_dir / "best", global_step)
        best_step = global_step
    steps_fh.close()
    log(f"done. steps={global_step} best_step={best_step} best_metric={best_metric} "
        f"elapsed={time.time() - t0:.0f}s | checkpoints under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

