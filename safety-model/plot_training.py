"""Render training curves from a run directory.

Reads train_steps.jsonl (step / loss / lr / elapsed), metrics.jsonl
(eval points) and train_info.json, writes training_curves.png next to them.
Run standalone any time, including while the training job is still writing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)

    steps = load_jsonl(run / "train_steps.jsonl")
    evals = load_jsonl(run / "metrics.jsonl")
    info_path = run / "train_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib unavailable: report, do not fail the job
        print(f"plot_training: matplotlib unavailable ({exc}); skipping curve PNG")
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    xs = [r["step"] for r in steps]
    ys = [r["loss"] for r in steps]

    ax = axes[0][0]
    ax.plot(xs, ys, color="tab:blue", alpha=0.35, lw=0.8, label="train loss (per step)")
    ema = []
    s = None
    for y in ys:
        s = y if s is None else 0.85 * s + 0.15 * y
        ema.append(s)
    ax.plot(xs, ema, color="tab:blue", lw=1.6, label="train loss (EMA)")
    evx = [r["step"] for r in evals if "val_loss" in r]
    evy = [r["val_loss"] for r in evals if "val_loss" in r]
    ax.scatter(evx, evy, color="tab:red", s=36, zorder=5, label="val loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title("loss curves"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.plot([r["step"] for r in steps], [r["lr"] for r in steps], color="tab:green", lw=1.2)
    ax.set_xlabel("step"); ax.set_ylabel("lr"); ax.set_title("learning rate")
    ax.grid(alpha=0.3); ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    ax = axes[1][0]
    e2 = [r for r in evals if "val_auroc" in r and r["val_auroc"] is not None]
    ax.plot([r["step"] for r in e2], [r["val_auroc"] for r in e2], marker="o", color="tab:orange", label="val_auroc")
    ax.plot([r["step"] for r in evals if "auth_acc" in r],
            [r["auth_acc"] for r in evals if "auth_acc" in r],
            marker="s", color="tab:purple", label="auth_acc")
    ax.plot([r["step"] for r in evals if "risk_level_acc" in r],
            [r["risk_level_acc"] for r in evals if "risk_level_acc" in r],
            marker="^", color="tab:brown", label="risk_level_acc")
    ax.set_xlabel("step"); ax.set_ylim(0, 1)
    ax.set_title("eval metrics"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.axis("off")
    lines = [
        f"run: {run.name}",
        f"target: {info.get('target_format', '?')}",
        f"base: {info.get('base_model', '?')}",
        f"device: {info.get('device', '?')}  dtype: {info.get('dtype', '?')}",
        f"trainable: {info.get('trainable_params', '?'):,}  train: {info.get('train_samples', '?')}  val: {info.get('val_samples', '?')}",
        f"effective batch: {info.get('effective_batch', '?')}  seq: {info.get('max_seq_len', '?')}",
        f"total steps: {info.get('total_steps', '?')}  warmup: {info.get('warmup_steps', '?')}",
        f"resumed: {info.get('resumed', '?')} from {info.get('resume_from', '-')}",
        f"optim: {json.dumps(info.get('optim', {}), ensure_ascii=False)}",
        f"steps recorded: {len(steps)}  evals: {len(evals)}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.02, 0.98 - 0.09 * i, line, transform=ax.transAxes, fontsize=9,
                family="monospace", va="top")

    fig.suptitle(f"AI6101 SFT {run.name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = run / "training_curves.png"
    fig.savefig(out, dpi=110)
    print(f"plot_training: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
