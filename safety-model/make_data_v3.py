"""Build data_v3 from all four teacher deliveries (worker_00..03).

Maximize training data while keeping a sound holdout:
per suite, exactly ONE suite::user_task group goes to val and ONE to test
(chosen deterministically near 7% of that suite's samples), everything else
is train. Group isolation: no suite::user_task appears in more than one split.
Val is shuffled so the first eval_gen_limit samples cover every suite.

Output: data_v3/{train,val,test}.jsonl + data_v3/build_report.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_delivery import convert_record  # noqa: E402
from safety_data import group_key, load_jsonl, validate_sample  # noqa: E402


def pick_val_test(groups: dict[str, list[dict]], target_share: float):
    """Pick one val and one test group per suite, closest to target_share of samples."""
    total = sum(len(v) for v in groups.values())
    target = total * target_share
    ordered = sorted(groups.items(), key=lambda kv: (abs(len(kv[1]) - target), kv[0]))
    val_key = ordered[0][0]
    rest = ordered[1:]
    test_key = min(rest, key=lambda kv: (abs(len(kv[1]) - target), kv[0]))[0]
    return val_key, test_key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deliveries", nargs="+",
                    default=[f"workers/worker_{i:02d}.jsonl" for i in range(4)])
    ap.add_argument("--out-dir", default="data_v3")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--target-share", type=float, default=0.07)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    errors: list[str] = []
    per_file: Counter = Counter()
    for path in args.deliveries:
        rows = load_jsonl(path)
        per_file[Path(path).name] = len(rows)
        for rec in rows:
            sample, errs = convert_record(rec, args.seed)
            errors.extend(errs)
            if sample is not None:
                samples.append(sample)

    by_suite: dict[str, dict[str, list[dict]]] = {}
    for s in samples:
        suite = s["meta"]["suite"]
        by_suite.setdefault(suite, {}).setdefault(group_key(s), []).append(s)

    val_keys: set[str] = set()
    test_keys: set[str] = set()
    plan = {}
    for suite, groups in sorted(by_suite.items()):
        val_key, test_key = pick_val_test(groups, args.target_share)
        val_keys.add(val_key)
        test_keys.add(test_key)
        plan[suite] = {
            "n_groups": len(groups),
            "samples": sum(len(v) for v in groups.values()),
            "val": (val_key, len(groups[val_key])),
            "test": (test_key, len(groups[test_key])),
        }
    for s in samples:
        key = group_key(s)
        if key in val_keys:
            s["meta"]["split"] = "val"
        elif key in test_keys:
            s["meta"]["split"] = "test"
        else:
            s["meta"]["split"] = "train"

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in samples:
        by_split[s["meta"]["split"]].append(s)

    rng = random.Random(args.seed)
    for sp in by_split:
        rng.shuffle(by_split[sp])
    for sp, rows in by_split.items():
        with open(out_dir / f"{sp}.jsonl", "w", encoding="utf-8") as fh:
            for s in rows:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    group_sets = {sp: set(group_key(s) for s in rows) for sp, rows in by_split.items()}
    violations = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        bad = group_sets[a] & group_sets[b]
        if bad:
            violations.append(f"{a}x{b}: {sorted(bad)}")
    id_sets = {sp: set(s["id"] for s in rows) for sp, rows in by_split.items()}
    id_overlap = {sp: len(id_sets[sp] & id_sets["train"]) for sp in ("val", "test")}

    checked, bad = 0, 0
    for sp in ("train", "val", "test"):
        for s in load_jsonl(out_dir / f"{sp}.jsonl"):
            checked += 1
            if validate_sample(s):
                bad += 1

    stats = {
        "seed": args.seed,
        "target_share": args.target_share,
        "deliveries": dict(per_file),
        "total_records": sum(per_file.values()),
        "converted": len(samples),
        "error_count": len(errors),
        "per_split": {k: len(v) for k, v in by_split.items()},
        "group_counts_per_split": {k: len(v) for k, v in group_sets.items()},
        "suite_plan": plan,
        "suite_split": dict(Counter(f'{s["meta"]["suite"]}/{s["meta"]["split"]}' for s in samples)),
        "auth_by_split": {sp: dict(Counter(r["label"]["authorization"] for r in rows)) for sp, rows in by_split.items()},
        "risk_by_split": {sp: dict(Counter(r["label"]["risk_level"] for r in rows)) for sp, rows in by_split.items()},
        "group_overlap_violations": violations,
        "id_overlap_with_train": id_overlap,
        "validation_checked": checked,
        "validation_errors": bad,
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 1 if (errors or violations or bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
