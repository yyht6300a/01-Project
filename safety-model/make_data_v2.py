"""Build data_v2 from the two new teacher deliveries (worker_00/01).

Why not one-file-train / one-file-test: the two files share 61 of ~64
suite::user_task groups and 800+ trajectories, so a file-level split leaks.
This script merges both files and splits per suite at the suite::user_task
group level (no group appears in more than one split). Val order is shuffled
so that the first eval_gen_limit samples (train_sft.py takes the first N)
cover all suites and label classes instead of one block of tasks.

Output: data_v2/{train,val,test}.jsonl + data_v2/build_report.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_delivery import convert_record, stratified_group_splits  # noqa: E402
from safety_data import group_key, load_jsonl, validate_sample  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deliveries", nargs="+",
                    default=["workers/worker_00.jsonl", "workers/worker_01.jsonl"])
    ap.add_argument("--out-dir", default="data_v2")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    errors: list[str] = []
    per_file: Counter = Counter()
    raw_by_file: dict[str, list[dict]] = {}
    for path in args.deliveries:
        rows = load_jsonl(path)
        per_file[Path(path).name] = len(rows)
        raw_by_file[Path(path).name] = rows
        for rec in rows:
            sample, errs = convert_record(rec, args.seed)
            errors.extend(errs)
            if sample is None:
                continue
            samples.append(sample)

    if errors:
        print(f"WARNING: {len(errors)} conversion errors, first 10:")
        for e in errors[:10]:
            print(" ", e)

    # Cross-file overlap report (evidence why file-level split is unsafe).
    names = list(raw_by_file)
    traj = [set(r["source"]["trajectory"] for r in raw_by_file[n]) for n in names]
    tasks = [set((r["source"]["suite"], r["source"]["user_task_id"]) for r in raw_by_file[n]) for n in names]
    ids = [set(r["sample_id"] for r in raw_by_file[n]) for n in names]
    overlap = {
        "files": names,
        "sample_id_overlap": len(ids[0] & ids[1]),
        "trajectory_overlap": len(traj[0] & traj[1]),
        "task_group_overlap": len(tasks[0] & tasks[1]),
    }

    stratified_group_splits(samples, args.seed, args.ratios)

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in samples:
        by_split[s["meta"]["split"]].append(s)

    # Isolation guards: group keys must be disjoint across splits.
    group_sets = {sp: set(group_key(s) for s in rows) for sp, rows in by_split.items()}
    violations = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        bad = group_sets[a] & group_sets[b]
        if bad:
            violations.append(f"{a}x{b}: {sorted(bad)}")
    id_sets = {sp: set(s["id"] for s in rows) for sp, rows in by_split.items()}
    id_violations = {sp: len(id_sets[sp] & id_sets["train"]) for sp in ("val", "test")}

    # Shuffle within splits (val first so the eval subset is representative).
    rng = random.Random(args.seed)
    for sp in by_split:
        rng.shuffle(by_split[sp])

    for sp, rows in by_split.items():
        with open(out_dir / f"{sp}.jsonl", "w", encoding="utf-8") as fh:
            for s in rows:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Post-write validation.
    checked = 0
    bad = 0
    for sp in ("train", "val", "test"):
        for s in load_jsonl(out_dir / f"{sp}.jsonl"):
            checked += 1
            if validate_sample(s):
                bad += 1

    # Novel-holdout note: tasks that never appeared in data_v1 (older run).
    prior = set()
    prior_dir = Path("data_v1")
    if prior_dir.exists():
        for sp in ("train", "val", "test"):
            p = prior_dir / f"{sp}.jsonl"
            if p.exists():
                for s in load_jsonl(p):
                    prior.add(group_key(s))
    test_groups = group_sets["test"]
    novel = sorted(test_groups - prior)

    stats = {
        "seed": args.seed,
        "deliveries": dict(per_file),
        "total_records": sum(per_file.values()),
        "converted": len(samples),
        "error_count": len(errors),
        "ratios": list(args.ratios),
        "cross_file_overlap": overlap,
        "per_split": {k: len(v) for k, v in by_split.items()},
        "group_counts_per_split": {k: len(v) for k, v in group_sets.items()},
        "suite_split": dict(Counter(f'{s["meta"]["suite"]}/{s["meta"]["split"]}' for s in samples)),
        "auth_by_split": {sp: dict(Counter(r["label"]["authorization"] for r in rows)) for sp, rows in by_split.items()},
        "risk_by_split": {sp: dict(Counter(r["label"]["risk_level"] for r in rows)) for sp, rows in by_split.items()},
        "group_overlap_violations": violations,
        "id_overlap_with_train": id_violations,
        "test_groups_not_in_data_v1": novel,
        "validation_checked": checked,
        "validation_errors": bad,
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 1 if (errors or violations or bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
