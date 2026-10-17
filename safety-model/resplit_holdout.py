#!/usr/bin/env python3
"""Re-split the placeholder test holdout: keep ~N test samples, move the rest back.

2026-10-16 decision: the current test.jsonl is a placeholder until the final
test delivery arrives (test data will be extended later). Keep only the whole
(suite, user_task) groups whose total size is closest to --test-target and
redistribute every remaining test group into train / val.

Rules (deterministic, whole-group only - a group never straddles splits):

  * test keeps the subset of test groups minimizing |total - test_target|
    (ties: larger total, then lexicographic group order);
  * val receives, from the remaining groups, the subset whose total is closest
    to round(remaining_total * val_share), with val_share taken from the
    split sizes currently on disk (ties: smaller total);
  * all other remaining groups move to train.

Moved samples get meta.split updated; file order is preserved (existing rows
first, appended rows in their former test order). Idempotent.

Usage (from safety-model/):

    py -3.10 resplit_holdout.py --data-dir data_v1 --test-target 40 \
        --backup-dir ../_archive/data_v1_splits_backup_2026-10-16
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from safety_data import load_jsonl

GroupKey = tuple[str, str]


def group_key(sample: dict) -> GroupKey:
    meta = sample["meta"]
    return (meta["suite"], meta["user_task"])


def group_samples(rows: list[dict]) -> dict[GroupKey, list[dict]]:
    groups: dict[GroupKey, list[dict]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    return groups


def closest_subset(candidates: list[tuple[GroupKey, int]], target: int, prefer_larger: bool) -> tuple[int, tuple[GroupKey, ...]]:
    """Subset of (key, size) minimizing |sum - target| (deterministic tie-break)."""
    sums: dict[int, tuple[GroupKey, ...]] = {0: ()}
    for key, size in sorted(candidates):
        additions: dict[int, tuple[GroupKey, ...]] = {}
        for total, keys in sums.items():
            candidate = total + size
            cand_keys = keys + (key,)
            if candidate not in sums and (candidate not in additions or cand_keys < additions[candidate]):
                additions[candidate] = cand_keys
        for total, keys in additions.items():
            sums.setdefault(total, keys)
    best_score = None
    best: tuple[int, tuple[GroupKey, ...]] = (0, ())
    for total, keys in sums.items():
        tie = -total if prefer_larger else total
        score = (abs(total - target), tie, keys)
        if best_score is None or score < best_score:
            best_score, best = score, (total, keys)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_v1")
    ap.add_argument("--test-target", type=int, default=40)
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_rows = load_jsonl(data_dir / "train.jsonl")
    val_rows = load_jsonl(data_dir / "val.jsonl")
    test_rows = load_jsonl(data_dir / "test.jsonl")

    train_groups = group_samples(train_rows)
    val_groups = group_samples(val_rows)
    test_groups = group_samples(test_rows)
    dup = (set(train_groups) & set(val_groups)) | (set(train_groups) & set(test_groups)) | (set(val_groups) & set(test_groups))
    if dup:
        raise SystemExit(f"input already leaks groups across splits: {sorted(dup)}")

    kept_total, kept_keys_tuple = closest_subset(
        [(key, len(rows)) for key, rows in test_groups.items()], args.test_target, prefer_larger=True
    )
    kept_keys = set(kept_keys_tuple)

    remaining = [(key, len(rows)) for key, rows in test_groups.items() if key not in kept_keys]
    remaining_total = sum(size for _, size in remaining)
    val_share = len(val_rows) / (len(train_rows) + len(val_rows))
    val_target = round(remaining_total * val_share)
    if remaining:
        val_total, val_keys_tuple = closest_subset(remaining, val_target, prefer_larger=False)
    else:
        val_total, val_keys_tuple = 0, ()
    val_keys = set(val_keys_tuple)

    moved_train = [r for r in test_rows if group_key(r) not in kept_keys and group_key(r) not in val_keys]
    moved_val = [r for r in test_rows if group_key(r) in val_keys]
    new_test = [r for r in test_rows if group_key(r) in kept_keys]
    for row in moved_train:
        row["meta"]["split"] = "train"
    for row in moved_val:
        row["meta"]["split"] = "val"
    new_train = train_rows + moved_train
    new_val = val_rows + moved_val

    new_keys = {
        "train": {group_key(r) for r in new_train},
        "val": {group_key(r) for r in new_val},
        "test": {group_key(r) for r in new_test},
    }
    assert not (new_keys["train"] & new_keys["val"]), "train/val group overlap after resplit"
    assert not (new_keys["train"] & new_keys["test"]), "train/test group overlap after resplit"
    assert not (new_keys["val"] & new_keys["test"]), "val/test group overlap after resplit"
    assert len(new_train) + len(new_val) + len(new_test) == len(train_rows) + len(val_rows) + len(test_rows)

    report = {
        "date": "2026-10-16",
        "decision": f"keep ~{args.test_target} test samples (placeholder holdout); move the rest back to train/val",
        "test_target": args.test_target,
        "kept_test_groups": {f"{suite}::{task}": len(test_groups[(suite, task)]) for suite, task in sorted(kept_keys)},
        "moved_to_val_groups": {f"{suite}::{task}": len(test_groups[(suite, task)]) for suite, task in sorted(val_keys)},
        "moved_to_train_groups": {
            f"{suite}::{task}": len(rows)
            for (suite, task), rows in sorted(test_groups.items())
            if (suite, task) not in kept_keys and (suite, task) not in val_keys
        },
        "before": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "after": {"train": len(new_train), "val": len(new_val), "test": len(new_test)},
        "kept_test_total": kept_total,
        "val_total_added": val_total,
        "val_topup_target": val_target,
        "val_share_on_disk": round(val_share, 4),
        "suite_split_after": dict(sorted(Counter(
            f"{r['meta']['suite']}/{r['meta']['split']}" for r in new_train + new_val + new_test
        ).items())),
        "dry_run": args.dry_run,
    }

    if args.backup_dir and not args.dry_run:
        backup = Path(args.backup_dir)
        backup.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            shutil.copy2(data_dir / name, backup / name)
        print(f"backup written: {backup.resolve()}")

    if not args.dry_run:
        for name, rows in (("train.jsonl", new_train), ("val.jsonl", new_val), ("test.jsonl", new_test)):
            with open(data_dir / name, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = Path(args.report) if args.report else data_dir / "resplit_report.json"
    if not args.dry_run:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
