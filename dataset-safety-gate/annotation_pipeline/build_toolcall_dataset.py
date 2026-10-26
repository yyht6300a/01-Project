#!/usr/bin/env python3

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


# AgentDojo v1.2.2: Train / Val / Test task counts
COUNTS = {
    "workspace": (28, 6, 6),
    "travel": (14, 3, 3),
    "banking": (11, 2, 3),
    "slack": (15, 3, 3),
}

SPLITS = ("train", "val", "test")

# Pilot samples already inspected during annotation development.
# Their corresponding User Tasks should not enter held-out sets.
REVIEWED_PILOT_IDS = {
    "P0050", "P0064", "P0024", "P0051",
    "P0031", "P0056", "P0009", "P0047",
    "P0058", "P0040", "P0016", "P0073",
    "P0060", "P0039", "P0043", "P0063",
}


# ============================================================
# 1. Create task-disjoint splits
# ============================================================

def make_split(args):
    from agentdojo.task_suite.load_suites import get_suites

    if args.manifest.exists():
        raise SystemExit(
            f"Refusing to overwrite split: {args.manifest}"
        )

    if not args.pilot.exists():
        raise SystemExit(f"Missing Pilot: {args.pilot}")

    reviewed = set(REVIEWED_PILOT_IDS)

    if args.seen_ids.exists():
        reviewed.update(
            x.strip()
            for x in args.seen_ids.read_text(
                encoding="utf-8"
            ).splitlines()
            if x.strip()
            and not x.lstrip().startswith("#")
        )

    seen_tasks = set()

    with args.pilot.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            if item.get("sample_id") in reviewed:
                src = item["source"]
                seen_tasks.add((
                    src["suite"],
                    src["user_task_id"]
                ))

    rng = random.Random(args.seed)
    suites = get_suites("v1.2.2")

    assignments = {
        name: {} for name in SPLITS
    }

    for suite, (n_train, n_val, n_test) in COUNTS.items():

        task_ids = sorted(suites[suite].user_tasks)

        if len(task_ids) != n_train + n_val + n_test:
            raise ValueError(
                f"Unexpected task count: {suite}"
            )

        eligible = [
            tid for tid in task_ids
            if (suite, tid) not in seen_tasks
        ]

        rng.shuffle(eligible)

        if len(eligible) < n_val + n_test:
            raise ValueError(
                f"Too few unseen tasks: {suite}"
            )

        val_ids = eligible[:n_val]

        test_ids = eligible[
            n_val:n_val + n_test
        ]

        holdout = set(val_ids + test_ids)

        train_ids = [
            tid for tid in task_ids
            if tid not in holdout
        ]

        assignments["train"][suite] = sorted(train_ids)
        assignments["val"][suite] = sorted(val_ids)
        assignments["test"][suite] = sorted(test_ids)

        print(
            suite,
            {
                s: len(assignments[s][suite])
                for s in SPLITS
            }
        )

    payload = {
        "benchmark_version": "v1.2.2",
        "seed": args.seed,
        "split_unit": ["suite", "user_task_id"],
        "reviewed_task_groups": [
            list(v) for v in sorted(seen_tasks)
        ],
        "splits": assignments,
    }

    args.manifest.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    args.manifest.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("Split saved:", args.manifest)


# ============================================================
# 2. Parse the original user task
# ============================================================

def task_text(messages):
    for msg in messages:

        if not isinstance(msg, dict):
            continue

        if msg.get("role") != "user":
            continue

        content = msg.get("content")

        if isinstance(content, str) and content.strip():
            return content

        # Support newer content-block formats.
        if isinstance(content, list):

            texts = [
                str(x.get("content", x.get("text", "")))
                for x in content
                if isinstance(x, dict)
                and x.get("type") == "text"
            ]

            result = "\n".join(texts)

            if result.strip():
                return result

    return None


# ============================================================
# 3. Read split manifest
# ============================================================

def get_assignments(path):

    manifest = json.loads(
        path.read_text(encoding="utf-8")
    )

    assignments = {}

    for split, by_suite in manifest["splits"].items():

        if split not in SPLITS:
            raise ValueError(
                f"Unknown split: {split}"
            )

        for suite, ids in by_suite.items():

            for tid in ids:

                key = (suite, tid)

                if key in assignments:
                    raise ValueError(
                        f"Task duplicated across splits: {key}"
                    )

                assignments[key] = split

    return assignments


# ============================================================
# 4. Extract individual tool calls
# ============================================================

def extract(args):

    assignments = get_assignments(args.manifest)

    if not assignments:
        raise SystemExit("Manifest has no tasks")

    base = args.runs_root / args.model

    paths = sorted(
        base.glob("*/user_task_*/*/*.json")
    )

    if args.max_files:
        paths = paths[:args.max_files]

    if not paths:
        raise SystemExit(
            f"No matching trajectories in {base}"
        )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    temps = {
        s: args.out_dir / f"{s}_unlabeled.jsonl.gz.tmp"
        for s in SPLITS
    }

    finals = {
        s: args.out_dir / f"{s}_unlabeled.jsonl.gz"
        for s in SPLITS
    }

    stats = Counter()
    errors = []
    seen_ids = set()

    handles = {
        s: gzip.open(
            temps[s],
            "wt",
            encoding="utf-8"
        )
        for s in SPLITS
    }

    try:

        for path in paths:

            stats["files_seen"] += 1

            rel = path.relative_to(args.runs_root)
            parts = rel.parts

            if len(parts) != 5:
                stats["skip_path_shape"] += 1
                continue

            model, suite, tid, attack_type, filename = parts

            split = assignments.get((suite, tid))

            if split is None:
                stats["skip_unknown_task"] += 1
                continue

            try:
                record = json.loads(
                    path.read_text(encoding="utf-8")
                )

            except (OSError, json.JSONDecodeError) as exc:

                stats["skip_bad_json"] += 1

                if len(errors) < 10:
                    errors.append(f"{rel}: {exc}")

                continue

            if not isinstance(record, dict):
                stats["skip_bad_root"] += 1
                continue

            if (
                record.get("suite_name", suite) != suite
                or record.get("user_task_id", tid) != tid
            ):
                stats["skip_metadata_mismatch"] += 1
                continue

            messages = record.get("messages")

            if not isinstance(messages, list):
                stats["skip_no_messages"] += 1
                continue

            user_task = task_text(messages)

            if not user_task:
                stats["skip_no_user_task"] += 1
                continue

            stats[f"{split}_trajectories"] += 1
            stats[f"{suite}_trajectories"] += 1

            # Find assistant messages containing tool calls.
            for mi, msg in enumerate(messages):

                if not isinstance(msg, dict):
                    continue

                if msg.get("role") != "assistant":
                    continue

                calls = msg.get("tool_calls") or []

                if not isinstance(calls, list):
                    stats["skip_nonlist_calls"] += 1
                    continue

                # Multiple tool calls in one assistant message
                # are extracted separately.
                for ci, call in enumerate(calls):

                    if (
                        not isinstance(call, dict)
                        or not isinstance(
                            call.get("function"), str
                        )
                        or not isinstance(
                            call.get("args"), dict
                        )
                    ):
                        stats["skip_malformed_call"] += 1
                        continue

                    identity = f"{rel}|{mi}|{ci}"

                    sid = hashlib.sha256(
                        identity.encode()
                    ).hexdigest()[:24]

                    if sid in seen_ids:
                        raise ValueError(
                            f"Duplicate sample ID: {sid}"
                        )

                    seen_ids.add(sid)

                    # IMPORTANT:
                    # history stops BEFORE the current
                    # assistant tool-call message.
                    #
                    # No current tool result, future message,
                    # or attack success label is included.

                    sample = {
                        "sample_id": sid,

                        "input": {
                            "user_task": user_task,

                            "history": messages[:mi],

                            "proposed_tool_call": {
                                "function": call["function"],
                                "args": call["args"],
                            },
                        },

                        "source": {
                            "model": model,
                            "suite": suite,
                            "user_task_id": tid,
                            "trajectory": str(rel),
                            "attack_type": attack_type,
                            "injection_task_id": Path(filename).stem,
                            "message_index": mi,
                            "call_index": ci,
                            "tool_call_id": call.get("id"),
                            "split": split,
                        },
                    }

                    handles[split].write(
                        json.dumps(
                            sample,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    stats[f"{split}_tool_calls"] += 1
                    stats[f"{suite}_tool_calls"] += 1

    finally:
        for handle in handles.values():
            handle.close()

    for s in SPLITS:
        temps[s].replace(finals[s])

    report = {
        "model": args.model,
        "max_files": args.max_files,
        "counts": dict(stats),
        "errors_first_10": errors,
    }

    report_path = (
        args.out_dir / "extraction_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        )
    )


# ============================================================
# 5. Audit the extracted dataset
# ============================================================

def audit(args):

    checked = Counter()
    observed_ids = set()

    assignments = get_assignments(args.manifest)

    for split in SPLITS:

        filepath = (
            args.out_dir
            / f"{split}_unlabeled.jsonl.gz"
        )

        with gzip.open(
            filepath,
            "rt",
            encoding="utf-8"
        ) as stream:

            for line in stream:

                item = json.loads(line)

                inp = item["input"]
                source = item["source"]

                key = (
                    source["suite"],
                    source["user_task_id"]
                )

                # Task-disjoint check
                assert assignments[key] == split
                assert source["split"] == split

                # Unique ID check
                assert item["sample_id"] not in observed_ids
                observed_ids.add(item["sample_id"])

                # History boundary check
                assert len(inp["history"]) == source["message_index"]

                assert isinstance(inp["user_task"], str)
                assert inp["user_task"].strip()

                assert isinstance(
                    inp["proposed_tool_call"]["args"],
                    dict
                )

                # No oracle labels
                assert "security" not in inp
                assert "utility" not in inp
                assert "target" not in item

                # Verify samples directly against raw JSON.
                if (
                    checked[f"{split}_verified_against_raw"]
                    < args.spotcheck
                ):

                    raw_path = (
                        args.runs_root
                        / source["trajectory"]
                    )

                    original = json.loads(
                        raw_path.read_text(
                            encoding="utf-8"
                        )
                    )["messages"]

                    mi = source["message_index"]
                    ci = source["call_index"]

                    assert inp["history"] == original[:mi]

                    raw_call = original[mi]["tool_calls"][ci]

                    assert inp["proposed_tool_call"] == {
                        "function": raw_call["function"],
                        "args": raw_call["args"],
                    }

                    checked[
                        f"{split}_verified_against_raw"
                    ] += 1

                checked[f"{split}_samples"] += 1

    print("AUDIT PASSED")
    print(dict(checked))


# ============================================================
# CLI
# ============================================================

def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "command",
        choices=["split", "extract", "audit"]
    )

    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/splits/task_split_seed42.json"
        )
    )

    p.add_argument(
        "--pilot",
        type=Path,
        default=Path(
            "data/annotated/pilot_indexed.jsonl"
        )
    )

    p.add_argument(
        "--seen-ids",
        type=Path,
        default=Path(
            "data/splits/seen_pilot_ids.txt"
        )
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("agentdojo/runs")
    )

    p.add_argument(
        "--model",
        default="gpt-4o-2024-05-13"
    )

    p.add_argument(
        "--max-files",
        type=int,
        default=0
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/toolcalls_v1")
    )

    p.add_argument(
        "--spotcheck",
        type=int,
        default=20
    )

    args = p.parse_args()

    if args.command == "split":
        make_split(args)

    elif args.command == "extract":
        extract(args)

    else:
        audit(args)


if __name__ == "__main__":
    main()
