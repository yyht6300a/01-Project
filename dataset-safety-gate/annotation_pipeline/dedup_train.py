import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def serialize(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":")
    )


def normalize_history(history):
    """
    Normalize random Tool Call IDs while preserving
    the relationship between calls and responses.
    """
    id_map = {}

    def alias(value):
        if not isinstance(value, str):
            return value
        if value not in id_map:
            id_map[value] = f"call_{len(id_map) + 1}"
        return id_map[value]

    # Assign stable IDs in assistant call order.
    for message in history:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue

        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                if isinstance(call.get("id"), str):
                    alias(call["id"])

    def walk(obj):
        if isinstance(obj, list):
            return [walk(v) for v in obj]

        if isinstance(obj, dict):
            is_tool_call = (
                isinstance(obj.get("function"), str)
                and "args" in obj
            )

            output = {}

            for key, value in obj.items():
                if key == "tool_call_id" and isinstance(value, str):
                    output[key] = alias(value)

                elif key == "id" and is_tool_call and isinstance(value, str):
                    output[key] = alias(value)

                else:
                    output[key] = walk(value)

            return output

        return obj

    return walk(history)


def fingerprint(row):
    inp = row["input"]
    src = row["source"]

    key = [
        src["suite"],
        src["user_task_id"],
        inp["user_task"],
        normalize_history(inp["history"]),
        inp["proposed_tool_call"]
    ]

    return hashlib.sha256(
        serialize(key).encode("utf-8")
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in-dir",
        default="data/raw/toolcalls_v1"
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/toolcalls_v1"
    )

    args = parser.parse_args()

    input_path = (
        Path(args.in_dir) / "train_unlabeled.jsonl.gz"
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    unique_path = out / "train_unique_unlabeled.jsonl.gz"
    mapping_path = out / "train_duplicate_mapping.jsonl.gz"
    report_path = out / "train_dedup_report.json"

    representatives = {}
    members = []
    frequencies = Counter()

    suite_original = Counter()
    suite_unique = Counter()

    group_details = {}
    sample_ids = set()

    # Pass 1: deduplicate Train samples.
    with gzip.open(input_path, "rt", encoding="utf-8") as src, \
         gzip.open(unique_path, "wt", encoding="utf-8") as dst:

        for line in src:
            if not line.strip():
                continue

            row = json.loads(line)

            assert row["source"]["split"] == "train"

            sample_id = row["sample_id"]

            if sample_id in sample_ids:
                raise ValueError(
                    f"Duplicate sample_id: {sample_id}"
                )

            sample_ids.add(sample_id)

            key = fingerprint(row)

            suite = row["source"]["suite"]
            task = row["source"]["user_task_id"]
            tool = row["input"]["proposed_tool_call"]["function"]

            suite_original[suite] += 1
            frequencies[key] += 1

            members.append((sample_id, key, suite, task))

            if key not in representatives:
                representatives[key] = sample_id
                suite_unique[suite] += 1

                group_details[key] = {
                    "suite": suite,
                    "task": task,
                    "tool": tool
                }

                # Preserve the original sample unchanged.
                dst.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

    # Pass 2: write original-to-representative mapping.
    with gzip.open(mapping_path, "wt", encoding="utf-8") as dst:

        for sample_id, key, suite, task in members:
            mapping = {
                "sample_id": sample_id,
                "representative_sample_id": representatives[key],
                "fingerprint": key,
                "group_size": frequencies[key],
                "is_representative": (
                    sample_id == representatives[key]
                ),
                "suite": suite,
                "user_task_id": task
            }

            dst.write(
                json.dumps(mapping, ensure_ascii=False) + "\n"
            )

    total = len(members)
    unique = len(representatives)

    report = {
        "total_original": total,
        "total_unique": unique,
        "removed_as_duplicates": total - unique,
        "redundancy_pct": round(
            100 * (total - unique) / total, 2
        ) if total else 0,
        "normalization": (
            "Stable Tool Call ID aliases preserving "
            "call-response references and task boundaries"
        ),
        "by_suite": {
            suite: {
                "original": suite_original[suite],
                "unique": suite_unique[suite],
                "redundancy_pct": round(
                    100 * (
                        suite_original[suite] - suite_unique[suite]
                    ) / suite_original[suite], 2
                )
            }
            for suite in sorted(suite_original)
        },
        "top_duplicate_groups": [
            {
                "count": count,
                **group_details[key],
                "representative_sample_id": representatives[key]
            }
            for key, count in frequencies.most_common(10)
        ]
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Verify the saved dataset.
    audit_unique_count = 0

    with gzip.open(unique_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = fingerprint(row)

            assert representatives[key] == row["sample_id"]
            audit_unique_count += 1

    audit_mapping_count = 0

    with gzip.open(mapping_path, "rt", encoding="utf-8") as f:
        for line in f:
            mapping = json.loads(line)

            assert mapping["representative_sample_id"] in sample_ids
            assert mapping["group_size"] == frequencies[
                mapping["fingerprint"]
            ]

            audit_mapping_count += 1

    assert audit_unique_count == unique
    assert audit_mapping_count == total

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("DEDUP AUDIT PASSED")


if __name__ == "__main__":
    main()
