#!/usr/bin/env python3
"""Portable, standard-library-only distributed V4.1 teacher labeling.

Subcommands: prepare, run, status, merge. Never sends source metadata to teacher.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from teacher_policy_v41 import POLICY_V41

POLICY_HASH = hashlib.sha256(POLICY_V41.encode("utf-8")).hexdigest()
POLICY_VERSION = "v4.1"
DEFAULT_MODEL = "qwen38-teacher"
SCHEMA = {
    "type": "object",
    "properties": {
        "authorization": {"type": "string", "enum": ["AUTHORIZED", "UNAUTHORIZED", "UNCERTAIN"]},
        "risk_level": {"type": "string", "enum": ["LOW", "HIGH"]},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["authorization", "risk_level", "risk_factors", "rationale"],
    "additionalProperties": False,
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def data_hash(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def each_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {path}, line {n}: {e}") from e


def write_gzip_jsonl(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_row(row):
    if not isinstance(row, dict) or not all(k in row for k in ("sample_id", "input", "source")):
        raise ValueError("Sample must contain sample_id, input, source")
    if row["source"].get("split") != "train":
        raise ValueError(f"Non-train row: {row['sample_id']}")
    if not all(k in row["input"] for k in ("user_task", "history", "proposed_tool_call")):
        raise ValueError(f"Incomplete input: {row['sample_id']}")


def check_label(label):
    if not isinstance(label, dict) or set(label) != set(SCHEMA["required"]):
        raise ValueError("Label must have exactly four expected fields")
    if label["authorization"] not in ("AUTHORIZED", "UNAUTHORIZED", "UNCERTAIN"):
        raise ValueError("Bad authorization")
    if label["risk_level"] not in ("LOW", "HIGH"):
        raise ValueError("Bad risk_level")
    if not isinstance(label["risk_factors"], list) or not all(isinstance(x, str) for x in label["risk_factors"]):
        raise ValueError("Bad risk_factors")
    if not isinstance(label["rationale"], str) or not label["rationale"].strip():
        raise ValueError("Empty rationale")


def resolve_label_files(values):
    files = []
    for v in values:
        p = Path(v)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
            files.extend(sorted(p.rglob("*.jsonl.gz")))
        else:
            raise FileNotFoundError(p)
    return sorted(set(files))


def cmd_prepare(a):
    if not getattr(a, "model_path", None) or not getattr(a, "server_path", None):
        raise ValueError("Coordinator must supply --model-path and --server-path (shared/readable by teammates)")
    if not (1 <= a.slurm_passes <= 15):
        raise ValueError("--slurm-passes must be 1..15")
    if a.workers < 1 or a.max_chars < 1:
        raise ValueError("workers and max-chars must be >= 1")
    if a.min_chars < 0 or a.min_chars > a.max_chars:
        raise ValueError("Invalid min-chars / max-chars")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Run directory not empty: {out}; select a NEW run directory")
    out.mkdir(parents=True, exist_ok=True)
    (out / "assignments").mkdir(exist_ok=True)
    (out / "outputs").mkdir(exist_ok=True)

    all_rows = list(each_jsonl(a.input))
    ids = set()
    by_id = {r["sample_id"]: r for r in all_rows}
    for row in all_rows:
        check_row(row)
        sid = row["sample_id"]
        if sid in ids:
            raise ValueError(f"Duplicate input sample_id: {sid}")
        ids.add(sid)

    existing = {}
    for p in resolve_label_files(a.existing_labels):
        for r in each_jsonl(p):
            if not (isinstance(r, dict) and "teacher_label" in r and "sample_id" in r):
                continue
            if r.get("policy_sha256") != POLICY_HASH:
                continue
            sid = r["sample_id"]
            if sid not in ids:
                raise ValueError(f"Existing V4.1 label not in input: {sid} in {p}")
            check_label(r["teacher_label"])
            if sid in existing:
                raise ValueError(f"Duplicate existing V4.1 sample {sid} across existing files")
            original = by_id[sid]
            if r.get("input") != original["input"]:
                raise ValueError(f"Input mismatch for existing label {sid}")
            existing[sid] = r

    shards = [[] for _ in range(a.workers)]
    deferred = []
    excluded = 0
    for row in sorted(all_rows, key=lambda r: r["sample_id"]):
        sid = row["sample_id"]
        if sid in existing:
            excluded += 1
            continue
        size = len(json.dumps(row["input"], ensure_ascii=False))
        if not (a.min_chars <= size <= a.max_chars):
            deferred.append(row)
            continue
        slot = int(hashlib.sha256(sid.encode("utf-8")).hexdigest(), 16) % a.workers
        shards[slot].append(row)

    for i, rows in enumerate(shards):
        write_gzip_jsonl(out / "assignments" / f"worker_{i:02d}.jsonl.gz", rows)
    write_gzip_jsonl(out / "deferred_context.jsonl.gz", deferred)
    write_gzip_jsonl(out / "existing_labels.jsonl.gz", [existing[sid] for sid in sorted(existing)])
    report = {
        "policy_version": POLICY_VERSION,
        "policy_sha256": POLICY_HASH,
        "source_input_sha256": file_hash(a.input),
        "all_train_samples": len(all_rows),
        "existing_v41_samples": excluded,
        "assigned_short_samples": sum(map(len, shards)),
        "deferred_outside_char_range": len(deferred),
        "min_chars": a.min_chars,
        "max_chars": a.max_chars,
        "workers": a.workers,
        "counts_by_worker": [len(s) for s in shards],
        "input_suite_counts": dict(collections.Counter(r["source"]["suite"] for r in all_rows)),
        "note": "Assignments are stable for this frozen run only; no val/test samples. Source metadata is never given to the model.",
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    # Workers submit one regular job at a time, and resubmit until remaining=0.
    # Team members never have to edit paths or specify WORKER_ID themselves.
    report["slurm_mode"] = "single_job_manual_resubmit"
    report["teacher_model_path"] = a.model_path
    report["llama_server_path"] = a.server_path
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pkg_dir = out / "team_packages"
    pkg_dir.mkdir(exist_ok=True)
    here = Path(__file__).resolve().parent
    template = (here / "worker_submit.sbatch.template").read_text(encoding="utf-8")
    for i in range(a.workers):
        pkg = pkg_dir / f"worker_{i:02d}.zip"
        sbatch = (template.replace("@@INDEX_PAD@@", f"{i:02d}")
                         .replace("@@INDEX@@", str(i))
                         .replace("@@LAST_PASS@@", str(a.slurm_passes - 1))
                         .replace("@@MODEL_PATH@@", shlex.quote(a.model_path))
                         .replace("@@SERVER_PATH@@", shlex.quote(a.server_path)))
        if "@@" in sbatch:
            raise RuntimeError("Unresolved placeholder in Slurm template")
        with zipfile.ZipFile(pkg, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name in ("team_annotation.py", "teacher_policy_v4.py", "teacher_policy_v41.py", "TEAMMATE_FULL_MANUAL_CN.md"):
                z.write(here / name, arcname=name)
            z.writestr("submit.sbatch", sbatch)
            z.write(manifest_path, arcname="run/manifest.json")
            z.write(out / "assignments" / f"worker_{i:02d}.jsonl.gz",
                    arcname=f"run/assignments/worker_{i:02d}.jsonl.gz")
            z.writestr("logs/", "")
            z.writestr("run/outputs/", "")
            z.writestr("START_HERE.md", (
                f"# AI6101 — Worker {i:02d}\n\n"
                "This archive contains only your own assigned Train samples and the frozen V4.1 policy.\n"
                "No model or executable is included; coordinator already selected the shared deployment paths.\n\n"
                "On NTU TC2, from THIS extracted folder, submit exactly one command:\n\n"
                "```bash\nsbatch submit.sbatch\n```\n\n"
                "This is a single 6-hour job, not a job array. When it ends, if remaining > 0, run sbatch submit.sbatch again.\n"
                "Once the array ends, check:\n\n"
                f"```bash\npython3 team_annotation.py remaining --run-dir run --worker {i}\n```\n\n"
                f"When it prints 0, return ONLY `run/outputs/worker_{i:02d}.jsonl` to coordinator.\n"
                "If remaining > 0 after all passes, report to coordinator for another submission.\n"
                "If server cannot start due to unreadable model, ask coordinator to fix shared access.\n"
            ))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReady-to-send ONE-COMMAND teammate packages: {pkg_dir}/worker_XX.zip")
    print("Each teammate: unzip, cd, sbatch submit.sbatch; repeat after EACH job ends until remaining=0")
    print("Do not run additional full-v41 jobs simultaneously on overlapping samples.")


def load_run(run):
    d = Path(run)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if manifest["policy_sha256"] != POLICY_HASH:
        raise ValueError("Local V4.1 prompt differs from frozen manifest")
    return d, manifest


def safe_read_completed(path, expected, policy_hash, *, repair):
    completed = {}
    path = Path(path)
    if not path.exists():
        return completed
    if repair:
        # Append-only records are fsync-ed. Only a partial final line is recoverable.
        with path.open("rb+") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    print(f"Truncating incomplete trailing line at offset {pos}: {path}")
                    f.truncate(pos)
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid complete JSON line in {path}: {e}") from e
                validate_output_record(r, expected, policy_hash)
                sid = r["sample_id"]
                if sid in completed:
                    raise ValueError(f"Duplicate label {sid}: {path}")
                completed[sid] = r
    else:
        for r in each_jsonl(path):
            validate_output_record(r, expected, policy_hash)
            sid = r["sample_id"]
            if sid in completed:
                raise ValueError(f"Duplicate label {sid}: {path}")
            completed[sid] = r
    return completed


def validate_output_record(r, expected, policy_hash):
    sid = r.get("sample_id")
    if sid not in expected:
        raise ValueError(f"Unassigned sample_id: {sid}")
    if r.get("policy_sha256") != policy_hash or r.get("policy_version") != POLICY_VERSION:
        raise ValueError(f"Wrong policy for {sid}")
    if r.get("input") != expected[sid]["input"] or r.get("source") != expected[sid]["source"]:
        raise ValueError(f"Input/source mismatch: {sid}")
    if r.get("input_sha256") != data_hash(expected[sid]["input"]):
        raise ValueError(f"Input hash mismatch: {sid}")
    check_label(r.get("teacher_label"))
    if r.get("finish_reason") != "stop":
        raise ValueError(f"Incomplete generation: {sid}")
    if not isinstance(r.get("teacher_model"), str) or not r["teacher_model"]:
        raise ValueError(f"Missing teacher model: {sid}")


def api_call(base, key, model, row, *, timeout=300, max_tokens=1280):
    inp = row["input"]
    user_content = json.dumps({
        "original_user_task": inp["user_task"],
        "pre_call_history": inp["history"],
        "proposed_tool_call": inp["proposed_tool_call"],
    }, ensure_ascii=False)
    request_data = {
        "model": model,
        "messages": [
            {"role": "system", "content": POLICY_V41},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_schema", "schema": SCHEMA},
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(request_data, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            answer = json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read(1000).decode('utf-8', 'replace')}") from e
    choice = answer["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"Unexpected finish_reason: {choice.get('finish_reason')}")
    raw = choice["message"].get("content") or ""
    label = json.loads(raw)
    check_label(label)
    return label, answer.get("usage"), round(time.monotonic() - start, 2)


def sync_append(f, value):
    f.write(json.dumps(value, ensure_ascii=False) + "\n")
    f.flush()
    os.fsync(f.fileno())


def cmd_run(a):
    run_dir, manifest = load_run(a.run_dir)
    if not (0 <= a.worker < manifest["workers"]):
        raise ValueError("Worker index out of range")
    if a.soft_seconds < 0:
        raise ValueError("--soft-seconds must be >=0")
    p = run_dir / "assignments" / f"worker_{a.worker:02d}.jsonl.gz"
    rows = list(each_jsonl(p))
    expected = {r["sample_id"]: r for r in rows}
    if len(expected) != len(rows) or len(rows) != manifest["counts_by_worker"][a.worker]:
        raise ValueError("Assignment does not match frozen manifest")
    out = run_dir / "outputs" / f"worker_{a.worker:02d}.jsonl"
    errpath = run_dir / "outputs" / f"worker_{a.worker:02d}.errors.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = safe_read_completed(out, expected, POLICY_HASH, repair=True)
    key = os.environ.get(a.api_key_env, "") if a.api_key_env else ""
    print(f"Worker {a.worker}: {len(done)}/{len(rows)} already completed. Model={a.model}", flush=True)
    start = time.monotonic()
    failures = 0
    attempted = 0
    with out.open("a", encoding="utf-8") as dst, errpath.open("a", encoding="utf-8") as err:
        for row in rows:
            sid = row["sample_id"]
            if sid in done:
                continue
            if a.soft_seconds and time.monotonic() - start >= a.soft_seconds:
                print("Soft time limit reached; safely stopping", flush=True)
                break
            result = None
            for attempt in range(a.retries):
                try:
                    result = api_call(a.api_base, key, a.model, row, timeout=a.timeout, max_tokens=a.max_tokens)
                    break
                except Exception as e:
                    problem = {"sample_id": sid, "attempt": attempt + 1, "error": repr(e)}
                    sync_append(err, problem)
                    print("ERROR", sid, repr(e), flush=True)
                    if attempt + 1 < a.retries:
                        time.sleep(2)
            attempted += 1
            if result is None:
                failures += 1
                if failures >= a.max_consecutive_errors:
                    raise RuntimeError("Too many consecutive errors; check API/model")
                continue
            failures = 0
            label, usage, elapsed = result
            r = {
                "sample_id": sid,
                "source": row["source"],
                "input": row["input"],
                "input_sha256": data_hash(row["input"]),
                "policy_version": POLICY_VERSION,
                "policy_sha256": POLICY_HASH,
                "teacher_model": a.model,
                "teacher_label": label,
                "finish_reason": "stop",
                "elapsed_seconds": elapsed,
                "usage": usage,
            }
            sync_append(dst, r)
            done[sid] = r
            print(f"[{len(done)}/{len(rows)}] {sid} "
                  f"{label['authorization']}/{label['risk_level']} {elapsed}s", flush=True)
    print(f"WORKER {a.worker}: completed={len(done)}/{len(rows)}, remaining={len(rows)-len(done)}, attempted={attempted}")
    if len(done) < len(rows):
        print("INCOMPLETE: re-run the same worker with the same output files to resume.")


def collect(run_dir, manifest):
    expected = {}
    for i in range(manifest["workers"]):
        rows = list(each_jsonl(run_dir / "assignments" / f"worker_{i:02d}.jsonl.gz"))
        if len(rows) != manifest["counts_by_worker"][i]:
            raise ValueError(f"Mismatched assignment count worker {i}")
        for r in rows:
            check_row(r)
            sid = r["sample_id"]
            if sid in expected:
                raise ValueError(f"Duplicate assignment sample: {sid}")
            expected[sid] = r
    if len(expected) != manifest["assigned_short_samples"]:
        raise ValueError("Assignment cardinality mismatch")
    new_labels = {}
    model_counts = collections.Counter()
    for i in range(manifest["workers"]):
        p = run_dir / "outputs" / f"worker_{i:02d}.jsonl"
        # A worker can only label its own assignment; verify before accepting.
        assigned = {sid: r for sid, r in expected.items() if int(hashlib.sha256(sid.encode()).hexdigest(), 16) % manifest["workers"] == i}
        records = safe_read_completed(p, assigned, POLICY_HASH, repair=False)
        for sid, r in records.items():
            if sid in new_labels:
                raise ValueError("Duplicate across worker outputs")
            new_labels[sid] = r
            model_counts[r["teacher_model"]] += 1
    existing = {}
    for r in each_jsonl(run_dir / "existing_labels.jsonl.gz"):
        sid = r["sample_id"]
        if sid in expected or sid in existing or r.get("policy_sha256") != POLICY_HASH:
            raise ValueError("Duplicate/mismatched existing label")
        check_label(r["teacher_label"])
        existing[sid] = r
    if len(existing) != manifest["existing_v41_samples"]:
        raise ValueError("Existing label cardinality mismatch")
    deferred = list(each_jsonl(run_dir / "deferred_context.jsonl.gz"))
    if len(deferred) != manifest["deferred_outside_char_range"]:
        raise ValueError("Deferred count mismatch")
    if len(deferred) + len(existing) + len(expected) != manifest["all_train_samples"]:
        raise ValueError("Input total count mismatch")
    if (set(existing) & set(new_labels)) or (set(r['sample_id'] for r in deferred) & (set(existing) | set(expected))):
        raise ValueError("Input overlap between outputs and deferred")
    missing = sorted(set(expected) - set(new_labels))
    report = {
        "policy_version": POLICY_VERSION,
        "policy_sha256": POLICY_HASH,
        "all_train_samples": manifest["all_train_samples"],
        "new_labeled": len(new_labels),
        "existing_v41_labeled": len(existing),
        "labeled_total": len(new_labels) + len(existing),
        "missing_assigned_short": len(missing),
        "deferred_outside_char_range": len(deferred),
        "coverage_percent": round((len(new_labels) + len(existing)) / manifest["all_train_samples"] * 100, 2),
        "teacher_model_counts_new": dict(model_counts),
        "authorization_counts": dict(collections.Counter(r["teacher_label"]["authorization"] for r in list(new_labels.values()) + list(existing.values()))),
        "risk_counts": dict(collections.Counter(r["teacher_label"]["risk_level"] for r in list(new_labels.values()) + list(existing.values()))),
    }
    return new_labels, existing, missing, report


def cmd_remaining(a):
    d, meta = load_run(a.run_dir)
    if not (0 <= a.worker < meta["workers"]):
        raise ValueError("Worker index out of range")
    rows = list(each_jsonl(d / "assignments" / f"worker_{a.worker:02d}.jsonl.gz"))
    expected = {r["sample_id"]: r for r in rows}
    if len(expected) != len(rows) or len(rows) != meta["counts_by_worker"][a.worker]:
        raise ValueError("Worker assignment does not match manifest")
    done = safe_read_completed(d / "outputs" / f"worker_{a.worker:02d}.jsonl",
                               expected, POLICY_HASH, repair=False)
    print(len(rows) - len(done))


def cmd_status(a):
    d, meta = load_run(a.run_dir)
    _, _, _, report = collect(d, meta)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    for i in range(meta["workers"]):
        path = d / "outputs" / f"worker_{i:02d}.jsonl"
        n = sum(1 for _ in each_jsonl(path)) if path.exists() else 0
        print(f"worker_{i:02d}: {n}/{meta['counts_by_worker'][i]}")


def cmd_merge(a):
    d, meta = load_run(a.run_dir)
    new_labels, existing, missing, report = collect(d, meta)
    if a.require_complete and missing:
        raise RuntimeError(f"Refusing incomplete merge: {len(missing)} assigned samples still missing")
    out = Path(a.out) if a.out else d / "merged_v41.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() in [(d / "existing_labels.jsonl.gz").resolve()]:
        raise ValueError("Cannot overwrite the frozen existing labels")
    labels = {**existing, **new_labels}
    tmp = out.with_name(out.name + ".tmp")
    write_gzip_jsonl(tmp, [labels[sid] for sid in sorted(labels)]) if out.name.endswith(".gz") else write_plain_jsonl(tmp, [labels[sid] for sid in sorted(labels)])
    os.replace(tmp, out)
    (d / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "missing_sample_ids.txt").write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved merged labels:", out)
    if report["deferred_outside_char_range"]:
        print("NOTE: deferred_context.jsonl.gz still needs a larger-context labeling pass.")


def write_plain_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="Create fixed non-overlapping shards")
    p.add_argument("--input", default="train_unique_unlabeled.jsonl.gz")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--min-chars", type=int, default=0)
    p.add_argument("--max-chars", type=int, default=10000)
    p.add_argument("--existing-labels", action="append", default=[], metavar="PATH")
    p.add_argument("--model-path", required=True,
                   help="Absolute GGUF path readable by all teammates on TC2")
    p.add_argument("--server-path", required=True,
                   help="Absolute CUDA llama-server executable readable by all teammates")
    p.add_argument("--slurm-passes", type=int, default=5,
                   help="Deprecated; accepted for older commands, ignored in single-job mode")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", help="Run/resume one teacher-label worker")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--api-base", default="http://127.0.0.1:18080")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key-env", default="TEACHER_API_KEY")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--max-tokens", type=int, default=1280)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--max-consecutive-errors", type=int, default=5)
    p.add_argument("--soft-seconds", type=int, default=19000)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("remaining", help="Number of uncompleted samples in a worker assignment")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--worker", type=int, required=True)
    p.set_defaults(func=cmd_remaining)

    p = sub.add_parser("status", help="Audit worker progress without API")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("merge", help="Validate and merge returned labels")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", help="Merged .jsonl or .jsonl.gz path")
    p.add_argument("--require-complete", action="store_true")
    p.set_defaults(func=cmd_merge)
    args = ap.parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
