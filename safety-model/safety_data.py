"""Canonical schema, loading, validation and deterministic group splitting for
safety-model training data.

Canonical sample (one JSON object per line):

    {
      "id": "...",                          # optional but recommended, unique
      "input":  {task, tool_name, tool_args, tool_metadata, trace},
      "label": {authorized, attacker_goal_match, needed_for_user_task,
                overlap, label_source, [risk], [reason]},
      "meta":  {suite, user_task, injection_task, attack_type, split, source}
    }

input serializes (see serialize_input) to exactly the user message of the
frozen gate interface (see ../接口/接口信息.md): json.dumps with
ensure_ascii=False and compact separators (",", ":").
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SERIALIZER_VERSION = "v1"
INPUT_FIELDS = ("task", "tool_name", "tool_args", "tool_metadata", "trace")
LABEL_SOURCES = ("ground_truth", "min_privilege", "env_diff")
SPLITS = ("train", "val", "test")
SOURCES = ("real_run", "synthetic", "perturbed")


def serialize_input(inp: dict) -> str:
    ordered = {k: inp[k] for k in INPUT_FIELDS}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def validate_sample(s: Any) -> list[str]:
    errs: list[str] = []

    def bad(msg: str) -> None:
        errs.append(msg)

    if not isinstance(s, dict):
        return ["sample is not an object"]
    inp, label, meta = s.get("input"), s.get("label"), s.get("meta")
    if not isinstance(inp, dict):
        bad("missing/invalid input")
        return errs
    for k in INPUT_FIELDS:
        if k not in inp:
            bad(f"input.{k} missing")
    if not isinstance(inp.get("task"), str) or not inp.get("task"):
        bad("input.task must be a non-empty string")
    if not isinstance(inp.get("tool_name"), str) or not inp.get("tool_name"):
        bad("input.tool_name must be a non-empty string")
    if not isinstance(inp.get("tool_args"), dict):
        bad("input.tool_args must be an object")
    tm = inp.get("tool_metadata")
    if not isinstance(tm, dict):
        bad("input.tool_metadata must be an object")
    else:
        for k in ("read_only", "touches_remote"):
            if not isinstance(tm.get(k), bool):
                bad(f"input.tool_metadata.{k} must be a boolean")
    tr = inp.get("trace")
    if not isinstance(tr, list):
        bad("input.trace must be a list")
    else:
        for i, call in enumerate(tr):
            if not (isinstance(call, dict) and isinstance(call.get("function"), str) and isinstance(call.get("args"), dict)):
                bad(f"input.trace[{i}] must be {{function: str, args: object}}")
    if not isinstance(label, dict):
        bad("missing/invalid label")
        return errs
    for k in ("authorized", "attacker_goal_match", "needed_for_user_task", "overlap"):
        if not isinstance(label.get(k), bool):
            bad(f"label.{k} must be a boolean")
    if label.get("label_source") not in LABEL_SOURCES:
        bad("label.label_source must be one of " + ", ".join(LABEL_SOURCES))
    agm, nut, ovl = label.get("attacker_goal_match"), label.get("needed_for_user_task"), label.get("overlap")
    if isinstance(agm, bool) and isinstance(nut, bool) and isinstance(ovl, bool) and ovl != (agm and nut):
        bad("label.overlap must equal attacker_goal_match AND needed_for_user_task")
    if "risk" in label and not (isinstance(label["risk"], (int, float)) and 0.0 <= float(label["risk"]) <= 1.0):
        bad("label.risk must be a float in [0, 1] when present")
    if "reason" in label and not isinstance(label["reason"], str):
        bad("label.reason must be a string when present")
    if "authorization" in label and label["authorization"] not in ("AUTHORIZED", "UNAUTHORIZED", "UNCERTAIN"):
        bad("label.authorization must be AUTHORIZED/UNAUTHORIZED/UNCERTAIN")
    if "risk_level" in label and label["risk_level"] not in ("LOW", "HIGH"):
        bad("label.risk_level must be LOW/HIGH")
    if "risk_factors" in label and not (isinstance(label["risk_factors"], list) and all(isinstance(x, str) for x in label["risk_factors"])):
        bad("label.risk_factors must be a list of strings")
    if "rationale" in label and not isinstance(label["rationale"], str):
        bad("label.rationale must be a string")
    if not isinstance(meta, dict):
        bad("missing/invalid meta")
        return errs
    for k in ("suite", "user_task", "injection_task", "attack_type", "split", "source"):
        if k not in meta:
            bad(f"meta.{k} missing")
    if meta.get("split") not in SPLITS:
        bad("meta.split must be one of " + ", ".join(SPLITS))
    if meta.get("source") not in SOURCES:
        bad("meta.source must be one of " + ", ".join(SOURCES))
    if "id" in s and not isinstance(s["id"], str):
        bad("id must be a string when present")
    return errs


def group_key(sample: dict) -> str:
    meta = sample.get("meta", {})
    return f"{meta.get('suite', '?')}::{meta.get('user_task', '?')}"


def _bucket(key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def assign_split(key: str, seed: int, ratios=(0.8, 0.1, 0.1)) -> str:
    """Deterministic group-level split. Same key + seed -> same split, always."""
    bucket = _bucket(key, seed)
    train_cut = int(round(ratios[0] * 100))
    val_cut = train_cut + int(round(ratios[1] * 100))
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def apply_splits(samples: list[dict], seed: int, ratios=(0.8, 0.1, 0.1)) -> list[dict]:
    for s in samples:
        s.setdefault("meta", {})["split"] = assign_split(group_key(s), seed, ratios)
    return samples


def bucket_of(label: dict) -> str:
    if label.get("overlap"):
        return "overlap"
    if label.get("attacker_goal_match"):
        return "attack"
    if label.get("needed_for_user_task"):
        return "needed"
    return "benign_other"


def dataset_stats(samples: list[dict]) -> dict:
    counter: Counter = Counter()
    for s in samples:
        counter["total"] += 1
        counter["suite:" + str(s.get("meta", {}).get("suite"))] += 1
        counter["split:" + str(s.get("meta", {}).get("split"))] += 1
        counter["source:" + str(s.get("meta", {}).get("source"))] += 1
        counter["bucket:" + bucket_of(s.get("label", {}))] += 1
        counter["authorized:" + str(bool(s.get("label", {}).get("authorized")))] += 1
    return dict(sorted(counter.items()))

