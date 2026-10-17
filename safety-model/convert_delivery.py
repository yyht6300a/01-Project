"""Convert teacher-delivered labels into canonical safety-model training data.

Input : workers/received/worker_*.jsonl (4-field teacher_label, input_sha256).
Output: data/train.jsonl, data/val.jsonl, data/test.jsonl + data/conversion_report.json

Mapping (D6 2026-10-15; target format v2 2026-10-16):
  authorization AUTHORIZED   -> authorized=True,  agm=False, nut=True   risk=0.05
  authorization UNAUTHORIZED -> authorized=False, agm=True,  nut=False  risk=0.95
  authorization UNCERTAIN    -> authorized=False, agm=False, nut=False  risk=0.50
  The trained target (prompts.answer_json) is the teacher's 4-field label:
  authorization / risk_level / risk_factors / rationale, copied verbatim.
  label.risk / label.reason are kept only for legacy stats (not trained on).
Tool metadata comes from the frozen table in ../??/????.md.
Group split: stratified per suite, sha256(seed|suite::user_task) ordering -> 80/10/10,
no task leakage between splits and every suite represented in val/test.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safety_data import group_key, load_jsonl, validate_sample  # noqa: E402

TOOL_META = {
    "banking": {
        "get_iban": {"read_only": True, "touches_remote": False},
        "get_balance": {"read_only": True, "touches_remote": False},
        "get_most_recent_transactions": {"read_only": True, "touches_remote": False},
        "get_scheduled_transactions": {"read_only": True, "touches_remote": False},
        "read_file": {"read_only": True, "touches_remote": False},
        "get_user_info": {"read_only": True, "touches_remote": False},
        "send_money": {"read_only": False, "touches_remote": True},
        "schedule_transaction": {"read_only": False, "touches_remote": True},
        "update_scheduled_transaction": {"read_only": False, "touches_remote": True},
        "update_password": {"read_only": False, "touches_remote": False},
        "update_user_info": {"read_only": False, "touches_remote": False},
    },
    "workspace": {
        "get_unread_emails": {"read_only": True, "touches_remote": False},
        "get_sent_emails": {"read_only": True, "touches_remote": False},
        "get_received_emails": {"read_only": True, "touches_remote": False},
        "get_draft_emails": {"read_only": True, "touches_remote": False},
        "search_emails": {"read_only": True, "touches_remote": False},
        "search_received_emails": {"read_only": True, "touches_remote": False},
        "search_contacts_by_name": {"read_only": True, "touches_remote": False},
        "search_contacts_by_email": {"read_only": True, "touches_remote": False},
        "get_current_day": {"read_only": True, "touches_remote": False},
        "search_calendar_events": {"read_only": True, "touches_remote": False},
        "get_day_calendar_events": {"read_only": True, "touches_remote": False},
        "search_files_by_filename": {"read_only": True, "touches_remote": False},
        "get_file_by_id": {"read_only": True, "touches_remote": False},
        "list_files": {"read_only": True, "touches_remote": False},
        "search_files": {"read_only": True, "touches_remote": False},
        "send_email": {"read_only": False, "touches_remote": True},
        "delete_email": {"read_only": False, "touches_remote": False},
        "create_calendar_event": {"read_only": False, "touches_remote": False},
        "cancel_calendar_event": {"read_only": False, "touches_remote": False},
        "reschedule_calendar_event": {"read_only": False, "touches_remote": False},
        "add_calendar_event_participants": {"read_only": False, "touches_remote": False},
        "append_to_file": {"read_only": False, "touches_remote": False},
        "create_file": {"read_only": False, "touches_remote": False},
        "delete_file": {"read_only": False, "touches_remote": False},
        "share_file": {"read_only": False, "touches_remote": True},
    },
    "travel": {
        "get_user_information": {"read_only": True, "touches_remote": False},
        "get_all_hotels_in_city": {"read_only": True, "touches_remote": False},
        "get_hotels_prices": {"read_only": True, "touches_remote": False},
        "get_rating_reviews_for_hotels": {"read_only": True, "touches_remote": False},
        "get_hotels_address": {"read_only": True, "touches_remote": False},
        "get_all_restaurants_in_city": {"read_only": True, "touches_remote": False},
        "get_cuisine_type_for_restaurants": {"read_only": True, "touches_remote": False},
        "get_restaurants_address": {"read_only": True, "touches_remote": False},
        "get_rating_reviews_for_restaurants": {"read_only": True, "touches_remote": False},
        "get_dietary_restrictions_for_all_restaurants": {"read_only": True, "touches_remote": False},
        "get_contact_information_for_restaurants": {"read_only": True, "touches_remote": False},
        "get_price_for_restaurants": {"read_only": True, "touches_remote": False},
        "check_restaurant_opening_hours": {"read_only": True, "touches_remote": False},
        "get_all_car_rental_companies_in_city": {"read_only": True, "touches_remote": False},
        "get_car_types_available": {"read_only": True, "touches_remote": False},
        "get_rating_reviews_for_car_rental": {"read_only": True, "touches_remote": False},
        "get_car_fuel_options": {"read_only": True, "touches_remote": False},
        "get_car_rental_address": {"read_only": True, "touches_remote": False},
        "get_car_price_per_day": {"read_only": True, "touches_remote": False},
        "get_flight_information": {"read_only": True, "touches_remote": False},
        "search_calendar_events": {"read_only": True, "touches_remote": False},
        "get_day_calendar_events": {"read_only": True, "touches_remote": False},
        "create_calendar_event": {"read_only": False, "touches_remote": False},
        "cancel_calendar_event": {"read_only": False, "touches_remote": False},
        "reserve_hotel": {"read_only": False, "touches_remote": True},
        "reserve_car_rental": {"read_only": False, "touches_remote": True},
        "reserve_restaurant": {"read_only": False, "touches_remote": True},
        "send_email": {"read_only": False, "touches_remote": True},
    },
    "slack": {
        "get_channels": {"read_only": True, "touches_remote": False},
        "read_channel_messages": {"read_only": True, "touches_remote": False},
        "read_inbox": {"read_only": True, "touches_remote": False},
        "get_users_in_channel": {"read_only": True, "touches_remote": False},
        "get_webpage": {"read_only": True, "touches_remote": True},
        "add_user_to_channel": {"read_only": False, "touches_remote": False},
        "remove_user_from_slack": {"read_only": False, "touches_remote": False},
        "send_direct_message": {"read_only": False, "touches_remote": True},
        "send_channel_message": {"read_only": False, "touches_remote": True},
        "invite_user_to_slack": {"read_only": False, "touches_remote": True},
        "post_webpage": {"read_only": False, "touches_remote": True},
    },
}

AUTH_FLAGS = {
    "AUTHORIZED": {"authorized": True, "attacker_goal_match": False, "needed_for_user_task": True, "overlap": False},
    "UNAUTHORIZED": {"authorized": False, "attacker_goal_match": True, "needed_for_user_task": False, "overlap": False},
    "UNCERTAIN": {"authorized": False, "attacker_goal_match": False, "needed_for_user_task": False, "overlap": False},
}
AUTH_RISK = {"AUTHORIZED": 0.05, "UNAUTHORIZED": 0.95, "UNCERTAIN": 0.50}


def first_sentence(text: str, cap: int = 180) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    match = re.match(r"^(.*?[.!?])(?:\s|$)", text)
    sentence = match.group(1) if match else text
    if len(sentence) > cap:
        sentence = sentence[:cap].rstrip(" ,;:") + "..."
    return sentence


def convert_record(rec: dict, seed: int) -> tuple[dict | None, list[str]]:
    errs: list[str] = []
    sid = rec.get("sample_id", "?")
    src = rec.get("source", {})
    inp = rec.get("input", {})
    tl = rec.get("teacher_label", {})
    suite = src.get("suite")
    call = inp.get("proposed_tool_call", {})
    tool = call.get("function")

    meta_entry = TOOL_META.get(suite, {}).get(tool)
    if meta_entry is None:
        errs.append(f"{sid}: unknown tool metadata for {suite}/{tool}")
        meta_entry = {"read_only": False, "touches_remote": False}

    auth = tl.get("authorization")
    if auth not in AUTH_FLAGS:
        errs.append(f"{sid}: bad authorization {auth!r}")
        return None, errs

    trace = []
    for msg in inp.get("history") or []:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                trace.append({"function": tc.get("function"), "args": tc.get("args") or {}})

    label = dict(AUTH_FLAGS[auth])
    label["label_source"] = "ground_truth"
    label["risk"] = AUTH_RISK[auth]
    reason = first_sentence(tl.get("rationale", ""))
    label["reason"] = reason or "No rationale provided."

    risk_level = tl.get("risk_level")
    if risk_level not in ("LOW", "HIGH"):
        errs.append(f"{sid}: bad risk_level {risk_level!r}")
        return None, errs
    label["authorization"] = auth
    label["risk_level"] = risk_level
    label["risk_factors"] = [str(f) for f in (tl.get("risk_factors") or [])]
    label["rationale"] = " ".join((tl.get("rationale") or "").split()) or "No rationale provided."

    sample = {
        "id": sid,
        "input": {
            "task": inp.get("user_task", ""),
            "tool_name": tool,
            "tool_args": call.get("args") or {},
            "tool_metadata": {"read_only": meta_entry["read_only"], "touches_remote": meta_entry["touches_remote"]},
            "trace": trace,
        },
        "label": label,
        "meta": {
            "suite": suite,
            "user_task": src.get("user_task_id"),
            "injection_task": src.get("injection_task_id"),
            "attack_type": src.get("attack_type"),
            "split": "train",
            "source": "real_run",
            "teacher_authorization": auth,
            "teacher_risk_level": tl.get("risk_level"),
        },
    }
    errs.extend(f"{sid}: {e}" for e in validate_sample(sample))
    return sample, errs


def stratified_group_splits(samples: list[dict], seed: int, ratios=(0.8, 0.1, 0.1)) -> None:
    """Assign splits per suite so val/test keep coverage of every suite."""
    import hashlib

    by_suite: dict[str, dict[str, list[dict]]] = {}
    for s in samples:
        by_suite.setdefault(s["meta"]["suite"], {}).setdefault(group_key(s), []).append(s)
    for groups in by_suite.values():
        keys = sorted(groups, key=lambda k: hashlib.sha256(f"{seed}|{k}".encode("utf-8")).hexdigest())
        n = len(keys)
        n_val = 0 if n < 3 else max(1, round(n * ratios[1]))
        n_test = 0 if n < 4 else max(1, round(n * ratios[2]))
        while n - n_val - n_test < 1:
            if n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break
        for i, key in enumerate(keys):
            if i < n - n_val - n_test:
                split = "train"
            elif i < n - n_test:
                split = "val"
            else:
                split = "test"
            for s in groups[key]:
                s["meta"]["split"] = split


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deliveries", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-uncertain", action="store_true")
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
            if sample is None:
                continue
            if args.drop_uncertain and sample["label"]["risk"] == 0.5:
                continue
            samples.append(sample)

    stratified_group_splits(samples, args.seed)

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in samples:
        by_split[s["meta"]["split"]].append(s)
    for split, rows in by_split.items():
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for s in rows:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats = {
        "seed": args.seed,
        "deliveries": dict(per_file),
        "total_records": sum(per_file.values()),
        "converted": len(samples),
        "errors": errors[:50],
        "error_count": len(errors),
        "per_split": {k: len(v) for k, v in by_split.items()},
        "suite_split": dict(Counter(f'{s["meta"]["suite"]}/{s["meta"]["split"]}' for s in samples)),
        "auth_counts": dict(Counter(s["label"]["risk"] for s in samples)),
        "teacher_risk_level": dict(Counter(s["meta"]["teacher_risk_level"] for s in samples)),
        "suite_counts": dict(Counter(s["meta"]["suite"] for s in samples)),
        "read_only_counts": dict(Counter(s["input"]["tool_metadata"]["read_only"] for s in samples)),
    }
    (out_dir / "conversion_report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=1))
    if errors:
        print(f"\nWARNING: {len(errors)} conversion errors; first 10:")
        for e in errors[:10]:
            print(" ", e)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
