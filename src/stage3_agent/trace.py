from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import canonical_json, state_hash, write_jsonl

GENESIS_HASH = state_hash([])


class EventLog:
    """Append-only, hash-chained event log using logical time."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self._audit_state: list[dict[str, Any]] = []

    def add(
        self,
        task: dict[str, Any],
        event_type: str,
        input_value: Any,
        output_value: Any,
        status: str = "ok",
        source_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        seq = len(self.events) + 1
        core = {
            "run_id": self.run_id,
            "task_id": task["task_id"],
            "session_id": task["session_id"],
            "seq": seq,
            "event_type": event_type,
            "timestamp_or_logical_index": seq,
            "input": input_value,
            "output": output_value,
            "status": status,
            "source_ids": source_ids or [],
        }
        pre_hash = state_hash(self._audit_state)
        self._audit_state.append(core)
        event = core | {"pre_state_hash": pre_hash, "post_state_hash": state_hash(self._audit_state)}
        self.events.append(event)
        return event

    def write(self, path: Path) -> None:
        write_jsonl(path, self.events)


def validate_event_chain(events: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    audit_state: list[dict[str, Any]] = []
    required = {
        "run_id", "task_id", "session_id", "seq", "event_type", "timestamp_or_logical_index",
        "input", "output", "status", "source_ids", "pre_state_hash", "post_state_hash"
    }
    for expected_seq, event in enumerate(events, 1):
        missing = required - event.keys()
        if missing:
            errors.append(f"event {expected_seq}: missing fields {sorted(missing)}")
            continue
        if event["seq"] != expected_seq or event["timestamp_or_logical_index"] != expected_seq:
            errors.append(f"event {expected_seq}: non-contiguous sequence/logical index")
        expected_pre = state_hash(audit_state)
        if event["pre_state_hash"] != expected_pre:
            errors.append(f"event {expected_seq}: pre_state_hash mismatch")
        core = {key: event[key] for key in event if key not in {"pre_state_hash", "post_state_hash"}}
        audit_state.append(core)
        expected_post = state_hash(audit_state)
        if event["post_state_hash"] != expected_post:
            errors.append(f"event {expected_seq}: post_state_hash mismatch")
    return errors


def safe_event_preview(event: dict[str, Any]) -> str:
    return canonical_json({key: event[key] for key in ("seq", "task_id", "event_type", "status")})
