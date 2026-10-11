from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class ActionError(ValueError):
    pass


@dataclass(frozen=True)
class Action:
    kind: str
    payload: dict[str, Any]


_EXACT_JSON_FENCE = re.compile(r"\A```json[ \t]*\r?\n(?P<body>.*)\r?\n```[ \t]*\Z", re.DOTALL)


def _unwrap_exact_json_fence(raw: str) -> str:
    """Accept only a single JSON fence occupying the complete model response."""
    match = _EXACT_JSON_FENCE.fullmatch(raw)
    return match.group("body") if match else raw


def parse_action(raw: str) -> Action:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ActionError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(_unwrap_exact_json_fence(raw), object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ActionError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict) or len(value) != 1:
        raise ActionError("action must be an object with exactly one of tool_call, memory_call, final")
    kind, payload = next(iter(value.items()))
    if kind not in {"tool_call", "memory_call", "final"} or not isinstance(payload, dict):
        raise ActionError(f"unsupported action: {kind}")
    if kind == "tool_call":
        if set(payload) != {"name", "arguments"} or not isinstance(payload["name"], str) or not isinstance(payload["arguments"], dict):
            raise ActionError("tool_call requires name:string and arguments:object")
    elif kind == "memory_call":
        operation = payload.get("operation")
        if operation not in {"search", "write", "invalidate"}:
            raise ActionError("memory_call operation must be search, write, or invalidate")
        if operation in {"search", "invalidate"} and not isinstance(payload.get("query"), str):
            raise ActionError(f"memory {operation} requires query:string")
        if operation == "write" and not isinstance(payload.get("content"), dict):
            raise ActionError("memory write requires content:object")
    else:
        if not isinstance(payload.get("answer"), dict) or not isinstance(payload.get("evidence_ids"), list):
            raise ActionError("final requires answer:object and evidence_ids:list")
        if not isinstance(payload.get("text"), str):
            raise ActionError("final requires text:string")
    return Action(kind, payload)
