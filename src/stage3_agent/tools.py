from __future__ import annotations

from typing import Any


class ToolError(ValueError):
    pass


class CampusTools:
    def __init__(self, fixture: dict[str, Any], max_input_chars: int = 256):
        self.fixture = fixture
        self.version = fixture["fixture_version"]
        self.source = fixture["source"]
        self.max_input_chars = max_input_chars

    def call(self, name: str, arguments: dict[str, Any], allowed: list[str]) -> list[dict[str, Any]]:
        if name not in allowed:
            raise ToolError(f"tool is not allowed for this task: {name}")
        if name not in {"lookup_schedule", "find_events", "lookup_policy"}:
            raise ToolError(f"unknown tool: {name}")
        for key, value in arguments.items():
            if not isinstance(key, str) or not isinstance(value, (str, bool)):
                raise ToolError("tool arguments must be strings or booleans")
            if isinstance(value, str) and len(value) > self.max_input_chars:
                raise ToolError(f"argument too long: {key}")
        if name == "lookup_schedule":
            self._keys(arguments, {"course", "day"})
            rows = self.fixture["schedules"]
        elif name == "find_events":
            self._keys(arguments, {"day", "query", "accessible"})
            rows = self.fixture["events"]
        else:
            self._keys(arguments, {"topic"})
            rows = self.fixture["policies"]
        result = [self._record(row) for row in rows if self._matches(row, arguments)]
        return result

    @staticmethod
    def _keys(arguments: dict[str, Any], allowed: set[str]) -> None:
        unknown = set(arguments) - allowed
        if unknown:
            raise ToolError(f"unsupported arguments: {sorted(unknown)}")

    @staticmethod
    def _matches(row: dict[str, Any], arguments: dict[str, Any]) -> bool:
        for key, wanted in arguments.items():
            if key == "query":
                if str(wanted).casefold() not in str(row.get("name", "")).casefold():
                    return False
            elif str(row.get(key, "")).casefold() != str(wanted).casefold():
                return False
        return True

    def _record(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "record_id": row["record_id"],
            "fixture_version": self.version,
            "source": self.source,
            "content": row,
        }
