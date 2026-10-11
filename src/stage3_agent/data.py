from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import read_json, read_jsonl

TASK_TYPES = {"multi_tool", "cross_session_memory", "conflict_update"}
TOOLS = {"lookup_schedule", "find_events", "lookup_policy"}
PUBLIC_TASK_FIELDS = {
    "task_id", "task_type", "session_id", "user_id", "fixture_version", "input", "allowed_tools",
}
FORBIDDEN_PROMPT_FIELDS = {
    "target_fields", "required_evidence_ids", "expected_behavior", "script_plan", "memory_fact",
    "setup_input", "prior_fixture_version", "setup_session_id"
}


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = read_jsonl(path)
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks: list[dict[str, Any]]) -> None:
    if len(tasks) < 12:
        raise ValueError("suite must contain at least 12 tasks")
    ids: set[str] = set()
    counts = {kind: 0 for kind in TASK_TYPES}
    required = {
        "task_id", "task_type", "session_id", "user_id", "fixture_version", "input",
        "allowed_tools", "required_evidence_ids", "target_fields", "expected_behavior", "script_plan",
    }
    for task in tasks:
        missing = sorted(required - task.keys())
        if missing:
            raise ValueError(f"{task.get('task_id', '<unknown>')}: missing {missing}")
        task_id = task["task_id"]
        if task_id in ids:
            raise ValueError(f"duplicate task_id: {task_id}")
        ids.add(task_id)
        kind = task["task_type"]
        if kind not in TASK_TYPES:
            raise ValueError(f"{task_id}: unsupported task_type {kind}")
        counts[kind] += 1
        unknown = set(task["allowed_tools"]) - TOOLS
        if unknown:
            raise ValueError(f"{task_id}: unknown allowed tools: {sorted(unknown)}")
        if not task["target_fields"] or any(
            not isinstance(key, str) or not isinstance(value, (str, bool))
            for key, value in task["target_fields"].items()
        ):
            raise ValueError(f"{task_id}: target fields must be non-empty string/boolean values")
        if not str(task["user_id"]).startswith("synthetic-"):
            raise ValueError(f"{task_id}: user_id must be explicitly synthetic")
    if any(value < 4 for value in counts.values()):
        raise ValueError(f"each task type needs at least four tasks: {counts}")


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    public = {key: task[key] for key in PUBLIC_TASK_FIELDS if key in task}
    public["answer_schema"] = {
        key: "boolean" if isinstance(value, bool) else "string"
        for key, value in task["target_fields"].items()
    }
    if FORBIDDEN_PROMPT_FIELDS & public.keys():
        raise AssertionError("evaluation data leaked into public task")
    return public


def fixture_path(repo_root: Path, version: str) -> Path:
    if version not in {"v1", "v2"}:
        raise ValueError(f"unsupported fixture version: {version}")
    return repo_root / "data" / "fixtures" / f"campus_{version}.json"


def load_fixture(repo_root: Path, version: str) -> dict[str, Any]:
    fixture = read_json(fixture_path(repo_root, version))
    if fixture.get("fixture_version") != version or fixture.get("synthetic") is not True:
        raise ValueError(f"invalid synthetic fixture: {version}")
    return fixture
