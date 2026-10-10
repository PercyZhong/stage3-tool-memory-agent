from pathlib import Path

import pytest

from stage3_agent.data import FORBIDDEN_PROMPT_FIELDS, load_fixture, load_tasks, public_task
from stage3_agent.tools import CampusTools, ToolError


ROOT = Path(__file__).resolve().parents[1]


def test_suite_has_four_tasks_per_category_and_no_prompt_leakage():
    tasks = load_tasks(ROOT / "data" / "tasks.jsonl")
    counts = {}
    for task in tasks:
        counts[task["task_type"]] = counts.get(task["task_type"], 0) + 1
        assert not (FORBIDDEN_PROMPT_FIELDS & public_task(task).keys())
    assert counts == {"multi_tool": 4, "cross_session_memory": 4, "conflict_update": 4}


def test_tools_are_local_filtered_and_versioned():
    tools = CampusTools(load_fixture(ROOT, "v2"))
    rows = tools.call("lookup_schedule", {"course": "CS101"}, ["lookup_schedule"])
    assert rows[0]["record_id"] == "SCH-CS101-R2"
    assert rows[0]["fixture_version"] == "v2"
    with pytest.raises(ToolError):
        tools.call("find_events", {}, ["lookup_schedule"])
    with pytest.raises(ToolError):
        tools.call("lookup_schedule", {"course": "x" * 257}, ["lookup_schedule"])
