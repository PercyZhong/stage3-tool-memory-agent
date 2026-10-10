from pathlib import Path

import pytest

from stage3_agent.data import FORBIDDEN_PROMPT_FIELDS, load_fixture, load_tasks, public_task
from stage3_agent.engine import assemble_prompt
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


def test_prompt_requires_grounding_and_contains_no_evaluation_targets():
    task = load_tasks(ROOT / "data" / "tasks.jsonl")[0]
    prompt, trimming = assemble_prompt(task, [], 12000)
    assert "call the relevant tools before final" in prompt
    assert '"tool_call":{"name":"find_events"' in prompt
    assert "Only cite record_id values" in prompt
    assert "CURRENT_TASK=" in prompt
    assert trimming["kept_history_items"] == 0
    for forbidden in FORBIDDEN_PROMPT_FIELDS:
        assert f'"{forbidden}"' not in prompt
