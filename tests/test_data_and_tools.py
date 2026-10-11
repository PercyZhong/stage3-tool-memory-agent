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
    photography = tools.call(
        "find_events", {"day": "Thursday", "query": "latest photography event"}, ["find_events"]
    )
    assert [row["record_id"] for row in photography] == ["EV-PHOTO-THU-R2"]
    music = tools.call("find_events", {"query": "Music Jam eligibility"}, ["find_events"])
    assert [row["record_id"] for row in music] == ["EV-MUSIC-FRI-R2"]


def test_prompt_requires_grounding_and_contains_no_evaluation_targets():
    task = load_tasks(ROOT / "data" / "tasks.jsonl")[0]
    prompt, trimming = assemble_prompt(task, [], 12000)
    assert "call the relevant tools before final" in prompt
    assert "Only cite record_id values" in prompt
    assert "CURRENT_TASK=" in prompt
    assert '"required_type":"tool_call"' in prompt
    assert '"allowed_names":["find_events","lookup_policy"]' in prompt
    assert prompt.endswith(
        '{"tool_call":{"name":"find_events","arguments":{}}}'
    )
    assert trimming["kept_history_items"] == 0
    for forbidden in FORBIDDEN_PROMPT_FIELDS:
        assert f'"{forbidden}"' not in prompt
    no_tools_prompt, _ = assemble_prompt(task, [], 12000, "no_tools")
    assert '"allowed_tools":[]' in no_tools_prompt
    no_memory_prompt, _ = assemble_prompt(task, [], 12000, "no_memory")
    assert '"persistent_memory_enabled":false' in no_memory_prompt


def test_prompt_exposes_exactly_one_state_dependent_next_action():
    tasks = load_tasks(ROOT / "data" / "tasks.jsonl")
    tool_task = tasks[0]
    after_first_tool = [{
        "tool_call": {"name": "find_events", "arguments": {"day": "Wednesday"}},
        "observation": [],
    }]
    prompt, _ = assemble_prompt(tool_task, after_first_tool, 12000)
    assert '"completed_actions":[{"arguments":{"day":"Wednesday"},"name":"find_events","type":"tool_call"}]' in prompt
    assert '"allowed_names":["lookup_policy"]' in prompt
    after_both_tools = after_first_tool + [{
        "tool_call": {"name": "lookup_policy", "arguments": {"topic": "photography"}},
        "observation": [],
    }]
    prompt, _ = assemble_prompt(tool_task, after_both_tools, 12000)
    assert '"required_type":"final"' in prompt
    assert "This turn is only the final answer" in prompt
    assert '"evidence_ids":[]' in prompt

    memory_task = next(task for task in tasks if task["task_type"] == "cross_session_memory")
    prompt, _ = assemble_prompt(memory_task, [], 12000)
    assert '"allowed_operations":["search"]' in prompt
    assert '"required_type":"memory_call"' in prompt
    assert "This turn is only a persistent-memory search" in prompt
    assert prompt.endswith(
        '{"memory_call":{"operation":"search","query":"short relevant keywords"}}'
    )
    after_search = [{
        "memory_call": {"operation": "search", "query": "accessibility preference"},
        "result": [],
    }]
    prompt, _ = assemble_prompt(memory_task, after_search, 12000)
    assert '"required_type":"tool_call"' in prompt
    assert "This turn is only a tool selection" in prompt

    update_task = next(task for task in tasks if task["task_type"] == "conflict_update")
    update_history = [{
        "memory_call": {"operation": "search", "query": "CS101 schedule"},
        "result": [],
    }, {
        "tool_call": {"name": "lookup_schedule", "arguments": {"course": "CS101"}},
        "observation": [{"record_id": "SCH-CS101-R2"}],
    }]
    prompt, _ = assemble_prompt(update_task, update_history, 12000)
    assert '"allowed_operations":["invalidate"]' in prompt
    assert "This turn is only a stale-memory invalidation" in prompt
    update_history.append({
        "memory_call": {"operation": "invalidate", "query": "CS101 schedule"},
        "result": [],
    })
    prompt, _ = assemble_prompt(update_task, update_history, 12000)
    assert '"required_type":"final"' in prompt
    assert '"evidence_ids":["SCH-CS101-R2"]' in prompt
