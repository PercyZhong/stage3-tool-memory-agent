import pytest

from stage3_agent.actions import ActionError, parse_action


def test_action_schema_rejects_arbitrary_or_malformed_actions():
    assert parse_action('{"tool_call":{"name":"find_events","arguments":{"day":"Tuesday"}}}').kind == "tool_call"
    with pytest.raises(ActionError):
        parse_action('{"shell":"rm -rf /"}')
    with pytest.raises(ActionError):
        parse_action("not json")
    with pytest.raises(ActionError):
        parse_action('{"final":{"answer":{},"evidence_ids":[]}}')
