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


def test_action_parser_accepts_only_an_exact_single_json_fence():
    action = parse_action(
        '```json\n{"tool_call":{"name":"find_events","arguments":{"day":"Tuesday"}}}\n```'
    )
    assert action.kind == "tool_call"
    assert action.payload["arguments"] == {"day": "Tuesday"}

    rejected = [
        'prefix\n```json\n{"tool_call":{"name":"find_events","arguments":{}}}\n```',
        '```json\n{"tool_call":{"name":"find_events","arguments":{}}}\n```\nsuffix',
        '```JSON\n{"tool_call":{"name":"find_events","arguments":{}}}\n```',
        '```json\n{"tool_call":{"name":"find_events","arguments":{}}\n```\n```json\n{}\n```',
    ]
    for raw in rejected:
        with pytest.raises(ActionError, match="invalid JSON"):
            parse_action(raw)


def test_action_parser_rejects_duplicate_keys_and_invalid_fenced_actions():
    with pytest.raises(ActionError, match="duplicate JSON key: tool_call"):
        parse_action(
            '{"tool_call":{"name":"find_events","arguments":{}},'
            '"tool_call":{"name":"lookup_policy","arguments":{}}}'
        )
    with pytest.raises(ActionError, match="duplicate JSON key: day"):
        parse_action('{"tool_call":{"name":"find_events","arguments":{"day":"Tuesday","day":"Friday"}}}')
    with pytest.raises(ActionError, match="duplicate JSON key: day"):
        parse_action(
            '```json\n{"tool_call":{"name":"find_events",'
            '"arguments":{"day":"Tuesday","day":"Friday"}}}\n```'
        )
    with pytest.raises(ActionError, match="exactly one"):
        parse_action(
            '```json\n{"tool_call":{"name":"find_events","arguments":{}},'
            '"final":{"answer":{},"evidence_ids":[],"text":"x"}}\n```'
        )
    with pytest.raises(ActionError, match="exactly one"):
        parse_action('```json\n{}\n```')
