from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .actions import ActionError, parse_action
from .canonical import artifact_hashes, read_json, read_jsonl, sha256_file, state_hash, write_json
from .trace import validate_event_chain

REQUIRED = [
    "config.json", "manifest.json", "tasks.jsonl", "events.jsonl", "responses.jsonl",
    "memory_before.json", "memory_after.json", "metrics.json", "examples.md", "errors.json",
]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Replay a recorded run without model, network, or live tools")
    value.add_argument("--run", type=Path, required=True)
    value.add_argument("--mode", choices=["recorded"], default="recorded")
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--from-task")
    return value


def _validate_hashes(run: Path, manifest: dict[str, Any], errors: list[str]) -> None:
    for name in REQUIRED:
        if not (run / name).is_file():
            errors.append(f"missing required source file: {name}")
    for name, expected in manifest.get("artifact_hashes", {}).items():
        path = run / name
        if not path.is_file():
            errors.append(f"manifest artifact missing: {name}")
        elif sha256_file(path) != expected:
            errors.append(f"artifact hash mismatch: {name}")
    for name, expected in manifest.get("source_hashes", {}).items():
        path = Path.cwd() / name
        if not path.is_file():
            errors.append(f"source input missing from checkout: {name}")
        elif sha256_file(path) != expected:
            errors.append(f"source input hash mismatch: {name}")


def validate_recording(run: Path, from_task: str | None = None) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    missing = [name for name in REQUIRED if not (run / name).is_file()]
    if missing:
        return [f"missing required source file: {name}" for name in missing], {"checked_events": 0, "checked_responses": 0}
    manifest = read_json(run / "manifest.json")
    _validate_hashes(run, manifest, errors)
    events = read_jsonl(run / "events.jsonl")
    responses = read_jsonl(run / "responses.jsonl")
    errors.extend(validate_event_chain(events))
    if len(events) != manifest.get("event_count"):
        errors.append("event_count differs from manifest")
    if len(responses) != manifest.get("response_count"):
        errors.append("response_count differs from manifest")

    response_position = 0
    reconstructed_memory: dict[str, dict[str, Any]] = {
        item["memory_id"]: item for item in read_json(run / "memory_before.json")
    }
    checked_events = 0
    active = from_task is None
    for index, event in enumerate(events):
        if event["event_type"] == "memory_write":
            item = event["output"]
            if isinstance(item, dict) and "memory_id" in item:
                reconstructed_memory[item["memory_id"]] = item
        elif event["event_type"] == "memory_invalidate":
            for item in event["output"]:
                reconstructed_memory[item["memory_id"]] = item
        if from_task and event["task_id"] == from_task:
            active = True
        if not active:
            if event["event_type"] == "model_response":
                response_position += 1
            continue
        checked_events += 1
        if event["event_type"] != "model_response":
            continue
        if response_position >= len(responses):
            errors.append(f"event {event['seq']}: no saved response remains")
            continue
        saved = responses[response_position]
        response_position += 1
        raw = event["output"].get("raw_response")
        if saved.get("raw_response") != raw or saved.get("task_id") != event["task_id"]:
            errors.append(f"event {event['seq']}: saved response mismatch")
        previous = events[index - 1] if index else None
        if previous is None or previous["event_type"] != "model_request":
            errors.append(f"event {event['seq']}: missing preceding model_request")
        elif saved.get("prompt_hash") != state_hash(previous["input"].get("prompt")):
            errors.append(f"event {event['seq']}: prompt hash mismatch")
        next_event = events[index + 1] if index + 1 < len(events) else None
        try:
            action = parse_action(raw)
        except ActionError:
            if next_event is None or next_event["event_type"] != "error":
                errors.append(f"event {event['seq']}: invalid response lacks error event")
            continue
        if next_event is None or next_event["event_type"] != "parsed_action":
            errors.append(f"event {event['seq']}: parsed_action missing")
            continue
        if next_event["output"] != {"kind": action.kind, "payload": action.payload}:
            errors.append(f"event {event['seq']}: parser output mismatch")
        execution = events[index + 2] if index + 2 < len(events) else None
        expected_types = {
            "tool_call": {"tool_observation", "tool_error"},
            "memory_call": {"memory_search", "memory_write", "memory_invalidate", "memory_error"},
            "final": {"final_answer"},
        }[action.kind]
        if execution is None or execution["event_type"] not in expected_types:
            errors.append(f"event {event['seq']}: action has no matching recorded transition")
        elif action.kind != "final" and execution["input"] != action.payload:
            errors.append(f"event {event['seq']}: recorded action arguments differ")
        elif action.kind == "final" and execution["output"] != action.payload:
            errors.append(f"event {event['seq']}: recorded final answer differs")

    if response_position != len(responses):
        errors.append(f"unconsumed saved responses: {len(responses) - response_position}")
    if from_task and not active:
        errors.append(f"from-task not found: {from_task}")
    expected_memory = read_json(run / "memory_after.json")
    actual_memory = sorted(reconstructed_memory.values(), key=lambda item: (item["namespace"], item["created_index"], item["memory_id"]))
    normalized_expected = sorted(expected_memory, key=lambda item: (item["namespace"], item["created_index"], item["memory_id"]))
    if actual_memory != normalized_expected:
        errors.append("reconstructed memory snapshot differs from memory_after.json")
    metrics = read_json(run / "metrics.json")
    finals = {event["task_id"]: event["output"] for event in events if event["event_type"] == "final_answer"}
    for task in metrics.get("tasks", []):
        if task.get("final") != finals.get(task["task_id"]):
            errors.append(f"final answer mismatch for {task['task_id']}")
    return errors, {"checked_events": checked_events, "checked_responses": response_position, "reconstructed_memories": len(actual_memory)}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"refusing to overwrite non-empty replay directory: {args.out}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        errors, counts = validate_recording(args.run.resolve(), args.from_task)
    except Exception as exc:
        errors, counts = [f"replay exception: {type(exc).__name__}: {exc}"], {"checked_events": 0, "checked_responses": 0}
    report = {"mode": "recorded", "source_run": str(args.run.resolve()), "ok": not errors, "error_count": len(errors), "errors": errors, **counts}
    write_json(args.out / "replay_report.json", report)
    if not errors:
        for name in REQUIRED:
            shutil.copyfile(args.run / name, args.out / name)
        copied = [name for name in REQUIRED if name != "manifest.json"] + ["replay_report.json"]
        write_json(args.out / "replay_manifest.json", {"artifact_hashes": artifact_hashes(args.out, copied), "source_manifest_hash": sha256_file(args.run / "manifest.json")})
    print(json.dumps(report, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
