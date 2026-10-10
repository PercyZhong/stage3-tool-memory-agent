from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from .backend import LocalHFBackend, ScriptedBackend, backend_metadata
from .canonical import artifact_hashes, sha256_file, state_hash, write_json, write_jsonl
from .data import fixture_path, load_fixture, load_tasks
from .engine import execute_task
from .memory import MemoryStore
from .tools import CampusTools
from .trace import EventLog

ARMS = {"full", "no_memory", "no_tools"}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run the stage-three synthetic campus agent")
    value.add_argument("--backend", choices=["scripted", "local_hf"], required=True)
    value.add_argument("--model-dir", type=Path)
    value.add_argument("--suite", type=Path, required=True)
    value.add_argument("--arm", choices=sorted(ARMS), required=True)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--max-actions", type=int, default=8)
    value.add_argument("--context-char-budget", type=int, default=12000)
    value.add_argument("--max-new-tokens", type=int, default=512)
    return value


def _repo_root(suite: Path) -> Path:
    path = suite.resolve()
    if path.parent.name == "data":
        return path.parent.parent
    return Path.cwd().resolve()


def _commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def _versions() -> dict[str, str]:
    names = ["torch", "transformers", "accelerate"]
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def aggregate(task_metrics: list[dict[str, Any]], arm: str, seed: int) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = {}
    for row in task_metrics:
        group = groups.setdefault(row["task_type"], {"attempted": 0, "completed": 0, "success": 0, "evidence_valid": 0, "failures": 0})
        group["attempted"] += 1
        group["completed"] += int(row["completed"])
        group["success"] += int(row["success"])
        group["evidence_valid"] += int(row["evidence_valid"])
        group["failures"] += int(not row["success"])
    return {
        "arm": arm, "seed": seed, "attempted": len(task_metrics),
        "completed": sum(int(row["completed"]) for row in task_metrics),
        "successes": sum(int(row["success"]) for row in task_metrics),
        "evidence_valid": sum(int(row["evidence_valid"]) for row in task_metrics),
        "by_task_type": groups, "tasks": task_metrics,
    }


def _examples(
    tasks: list[dict[str, Any]], responses: list[dict[str, Any]], metrics: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> str:
    metric_by_id = {row["task_id"]: row for row in metrics}
    explanations = {
        "memory_write": "persist a setup or model-requested memory",
        "model_request": "assemble the bounded, target-free prompt",
        "model_response": "record the backend's raw response",
        "parsed_action": "parse and schema-check exactly one action",
        "tool_observation": "record the deterministic local tool result",
        "tool_error": "record a rejected or disabled tool call",
        "memory_search": "retrieve ranked active records for this user namespace",
        "memory_invalidate": "mark matching stale records invalid without overwriting them",
        "memory_error": "record a disabled memory operation",
        "error": "record an invalid or duplicate action",
        "final_answer": "record structured fields, citations, and user text",
        "termination": "record an explicit non-success termination",
    }
    lines = [
        "# Trace examples", "",
        "These are synthetic tasks. Every event for each selected task is listed below; full canonical inputs, outputs, hashes, and source IDs are in `events.jsonl`.", "",
    ]
    for task_type in ("multi_tool", "cross_session_memory", "conflict_update"):
        candidates = [task for task in tasks if task["task_type"] == task_type]
        success = next((task for task in candidates if metric_by_id[task["task_id"]]["success"]), None)
        failure = next((task for task in candidates if not metric_by_id[task["task_id"]]["success"]), None)
        lines.extend([f"## {task_type}", ""])
        for label, task in (("success", success), ("failure/boundary", failure)):
            if task is None:
                lines.extend([f"- {label}: none observed in this run.", ""])
                continue
            task_responses = [row for row in responses if row["task_id"] == task["task_id"]]
            lines.extend([
                f"### {label}: `{task['task_id']}`", "",
                f"- Input: {task['input']}",
                f"- Model steps: {len(task_responses)}",
                f"- Success: {metric_by_id[task['task_id']]['success']}",
                f"- Evidence valid: {metric_by_id[task['task_id']]['evidence_valid']}", "",
            ])
            task_events = [event for event in events if event["task_id"] == task["task_id"]]
            lines.append("Complete event sequence:")
            lines.append("")
            for event in task_events:
                explanation = explanations.get(event["event_type"], "record the state-machine transition")
                lines.append(f"1. seq `{event['seq']}` · `{event['event_type']}` · `{event['status']}` — {explanation}.")
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.backend == "local_hf" and args.model_dir is None:
        parser().error("--model-dir is required for local_hf")
    if args.out.exists() and any(args.out.iterdir()):
        print(f"refusing to overwrite non-empty output directory: {args.out}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    suite = args.suite.resolve()
    repo_root = _repo_root(suite)
    tasks = load_tasks(suite)
    run_id = str(uuid.uuid4())
    log = EventLog(run_id)
    responses: list[dict[str, Any]] = []
    task_metrics: list[dict[str, Any]] = []
    all_memory_after: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    memory_path = args.out / "memory.sqlite3"

    shared_backend = None
    if args.backend == "local_hf":
        shared_backend = LocalHFBackend(args.model_dir.resolve(), args.seed, args.max_new_tokens)

    config = {
        "run_id": run_id, "backend": args.backend, "arm": args.arm, "seed": args.seed,
        "suite": str(suite), "model": backend_metadata(shared_backend, args.model_dir) if shared_backend else {"backend": "scripted"},
        "max_actions": args.max_actions, "context_char_budget": args.context_char_budget,
        "max_new_tokens": args.max_new_tokens, "temperature": 0.0, "code_commit": _commit(repo_root),
        "environment": _versions(), "fixture_timeline": "task-declared v1/v2; identical suite order for every arm",
    }
    write_json(args.out / "config.json", config)
    shutil.copyfile(suite, args.out / "tasks.jsonl")
    write_json(args.out / "memory_before.json", [])

    for task in tasks:
        namespace = f"{run_id}:{args.arm}:{task['user_id']}"
        memory = MemoryStore(memory_path, namespace)
        setup_written = False
        try:
            if args.arm != "no_memory" and task.get("memory_fact"):
                setup_content = dict(task["memory_fact"])
                setup_content["setup_input"] = task["setup_input"]
                item = memory.write(setup_content, len(log.events) + 1, f"{task['task_id']}:{task['setup_session_id']}")
                log.add(
                    task, "memory_write",
                    {"setup_session_id": task["setup_session_id"], "input": task["setup_input"]}, item,
                    source_ids=[item["memory_id"]],
                )
                setup_written = True
            fixture = load_fixture(repo_root, task["fixture_version"])
            tools = CampusTools(fixture)
            backend = shared_backend if shared_backend is not None else ScriptedBackend(task, args.arm)
            final, metrics = execute_task(
                task, backend, tools, memory, log, responses, args.arm,
                args.max_actions, args.context_char_budget,
            )
            task_metrics.append(metrics | {"final": final})
            if setup_written:
                task_metrics[-1]["memory_writes"] += 1
            all_memory_after.extend(memory.snapshot())
        except Exception as exc:
            error = {"task_id": task["task_id"], "type": type(exc).__name__, "message": str(exc)}
            errors.append(error)
            log.add(task, "termination", {}, error, status="error")
            task_metrics.append({
                "task_id": task["task_id"], "task_type": task["task_type"], "success": False,
                "evidence_valid": False, "tool_calls": 0, "memory_reads": 0, "memory_writes": 0,
                "stale_fact_error": False, "invalid_action": 0, "step_limit": False,
                "latency_seconds": 0.0, "completed": False, "final": None,
            })
        finally:
            memory.close()

    log.write(args.out / "events.jsonl")
    write_jsonl(args.out / "responses.jsonl", responses)
    write_json(args.out / "memory_after.json", all_memory_after)
    metrics = aggregate(task_metrics, args.arm, args.seed)
    write_json(args.out / "metrics.json", metrics)
    write_json(args.out / "errors.json", errors)
    (args.out / "examples.md").write_text(_examples(tasks, responses, task_metrics, log.events), encoding="utf-8")
    memory_path.unlink(missing_ok=True)

    fixture_files = sorted({fixture_path(repo_root, task["fixture_version"]) for task in tasks})
    source_hashes = {str(path.relative_to(repo_root)).replace("\\", "/"): sha256_file(path) for path in fixture_files}
    source_hashes[str(suite.relative_to(repo_root)).replace("\\", "/")] = sha256_file(suite)
    artifacts = ["config.json", "tasks.jsonl", "events.jsonl", "responses.jsonl", "memory_before.json", "memory_after.json", "metrics.json", "examples.md", "errors.json"]
    manifest = {
        "schema_version": 1, "run_id": run_id, "source_hashes": source_hashes,
        "artifact_hashes": artifact_hashes(args.out, artifacts),
        "event_count": len(log.events), "response_count": len(responses),
        "task_ids": [task["task_id"] for task in tasks],
        "fixture_set_hash": state_hash(source_hashes),
    }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps({"run": str(args.out), "completed": metrics["completed"], "attempted": metrics["attempted"], "successes": metrics["successes"], "errors": len(errors)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
