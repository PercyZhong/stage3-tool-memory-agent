import json
import shutil
from pathlib import Path

from stage3_agent.replay import main as replay_main
from stage3_agent.run import main as run_main
from stage3_agent.verify import main as verify_main


ROOT = Path(__file__).resolve().parents[1]


def test_scripted_run_replay_verify_and_tamper_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    run_dir = tmp_path / "run"
    replay_dir = tmp_path / "replay"
    assert run_main([
        "--backend", "scripted", "--suite", str(ROOT / "data" / "tasks.jsonl"),
        "--arm", "full", "--seed", "42", "--out", str(run_dir),
    ]) == 0
    expected = {
        "config.json", "manifest.json", "tasks.jsonl", "events.jsonl", "responses.jsonl",
        "memory_before.json", "memory_after.json", "metrics.json", "examples.md", "errors.json",
    }
    assert expected <= {path.name for path in run_dir.iterdir()}
    assert replay_main(["--run", str(run_dir), "--mode", "recorded", "--out", str(replay_dir)]) == 0
    assert verify_main(["--run", str(run_dir), "--replay", str(replay_dir)]) == 0

    tampered = tmp_path / "tampered"
    shutil.copytree(run_dir, tampered)
    events = (tampered / "events.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(events[0])
    first["status"] = "tampered"
    events[0] = json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    (tampered / "events.jsonl").write_text("\n".join(events) + "\n", encoding="utf-8")
    failed_replay = tmp_path / "failed-replay"
    assert replay_main(["--run", str(tampered), "--mode", "recorded", "--out", str(failed_replay)]) != 0
    report = json.loads((failed_replay / "replay_report.json").read_text(encoding="utf-8"))
    assert not report["ok"]
    assert any("hash mismatch" in error for error in report["errors"])


def test_all_three_scripted_arms_are_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    successes = {}
    for arm in ("full", "no_memory", "no_tools"):
        run_dir = tmp_path / arm
        assert run_main([
            "--backend", "scripted", "--suite", str(ROOT / "data" / "tasks.jsonl"),
            "--arm", arm, "--out", str(run_dir),
        ]) == 0
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        successes[arm] = metrics["successes"]
        namespaces = {item["namespace"] for item in json.loads((run_dir / "memory_after.json").read_text(encoding="utf-8"))}
        assert all(f":{arm}:" in namespace for namespace in namespaces)
    assert successes == {"full": 12, "no_memory": 8, "no_tools": 0}


def test_invalid_backend_terminates_without_inventing_final(tmp_path, monkeypatch):
    class InvalidBackend:
        name = "scripted"

        def generate(self, prompt):
            del prompt
            return "not valid JSON"

    monkeypatch.setattr("stage3_agent.run.ScriptedBackend", lambda task, arm: InvalidBackend())
    monkeypatch.chdir(ROOT)
    run_dir = tmp_path / "invalid-run"
    replay_dir = tmp_path / "invalid-replay"
    assert run_main([
        "--backend", "scripted", "--suite", str(ROOT / "data" / "tasks.jsonl"),
        "--arm", "full", "--out", str(run_dir),
    ]) == 0
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["completed"] == 12
    assert metrics["successes"] == 0
    assert all(task["final"] is None for task in metrics["tasks"])
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    terminations = [event for event in events if event["event_type"] == "termination"]
    assert len(terminations) == 12
    assert all(event["output"]["reason"] == "invalid_action_limit" for event in terminations)
    assert replay_main(["--run", str(run_dir), "--mode", "recorded", "--out", str(replay_dir)]) == 0
    assert verify_main(["--run", str(run_dir), "--replay", str(replay_dir)]) == 0
