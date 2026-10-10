from __future__ import annotations

import argparse
import json
from pathlib import Path

from .canonical import read_json, read_jsonl, sha256_file
from .trace import validate_event_chain


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Verify source run against its recorded replay")
    value.add_argument("--run", type=Path, required=True)
    value.add_argument("--replay", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    errors: list[str] = []
    report_path = args.replay / "replay_report.json"
    if not report_path.is_file():
        errors.append("replay_report.json missing")
    else:
        report = read_json(report_path)
        if not report.get("ok") or report.get("error_count") != 0:
            errors.append("recorded replay reports errors")
    for name in ("events.jsonl", "responses.jsonl", "memory_after.json", "metrics.json"):
        source, replay = args.run / name, args.replay / name
        if not source.is_file() or not replay.is_file():
            errors.append(f"comparison artifact missing: {name}")
        elif sha256_file(source) != sha256_file(replay):
            errors.append(f"comparison artifact differs: {name}")
    if (args.replay / "events.jsonl").is_file():
        errors.extend(validate_event_chain(read_jsonl(args.replay / "events.jsonl")))
    result = {"ok": not errors, "error_count": len(errors), "errors": errors}
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
