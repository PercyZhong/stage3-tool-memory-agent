from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import read_json, write_json


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Aggregate completed experiment runs")
    value.add_argument("--runs", nargs="+", type=Path, required=True)
    value.add_argument("--out", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for run in args.runs:
        path = run / "metrics.json"
        if not path.is_file():
            missing.append(str(run))
            continue
        metrics = read_json(path)
        for task_type, values in metrics["by_task_type"].items():
            rows.append({"run": str(run), "arm": metrics["arm"], "seed": metrics["seed"], "task_type": task_type, **values})
    report = {"runs_requested": len(args.runs), "runs_loaded": len(args.runs) - len(missing), "missing_runs": missing, "groups": rows}
    write_json(args.out / "summary.json", report)
    with (args.out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["run", "arm", "seed", "task_type", "attempted", "completed", "success", "evidence_valid", "failures"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Experiment summary", "", f"Loaded {report['runs_loaded']} of {report['runs_requested']} requested runs.", ""]
    if missing:
        lines.append("Missing runs are retained as missing rather than excluded: " + ", ".join(missing))
    for row in rows:
        lines.append(f"- {row['arm']} seed={row['seed']} {row['task_type']}: success {row['success']}/{row['attempted']}, evidence {row['evidence_valid']}/{row['attempted']}, completed {row['completed']}/{row['attempted']}")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
