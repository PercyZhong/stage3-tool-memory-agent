# Stage 3: auditable tool-and-memory agent

This repository implements a small, single-agent teaching experiment inspired by ReAct and MemGPT. It operates only on versioned, explicitly synthetic campus fixtures. It does not access a real campus service, execute arbitrary code, or require a network at inference time.

The system provides:

- a schema-checked ReAct loop with a maximum of eight actions;
- deterministic read-only campus tools;
- working-context trimming plus user-isolated SQLite long-term memory;
- canonical, hash-chained JSONL events and artifact manifests;
- strict recorded replay that does not load a model, use a network, or rerun tools;
- three arms (`full`, `no_memory`, `no_tools`) and two declared seeds (42, 123);
- 12 hand-authored synthetic tasks across multi-tool, cross-session-memory, and stale-update categories.

The scripted backend is an engineering oracle for offline tests. It is not a substitute for the evaluated model. Formal results must use `local_hf`, and must be reported as observed even when the small model fails to produce valid actions.

## Safety and data boundaries

`local_hf` accepts only an existing local directory and calls Transformers with `local_files_only=True`. Nothing downloads weights. Model prompts contain the current public task fields plus the final answer's field names and primitive types so exact scoring has a declared output contract. Target values, required evidence, expected behavior, setup facts, and scripted actions are excluded. Tools accept limited typed arguments and read local JSON only. Tool output is presented as untrusted data.

For small instruct models, `local_hf` uses a state-derived assistant prefix containing only the required JSON action wrapper and any tool name or memory operation already selected by the visible state machine. Generated arguments and answer content are not supplied by the runner; the complete prefixed response is logged and replayed through the same strict parser.

No real student data appears here. User IDs all begin with `synthetic-`. Model weights, runtime SQLite files, complete `runs/`, archives, caches, environment files, and secrets are ignored by Git.

## Windows engineering checks

From PowerShell in this repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m stage3_agent.run --backend scripted --suite data/tasks.jsonl --arm full --out runs/smoke
.\.venv\Scripts\python -m stage3_agent.replay --run runs/smoke --mode recorded --out runs/replay_smoke
.\.venv\Scripts\python -m stage3_agent.verify --run runs/smoke --replay runs/replay_smoke
```

The test suite copies and tampers with a trace and asserts that replay exits nonzero. It also runs all three scripted arms and checks their isolated memory namespaces.

## Stable command-line interface

```text
python -m stage3_agent.preflight --model-dir /absolute/local/model
python -m stage3_agent.run --backend scripted --suite data/tasks.jsonl --arm full --out runs/smoke
python -m stage3_agent.run --backend local_hf --model-dir /absolute/local/model --suite data/tasks.jsonl --arm full --seed 42 --out runs/full_s42
python -m stage3_agent.replay --run runs/full_s42 --mode recorded --out runs/replay_full_s42
python -m stage3_agent.verify --run runs/full_s42 --replay runs/replay_full_s42
python -m stage3_agent.evaluate --runs runs/full_s42 runs/no_memory_s42 runs/no_tools_s42 --out reports/s42
```

Every run writes `config.json`, `manifest.json`, `tasks.jsonl`, `events.jsonl`, `responses.jsonl`, `memory_before.json`, `memory_after.json`, `metrics.json`, `examples.md`, and `errors.json`. A run directory must be absent or empty; commands refuse to overwrite prior evidence.

## Model and commit provenance

Set `STAGE3_MODEL_DIR` to an absolute, authorized, already-present local instruction-model directory. The initial suggestion is a local snapshot corresponding to Qwen2.5-0.5B-Instruct; the code never assumes it exists. Preflight checks Python, packages, weight/tokenizer files, and CUDA before a formal run. The run config records the resolved model path, package versions, seed, decoding parameters, fixture hashes, and current Git commit.

For a formal experiment, check out the exact intended commit and preserve the resulting config/manifest. Do not edit a completed run. Use the replay and verify commands in a separate directory.

The full Windows → GitHub → Linux procedure, six-run matrix, replay, reporting, and weight-free archive commands are in [docs/LINUX_RUNBOOK.md](docs/LINUX_RUNBOOK.md). The evaluation design is in [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Interpretation limits

This is a small synthetic benchmark and a method demonstration. It neither reproduces the original ReAct/MemGPT paper results nor predicts real-campus performance. `full` is not assumed to win. Only Linux runs completed with a real local model count as model experiments; scripted Windows checks establish engineering behavior only.
