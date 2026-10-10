# Linux runbook (Python 3.11, ordinary user, offline model)

These commands do not require administrator access and do not require GitHub CLI on Linux. They assume the repository itself is reachable by `git`; model loading never contacts Hugging Face.

## 1. Clone and pin the code

```bash
cd "$HOME"
git clone https://github.com/PercyZhong/stage3-tool-memory-agent.git
cd stage3-tool-memory-agent
git pull --ff-only origin main
git rev-parse HEAD
git status --short
```

Record the printed commit. For strict reproducibility, later reruns should check out that exact SHA.

## 2. Create a user environment

First inspect the available interpreter and already-installed scientific packages:

```bash
command -v python3.11
python3.11 --version
python3.11 -c 'import sys; print(sys.executable)'
python3.11 -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Use the server's existing user-visible packages to avoid redownloading its CUDA build:

```bash
mkdir -p "$HOME/.venvs"
python3.11 -m venv --system-site-packages "$HOME/.venvs/stage3-agent"
source "$HOME/.venvs/stage3-agent/bin/activate"
python -m pip install --no-deps -e .
python -c 'import pytest, torch, transformers, accelerate; print(pytest.__version__, torch.__version__, transformers.__version__, accelerate.__version__)'
```

If the final import fails, stop and record the missing package. Install only from an experiment-lab-approved local wheelhouse, if one exists; for example:

```bash
python -m pip install --no-index --find-links /absolute/approved/wheelhouse 'pytest>=8.2,<9' 'transformers>=4.46,<5' 'accelerate>=1,<2'
```

Do not request system installation and do not silently fall back to an online model download.

## 3. Run no-model checks

```bash
python -m pytest -q
python -m stage3_agent.run --backend scripted --suite data/tasks.jsonl --arm full --out runs/smoke
python -m stage3_agent.replay --run runs/smoke --mode recorded --out runs/replay_smoke
python -m stage3_agent.verify --run runs/smoke --replay runs/replay_smoke
```

## 4. Select one free GPU and preflight the local model

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=<ONE_FREE_GPU_INDEX>
export STAGE3_MODEL_DIR=/absolute/authorized/local/model/snapshot
test -d "$STAGE3_MODEL_DIR"
python -m stage3_agent.preflight --model-dir "$STAGE3_MODEL_DIR"
```

Do not continue if preflight is nonzero. A missing model directory is an expected stop condition, not permission to download from Hugging Face. The preflight requires Python 3.11, local config/tokenizer/weight files, `torch`, `transformers`, `accelerate`, and CUDA.

## 5. Capture the environment

```bash
{
  date -u
  git rev-parse HEAD
  command -v python
  python --version
  python -m pip freeze
  nvidia-smi
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'STAGE3_MODEL_DIR=%s\n' "$STAGE3_MODEL_DIR"
} > environment.txt
```

Review `environment.txt` before archiving and remove any unexpected sensitive path component. It must not contain tokens or environment secrets.

## 6. Run the fixed 2-seed × 3-arm matrix

The checked-in script runs each model arm, immediately performs recorded replay and verification, summarizes each seed, then writes an all-seed summary:

```bash
bash scripts/run_experiments.sh
```

Equivalent run names are:

```text
runs/full_s42          runs/no_memory_s42          runs/no_tools_s42
runs/full_s123         runs/no_memory_s123         runs/no_tools_s123
```

The script stops at the first failed run/replay/verification. Do not delete or selectively omit a failure. Inspect the run's `errors.json`, `events.jsonl`, `metrics.json`, and replay report, record the cause, and only resume into new output directories after resolving it.

## 7. Manual replay, verification, and summary commands

For a single completed run:

```bash
python -m stage3_agent.replay --run runs/full_s42 --mode recorded --out runs/replay_full_s42
python -m stage3_agent.verify --run runs/full_s42 --replay runs/replay_full_s42
```

For per-seed and combined summaries:

```bash
python -m stage3_agent.evaluate --runs runs/full_s42 runs/no_memory_s42 runs/no_tools_s42 --out reports/s42
python -m stage3_agent.evaluate --runs runs/full_s123 runs/no_memory_s123 runs/no_tools_s123 --out reports/s123
python -m stage3_agent.evaluate --runs runs/full_s42 runs/no_memory_s42 runs/no_tools_s42 runs/full_s123 runs/no_memory_s123 runs/no_tools_s123 --out reports/all_seeds
```

## 8. Inspect and archive without weights

```bash
find runs reports -type f -maxdepth 3 -print | sort
find runs -name manifest.json -print -exec sha256sum {} \;
bash scripts/archive_results.sh
```

The archive is written under `$HOME/stage3-archives/` by default, followed by a `.sha256` file. It contains only `runs/`, `reports/`, and the reviewed `environment.txt`; it does not include the model directory or cache. Preserve both files and report their paths and digest.
