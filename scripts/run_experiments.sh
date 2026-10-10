#!/usr/bin/env bash
set -euo pipefail

: "${STAGE3_MODEL_DIR:?Set STAGE3_MODEL_DIR to an authorized local model directory}"
SEEDS=(42 123)
ARMS=(full no_memory no_tools)

for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    run_dir="runs/${arm}_s${seed}"
    replay_dir="runs/replay_${arm}_s${seed}"
    python -m stage3_agent.run \
      --backend local_hf \
      --model-dir "$STAGE3_MODEL_DIR" \
      --suite data/tasks.jsonl \
      --arm "$arm" \
      --seed "$seed" \
      --out "$run_dir"
    python -m stage3_agent.replay --run "$run_dir" --mode recorded --out "$replay_dir"
    python -m stage3_agent.verify --run "$run_dir" --replay "$replay_dir"
  done
  python -m stage3_agent.evaluate \
    --runs "runs/full_s${seed}" "runs/no_memory_s${seed}" "runs/no_tools_s${seed}" \
    --out "reports/s${seed}"
done

python -m stage3_agent.evaluate \
  --runs runs/full_s42 runs/no_memory_s42 runs/no_tools_s42 \
         runs/full_s123 runs/no_memory_s123 runs/no_tools_s123 \
  --out reports/all_seeds
