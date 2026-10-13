#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec 9>rank-v3.runner.lock
flock -n 9 || exit 0
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=python
mkdir -p logs-rank-v3 runs-rank-v3
"$PY" -B code/scripts/test_hgt_rank_v3.py >logs-rank-v3/numeric-tests.log 2>&1
for method in type-aware-prompt-hgt pure-hgt; do
  extra=()
  if [ "$method" = type-aware-prompt-hgt ]; then
    extra=(--features ../features/combined_features.npy --features-manifest inputs/type-aware.json)
  fi
  "$PY" -B code/scripts/train_hgt_rank_v3.py \
      --protocol-dir protocol --prepared-dir prepared \
      --output-dir "runs-rank-v3/smoke-$method" --seed 42 --experiment-name "$method" \
      "${extra[@]}" --smoke-batches 1 >>"logs-rank-v3/smoke-$method.log" 2>&1
  for seed in 42 43 44; do
    out="runs-rank-v3/$method/seed-$seed"
    if [ -f "$out/experiment_record.json" ]; then
      echo "Already completed $method $seed"
      continue
    fi
    echo "START $method seed=$seed $(date -Iseconds)"
    "$PY" -B code/scripts/train_hgt_rank_v3.py \
      --protocol-dir protocol --prepared-dir prepared --output-dir "$out" \
      --seed "$seed" --experiment-name "$method" "${extra[@]}" \
      --hidden-dim 128 --heads 4 --layers 2 --dropout 0.2 --neighbors 15 10 \
      --batch-size 2048 --inference-batch-size 4096 --epochs 30 --patience 5 \
      --lr 0.001 --weight-decay 0.0001 --workers 0 --device cuda \
      >>"logs-rank-v3/$method-seed-$seed.log" 2>&1
  done
done
"$PY" -B code/scripts/summarize_hgt_rank_v3.py --root runs-rank-v3 \
    >logs-rank-v3/summary.log 2>&1
echo "ALL_V3_RUNS_COMPLETE $(date -Iseconds)"
