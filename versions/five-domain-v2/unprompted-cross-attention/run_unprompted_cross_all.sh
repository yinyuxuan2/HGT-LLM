#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$ROOT")"
exec 9>"$BASE/runner.lock"
flock -n 9 || exit 0
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=python
mkdir -p "$ROOT/logs" "$ROOT/runs"
cd "$BASE"
"$PY" -B code/scripts/verify_deployment_v2.py
cd "$ROOT"
"$PY" -B code/scripts/test_cross_attention.py >logs/unit-tests.log 2>&1
if [ ! -f logs/smoke-passed ]; then
  "$PY" -B code/scripts/smoke_cross_attention.py --fixtures "$BASE/smoke-test" --output "$ROOT/synthetic-smoke" >logs/smoke.log 2>&1
  touch logs/smoke-passed
fi
for seed in 42 43 44; do
  out="$ROOT/runs/seed-$seed"
  if [ -f "$out/experiment_record.json" ]; then
    "$PY" -B "$BASE/code/scripts/verify_deployment_v2.py" --record "$out/experiment_record.json"
    continue
  fi
  echo "START unprompted-cross-hgt seed=$seed $(date -Iseconds)"
  "$PY" -B code/scripts/train_hgt_cross_attention_v2.py \
    --protocol-dir "$BASE/protocol" --prepared-dir "$BASE/prepared" \
    --features "$BASE/../features/unprompted_combined_features.npy" --features-manifest "$BASE/inputs/unprompted.json" \
    --output-dir "$out" --seed "$seed" --experiment-name unprompted-cross-hgt \
    --hidden-dim 128 --heads 4 --layers 2 --dropout 0.2 --neighbors 15 10 \
    --batch-size 2048 --inference-batch-size 4096 --epochs 30 --patience 5 \
    --lr 0.001 --weight-decay 0.0001 --workers 0 --device cuda \
    --cross-init 0.5 --cross-pack-budget 8192 >>"logs/seed-$seed.log" 2>&1
  "$PY" -B "$BASE/code/scripts/verify_deployment_v2.py" --record "$out/experiment_record.json"
done
echo "UNPROMPTED_CROSS_THREE_SEEDS_COMPLETE $(date -Iseconds)"
