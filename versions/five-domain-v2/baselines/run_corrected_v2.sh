#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec 9>runner.lock
flock -n 9 || exit 0
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=python
mkdir -p logs runs
if [ ! -f protocol/manifest.json ]; then
  "$PY" -B code/scripts/revise_protocol_v2.py --source ../protocol --output protocol >logs/protocol.log 2>&1
fi
"$PY" -B code/scripts/verify_deployment_v2.py
"$PY" -B code/scripts/test_training_evaluation_v2.py >logs/numeric-tests.log 2>&1
if [ ! -f logs/smoke-passed ]; then
  "$PY" -B code/scripts/smoke_corrected_v2.py --output smoke-test >logs/smoke-test.log 2>&1
  touch logs/smoke-passed
fi
for method in pure-hgt unprompted-bge-m3-hgt type-aware-prompt-hgt uniform-prompt-bge-m3-hgt; do
  extra=()
  case "$method" in
    unprompted-bge-m3-hgt) extra=(--features ../features/unprompted_combined_features.npy --features-manifest inputs/unprompted.json);;
    type-aware-prompt-hgt) extra=(--features ../features/combined_features.npy --features-manifest inputs/type-aware.json);;
    uniform-prompt-bge-m3-hgt)
      if [ ! -f inputs/uniform.ready ]; then
        echo 'WAITING_FOR_UNIFORM_FEATURES: first three methods completed; restart runner after verified upload.'
        exit 0
      fi
      extra=(--features inputs/uniform_combined_features.npy --features-manifest inputs/uniform.json);;
  esac
  for seed in 42 43 44; do
    out="runs/$method/seed-$seed"
    if [ -f "$out/experiment_record.json" ]; then
      "$PY" -B code/scripts/verify_deployment_v2.py --record "$out/experiment_record.json"
      continue
    fi
    echo "START $method seed=$seed $(date -Iseconds)"
    "$PY" -B code/scripts/train_hgt_corrected_v2.py \
      --protocol-dir protocol --prepared-dir prepared --output-dir "$out" \
      --seed "$seed" --experiment-name "$method" "${extra[@]}" \
      --hidden-dim 128 --heads 4 --layers 2 --dropout 0.2 --neighbors 15 10 \
      --batch-size 2048 --inference-batch-size 4096 --epochs 30 --patience 5 \
      --lr 0.001 --weight-decay 0.0001 --workers 0 --device cuda \
      >>"logs/$method-seed-$seed.log" 2>&1
    "$PY" -B code/scripts/verify_deployment_v2.py --record "$out/experiment_record.json"
  done
done
echo "ALL_TWELVE_RUNS_COMPLETE $(date -Iseconds)"
