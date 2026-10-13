#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SEED" >&2
  exit 2
fi
SEED="$1"
case "$SEED" in 42|43|44) ;; *) echo "seed must be 42, 43, or 44" >&2; exit 2 ;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$ROOT/runs-unprompted/seed-$SEED" "$ROOT/logs-unprompted"
python "$ROOT/code/scripts/train_type_aware_hgt_link.py" \
  --protocol-dir "$ROOT/protocol" \
  --prepared-dir "$ROOT/prepared" \
  --features "$ROOT/features/unprompted_combined_features.npy" \
  --output-dir "$ROOT/runs-unprompted/seed-$SEED" \
  --experiment-name unprompted-bge-m3-hgt \
  --seed "$SEED" \
  --hidden-dim 128 --heads 4 --layers 2 --dropout 0.2 \
  --neighbors 15 10 --batch-size 2048 --inference-batch-size 4096 \
  --epochs 30 --patience 5 --lr 0.001 --weight-decay 0.0001 \
  --workers 0 --device cuda \
  2>&1 | tee -a "$ROOT/logs-unprompted/seed-$SEED.log"
