#!/usr/bin/env bash
set -euo pipefail
SEED="${1:?usage: run_pure_hgt_seed.sh SEED}"
case "$SEED" in 42|43|44) ;; *) exit 2 ;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/runs-pure-hgt/seed-$SEED" "$ROOT/logs-pure-hgt"
python "$ROOT/code-pure-hgt/scripts/train_type_aware_hgt_link.py" \
  --protocol-dir "$ROOT/protocol" --prepared-dir "$ROOT/prepared" \
  --output-dir "$ROOT/runs-pure-hgt/seed-$SEED" --experiment-name pure-hgt \
  --seed "$SEED" --hidden-dim 128 --heads 4 --layers 2 --dropout 0.2 \
  --neighbors 15 10 --batch-size 2048 --inference-batch-size 4096 \
  --epochs 30 --patience 5 --lr 0.001 --weight-decay 0.0001 --workers 0 --device cuda \
  2>&1 | tee -a "$ROOT/logs-pure-hgt/seed-$SEED.log"
