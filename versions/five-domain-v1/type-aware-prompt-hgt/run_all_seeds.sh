#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for seed in 42 43 44; do
  if [[ -f "$ROOT/runs/seed-$seed/experiment_record.json" ]]; then
    echo "seed $seed already complete; skipping"
  else
    "$ROOT/run_seed.sh" "$seed" 0
  fi
done
