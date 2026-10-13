#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/logs-pure-hgt"
exec 9>"$ROOT/logs-pure-hgt/runner.lock"
flock -n 9 || { echo "pure HGT runner already active"; exit 1; }
for seed in 42 43 44; do
  if [[ -f "$ROOT/runs-pure-hgt/seed-$seed/experiment_record.json" ]]; then
    echo "seed $seed already complete; skipping"
  else
    bash "$ROOT/run_pure_hgt_seed.sh" "$seed"
  fi
done
python "$ROOT/code-pure-hgt/scripts/summarize_pure_hgt.py" --root "$ROOT"
