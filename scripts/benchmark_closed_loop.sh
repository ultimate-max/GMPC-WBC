#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/results/closed_loop"
for repetition in 1 2 3; do
  if [ "$repetition" = 2 ]; then methods="admm30 admm10 admm5 central"; else methods="central admm5 admm10 admm30"; fi
  for method in $methods; do
    name="bench_${method}_${repetition}"
    case "$method" in
      central) options=(--mode central --central 500) ;;
      admm5) options=(--mode admm --outer 5 --inner 100) ;;
      admm10) options=(--mode admm --outer 10 --inner 100) ;;
      admm30) options=(--mode admm --outer 30 --inner 100) ;;
    esac
    echo "Running $name (6 simulated seconds)"
    if bash "$ROOT/scripts/run_closed_loop.sh" "${options[@]}" --duration 6 --name "$name" > "$ROOT/results/closed_loop/$name.log" 2>&1; then
      tail -1 "$ROOT/results/closed_loop/$name.log"
    else
      echo "FAILED $name; retaining log and partial trace"
      tail -3 "$ROOT/results/closed_loop/$name.log"
    fi
  done
done
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate b2z1_mpc
python "$ROOT/scripts/report_closed_loop.py"
