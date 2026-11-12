#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
python relu_hgmpc/test_prioritized.py
for dim in 3 6; do
  for rep in 1 2; do
    python relu_hgmpc/run_prioritized.py --task-dim "$dim" --duration 12 --name "xie_${dim}d_final_r${rep}" > "results/prioritized/xie_${dim}d_final_r${rep}.log" 2>&1
  done
done
python relu_hgmpc/compare_prioritized.py
