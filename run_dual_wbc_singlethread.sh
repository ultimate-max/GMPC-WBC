#!/usr/bin/env bash
set -euo pipefail
cd /home/ganhe/project/b2z1_mpc
source relu_hgmpc/env.sh
method="${1:-gmpc_dual}"
scenario="${2:-normal}"
run_name="${3:-manual_${method}_${scenario}}"
python relu_hgmpc/run_dual_wbc_comparison.py --method "$method" --case "$scenario" \
    --name "$run_name" --joint-damping 5 --smoothing 1 --duration 12 --wbc-hz 200 --schedule ratio
