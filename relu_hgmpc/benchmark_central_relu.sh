#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
cd /home/ganhe/project/b2z1_mpc
mkdir -p results/central_relu
# Sequential ABBA order avoids simultaneous GPU/CPU benchmark contention.
for pair in osqp:1 reluqp:1 reluqp:2 osqp:2; do
    backend=${pair%:*}
    repeat=${pair#*:}
    name=central_${backend}_12s_r${repeat}
    python relu_hgmpc/run_split_gmpc.py --mode central --backend "$backend" --duration 12 --name "$name" > "results/central_relu/$name.log" 2>&1
    printf '%s completed\n' "$name"
done
