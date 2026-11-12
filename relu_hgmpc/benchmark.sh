#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
python relu_hgmpc/test_solver.py > results/relu_hgmpc/solver_verification.json
python relu_hgmpc/test_model.py
for iter in 15 100; do
  for mode in baseline hgmpc; do
    python relu_hgmpc/run.py --mode "$mode" --duration 12 --iterations "$iter" --name "${mode}_i${iter}_final_12s" > "results/relu_hgmpc/${mode}_i${iter}_final_12s.log" 2>&1
    tail -n 2 "results/relu_hgmpc/${mode}_i${iter}_final_12s.log"
  done
done
