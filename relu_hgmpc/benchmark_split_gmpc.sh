#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
mkdir -p results/split_gmpc
python relu_hgmpc/test_split_gmpc.py > results/split_gmpc/verification.log
python relu_hgmpc/run_split_gmpc.py --mode central --duration 12 --name central_12s > results/split_gmpc/central_12s.log
python relu_hgmpc/run_split_gmpc.py --mode distributed --parallel --iterations 30 --duration 12 --name distributed30_12s > results/split_gmpc/distributed30_12s.log
python relu_hgmpc/run_split_gmpc.py --mode distributed --parallel --iterations 100 --duration 12 --name distributed100_12s > results/split_gmpc/distributed100_12s.log
python relu_hgmpc/report_split_gmpc.py
