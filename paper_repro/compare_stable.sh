#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/paper_repro/env.sh
# Run only after stability validation and with no concurrent builds/simulations.
for run in 1 2 3; do
  if [ "$run" = 2 ]; then order='distributed central'; else order='central distributed'; fi
  for mode in $order; do
    name=latency_${mode}_${run}
    paper_repro/build/paper_rti "$PWD" --mode "$mode" --profile stable --duration 6 --name "$name" > "results/interface_audit/$name.log" 2>&1
    cp "results/paper_repro/$name.csv" results/interface_audit/
    tail -n 1 "results/interface_audit/$name.log"
  done
done
