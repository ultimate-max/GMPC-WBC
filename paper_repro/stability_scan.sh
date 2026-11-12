#!/usr/bin/env bash
set -u
source /home/ganhe/project/b2z1_mpc/paper_repro/env.sh
mkdir -p results/paper_repro/diagnostics
for sigma in 100 10000; do
  for up in 10000 100000 1000000; do
    name=scan_s${sigma}_u${up}
    paper_repro/build/paper_rti "$PWD" --mode distributed --sigma "$sigma" --rho 10000 --input-prox "$up" --bootstrap 40 --duration 3 --name "$name" > "results/paper_repro/diagnostics/$name.log" 2>&1
    status=$?
    printf '%s exit=%s ' "$name" "$status"
    tail -n 1 "results/paper_repro/diagnostics/$name.log"
  done
done
