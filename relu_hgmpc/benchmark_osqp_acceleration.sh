#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
mkdir -p results/osqp_acceleration
run_case() {
  local method="$1" scale="$2" repeat="$3"
  local name="${method}_scale${scale}_r${repeat}"
  python relu_hgmpc/run_osqp_acceleration.py --optimization "$method" --scale "$scale" --name "$name" > "results/osqp_acceleration/${name}.log" 2>&1
  printf '%s\n' "finished $name"
}
# Sequential/reversed order. Every run has exactly the same model and tolerance.
for scale in 1 2; do
  for method in baseline reuse eliminate combined; do run_case "$method" "$scale" 1; done
  for method in combined eliminate reuse baseline; do run_case "$method" "$scale" 2; done
done
