#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
mkdir -p results/paper_gmpc
run_case() {
  local variant="$1" repeat="$2" model="$1" scale="${3:-1}" suffix=""
  local options=()
  if [[ "$variant" == *_cached ]]; then model="${variant%_cached}"; options+=(--cached); fi
  if [[ "$scale" != 1 ]]; then suffix="_scale${scale}"; fi
  local name="aligned_${variant}${suffix}_r${repeat}"
  python relu_hgmpc/run_paper_gmpc.py --model "$model" "${options[@]}" --duration 12 --scale "$scale" --name "$name" > "results/paper_gmpc/${name}.log" 2>&1
  printf '%s\n' "finished $name"
}
# Sequential, then reversed order; no concurrent controller benchmarking.
for variant in frozen frozen_cached geometry_cached paper paper_cached; do run_case "$variant" 1; done
for variant in paper_cached paper geometry_cached frozen_cached frozen; do run_case "$variant" 2; done
run_case frozen_cached 1 2
run_case paper_cached 1 2
run_case paper_cached 2 2
run_case frozen_cached 2 2
