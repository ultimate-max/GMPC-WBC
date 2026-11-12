#!/usr/bin/env bash
set -u
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
mkdir -p results/floating_base
# Serialized and alternated: physical trajectories repeat deterministically;
# elapsed times are timing repeats, not independent robustness trials.
failed=0
for scale in 1 2; do
  for repeat in 1 2; do
    if [ "$repeat" = 1 ]; then
      methods="wbc_threshold wbc_hysteresis hgmpc"
    else
      methods="hgmpc wbc_hysteresis wbc_threshold"
    fi
    for method in $methods; do
      name="${method}_scale${scale}_r${repeat}"
      if ! python relu_hgmpc/run_floating_base.py --controller "$method" --scale "$scale" \
        --duration 12 --name "$name" > "results/floating_base/${name}.log" 2>&1; then
        failed=$((failed+1))
      fi
    done
  done
done
printf 'Failed runs: %s\n' "$failed"
test "$failed" = 0
