#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source relu_hgmpc/env.sh
exec python relu_hgmpc/run_singlethread_anchor.py \
  --method anchor --wbc-hz 200 --schedule budget --duration 12 \
  --name "singlethread_$(date +%Y%m%d_%H%M%S)" "$@"
