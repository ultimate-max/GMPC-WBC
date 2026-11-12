#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source relu_hgmpc/env.sh
exec python relu_hgmpc/run_mrt_wbc.py "$@"
