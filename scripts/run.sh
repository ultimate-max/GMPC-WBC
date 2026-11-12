#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate b2z1_mpc
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
"$ROOT/build/b2z1_demo" "$ROOT" "${1:---benchmark}"
