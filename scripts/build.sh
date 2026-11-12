#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate b2z1_mpc
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
python "$ROOT/scripts/prepare_model.py"
cmake -S "$ROOT" -B "$ROOT/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build "$ROOT/build" -j2
ctest --test-dir "$ROOT/build" --output-on-failure
python "$ROOT/scripts/check_mujoco.py"
