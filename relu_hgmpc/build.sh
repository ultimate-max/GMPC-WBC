#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
cmake -S relu_hgmpc -B relu_hgmpc/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build relu_hgmpc/build -j1
