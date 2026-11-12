#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/paper_repro/env.sh
python paper_repro/generate.py
python paper_repro/verify.py
cmake -S paper_repro -B paper_repro/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build paper_repro/build -j4
