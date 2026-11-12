#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/miniconda3/etc/profile.d/conda.sh
conda activate b2z1_mpc
cd /home/ganhe/project/b2z1_mpc
if [ ! -d third_party/acados/.git ]; then
  git clone --depth 1 https://github.com/acados/acados.git third_party/acados
  git -C third_party/acados fetch --depth 1 origin 51db6db146217e6624d7575f34e0421f28bc76a0
  git -C third_party/acados checkout 51db6db146217e6624d7575f34e0421f28bc76a0
fi
git -C third_party/acados submodule update --init --recursive --depth 1
cmake -S third_party/acados -B third_party/acados/build_ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DACADOS_WITH_OPENMP=OFF -DBLASFEO_TARGET=X64_INTEL_HASWELL -DACADOS_INSTALL_DIR="$PWD/third_party/acados" -DCMAKE_INSTALL_PREFIX="$PWD/third_party/acados"
cmake --build third_party/acados/build_ninja --parallel 6 >/tmp/b2z1_acados_build.log 2>&1
cmake --install third_party/acados/build_ninja >>/tmp/b2z1_acados_build.log 2>&1
python -m pip install -e third_party/acados/interfaces/acados_template
git -C third_party/acados rev-parse HEAD
