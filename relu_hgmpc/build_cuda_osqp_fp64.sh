#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
osqp_project=/home/ganhe/project/b2z1_mpc
export CMAKE_CUDA_COMPILER="$osqp_project/toolchains/cuda121/bin/nvcc"
export CUDA_TOOLKIT_ROOT_DIR="$osqp_project/toolchains/cuda121"
export CUDAHOSTCXX="$CUDA_TOOLKIT_ROOT_DIR/bin/x86_64-conda-linux-gnu-g++"
export CC="$CUDA_TOOLKIT_ROOT_DIR/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDAHOSTCXX"
# Avoid CMake searching WindowsApps through WSL's slow /mnt/c PATH entries.
export PATH="$CUDA_TOOLKIT_ROOT_DIR/bin:/home/ganhe/miniconda3/envs/b2z1_mpc/bin:/usr/local/bin:/usr/bin:/bin"
export CMAKE_BUILD_PARALLEL_LEVEL=4
python "$osqp_project/relu_hgmpc/tighten_cuda_cg.py"
python - <<'PY'
from pathlib import Path
prefix=Path('/home/ganhe/project/b2z1_mpc/toolchains/cuda121')
runtime=Path('/home/ganhe/miniconda3/envs/frictionest/lib/python3.11/site-packages/nvidia')
for name in ('cublas','cusparse'):
    for src in (runtime/name/'include').glob('*.h'):
        dst=prefix/'include'/src.name
        if not dst.exists():dst.symlink_to(src)
    for src in (runtime/name/'lib').glob('*.so*'):
        dst=prefix/'lib'/src.name
        if not dst.exists():dst.symlink_to(src)
        base=prefix/'lib'/(src.name.split('.so')[0]+'.so')
        if not base.exists():base.symlink_to(src)
PY
python -m pip wheel "$osqp_project/third_party/osqp_python_cuda113/backend/cuda" --no-deps \
    --wheel-dir "$osqp_project/relu_hgmpc/deps/wheels_fp64_safe" \
    --config-settings=cmake.define.OSQP_USE_FLOAT=OFF \
    --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_OSQP="$osqp_project/third_party/osqp_core_cuda100" \
    --config-settings=cmake.define.CUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT_DIR" \
    --config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
    --config-settings=build-dir="$osqp_project/third_party/osqp_python_cuda113/build_cuda_fp64_gcc12"
python -m pip install --no-deps --upgrade --target "$osqp_project/relu_hgmpc/deps/osqp_cuda_fp64_safe" \
    "$osqp_project/relu_hgmpc/deps/wheels_fp64_safe/"*.whl
