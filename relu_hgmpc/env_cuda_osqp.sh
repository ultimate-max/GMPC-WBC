#!/usr/bin/env bash
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
export PYTHONPATH=/home/ganhe/project/b2z1_mpc/relu_hgmpc/deps/osqp_cuda_fp64_safe:${PYTHONPATH:-}
# Runtime libraries already supplied by the existing PyTorch environment.
osqp_cuda_libraries=/home/ganhe/miniconda3/envs/frictionest/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="$osqp_cuda_libraries/cuda_runtime/lib:$osqp_cuda_libraries/cublas/lib:$osqp_cuda_libraries/cusparse/lib:$osqp_cuda_libraries/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
# Installing an optional backend must not silently change old experiments.
export OSQP_ALGEBRA_BACKEND=builtin
