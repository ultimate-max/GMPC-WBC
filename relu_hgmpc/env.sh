#!/usr/bin/env bash
source /home/ganhe/project/b2z1_mpc/paper_repro/env.sh
export PYTHONPATH=/home/ganhe/project/b2z1_mpc/relu_hgmpc/build:${PYTHONPATH:-}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RELU_TORCH_SITE=/home/ganhe/miniconda3/envs/frictionest/lib/python3.11/site-packages
