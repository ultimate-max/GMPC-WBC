#!/usr/bin/env bash
source /home/ganhe/miniconda3/etc/profile.d/conda.sh
conda activate b2z1_mpc
export ACADOS_SOURCE_DIR=/home/ganhe/project/b2z1_mpc/third_party/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=3
export OMP_DYNAMIC=FALSE
cd /home/ganhe/project/b2z1_mpc
