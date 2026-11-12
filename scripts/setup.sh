#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx b2z1_mpc; then
  conda env create -f "$ROOT/environment.yml"
fi
mkdir -p "$ROOT/third_party"
if [ ! -d "$ROOT/third_party/unitree_ros/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_ros.git "$ROOT/third_party/unitree_ros"
  git -C "$ROOT/third_party/unitree_ros" sparse-checkout set robots/b2_description robots/z1_description
  git -C "$ROOT/third_party/unitree_ros" checkout 725b59b6de39a26c425632120c89f23c49a36f1e
fi
if [ ! -d "$ROOT/third_party/osqp/.git" ]; then
  git clone --depth 1 --branch v1.0.0 https://github.com/osqp/osqp.git "$ROOT/third_party/osqp"
fi
bash "$ROOT/scripts/build.sh"
