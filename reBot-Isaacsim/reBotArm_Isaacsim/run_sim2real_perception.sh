#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${REBOT_CONDA_ENV:-rebotarm_gpu}"
CONDA_BASE="$(conda info --base)"

# 激活环境后直接 exec Python，避免 conda run 的父子进程层级截断 Ctrl+C。
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# 隔离 ROS 路径，避免其 NumPy/Pinocchio 覆盖 rebotarm_gpu 中的依赖。
exec env \
  -u PYTHONPATH \
  -u LD_LIBRARY_PATH \
  -u AMENT_PREFIX_PATH \
  -u CMAKE_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/rebot-matplotlib \
  python "${SCRIPT_DIR}/sim2real_perception.py" "$@"
