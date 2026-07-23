#!/usr/bin/env bash
set -euo pipefail

# 直接相机 + GraspNet 调试入口。
# 这个脚本不需要 ROS2；清掉 ROS2 PYTHONPATH，保留 conda/cmeel pinocchio
# 和 pyorbbecsdk 动态库路径，避免 native 扩展冲突。

GRASP_ROOT="/home/yyf/Desktop/pythonProject/rebot/rebot_grasp"
CONDA_ENV="rebotarm_gpu"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] 未找到 conda.sh，请先确认 conda 安装路径。" >&2
  exit 1
fi

conda activate "${CONDA_ENV}"
cd "${GRASP_ROOT}"

CMEEL_LIB="${CONDA_PREFIX}/lib/python3.10/site-packages/cmeel.prefix/lib"
ORBBEC_LIB="${GRASP_ROOT}/sdk/pyorbbecsdk/install/lib"

exec env -u PYTHONPATH \
  LD_LIBRARY_PATH="${CMEEL_LIB}:${ORBBEC_LIB}:${LD_LIBRARY_PATH:-}" \
  python scripts/graspnet_camera_demo.py "$@"
