#!/usr/bin/env bash
set -euo pipefail

# 手眼标定统一入口。
# collect_handeye_eih.py 不依赖 ROS2；清掉 ROS2 PYTHONPATH，
# 避免误导入 /opt/ros/humble 的 pinocchio 导致 NumPy/native 扩展崩溃。

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
  python scripts/collect_handeye_eih.py "$@"
