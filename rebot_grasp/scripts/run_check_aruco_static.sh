#!/usr/bin/env bash
set -euo pipefail

# ArUco 静态稳定性检查入口。
# 默认先移动到 config/default.yaml 的 robot.ready_pose，让相机能看到 ArUco；
# 如需只开相机、不移动机械臂，命令后加 --no-move-ready。

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

ORBBEC_LIB="${GRASP_ROOT}/sdk/pyorbbecsdk/install/lib"
CMEEL_LIB="${CONDA_PREFIX}/lib/python3.10/site-packages/cmeel.prefix/lib"

exec env -u PYTHONPATH \
  LD_LIBRARY_PATH="${CMEEL_LIB}:${ORBBEC_LIB}:${LD_LIBRARY_PATH:-}" \
  python scripts/check_aruco_static.py --move-ready "$@"
