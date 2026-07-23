#!/usr/bin/env bash
set -euo pipefail

# 直接相机 + 真机抓取统一入口。
# main.py 会读取 config/default.yaml 的 perception.backend：
#   obb/sam   -> 传统短轴抓取；
#   graspnet  -> 自动转到 scripts/grasp.py，并打开 GraspNet 候选抓取窗口。
# main.py 不订阅 ROS2 topic，因此必须清掉 ROS2 注入的 PYTHONPATH，
# 否则 reBotArm_control_py 可能误导入 /opt/ros/humble 的 pinocchio，
# 在 NumPy 2.x 环境下触发 native 崩溃。

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
  python scripts/main.py
