#!/usr/bin/env bash
set -euo pipefail

# 真机 TCP 位姿测试入口。
# 默认使用 config/default.yaml 的 execution_compensation_* 后再发送。
# 如需测原始未补偿位姿，在命令末尾加 --raw。
# 只运行 SDK 机械臂位姿控制，不打开相机，不依赖 ROS2。
# ./scripts/run_move_pose_error_test.sh --interactive

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

exec env -u PYTHONPATH \
  LD_LIBRARY_PATH="${CMEEL_LIB}:${LD_LIBRARY_PATH:-}" \
  python scripts/move_pose_error_test.py --apply-compensation "$@"
