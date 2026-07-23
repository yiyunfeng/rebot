#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${REBOT_CONDA_ENV:-rebotarm_gpu}"
CONDA_BASE="$(conda info --base)"

echo "Real robot: B601-DM using the serial channel from the SDK YAML"
echo "Confirm joint/gripper limits, clear workspace, and keep the emergency stop ready."
read -r -p "Type RUN REAL to continue: " CONFIRMATION
if [[ "${CONFIRMATION}" != "RUN REAL" ]]; then
  echo "Cancelled before connecting hardware."
  exit 0
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# 为感知进程创建独立进程组；退出时终止整个组，确保相机一定被释放。
setsid env \
  -u PYTHONPATH \
  -u LD_LIBRARY_PATH \
  -u AMENT_PREFIX_PATH \
  -u CMAKE_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/rebot-matplotlib \
  python "${SCRIPT_DIR}/sim2real_perception.py" --source real &
PERCEPTION_PID=$!

cleanup() {
  trap - EXIT INT TERM
  kill -TERM -- "-${PERCEPTION_PID}" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 -- "-${PERCEPTION_PID}" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  kill -KILL -- "-${PERCEPTION_PID}" 2>/dev/null || true
  wait "${PERCEPTION_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

env \
  -u PYTHONPATH \
  -u LD_LIBRARY_PATH \
  -u AMENT_PREFIX_PATH \
  -u CMAKE_PREFIX_PATH \
  REBOT_REAL_CONFIRMED=1 \
  MPLCONFIGDIR=/tmp/rebot-matplotlib \
  python "${SCRIPT_DIR}/real_grasp_executor.py"
