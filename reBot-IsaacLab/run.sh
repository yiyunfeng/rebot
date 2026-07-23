#!/usr/bin/env bash
# reBot-IsaacLab 统一入口。
# 日常只需要记住本脚本，避免多个 run_*.sh 入口互相混淆。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage:
  ./run.sh <command>

Main commands:
  build-asset      检查香蕉物理 USD
  collect-teacher  从 reBot-Isaacsim /tmp 输出采集 IsaacSim teacher 数据
  train-bc         用 teacher 数据做 RGB-D 行为克隆预训练
  train            训练 RGB-D PPO
  evaluate         评估最新 RGB-D checkpoint 成功率
  export           导出最新 RGB-D checkpoint 为 TorchScript
  report           生成 HTML 可视化报告
  watch            打开 Isaac Sim GUI 看最新策略

Real robot:
  real-dry-run     只读真机相机并打印网络动作，不连接机械臂
  real-execute     真机执行入口；仍需 REBOT_REAL_POLICY_ENABLE=1 和人工确认

Utility:
  test             运行轻量单元测试，不启动 Isaac Sim
  quick            短测闭环：build -> collect -> bc -> train -> evaluate -> export -> report
  help             显示本帮助

Examples:
  REBOT_ISAACSIM_TEACHER_PLANS=3 ./run.sh collect-teacher
  REBOT_RGBD_NUM_ENVS=8 REBOT_RGBD_ITERATIONS=200 ./run.sh train
  REBOT_RGBD_GUI=1 REBOT_RGBD_NUM_ENVS=4 REBOT_RGBD_ITERATIONS=50 ./run.sh train
  ./run.sh watch
EOF
}

isaaclab_python() {
  export PYTHONPATH="${PROJECT_DIR}/source/rebot_isaaclab:${PYTHONPATH:-}"
  cd "${PROJECT_DIR}"
  conda run -n isaaclab --no-capture-output python "$@"
}

real_python() {
  export PYTHONPATH="${PROJECT_DIR}/source/rebot_isaaclab:${REPO_ROOT}/rebot_grasp:${PYTHONPATH:-}"
  cd "${PROJECT_DIR}"
  conda run -n rebotarm_gpu --no-capture-output python "$@"
}

command="${1:-help}"
case "${command}" in
  build-asset)
    export PYTHONPATH="${PROJECT_DIR}/source/rebot_isaaclab:${PYTHONPATH:-}"
    cd "${PROJECT_DIR}"
    python3 "${PROJECT_DIR}/scripts/build_banana_asset.py"
    ;;
  collect-teacher)
    isaaclab_python "${PROJECT_DIR}/scripts/collect_isaacsim_teacher.py"
    ;;
  train-bc)
    isaaclab_python "${PROJECT_DIR}/scripts/train_rgbd_bc.py"
    ;;
  train)
    isaaclab_python "${PROJECT_DIR}/scripts/train_rgbd_ppo.py"
    ;;
  evaluate)
    isaaclab_python "${PROJECT_DIR}/scripts/evaluate_rgbd_policy.py"
    ;;
  export)
    isaaclab_python "${PROJECT_DIR}/scripts/export_rgbd_policy.py"
    ;;
  report)
    isaaclab_python "${PROJECT_DIR}/scripts/make_visual_report.py"
    ;;
  watch)
    isaaclab_python "${PROJECT_DIR}/scripts/watch_rgbd_policy.py"
    ;;
  real-dry-run)
    real_python "${PROJECT_DIR}/scripts/real_rgbd_policy_dry_run.py"
    ;;
  real-execute)
    echo "[Safety] This can move the real robot when REBOT_REAL_POLICY_ENABLE=1."
    echo "[Safety] Confirm DM model, serial channel, joint/gripper limits, clear workspace and emergency stop."
    real_python "${PROJECT_DIR}/scripts/real_rgbd_policy_executor.py"
    ;;
  test)
    export PYTHONPATH="${PROJECT_DIR}/source/rebot_isaaclab:${PYTHONPATH:-}"
    export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
    cd "${PROJECT_DIR}"
    conda run -n isaaclab --no-capture-output python -m pytest -q tests
    ;;
  quick)
    "${PROJECT_DIR}/run.sh" build-asset
    "${PROJECT_DIR}/run.sh" collect-teacher
    "${PROJECT_DIR}/run.sh" train-bc
    "${PROJECT_DIR}/run.sh" train
    "${PROJECT_DIR}/run.sh" evaluate
    "${PROJECT_DIR}/run.sh" export
    "${PROJECT_DIR}/run.sh" report
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "[run.sh] unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
