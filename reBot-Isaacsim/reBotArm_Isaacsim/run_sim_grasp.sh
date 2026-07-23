#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_sim_rgbd.sh" &
ISAAC_PID=$!

cleanup() {
  kill "${ISAAC_PID}" 2>/dev/null || true
  wait "${ISAAC_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${SCRIPT_DIR}/run_sim2real_perception.sh" --source sim
