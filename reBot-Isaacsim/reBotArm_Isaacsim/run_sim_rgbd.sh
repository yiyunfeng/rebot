#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-/home/yyf/isaacsim}"

# Isaac API 只能在其官方 Python 环境中加载。
exec "${ISAACSIM_ROOT}/python.sh" "${SCRIPT_DIR}/isaacsim_rgbd_exporter.py" \
  "$@" --/rtx/verifyDriverVersion/enabled=false
