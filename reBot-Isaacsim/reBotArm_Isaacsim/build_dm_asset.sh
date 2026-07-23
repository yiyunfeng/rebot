#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-/home/yyf/isaacsim}"

exec "${ISAACSIM_ROOT}/python.sh" "${SCRIPT_DIR}/build_dm_asset.py" \
  --/rtx/verifyDriverVersion/enabled=false
