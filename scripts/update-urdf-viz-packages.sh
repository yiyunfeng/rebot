#!/usr/bin/env bash
# 自动从 ROS2 工作空间的 colcon list 生成 urdf-visualizer.packages 配置
# 用法: ./scripts/update-urdf-viz-packages.sh [workspace_root] [--write]

set -euo pipefail

ROOT="${1:-$(pwd)}"
WRITE=false
[[ "${2:-}" == "--write" ]] && WRITE=true

# 需要先 source ROS2 环境
if ! command -v colcon &>/dev/null; then
    if [[ -f /opt/ros/humble/setup.bash ]]; then
        source /opt/ros/humble/setup.bash
    elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
        source /opt/ros/jazzy/setup.bash
    else
        echo "错误: 找不到 colcon 和 ROS2 环境，请先 source /opt/ros/<distro>/setup.bash"
        exit 1
    fi
fi

cd "$ROOT"

# 从 colcon list 生成 packages JSON
readarray -t PACKAGES < <(colcon list 2>/dev/null | awk '{printf "        \"%s\": \"%s\"", $1, $2}')

if [[ ${#PACKAGES[@]} -eq 0 ]]; then
    echo "错误: colcon list 为空，当前目录可能不是 ROS2 工作空间"
    exit 1
fi

# 生成 JSON 块（去掉每行末尾逗号，最后一行除外）
JSON_BLOCK=""
for i in "${!PACKAGES[@]}"; do
    if [[ $i -lt $(( ${#PACKAGES[@]} - 1 )) ]]; then
        JSON_BLOCK+="${PACKAGES[$i]},\n"
    else
        JSON_BLOCK+="${PACKAGES[$i]}\n"
    fi
done

echo "检测到 ${#PACKAGES[@]} 个包："
colcon list 2>/dev/null
echo ""
echo "--- urdf-visualizer.packages 配置 ---"
echo -e "\"urdf-visualizer.packages\": {\n${JSON_BLOCK}}"

# 可选：直接写入 .vscode/settings.json
if $WRITE; then
    VSCODE_DIR="$ROOT/.vscode"
    SETTINGS_FILE="$VSCODE_DIR/settings.json"

    if [[ ! -f "$SETTINGS_FILE" ]]; then
        mkdir -p "$VSCODE_DIR"
        echo "{}" > "$SETTINGS_FILE"
    fi

    # 用 python3 合并 JSON（保持缩进）
    python3 -c "
import json, sys

with open('$SETTINGS_FILE') as f:
    try:
        settings = json.load(f)
    except json.JSONDecodeError:
        settings = {}

# 生成 packages 映射
packages = {}
$(colcon list 2>/dev/null | awk '{printf "packages[\"%s\"] = \"%s\"\n", $1, $2}')

settings['urdf-visualizer.packages'] = packages

with open('$SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=4, ensure_ascii=False)
    f.write('\n')

print('已写入 $SETTINGS_FILE')
"
fi
