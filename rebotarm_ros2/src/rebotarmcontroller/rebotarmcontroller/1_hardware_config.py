"""
hardware_config 模块 — 硬件配置解析与合并
==========================================

本模块负责：
  1. 定位并加载 ROS 硬件配置 YAML 文件
  2. 根据指定的 model name 查找对应的 SDK 配置文件
  3. 深度合并 override 配置
  4. 计算运行时增益参数（Kp/Kd/重力补偿等）
  5. 将解析结果写入临时文件供 SDK 读取
  6. 同步配置到 SDK 的运动学/动力学缓存

**配置层级结构**（优先级从高到低）：
  SDK 默认 → model overrides → channel 覆盖 → _runtime 计算值
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)


def resolve_hardware_config(
    hardware_config: str | None,
    model: str,
    channel: str,
) -> tuple[Path, dict[str, Any]]:
    """硬件配置解析入口。返回 (临时配置文件路径, 合并后的完整配置字典)。"""
    sdk_root = _ensure_rebot_sdk_in_syspath()
    model_name, data = _load_ros_hardware_config(sdk_root, hardware_config, model, channel)
    path = _write_resolved_hardware_config(model_name, data)
    _sync_sdk_robot_model_config(data)
    return path, copy.deepcopy(data)


# ══════════════════════════════════════════════════════════════════════
# 工作区路径定位
# ══════════════════════════════════════════════════════════════════════

def _workspace_root() -> Path:
    """
    定位工作区根目录：从当前文件向上遍历父目录，找到包含
    "third_party/reBotArm_control_py" 的目录。回退到向上 3 级。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "third_party" / "reBotArm_control_py").is_dir():
            return parent
    return here.parents[3]


def _ensure_rebot_sdk_in_syspath() -> Path:
    """确保 reBotArm SDK 在 sys.path 最前面。SDK 不存在时给出 clone 提示。"""
    root = _workspace_root() / "third_party" / "reBotArm_control_py"
    if not (root / "reBotArm_control_py").is_dir():
        raise FileNotFoundError(
            f"Cannot find reBotArm_control_py at {root}. Clone it first:\n"
            "  git clone https://github.com/vectorBH6/reBotArm_control_py.git "
            "third_party/reBotArm_control_py"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


# ══════════════════════════════════════════════════════════════════════
# 配置路径解析
# ══════════════════════════════════════════════════════════════════════

def _default_hardware_config_path() -> Path:
    """
    默认硬件配置路径。
    查找顺序：1) ament 包路径  2) workspace 回退路径
    """
    try:
        path = (
            Path(get_package_share_directory("rebotarm_bringup"))
            / "config" / "rebotarm_hardware.yaml"
        )
        if path.exists():
            return path
    except PackageNotFoundError:
        pass
    return (
        _workspace_root() / "src" / "rebotarm_bringup"
        / "config" / "rebotarm_hardware.yaml"
    )


# ══════════════════════════════════════════════════════════════════════
# 配置合并
# ══════════════════════════════════════════════════════════════════════

def _deep_merge(base: Any, override: Any) -> Any:
    """
    深度合并两个字典。相同 key 且都是 dict → 递归，否则覆盖。
    示例：{a:1, b:{x:2}} + {b:{y:3}} → {a:1, b:{x:2, y:3}}
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged.get(key), value)
        return merged
    return copy.deepcopy(override)


# ══════════════════════════════════════════════════════════════════════
# 核心加载逻辑
# ══════════════════════════════════════════════════════════════════════

def _load_ros_hardware_config(
    sdk_root: Path,
    hardware_config: str | None,
    model: str,
    channel: str,
) -> tuple[str, dict[str, Any]]:
    """
    加载 ROS 配置 + 合并 SDK 配置 + overrides + 计算运行时参数。

    流程：
      1. 加载 ROS YAML（用户路径或默认路径）
      2. 确定 model name（参数 > default_model > "dm"）
      3. 读取对应 model 的 SDK 配置文件
      4. 深度合并 model overrides
      5. 注入 channel
      6. 计算 _runtime 增益
    """
    # 1. 定位配置路径
    config_path = (
        Path(hardware_config).expanduser()
        if hardware_config
        else _default_hardware_config_path()
    )
    if not config_path.exists():
        raise FileNotFoundError(f"ROS hardware config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        ros_config = yaml.safe_load(f) or {}

    # 2. 确定 model
    model_name = (model or ros_config.get("default_model") or "dm").strip().lower()
    models = ros_config.get("models", {})
    if model_name not in models:
        choices = ", ".join(sorted(models))
        raise ValueError(f"unknown hardware model {model_name!r}; choices: {choices}")

    # 3. 读取 SDK 配置
    model_config = models[model_name] or {}
    sdk_config = model_config.get("sdk_config")
    if not sdk_config:
        raise ValueError(f"models.{model_name}.sdk_config is required")

    sdk_config_path = Path(str(sdk_config)).expanduser()
    if not sdk_config_path.is_absolute():
        sdk_config_path = sdk_root / "config" / sdk_config_path
    if not sdk_config_path.exists():
        raise FileNotFoundError(f"SDK hardware config not found: {sdk_config_path}")

    with open(sdk_config_path, "r", encoding="utf-8") as f:
        merged = yaml.safe_load(f) or {}

    # 4-6. 合并 overrides → 注入 channel → 计算 _runtime
    merged = _deep_merge(merged, model_config.get("overrides", {}) or {})
    if channel:
        merged["channel"] = channel
    _add_runtime_config(merged)

    return model_name, merged


# ══════════════════════════════════════════════════════════════════════
# 运行时配置计算
# ══════════════════════════════════════════════════════════════════════

def _add_runtime_config(data: dict[str, Any]) -> None:
    """
    计算并注入 _runtime 字典，包含：
      - control.mit_kp / mit_kd（每关节 MIT 增益）
      - gravity_compensation.kp/kd/joint_direction/tau_scale

    增益查找优先级：
      control 全局配置 > 单个关节 MIT 配置 > 默认 0.0
    """
    arm_joints = _arm_joint_names(data)
    n = len(arm_joints)
    gravity_config = data.get("gravity_compensation", {}) or {}
    control_config = data.get("control", {}) or {}

    data["_runtime"] = {
        "control": {
            "arm_control_mode": _arm_control_mode(data),
            "mit_kp": _control_gain(data, arm_joints, control_config, "mit_kp", "kp"),
            "mit_kd": _control_gain(data, arm_joints, control_config, "mit_kd", "kd"),
        },
        "gravity_compensation": {
            "kp": _gravity_gain(data, arm_joints, gravity_config, "kp"),
            "kd": _gravity_gain(data, arm_joints, gravity_config, "kd"),
            "joint_direction": _runtime_vector(
                gravity_config.get("joint_direction", 1.0), n,
                "gravity_compensation.joint_direction",
            ),
            "tau_scale": _runtime_vector(
                gravity_config.get("tau_scale", 1.0), n,
                "gravity_compensation.tau_scale",
            ),
        },
    }


def _arm_control_mode(data: dict[str, Any]) -> str:
    """确定控制模式。仅接受 'posvel' 或 'mit'。'pos_vel' 自动规范化为 'posvel'。"""
    mode = str(
        (data.get("control", {}) or {}).get("arm_control_mode", "posvel")
    ).strip().lower()
    if mode == "pos_vel":
        mode = "posvel"
    if mode not in ("posvel", "mit"):
        raise ValueError("control.arm_control_mode must be 'posvel' or 'mit'")
    return mode


def _arm_joint_names(data: dict[str, Any]) -> list[str]:
    """从 groups.arm.joints 提取关节名列表。"""
    joints = data.get("groups", {}).get("arm", {}).get("joints", [])
    if not joints:
        raise ValueError("hardware config must define groups.arm.joints")
    return [str(name) for name in joints]


# ══════════════════════════════════════════════════════════════════════
# 增益参数解析
# ══════════════════════════════════════════════════════════════════════

def _gravity_gain(
    data: dict[str, Any], arm_joints: list[str],
    gravity_config: dict[str, Any], key: str,
) -> list[float]:
    """获取重力补偿增益。优先级：gravity 配置 > 各关节 MIT 配置 > 0.0。"""
    if key in gravity_config:
        return _runtime_vector(gravity_config[key], len(arm_joints),
                               f"gravity_compensation.{key}")
    joint_map = {str(joint.get("name")): joint for joint in data.get("joints", [])}
    gains = []
    for name in arm_joints:
        joint = joint_map.get(name)
        if joint is None:
            raise ValueError(f"groups.arm references unknown joint {name!r}")
        gains.append(float((joint.get("MIT", {}) or {}).get(key, 0.0)))
    return gains


def _control_gain(
    data: dict[str, Any], arm_joints: list[str],
    control_config: dict[str, Any], config_key: str, mit_key: str,
) -> list[float]:
    """获取控制增益。优先级：control 配置 > 各关节 MIT 配置 > 0.0。"""
    if config_key in control_config:
        return _runtime_vector(control_config[config_key], len(arm_joints),
                               f"control.{config_key}")
    joint_map = {str(joint.get("name")): joint for joint in data.get("joints", [])}
    gains = []
    for name in arm_joints:
        joint = joint_map.get(name)
        if joint is None:
            raise ValueError(f"groups.arm references unknown joint {name!r}")
        gains.append(float((joint.get("MIT", {}) or {}).get(mit_key, 0.0)))
    return gains


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _runtime_vector(value: Any, size: int, label: str) -> list[float]:
    """
    标量/列表 → 指定长度的浮点向量。
    - 标量 → [value] * size（广播到所有关节）
    - 单元素列表 → [v[0]] * size
    - size 元素列表 → 直接使用
    """
    if isinstance(value, (int, float)):
        return [float(value)] * size
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a scalar or {size} values")
    values = [float(item) for item in value]
    if len(values) == 1:
        return values * size
    if len(values) != size:
        raise ValueError(f"{label} must be a scalar or {size} values")
    return values


# ══════════════════════════════════════════════════════════════════════
# 临时文件输出
# ══════════════════════════════════════════════════════════════════════

_resolved_config_dir: Path | None = None


def _write_resolved_hardware_config(model: str, data: dict[str, Any]) -> Path:
    """
    将解析后配置写入临时文件供 SDK 读取。
    SDK 需要从文件读取配置，因此必须物化到磁盘。
    tempfile.mkdtemp 确保路径唯一，进程结束后由系统回收。
    """
    global _resolved_config_dir
    if _resolved_config_dir is None:
        _resolved_config_dir = Path(tempfile.mkdtemp(prefix="rebotarm_ros2_"))
    tmp_path = _resolved_config_dir / f"{model}_hardware.yaml"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return tmp_path


# ══════════════════════════════════════════════════════════════════════
# SDK 模型同步
# ══════════════════════════════════════════════════════════════════════

def _sync_sdk_robot_model_config(data: dict[str, Any]) -> None:
    """
    同步配置到 SDK 运动学/动力学模块缓存。
    - robot_model._hw_cfg_cache → 运动学硬件配置缓存
    - dynamics_model._CACHED_MODEL → 清除动力学缓存，下次强制重算
    """
    import reBotArm_control_py.kinematics.robot_model as robot_model
    import reBotArm_control_py.dynamics.robot_model as dynamics_model

    robot_model._hw_cfg_cache = copy.deepcopy(data)
    dynamics_model._CACHED_MODEL = None
