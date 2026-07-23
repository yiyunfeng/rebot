"""相机配置与手眼标定相关的共享辅助函数。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

# 尝试相对导入，支持包内和直接脚本运行两种方式
try:
    from ..drivers.camera import CameraDriver, make_camera
except ImportError:
    from drivers.camera import CameraDriver, make_camera


def load_config(path: str | Path) -> dict[str, Any]:
    """加载 YAML 配置文件。

    使用 yaml.safe_load 安全加载（禁用任意 Python 对象反序列化），
    避免 unsafe YAML 加载导致的安全风险。

    参数：
        path: 配置文件路径

    返回：
        解析后的配置字典

    异常：
        FileNotFoundError: 配置文件不存在时抛出
    """
    # 展开 ~ 并转为 Path 对象
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    # 使用 safe_load 而非 load，防止任意代码执行
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_hand_eye(project_root: str | Path, cam_type: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    """加载手眼标定结果（相机到 TCP 的变换矩阵）。

    从 config/calibration/<cam_type>/hand_eye.npz 文件中读取：
    - T_result: 4x4 齐次变换矩阵（相机坐标系 -> TCP/法兰坐标系）
    - mode:     手眼标定模式（如 "eye_in_hand" 或 "eye_to_hand"）

    参数：
        project_root: 项目根目录
        cam_type:     相机类型名（如 "realsense"），用作标定子目录名

    返回：
        (T_result, mode) 的元组：
        - T_result: 4x4 float64 齐次变换矩阵，文件不存在时返回 None
        - mode:     手眼标定模式字符串，文件不存在时返回 None
    """
    # 构造标定文件路径：config/calibration/<cam_type>/hand_eye.npz
    hand_eye_path = Path(project_root) / "config" / "calibration" / str(cam_type).lower() / "hand_eye.npz"
    if not hand_eye_path.exists():
        return None, None

    # allow_pickle=False 防止 pickle 反序列化安全风险
    data = np.load(str(hand_eye_path), allow_pickle=False)
    T = data["T_result"].astype(np.float64)
    # mode 是 (1,) 形状的数组，取第一个元素并转为 str
    mode = str(data["mode"][0])
    return T, mode


def hand_eye_compensation_matrix(cfg: dict[str, Any]) -> np.ndarray:
    """从配置中读取手眼标定补偿量，构建 4x4 平移补偿矩阵。

    补偿矩阵用于微调手眼标定结果中的平移误差，结构如下：
        T = [[1, 0, 0, dx],
             [0, 1, 0, dy],
             [0, 0, 1, dz],
             [0, 0, 0, 1 ]]

    其中 dx/dy/dz 从配置的 calibration.hand_eye_compensation_m 中读取。

    参数：
        cfg: 配置字典

    返回：
        4x4 float64 齐次变换矩阵（纯平移，无旋转成分）
    """
    # 读取校准补偿配置节
    calibration = cfg.get("calibration") or {}
    compensation = calibration.get("hand_eye_compensation_m") or {}
    # 构建 4x4 单位矩阵
    T = np.eye(4, dtype=np.float64)
    # 设置平移分量
    T[:3, 3] = [
        float(compensation.get("x", 0.0)),
        float(compensation.get("y", 0.0)),
        float(compensation.get("z", 0.0)),
    ]
    return T


def compose_cam_to_base_transform(T_tcp2base: np.ndarray, T_hand_eye: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """组合完整的相机到基座坐标变换链。

    变换链：相机（cam）-> TCP（法兰）-> 基座（base）+ 补偿量。

    即：T_cam2base = T_compensation @ T_tcp2base @ T_hand_eye

    其中：
    - T_hand_eye:   相机 -> TCP 的变换（手眼标定结果）
    - T_tcp2base:   TCP -> 基座的变换（从机器人正运动学获取）
    - T_compensation: 平移补偿矩阵（从配置文件读取）

    参数：
        T_tcp2base: TCP/法兰 -> 基座的 4x4 齐次变换矩阵
        T_hand_eye: 相机 -> TCP 的 4x4 齐次变换矩阵（手眼标定结果）
        cfg:        配置字典（用于读取补偿值）

    返回：
        4x4 float64 齐次变换矩阵：相机 -> 基座
    """
    # 读取平移补偿矩阵
    T_compensation = hand_eye_compensation_matrix(cfg)
    # 矩阵乘法链：补偿 @ TCP2base @ 手眼
    return T_compensation @ np.asarray(T_tcp2base, dtype=np.float64) @ np.asarray(T_hand_eye, dtype=np.float64)


def configure_camera(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    realsense_default_fps: int | None = 15,
) -> dict[str, Any]:
    """合并配置文件与命令行参数，生成最终的相机配置。

    参数合并策略：
        - 命令行参数（args）具有最高优先级，直接覆盖配置字典中的对应项。
        - 如果命令行未指定 FPS 且相机类型为 RealSense，使用默认 FPS。
        - 修改后的配置直接写回 cfg["camera"]，实现原地更新。

    参数：
        cfg:                  完整的配置字典（会被原地修改）
        args:                 解析后的命令行参数（argparse.Namespace）
        realsense_default_fps: RealSense 相机未指定 FPS 时的默认帧率

    返回：
        原地修改后的 cfg 字典（与入参同一个对象）

    异常：
        ValueError: 未配置 camera.type 时抛出
    """
    # 确保 camera 节存在，不存在时创建空字典
    cam_cfg = cfg.setdefault("camera", {})
    # 从命令行读取可选的覆盖参数
    camera_type = getattr(args, "camera_type", None)
    width = getattr(args, "width", None)
    height = getattr(args, "height", None)
    fps = getattr(args, "fps", None)

    # 命令行指定的相机类型覆盖配置
    if camera_type is not None:
        cam_cfg["type"] = camera_type
    cam_type = str(cam_cfg.get("type", "")).lower()
    if not cam_type:
        raise ValueError("camera.type is missing in config; pass --camera-type or set it in YAML")

    # 命令行指定的分辨率覆盖配置（颜色图和深度图使用相同分辨率）
    if width is not None:
        cam_cfg["color_width"] = int(width)
        cam_cfg["depth_width"] = int(width)
    if height is not None:
        cam_cfg["color_height"] = int(height)
        cam_cfg["depth_height"] = int(height)

    # FPS：命令行 > 配置 > RealSense 默认值
    if fps is not None:
        cam_cfg["fps"] = int(fps)
    elif camera_type is not None and realsense_default_fps is not None and "realsense" in cam_type:
        cam_cfg["fps"] = int(realsense_default_fps)
    return cfg


def create_camera_from_args(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    *,
    realsense_default_fps: int | None = 15,
) -> CameraDriver:
    """根据配置和命令行参数创建相机驱动实例。

    该函数组合了 configure_camera 和 make_camera 两个步骤：
        1. 首先调用 configure_camera 合并配置与命令行参数
        2. 然后调用 make_camera 根据合并后的配置实例化相机驱动

    参数：
        cfg:                  完整的配置字典
        args:                 解析后的命令行参数
        realsense_default_fps: RealSense 相机的默认帧率

    返回：
        已配置并初始化的 CameraDriver 实例
    """
    # 先合并配置，再创建相机实例
    return make_camera(configure_camera(cfg, args, realsense_default_fps=realsense_default_fps))
