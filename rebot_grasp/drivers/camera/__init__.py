"""相机驱动统一入口：根据 YAML 中的 ``camera.type`` 创建对应 RGB-D 相机。"""

from __future__ import annotations

from pathlib import Path

from .base import CameraDriver, CameraFrameError
from .orbbec_dabai_dcw import OrbbecDaBaiDCW
from .orbbec_gemini2 import OrbbecGemini2
from .realsense import RealsenseCamera

__all__ = [
    "CameraDriver",
    "CameraFrameError",
    "OrbbecDaBaiDCW",
    "OrbbecGemini2",
    "RealsenseCamera",
    "make_camera",
]


def make_camera(cfg: dict) -> CameraDriver:
    """从 ``config/default.yaml`` 的相机配置创建对应驱动。

    所有具体驱动都实现 :class:`CameraDriver`，所以上层代码只需调用统一的
    ``open/get_frame/close`` 接口，不需要了解不同厂商 SDK 的区别。
    """
    cam_cfg  = cfg.get("camera", {})
    cam_type = cam_cfg.get("type", "").lower()
    w   = cam_cfg.get("color_width",  1280)
    h   = cam_cfg.get("color_height", 720)
    dw  = cam_cfg.get("depth_width")
    dh  = cam_cfg.get("depth_height")
    fps = cam_cfg.get("fps", 30)

    _root     = Path(__file__).resolve().parent.parent.parent
    calib_dir = str(_root / "config" / "calibration" / cam_type)

    # 型号名允许带具体后缀，例如 realsense_d435i，因此使用关键词匹配。
    if "dabai" in cam_type or "dcw" in cam_type:
        return OrbbecDaBaiDCW(w, h, fps, calib_dir=calib_dir, depth_width=dw, depth_height=dh)
    elif "orbbec" in cam_type:
        return OrbbecGemini2(w, h, fps, calib_dir=calib_dir)
    elif "realsense" in cam_type:
        return RealsenseCamera(w, h, fps, calib_dir=calib_dir)
    else:
        raise ValueError(
            f"Unsupported camera type: {cam_type!r}\n"
            f"Set camera.type in config/default.yaml to:\n"
            f"  orbbec_dabai_dcw | orbbec_gemini2 | realsense_d435i | realsense_d405"
        )
