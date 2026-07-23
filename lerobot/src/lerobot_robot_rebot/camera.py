"""复用 rebot_grasp 驱动的单台同步 RGB-D 相机适配。"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class ReBotRGBDConfig:
    width: int = 640
    height: int = 360
    fps: int = 30
    min_depth_mm: int = 150
    max_depth_mm: int = 2000
    frame_retries: int = 12
    rebot_grasp_path: Path | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("相机 width、height、fps 必须为正数")
        if not 0 <= self.min_depth_mm < self.max_depth_mm <= 65535:
            raise ValueError("深度范围必须满足 0 <= min_depth_mm < max_depth_mm <= 65535")
        if self.frame_retries <= 0:
            raise ValueError("frame_retries 必须为正数")


def depth_mm_to_model_image(
    depth_mm: np.ndarray, min_depth_mm: int, max_depth_mm: int
) -> np.ndarray:
    """固定量程映射：无效像素为 0，有效毫米深度映射到 [1, 255]。"""

    if depth_mm.ndim != 2 or depth_mm.dtype != np.uint16:
        raise ValueError(f"depth_mm 必须为 HxW uint16，实际 {depth_mm.shape}/{depth_mm.dtype}")
    valid = (depth_mm >= min_depth_mm) & (depth_mm <= max_depth_mm) & (depth_mm > 0)
    clipped = np.clip(depth_mm.astype(np.float32), min_depth_mm, max_depth_mm)
    scaled = 1.0 + (clipped - min_depth_mm) * 254.0 / (max_depth_mm - min_depth_mm)
    gray = np.zeros(depth_mm.shape, dtype=np.uint8)
    gray[valid] = np.rint(scaled[valid]).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _default_rebot_grasp_path() -> Path:
    return Path(__file__).resolve().parents[3] / "rebot_grasp"


def _load_driver(root: Path | None) -> type:
    driver_root = (root or _default_rebot_grasp_path()).expanduser().resolve()
    if not driver_root.is_dir():
        raise FileNotFoundError(f"未找到 rebot_grasp: {driver_root}")
    if str(driver_root) not in sys.path:
        sys.path.insert(0, str(driver_root))
    module = importlib.import_module("drivers.camera.orbbec_dabai_dcw")
    return module.OrbbecDaBaiDCW


class ReBotRGBDCamera:
    """一次 SDK 取帧同时生成 RGB、模型深度和原始毫米深度。"""

    def __init__(self, config: ReBotRGBDConfig):
        self.config = config
        self._driver: Any | None = None
        self._depth_mm: np.ndarray | None = None

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    def connect(self) -> None:
        if self.is_connected:
            return
        driver_class = _load_driver(self.config.rebot_grasp_path)
        driver = driver_class(
            color_width=self.config.width,
            color_height=self.config.height,
            depth_width=self.config.width,
            depth_height=self.config.height,
            fps=self.config.fps,
        )
        try:
            driver.open()
        except Exception:
            # open() 可能在 Pipeline 已创建后失败，必须主动释放 USB 资源。
            driver.close()
            raise
        self._driver = driver

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if self._driver is None:
            raise RuntimeError("RGB-D 相机尚未连接")
        # DaBai DCW 不支持硬件帧同步，启动时或运行中可能短暂缺少其中一帧。
        # 只重试取帧，不复用旧深度，保证返回的 RGB/Depth 来自同一组 frameset。
        color_bgr = depth_mm = None
        for _ in range(self.config.frame_retries):
            color_bgr, depth_mm = self._driver.get_frame()
            if color_bgr is not None and depth_mm is not None:
                break
        if color_bgr is None or depth_mm is None:
            raise TimeoutError(
                f"Orbbec 连续 {self.config.frame_retries} 次未返回完整 RGB-D 帧"
            )
        if color_bgr.shape[:2] != depth_mm.shape:
            raise RuntimeError(
                f"D2C 后 RGB/Depth 尺寸不一致: {color_bgr.shape[:2]} vs {depth_mm.shape}"
            )
        if color_bgr.shape[:2] != (self.config.height, self.config.width):
            raise RuntimeError(
                f"相机输出尺寸 {color_bgr.shape[:2]} 与配置 {(self.config.height, self.config.width)} 不一致"
            )

        self._depth_mm = np.ascontiguousarray(depth_mm, dtype=np.uint16)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        model_depth = depth_mm_to_model_image(
            self._depth_mm,
            self.config.min_depth_mm,
            self.config.max_depth_mm,
        )
        return color_rgb, model_depth

    def read_depth_mm(self) -> np.ndarray:
        if self._depth_mm is None:
            raise RuntimeError("尚未读取到深度帧")
        return self._depth_mm.copy()

    def disconnect(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
