"""Orbbec Gemini 2 相机驱动。

OrbbecGemini2 类：
  open() / close(): 管理 pyorbbecsdk Pipeline 的生命周期.
  get_frame(): 返回深度对齐到彩色图像的 BGR 和深度毫米帧.
  K / D: 从 SDK 读取的相机内参矩阵和畸变系数.

Gemini 2 相比 DaBai DCW 的简化点：
  - SDK 原生库路径通常与 pyorbbecsdk 安装一致，无需额外预加载.
  - 支持更高分辨率 (1280x720) 的彩色流.
  - 打开流程更简化：直接选择 MJPG/RGB 彩色格式 + Y16 深度格式 + 硬件 D2C 对齐.
"""

from __future__ import annotations

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple

from .base import CameraDriver, CameraFrameError


class OrbbecGemini2(CameraDriver):
    """Orbbec Gemini 2 RGB-D 相机驱动。

    Gemini 2 是一款中高端 RGB-D 相机，支持 1280x720 高清 RGB 和
    硬件 D2C（深度到彩色）对齐，适用于高精度视觉抓取和标定场景。
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        calib_dir: Optional[str] = None,
    ) -> None:
        """保存分辨率、帧率和标定目录；此时不会连接相机。"""
        self._w = width
        self._h = height
        self._fps = fps
        self._calib_dir = Path(calib_dir) if calib_dir else None

        # SDK Pipeline 实例
        self._pipeline = None
        # 深度缩放因子：原始值 * depth_scale_mm = 毫米
        self._depth_scale_mm: float = 1.0
        # 相机内参矩阵 K (3x3) 和畸变系数 D
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._aruco = None
        self._reset_frame_failures()

    # ==========================================
    # 生命周期：打开/关闭
    # ==========================================

    def open(self) -> None:
        """打开 Gemini 2 相机 Pipeline。

        简化打开流程（相比 DaBai DCW）：
          1. 导入 pyorbbecsdk，抑制 SDK 初始化噪声.
          2. 创建 Pipeline，选择彩色流（MJPG 优先，回退 RGB）.
          3. 选择深度流（Y16 格式）.
          4. 启用硬件 D2C 深度对齐 (OBAlignMode.HW_MODE).
          5. 从 SDK 读取相机内参，从本地文件或默认值加载畸变系数.

        异常：
            RuntimeError: 相机未连接、无权限或 SDK 未安装时抛出.
        """
        # 先导入以保持原生加载报错可见（在 stderr 被重定向前）
        try:
            from pyorbbecsdk import (
                Pipeline, Config,
                OBSensorType, OBFormat, OBAlignMode,
                Context,
            )
        except ImportError as e:
            raise RuntimeError(f"pyorbbecsdk is not installed: {e}") from e

        # 抑制 SDK 初始化期间的 stderr 原生库噪声
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)

        try:
            # 设置 SDK 日志级别为 FATAL
            try:
                from pyorbbecsdk import OBLogSeverity
                Context().set_logger_severity(OBLogSeverity.FATAL)
            except Exception:
                pass

            # 创建 Pipeline
            try:
                self._pipeline = Pipeline()
            except Exception as e:
                raise RuntimeError(
                    f"Orbbec camera not found: {e}\n"
                    "  Check USB connection and udev permissions.\n"
                    "  Permission quick fix: sudo chmod a+rw /dev/bus/usb/*/*"
                ) from e

            cfg = Config()

            # 彩色流：MJPG（硬件压缩，带宽低）优先，回退到 RGB（原始数据）
            plist = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            cp = None
            for fmt in (OBFormat.MJPG, OBFormat.RGB):
                try:
                    cp = plist.get_video_stream_profile(self._w, self._h, fmt, self._fps)
                    break
                except Exception:
                    pass
            if cp is None:
                cp = plist.get_default_video_stream_profile()
            cfg.enable_stream(cp)

            # 深度流：Y16 格式（16 位无符号整数，每个单位代表 depth_scale 毫米）
            dplist = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                dp = dplist.get_video_stream_profile(self._w, self._h, OBFormat.Y16, self._fps)
            except Exception:
                dp = dplist.get_default_video_stream_profile()
            cfg.enable_stream(dp)

            # 硬件 D2C 深度到彩色对齐
            cfg.set_align_mode(OBAlignMode.HW_MODE)
            self._pipeline.start(cfg)
            self._reset_frame_failures()

            # 从 SDK 读取 RGB 相机内参，构建内参矩阵 K (3x3)
            intr = self._pipeline.get_camera_param().rgb_intrinsic
            self._K = np.array([
                [intr.fx, 0,       intr.cx],
                [0,       intr.fy, intr.cy],
                [0,       0,       1      ],
            ], dtype=np.float64)

            # 加载畸变系数（本地文件优先，异常大值时回退到零畸变）
            self._D = self._load_distortion()

        finally:
            os.dup2(saved, 2)
            os.close(saved)

    def close(self) -> None:
        """关闭相机 Pipeline，停止流传输。"""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    # ==========================================
    # 帧获取
    # ==========================================

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取一帧对齐后的 RGB 和深度图像。

        返回:
            (color_bgr, depth_mm):
              - color_bgr: uint8 BGR 彩色图像 (H, W, 3).
              - depth_mm: uint16 深度图像 (H, W)，单位为毫米.

        流程:
          1. 等待下一组帧（超时 500ms）.
          2. 彩色帧格式转换：MJPG -> imdecode, RGB -> cvtColor.
          3. 深度帧原始值 * depth_scale 转换为毫米并裁切到 uint16 范围.
        """
        if self._pipeline is None:
            return None, None
        try:
            from pyorbbecsdk import OBFormat
            # 等待下一组帧，超时 500ms
            frames = self._pipeline.wait_for_frames(500)
            if frames is None:
                self._record_frame_failure("wait_for_frames timeout")
                return None, None

            # 彩色帧格式转换
            color_bgr = None
            cf = frames.get_color_frame()
            if cf is not None:
                w, h = cf.get_width(), cf.get_height()
                raw = np.asanyarray(cf.get_data(), dtype=np.uint8)
                fmt = cf.get_format()
                try:
                    if fmt == OBFormat.MJPG:
                        # JPEG 压缩格式：用 cv2.imdecode 解码
                        color_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    elif fmt == OBFormat.RGB:
                        # RGB -> BGR 颜色空间转换
                        color_bgr = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
                    else:
                        # 未知格式：直接 reshape 尝试
                        color_bgr = raw.reshape(h, w, 3)
                except Exception:
                    pass

            # 深度帧格式转换：原始值 -> 毫米
            depth_mm = None
            df = frames.get_depth_frame()
            if df is not None:
                dw, dh = df.get_width(), df.get_height()
                # 16 位无符号整数深度值
                depth_raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(dh, dw)
                depth_scale = self._depth_scale_mm
                try:
                    depth_scale = float(df.get_depth_scale())
                    self._depth_scale_mm = depth_scale
                except Exception:
                    pass
                # 原始值 * depth_scale -> 毫米，四舍五入后裁切到 [0, 65535]
                depth_mm = np.clip(
                    np.rint(depth_raw.astype(np.float32) * depth_scale),
                    0,
                    np.iinfo(np.uint16).max,
                ).astype(np.uint16)

            if color_bgr is None or depth_mm is None:
                self._record_frame_failure("missing color or depth frame")
            else:
                self._reset_frame_failures()
            return color_bgr, depth_mm
        except CameraFrameError:
            raise
        except Exception as exc:
            self._record_frame_failure(str(exc))
            return None, None

    # ==========================================
    # 内参属性
    # ==========================================

    @property
    def K(self) -> np.ndarray:
        """相机内参矩阵 K (3x3)，格式: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]."""
        if self._K is None:
            raise RuntimeError("Camera is not open")
        return self._K

    @property
    def D(self) -> np.ndarray:
        """畸变系数向量 (1, N)，包含 k1, k2, p1, p2, k3 等."""
        if self._D is None:
            raise RuntimeError("Camera is not open")
        return self._D

    # ==========================================
    # 内部实现
    # ==========================================

    def _load_distortion(self) -> np.ndarray:
        """加载畸变系数：优先本地标定文件，异常大值时回退零畸变。

        策略：
          1. 如果配置了 calib_dir 且其中存在 intrinsics.npz，
             加载其中的 dist_coeffs.
          2. 如果 k1 绝对值 > 5.0，视为无效数据，回退到零畸变.
          3. 默认返回 (1, 5) 的全零畸变向量.
        """
        if self._calib_dir is not None:
            npz_path = self._calib_dir / "intrinsics.npz"
            if npz_path.exists():
                try:
                    data = np.load(str(npz_path))
                    D = data["dist_coeffs"].flatten()
                    # k1 通常应在 [-1, 1] 量级，>5 视为无效
                    if abs(D[0]) > 5.0:
                        print(f"[OrbbecGemini2] Invalid k1={D[0]:.2f}; using zero distortion")
                        return np.zeros((1, 5), dtype=np.float64)
                    return D.reshape(1, -1)
                except Exception:
                    pass
        # 默认零畸变
        return np.zeros((1, 5), dtype=np.float64)
