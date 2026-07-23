"""Intel RealSense 相机驱动。

RealsenseCamera 类：
  open() / close(): 管理 pyrealsense2 Pipeline 的生命周期.
  get_frame(): 返回深度对齐到彩色图像的 BGR 和深度毫米帧.
  K / D: 相机内参矩阵和畸变系数.

多分辨率回退策略：
  _profile_candidates() 返回按优先级排列的 (宽, 高, 帧率) 候选列表。
  如果请求的分辨率不可用，自动回退到较低分辨率，确保在各种硬件上都能正常打开。
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .base import CameraDriver, CameraFrameError


class RealsenseCamera(CameraDriver):
    """Intel RealSense RGB-D 相机驱动（基于 pyrealsense2）。

    支持的型号：D435i, D405 等 RealSense D400 系列。
    特性：
      - 多分辨率自动回退（当请求的分辨率不支持时自动降级）.
      - 深度到彩色硬件对齐 (rs.align).
      - 深度值自动转换为毫米.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        calib_dir: Optional[str] = None,
    ) -> None:
        """保存首选流配置；设备连接失败时 ``open`` 会尝试备用分辨率。"""
        self._w = width
        self._h = height
        self._fps = fps
        self._calib_dir = Path(calib_dir) if calib_dir else None

        # pyrealsense2 Pipeline 实例
        self._pipeline = None
        # 深度到彩色对齐器
        self._align = None
        # 深度缩放因子：原始 uint16 值 * depth_scale_mm = 毫米
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
        """打开 RealSense 相机 Pipeline。

        打开流程：
          1. 按 _profile_candidates() 返回的优先级列表依次尝试连接.
          2. 对每个候选分辨率 (w, h, fps)，配置彩色流 (bgr8) 和深度流 (z16).
          3. 成功连接后，创建对齐器 (align to color) 并读取相机内参.
          4. 从 depth_sensor 获取 depth_scale（米/单位），转为毫米缩放因子.

        depth_scale 的含义：
          RealSense 深度传感器输出的原始值是固定单位（如 1mm 或 0.1mm）。
          depth_scale 表示每个单位对应的米数（例如 0.001 表示 1mm/单位）。
          乘以 1000 后得到 depth_scale_mm（毫米/单位）.

        异常：
            RuntimeError: 所有候选分辨率都连接失败时抛出.
        """
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(f"pyrealsense2 is not installed: {e}") from e

        # 多分辨率回退：按优先级尝试每个候选配置
        errors = []
        for width, height, fps in self._profile_candidates():
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            try:
                profile = pipeline.start(config)
                # 连接成功
                self._pipeline = pipeline
                # 创建深度到彩色对齐器（将深度图像对齐到彩色图像像素网格）
                self._align = rs.align(rs.stream.color)
                self._w, self._h, self._fps = width, height, fps
                self._reset_frame_failures()
                break
            except RuntimeError as e:
                errors.append(f"{width}x{height}@{fps}: {e}")
                try:
                    pipeline.stop()
                except Exception:
                    pass
        else:
            # for-else: 所有候选都失败
            raise RuntimeError(
                "RealSense camera not found or no RGB-D profile is available:\n  "
                + "\n  ".join(errors)
                + "\n  Check USB connection or other camera clients."
            )

        # 读取彩色流的相机内参（焦距 fx/fy 和主点 cx/cy）
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        # pyrealsense2 不同版本的属性名可能不同 (ppx/cx, ppy/cy)
        cx = getattr(intr, "ppx", getattr(intr, "cx", None))
        cy = getattr(intr, "ppy", getattr(intr, "cy", None))
        if cx is None or cy is None:
            raise RuntimeError(f"Invalid RealSense intrinsics: {intr!r}")
        self._K = np.array([
            [intr.fx, 0,        cx],
            [0,        intr.fy, cy],
            [0,        0,       1      ],
        ], dtype=np.float64)

        # 深度缩放因子：原始 uint16 * depth_scale (m/unit) * 1000 -> mm
        ds = profile.get_device().first_depth_sensor().get_depth_scale()
        self._depth_scale_mm = ds * 1000.0

        # 加载畸变系数（本地文件优先，默认零畸变）
        self._D = self._load_distortion()
        print(f"[RealsenseCamera] Ready ({intr.width}x{intr.height}, "
              f"depth_scale={ds:.6f} m/unit)")

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
          1. 等待下一组帧（超时 500ms），通过 align 将深度对齐到彩色.
          2. 从对齐帧中提取彩色图像 (BGR8) 和深度图像 (Z16).
          3. 深度原始值 * depth_scale_mm -> 毫米值.
        """
        if self._pipeline is None:
            return None, None
        try:
            # 等待下一组帧，超时 500ms
            frames = self._pipeline.wait_for_frames(500)
            # 将对齐器应用到帧组：将深度帧的每个像素重新映射到彩色帧坐标
            aligned = self._align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                self._record_frame_failure("missing color or depth frame")
                return None, None

            # 从对齐帧中提取原始数据
            color_bgr = np.asanyarray(cf.get_data())         # uint8 类型的 BGR 图像
            depth_raw = np.asanyarray(df.get_data())         # uint16, 深度单位
            # 深度原始值 * depth_scale_mm -> 毫米
            depth_mm  = (depth_raw * self._depth_scale_mm).astype(np.uint16)
            self._reset_frame_failures()
            return color_bgr, depth_mm
        except CameraFrameError:
            raise
        except RuntimeError as exc:
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
                        print(f"[RealsenseCamera] Invalid k1={D[0]:.2f}; using zero distortion")
                        return np.zeros((1, 5), dtype=np.float64)
                    return D.reshape(1, -1)
                except Exception:
                    pass
        # 默认零畸变
        return np.zeros((1, 5), dtype=np.float64)

    def _profile_candidates(self) -> list[tuple[int, int, int]]:
        """返回按优先级排列的 (宽, 高, 帧率) 候选列表。

        多分辨率回退策略：
          当请求的分辨率/帧率在当前硬件上不可用时，自动依次尝试较低分辨率，
          确保 D405 (848x480) 和 D435i (1280x720) 等不同型号都能正常连接。

        优先级顺序:
          1. 用户请求的分辨率 (self._w, self._h, self._fps).
          2. 1280x720@15  (D435i 常见备用配置).
          3. 848x480@30   (D405 原生分辨率).
          4. 640x480@30   (VGA 通用回退).
          5. 640x480@15   (最低保底配置).

        返回：
            去重后的候选配置列表.
        """
        requested = (int(self._w), int(self._h), int(self._fps))
        candidates = [
            requested,
            (1280, 720, 15),
            (848, 480, 30),
            (640, 480, 30),
            (640, 480, 15),
        ]
        # 去重（保持顺序）
        unique = []
        seen = set()
        for item in candidates:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique
