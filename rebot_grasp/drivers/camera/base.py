"""RGB-D 相机共用接口。

CameraDriver：
  open() / close()：启动或停止相机数据流。
  get_frame()：返回一帧 BGR 彩色图和一帧毫米单位的深度图。
  K / D：相机内参矩阵和畸变系数。
  warm_up()：推理或标定前丢弃相机刚启动时的不稳定帧。
  setup_aruco() / detect_aruco() / draw_aruco()：可选的 ArUco 辅助功能。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class CameraFrameError(RuntimeError):
    """相机连续多次取帧失败后抛出的异常。"""


class CameraDriver(ABC):
    """彩色与深度相机的统一抽象接口。"""

    _FRAME_FAIL_WARN = 10
    _FRAME_FAIL_LIMIT = 60

    def _reset_frame_failures(self) -> None:
        """成功取帧或重新打开相机后，将连续失败次数清零。"""
        self._frame_failures = 0

    def _record_frame_failure(self, reason: str) -> None:
        """累计连续取帧失败；先告警，达到上限后抛错，避免无限空转。"""
        count = int(getattr(self, "_frame_failures", 0)) + 1
        self._frame_failures = count
        if count == self._FRAME_FAIL_WARN:
            print(f"[Camera] {count} consecutive frame failures: {reason}")
        if count >= self._FRAME_FAIL_LIMIT:
            raise CameraFrameError(
                f"Camera failed for {count} consecutive frames: {reason}\n"
                "  Check connection, USB permissions, or other camera clients."
            )

    # 生命周期

    @abstractmethod
    def open(self) -> None:
        """打开相机数据流。"""

    @abstractmethod
    def close(self) -> None:
        """停止数据流并释放相机资源。"""

    # 图像帧

    @abstractmethod
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """返回一对彩色帧和深度帧。

        返回：
            color_bgr：uint8 类型的 BGR 图像，取帧失败时为 None。
            depth_mm：uint16 类型、毫米单位的深度图，取帧失败时为 None。
        """

    # 相机内参

    @property
    @abstractmethod
    def K(self) -> np.ndarray:
        """返回形状为 (3, 3)、类型为 float64 的相机内参矩阵。"""

    @property
    @abstractmethod
    def D(self) -> np.ndarray:
        """返回形状为 (1, N)、类型为 float64 的畸变系数。"""

    # 辅助功能

    def warm_up(self, n_frames: int = 20) -> None:
        """丢弃刚启动的若干帧，等待自动曝光等参数稳定。"""
        # 相机刚启动时自动曝光和白平衡尚未稳定，主动丢弃前若干帧。
        for _ in range(n_frames):
            self.get_frame()

    def setup_aruco(
        self,
        marker_length_m: float,
        dict_id: int = 0,
        target_marker_id: Optional[int] = None,
    ) -> None:
        """使用当前相机标定参数创建 ArUco 检测器。

        参数：
            marker_length_m：标记边长，单位为米。
            dict_id：cv2.aruco 字典编号。
            target_marker_id：只检测该编号；为 None 时选择最近的标记。
        """
        # 延迟导入避免普通抓取不使用 ArUco 时也加载标定模块。
        from calibration.aruco_pose import ArUcoDetector
        self._aruco = ArUcoDetector(marker_length_m, dict_id, target_marker_id)

    def detect_aruco(self, bgr: np.ndarray):
        """检测一张 ArUco 标记；调用前必须先执行 setup_aruco()。"""
        return self._aruco.detect(bgr, self.K, self.D)

    def draw_aruco(self, bgr: np.ndarray) -> np.ndarray:
        """绘制检测到的 ArUco 标记；调用前必须先执行 setup_aruco()。"""
        return self._aruco.draw_detected(bgr, self.K, self.D)

    # 上下文管理器

    def __enter__(self) -> "CameraDriver":
        """进入 ``with`` 代码块时自动打开相机。"""
        self.open()
        return self

    def __exit__(self, *_) -> None:
        """离开 ``with`` 代码块时自动释放相机资源。"""
        self.close()
