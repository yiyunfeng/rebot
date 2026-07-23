"""ArUco 标记检测与位姿估计。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MarkerPose:
    """单个 ArUco 标记的检测结果。"""
    id: int
    T_marker2cam: np.ndarray   # (4, 4) 标记到相机坐标系的变换


class ArUcoDetector:
    """ArUco 标记检测器，返回最接近相机的目标标记的位姿。

    参数：
        marker_length_m: 标记实际边长（米，黑色边框外沿）
        aruco_dict_id:   cv2.aruco 字典 ID，默认 DICT_4X4_50 = 0
        target_marker_id: 指定只检测该 ID 的标记；None = 使用检测到的第一个
    """

    def __init__(
        self,
        marker_length_m: float = 0.05,
        aruco_dict_id: int = 0,
        target_marker_id: Optional[int] = None,
    ) -> None:
        """保存标记参数，并预先建立检测器和四个三维角点。

        ``_object_points`` 使用标记中心作为原点、米作为单位。后续 PnP 会把
        这些已知三维角点与图像中的四个二维角点配对，求出标记相对相机的位姿。
        """
        self._length = marker_length_m
        self._tid = target_marker_id
        self._dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self._params = self._create_detector_params()
        # OpenCV 新版提供 ArucoDetector 对象；旧版没有时保留 None，检测时调用旧函数接口。
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params) if hasattr(cv2.aruco, "ArucoDetector") else None

        # solvePnP 需要标记四角在真实三维空间中的已知坐标。
        # 原点放在标记中心，Z=0 表示四个角都位于同一张平面上。
        half = self._length / 2.0
        self._object_points = np.array(
            [
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _create_detector_params():
        """创建兼容新旧 OpenCV 的参数对象，并启用亚像素角点优化。"""
        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()

        # 手眼标定依赖角点稳定性。默认角点是像素级，30mm 小标记在 640x360
        # 画面里只有几十个像素宽，亚像素 refinement 可以明显降低 PnP 抖动。
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
        return params

    def _detect_markers(self, gray: np.ndarray):
        """调用当前 OpenCV 版本可用的 ArUco 检测接口。"""
        if self._detector is not None:
            return self._detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self._dict, parameters=self._params)

    def _estimate_pose(self, corner: np.ndarray, K: np.ndarray, D: np.ndarray):
        """用四个角点做 PnP，返回 marker -> camera 的旋转向量和平移。"""
        # ArUco 返回的角点通常带一层 batch 维度，统一整理为 (4, 2)。
        image_points = np.asarray(corner, dtype=np.float64).reshape(4, 2)

        # IPPE_SQUARE 专门针对正方形平面目标，通常比通用迭代法更稳定。
        ok, rvec, tvec = cv2.solvePnP(
            self._object_points,
            image_points,
            np.asarray(K, dtype=np.float64),
            np.asarray(D, dtype=np.float64).reshape(-1, 1),
            flags=getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
        )
        if not ok:
            # 当前 OpenCV/画面条件下 IPPE 失败时，再用默认迭代 PnP 做一次回退。
            ok, rvec, tvec = cv2.solvePnP(
                self._object_points,
                image_points,
                np.asarray(K, dtype=np.float64),
                np.asarray(D, dtype=np.float64).reshape(-1, 1),
            )
        if not ok:
            return None, None
        return rvec.reshape(3), tvec.reshape(3)

    def detect_all(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
    ) -> list[MarkerPose]:
        """返回当前画面中所有检测到的 ArUco 位姿。

        注意：如果画面里有多个相同 ID 的 marker，它们在结果里只能看到相同
        id，无法区分对应哪一个物理纸片；手眼标定时不能把这些结果混用。
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect_markers(gray)
        if ids is None or len(ids) == 0:
            return []

        poses: list[MarkerPose] = []
        # corners 与 ids 顺序一一对应：每组四个像素角点属于同位置的 marker ID。
        for corner, mid in zip(corners, ids.flatten()):
            rvec, tvec = self._estimate_pose(corner, K, D)
            if rvec is None:
                continue
            # solvePnP 返回 Rodrigues 旋转向量，先转成 3×3 旋转矩阵，再组成 4×4 位姿。
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = tvec
            poses.append(MarkerPose(id=int(mid), T_marker2cam=T))

        return poses

    def detect(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
    ) -> Optional[MarkerPose]:
        """
        在图像中检测 ArUco 标记并返回位姿。

        参数：
            bgr: BGR 图像 (H, W, 3)
            K:   相机内参矩阵 (3, 3)
            D:   畸变系数 (1, N) 或 (N,)

        返回：
            MarkerPose 或 None（未检测到）
        """
        poses = self.detect_all(bgr, K, D)
        if not poses:
            return None

        # 筛选目标 ID
        if self._tid is not None:
            poses = [pose for pose in poses if pose.id == self._tid]
            if not poses:
                return None

        # marker -> camera 的平移 Z 是沿相机光轴的距离；多标记时取最近（Z 最小）的。
        return min(poses, key=lambda pose: float(pose.T_marker2cam[2, 3]))

    def draw_detected(
        self,
        bgr: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
        axis_length: float = 0.03,
    ) -> np.ndarray:
        """
        在图像上绘制检测到的所有 ArUco 标记（框 + 坐标轴）。

        返回：
            带标注的 BGR 图像副本
        """
        vis = bgr.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect_markers(gray)

        if ids is None or len(ids) == 0:
            return vis

        # 先画检测到的四边形和 ID，再逐个重新估计位姿并画 XYZ 坐标轴。
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        for corner in corners:
            rvec, tvec = self._estimate_pose(corner, K, D)
            if rvec is not None:
                cv2.drawFrameAxes(vis, K, D, rvec, tvec, axis_length)

        return vis
