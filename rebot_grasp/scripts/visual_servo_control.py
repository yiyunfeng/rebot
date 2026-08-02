#!/usr/bin/env python3
"""ArUco 眼在手视觉伺服。

运行方式只有：

    python3 scripts/visual_servo_control.py
    存在一定的问题，重力补偿对joint不准，应该是模型结构不准

程序启动后先进入 MIT 重力补偿拖动模式。先手动把夹爪拖到水平姿态；连续
识别到 ArUco 后锁定此刻的 TCP 姿态，后续只做平移，使标记位于相机光心且
距离保持为设定值。ArUco 平面倾斜或纸面内旋转都不会带动夹爪转动。短暂漏检
不会启停，确认丢失后恢复重力补偿拖动；再次出现会重新锁定当时的 TCP 姿态
并自动跟随。一次 Ctrl+C 会停止跟随、回 home、失能并断开。

视觉到控制的变换链全部使用齐次矩阵，不再添加经验方向符号：

    T_marker_base = T_tcp_base @ T_camera_tcp @ T_marker_camera

自动接管时锁定 TCP 旋转 R_tcp_fixed，相机旋转也随之固定：

    R_camera_fixed = R_tcp_fixed @ R_camera_tcp

相机坐标中 ArUco 中心为 [x, y, z]。要让它回到画面中心并保持距离 d，
相机需要平移 [x, y, z-d]；写到 base 系的等价期望位置为：

    p_camera_desired = p_marker_base - d * R_camera_fixed[:, 2]
    T_tcp_base_desired = T_camera_base_desired @ inverse(T_camera_tcp)

其中 T_camera_base_desired 的旋转始终是 R_camera_fixed，因此求 IK 后 TCP
姿态始终不变，只有位置跟随。再由 500 Hz MIT 位置/速度/重力控制跟踪：

    tau = Kp*(q_command-q) + Kd*(qd_command-qd) + g(q)

MIT 增益、重力补偿和 27/7 N*m 峰值边界与刚刚真机跑通的位姿运动基线保持
一致。本脚本没有环境碰撞检测；运行前必须清空工作空间并确认急停可用。
"""

from __future__ import annotations

import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SDK_ROOT = REPOSITORY_ROOT / "third_party" / "reBotArm_control_py"
for search_path in (PROJECT_ROOT, SDK_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from drivers.camera import CameraFrameError, make_camera  # noqa: E402

# =============================================================================
# 用户配置：正常使用只改这两个值
# =============================================================================

# 相机到 ArUco 平面的期望距离，单位 m。
FOLLOW_DISTANCE_M = 0.3

# TCP 位置目标的最大跟随速度，单位 m/s。
FOLLOW_SPEED_M_S = 0.10

# =============================================================================
# 内部固定值：控制基线，不作为命令行参数
# =============================================================================

CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

ARUCO_MARKER_LENGTH_M = 0.100
ARUCO_DICTIONARY_ID = 0
ARUCO_MARKER_ID = 0
ARUCO_MIN_AREA_PX2 = 300.0
ARUCO_MAX_REPROJECTION_ERROR_PX = 2.5
CAMERA_WARMUP_FRAMES = 20

# 目标需要连续出现 3 帧才接管机械臂。连续漏检 5 帧且已经超过 0.15 s，或
# 最后一次观测超过 0.5 s，才确认丢失；单帧曝光/USB 抖动不会触发启停。
TARGET_CONFIRM_FRAMES = 3
TARGET_LOST_FRAMES = 5
TARGET_LOST_MIN_TIME_S = 0.15
TARGET_LOST_TIMEOUT_S = 0.50
# 一阶低通：值越大越跟手，值越小越平滑。30 FPS 下取 0.50，可明显减少
# 旧值 0.35 带来的滞后，同时保留足够的 RGB-D 深度抖动抑制。
TARGET_POSITION_FILTER_ALPHA = 0.50

# 自动接管前确认机械臂已经停止。否则用户仍在手动拖动时从 Kp=2 突然切到
# Kp=120，会把手动速度变成很大的位置误差。
ACQUIRE_MAX_JOINT_SPEED_RAD_S = 0.05
ACQUIRE_STABLE_TIME_S = 0.25

# 位置目标先在 Cartesian 空间限速。姿态在接管时锁定，不参与视觉控制；
# FOLLOW_SPEED_M_S 直接约束 TCP 平移目标，不是仅用于显示。
TCP_POSITION_DEADBAND_M = 0.003

# 与刚刚真机通过的 MIT 位姿运动参数相同。
CONTROL_RATE_HZ = 500.0
TRACK_KP = np.array([120.0, 120.0, 120.0, 18.0, 18.0, 18.0])
TRACK_KD = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0])
DRAG_KP = np.full(6, 2.0, dtype=np.float64)
DRAG_KD = np.full(6, 1.5, dtype=np.float64)
MOTOR_PEAK_TORQUES_NM = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0])

# 当前 URDF 没有单独建模眼在手相机。项目 SDK 的 RebotArmEndPose 控制器
# 对 tau_g[1]（joint2）和 tau_g[2]（joint3）使用 1.55 倍作为参考；当前按
# 真机调试值取 1.30，其余四轴仍使用模型原值。
GRAVITY_TORQUE_SCALE = np.array([1.0, 1.3, 1.3, 1.0, 1.0, 1.0])

# 关节参考仍使用制动速度曲线，不会直接阶跃。0.30 rad/s 和 0.80 rad/s^2
# 让机械臂约 0.375 s 加速到上限，比旧配置更适合 0.10 m/s 的视觉跟随。
MAX_JOINT_REFERENCE_SPEED_RAD_S = 0.30
MAX_JOINT_REFERENCE_ACCELERATION_RAD_S2 = 0.80

HOME_JOINTS_RAD = np.zeros(6, dtype=np.float64)
HOME_TIMEOUT_S = 15.0
HOME_POSITION_TOLERANCE_RAD = 0.040
HOME_SPEED_TOLERANCE_RAD_S = 0.050
HOME_STABLE_TIME_S = 0.40

JOINT_LIMIT_TOLERANCE_RAD = 0.02
MAX_FEEDBACK_SPEED_RAD_S = 2.0
STALL_POSITION_ERROR_RAD = 0.08
STALL_SPEED_RAD_S = 0.015
STALL_TORQUE_UTILIZATION = 0.95
STALL_TIME_S = 1.0
CONTROL_LOG_INTERVAL_S = 0.5
VISION_LOG_INTERVAL_S = 0.5
VISION_RATE_WINDOW_S = 1.0

MODE_DRAG = "gravity_drag"
MODE_TRACK = "visual_track"


def validate_user_config() -> None:
    """只校验文件顶部两个用户参数。"""
    if not 0.20 <= FOLLOW_DISTANCE_M <= 1.00:
        raise ValueError("FOLLOW_DISTANCE_M 必须在 [0.20, 1.00] m 内")
    if not 0.001 <= FOLLOW_SPEED_M_S <= 0.10:
        raise ValueError("FOLLOW_SPEED_M_S 必须在 [0.001, 0.10] m/s 内")


def validate_transform(transform: np.ndarray, name: str) -> np.ndarray:
    """校验并复制 4x4 刚体变换。"""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 必须是有限的 4x4 矩阵")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} 最后一行不是齐次变换格式")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} 旋转部分不是正交矩阵")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError(f"{name} 旋转部分不是 SO(3)")
    return matrix.copy()


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """利用刚体结构计算逆变换，避免通用矩阵求逆引入无意义误差。"""
    matrix = validate_transform(transform, "transform")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """把旋转矩阵转换为轴角向量。"""
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    return vector.reshape(3)


def rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    """把轴角向量转换为旋转矩阵。"""
    rotation, _ = cv2.Rodrigues(np.asarray(vector, dtype=np.float64).reshape(3))
    return rotation


def desired_marker_to_camera(distance_m: float = FOLLOW_DISTANCE_M) -> np.ndarray:
    """返回 marker 在理想相机画面中的位姿。

    OpenCV 相机坐标为 X 向右、Y 向下、Z 向前。这里定义的 marker 角点坐标
    为 X 向右、Y 向上，marker 正面法向 +Z 朝向相机。因此正对标记时：

        R_marker_camera = diag(1, -1, -1)
        t_marker_camera = [0, 0, distance]

    这同时表达了“画面居中、距离固定、光轴垂直标记平面”。
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = [0.0, 0.0, float(distance_m)]
    return transform


def compute_desired_tcp_pose(
    marker_to_base: np.ndarray,
    camera_to_tcp: np.ndarray,
    fixed_tcp_to_base: np.ndarray,
    distance_m: float = FOLLOW_DISTANCE_M,
) -> np.ndarray:
    """计算“固定 TCP 姿态、标记居中且定距”的期望 TCP 位姿。

    fixed_tcp_to_base 是自动接管瞬间的 TCP 位姿。本函数只使用它的旋转，
    后续无论 ArUco 平面如何倾斜，期望 TCP 旋转都保持不变。

    固定相机旋转后，期望位置只由 ArUco 中心决定。令相机光轴单位向量为
    z_camera_base，则把相机放到

        p_marker_base - distance * z_camera_base

    就会让标记中心落在光轴上，并保持指定的光轴方向距离。
    """
    marker_base = validate_transform(marker_to_base, "T_marker_base")
    camera_tcp = validate_transform(camera_to_tcp, "T_camera_tcp")
    fixed_tcp = validate_transform(fixed_tcp_to_base, "T_tcp_base_fixed")
    fixed_camera_rotation = fixed_tcp[:3, :3] @ camera_tcp[:3, :3]
    camera_z_base = fixed_camera_rotation[:, 2]

    desired_camera = np.eye(4, dtype=np.float64)
    desired_camera[:3, :3] = fixed_camera_rotation
    desired_camera[:3, 3] = (
        marker_base[:3, 3] - float(distance_m) * camera_z_base
    )
    desired = desired_camera @ invert_transform(camera_tcp)
    return validate_transform(desired, "T_tcp_base_desired")


class PoseCommandLimiter:
    """只平滑 TCP 位置；接管时锁定的 TCP 姿态始终不变。"""

    def __init__(self) -> None:
        self.pose: Optional[np.ndarray] = None
        self.timestamp: Optional[float] = None

    def reset(self, current_tcp_pose: np.ndarray, timestamp: float) -> None:
        self.pose = validate_transform(current_tcp_pose, "current_tcp_pose")
        self.timestamp = float(timestamp)

    def clear(self) -> None:
        self.pose = None
        self.timestamp = None

    def step(self, desired_pose: np.ndarray, timestamp: float) -> np.ndarray:
        """以平移速度上限向最新期望位置走一步，不修改当前姿态。"""
        desired = validate_transform(desired_pose, "desired_pose")
        if self.pose is None or self.timestamp is None:
            self.reset(desired, timestamp)
            return desired

        # 相机长时间卡顿后不能把全部积累位移一次补发；单步最多按 0.1 s 算。
        dt = float(np.clip(float(timestamp) - self.timestamp, 1e-3, 0.1))
        current = self.pose.copy()

        delta_position = desired[:3, 3] - current[:3, 3]
        distance = float(np.linalg.norm(delta_position))
        movable_distance = max(0.0, distance - TCP_POSITION_DEADBAND_M)
        if movable_distance > 0.0:
            step_length = min(FOLLOW_SPEED_M_S * dt, movable_distance)
            current[:3, 3] += delta_position * (step_length / distance)

        self.pose = current
        self.timestamp = float(timestamp)
        return current.copy()


class MarkerPoseFilter:
    """在 base 系滤波标记位姿，并要求连续确认。

    不能直接对 camera 系目标做长时间滤波，因为相机本身在运动。先通过手眼
    变换把每帧标记换到 base 系，再滤波，静止标记在 base 系应保持不动。
    """

    def __init__(self) -> None:
        self.pose: Optional[np.ndarray] = None
        self.last_seen: Optional[float] = None
        self.confirmed_frames = 0

    def reset(self) -> None:
        self.pose = None
        self.last_seen = None
        self.confirmed_frames = 0

    def update(self, marker_to_base: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        measurement = validate_transform(marker_to_base, "T_marker_base")
        now = float(timestamp)
        if self.last_seen is not None and now - self.last_seen > TARGET_LOST_TIMEOUT_S:
            self.reset()

        if self.pose is None:
            self.pose = measurement
        else:
            alpha_p = TARGET_POSITION_FILTER_ALPHA
            self.pose[:3, 3] = (
                alpha_p * measurement[:3, 3]
                + (1.0 - alpha_p) * self.pose[:3, 3]
            )
            # 控制只使用标记中心位置；ArUco 姿态不参与末端姿态控制。

        self.last_seen = now
        self.confirmed_frames += 1
        return self.current(now)

    def current(self, timestamp: float) -> Optional[np.ndarray]:
        if self.pose is None or self.last_seen is None:
            return None
        if float(timestamp) - self.last_seen > TARGET_LOST_TIMEOUT_S:
            return None
        if self.confirmed_frames < TARGET_CONFIRM_FRAMES:
            return None
        return self.pose.copy()

    def age(self, timestamp: float) -> Optional[float]:
        if self.last_seen is None:
            return None
        return max(0.0, float(timestamp) - self.last_seen)


@dataclass
class TargetDetection:
    """一帧 ArUco 位姿和预览所需信息。"""

    marker_to_camera: np.ndarray
    center_px: tuple[int, int]
    corners_px: np.ndarray
    source: str
    reprojection_error_px: float

    @property
    def position_camera(self) -> np.ndarray:
        return self.marker_to_camera[:3, 3]


class ArucoTracker:
    """检测指定 ArUco，并用 RGB-D 中心深度改善平移估计。"""

    def __init__(self) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV 缺少 aruco；需要 opencv-contrib-python")
        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_ID)
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.parameters = cv2.aruco.DetectorParameters()
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 50
        self.parameters.cornerRefinementMinAccuracy = 0.01
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        half = ARUCO_MARKER_LENGTH_M / 2.0
        self.object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )
        self.status = "waiting"

    def _detect_markers(self, gray: np.ndarray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.parameters,
        )

    def _solve_pnp(
        self,
        corners_px: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> Optional[tuple[np.ndarray, float]]:
        image_points = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
            flags=getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
        )
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                self.object_points,
                image_points,
                np.asarray(camera_matrix, dtype=np.float64),
                np.asarray(distortion, dtype=np.float64),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        if not ok or float(tvec[2, 0]) <= 0.0:
            return None

        projected, _ = cv2.projectPoints(
            self.object_points,
            rvec,
            tvec,
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
        )
        residual = projected.reshape(4, 2) - image_points
        reprojection_error = float(
            np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_from_vector(rvec.reshape(3))
        transform[:3, 3] = tvec.reshape(3)
        return transform, reprojection_error

    @staticmethod
    def _position_from_depth(
        corners_px: np.ndarray,
        depth_mm: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> Optional[np.ndarray]:
        """在标记中央区域取深度中位数，再反投影标记中心。"""
        if depth_mm is None or depth_mm.ndim != 2:
            return None
        height, width = depth_mm.shape
        points = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
        center = np.mean(points, axis=0)
        inner = center + 0.60 * (points - center)
        inner[:, 0] = np.clip(inner[:, 0], 0, width - 1)
        inner[:, 1] = np.clip(inner[:, 1], 0, height - 1)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(inner).astype(np.int32), 1)
        samples = np.asarray(depth_mm, dtype=np.float64)[mask > 0]
        samples = samples[np.isfinite(samples) & (samples > 0.0)]
        if len(samples) < 25:
            return None

        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        tolerance = max(8.0, 3.0 * 1.4826 * mad)
        inliers = samples[np.abs(samples - median) <= tolerance]
        if len(inliers) < 25:
            return None
        depth_m = float(np.median(inliers) / 1000.0)
        if not 0.15 <= depth_m <= 1.20:
            return None

        normalized = cv2.undistortPoints(
            np.asarray([[center]], dtype=np.float64),
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
        ).reshape(2)
        return np.array(
            [normalized[0] * depth_m, normalized[1] * depth_m, depth_m],
            dtype=np.float64,
        )

    def detect(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> Optional[TargetDetection]:
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect_markers(gray)
        if ids is None:
            self.status = "aruco_not_found"
            return None

        candidates: list[TargetDetection] = []
        rejection = "target_id_not_found"
        for corner, marker_id in zip(corners, ids.flatten()):
            if int(marker_id) != ARUCO_MARKER_ID:
                continue
            points = np.asarray(corner, dtype=np.float64).reshape(4, 2)
            area = abs(float(cv2.contourArea(points.astype(np.float32))))
            if area < ARUCO_MIN_AREA_PX2:
                rejection = f"small_area:{area:.0f}px2"
                continue
            solved = self._solve_pnp(points, camera_matrix, distortion)
            if solved is None:
                rejection = "pnp_failed"
                continue
            marker_camera, reprojection_error = solved
            if reprojection_error > ARUCO_MAX_REPROJECTION_ERROR_PX:
                rejection = f"reprojection:{reprojection_error:.2f}px"
                continue

            # 正面朝向相机时 marker +Z 法向与 camera->marker 向量点积为负。
            # 拒绝平面 PnP 的背面解，防止姿态突然翻转 180 度。
            if (
                np.dot(marker_camera[:3, 2], marker_camera[:3, 3])
                >= 0.0
            ):
                rejection = "pnp_back_facing"
                continue

            depth_position = self._position_from_depth(
                points,
                depth_mm,
                camera_matrix,
                distortion,
            )
            source = "pnp"
            if depth_position is not None:
                marker_camera[:3, 3] = depth_position
                source = "rgbd"
            center = np.mean(points, axis=0)
            candidates.append(
                TargetDetection(
                    marker_to_camera=marker_camera,
                    center_px=(int(round(center[0])), int(round(center[1]))),
                    corners_px=np.rint(points).astype(np.int32),
                    source=source,
                    reprojection_error_px=reprojection_error,
                )
            )

        if not candidates:
            self.status = rejection
            return None
        detection = min(candidates, key=lambda value: value.position_camera[2])
        self.status = f"aruco_ok_{detection.source}"
        return detection


def next_joint_reference(
    q_reference: np.ndarray,
    qd_reference: np.ndarray,
    q_goal: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """生成连续的关节位置/速度参考。

    对每轴先由剩余距离计算不会越过目标的制动速度：

        |qd_brake| = sqrt(2 * a_max * |q_goal-q_ref|)

    再同时施加速度上限和每周期加速度上限。视觉目标即使每帧更新，q_ref 和
    qd_ref 仍连续，因此不会形成反复启动、急停。
    """
    q_ref = np.asarray(q_reference, dtype=np.float64).copy()
    qd_ref = np.asarray(qd_reference, dtype=np.float64).copy()
    goal = np.asarray(q_goal, dtype=np.float64)
    error = goal - q_ref
    braking_speed = np.sqrt(
        2.0 * MAX_JOINT_REFERENCE_ACCELERATION_RAD_S2 * np.abs(error)
    )
    desired_speed = np.sign(error) * np.minimum(
        braking_speed,
        MAX_JOINT_REFERENCE_SPEED_RAD_S,
    )
    max_velocity_change = MAX_JOINT_REFERENCE_ACCELERATION_RAD_S2 * float(dt)
    qd_next = qd_ref + np.clip(
        desired_speed - qd_ref,
        -max_velocity_change,
        max_velocity_change,
    )
    step = qd_next * float(dt)
    q_next = q_ref + step
    at_goal = np.abs(error) <= 1e-12
    crossed = (np.abs(error) > 0.0) & (error * (error - step) <= 0.0)
    # 跨过目标时只把位置参考停在目标；速度参考不能瞬间清零。后续周期在
    # q_ref=goal 处继续按 a_max 递减 qd_ref，避免到点时产生一次急刹。
    q_next[crossed | at_goal] = goal[crossed | at_goal]
    return q_next, qd_next


def compute_mit_command(
    q: np.ndarray,
    qd: np.ndarray,
    q_reference: np.ndarray,
    qd_reference: np.ndarray,
    gravity: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把目标变成满足 27/7 N*m 峰值边界的 MIT 命令。

    总扭矩只按厂家峰值边界处理。先给重力和速度项分配扭矩，再限制位置项；
    反算 q_command 后，电机内部仍执行标准 MIT 控制律。
    """
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    q_ref = np.asarray(q_reference, dtype=np.float64)
    qd_ref = np.asarray(qd_reference, dtype=np.float64)
    tau_g = np.asarray(gravity, dtype=np.float64)
    kp = np.asarray(kp, dtype=np.float64)
    kd = np.asarray(kd, dtype=np.float64)
    if np.any(np.abs(tau_g) >= MOTOR_PEAK_TORQUES_NM):
        raise RuntimeError("模型重力达到电机峰值扭矩")

    velocity_error = qd_ref - qd
    velocity_torque = np.clip(
        kd * velocity_error,
        -MOTOR_PEAK_TORQUES_NM - tau_g,
        MOTOR_PEAK_TORQUES_NM - tau_g,
    )
    kd_effective = kd.copy()
    moving = np.abs(velocity_error) > 1e-9
    kd_effective[moving] = np.clip(
        velocity_torque[moving] / velocity_error[moving],
        0.0,
        kd[moving],
    )
    tau_without_position = tau_g + kd_effective * velocity_error

    position_torque = np.clip(
        kp * (q_ref - q),
        -MOTOR_PEAK_TORQUES_NM - tau_without_position,
        MOTOR_PEAK_TORQUES_NM - tau_without_position,
    )
    q_command = q + position_torque / kp
    tau_total = tau_without_position + position_torque
    if np.any(np.abs(tau_total) > MOTOR_PEAK_TORQUES_NM + 1e-9):
        raise RuntimeError("MIT 总扭矩限幅内部错误")
    return q_command, qd_ref.copy(), kd_effective, tau_total


def read_initial_arm_state(
    robot: Any,
    timeout_s: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """首次连接时逐轴读取反馈；函数本身不使能、不发送运动命令。"""
    deadline = time.monotonic() + timeout_s
    missing = list(robot.arm.joint_names)
    while time.monotonic() < deadline:
        states = [robot._motor_map[name].get_state() for name in robot.arm.joint_names]
        if all(state is not None for state in states):
            robot.arm.get_positions(request_feedback=True)
        else:
            for config in robot.arm._jcfgs:
                motor = robot._motor_map[config.name]
                motor.request_feedback()
                robot._ctrl_map[config.vendor].poll_feedback_once()
                time.sleep(0.005)
        states = [robot._motor_map[name].get_state() for name in robot.arm.joint_names]
        if all(state is not None for state in states):
            q = np.array([state.pos for state in states], dtype=np.float64)
            qd = np.array([state.vel for state in states], dtype=np.float64)
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
                raise RuntimeError("arm 反馈包含 NaN/Inf")
            return q, qd
        missing = [
            name for name, state in zip(robot.arm.joint_names, states) if state is None
        ]
        time.sleep(0.02)
    raise RuntimeError(f"未收到全部 arm 电机反馈，缺少: {missing}")


class RobotVisualServo:
    """真机生命周期、IK 和 500 Hz MIT 控制器。"""

    def __init__(self) -> None:
        if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
            raise RuntimeError("当前 Pinocchio 环境要求 NumPy < 2")
        try:
            import pinocchio as pin
            from reBotArm_control_py.actuator import RebotArm
            from reBotArm_control_py.dynamics import compute_generalized_gravity
            from reBotArm_control_py.kinematics import (
                get_end_effector_frame_id,
                load_robot_model,
                pad_q_for_model,
            )
            from reBotArm_control_py.kinematics.inverse_kinematics import (
                IKParams,
                solve_ik,
            )
        except Exception as exc:
            raise RuntimeError("无法加载 Pinocchio/reBotArm SDK") from exc

        self.pin = pin
        self.robot = RebotArm()
        self.model = load_robot_model()
        self.pad_q_for_model = pad_q_for_model
        self.compute_gravity = compute_generalized_gravity
        self.solve_ik_fn = solve_ik
        self.ik_params_type = IKParams
        self.ee_frame_id = get_end_effector_frame_id(self.model)
        self.gravity_data = self.model.createData()
        self.fk_data = self.model.createData()
        self.ik_data = self.model.createData()

        self.command_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.mode = MODE_DRAG
        self.q_goal: Optional[np.ndarray] = None
        self.q_reference: Optional[np.ndarray] = None
        self.qd_reference = np.zeros(6, dtype=np.float64)
        self.latest_q: Optional[np.ndarray] = None
        self.latest_qd: Optional[np.ndarray] = None
        self.error: Optional[Exception] = None
        self.stop_event = threading.Event()
        self.connected = False
        self.running = False
        self.last_log = 0.0
        self.log_cycles = 0
        self.stall_started: Optional[float] = None

    @staticmethod
    def _angles_near_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        delta = np.asarray(values) - np.asarray(reference)
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        return np.asarray(reference) + delta

    def _check_feedback(self, q: np.ndarray, qd: np.ndarray) -> None:
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            raise RuntimeError("关节反馈包含 NaN/Inf")
        if np.any(np.abs(qd) > MAX_FEEDBACK_SPEED_RAD_S):
            raise RuntimeError("关节反馈速度超过 2 rad/s")
        lower = self.model.lowerPositionLimit[:6] - JOINT_LIMIT_TOLERANCE_RAD
        upper = self.model.upperPositionLimit[:6] + JOINT_LIMIT_TOLERANCE_RAD
        invalid = np.flatnonzero((q < lower) | (q > upper))
        if invalid.size:
            names = [self.robot.arm.joint_names[index] for index in invalid]
            raise RuntimeError(f"关节反馈越过 URDF 限位: {names}")

    def _gravity(self, q: np.ndarray) -> np.ndarray:
        q_model = self.pad_q_for_model(self.model, q, 6)
        gravity_model = self.compute_gravity(
            model=self.model,
            q=q_model,
            data=self.gravity_data,
        )[:6]
        return gravity_model * GRAVITY_TORQUE_SCALE

    def _send(
        self,
        q_command: np.ndarray,
        qd_command: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        gravity: np.ndarray,
    ) -> None:
        """逐电机发送；SDK 批量接口会吞 CallError，所以这里不用批量接口。"""
        for index, joint_name in enumerate(self.robot.arm.joint_names):
            try:
                self.robot._motor_map[joint_name].send_mit(
                    float(q_command[index]),
                    float(qd_command[index]),
                    float(kp[index]),
                    float(kd[index]),
                    float(gravity[index]),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{joint_name} MIT 下发失败: {type(exc).__name__}: {exc}"
                ) from exc

    def _enable_with_gravity_drag(self, q_hold: np.ndarray) -> None:
        """逐轴使能后立即发送重力帧，避免六轴使能过程中的下坠空窗。"""
        gravity = self._gravity(q_hold)
        if np.any(np.abs(gravity) >= MOTOR_PEAK_TORQUES_NM):
            raise RuntimeError("启动位姿模型重力达到电机峰值")
        for index, joint_name in enumerate(self.robot.arm.joint_names):
            motor = self.robot._motor_map[joint_name]
            motor.enable()
            motor.send_mit(
                float(q_hold[index]),
                0.0,
                float(DRAG_KP[index]),
                float(DRAG_KD[index]),
                float(gravity[index]),
            )
        self._send(q_hold, np.zeros(6), DRAG_KP, DRAG_KD, gravity)

    def start(self) -> None:
        """连接机械臂并直接进入可手动拖动的 MIT 重力补偿。"""
        print("[Hardware] Review before motion:")
        print(f"  yaml:    {self.robot.hardware_yaml}")
        print(f"  channel: {self.robot._channel}")
        print(f"  joints:  {self.robot.arm.joint_names}")
        print(f"  models:  {[item.model for item in self.robot.arm._jcfgs]}")
        print(f"  peak command cap: {MOTOR_PEAK_TORQUES_NM.tolist()} N*m")
        print(f"  gravity torque scale: {GRAVITY_TORQUE_SCALE.tolist()}")
        print(f"  track Kp: {TRACK_KP.tolist()}")
        print(f"  track Kd: {TRACK_KD.tolist()}")
        if any(item.vendor != "damiao" for item in self.robot.arm._jcfgs):
            raise RuntimeError("本脚本只验证了 Damiao 六轴 arm")
        try:
            self.robot.connect()
            self.connected = True
            self.robot.disable_all()
            q_start, qd_start = read_initial_arm_state(self.robot)
            self._check_feedback(q_start, qd_start)

            if not self.robot.arm.mode_mit(kp=DRAG_KP, kd=DRAG_KD):
                raise RuntimeError("部分电机切换 MIT 模式失败")
            self.robot.disable_all()
            time.sleep(0.1)
            q_before_enable, _ = read_initial_arm_state(self.robot)
            self._enable_with_gravity_drag(q_before_enable)
            q_enable, qd_enable = read_initial_arm_state(self.robot)
            self._check_feedback(q_enable, qd_enable)

            with self.state_lock:
                self.latest_q = q_enable.copy()
                self.latest_qd = qd_enable.copy()
            with self.command_lock:
                self.mode = MODE_DRAG
                self.q_goal = q_enable.copy()
                self.q_reference = q_enable.copy()
                self.qd_reference.fill(0.0)

            print(
                "[Startup] gravity drag ready: "
                f"q={np.array2string(q_enable, precision=3)}, "
                f"disabled_drop={np.max(np.abs(q_before_enable-q_start)):.4f}rad"
            )
            self.robot.start_control_loop(self, rate=CONTROL_RATE_HZ)
            self.running = True
            print(
                "[Motion] MIT gravity compensation drag active: "
                "pos=q, Kp=2, Kd=1.5, tau=g(q)."
            )
        except Exception:
            self.shutdown()
            raise

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        with self.state_lock:
            if self.latest_q is None or self.latest_qd is None:
                raise RuntimeError("机械臂反馈尚未初始化")
            return self.latest_q.copy(), self.latest_qd.copy()

    def goal_seed(self) -> np.ndarray:
        with self.command_lock:
            if self.q_goal is not None:
                return self.q_goal.copy()
        return self.state()[0]

    def set_joint_goal(self, q_goal: np.ndarray) -> None:
        goal = np.asarray(q_goal, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(goal)):
            raise ValueError("IK 目标包含 NaN/Inf")
        lower = self.model.lowerPositionLimit[:6]
        upper = self.model.upperPositionLimit[:6]
        if np.any((goal < lower) | (goal > upper)):
            raise ValueError("IK 目标越过 URDF 限位")
        q_actual, _ = self.state()
        with self.command_lock:
            if self.mode == MODE_DRAG or self.q_reference is None:
                self.q_reference = q_actual.copy()
                self.qd_reference.fill(0.0)
            self.q_goal = goal.copy()
            self.mode = MODE_TRACK

    def enter_gravity_drag(self) -> None:
        """确认目标丢失后撤销位置目标，下一周期 pos 直接等于实测 q。"""
        q, _ = self.state()
        with self.command_lock:
            self.mode = MODE_DRAG
            self.q_goal = q.copy()
            self.q_reference = q.copy()
            self.qd_reference.fill(0.0)
        self.stall_started = None

    def current_tcp_pose(self) -> np.ndarray:
        q, _ = self.state()
        q_model = self.pad_q_for_model(self.model, q, 6)
        self.pin.forwardKinematics(self.model, self.fk_data, q_model)
        self.pin.updateFramePlacements(self.model, self.fk_data)
        pose = self.fk_data.oMf[self.ee_frame_id]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = pose.rotation
        transform[:3, 3] = pose.translation
        return transform

    def solve_tcp_pose(
        self,
        tcp_to_base: np.ndarray,
        q_seed: np.ndarray,
    ) -> tuple[Optional[np.ndarray], float]:
        target = validate_transform(tcp_to_base, "T_tcp_base_target")
        result = self.solve_ik_fn(
            self.model,
            self.ik_data,
            self.ee_frame_id,
            self.pin.SE3(target[:3, :3], target[:3, 3]),
            np.asarray(q_seed, dtype=np.float64),
            self.ik_params_type(
                # SDK 求解器达到 tolerance 后仍会跑满 max_iter；视觉目标每次
                # 只前进几毫米/不到一度，30 次已足够，避免 500 次 Python
                # 循环把 500 Hz MIT 线程拖到约 100 Hz。
                max_iter=30,
                tolerance=1e-4,
                step_size=0.45,
                damping=1e-5,
            ),
            controlled_joints=6,
        )
        if not result.success:
            return None, float(result.error)
        q_goal = np.asarray(result.q[:6], dtype=np.float64)
        lower = self.model.lowerPositionLimit[:6]
        upper = self.model.upperPositionLimit[:6]
        if np.any((q_goal < lower) | (q_goal > upper)):
            return None, float(result.error)
        return q_goal, float(result.error)

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise self.error

    def __call__(self, _robot: Any, dt: float) -> None:
        if self.stop_event.is_set():
            return
        try:
            # 先触发本周期反馈，再直接检查六个电机状态。JointGroup 的便捷
            # getter 会把缺失状态填成 0；真机控制不能把“掉线”误当成“零位”。
            self.robot.arm.get_positions(request_feedback=True)
            states = [
                self.robot._motor_map[name].get_state()
                for name in self.robot.arm.joint_names
            ]
            if any(state is None for state in states):
                missing = [
                    name
                    for name, state in zip(self.robot.arm.joint_names, states)
                    if state is None
                ]
                raise RuntimeError(f"控制周期缺少电机反馈: {missing}")
            q_raw = np.array([state.pos for state in states], dtype=np.float64)
            qd = np.array([state.vel for state in states], dtype=np.float64)
            with self.state_lock:
                previous = None if self.latest_q is None else self.latest_q.copy()
            q = q_raw if previous is None else self._angles_near_reference(q_raw, previous)
            self._check_feedback(q, qd)
            with self.state_lock:
                self.latest_q = q.copy()
                self.latest_qd = qd.copy()

            with self.command_lock:
                mode = self.mode
                if mode == MODE_DRAG:
                    q_reference = q.copy()
                    qd_reference = np.zeros(6, dtype=np.float64)
                    self.q_reference = q_reference.copy()
                    self.qd_reference = qd_reference.copy()
                    kp, kd = DRAG_KP, DRAG_KD
                else:
                    if self.q_goal is None or self.q_reference is None:
                        raise RuntimeError("视觉跟随关节目标未初始化")
                    q_reference, qd_reference = next_joint_reference(
                        self.q_reference,
                        self.qd_reference,
                        self.q_goal,
                        dt,
                    )
                    q_reference = np.clip(
                        q_reference,
                        self.model.lowerPositionLimit[:6],
                        self.model.upperPositionLimit[:6],
                    )
                    self.q_reference = q_reference.copy()
                    self.qd_reference = qd_reference.copy()
                    kp, kd = TRACK_KP, TRACK_KD

            gravity = self._gravity(q)
            q_command, qd_command, kd_effective, tau_total = compute_mit_command(
                q,
                qd,
                q_reference,
                qd_reference,
                gravity,
                kp,
                kd,
            )
            self._send(q_command, qd_command, kp, kd_effective, gravity)

            now = time.monotonic()
            torque_utilization = float(
                np.max(np.abs(tau_total) / MOTOR_PEAK_TORQUES_NM)
            )
            is_stalled = (
                mode == MODE_TRACK
                and np.max(np.abs(q_reference - q)) > STALL_POSITION_ERROR_RAD
                and np.max(np.abs(qd)) < STALL_SPEED_RAD_S
                and torque_utilization > STALL_TORQUE_UTILIZATION
            )
            if is_stalled:
                if self.stall_started is None:
                    self.stall_started = now
                elif now - self.stall_started >= STALL_TIME_S:
                    raise RuntimeError(
                        "MIT 堵转保护触发: "
                        f"|q_ref-q|max={np.max(np.abs(q_reference-q)):.4f}rad, "
                        f"|qd|max={np.max(np.abs(qd)):.4f}rad/s, "
                        f"peak_util={torque_utilization:.1%}"
                    )
            else:
                self.stall_started = None

            self.log_cycles += 1
            if self.last_log == 0.0:
                self.last_log = now
                self.log_cycles = 0
            elif now - self.last_log >= CONTROL_LOG_INTERVAL_S:
                rate = self.log_cycles / (now - self.last_log)
                self.last_log = now
                self.log_cycles = 0
                print(
                    f"[Control] mode={mode} rate={rate:.1f}Hz "
                    f"|q_ref-q|max={np.max(np.abs(q_reference-q)):.4f}rad "
                    f"|qd_ref|max={np.max(np.abs(qd_reference)):.3f}rad/s "
                    f"|qd|max={np.max(np.abs(qd)):.3f}rad/s "
                    f"tau_g(2,3)=({gravity[1]:+.2f},{gravity[2]:+.2f})N*m "
                    f"peak_util={torque_utilization:.1%}"
                )
        except Exception as exc:
            if self.error is None:
                self.error = exc
            self.stop_event.set()
            try:
                self.robot.disable_all()
            except Exception:
                pass

    def return_home(self) -> None:
        """保持 MIT 模式回 home；无需第二次 Ctrl+C。"""
        if not self.running:
            return
        print(f"[Motion] return_home MIT target={HOME_JOINTS_RAD.tolist()}")
        self.set_joint_goal(HOME_JOINTS_RAD)
        deadline = time.monotonic() + HOME_TIMEOUT_S
        stable_since: Optional[float] = None
        while time.monotonic() < deadline:
            self.raise_if_failed()
            q, qd = self.state()
            reached = (
                np.max(np.abs(HOME_JOINTS_RAD - q)) <= HOME_POSITION_TOLERANCE_RAD
                and np.max(np.abs(qd)) <= HOME_SPEED_TOLERANCE_RAD_S
            )
            if reached:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= HOME_STABLE_TIME_S:
                    print(
                        "[Motion] return_home reached: "
                        f"max|q|={np.max(np.abs(q)):.4f}rad"
                    )
                    return
            else:
                stable_since = None
            time.sleep(0.02)
        q, qd = self.state()
        raise RuntimeError(
            "return_home 到位超时: "
            f"max|q|={np.max(np.abs(q)):.4f}rad, "
            f"max|qd|={np.max(np.abs(qd)):.4f}rad/s"
        )

    def shutdown(self) -> None:
        self.stop_event.set()
        errors: list[str] = []
        if self.connected:
            for name, action in (
                ("stop_control_loop", self.robot.stop_control_loop),
                ("disable_all", self.robot.disable_all),
                ("disconnect", self.robot.disconnect),
            ):
                try:
                    action()
                except Exception as exc:
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
        self.running = False
        self.connected = False
        if errors:
            print("[Hardware] cleanup errors: " + "; ".join(errors), file=sys.stderr)
        else:
            print("[Hardware] Arm disabled and disconnected.")


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("配置根节点必须是 mapping")
    return config


def load_eye_in_hand_transform(config: dict[str, Any]) -> tuple[np.ndarray, Path]:
    """读取 camera -> TCP 手眼矩阵。"""
    camera_type = str((config.get("camera") or {}).get("type", "")).lower()
    if not camera_type:
        raise ValueError("camera.type 未配置")
    path = PROJECT_ROOT / "config" / "calibration" / camera_type / "hand_eye.npz"
    if not path.is_file():
        raise FileNotFoundError(f"缺少手眼标定: {path}")
    with np.load(str(path), allow_pickle=False) as data:
        if "T_result" not in data or "mode" not in data:
            raise ValueError(f"手眼标定缺少 T_result/mode: {path}")
        mode = str(np.asarray(data["mode"]).reshape(-1)[0]).lower()
        if mode != "eye_in_hand":
            raise ValueError(f"需要 eye_in_hand 标定，实际为 {mode}")
        transform = validate_transform(data["T_result"], "T_camera_tcp")
    return transform, path


def draw_overlay(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    detection: Optional[TargetDetection],
    mode: str,
    status: str,
    missing_frames: int,
    camera_fps: float = 0.0,
    aruco_fps: float = 0.0,
    aruco_valid_rate: float = 0.0,
) -> np.ndarray:
    """显示检测框、光心、距离误差和控制状态。"""
    view = image.copy()
    K = np.asarray(camera_matrix, dtype=np.float64)
    center = (int(round(K[0, 2])), int(round(K[1, 2])))
    cv2.drawMarker(
        view,
        center,
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )
    if detection is not None:
        cv2.polylines(view, [detection.corners_px], True, (0, 255, 0), 2)
        cv2.circle(view, detection.center_px, 5, (0, 0, 255), -1)
        cv2.arrowedLine(view, center, detection.center_px, (0, 0, 255), 2)
        position = detection.position_camera
        du = detection.center_px[0] - center[0]
        dv = detection.center_px[1] - center[1]
        lines = [
            (
                f"aruco:{ARUCO_MARKER_ID} {detection.source} "
                f"xyz=({position[0]:+.3f},{position[1]:+.3f},{position[2]:.3f})m"
            ),
            (
                f"center_error=({du:+d},{dv:+d})px "
                f"depth_error={position[2]-FOLLOW_DISTANCE_M:+.3f}m"
            ),
        ]
    else:
        lines = ["aruco not detected", "center/depth error unavailable"]
    lines.extend(
        [
            f"mode={mode} missing={missing_frames} {status}",
            (
                f"camera={camera_fps:.1f}fps aruco={aruco_fps:.1f}fps "
                f"valid={aruco_valid_rate:.0%}"
            ),
            (
                f"distance={FOLLOW_DISTANCE_M:.3f}m "
                f"speed={FOLLOW_SPEED_M_S:.3f}m/s | Ctrl+C: home and exit"
            ),
        ]
    )
    for index, text in enumerate(lines):
        cv2.putText(
            view,
            text,
            (10, 28 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255) if index >= 2 else (0, 255, 0),
            2,
        )
    return view


def report_exception(stage: str, exc: Exception) -> None:
    print(f"[Fatal] stage={stage}: {type(exc).__name__}: {exc}", file=sys.stderr)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def main() -> int:
    if len(sys.argv) != 1:
        print("本脚本不接受命令行参数，请修改文件顶部两个参数。", file=sys.stderr)
        return 2

    stage = "load configuration"
    try:
        validate_user_config()
        config = load_yaml_config(CONFIG_PATH)
        camera_to_tcp, hand_eye_path = load_eye_in_hand_transform(config)
        tracker = ArucoTracker()
        camera = make_camera(config)
    except Exception as exc:
        report_exception(stage, exc)
        return 2

    print(
        "[Servo] ArUco center -> fixed-orientation TCP translation -> IK -> MIT"
    )
    print(
        f"[Servo] ID={ARUCO_MARKER_ID}, edge={ARUCO_MARKER_LENGTH_M*1000:.0f}mm, "
        f"distance={FOLLOW_DISTANCE_M:.3f}m, speed={FOLLOW_SPEED_M_S:.3f}m/s"
    )
    print(f"[Calibration] camera->TCP: {hand_eye_path}")
    print(
        "[Safety] Confirm clear workspace, correct /dev/ttyACM*, rigid camera "
        "mount and working emergency stop. No environment collision checking."
    )

    exit_event = threading.Event()
    ctrl_c_requested = False

    def on_ctrl_c(_signum=None, _frame=None) -> None:
        nonlocal ctrl_c_requested
        ctrl_c_requested = True
        if not exit_event.is_set():
            print("\n[Exit] Ctrl+C: stop vision, return home once, disable and exit.")
        exit_event.set()

    def on_terminate(_signum=None, _frame=None) -> None:
        exit_event.set()

    signal.signal(signal.SIGINT, on_ctrl_c)
    signal.signal(signal.SIGTERM, on_terminate)

    robot: Optional[RobotVisualServo] = None
    camera_open = False
    window_open = False
    exit_code = 0
    tracking = False
    missing_frames = 0
    last_vision_log = 0.0
    marker_filter = MarkerPoseFilter()
    pose_limiter = PoseCommandLimiter()
    fixed_tcp_pose: Optional[np.ndarray] = None
    acquire_stable_since: Optional[float] = None
    status = "gravity drag; waiting for ArUco"
    last_preview: Optional[np.ndarray] = None

    # 这里统计的是视觉主循环的真实吞吐量，不是相机配置中声明的 30 FPS：
    # camera_fps = 进入 ArUco 检测的彩色帧数 / 时间
    # aruco_fps  = ID、面积、PnP 等检查全部通过的帧数 / 时间
    # valid_rate = 有效 ArUco 帧数 / 实际彩色帧数
    rate_window_started = time.monotonic()
    camera_frames_in_window = 0
    aruco_frames_in_window = 0
    actual_camera_fps = 0.0
    actual_aruco_fps = 0.0
    aruco_valid_rate = 0.0

    try:
        stage = "start MIT gravity compensation"
        robot = RobotVisualServo()
        robot.start()

        stage = "open camera"
        camera.open()
        camera_open = True
        camera.warm_up(CAMERA_WARMUP_FRAMES)
        print(
            "[Camera] ready: "
            f"K=({camera.K[0,0]:.1f},{camera.K[1,1]:.1f},"
            f"{camera.K[0,2]:.1f},{camera.K[1,2]:.1f})"
        )
        cv2.namedWindow("reBot ArUco Visual Servo", cv2.WINDOW_AUTOSIZE)
        window_open = True
        print(
            "[Servo] First drag the gripper to a horizontal pose and stop. "
            "Three valid ArUco frames lock that TCP orientation and start "
            "automatic center/distance tracking; no key press is required."
        )

        stage = "visual servo loop"
        while not exit_event.is_set():
            robot.raise_if_failed()
            detection: Optional[TargetDetection] = None
            color_bgr: Optional[np.ndarray] = None
            try:
                color_bgr, depth_mm = camera.get_frame()
                if color_bgr is not None:
                    camera_frames_in_window += 1
                    last_preview = color_bgr.copy()
                    detection = tracker.detect(
                        color_bgr,
                        depth_mm,
                        camera.K,
                        camera.D,
                    )
                    if detection is not None:
                        aruco_frames_in_window += 1
            except CameraFrameError as exc:
                tracker.status = f"camera_error:{exc}"

            now = time.monotonic()
            rate_elapsed = now - rate_window_started
            if rate_elapsed >= VISION_RATE_WINDOW_S:
                actual_camera_fps = camera_frames_in_window / rate_elapsed
                actual_aruco_fps = aruco_frames_in_window / rate_elapsed
                aruco_valid_rate = (
                    aruco_frames_in_window / camera_frames_in_window
                    if camera_frames_in_window > 0
                    else 0.0
                )
                rate_window_started = now
                camera_frames_in_window = 0
                aruco_frames_in_window = 0

            current_tcp = robot.current_tcp_pose()
            if detection is not None:
                marker_to_base = (
                    current_tcp @ camera_to_tcp @ detection.marker_to_camera
                )
                marker_filter.update(marker_to_base, now)
                missing_frames = 0
            else:
                missing_frames += 1

            marker_pose = marker_filter.current(now)
            observation_age = marker_filter.age(now)
            target_lost = tracking and (
                observation_age is None
                or observation_age > TARGET_LOST_TIMEOUT_S
                or (
                    missing_frames >= TARGET_LOST_FRAMES
                    and observation_age >= TARGET_LOST_MIN_TIME_S
                )
            )

            if target_lost:
                robot.enter_gravity_drag()
                tracking = False
                marker_filter.reset()
                pose_limiter.clear()
                fixed_tcp_pose = None
                acquire_stable_since = None
                status = "target lost; gravity drag resumed"
                print("[Servo] Target lost: gravity compensation drag resumed.")
            elif marker_pose is not None:
                _, current_qd = robot.state()
                ready_to_track = tracking
                if not tracking and detection is None:
                    acquire_stable_since = None
                    status = "confirmed target; waiting for a fresh frame"
                elif not tracking and (
                    np.max(np.abs(current_qd)) > ACQUIRE_MAX_JOINT_SPEED_RAD_S
                ):
                    acquire_stable_since = None
                    status = (
                        "target ready; waiting for arm to stop "
                        f"qd={np.max(np.abs(current_qd)):.3f}rad/s"
                    )
                elif not tracking:
                    if acquire_stable_since is None:
                        acquire_stable_since = now
                    ready_to_track = (
                        now - acquire_stable_since >= ACQUIRE_STABLE_TIME_S
                    )
                    status = "target ready; confirming arm is stationary"

                if ready_to_track:
                    # 接管瞬间锁定 TCP 姿态。之后 ArUco 的任何旋转都不会生成
                    # 末端旋转命令；用户应在此之前把夹爪手动拖到水平姿态。
                    if fixed_tcp_pose is None:
                        fixed_tcp_pose = current_tcp.copy()
                    raw_tcp_goal = compute_desired_tcp_pose(
                        marker_pose,
                        camera_to_tcp,
                        fixed_tcp_pose,
                    )
                    if not tracking:
                        pose_limiter.reset(current_tcp, now - 1.0 / 30.0)
                    last_accepted_pose = pose_limiter.pose.copy()
                    limited_tcp_goal = pose_limiter.step(raw_tcp_goal, now)
                    q_goal, ik_error = robot.solve_tcp_pose(
                        limited_tcp_goal,
                        robot.goal_seed(),
                    )
                    if q_goal is not None:
                        robot.set_joint_goal(q_goal)
                        if not tracking:
                            print(
                                "[Servo] Target acquired: TCP orientation locked; "
                                "automatic center/distance tracking started."
                            )
                        tracking = True
                        acquire_stable_since = None
                        position_error = float(
                            np.linalg.norm(
                                raw_tcp_goal[:3, 3] - current_tcp[:3, 3]
                            )
                        )
                        status = (
                            f"following position_error={position_error:.3f}m "
                            f"orientation=locked ik={ik_error:.1e}"
                        )
                    else:
                        # 不可达帧不能继续推进 Cartesian 限速器，否则它会在
                        # 后台越走越远，后续所有 IK 都从不可达目标开始。
                        pose_limiter.reset(last_accepted_pose, now)
                        if not tracking:
                            # 尚未真正接管时允许用户继续拖动；下一次尝试应锁定
                            # 新的实际姿态，不能沿用一次失败尝试的旧姿态。
                            fixed_tcp_pose = None
                        status = (
                            f"IK rejected error={ik_error:.3e}; holding last goal"
                        )
            elif detection is not None:
                acquire_stable_since = None
                status = (
                    f"confirming {marker_filter.confirmed_frames}/"
                    f"{TARGET_CONFIRM_FRAMES}"
                )
            elif tracking:
                status = "brief target dropout; holding last smooth goal"
            else:
                acquire_stable_since = None
                status = "gravity drag; waiting for ArUco"

            mode = MODE_TRACK if tracking else MODE_DRAG
            if now - last_vision_log >= VISION_LOG_INTERVAL_S:
                last_vision_log = now
                raw = (
                    np.array2string(detection.position_camera, precision=3)
                    if detection is not None
                    else "none"
                )
                print(
                    f"[Vision] camera={actual_camera_fps:.1f}fps "
                    f"aruco={actual_aruco_fps:.1f}fps "
                    f"valid={aruco_valid_rate:.0%} "
                    f"detector={tracker.status} raw={raw} "
                    f"missing={missing_frames} mode={mode} status={status}"
                )

            preview_source = color_bgr if color_bgr is not None else last_preview
            if preview_source is not None:
                preview = draw_overlay(
                    preview_source,
                    camera.K,
                    detection,
                    mode,
                    status,
                    missing_frames,
                    actual_camera_fps,
                    actual_aruco_fps,
                    aruco_valid_rate,
                )
                cv2.imshow("reBot ArUco Visual Servo", preview)
                cv2.waitKey(1)
    except Exception as exc:
        report_exception(stage, exc)
        exit_code = 4
    finally:
        if camera_open:
            try:
                camera.close()
            except Exception as exc:
                print(f"[Camera] close failed: {exc}", file=sys.stderr)
            camera_open = False

        if robot is not None:
            if (
                ctrl_c_requested
                and robot.running
                and robot.error is None
            ):
                try:
                    stage = "return home"
                    robot.return_home()
                except Exception as exc:
                    report_exception(stage, exc)
                    exit_code = 5
            robot.shutdown()
        if window_open:
            cv2.destroyAllWindows()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
