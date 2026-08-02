from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import visual_servo_control as servo  # noqa: E402


def make_transform(
    translation: tuple[float, float, float],
    rotation_vector_value: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = servo.rotation_from_vector(
        np.asarray(rotation_vector_value, dtype=np.float64)
    )
    transform[:3, 3] = translation
    return transform


def test_user_configuration_has_only_expected_ranges(monkeypatch) -> None:
    servo.validate_user_config()
    assert 0.20 <= servo.FOLLOW_DISTANCE_M <= 1.00
    assert 0.001 <= servo.FOLLOW_SPEED_M_S <= 0.10

    monkeypatch.setattr(servo, "FOLLOW_SPEED_M_S", 0.5)
    with pytest.raises(ValueError, match="FOLLOW_SPEED_M_S"):
        servo.validate_user_config()


def test_desired_marker_pose_means_center_distance_and_perpendicular() -> None:
    desired = servo.desired_marker_to_camera(0.40)
    np.testing.assert_array_equal(desired[:3, 3], [0.0, 0.0, 0.40])
    np.testing.assert_array_equal(desired[:3, :3], np.diag([1.0, -1.0, -1.0]))
    assert np.linalg.det(desired[:3, :3]) == pytest.approx(1.0)


def test_pose_chain_returns_current_tcp_when_marker_is_already_ideal() -> None:
    """目标已经居中、定距、正对时，几何链不得产生任何运动。"""
    tcp_to_base = make_transform((0.32, -0.08, 0.27), (0.2, -0.1, 0.3))
    camera_to_tcp = make_transform((-0.07, 0.02, 0.055), (-0.4, 0.2, -0.1))
    marker_to_camera = servo.desired_marker_to_camera(0.30)
    marker_to_base = tcp_to_base @ camera_to_tcp @ marker_to_camera

    recovered = servo.compute_desired_tcp_pose(
        marker_to_base,
        camera_to_tcp,
        tcp_to_base,
        0.30,
    )

    np.testing.assert_allclose(recovered, tcp_to_base, atol=1e-12)


def test_pose_chain_moves_camera_toward_right_and_far_target_without_sign_hack() -> None:
    """相机和 TCP 重合时，右侧/远处目标应直接得到 +X/+Z TCP 位移。"""
    current_tcp = np.eye(4)
    camera_to_tcp = np.eye(4)
    observed = servo.desired_marker_to_camera(0.30)
    observed[0, 3] = 0.05
    observed[2, 3] = 0.40
    marker_to_base = current_tcp @ camera_to_tcp @ observed

    desired_tcp = servo.compute_desired_tcp_pose(
        marker_to_base,
        camera_to_tcp,
        current_tcp,
        0.30,
    )

    np.testing.assert_allclose(desired_tcp[:3, 3], [0.05, 0.0, 0.10], atol=1e-12)


def test_in_plane_marker_rotation_does_not_roll_the_tcp() -> None:
    """纸面内旋转 ArUco 不改变平面法向，不应命令末端滚转。"""
    current_tcp = make_transform((0.30, 0.0, 0.25), (0.1, -0.2, 0.3))
    camera_to_tcp = make_transform((-0.07, 0.02, 0.05), (-0.3, 0.1, 0.2))
    marker_camera = servo.desired_marker_to_camera(0.30)
    marker_camera[:3, :3] = (
        marker_camera[:3, :3]
        @ servo.rotation_from_vector(np.array([0.0, 0.0, np.pi]))
    )
    marker_base = current_tcp @ camera_to_tcp @ marker_camera

    desired_tcp = servo.compute_desired_tcp_pose(
        marker_base,
        camera_to_tcp,
        current_tcp,
        0.30,
    )

    np.testing.assert_allclose(desired_tcp, current_tcp, atol=1e-12)


def test_marker_plane_tilt_does_not_rotate_the_tcp() -> None:
    """标记平面发生倾斜时，夹爪姿态仍必须等于接管时的固定姿态。"""
    fixed_tcp = make_transform((0.30, 0.0, 0.25), (0.1, -0.2, 0.3))
    camera_to_tcp = make_transform((-0.07, 0.02, 0.05), (-0.3, 0.1, 0.2))
    marker_camera = servo.desired_marker_to_camera(0.35)
    marker_camera[:3, :3] = (
        marker_camera[:3, :3]
        @ servo.rotation_from_vector(np.array([0.4, -0.3, 0.2]))
    )
    marker_base = fixed_tcp @ camera_to_tcp @ marker_camera

    desired_tcp = servo.compute_desired_tcp_pose(
        marker_base,
        camera_to_tcp,
        fixed_tcp,
        0.30,
    )

    np.testing.assert_allclose(desired_tcp[:3, :3], fixed_tcp[:3, :3], atol=1e-12)


def test_pose_limiter_obeys_speed_and_keeps_locked_orientation(monkeypatch) -> None:
    monkeypatch.setattr(servo, "FOLLOW_SPEED_M_S", 0.05)
    limiter = servo.PoseCommandLimiter()
    limiter.reset(np.eye(4), 1.0)
    desired = make_transform((0.20, 0.0, 0.0), (0.0, 0.0, 1.0))

    first = limiter.step(desired, 1.02)
    second = limiter.step(desired, 1.04)

    assert np.linalg.norm(first[:3, 3]) <= 0.05 * 0.02 + 1e-12
    assert np.linalg.norm(second[:3, 3] - first[:3, 3]) <= 0.05 * 0.02 + 1e-12
    np.testing.assert_array_equal(first[:3, :3], np.eye(3))
    np.testing.assert_array_equal(second[:3, :3], np.eye(3))


def test_marker_filter_works_in_base_frame_and_requires_three_frames() -> None:
    target_filter = servo.MarkerPoseFilter()
    marker = make_transform((0.44, 0.00, 0.05), (0.0, 0.0, 0.1))

    assert target_filter.update(marker, 0.00) is None
    assert target_filter.update(marker, 0.03) is None
    filtered = target_filter.update(marker, 0.06)
    np.testing.assert_allclose(filtered, marker, atol=1e-12)
    assert target_filter.current(0.20) is not None
    assert target_filter.current(0.57) is None


def test_joint_reference_limits_speed_and_acceleration_without_reference_lead_cap() -> None:
    q = np.zeros(6)
    qd = np.zeros(6)
    goal = np.array([0.50, -0.50, 0.40, -0.30, 0.20, -0.10])
    dt = 0.002

    for _ in range(2000):
        previous_qd = qd.copy()
        q, qd = servo.next_joint_reference(q, qd, goal, dt)
        assert np.max(np.abs(qd)) <= servo.MAX_JOINT_REFERENCE_SPEED_RAD_S + 1e-12
        assert np.max(np.abs(qd - previous_qd)) <= (
            servo.MAX_JOINT_REFERENCE_ACCELERATION_RAD_S2 * dt + 1e-12
        )
    # 关节参考可以完整走到目标；不存在旧版 0.05/0.08 rad 领先量截断。
    np.testing.assert_allclose(q, goal, atol=2e-3)


def test_mit_command_uses_motor_peak_not_nine_nm_as_motion_cap() -> None:
    q = np.zeros(6)
    qd = np.zeros(6)
    q_reference = np.array([0.20, -0.20, 0.20, -0.20, 0.20, -0.20])
    gravity = np.array([0.0, -1.0, -7.0, -1.0, 0.0, 0.0])

    q_command, _, _, tau_total = servo.compute_mit_command(
        q,
        qd,
        q_reference,
        np.zeros(6),
        gravity,
        servo.TRACK_KP,
        servo.TRACK_KD,
    )

    assert np.max(np.abs(tau_total[:3])) > 9.0
    assert np.all(np.abs(tau_total) <= servo.MOTOR_PEAK_TORQUES_NM + 1e-12)
    assert np.max(np.abs(q_command[:3] - q[:3])) > 0.08


def test_gravity_drag_is_pos_equal_current_q_plus_gravity() -> None:
    q = np.array([0.1, -0.2, -0.3, 0.1, 0.0, -0.1])
    qd = np.zeros(6)
    gravity = np.array([0.0, -1.2, -7.1, -1.8, 0.0, 0.0])

    q_command, qd_command, kd, tau_total = servo.compute_mit_command(
        q,
        qd,
        q,
        np.zeros(6),
        gravity,
        servo.DRAG_KP,
        servo.DRAG_KD,
    )

    np.testing.assert_allclose(q_command, q)
    np.testing.assert_array_equal(qd_command, np.zeros(6))
    np.testing.assert_array_equal(kd, servo.DRAG_KD)
    np.testing.assert_allclose(tau_total, gravity)


def test_robot_gravity_applies_configured_joint_scales(monkeypatch) -> None:
    """模型重力必须逐轴乘上当前真机调试比例。"""
    robot = servo.RobotVisualServo()
    gravity_model = np.array([0.0, -2.0, -7.0, -1.0, 0.1, -0.1])
    monkeypatch.setattr(
        robot,
        "compute_gravity",
        lambda **_kwargs: gravity_model.copy(),
    )

    gravity_command = robot._gravity(np.zeros(6))

    np.testing.assert_allclose(
        gravity_command,
        gravity_model * servo.GRAVITY_TORQUE_SCALE,
    )


def test_actual_model_ik_accepts_pose_generated_by_hand_eye_chain() -> None:
    """用项目真实 URDF 和手眼矩阵回验整条位姿链，但不连接硬件。"""
    robot = servo.RobotVisualServo()
    q = np.array([-0.0001, -0.5251, -0.4269, -0.0982, 0.0, 0.0])
    with robot.state_lock:
        robot.latest_q = q.copy()
        robot.latest_qd = np.zeros(6)
    tcp_to_base = robot.current_tcp_pose()
    config = servo.load_yaml_config(servo.CONFIG_PATH)
    camera_to_tcp, _ = servo.load_eye_in_hand_transform(config)
    marker_to_base = (
        tcp_to_base
        @ camera_to_tcp
        @ servo.desired_marker_to_camera(servo.FOLLOW_DISTANCE_M)
    )
    tcp_goal = servo.compute_desired_tcp_pose(
        marker_to_base,
        camera_to_tcp,
        tcp_to_base,
    )

    q_goal, error = robot.solve_tcp_pose(tcp_goal, q)

    assert q_goal is not None
    assert error < 1e-4
    np.testing.assert_allclose(q_goal, q, atol=2e-3)


@pytest.mark.skipif(
    not hasattr(cv2, "aruco"),
    reason="OpenCV was built without aruco",
)
def test_synthetic_aruco_returns_full_front_facing_pose() -> None:
    tracker = servo.ArucoTracker()
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(
            tracker.dictionary,
            servo.ARUCO_MARKER_ID,
            200,
        )
    else:
        marker = cv2.aruco.drawMarker(
            tracker.dictionary,
            servo.ARUCO_MARKER_ID,
            200,
        )
    image[140:340, 220:420] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    K = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    )

    detection = tracker.detect(image, None, K, np.zeros(5))

    assert detection is not None
    assert detection.center_px == pytest.approx((320, 240), abs=1)
    assert detection.position_camera[2] == pytest.approx(0.30, abs=0.01)
    assert np.dot(
        detection.marker_to_camera[:3, 2],
        detection.marker_to_camera[:3, 3],
    ) < 0.0
    angle_error = np.linalg.norm(
        servo.rotation_vector(
            servo.desired_marker_to_camera()[:3, :3].T
            @ detection.marker_to_camera[:3, :3]
        )
    )
    assert angle_error < np.deg2rad(1.0)


@pytest.mark.skipif(
    not hasattr(cv2, "aruco"),
    reason="OpenCV was built without aruco",
)
def test_synthetic_aruco_uses_aligned_depth_for_translation() -> None:
    tracker = servo.ArucoTracker()
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(
            tracker.dictionary,
            servo.ARUCO_MARKER_ID,
            200,
        )
    else:
        marker = cv2.aruco.drawMarker(
            tracker.dictionary,
            servo.ARUCO_MARKER_ID,
            200,
        )
    image[140:340, 220:420] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[140:340, 220:420] = 425
    K = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    )

    detection = tracker.detect(image, depth, K, np.zeros(5))

    assert detection is not None
    assert detection.source == "rgbd"
    np.testing.assert_allclose(
        detection.position_camera,
        [0.0, 0.0, 0.425],
        atol=1e-3,
    )


def test_initial_feedback_performs_per_motor_handshake(monkeypatch) -> None:
    class FakeMotor:
        def __init__(self, position: float) -> None:
            self.position = position
            self.requests = 0

        def request_feedback(self) -> None:
            self.requests += 1

        def get_state(self):
            if self.requests < 2:
                return None
            return SimpleNamespace(pos=self.position, vel=0.0)

    class FakeController:
        def poll_feedback_once(self) -> None:
            pass

    names = [f"joint{index}" for index in range(1, 7)]
    motors = {name: FakeMotor(float(index)) for index, name in enumerate(names)}
    arm = SimpleNamespace(
        joint_names=names,
        _jcfgs=[SimpleNamespace(name=name, vendor="damiao") for name in names],
    )
    robot = SimpleNamespace(
        arm=arm,
        _motor_map=motors,
        _ctrl_map={"damiao": FakeController()},
    )
    monkeypatch.setattr(servo.time, "sleep", lambda _seconds: None)

    q, qd = servo.read_initial_arm_state(robot)

    np.testing.assert_array_equal(q, np.arange(6, dtype=np.float64))
    np.testing.assert_array_equal(qd, np.zeros(6))


def test_overlay_marks_calibrated_optical_center() -> None:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    K = np.array([[359.0, 0.0, 320.0], [0.0, 359.0, 178.0], [0.0, 0.0, 1.0]])
    output = servo.draw_overlay(
        image,
        K,
        detection=None,
        mode=servo.MODE_DRAG,
        status="waiting",
        missing_frames=0,
    )
    np.testing.assert_array_equal(output[178, 320], [0, 255, 255])
