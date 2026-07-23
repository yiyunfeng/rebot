from unittest.mock import patch

import numpy as np
import pytest

from lerobot_robot_rebot import JOINT_NAMES, ReBotB601, ReBotB601Config


class FakeGroup:
    def __init__(self) -> None:
        self.mode: str | None = None
        self.commands: list[tuple[str, tuple, dict]] = []

    def mode_mit(self) -> None:
        self.mode = "mit"
        return self.mode_ok

    def mode_pos_vel(self, velocity_limits) -> None:
        self.mode = "pos_vel"
        self.velocity_limits = np.asarray(velocity_limits)
        return self.mode_ok

    def send_mit(self, *args, **kwargs) -> None:
        self.commands.append(("mit", args, kwargs))

    def send_pos_vel(self, *args, **kwargs) -> None:
        self.commands.append(("pos_vel", args, kwargs))


class FakeArm:
    instances: list["FakeArm"] = []
    fail_mode_switch = False
    initial_state = np.array([2.79, -1.0, -1.0, 0.0, 0.0, 0.0, -2.0], dtype=np.float64)

    def __init__(self, hw_yaml=None) -> None:
        self.hw_yaml = hw_yaml
        self.arm = FakeGroup()
        self.gripper = FakeGroup()
        self.arm.mode_ok = not self.fail_mode_switch
        self.gripper.mode_ok = not self.fail_mode_switch
        self.state = self.initial_state.copy()
        self.connected = False
        self.enabled = False
        self.loop_active = False
        self.estopped = False
        self.control_fn = None
        FakeArm.instances.append(self)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.enabled = False

    def get_state(self, request_feedback=True):
        zeros = np.zeros(7, dtype=np.float64)
        return self.state.copy(), zeros, zeros

    def enable_all(self) -> None:
        self.enabled = True

    def start_control_loop(self, control_fn, rate) -> None:
        self.control_fn = control_fn
        self.rate = rate
        self.loop_active = True

    def stop_control_loop(self) -> None:
        self.loop_active = False

    def estop(self) -> None:
        self.estopped = True
        self.enabled = False


class FakeCamera:
    def __init__(self, config) -> None:
        self.config = config
        self.is_connected = False
        self.depth_mm = np.full((config.height, config.width), 500, dtype=np.uint16)

    def connect(self) -> None:
        self.is_connected = True

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        shape = (self.config.height, self.config.width, 3)
        return np.zeros(shape, dtype=np.uint8), np.ones(shape, dtype=np.uint8)

    def read_depth_mm(self) -> np.ndarray:
        return self.depth_mm.copy()

    def disconnect(self) -> None:
        self.is_connected = False


@pytest.fixture(autouse=True)
def fake_hardware():
    FakeArm.instances.clear()
    FakeArm.fail_mode_switch = False
    FakeArm.initial_state = np.array(
        [2.79, -1.0, -1.0, 0.0, 0.0, 0.0, -2.0], dtype=np.float64
    )

    def gravity(q):
        return np.arange(1, 7, dtype=np.float64)

    with (
        patch("lerobot_robot_rebot.robot._load_sdk", return_value=(FakeArm, gravity)),
        patch("lerobot_robot_rebot.robot.ReBotRGBDCamera", FakeCamera),
    ):
        yield


def make_robot(tmp_path, mode: str) -> ReBotB601:
    config = ReBotB601Config(
        id=f"test-{mode}",
        calibration_dir=tmp_path,
        operating_mode=mode,
    )
    return ReBotB601(config)


def test_readonly_connect_does_not_enable_motors(tmp_path) -> None:
    robot = make_robot(tmp_path, "readonly")
    robot.connect()

    arm = FakeArm.instances[-1]
    assert robot.is_connected
    assert not arm.enabled
    assert not arm.loop_active
    observation = robot.get_observation()
    assert observation["joint1.pos"] == pytest.approx(2.79)
    assert observation["main_rgb"].shape == (360, 640, 3)
    assert observation["main_depth"].shape == (360, 640, 3)
    np.testing.assert_array_equal(robot.get_depth_mm(), robot.camera.depth_mm)

    robot.disconnect()
    assert not robot.is_connected


def test_deploy_clips_relative_and_absolute_limits(tmp_path) -> None:
    robot = make_robot(tmp_path, "deploy")
    robot.connect()
    arm = FakeArm.instances[-1]

    assert arm.arm.mode == "pos_vel"
    assert arm.gripper.mode == "mit"
    action = {f"{name}.pos": 10.0 for name in JOINT_NAMES}
    sent = robot.send_action(action)

    assert sent["joint1.pos"] == pytest.approx(2.8)
    assert sent["joint2.pos"] == pytest.approx(-0.92)
    assert sent["gripper.pos"] == pytest.approx(0.0)
    arm.control_fn(arm, 1 / arm.rate)
    assert arm.arm.commands[-1][0] == "pos_vel"
    assert arm.gripper.commands[-1][0] == "mit"
    assert arm.gripper.commands[-1][2]["kp"] == [0.0]
    assert arm.gripper.commands[-1][2]["tau"] == [robot.config.gripper_close_torque]
    robot.disconnect()


def test_deploy_policy_opens_gripper_without_arm_step_limit(tmp_path) -> None:
    robot = make_robot(tmp_path, "deploy")
    robot.connect()
    arm = FakeArm.instances[-1]
    action = {
        f"{name}.pos": float(value)
        for name, value in zip(JOINT_NAMES, arm.state, strict=True)
    }
    action["gripper.pos"] = -4.0

    sent = robot.send_action(action)
    arm.control_fn(arm, 1 / arm.rate)

    assert sent["gripper.pos"] == pytest.approx(robot.config.gripper_open_position)
    assert arm.gripper.commands[-1][1][0] == [robot.config.gripper_open_position]
    assert arm.gripper.commands[-1][2]["kp"] == [robot.config.gripper_position_kp]
    robot.disconnect()


def test_deploy_rejects_bad_actions(tmp_path) -> None:
    robot = make_robot(tmp_path, "deploy")
    robot.connect()

    with pytest.raises(ValueError, match="动作字段不匹配"):
        robot.send_action({"joint1.pos": 0.0})

    action = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
    action["joint4.pos"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        robot.send_action(action)
    robot.disconnect()


def test_deploy_rejects_failed_mode_switch_before_enable(tmp_path) -> None:
    FakeArm.fail_mode_switch = True
    robot = make_robot(tmp_path, "deploy")

    with pytest.raises(RuntimeError, match="控制模式切换失败"):
        robot.connect()

    arm = FakeArm.instances[-1]
    assert not arm.enabled
    assert not arm.loop_active


def test_deploy_rejects_out_of_range_initial_state(tmp_path) -> None:
    FakeArm.initial_state[0] = 3.0
    robot = make_robot(tmp_path, "deploy")

    with pytest.raises(RuntimeError, match="初始关节位置超限"):
        robot.connect()

    arm = FakeArm.instances[-1]
    assert not arm.enabled
    assert not arm.loop_active


def test_teach_uses_gravity_and_force_limited_gripper(tmp_path) -> None:
    robot = make_robot(tmp_path, "teach")
    robot.connect()
    arm = FakeArm.instances[-1]

    assert arm.arm.mode == "mit"
    robot.close_gripper()
    arm.control_fn(arm, 1 / arm.rate)

    np.testing.assert_allclose(arm.arm.commands[-1][2]["tau"], np.arange(1, 7))
    assert arm.gripper.commands[-1][2]["kp"] == [0.0]
    assert arm.gripper.commands[-1][2]["kd"] == [robot.config.gripper_close_kd]
    assert arm.gripper.commands[-1][2]["tau"] == [robot.config.gripper_close_torque]

    robot.open_gripper()
    arm.control_fn(arm, 0.1)
    assert arm.gripper.commands[-1][1][0] == [robot.config.gripper_open_position]
    assert arm.gripper.commands[-1][2]["kp"] == [robot.config.gripper_position_kp]
    assert arm.gripper.commands[-1][2]["tau"] == [0.0]
    robot.disconnect()


def test_emergency_stop_stops_control_loop(tmp_path) -> None:
    robot = make_robot(tmp_path, "teach")
    robot.connect()
    arm = FakeArm.instances[-1]

    robot.emergency_stop()

    assert arm.estopped
    assert not arm.loop_active
    with pytest.raises(RuntimeError, match="急停"):
        robot.get_observation()
    robot.disconnect()


def test_config_rejects_unsafe_limits(tmp_path) -> None:
    with pytest.raises(ValueError, match="joint_limits"):
        ReBotB601Config(
            id="bad-limits",
            calibration_dir=tmp_path,
            joint_limits={"joint1": (-1.0, 1.0)},
        )

