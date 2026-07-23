"""reBot Arm B601 的 LeRobot 硬件适配。"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
import time
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .camera import ReBotRGBDCamera
from .config import JOINT_NAMES, ReBotB601Config

logger = logging.getLogger(__name__)


def _default_sdk_path() -> Path:
    # 当前文件位于 rebot/lerobot/src/lerobot_robot_rebot，parents[3] 是 rebot 根目录。
    return Path(__file__).resolve().parents[3] / "third_party" / "reBotArm_control_py"


def _load_sdk(sdk_path: Path | None) -> tuple[type, Callable[..., np.ndarray]]:
    """延迟导入 SDK，让配置检查和数据处理测试不依赖真机。"""

    root = (sdk_path or _default_sdk_path()).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"未找到 reBotArm SDK: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    actuator = importlib.import_module("reBotArm_control_py.actuator")
    dynamics = importlib.import_module("reBotArm_control_py.dynamics")
    return actuator.RebotArm, dynamics.compute_generalized_gravity


class ReBotB601(Robot):
    """B601 适配器，关节位置、速度和策略动作均使用 rad。

    ``teach`` 模式使用 MIT 重力补偿，方便直接拖动机械臂；``deploy`` 模式使用
    POS_VEL 接收策略位置目标。一个连接期间控制模式保持不变，切换模式必须断开重连，
    防止将位置动作误发给 MIT 力矩接口。
    """

    config_class = ReBotB601Config
    name = "rebot_b601"

    def __init__(self, config: ReBotB601Config):
        super().__init__(config)
        self.config = config
        self.camera = ReBotRGBDCamera(config.camera)
        self._arm: Any | None = None
        self._connected = False
        self._state_lock = threading.Lock()
        self._state = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._velocity = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._torque = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._target = self._state.copy()
        self._state_time = 0.0
        self._gripper_command = "hold"
        self._control_error: BaseException | None = None
        self._compute_gravity: Callable[..., np.ndarray] | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        height, width = self.config.camera.height, self.config.camera.width
        motors = {f"{name}.pos": float for name in JOINT_NAMES}
        return {
            **motors,
            "main_rgb": (height, width, 3),
            # 深度图保留为三通道 uint8，便于直接复用视觉模型的 image encoder。
            "main_depth": (height, width, 3),
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in JOINT_NAMES}

    @property
    def is_connected(self) -> bool:
        return self._connected and self._arm is not None and self.camera.is_connected

    @property
    def is_calibrated(self) -> bool:
        # 机械零点由厂商 SDK 与硬件配置管理，插件不会擅自重写零点。
        return True

    def calibrate(self) -> None:
        logger.info("B601 使用现有硬件零点，不执行 LeRobot 自动标定")

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._control_error = None
        arm_class, compute_gravity = _load_sdk(self.config.sdk_path)
        hardware_yaml = str(self.config.hardware_yaml) if self.config.hardware_yaml else None
        arm = arm_class(hw_yaml=hardware_yaml)

        try:
            arm.connect()
            self._arm = arm
            self._compute_gravity = compute_gravity
            self._refresh_state(request_feedback=True)
            with self._state_lock:
                self._target = self._state.copy()
                initial_state = self._state.copy()

            # 在切换模式和使能前核对实测位置。反馈异常或零点配置错误时，宁可拒绝启动，
            # 也不能让 POS_VEL 控制器把机械臂拉向错误目标。
            invalid = [
                f"{name}={value:.4f}（范围 {lower:.4f}~{upper:.4f}）"
                for name, value in zip(JOINT_NAMES, initial_state, strict=True)
                for lower, upper in [self.config.joint_limits[name]]
                if not lower <= value <= upper
            ]
            if invalid:
                raise RuntimeError("B601 初始关节位置超限，拒绝使能: " + ", ".join(invalid))

            self.camera.connect()
            # 控制线程可能立即执行；先标记连接，确保线程内安全检查状态一致。
            self._connected = True
            self.configure()
        except Exception:
            self._connected = False
            if self.camera.is_connected:
                self.camera.disconnect()
            try:
                arm.disconnect()
            except Exception:
                logger.exception("B601 连接失败后的回滚也发生异常")
            self._arm = None
            raise

        logger.info("%s 已连接，模式=%s", self, self.config.operating_mode)

    def configure(self) -> None:
        if self._arm is None:
            raise RuntimeError("必须先连接 B601，再配置控制模式")
        if self.config.operating_mode == "readonly":
            return

        if self.config.operating_mode == "teach":
            mode_ok = self._arm.arm.mode_mit()
        else:
            velocity_limits = np.asarray(self.config.arm_velocity_limits, dtype=np.float64)
            mode_ok = self._arm.arm.mode_pos_vel(velocity_limits)
        gripper_mode_ok = self._arm.gripper.mode_mit()
        if not mode_ok or not gripper_mode_ok:
            raise RuntimeError(
                "B601 控制模式切换失败，拒绝使能: "
                f"arm={bool(mode_ok)}, gripper={bool(gripper_mode_ok)}"
            )

        # 先以当前反馈作为目标，再使能，避免启动瞬间向旧目标跳变。
        with self._state_lock:
            self._target = self._state.copy()
        self._arm.enable_all()
        self._arm.start_control_loop(self._control_step, rate=self.config.control_rate_hz)

    def _refresh_state(self, request_feedback: bool) -> None:
        if self._arm is None:
            raise RuntimeError("B601 尚未连接")
        position, velocity, torque = self._arm.get_state(request_feedback=request_feedback)
        expected_shape = (len(JOINT_NAMES),)
        arrays = {"位置": position, "速度": velocity, "力矩": torque}
        for label, array in arrays.items():
            if array.shape != expected_shape:
                raise RuntimeError(f"B601 {label}应为 {expected_shape}，实际 {array.shape}")
            if not np.all(np.isfinite(array)):
                raise RuntimeError(f"B601 {label}包含 NaN 或 Inf")

        with self._state_lock:
            self._state = position.copy()
            self._velocity = velocity.copy()
            self._torque = torque.copy()
            self._state_time = time.monotonic()

    def _control_step(self, arm: Any, dt: float) -> None:
        """SDK 唯一发送循环；异常时直接失能，不在本线程 join 自己。"""

        try:
            self._refresh_state(request_feedback=False)
            with self._state_lock:
                state = self._state.copy()
                target = self._target.copy()
                gripper_command = self._gripper_command

            if self.config.operating_mode == "teach":
                if self._compute_gravity is None:
                    raise RuntimeError("重力补偿函数尚未加载")
                gravity = np.asarray(self._compute_gravity(q=state[:6]), dtype=np.float64)
                if gravity.shape != (6,) or not np.all(np.isfinite(gravity)):
                    raise RuntimeError("重力补偿计算结果无效")
                gravity *= np.asarray(self.config.gravity_scale, dtype=np.float64)
                arm.arm.send_mit(
                    pos=state[:6],
                    vel=np.zeros(6),
                    kp=np.full(6, self.config.teach_kp),
                    kd=np.full(6, self.config.teach_kd),
                    tau=gravity,
                )
            elif self.config.operating_mode == "deploy":
                arm.arm.send_pos_vel(
                    target[:6],
                    vlim=np.asarray(self.config.arm_velocity_limits, dtype=np.float64),
                )
            else:
                raise RuntimeError("readonly 模式不应启动控制循环")

            self._send_gripper(arm, state[6], target[6], gripper_command)
        except BaseException as exc:
            self._control_error = exc
            arm.estop()
            raise

    def _send_gripper(self, arm: Any, position: float, target: float, command: str) -> None:
        if command == "close":
            if position < self.config.gripper_close_position - self.config.gripper_close_tolerance:
                # 正向前馈力矩有上限；碰到物体后不会继续增大夹持力。
                arm.gripper.send_mit(
                    [position],
                    kp=[0.0],
                    kd=[self.config.gripper_close_kd],
                    tau=[self.config.gripper_close_torque],
                )
                return
            with self._state_lock:
                self._target[6] = position
                self._gripper_command = "hold"
            target = position
        elif command == "open":
            # 直接下发完整开爪目标。若每周期只给当前位置前方一个极小增量，
            # 位置误差产生的力矩不足以克服夹爪摩擦，表现为按 O 但不运动。
            target = self.config.gripper_open_position
            with self._state_lock:
                self._target[6] = target

        arm.gripper.send_mit(
            [target],
            kp=[self.config.gripper_position_kp],
            kd=[self.config.gripper_kd],
            tau=[0.0],
        )

    def _raise_if_control_failed(self) -> None:
        if self._control_error is not None:
            raise RuntimeError("B601 控制循环已触发急停") from self._control_error
        if self.config.operating_mode == "readonly":
            return
        age = time.monotonic() - self._state_time
        if age > self.config.feedback_timeout_s:
            self.emergency_stop()
            raise RuntimeError(f"B601 反馈超时 {age:.3f}s，已触发急停")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self.config.operating_mode == "readonly":
            self._refresh_state(request_feedback=True)
        else:
            self._raise_if_control_failed()

        with self._state_lock:
            state = self._state.copy()
        color_rgb, model_depth = self.camera.read()
        observation: RobotObservation = {
            f"{name}.pos": float(value)
            for name, value in zip(JOINT_NAMES, state, strict=True)
        }
        observation["main_rgb"] = color_rgb
        observation["main_depth"] = model_depth
        return observation

    def get_depth_mm(self) -> np.ndarray:
        """返回最近一次 observation 对应的原始 uint16 毫米深度。"""

        return self.camera.read_depth_mm()

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.operating_mode != "deploy":
            raise RuntimeError("send_action 仅允许在 deploy 模式使用")
        self._raise_if_control_failed()

        expected = set(self.action_features)
        if set(action) != expected:
            missing = sorted(expected - set(action))
            extra = sorted(set(action) - expected)
            raise ValueError(f"动作字段不匹配，缺少={missing}，多余={extra}")

        requested = np.asarray([action[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float64)
        if not np.all(np.isfinite(requested)):
            raise ValueError("动作包含 NaN 或 Inf")

        with self._state_lock:
            present = self._state.copy()
        # 六个机械臂关节先限制单步位移，再限制绝对范围。
        arm_relative = np.clip(
            requested[:6],
            present[:6] - self.config.max_relative_target,
            present[:6] + self.config.max_relative_target,
        )
        safe_arm = np.asarray(
            [
                np.clip(value, *self.config.joint_limits[name])
                for name, value in zip(JOINT_NAMES[:6], arm_relative, strict=True)
            ],
            dtype=np.float64,
        )

        # 示教数据中的夹爪是 O/C 二值意图，单帧实测变化可达约 0.7rad，不能套用
        # 机械臂的 0.01~0.08rad 小步限幅。开爪复用已验证的完整位置目标；闭爪
        # 复用 0.6N·m 限力模式，接触物体后不会继续提高夹持力。
        gripper_delta = requested[6] - present[6]
        if gripper_delta > self.config.gripper_policy_deadband:
            gripper_target = self.config.gripper_close_position
            gripper_command = "close"
        elif gripper_delta < -self.config.gripper_policy_deadband:
            gripper_target = self.config.gripper_open_position
            gripper_command = "open"
        else:
            gripper_target = float(
                np.clip(requested[6], *self.config.joint_limits["gripper"])
            )
            gripper_command = "hold"

        safe = np.append(safe_arm, gripper_target)
        with self._state_lock:
            self._target = safe
            self._gripper_command = gripper_command
        return {
            f"{name}.pos": float(value)
            for name, value in zip(JOINT_NAMES, safe, strict=True)
        }

    def open_gripper(self) -> None:
        self._require_teach_mode()
        with self._state_lock:
            self._gripper_command = "open"

    def close_gripper(self) -> None:
        self._require_teach_mode()
        with self._state_lock:
            self._gripper_command = "close"

    def hold_gripper(self) -> None:
        self._require_teach_mode()
        with self._state_lock:
            self._target[6] = self._state[6]
            self._gripper_command = "hold"

    def _require_teach_mode(self) -> None:
        if not self.is_connected or self.config.operating_mode != "teach":
            raise RuntimeError("夹爪示教命令仅允许在已连接的 teach 模式使用")
        self._raise_if_control_failed()

    def emergency_stop(self) -> None:
        """停止发送线程并失能全部电机；可由键盘或安全检查调用。"""

        if self._arm is not None:
            # 先失能，再等待发送线程退出，避免 join 最长等待期间电机仍保持使能。
            self._arm.estop()
            self._arm.stop_control_loop()
        self._control_error = RuntimeError("用户或安全检查触发急停")

    def disconnect(self) -> None:
        if self._arm is None and not self.camera.is_connected:
            return
        arm = self._arm
        self._connected = False
        try:
            if arm is not None:
                arm.stop_control_loop()
                arm.disconnect()
        finally:
            if self.camera.is_connected:
                self.camera.disconnect()
            self._arm = None
        logger.info("%s 已断开", self)
