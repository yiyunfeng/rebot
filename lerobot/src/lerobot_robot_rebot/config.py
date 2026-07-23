"""B601 硬件、安全与控制模式配置。"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from lerobot.robots.config import RobotConfig

from .camera import ReBotRGBDConfig

JointLimit: TypeAlias = tuple[float, float]
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper")
DEFAULT_JOINT_LIMITS: dict[str, JointLimit] = {
    "joint1": (-2.8, 2.8),
    "joint2": (-3.14, 0.0),
    "joint3": (-3.14, 0.0),
    "joint4": (-1.87, 1.57),
    "joint5": (-1.57, 1.57),
    "joint6": (-3.14, 3.14),
    "gripper": (-5.0, 0.0),
}


@RobotConfig.register_subclass("rebot_b601")
@dataclass
class ReBotB601Config(RobotConfig):
    """默认只读；必须显式选择 teach/deploy 才允许使能电机。"""

    operating_mode: Literal["readonly", "teach", "deploy"] = "readonly"
    sdk_path: Path | None = None
    hardware_yaml: Path | None = None
    camera: ReBotRGBDConfig = field(default_factory=ReBotRGBDConfig)
    joint_limits: dict[str, JointLimit] = field(default_factory=lambda: DEFAULT_JOINT_LIMITS.copy())
    max_relative_target: float = 0.08
    arm_velocity_limits: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    control_rate_hz: float = 500.0
    feedback_timeout_s: float = 0.2
    teach_kp: float = 2.0
    teach_kd: float = 1.0
    gravity_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # 与 rebot_grasp 一致，避开 -5.0 rad 机械硬限位，使用 98% 开度。
    gripper_open_position: float = -4.9
    gripper_close_position: float = 0.0
    gripper_position_kp: float = 5.0
    gripper_kd: float = 1.0
    gripper_close_kd: float = 0.5
    gripper_close_torque: float = 0.6
    gripper_close_tolerance: float = 0.03
    gripper_policy_deadband: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if set(self.joint_limits) != set(JOINT_NAMES):
            raise ValueError(f"joint_limits 必须且只能包含 {JOINT_NAMES}")
        for name, (lower, upper) in self.joint_limits.items():
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"{name} 的关节限制必须是递增的有限值")
        if len(self.arm_velocity_limits) != 6 or any(value <= 0 for value in self.arm_velocity_limits):
            raise ValueError("arm_velocity_limits 必须包含 6 个正数")
        if len(self.gravity_scale) != 6:
            raise ValueError("gravity_scale 必须包含 6 个数")
        if self.max_relative_target <= 0 or self.feedback_timeout_s <= 0:
            raise ValueError("动作增量和反馈超时必须为正数")
        if not 0 < self.gripper_close_torque <= 1.0:
            raise ValueError("gripper_close_torque 必须位于 (0, 1.0] N·m")
        if self.gripper_policy_deadband <= 0:
            raise ValueError("gripper_policy_deadband 必须为正数")
        gripper_lower, gripper_upper = self.joint_limits["gripper"]
        if not gripper_lower <= self.gripper_open_position <= gripper_upper:
            raise ValueError("gripper_open_position 超出夹爪限制")
        if not gripper_lower <= self.gripper_close_position <= gripper_upper:
            raise ValueError("gripper_close_position 超出夹爪限制")
