"""机械臂控制 API。

运动方式：
    move_joints: 直接发关节轨迹，不做 MoveIt 规划。
    moveit_to_joints: MoveIt 规划到关节目标。
    moveit_to_pose: MoveIt 规划到 TCP 位姿约束，可能比关节目标慢。
    moveit_execute: 执行调用方构造的 MoveIt Constraints。
    solve_ik / solve_ik_hardware: 只求逆解，不执行。
    open_gripper / close_gripper: 控制夹爪。

视觉抓取推荐直接写：solve_ik(pose) -> moveit_to_joints(joints)。

示例：
    node = rclpy.create_node("my_node")
    arm = RealController(node)
    arm.enable()
    arm.open_gripper()
    arm.move_joints([0.0, -0.05, -0.05, 0.0, 0.0, 0.0])
    arm.moveit_to_pose(target_pose)
    ik = arm.solve_ik(target_pose)
    if ik is not None:
        arm.moveit_to_joints(ik)
"""

from __future__ import annotations

import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.action import GripperCommand as GripperAction
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetMotionPlan, GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf_transformations import quaternion_from_euler
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rebotarm_msgs.srv import MoveToPoseIK

# ── 默认常量 ──
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]  # 六轴关节名
GRIPPER_JOINTS = ["gripper_joint1", "gripper_joint2"]                      # 两指夹爪关节名
HOME_JOINTS = [0.0, -0.05, -0.05, 0.0, 0.0, 0.0]                          # Home 位姿（弧度）
GRIPPER_OPEN = 0.06                    # 夹爪张开宽度（模拟值）
GRIPPER_CLOSE = 0.0                    # 夹爪闭合宽度（模拟值）
MAX_GRIPPER_WIDTH = 0.09               # 最大开口宽度（硬件映射用）
HW_OPEN_GRIPPER = -5.0                 # 硬件张开指令值（DM 型号）
HW_CLOSE_GRIPPER = 0.0                 # 硬件闭合指令值（DM 型号）
GRIPPER_MAX_EFFORT = 10.0              # 夹爪最大力矩
NAMESPACE = "rebotarm"                 # 机械臂命名空间
TASK_FRAME = "base_link"               # 规划参考坐标系

# ── DM 机械臂关节角全局限制（单位：rad） ──
#
# 来源：reBot_B601_DM_with_gripper_gazebo.urdf 中 joint1~joint6 的
# <limit lower="..." upper="...">。RealController 是多个真机流程共用的
# 底层接口，所以这里做最后一道兜底：直接关节控制、IK 返回值、MoveIt
# 规划轨迹在发给真机前都必须满足这些限制。
#
# 注意：这只限制关节旋转角度，不负责碰撞检测、速度限制或 TCP 工作空间限制。
# 如果 URDF 里的 DM 关节限位更新，这里也要同步更新。
JOINT_POSITION_LIMITS = {
    "joint1": (-2.8, 2.8),
    "joint2": (-3.14, 0.0),
    "joint3": (-3.14, 0.0),
    "joint4": (-1.87, 1.57),
    "joint5": (-1.57, 1.57),
    "joint6": (-3.14, 3.14),
}
# 浮点计算会有极小误差，给 0.0001 rad 容差，避免刚好贴边的合法解被误拒。
JOINT_LIMIT_EPS = 1e-4
# MoveIt 可能把真实当前关节状态作为轨迹第 0 点插入。真机编码器/URDF 零位
# 存在细小偏差时，起始点可能刚好越过 URDF 边界，例如 joint2=+0.018 rad。
# 这个容差只用于轨迹第 0 点，后续轨迹点仍使用严格限位，避免继续向越界方向运动。
START_STATE_LIMIT_TOLERANCE = 0.03


# ==============================================================================
# 工具函数
# ==============================================================================

def make_pose_from_rpy(
    x: float, y: float, z: float, rpy: tuple[float, float, float]
) -> Pose:
    """由位置 + RPY 欧拉角构造 Pose 消息"""
    qx, qy, qz, qw = quaternion_from_euler(*rpy)  # RPY → 四元数
    return Pose(
        position=Point(x=float(x), y=float(y), z=float(z)),
        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )


def make_sphere(radius: float) -> SolidPrimitive:
    """创建球体几何基元，用于位置约束的容差区域"""
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE  # 几何类型：球体
    sphere.dimensions = [radius]         # 球体尺寸：半径（米）
    return sphere


def nearest_angle(angle: float, reference: float) -> float:
    """将 angle 调整到 reference 的 ±π 范围内，避免角度跳变"""
    while angle - reference > math.pi:
        angle -= 2.0 * math.pi  # 向下修正一个整圈
    while angle - reference < -math.pi:
        angle += 2.0 * math.pi  # 向上修正一个整圈
    return angle


def nearest_joint_values(
    values: list[float], reference: list[float]
) -> list[float]:
    """将每个关节角调整到参考值的最近等效角，避免整圈旋转"""
    return [nearest_angle(float(v), float(r)) for v, r in zip(values, reference)]


def clamp_joint_values_near_limits(
    values: list[float], joint_names: list[str] = ARM_JOINTS
) -> list[float]:
    """把限位附近的小误差钳到合法边界。"""
    fixed: list[float] = []
    for joint_name, value in zip(joint_names, values):
        lower, upper = JOINT_POSITION_LIMITS[joint_name]
        value = float(value)
        if lower - START_STATE_LIMIT_TOLERANCE <= value < lower:
            value = lower
        elif upper < value <= upper + START_STATE_LIMIT_TOLERANCE:
            value = upper
        fixed.append(value)
    return fixed


def equivalent_joint_values_within_limits(
    values: list[float], joint_names: list[str]
) -> list[float]:
    """把等效关节角拉回 URDF 限位范围。

    MoveIt IK 返回的角度和 nearest_angle 处理后的角度，可能是同一个物理
    姿态的不同 2π 等效表示。例如 joint6=-3.6874 超出 [-3.14, 3.14]，
    但加 2π 后是 2.5958，姿态等效且满足 URDF 限位。

    这里只做“等效角归位”，不修改不可等效修复的真实越界解。
    """
    fixed: list[float] = []
    for joint_name, value in zip(joint_names, values):
        lower, upper = JOINT_POSITION_LIMITS.get(joint_name, (-math.inf, math.inf))
        candidate = float(value)
        if math.isfinite(lower) and math.isfinite(upper):
            turns_min = math.ceil(
                (lower - JOINT_LIMIT_EPS - candidate) / (2.0 * math.pi)
            )
            turns_max = math.floor(
                (upper + JOINT_LIMIT_EPS - candidate) / (2.0 * math.pi)
            )
            if turns_min <= turns_max:
                turns = min(range(turns_min, turns_max + 1), key=abs)
                candidate += turns * 2.0 * math.pi
        fixed.append(candidate)
    return fixed


def normalize_joint_values(
    values: list[float],
    reference: list[float],
    joint_names: list[str] = ARM_JOINTS,
) -> list[float]:
    """选择最近的 2*pi 等效角，并修正限位附近的小误差。"""
    normalized = nearest_joint_values(values, reference)
    normalized = equivalent_joint_values_within_limits(normalized, joint_names)
    return clamp_joint_values_near_limits(normalized, joint_names)


def build_single_point_trajectory(
    joints: list[str],
    positions: list[float],
    seconds: float,
) -> JointTrajectory:
    """构造一个单点 JointTrajectory，用于发送简单关节位置指令。

    Args:
        joints: 关节名称列表
        positions: 目标位置（弧度），与 joints 一一对应
        seconds: 运动时长（秒）
    """
    traj = JointTrajectory()
    traj.joint_names = list(joints)                     # 关节名列表
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in positions]     # 目标位置
    sec = int(seconds)                                   # 秒整数部分
    point.time_from_start = Duration(
        sec=sec,
        nanosec=int((seconds - sec) * 1_000_000_000),   # 纳秒部分：小数 × 10^9
    )
    traj.points = [point]                                # 单轨迹点
    return traj


# ==============================================================================
# RealController — 真实机械臂控制 API
# ==============================================================================

class RealController:
    """真实机械臂控制 API。

    通过组合模式持有一个 Node 引用，所有 ROS 通信操作委托给该 Node。
    调用方负责创建 Node 并传入。

    用法示例:
        node = rclpy.create_node("my_app")
        arm = RealController(node)

        # 基本操作
        arm.enable()                                     # 电机使能
        arm.open_gripper()                               # 张开夹爪
        arm.close_gripper()                              # 闭合夹爪
        arm.home()                                       # 回 Home

        # 关节控制
        arm.move_joints([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])

        # 位姿控制
        pose = make_pose_from_rpy(0.3, 0.1, 0.2, (0, 1.57, 0))
        arm.moveit_to_pose(pose)

        # IK 求解
        ik = arm.solve_ik(pose)                          # MoveIt IK
        ik = arm.solve_ik_hardware(pose)                  # SDK IK（硬件）
    """

    # ── 可配置参数 ──
    move_duration: float = 3.0           # 单段关节运动时长（秒）
    arm_result_timeout: float = 20.0     # 机械臂执行超时（秒）
    
    position_tolerance: float = 0.015    # 位置约束容差（米）
    orientation_tolerance: float = 0.2   # 姿态约束容差（弧度）
    planning_attempts: int = 5           # 最大规划尝试次数
    planning_time: float = 5.0           # 每次规划超时（秒）
    ik_request_timeout: int = 5          # /compute_ik 内部求解超时（秒）
    ik_service_timeout: float = 6.0      # 等待 /compute_ik 返回的最长时间（秒）
    
    #路径规划Moveit的配置
    velocity_scaling: float = 0.5        # 全局速度倍率（0-1）
    acceleration_scaling: float = 0.4    # 全局加速度倍率（0-1）

    def __init__(self, node: Node) -> None:
        """初始化控制器。

        Args:
            node: 调用方的 rclpy Node 实例，所有 ROS 通信绑定到该 Node
        """
        self._node = node
        self._logger = node.get_logger()

        # ── 订阅关节状态 ──
        # Gazebo 仿真发布 /joint_states；真机链路发布 /rebotarm/joint_states。
        # 两个都监听，保证 MoveIt 规划起点使用当前真实姿态，而不是 HOME 兜底值。
        self._latest_joints: dict[str, float] = {}
        for topic in ("/joint_states", f"/{NAMESPACE}/joint_states"):
            node.create_subscription(
                JointState,
                topic,
                self._joint_state_cb,
                qos_profile_sensor_data,
            )

        # ── 机械臂轨迹执行（无路径规划） ──
        self._arm_client = ActionClient(
            node, FollowJointTrajectory,
            f"/{NAMESPACE}/follow_joint_trajectory",
        )

        # ── 夹爪控制 ──
        self._gripper_client = ActionClient(
            node, GripperAction, f"/{NAMESPACE}/gripper/command"
        )

        # ── IK 求解（两种后端） ──
        self._ik_client = node.create_client(GetPositionIK, "/compute_ik")
        self._hw_ik_client = node.create_client(
            MoveToPoseIK, f"/{NAMESPACE}/move_to_pose_ik"
        )

        # ── MoveIt 路径规划 ──
        self._planner = node.create_client(GetMotionPlan, "/plan_kinematic_path")
        self._executor = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")

    # ==================================================================
    # 关节状态
    # ==================================================================

    def _joint_state_cb(self, msg: JointState) -> None:
        """订阅回调：缓存最新关节角度"""
        for name, position in zip(msg.name, msg.position):
            self._latest_joints[name] = float(position)

    def get_current_joints(self) -> list[float]:
        """获取当前六轴关节角度列表，未收到数据时回退到 HOME"""
        return [
            self._latest_joints.get(name, HOME_JOINTS[i])
            for i, name in enumerate(ARM_JOINTS)
        ]

    def _current_robot_state(self) -> RobotState:
        """构造当前 RobotState（用于运动规划的起始状态）"""
        current = self.get_current_joints()
        planning = clamp_joint_values_near_limits(current)
        for name, measured, corrected in zip(ARM_JOINTS, current, planning):
            if measured != corrected:
                self._logger.warn(
                    f"规划起点 {name}={measured:.4f} 轻微越界，"
                    f"按合法边界 {corrected:.4f} 规划"
                )
        state = RobotState()
        state.joint_state.name = list(ARM_JOINTS)              # 关节名
        state.joint_state.position = planning
        state.is_diff = False  # 非增量，完整状态
        return state

    def _wait_future(self, future, timeout_sec: float) -> bool:
        """等待 ROS future 完成，兼容普通脚本和多线程回调两种用法。"""
        executor = getattr(self._node, "executor", None)
        if executor is not None and getattr(executor, "is_spinning", False):
            deadline = time.monotonic() + float(timeout_sec)
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
        else:
            rclpy.spin_until_future_complete(
                self._node,
                future,
                timeout_sec=float(timeout_sec),
            )
        return future.done()

    def _joints_within_limits(self, joints: list[float], label: str) -> bool:
        """检查六轴目标角是否满足 DM URDF 定义的关节旋转范围。

        用于直接关节控制和 IK 返回值检查。调用前通常已经通过
        normalize_joint_values 选取了离当前姿态最近的等效角，因此这里检查的
        是最终准备发送或返回给上层流程的真实目标角。
        """
        for name, value in zip(ARM_JOINTS, joints):
            lower, upper = JOINT_POSITION_LIMITS[name]
            if lower - JOINT_LIMIT_EPS <= float(value) <= upper + JOINT_LIMIT_EPS:
                continue
            self._logger.error(
                f"{label}: {name}={value:.4f} 超出关节角限制 "
                f"[{lower:.4f}, {upper:.4f}]"
            )
            return False
        return True

    def _trajectory_within_joint_limits(self, trajectory: JointTrajectory) -> bool:
        """检查 MoveIt 规划出的整条轨迹是否满足 DM 六轴关节角限制。

        检查时机在 _fix_trajectory_continuity 之后、ExecuteTrajectory 之前。
        轨迹点 0 通常是当前真实起始状态，允许一个很小的起始误差容差；
        从轨迹点 1 开始必须严格满足 URDF 限位，避免底层通用接口把不合法
        角度发给真机。
        """
        joint_names = list(trajectory.joint_names)
        for point_index, point in enumerate(trajectory.points):
            positions = list(point.positions)
            for joint_name, (lower, upper) in JOINT_POSITION_LIMITS.items():
                if joint_name not in joint_names:
                    continue
                joint_index = joint_names.index(joint_name)
                if joint_index >= len(positions):
                    self._logger.error(
                        f"MoveIt 轨迹点 {point_index} 缺少 {joint_name} 的 position，拒绝执行"
                    )
                    return False
                value = float(positions[joint_index])
                if lower - JOINT_LIMIT_EPS <= value <= upper + JOINT_LIMIT_EPS:
                    continue
                if (
                    point_index == 0
                    and lower - START_STATE_LIMIT_TOLERANCE
                    <= value
                    <= upper + START_STATE_LIMIT_TOLERANCE
                ):
                    self._logger.warn(
                        f"MoveIt 轨迹起始点 {joint_name}={value:.4f} 轻微超出关节角限制 "
                        f"[{lower:.4f}, {upper:.4f}]，按当前真机状态容差放行"
                    )
                    continue
                self._logger.error(
                    f"MoveIt 轨迹点 {point_index}: {joint_name}={value:.4f} "
                    f"超出关节角限制 [{lower:.4f}, {upper:.4f}]，拒绝执行"
                )
                return False
        return True

    # ==================================================================
    # 高层便捷 API（主要调用接口）
    # ==================================================================

    def enable(self) -> None:
        """电机使能（上电/解锁）"""
        client = self._node.create_client(Trigger, f"/{NAMESPACE}/enable")
        if not client.wait_for_service(timeout_sec=5.0):
            self._logger.warn("enable service not ready")
            return
        future = client.call_async(Trigger.Request())
        self._wait_future(future, 10.0)
        res = future.result()
        if res is not None and res.success:
            self._logger.info("Robot enabled")
        else:
            msg = res.message if res is not None else "no response"
            self._logger.warn(f"Enable failed: {msg}")

    def open_gripper(self) -> bool:
        """张开夹爪"""
        return self.move_gripper(GRIPPER_OPEN, "open")

    def close_gripper(self) -> bool:
        """闭合夹爪"""
        return self.move_gripper(GRIPPER_CLOSE, "close")

    def home(self) -> bool:
        """通过 MoveIt 规划回到 Home 位姿"""
        constraints = Constraints(name="home")
        constraints.joint_constraints = [
            JointConstraint(
                joint_name=name,
                position=pos,
                tolerance_above=0.02,
                tolerance_below=0.02,
                weight=1.0,
            )
            for name, pos in zip(ARM_JOINTS, HOME_JOINTS)
        ]
        return self.moveit_execute(constraints)

    def move_joints(self, joints: list[float], duration: float | None = None) -> bool:
        """直接移动六轴关节到目标角度（无路径规划，使用 FollowJointTrajectory）。

        Args:
            joints: 六轴目标角度（弧度）
            duration: 运动时长（秒），默认使用 self.move_duration
        """
        if duration is None:
            duration = self.move_duration
        return self._move_arm_direct(joints, duration)

    def moveit_to_pose(self, pose: Pose) -> bool:
        """MoveIt 规划到 TCP 位姿约束。"""
        return self._go_pose(pose)

    def moveit_to_joints(self, joints: list[float], label: str = "joint_goal") -> bool:
        """MoveIt 规划到六轴关节目标。"""
        joints = [float(v) for v in joints]
        if not self._joints_within_limits(joints, label):
            return False

        constraints = Constraints(name=label)
        constraints.joint_constraints = [
            JointConstraint(
                joint_name=name,
                position=position,
                tolerance_above=0.02,
                tolerance_below=0.02,
                weight=1.0,
            )
            for name, position in zip(ARM_JOINTS, joints)
        ]
        self._logger.info(
            f"MoveIt joint goal {label}: {[round(v, 4) for v in joints]}"
        )
        return self.moveit_execute(constraints)

    # ==================================================================
    # IK 求解
    # ==================================================================

    def solve_ik(
        self, pose: Pose, seed: list[float] | None = None
    ) -> list[float] | None:
        """MoveIt IK：TCP 位姿 -> 六轴关节角，失败返回 None。"""
        if seed is None:
            seed = self.get_current_joints()

        if not self._ik_client.wait_for_service(timeout_sec=3.0):
            self._logger.error("IK: /compute_ik not ready")
            return None

        # 构造 IK 请求
        state = RobotState()
        state.joint_state.name = list(ARM_JOINTS)
        state.joint_state.position = [float(v) for v in seed]
        request = GetPositionIK.Request()
        request.ik_request.group_name = "arm"                   # 规划组
        request.ik_request.robot_state = state                  # 种子状态
        request.ik_request.ik_link_name = "gripper_tcp"         # 目标 link
        request.ik_request.avoid_collisions = False             # 不避障
        request.ik_request.pose_stamped = PoseStamped(
            header=Header(frame_id=TASK_FRAME),
            pose=pose,
        )
        request.ik_request.timeout = Duration(sec=int(self.ik_request_timeout))

        future = self._ik_client.call_async(request)
        self._wait_future(future, float(self.ik_service_timeout))
        if not future.done() or future.result() is None:
            self._logger.error(f"IK: service timeout after {self.ik_service_timeout:.1f}s")
            return None

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self._logger.error(f"IK: failed, code={response.error_code.val}")
            return None

        # 提取关节角并归一化到最近等效角
        joint_map = dict(zip(
            response.solution.joint_state.name,
            response.solution.joint_state.position,
        ))
        joints = [float(joint_map[name]) for name in ARM_JOINTS]
        joints = normalize_joint_values(joints, seed)
        if not self._joints_within_limits(joints, "IK solution"):
            return None
        self._logger.info(f"IK solution: {[round(v, 4) for v in joints]}")
        return joints

    def solve_ik_hardware(
        self, pose: Pose, seed: list[float] | None = None
    ) -> list[float] | None:
        """硬件 SDK IK：TCP 位姿 -> 六轴关节角，失败返回 None。"""
        if seed is None:
            seed = self.get_current_joints()

        if not self._hw_ik_client.wait_for_service(timeout_sec=5.0):
            self._logger.warn("SDK IK: /move_to_pose_ik not ready")
            return None

        req = MoveToPoseIK.Request()
        req.target_pose = pose
        future = self._hw_ik_client.call_async(req)
        self._wait_future(future, 5.0)
        res = future.result()
        if res is None or not res.success:
            self._logger.error("SDK IK: failed")
            return None

        joints = [float(v) for v in res.q_solution]
        joints = normalize_joint_values(joints, seed)
        if not self._joints_within_limits(joints, "SDK IK solution"):
            return None
        self._logger.info(f"SDK IK solution: {[round(v, 4) for v in joints]}")
        return joints

    # ==================================================================
    # 夹爪控制
    # ==================================================================

    def move_gripper(self, sim_position: float, label: str = "") -> bool:
        """控制夹爪移动到目标宽度（模拟值 → 硬件指令自动映射）。

        Args:
            sim_position: 模拟夹爪宽度（0 ~ MAX_GRIPPER_WIDTH）
            label: 日志标签
        """
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self._logger.warn(f"[{label}] gripper action not ready")
            return False

        # 模拟宽度 → 硬件指令值
        hardware_position = self._gripper_to_hardware(sim_position)
        goal = GripperAction.Goal()
        goal.command.position = hardware_position                # 硬件目标位置
        goal.command.max_effort = GRIPPER_MAX_EFFORT             # 最大力矩
        self._logger.info(
            f"[{label}] gripper sim={sim_position:.4f} → hw={hardware_position:.4f}"
        )

        # 发送 goal
        future = self._gripper_client.send_goal_async(goal)
        self._wait_future(future, 5.0)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._logger.warn(f"[{label}] gripper goal rejected")
            return False

        # 等待执行完成
        result_future = goal_handle.get_result_async()
        self._wait_future(result_future, 5.0)
        if not result_future.done() or result_future.result() is None:
            self._logger.warn(f"[{label}] gripper timeout")
            return False

        result = result_future.result().result
        self._logger.info(
            f"[{label}] gripper done | reached={result.reached_goal} "
            f"stalled={result.stalled} pos={result.position:.4f} "
            f"effort={result.effort:.4f}"
        )
        if not result.reached_goal:
            self._logger.warn(f"[{label}] gripper did not reach target")
            return False
        return True

    def _gripper_to_hardware(self, sim_position: float) -> float:
        """将模拟夹爪宽度线性映射到硬件指令值。

        映射关系：
            sim 0.0     → HW_CLOSE_GRIPPER  (DM: 0.0)
            sim max_width → HW_OPEN_GRIPPER   (DM: -5.0)
        """
        max_width = MAX_GRIPPER_WIDTH
        ratio = 0.0 if max_width <= 0.0 else 2.0 * float(sim_position) / max_width
        ratio = max(0.0, min(1.0, ratio))  # 限制在 [0, 1]
        return HW_CLOSE_GRIPPER + (HW_OPEN_GRIPPER - HW_CLOSE_GRIPPER) * ratio

    # ==================================================================
    # 机械臂直接关节控制
    # ==================================================================

    def _move_arm_direct(self, joints: list[float], duration: float) -> bool:
        """通过 FollowJointTrajectory action 直接移动关节（无路径规划）。

        Args:
            joints: 六轴目标角度（弧度）
            duration: 运动时长（秒）
        """
        current = self.get_current_joints()
        joints = normalize_joint_values(joints, current)
        if not self._joints_within_limits(joints, "[arm] target"):
            return False

        self._logger.info(
            f"[arm] {[round(v, 4) for v in current]} "
            f"→ {[round(v, 4) for v in joints]} "
            f"({duration:.1f}s)"
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = build_single_point_trajectory(ARM_JOINTS, joints, duration)

        future = self._arm_client.send_goal_async(goal)
        self._wait_future(future, 5.0)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._logger.error("[arm] goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        self._wait_future(result_future, self.arm_result_timeout)
        if not result_future.done():
            self._logger.error(f"[arm] timeout ({self.arm_result_timeout:.1f}s)")
            return False

        result = result_future.result()
        if result is None or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            code = result.result.error_code if result is not None else "none"
            self._logger.error(f"[arm] failed, code={code}")
            return False

        self._logger.info("[arm] done")
        return True

    # ==================================================================
    # MoveIt 路径规划
    # ==================================================================

    def _go_pose(self, pose: Pose) -> bool:
        """通过 TCP 位置+姿态约束让 MoveIt 规划到目标位姿并执行。

        Args:
            pose: TCP 目标位姿（base_link 坐标系）
        """
        constraints = Constraints(name="gripper_tcp_pose")

        # ── 位置约束：TCP 在目标位置 tolerance 球体内 ──
        pc = PositionConstraint()
        pc.header = Header(frame_id=TASK_FRAME)
        pc.link_name = "gripper_tcp"
        pc.weight = 1.0
        pc.constraint_region = BoundingVolume(
            primitives=[make_sphere(self.position_tolerance)],  # 球体容差
            primitive_poses=[Pose(
                position=pose.position,
                orientation=Quaternion(w=1.0),
            )],
        )

        # ── 姿态约束：TCP 朝向偏差 < orientation_tolerance rad ──
        oc = OrientationConstraint()
        oc.header = Header(frame_id=TASK_FRAME)
        oc.link_name = "gripper_tcp"
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = self.orientation_tolerance
        oc.absolute_y_axis_tolerance = self.orientation_tolerance
        oc.absolute_z_axis_tolerance = self.orientation_tolerance
        oc.weight = 1.0

        constraints.position_constraints = [pc]
        constraints.orientation_constraints = [oc]

        return self.moveit_execute(constraints)

    def moveit_execute(self, constraints: Constraints) -> bool:
        """向 move_group 发送规划请求，规划成功则执行轨迹。

        Args:
            constraints: 目标约束
        """
        if not self._planner.wait_for_service(timeout_sec=20.0):
            self._logger.error("/plan_kinematic_path not ready")
            return False
        if not self._executor.wait_for_server(timeout_sec=20.0):
            self._logger.error("/execute_trajectory not ready")
            return False

        # ── 1. 构造规划请求 ──
        request = MotionPlanRequest()
        request.group_name = "arm"
        request.start_state = self._current_robot_state()
        request.goal_constraints = [constraints]
        request.num_planning_attempts = self.planning_attempts
        request.allowed_planning_time = self.planning_time
        request.max_velocity_scaling_factor = self.velocity_scaling
        request.max_acceleration_scaling_factor = self.acceleration_scaling

        # ── 2. 调用规划服务 ──
        plan_request = GetMotionPlan.Request()
        plan_request.motion_plan_request = request
        future = self._planner.call_async(plan_request)
        # MoveIt/OMPL 在多次 planning_attempts 时，实际耗时可能接近
        # planning_time * planning_attempts。之前这里固定等 10s，会在
        # move_group 仍然规划时误判为 "no response"，随后重复发新请求，
        # 让 move_group 积压更多规划任务。这里按当前全局规划参数计算等待时间。
        planner_response_timeout = max(
            10.0,
            float(self.planning_time) * float(self.planning_attempts) + 5.0,
        )
        self._wait_future(future, planner_response_timeout)

        response = future.result()
        if response is None:
            self._logger.error(
                f"MoveIt planner: no response after {planner_response_timeout:.1f}s"
            )
            return False

        plan = response.motion_plan_response
        if plan.error_code.val != MoveItErrorCodes.SUCCESS:
            self._logger.error(f"MoveIt planner: failed, code={plan.error_code.val}")
            return False

        # ── 3. 修正轨迹中的角度跳变 ──
        self._fix_trajectory_continuity(plan.trajectory.joint_trajectory)
        if not self._trajectory_within_joint_limits(plan.trajectory.joint_trajectory):
            return False

        # ── 4. 执行轨迹 ──
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = plan.trajectory
        future = self._executor.send_goal_async(goal)
        self._wait_future(future, 5.0)

        handle = future.result()
        if handle is None or not handle.accepted:
            self._logger.error("execute_trajectory: goal rejected")
            return False

        result_future = handle.get_result_async()
        self._wait_future(result_future, 30.0)
        if not result_future.done():
            self._logger.error("execute_trajectory: timeout")
            return False

        result = result_future.result().result
        ok = result.error_code.val == MoveItErrorCodes.SUCCESS
        self._logger.info(f"ExecuteTrajectory result={result.error_code.val}, success={ok}")
        return ok

    def _fix_trajectory_continuity(self, trajectory: JointTrajectory) -> None:
        """修正轨迹点中的角度跳变，使相邻点角度变化不超过 π（最近等效角）。

        防止机械臂从 3.1 rad 跳到 -3.0 rad 时发生整圈旋转。
        """
        if not trajectory.points:
            return

        arm_indexes = [
            trajectory.joint_names.index(name)
            for name in ARM_JOINTS
            if name in trajectory.joint_names
        ]
        if not arm_indexes:
            return

        # 以当前关节角度为起始参考
        reference_map = dict(zip(ARM_JOINTS, self.get_current_joints()))
        reference = [reference_map[trajectory.joint_names[idx]] for idx in arm_indexes]
        for point in trajectory.points:
            positions = list(point.positions)
            current = [positions[idx] for idx in arm_indexes]
            continuous = normalize_joint_values(
                current,
                reference,
                [trajectory.joint_names[idx] for idx in arm_indexes],
            )
            for idx, val in zip(arm_indexes, continuous):
                positions[idx] = val
            point.positions = positions
            reference = continuous
