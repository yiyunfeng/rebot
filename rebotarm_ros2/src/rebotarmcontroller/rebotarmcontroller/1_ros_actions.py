"""
ros_actions 模块 — 机械臂 ROS2 Action 服务端
=============================================

本模块将复杂、长时间运行的机械臂操作暴露为 ROS2 Action 接口。
与 Service（一次性请求-响应）不同，Action 支持：
  - 执行过程中的状态反馈（feedback）
  - 客户端取消请求（cancel）
  - 执行完成/中止/取消等终态

**提供的 Action 列表**：
  | Action 名                              | 类型                      | 功能                         |
  |----------------------------------------|---------------------------|------------------------------|
  | /{namespace}/move_to_pose              | MoveToPose                | 笛卡尔轨迹移动到目标位姿      |
  | /{namespace}/follow_joint_trajectory   | FollowJointTrajectory     | 关节空间轨迹跟踪              |
  | /{namespace}/gripper/command           | GripperCommand            | 夹爪开合控制（带失速检测）    |
"""

from __future__ import annotations

import time

import numpy as np
from control_msgs.action import FollowJointTrajectory, GripperCommand  # ROS 标准 action 类型
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rebotarm_msgs.action import MoveToPose  # 自定义 action 类型

from .conversions import pose_to_xyz_rpy


class ArmActions:
    """
    机械臂 Action 服务端 —— 封装 HardwareManager 操作到 Action 接口。

    三组回调：
      - goal_callback:   新目标到达时判断是否接受（仲裁门控）
      - execute_callback:执行目标（带反馈发布 + 取消轮询）
      - cancel_callback: 取消请求（始终 ACCEPT，实际取消在 execute 中轮询处理）
    """

    def __init__(self, node, hardware, namespace: str) -> None:
        """初始化三个 Action 服务端。"""
        self._node = node
        self._hardware = hardware
        self._namespace = namespace

        # ----- MoveToPose Action -----
        # 笛卡尔空间轨迹规划 + 执行
        self._move_to_pose_server = ActionServer(
            node, MoveToPose, f"/{namespace}/move_to_pose",
            execute_callback=self.execute_move_to_pose,
            goal_callback=self.arm_goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=node.reentrant_group,
        )
        # ----- FollowJointTrajectory Action -----
        # 关节空间在线轨迹插值 + 跟踪
        self._follow_joint_trajectory_server = ActionServer(
            node, FollowJointTrajectory, f"/{namespace}/follow_joint_trajectory",
            execute_callback=self.execute_follow_joint_trajectory,
            goal_callback=self.arm_goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=node.reentrant_group,
        )
        # ----- GripperCommand Action -----
        # 夹爪开合（带 position/effort 反馈 + 失速检测）
        self._gripper_command_server = ActionServer(
            node, GripperCommand, f"/{namespace}/gripper/command",
            execute_callback=self.execute_gripper_command,
            goal_callback=self.gripper_goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=node.reentrant_group,
        )

    # ══════════════════════════════════════════════════════════════════
    # Goal 门控回调 — 根据状态机决定是否接受新目标
    # ══════════════════════════════════════════════════════════════════

    def arm_goal_callback(self, _goal_request):
        """
        机械臂运动 goal 门控。
        拒绝条件：TRAJ_RUNNING（已有轨迹在运行，不允许嵌套）/
        GRAVITY_COMP（重力补偿中）/ SAFE_HOMING（安全回零中）。
        """
        return self._gate_goal(
            ("TRAJ_RUNNING", "GRAVITY_COMP", "SAFE_HOMING"), "arm motion"
        )

    def gripper_goal_callback(self, _goal_request):
        """
        夹爪 goal 门控。
        与机械臂不同：TRAJ_RUNNING 时允许夹爪操作（不阻塞），
        仅在重力补偿和安全回零时拒绝。
        """
        return self._gate_goal(("GRAVITY_COMP", "SAFE_HOMING"), "gripper")

    def _gate_goal(self, blocked, label):
        """核心门控逻辑：检查状态机，返回 ACCEPT 或 REJECT。"""
        state = self._hardware.state_machine
        if state in blocked:
            self._node.get_logger().warn(f"rejecting {label} goal in state {state}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        """取消请求：始终接受（实际取消在 execute 回调轮询 is_cancel_requested）。"""
        return CancelResponse.ACCEPT

    # ══════════════════════════════════════════════════════════════════
    # MoveToPose — 笛卡尔轨迹移动
    # ══════════════════════════════════════════════════════════════════

    def _fail_move_to_pose(self, goal_handle, result, message, *, canceled=False):
        """
        统一的 MoveToPose 失败/取消处理。
        - SAFE_HOMING 中不重置状态机（被外部抢占）
        - canceled=True → goal_handle.canceled()，否则 abort()
        - 附上当前末端位姿供客户端参考
        """
        if self._hardware.state_machine != "SAFE_HOMING":
            self._hardware.set_state_machine("IDLE")
            self._node.publish_arm_status()
        if canceled:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        result.success = False
        result.message = message
        result.final_pose = self._hardware.current_pose()
        return result

    def execute_move_to_pose(self, goal_handle):
        """
        执行 MoveToPose Action。

        流程：
          1. Pose → xyz+rpy 转换
          2. hardware.move_to_pose_traj() 轨迹规划
          3. 轮询等待轨迹完成（50Hz），期间检测：
             - SAFE_HOMING 抢占 → stop_motion
             - 客户端取消 → stop_motion + hold_current_position
             - 超时（duration + 2s 缓冲）→ stop_motion + hold_current_position
        """
        goal = goal_handle.request
        result = MoveToPose.Result()

        try:
            x, y, z, roll, pitch, yaw = pose_to_xyz_rpy(goal.target_pose)
            ok = self._hardware.move_to_pose_traj(
                x, y, z, roll, pitch, yaw, float(goal.duration)
            )
        except Exception as exc:
            self._hardware.hold_current_position()
            return self._fail_move_to_pose(goal_handle, result, str(exc))

        if not ok:
            return self._fail_move_to_pose(goal_handle, result, "trajectory planning failed")
        self._node.publish_arm_status()

        # 轮询等待轨迹完成（50Hz）
        deadline = time.monotonic() + max(float(goal.duration), 0.0) + 2.0
        while self._hardware.motion_active():
            if self._hardware.state_machine == "SAFE_HOMING":
                self._hardware.stop_motion()
                break
            if goal_handle.is_cancel_requested:
                self._hardware.stop_motion()
                self._hardware.hold_current_position()
                return self._fail_move_to_pose(
                    goal_handle, result, "move_to_pose canceled", canceled=True
                )
            if time.monotonic() > deadline:
                self._hardware.stop_motion()
                self._hardware.hold_current_position()
                return self._fail_move_to_pose(goal_handle, result, "move_to_pose timeout")
            time.sleep(0.02)

        if self._hardware.state_machine == "SAFE_HOMING":
            return self._fail_move_to_pose(
                goal_handle, result, "move_to_pose preempted by safe_home"
            )

        # 成功
        positions = self._hardware.get_joint_positions()
        velocities = self._hardware.get_joint_velocities()
        result.success = True
        result.message = (
            "move_to_traj accepted "
            f"positions={[float(v) for v in positions]} "
            f"velocities={[float(v) for v in velocities]}"
        )
        result.final_pose = self._hardware.current_pose()
        self._hardware.set_state_machine("IDLE")
        self._node.publish_arm_status()
        goal_handle.succeed()
        return result

    # ══════════════════════════════════════════════════════════════════
    # FollowJointTrajectory — 关节轨迹跟踪
    # ══════════════════════════════════════════════════════════════════

    def execute_follow_joint_trajectory(self, goal_handle):
        """
        执行 FollowJointTrajectory Action。

        参数校验（失败均 INVALID_GOAL）：
          1. joint_names 和 points 非空
          2. joint_names 顺序必须与 hardware.joint_names 完全一致
          3. 每个 point 的 positions 维度必须与关节数一致

        执行逻辑：
          1. begin_trajectory_stream() → TRAJ_RUNNING
          2. 逐段插值：
             - ratio = (now - t_start) / (t_end - t_start) ∈ [0, 1]
             - q_target = q_start + (q_end - q_start) * ratio
          3. 50Hz 发布 feedback（期望/实际/误差）
          4. 支持取消 和 SAFE_HOMING 抢占

        细节：如果第一个 points[0] 的时间 > 0，自动插入当前位置(t=0)
        作为起始点，避免机械臂从任意位置跳变到第一个目标。
        """
        goal = goal_handle.request
        result = FollowJointTrajectory.Result()
        trajectory = goal.trajectory
        joint_names = list(trajectory.joint_names)

        # ----- 参数校验 -----
        if not joint_names or not trajectory.points:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "trajectory must include joint_names and points"
            return result

        if joint_names != self._hardware.joint_names:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = (
                f"trajectory joint_names must be {self._hardware.joint_names}"
            )
            return result

        targets = [np.array(point.positions, dtype=np.float64) for point in trajectory.points]
        if any(len(target) != len(self._hardware.joint_names) for target in targets):
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "point positions length must match joint_names"
            return result

        try:
            # 进入轨迹模式
            self._hardware.begin_trajectory_stream()
            self._node.publish_arm_status()
            start = time.monotonic()
            point_times = [
                float(point.time_from_start.sec)
                + float(point.time_from_start.nanosec) * 1e-9
                for point in trajectory.points
            ]
            # 如果第一个点的时间戳 > 0，插入当前位置作为起始点（t=0）
            if point_times[0] > 0.0:
                targets.insert(0, self._hardware.get_joint_positions().copy())
                point_times.insert(0, 0.0)

            # 逐段线性插值
            for index in range(1, len(targets)):
                q0 = targets[index - 1]    # 段起始位置
                q1 = targets[index]        # 段结束位置
                t0 = point_times[index - 1]
                t1 = max(point_times[index], t0)

                while True:
                    # SAFE_HOMING 抢占
                    if self._hardware.state_machine == "SAFE_HOMING":
                        goal_handle.abort()
                        result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                        result.error_string = "follow_joint_trajectory preempted by safe_home"
                        return result

                    now = time.monotonic() - start
                    ratio = 1.0 if t1 <= t0 else max(0.0, min(1.0, (now - t0) / (t1 - t0)))
                    target = q0 + (q1 - q0) * ratio
                    self._hardware.set_joint_position_target(target)

                    # 发布 feedback（期望 vs 实际 vs 误差）
                    positions = self._hardware.get_joint_positions()
                    velocities = self._hardware.get_joint_velocities()
                    feedback = FollowJointTrajectory.Feedback()
                    feedback.header.stamp = self._node.get_clock().now().to_msg()
                    feedback.joint_names = self._hardware.joint_names
                    feedback.desired.positions = [float(v) for v in target]
                    feedback.actual.positions = [float(v) for v in positions]
                    feedback.actual.velocities = [float(v) for v in velocities]
                    feedback.error.positions = [float(v) for v in target - positions]
                    goal_handle.publish_feedback(feedback)

                    # 取消检测
                    if goal_handle.is_cancel_requested:
                        self._hardware.hold_current_position()
                        goal_handle.canceled()
                        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                        result.error_string = "follow_joint_trajectory canceled"
                        return result

                    if now >= t1:
                        break
                    time.sleep(0.02)  # 50Hz
        except Exception as exc:
            self._hardware.hold_current_position()
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            result.error_string = f"execution failed: {exc}"
            return result
        finally:
            if self._hardware.state_machine != "SAFE_HOMING":
                self._hardware.set_state_machine("IDLE")
                self._node.publish_arm_status()

        # 成功
        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        positions = self._hardware.get_joint_positions()
        velocities = self._hardware.get_joint_velocities()
        result.error_string = (
            "joint target accepted "
            f"positions={[float(v) for v in positions]} "
            f"velocities={[float(v) for v in velocities]}"
        )
        return result

    # ══════════════════════════════════════════════════════════════════
    # GripperCommand — 夹爪控制 Action
    # ══════════════════════════════════════════════════════════════════

    def execute_gripper_command(self, goal_handle):
        """
        执行夹爪控制 Action。

        流程：
          1. 发送目标位置 → hardware.set_gripper_target()
          2. 20Hz 轮询等待到达或失速（最大 5 秒）
          3. 发布 position/effort/stalled/reached_goal feedback

        失速检测（stalled）：位置变化 < 1e-4 且 effort >= max_effort
        """
        goal = goal_handle.request.command
        result = GripperCommand.Result()
        feedback = GripperCommand.Feedback()

        # 发送目标
        try:
            self._hardware.set_gripper_target(goal.position)
        except Exception:
            goal_handle.abort()
            result.position = 0.0
            result.effort = 0.0
            result.stalled = False
            result.reached_goal = False
            return result

        # 轮询等待（20Hz，最大 5 秒）
        start = time.monotonic()
        last_pos = self._hardware.get_gripper_state()[0]
        stalled = False
        while time.monotonic() - start < 5.0:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                pos, _, effort, _ = self._hardware.get_gripper_state()
                result.position = pos
                result.effort = effort
                result.stalled = stalled
                result.reached_goal = False
                return result

            pos, _, effort, _ = self._hardware.get_gripper_state()
            reached = self._hardware.gripper_reached_target()
            # 失速判定：位置无明显变化 且 力矩达到上限
            stalled = abs(pos - last_pos) < 1e-4 and abs(effort) >= float(goal.max_effort)
            feedback.position = pos
            feedback.effort = effort
            feedback.stalled = stalled
            feedback.reached_goal = reached
            goal_handle.publish_feedback(feedback)
            if reached:
                break
            last_pos = pos
            time.sleep(0.05)

        pos, _, effort, _ = self._hardware.get_gripper_state()
        result.position = pos
        result.effort = effort
        result.stalled = stalled
        result.reached_goal = self._hardware.gripper_reached_target()
        goal_handle.succeed()
        return result
