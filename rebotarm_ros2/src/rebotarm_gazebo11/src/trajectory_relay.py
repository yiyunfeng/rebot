"""
轨迹中继节点：将 MoveIt 的轨迹指令转发给硬件机械臂。

用途：在 "gazebo_to_hardware" 模式下使用。
MoveIt 在 Gazebo 仿真中规划轨迹，但执行命令需要发送给真实的硬件机械臂。
这个节点充当"中转站"：

    MoveIt → [trajectory_relay] → 硬件机械臂

核心概念 — Action 中继：
    Action 是 ROS 2 中"带反馈的异步任务"机制。本节点同时作为:
    - Action Server（对 MoveIt）：接收轨迹执行请求
    - Action Client（对硬件）：将请求转发给硬件控制器

夹爪的特别处理：
    仿真中夹爪使用 FollowJointTrajectory（两个 prismatic joint 各发一个位置值），
    但硬件夹爪使用 GripperCommand（单一 position 值表示开合程度）。
    本节点负责将仿真格式自动转换为硬件格式。
"""

from __future__ import annotations

import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class TrajectoryRelay(Node):
    """MoveIt 与硬件之间的轨迹中继节点。

    中继两种 Action：
    - arm:  FollowJointTrajectory → FollowJointTrajectory（直接转发）
    - gripper: FollowJointTrajectory → GripperCommand（格式转换后转发）
    """

    def __init__(self) -> None:
        super().__init__("trajectory_relay")

        # === 参数声明 ===
        # arm 中继：输入和输出的 Action 名称
        self.declare_parameter(
            "arm_input_action", "/rebotarm_controller/follow_joint_trajectory"
        )
        self.declare_parameter(
            "arm_output_action", "/rebotarm/follow_joint_trajectory"
        )

        # gripper 中继：输入和输出的 Action 名称
        self.declare_parameter(
            "gripper_input_action", "/gripper_controller/follow_joint_trajectory"
        )
        self.declare_parameter("gripper_output_action", "/rebotarm/gripper/command")

        # 功能开关：可以只启 arm 或只启 gripper
        self.declare_parameter("enable_arm", True)
        self.declare_parameter("enable_gripper", True)

        # gripper 转换参数
        self.declare_parameter("max_gripper_width", 0.09)
        self.declare_parameter("hardware_open_gripper_position", -5.0)
        self.declare_parameter("hardware_closed_gripper_position", 0.0)
        self.declare_parameter("gripper_max_effort", 10.0)

        # === Action Client：连接到硬件（转发目标用） ===
        self._arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("arm_output_action").value),
        )
        self._gripper_client = ActionClient(
            self,
            GripperCommand,
            str(self.get_parameter("gripper_output_action").value),
        )

        # === Action Server：对 MoveIt 暴露的接口（接收目标） ===
        self._servers = []

        if bool(self.get_parameter("enable_arm").value):
            self._servers.append(
                ActionServer(
                    self,
                    FollowJointTrajectory,
                    str(self.get_parameter("arm_input_action").value),
                    execute_callback=self._execute_arm,
                    goal_callback=self._always_accept,
                    cancel_callback=self._always_accept_cancel,
                )
            )

        if bool(self.get_parameter("enable_gripper").value):
            self._servers.append(
                ActionServer(
                    self,
                    FollowJointTrajectory,
                    str(self.get_parameter("gripper_input_action").value),
                    execute_callback=self._execute_gripper,
                    goal_callback=self._always_accept,
                    cancel_callback=self._always_accept_cancel,
                )
            )

    # ------------------------------------------------------------------
    # Action 生命周期回调（总是接受，不做额外判断）
    # ------------------------------------------------------------------

    def _always_accept(self, _goal_request):
        """总是接受新的 Goal 请求。"""
        return GoalResponse.ACCEPT

    def _always_accept_cancel(self, _goal_handle):
        """总是接受取消请求。"""
        return CancelResponse.ACCEPT

    # ------------------------------------------------------------------
    # Arm 中继：直接转发轨迹（无格式转换）
    # ------------------------------------------------------------------

    async def _execute_arm(self, goal_handle):
        """Arm 轨迹中继：将 MoveIt 的轨迹直接转发给硬件。

        async 函数 = 异步执行。ROS 2 的 ActionServer 在后台调用它，
        不会阻塞其他操作。await 表示"等待这个操作完成，但不占着线程"。
        """
        result = FollowJointTrajectory.Result()

        # 1. 等待硬件 arm 控制器就绪（最多等 5 秒）
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "硬件 arm 控制器不可用"
            return result

        # 2. 将 MoveIt 的轨迹目标原样发给硬件
        send_future = self._arm_client.send_goal_async(goal_handle.request)
        hardware_goal = await send_future

        if hardware_goal is None or not hardware_goal.accepted:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "硬件 arm 控制器拒绝了目标"
            return result

        # 3. 等待硬件执行完成，将结果原样返回给 MoveIt
        hardware_result = await hardware_goal.get_result_async()
        result = hardware_result.result

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    # ------------------------------------------------------------------
    # Gripper 中继：格式转换 + 转发
    # ------------------------------------------------------------------

    async def _execute_gripper(self, goal_handle):
        """Gripper 轨迹中继：格式转换后转发给硬件。

        转换原因：
            仿真格式 = gripper_joint1 主控位置
            硬件格式 = 单一 position 值 + max_effort
        """
        result = FollowJointTrajectory.Result()

        # 1. 等待硬件 gripper 控制器就绪
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "硬件 gripper 控制器不可用"
            return result

        # 2. 格式转换：仿真轨迹 → 硬件位置值
        try:
            hardware_position = self._sim_to_hardware_gripper(
                goal_handle.request.trajectory
            )
        except ValueError as exc:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = str(exc)
            return result

        # 3. 构造 GripperCommand 目标
        goal = GripperCommand.Goal()
        goal.command.position = hardware_position
        goal.command.max_effort = float(
            self.get_parameter("gripper_max_effort").value
        )

        # 4. 发送给硬件并等待结果
        send_future = self._gripper_client.send_goal_async(goal)
        hardware_goal = await send_future

        if hardware_goal is None or not hardware_goal.accepted:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "硬件 gripper 控制器拒绝了目标"
            return result

        hardware_result = await hardware_goal.get_result_async()

        if hardware_result.result.reached_goal:
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            result.error_string = "硬件夹爪已到达目标位置"
        else:
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            result.error_string = "硬件夹爪未到达目标位置"
        return result

    # ------------------------------------------------------------------
    # 夹爪位置转换：仿真格式 → 硬件格式
    # ------------------------------------------------------------------

    def _sim_to_hardware_gripper(self, trajectory) -> float:
        """将仿真夹爪轨迹转换为硬件夹爪位置值。

        仿真格式:
            FollowJointTrajectory 中通常只有 gripper_joint1 主控关节。
            旧配置如果仍传入 gripper_joint1 + gripper_joint2，本函数也兼容。
            0 = 完全闭合, max_gripper_width/2 = 完全张开

        硬件格式:
            GripperCommand 中只有一个 position 值：
            - hardware_closed_position = 闭合
            - hardware_open_position = 张开

        转换步骤（4 步）:
            ① 取轨迹最后一个点的关节位置
            ② 计算夹爪主控位置（多关节输入时取平均值）
            ③ 归一化为 [0, 1]（0=闭合, 1=最开）
            ④ 线性映射到硬件范围 [closed, open]

        Args:
            trajectory: 仿真夹爪的 JointTrajectory 消息。

        Returns:
            硬件夹爪的目标位置值。

        Raises:
            ValueError: 轨迹数据为空或格式不正确。
        """
        # ① 取最后一个轨迹点
        if not trajectory.points:
            raise ValueError("夹爪轨迹为空（至少需要一个轨迹点）")
        final_positions = list(trajectory.points[-1].positions)
        if not final_positions:
            raise ValueError("夹爪轨迹最后一个点没有位置数据")

        # ② 计算左右夹爪的平均偏移
        sim_position = sum(float(p) for p in final_positions) / len(final_positions)

        # ③ 归一化为 [0, 1]
        # 每个 joint 偏移范围 = [0, max_width/2]
        # 归一化公式: ratio = 2 * sim_position / max_width
        max_width = float(self.get_parameter("max_gripper_width").value)
        if max_width <= 0.0:
            ratio = 0.0
        else:
            ratio = (2.0 * sim_position) / max_width
            ratio = max(0.0, min(1.0, ratio))  # 限幅在 [0, 1]

        # ④ 映射到硬件范围
        # 公式: hardware = closed + (open - closed) * ratio
        open_pos = float(
            self.get_parameter("hardware_open_gripper_position").value
        )
        closed_pos = float(
            self.get_parameter("hardware_closed_gripper_position").value
        )
        return closed_pos + (open_pos - closed_pos) * ratio


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryRelay()

    # 使用 MultiThreadedExecutor（4 线程）
    # 原因：async Action 回调需要并发执行，单线程 executor 会卡住
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
