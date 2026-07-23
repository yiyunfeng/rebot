"""
关节状态镜像节点：将硬件机械臂的关节状态同步到 Gazebo 仿真控制器。

用途：在 "twin"（数字孪生）或 "gazebo_to_hardware" 模式下使用。
硬件机械臂通过 /rebotarm/joint_states 发布真实的关节角度，
本节点将这些角度"镜像"到 Gazebo 中的虚拟机械臂控制器，
让仿真中的机械臂跟随真实机械臂同步运动。

工作原理：
    1. 订阅 /rebotarm/joint_states 话题（硬件发布的关节状态）
    2. 将关节角度拆分为 arm（关节1-6）和 gripper（夹爪）两组
    3. 分别发布到 Gazebo 中对应的 joint_trajectory 话题
    4. 做频率限制，避免发布过快
"""

from __future__ import annotations

import time

from builtin_interfaces.msg import Duration
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class JointStateMirror(Node):
    """将硬件关节状态镜像到 Gazebo 控制器的节点。"""

    def __init__(self) -> None:
        super().__init__("joint_state_mirror")

        # --- 可配置参数 ---
        self.declare_parameter("source_joint_states", "/rebotarm/joint_states")
        self.declare_parameter(
            "arm_command_topic", "/gazebo_rebotarm_controller/joint_trajectory"
        )
        self.declare_parameter(
            "gripper_command_topic", "/gazebo_gripper_controller/joint_trajectory"
        )
        self.declare_parameter(
            "arm_joint_names",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        )
        self.declare_parameter("gripper_joint_names", ["gripper_joint1", "gripper_joint2"])
        self.declare_parameter("publish_rate", 20.0)   # 最大发布频率（Hz）
        self.declare_parameter("point_duration", 0.1)  # 每个轨迹点的时长（秒）

        # 读取关节名称列表
        self._arm_joints = [
            str(name) for name in self.get_parameter("arm_joint_names").value
        ]
        self._gripper_joints = [
            str(name) for name in self.get_parameter("gripper_joint_names").value
        ]

        # 频率限制：两次发布之间至少间隔 min_period 秒
        publish_rate = max(float(self.get_parameter("publish_rate").value), 1.0)
        self._min_period = 1.0 / publish_rate
        self._point_duration = max(
            float(self.get_parameter("point_duration").value), 0.02
        )
        self._last_publish_time = 0.0

        # --- 发布者：分别向 arm 和 gripper 控制器发送轨迹命令 ---
        self._arm_pub = self.create_publisher(
            JointTrajectory,
            str(self.get_parameter("arm_command_topic").value),
            10,
        )
        self._gripper_pub = self.create_publisher(
            JointTrajectory,
            str(self.get_parameter("gripper_command_topic").value),
            10,
        )

        # --- 订阅者：接收硬件发布的关节状态 ---
        self.create_subscription(
            JointState,
            str(self.get_parameter("source_joint_states").value),
            self._on_joint_state,
            qos_profile_sensor_data,
        )

    def _on_joint_state(self, msg: JointState) -> None:
        """收到关节状态消息时的回调：做频率限制后分两组发布。"""
        now = time.monotonic()
        # 频率限制：距离上次发布不足 min_period 就跳过
        if now - self._last_publish_time < self._min_period:
            return
        self._last_publish_time = now

        # 将关节名称和角度值配对成字典 {"joint1": 0.5, "joint2": -1.2, ...}
        positions = dict(zip(msg.name, msg.position))

        # 分别发布 arm 和 gripper 的轨迹命令
        self._publish_trajectory(self._arm_pub, self._arm_joints, positions)
        self._publish_trajectory(self._gripper_pub, self._gripper_joints, positions)

    def _publish_trajectory(
        self,
        publisher,
        joint_names: list[str],
        positions: dict[str, float],
    ) -> None:
        """将一组关节的目标位置发布为 JointTrajectory 消息。

        Args:
            publisher: ROS 2 发布者。
            joint_names: 要发布的关节名列表。
            positions: 所有关节的 {名称: 角度} 字典。
        """
        # 确保所有需要的关节都有值
        if not joint_names or not all(name in positions for name in joint_names):
            return

        # 构造 JointTrajectory 消息
        trajectory = JointTrajectory()
        trajectory.joint_names = list(joint_names)
        trajectory.header.stamp = self.get_clock().now().to_msg()

        point = JointTrajectoryPoint()
        point.positions = [float(positions[name]) for name in joint_names]

        # 设置轨迹点的到达时间
        # Duration 由秒 + 纳秒两部分组成
        duration_sec = int(self._point_duration)
        duration_nsec = int((self._point_duration - duration_sec) * 1e9)
        point.time_from_start = Duration(sec=duration_sec, nanosec=duration_nsec)

        trajectory.points = [point]
        publisher.publish(trajectory)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointStateMirror()
    try:
        rclpy.spin(node)  # 让节点持续运行，等待消息
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
