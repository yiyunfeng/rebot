"""将 Gazebo 机械臂移动到 YAML 中的命名关节姿态。"""

from __future__ import annotations

import sys

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rebotarm_gazebo.real_controller import (
    ARM_JOINTS,
    JOINT_LIMIT_EPS,
    JOINT_POSITION_LIMITS,
)


class JointPoseCommander(Node):
    def __init__(self) -> None:
        super().__init__("joint_pose_commander")

        self.declare_parameter("pose_name", "table_view")
        self.declare_parameter("joint_names", list(ARM_JOINTS))
        self.declare_parameter(
            "table_view",
            [-0.0000084715, -0.4461844044, -0.7145121820, 0.9683351239, -0.0000088457, 0.0000936787],
        )
        self.declare_parameter("command_action", "/rebotarm_controller/follow_joint_trajectory")
        self.declare_parameter("move_duration", 3.0)
        self._client = ActionClient(
            self, FollowJointTrajectory, str(self.get_parameter("command_action").value)
        )

    def run(self) -> bool:
        pose_name = str(self.get_parameter("pose_name").value)
        joint_names = [str(v) for v in self.get_parameter("joint_names").value]
        target = [float(v) for v in self.get_parameter(pose_name).value]
        duration = float(self.get_parameter("move_duration").value)

        if len(joint_names) != len(target) or duration <= 0.0:
            self.get_logger().error("关节数量不一致或 move_duration 非法")
            return False
        for name, value in zip(joint_names, target):
            lower, upper = JOINT_POSITION_LIMITS[name]
            if not lower - JOINT_LIMIT_EPS <= value <= upper + JOINT_LIMIT_EPS:
                self.get_logger().error(f"{name}={value:.4f} 超出限制 [{lower:.4f}, {upper:.4f}]")
                return False
        if not self._client.wait_for_server(timeout_sec=8.0):
            self.get_logger().error("FollowJointTrajectory action 不可用")
            return False

        point = JointTrajectoryPoint(positions=target)
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int(duration % 1 * 1_000_000_000)
        goal = FollowJointTrajectory.Goal(
            trajectory=JointTrajectory(joint_names=joint_names, points=[point])
        )
        self.get_logger().info(f"移动到 {pose_name}: {[round(v, 4) for v in target]}")
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("轨迹 goal 被拒绝")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration + 5.0)
        result = result_future.result()
        ok = result is not None and result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        self.get_logger().info("姿态执行完成" if ok else "姿态执行失败或超时")
        return ok


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = JointPoseCommander()
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
