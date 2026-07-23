#!/usr/bin/env python3
"""
move_to_pose 示例 — 笛卡尔空间末端位姿移动
============================================

通过 MoveToPose Action 将机械臂末端移动到指定位姿。

**参数**：
  --namespace  命名空间（默认 rebotarm）
  --x/y/z      目标位置(米)，默认 0.30/0.0/0.30
  --qx/qy/qz/qw 目标姿态四元数，默认 0,0,0,1（水平握持）
  --duration   运动时长(秒)，默认 2.0
"""

from __future__ import annotations

import argparse
import time

import rclpy
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rebotarm_msgs.action import MoveToPose
from sensor_msgs.msg import JointState


class DemoMoveToPose(Node):
    """
    MoveToPose 演示节点。

    流程：等待 joint_states → 等待 Action Server → 构造 Pose Goal → 发送 → 等结果
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("move_to_pose")
        self._namespace = args.namespace.strip("/")
        self._target = args
        self._latest_joint_state: JointState | None = None

        # 订阅 joint_states 用于确认控制器在线
        self.create_subscription(
            JointState, f"/{self._namespace}/joint_states",
            self._joint_state_cb, qos_profile_sensor_data,
        )
        self._move_to_pose = ActionClient(
            self, MoveToPose, f"/{self._namespace}/move_to_pose",
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def run(self) -> bool:
        """执行 MoveToPose：等待 → 构造 Goal → 发送 → 等结果。"""
        if not self._wait_for_joint_state():
            self.get_logger().error("joint_states not available")
            return False
        if not self._move_to_pose.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("move_to_pose action not available")
            return False

        # 构造 Goal：Pose(位置+四元数) + duration
        goal = MoveToPose.Goal()
        goal.target_pose = Pose()
        goal.target_pose.position.x = float(self._target.x)
        goal.target_pose.position.y = float(self._target.y)
        goal.target_pose.position.z = float(self._target.z)
        goal.target_pose.orientation.x = float(self._target.qx)
        goal.target_pose.orientation.y = float(self._target.qy)
        goal.target_pose.orientation.z = float(self._target.qz)
        goal.target_pose.orientation.w = float(self._target.qw)
        goal.duration = float(self._target.duration)

        # 异步发送 → 等待 Goal 被接收
        send_future = self._move_to_pose.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("goal rejected")
            return False

        # 等待执行结果
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info(f"success={result.success} message={result.message}")
        return bool(result.success)

    def _wait_for_joint_state(self, timeout_sec: float = 5.0) -> bool:
        """10Hz 轮询等待首个 joint_state 消息到达。"""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and self._latest_joint_state is None:
            if time.monotonic() > deadline:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._latest_joint_state is not None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="rebotarm")
    parser.add_argument("--x", type=float, default=0.30)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.30)
    parser.add_argument("--qx", type=float, default=0.0)
    parser.add_argument("--qy", type=float, default=0.0)
    parser.add_argument("--qz", type=float, default=0.0)
    parser.add_argument("--qw", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init()
    node = DemoMoveToPose(args)
    try:
        ok = node.run()
    except Exception as exc:
        node.get_logger().error(str(exc))
        ok = False
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
