#!/usr/bin/env python3
"""
move_to 示例 — 关节空间轨迹移动
================================

通过 FollowJointTrajectory Action 在关节空间移动机械臂。

**两种模式**：
  模式 A: `move_to.py 0.0 0.0 0.0 0.0 0.0 0.0 --duration 3.0`（全关节）
  模式 B: `move_to.py --joint joint3 --position 0.5 --duration 1.0`（单关节）
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

_NAMESPACE = "rebotarm"


def _duration_msg(seconds: float) -> Duration:
    """浮点秒数 → ROS2 Duration(sec + nanosec)。例 2.5s → sec=2, nanosec=500_000_000。"""
    sec = int(seconds)
    nanosec = int((float(seconds) - sec) * 1e9)
    return Duration(sec=sec, nanosec=nanosec)


class DemoMoveTo(Node):
    """关节空间移动演示节点。"""

    def __init__(
        self,
        target_positions: list[float],
        joint_name: str | None,
        joint_position: float | None,
        duration: float,
    ) -> None:
        super().__init__("move_to")
        self._target_positions = target_positions
        self._joint_name = joint_name
        self._joint_position = joint_position
        self._duration = max(float(duration), 0.1)  # 最小 0.1s
        self._latest_joint_state: JointState | None = None
        self._last_feedback_log = 0.0  # feedback 日志限流

        self.create_subscription(
            JointState, f"/{_NAMESPACE}/joint_states",
            self._joint_state_cb, qos_profile_sensor_data,
        )
        self._follow_joint_trajectory = ActionClient(
            self, FollowJointTrajectory, f"/{_NAMESPACE}/follow_joint_trajectory",
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def run(self) -> bool:
        """等待 joint_state → 等待 Server → 构造单点轨迹 → 发送 → 等结果。"""
        if not self._wait_for_joint_state():
            self.get_logger().error("joint_states not available")
            return False
        if not self._follow_joint_trajectory.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("follow_joint_trajectory action not available")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = self._make_trajectory()

        send_future = self._follow_joint_trajectory.send_goal_async(
            goal, feedback_callback=self._feedback_cb,
        )
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info(f"error_code={result.error_code} message={result.error_string}")
        return result.error_code == FollowJointTrajectory.Result.SUCCESSFUL

    def _wait_for_joint_state(self, timeout_sec: float = 5.0) -> bool:
        """10Hz 轮询等待首个 joint_state 消息。"""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and self._latest_joint_state is None:
            if time.monotonic() > deadline:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._latest_joint_state is not None

    def _make_trajectory(self) -> JointTrajectory:
        """
        构造单点关节轨迹。

        策略：从 joint_state 取当前位置 → 解析目标位置 →
             创建单个 point（time_from_start = duration，控制器线性插值）。
        gripper_ 前缀的关节名将被过滤（不属于机械臂关节）。
        """
        assert self._latest_joint_state is not None
        joint_names, current = self._arm_joint_state()
        target = self._resolve_target(joint_names, current)

        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in target]
        point.time_from_start = _duration_msg(self._duration)
        trajectory.points = [point]

        self.get_logger().info(
            "moving joints to "
            + ", ".join(f"{name}={value:+.3f}" for name, value in zip(joint_names, target))
        )
        return trajectory

    def _arm_joint_state(self) -> tuple[list[str], np.ndarray]:
        """从 JointState 提取机械臂关节（过滤 gripper_ 虚拟关节）。"""
        assert self._latest_joint_state is not None
        pairs = [
            (name, position)
            for name, position in zip(self._latest_joint_state.name, self._latest_joint_state.position)
            if not name.startswith("gripper_")
        ]
        return [name for name, _ in pairs], np.array([p for _, p in pairs], dtype=np.float64)

    def _resolve_target(self, joint_names: list[str], current: np.ndarray) -> np.ndarray:
        """
        根据参数决定目标位置向量。
        模式 A: target_positions 长度必须匹配
        模式 B: 复制当前位置，替换指定关节目标
        交叉错误检测：全关节位和 --joint 互斥、--joint 需要 --position、未知关节等。
        """
        assert self._latest_joint_state is not None
        if self._joint_name is not None:
            if self._target_positions:
                raise ValueError("use either 6 joint positions or --joint/--position")
            if self._joint_position is None:
                raise ValueError("--joint requires --position")
            if self._joint_name not in joint_names:
                raise ValueError(f"unknown joint: {self._joint_name}")
            target = current.copy()
            target[joint_names.index(self._joint_name)] = float(self._joint_position)
            return target
        if self._joint_position is not None:
            raise ValueError("--position requires --joint")
        if len(self._target_positions) != len(current):
            raise ValueError(f"expected {len(current)} absolute joint positions, got {len(self._target_positions)}")
        return np.array(self._target_positions, dtype=np.float64)

    def _feedback_cb(self, feedback_msg) -> None:
        """Feedback 回调：每 0.5s 最多输出一条日志（避免 50Hz 刷屏）。"""
        now = time.monotonic()
        if now - self._last_feedback_log < 0.5:
            return
        self._last_feedback_log = now
        actual = feedback_msg.feedback.actual.positions
        if actual:
            self.get_logger().info("actual=" + ", ".join(f"{v:+.3f}" for v in actual))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positions", nargs="*", type=float, help="6-axis joint target positions (radians)")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--joint", help="single joint name (e.g. joint3)")
    parser.add_argument("--position", type=float, help="target position for --joint (radians)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init()
    node = DemoMoveTo(args.positions, args.joint, args.position, args.duration)
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
