#!/usr/bin/env python3
"""读取 default.yaml 的 robot.ready_pose，通过 ROS2 MoveToPose Action 移动真机。"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rebotarm_msgs.action import MoveToPose
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """将固定轴 RPY（弧度）转换为 ROS 四元数 x/y/z/w。"""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class ReadyPoseMover(Node):
    """等待真机关节状态后发送一次笛卡尔 ready_pose。"""

    def __init__(self, namespace: str, ready: dict) -> None:
        """创建关节状态订阅、使能服务客户端和位姿 Action 客户端。"""
        super().__init__("rebot_grasp_ready_pose_mover")
        self._namespace = namespace.strip("/")
        self._ready = ready
        self._has_joint_state = False
        self.create_subscription(
            JointState,
            f"/{self._namespace}/joint_states",
            self._joint_state_cb,
            qos_profile_sensor_data,
        )
        self._client = ActionClient(self, MoveToPose, f"/{self._namespace}/move_to_pose")
        self._enable_client = self.create_client(Trigger, f"/{self._namespace}/enable")

    def _joint_state_cb(self, _msg: JointState) -> None:
        """收到任意关节状态即表示硬件反馈链路已经就绪。"""
        self._has_joint_state = True

    def run(self) -> bool:
        """依次等待反馈、使能机械臂、发送 ready_pose，并等待执行结果。"""
        # 先等 joint_states，确认驱动节点、硬件反馈和 namespace 都正确。
        deadline = time.monotonic() + 10.0
        while rclpy.ok() and not self._has_joint_state and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self._has_joint_state:
            self.get_logger().error(f"/{self._namespace}/joint_states 未就绪")
            return False
        # Action 和 enable service 分别检查，报错时能明确是哪条 ROS 2 通道未就绪。
        if not self._client.wait_for_server(timeout_sec=8.0):
            self.get_logger().error(f"/{self._namespace}/move_to_pose Action 未就绪")
            return False
        if not self._enable_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"/{self._namespace}/enable 服务未就绪")
            return False
        # 服务调用是异步的，这里 spin 当前节点直到收到响应或超时。
        enable_future = self._enable_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, enable_future, timeout_sec=10.0)
        enable_response = enable_future.result()
        if enable_response is None or not enable_response.success:
            message = enable_response.message if enable_response is not None else "no response"
            self.get_logger().error(f"真机 enable 失败: {message}")
            return False

        x, y, z = (float(self._ready[name]) for name in ("x", "y", "z"))
        roll, pitch, yaw = (
            float(self._ready.get(name, 0.0)) for name in ("roll", "pitch", "yaw")
        )
        duration = float(self._ready.get("duration", 3.0))
        if duration <= 0.0:
            raise ValueError("robot.ready_pose.duration 必须大于 0")

        # ROS Pose 使用四元数而配置使用 RPY，发送 Action 前完成格式转换。
        qx, qy, qz, qw = quaternion_from_rpy(roll, pitch, yaw)
        goal = MoveToPose.Goal(
            target_pose=Pose(
                position=Point(x=x, y=y, z=z),
                orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
            ),
        )
        goal.duration = duration
        self.get_logger().info(
            f"发送笛卡尔 ready_pose: xyz=({x:.3f}, {y:.3f}, {z:.3f}), "
            f"rpy=({roll:.3f}, {pitch:.3f}, {yaw:.3f}), duration={duration:.1f}s"
        )

        # 先等待服务端接受目标，再单独等待动作执行结果。
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=8.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("ready_pose goal 被拒绝")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration + 15.0)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            self.get_logger().error("ready_pose 执行超时")
            return False
        result = wrapped_result.result
        if not result.success:
            self.get_logger().error(f"ready_pose 执行失败: {result.message}")
            return False
        self.get_logger().info(f"ready_pose 执行完成: {result.message}")
        return True


def main() -> None:
    """读取配置并管理 ROS 2 节点的创建、运行和销毁。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--namespace", default="rebotarm")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    ready = (config.get("robot") or {}).get("ready_pose")
    if not isinstance(ready, dict):
        raise SystemExit("[ERROR] 配置中缺少 robot.ready_pose")
    missing = [name for name in ("x", "y", "z") if name not in ready]
    if missing:
        raise SystemExit(f"[ERROR] robot.ready_pose 缺少字段: {', '.join(missing)}")

    # rclpy 资源无论成功失败都在 finally 中销毁，避免节点残留。
    rclpy.init()
    node = ReadyPoseMover(args.namespace, ready)
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
