#!/usr/bin/env python3
"""
gravity_compensation 示例 — 在 ROS 控制器上启动重力补偿模式
===========================================================

本示例演示如何通过 ROS2 Service 调用控制器的重力补偿功能。

**操作流程**：
  1. 调用 /rebotarm/enable 使能机械臂
  2. 调用 /rebotarm/gravity_compensation/start 启动重力补偿
  3. 等待用户 Ctrl+C 停止
  4. 安全清理：safe_home → disable

**依赖的服务**（均由 reBotArmController 节点提供）：
  - /rebotarm/enable                       (std_srvs/Trigger)
  - /rebotarm/gravity_compensation/start   (std_srvs/Trigger)
  - /rebotarm/safe_home                    (std_srvs/Trigger)
  - /rebotarm/disable                      (std_srvs/Trigger)
"""

from __future__ import annotations

import signal

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions  # 控制 ROS2 自身信号处理行为
from std_srvs.srv import Trigger  # 标准触发服务

_NAMESPACE = "rebotarm"


def _call_trigger(
    node: Node,
    client,
    label: str,
    timeout_sec: float = 5.0,
) -> bool:
    """
    调用 Trigger 类型 Service 的通用辅助函数。

    流程：wait_for_service(5s) → call_async → spin_until_future_complete → 检查结果
    """
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error(f"{label} service not available")
        return False
    future = client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done():
        node.get_logger().error(f"{label} timed out")
        return False
    result = future.result()
    if result is None or not result.success:
        message = result.message if result is not None else "no response"
        node.get_logger().error(f"{label} failed: {message}")
        return False
    node.get_logger().info(message if (message := result.message) else f"{label} OK")
    return True


def main() -> None:
    """
    主函数：启动重力补偿 + 等待用户中断 + 安全清理。

    信号处理：
      - SignalHandlerOptions.NO 禁止 rclpy 默认 SIGINT，由本脚本自行管理
      - 注册 SIGINT/SIGTERM → request_stop 设置标志位
      - 主循环检测 stop_requested 退出
      - finally 块保证 safe_home(超时35s) → disable 始终执行
    """
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("gravity_compensation")

    # ----- 自定义信号处理 -----
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        """信号 handler：设置停止标志（幂等）。"""
        nonlocal stop_requested
        if not stop_requested:
            node.get_logger().info("stop requested, shutting down gravity compensation")
        stop_requested = True

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    # ----- 创建 Service 客户端 -----
    enable_client = node.create_client(Trigger, f"/{_NAMESPACE}/enable")
    start_client = node.create_client(Trigger, f"/{_NAMESPACE}/gravity_compensation/start")
    safe_home_client = node.create_client(Trigger, f"/{_NAMESPACE}/safe_home")
    disable_client = node.create_client(Trigger, f"/{_NAMESPACE}/disable")

    gc_started = False  # 标记重力补偿是否已启动
    try:
        if not _call_trigger(node, enable_client, "enable"):
            raise SystemExit(1)
        if not _call_trigger(node, start_client, "start gravity compensation"):
            raise SystemExit(1)
        gc_started = True

        node.get_logger().info("press Ctrl+C to stop gravity compensation")
        while rclpy.ok() and not stop_requested:
            rclpy.spin_once(node, timeout_sec=0.2)  # 200ms 轮询，避免忙等
    finally:
        # ----- 清理（safe_home 超时 35s 因回零可能较慢）-----
        if gc_started:
            try:
                _call_trigger(node, safe_home_client, "safe_home", timeout_sec=35.0)
            except Exception as exc:
                node.get_logger().warn(f"safe_home cleanup failed: {exc}")
            try:
                _call_trigger(node, disable_client, "disable")
            except Exception as exc:
                node.get_logger().warn(f"disable cleanup failed: {exc}")
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
