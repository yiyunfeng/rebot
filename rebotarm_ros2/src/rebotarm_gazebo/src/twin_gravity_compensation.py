"""Twin 模式专用的重力补偿守护节点。

这个节点只做三件事：
1. enable 真机控制器；
2. 启动 controller 内部重力补偿；
3. 节点退出时停止重力补偿，并按参数决定是否 safe_home / disable。

默认会 safe_home，和 rebotarmcontroller 的 GravityCompensation 示例行为更接近；
默认也会 disable，关闭 twin 后真机回到安全失能状态。
"""

from __future__ import annotations

import signal

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_srvs.srv import Trigger


class TwinGravityCompensation(Node):
    """启动并维持真机重力补偿，供 Gazebo twin 镜像使用。"""

    def __init__(self) -> None:
        super().__init__("twin_gravity_compensation")
        self.declare_parameter("namespace", "rebotarm")
        self.declare_parameter("safe_home_on_exit", True)
        self.declare_parameter("disable_on_exit", True)
        namespace = str(self.get_parameter("namespace").value).strip("/")

        self._enable = self.create_client(Trigger, f"/{namespace}/enable")
        self._start = self.create_client(
            Trigger, f"/{namespace}/gravity_compensation/start"
        )
        self._stop = self.create_client(
            Trigger, f"/{namespace}/gravity_compensation/stop"
        )
        self._safe_home = self.create_client(Trigger, f"/{namespace}/safe_home")
        self._disable = self.create_client(Trigger, f"/{namespace}/disable")
        self._started = False
        self._stop_requested = False
        self._safe_home_on_exit = bool(self.get_parameter("safe_home_on_exit").value)
        self._disable_on_exit = bool(self.get_parameter("disable_on_exit").value)

    def run(self) -> bool:
        if not self._call(self._enable, "enable"):
            return False
        if not self._call(self._start, "start gravity compensation"):
            return False

        self._started = True
        self.get_logger().info(
            "twin gravity compensation is active; Gazebo will mirror joint_states"
        )
        while rclpy.ok() and not self._stop_requested:
            rclpy.spin_once(self, timeout_sec=0.2)
        return True

    def request_stop(self) -> None:
        self._stop_requested = True

    def cleanup(self) -> None:
        if self._started:
            self._call(self._stop, "stop gravity compensation", timeout_sec=5.0)
            self._started = False
        if self._safe_home_on_exit:
            self._call(self._safe_home, "safe_home", timeout_sec=35.0)
        if self._disable_on_exit:
            self._call(self._disable, "disable", timeout_sec=5.0)

    def _call(self, client, label: str, timeout_sec: float = 10.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f"{label} service not available")
            return False

        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            self.get_logger().error(f"{label} timed out")
            return False

        result = future.result()
        if result is None or not result.success:
            message = result.message if result is not None else "no response"
            self.get_logger().error(f"{label} failed: {message}")
            return False

        message = result.message or f"{label} OK"
        self.get_logger().info(message)
        return True


def main(args=None) -> None:
    del args
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = TwinGravityCompensation()

    def _signal_handler(_signum, _frame) -> None:
        node.request_stop()

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    ok = False
    try:
        ok = node.run()
    finally:
        node.cleanup()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
