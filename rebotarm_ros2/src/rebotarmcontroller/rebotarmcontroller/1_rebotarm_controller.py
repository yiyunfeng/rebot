"""
rebotarm_controller 模块 — reBotArm 控制器主节点
================================================

ROS2 主节点，负责：
  1. 参数声明与解析（硬件配置、命名空间、仲裁模式等）
  2. 依赖注入——组装 HardwareManager → Publisher / Service / Action / MotorPassthrough
  3. 生命周期管理——启动 MultiThreadedExecutor → 关机时 safe_home + 清理

**模块依赖图**：
  reBotArmController
    ├── HardwareManager      (硬件抽象层)
    ├── JointStatePublisher   (状态发布)
    ├── ArmServices           (Service 接口)
    ├── ArmActions            (Action 接口)
    └── MotorPassthrough      (直通指令转发)
"""

from __future__ import annotations

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # 传感器数据 QoS 预设

from .hardware_manager import HardwareManager
from .motor_passthrough import MotorPassthrough
from .ros_actions import ArmActions
from .ros_publishers import JointStatePublisher
from .ros_services import ArmServices


class reBotArmController(Node):
    """
    reBotArm 控制器 ROS2 Node。

    架构设计：
      - 使用两个回调组分离调度策略
        - reentrant_group: 可重入（多线程并发），用于实时指令和控制回路
        - slow_group:      互斥（单线程串行），用于 enable/disable/零点等慢速危险操作
      - MultiThreadedExecutor(4) 驱动多线程回调
      - 所有子模块通过依赖注入组装（构造函数接收 node + hardware 引用）
    """

    def __init__(self) -> None:
        super().__init__("reBotArmController")

        # ═════════════════════════════════════════════════════════════
        # 回调组配置
        # ═════════════════════════════════════════════════════════════
        self.reentrant_group = ReentrantCallbackGroup()
        self.slow_group = MutuallyExclusiveCallbackGroup()
        self.sensor_qos = qos_profile_sensor_data

        # ═════════════════════════════════════════════════════════════
        # 参数声明（ROS2 parameter API）
        # ═════════════════════════════════════════════════════════════
        self.declare_parameter("hardware_config", "")          # 硬件配置文件路径
        self.declare_parameter("model", "")                    # 机械臂型号（如 "dm"）
        self.declare_parameter("channel", "")                  # 通信通道
        self.declare_parameter("joint_state_rate", 100.0)      # 状态发布频率（Hz）
        self.declare_parameter("arm_namespace", "rebotarm")   # 命名空间
        self.declare_parameter("cmd_arbitration", "reject")   # 直通指令仲裁模式
        self.declare_parameter("frame_id", "base_link")       # TF 基座系
        self.declare_parameter("ee_frame_id", "end_link")     # TF 末端系
        self.declare_parameter("disable_after_safe_home", True)

        # ═════════════════════════════════════════════════════════════
        # 参数解析 + 校验
        # ═════════════════════════════════════════════════════════════
        hardware_config = self.get_parameter("hardware_config").value or None
        model = str(self.get_parameter("model").value or "")
        channel = str(self.get_parameter("channel").value or "")
        self.arm_namespace = str(
            self.get_parameter("arm_namespace").value or "rebotarm"
        ).strip("/")
        joint_state_rate = float(self.get_parameter("joint_state_rate").value)
        cmd_arbitration = str(self.get_parameter("cmd_arbitration").value or "reject")
        self.disable_after_safe_home = bool(
            self.get_parameter("disable_after_safe_home").value
        )
        # 只接受 "reject" 或 "preempt"，否则回退到 "reject"
        if cmd_arbitration not in ("reject", "preempt"):
            self.get_logger().warn(
                f"unsupported cmd_arbitration={cmd_arbitration!r}; using 'reject'"
            )
            cmd_arbitration = "reject"

        # ═════════════════════════════════════════════════════════════
        # 依赖注入——按依赖顺序组装
        # ═════════════════════════════════════════════════════════════

        # 1. 硬件管理层（最先创建，后续模块依赖它）
        self.hardware = HardwareManager(
            hardware_config=hardware_config, model=model, channel=channel,
        )
        self.hardware.connect()  # 建立硬件连接 + 启动控制回路

        # 2. 状态发布层
        self.joint_state_publisher = JointStatePublisher(
            self, self.hardware, self.arm_namespace, joint_state_rate,
        )
        # 3. Service 接口层
        self.arm_services = ArmServices(self, self.hardware, self.arm_namespace)
        # 4. Action 接口层
        self.arm_actions = ArmActions(self, self.hardware, self.arm_namespace)
        # 5. 电机直通层（订阅 cmd 话题）
        self.motor_passthrough = MotorPassthrough(
            self, self.hardware, self.arm_namespace, cmd_arbitration,
        )

        self.get_logger().info(
            f"reBotArmController started: namespace=/{self.arm_namespace}, "
            f"joints={self.hardware.joint_names}"
        )

    # ═════════════════════════════════════════════════════════════════
    # 公共接口（供子模块回调使用）
    # ═════════════════════════════════════════════════════════════════

    def publish_arm_status(self, *, read_hardware: bool = True) -> None:
        """
        发布机械臂整体状态（latched topic）。
        子模块（MotorPassthrough / ArmServices）回调中调用的统一入口。
        """
        self.joint_state_publisher.publish_status(read_hardware=read_hardware)

    def shutdown(self) -> None:
        """
        安全关机流程：safe_home → stop_control_loop → [disable] → disconnect
        """
        self.hardware.shutdown(
            disable_after_safe_home=self.disable_after_safe_home,
        )


def main(args=None) -> None:
    """
    控制器入口函数。
    执行流程：
      1. rclpy.init()          — 初始化 ROS2 客户端库
      2. reBotArmController()  — 创建节点 + 初始化硬件 + 组装子模块
      3. MultiThreadedExecutor(4) — 4 线程并发处理回调
      4. executor.spin()       — 事件循环（阻塞直到 SIGINT）
      5. finally 块            — 安全关机 + 资源释放（顺序至关重要）
    """
    rclpy.init(args=args)
    node = reBotArmController()
    # 4 线程并发处理：状态发布 / Action / Service / MotorPassthrough 回调
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.shutdown()        # 安全回零 → 停止回路 → 失能 → 断开
        executor.shutdown()    # 停止执行器
        node.destroy_node()    # 销毁节点
        rclpy.shutdown()      # 关闭 ROS2 客户端库


if __name__ == "__main__":
    main()
