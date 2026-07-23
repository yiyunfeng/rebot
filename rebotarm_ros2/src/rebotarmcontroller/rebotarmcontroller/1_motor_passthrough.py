"""
motor_passthrough 模块 — 电机直通指令中间层
===========================================

本模块实现了 ROS2 话题订阅 与 底层硬件接口 之间的转发桥梁。
核心职责：监听关节/夹爪控制指令话题 → 校验仲裁权限 → 转发给硬件驱动层。

**工作流程**：
  1. 订阅 ROS2 话题（JointMitCmd / JointPosVelCmd）
  2. 每次收到消息，先检查当前状态机是否允许发送底层指令
  3. 若状态为 GRAVITY_COMP / SAFE_HOMING → 拒绝
  4. 若状态为 TRAJ_RUNNING（轨迹正在执行）：
      - 「拒绝模式」→ 拒绝
      - 「抢占模式」→ 停止当前轨迹，执行新指令
  5. 允许 → 调用硬件接口发送指令
  6. 无论成功/失败，最终广播一次机械臂状态

**两种控制模式**：
  - MIT 模式（力矩-位置-阻抗控制）：需要目标位置、速度、Kp、Kd、力矩前馈
  - POS_VEL 模式（位置-速度控制）：需要目标位置、速度限制
"""

from __future__ import annotations  # PEP 563: 延迟求值类型注解，支持前向引用

from rclpy.qos import QoSProfile, ReliabilityPolicy  # ROS2 QoS 配置
from rebotarm_msgs.msg import (
    JointMitCmd,        # 关节 MIT 控制指令消息（Mass-Impedance-Torque）
    JointPosVelCmd,     # 关节位置-速度控制指令消息
)


class MotorPassthrough:
    """
    电机直通层 —— 将 ROS2 消息透明转发到硬件驱动层，附带仲裁逻辑。

    设计意图：
      - 解耦话题层和硬件层（话题格式变化不影响硬件接口）
      - 集中仲裁逻辑（同一状态机下，仅一个指令源可控制电机）
      - 支持轨迹抢占（手动操作可打断正在运行的轨迹）

    Attributes:
        _node:          ROS2 Node 引用，用于创建订阅 / 日志 / 状态发布
        _hardware:      HardwareManager 实例，封装真实/仿真电机控制
        _arbitration:   仲裁模式字符串，通常为 "reject" 或 "preempt"
        _subscriptions: 保存所有创建的订阅对象列表（防 GC 回收）
    """

    def __init__(self, node, hardware, namespace: str, arbitration: str) -> None:
        """
        初始化电机直通层。

        Args:
            node:        ROS2 Node 对象，提供 create_subscription / get_logger 等接口
            hardware:    HardwareManager 实例，封装底层电机控制 send_joint_mit_cmd 等
            namespace:   命名空间前缀，用于构造话题名（如 "/my_arm/joints/joint1/cmd/mit"）
            arbitration: 仲裁模式：
                            "reject"  → 轨迹运行时直接拒绝直通指令
                            "preempt" → 轨迹运行时停止轨迹，执行直通指令（抢占）
        """
        # ===== 注入依赖 =====
        self._node = node
        self._hardware = hardware
        self._arbitration = arbitration

        # ===== QoS 配置 =====
        # depth=10  → 队列深度：最多缓存 10 条未处理消息
        # RELIABLE  → 可靠性策略：保证消息不丢失（TCP 重传）
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # 保存所有订阅引用，防止 Python GC 回收未绑定变量的订阅对象
        self._subscriptions = []

        # ═══════════════════════════════════════════════════════════════
        # 定义命令配置元组列表
        # 每个元组：(消息类型, 话题后缀, 回调工厂函数)
        # 回调工厂接收 (硬件引用, 关节名?, 消息) → 调用对应的硬件接口
        # ═══════════════════════════════════════════════════════════════

        # ----- 关节命令配置 -----
        # 每个关节独立订阅话题：/{namespace}/joints/{joint_name}/{label}
        joint_commands = (
            (
                JointMitCmd,
                "cmd/mit",
                # lambda 参数说明：hw=硬件引用, name=关节名, msg=ROS消息
                # send_joint_mit_cmd 将 MIT 消息的 5 个字段转发给底层
                lambda hw, name, msg: hw.send_joint_mit_cmd(
                    name,       # 关节名称（如 "joint1"）
                    msg.pos,    # 目标位置（弧度）
                    msg.vel,    # 目标速度（弧度/秒）
                    msg.kp,     # 比例增益（刚度系数）
                    msg.kd,     # 微分增益（阻尼系数）
                    msg.tau,    # 前馈力矩（Nm）
                ),
            ),
            (
                JointPosVelCmd,
                "cmd/pos_vel",
                # send_joint_pos_vel_cmd 将位置-速度消息转发给底层
                lambda hw, name, msg: hw.send_joint_pos_vel_cmd(
                    name,
                    msg.pos,    # 目标位置（弧度）
                    msg.vlim,   # 速度限制（弧度/秒）
                ),
            ),
        )

        # ----- 夹爪命令配置 -----
        # 夹爪只有一个，无关节名参数：/{namespace}/gripper/{label}
        gripper_commands = (
            (
                JointMitCmd,
                "cmd/mit",
                # 夹爪 MIT 控制：无关节名，直接发送给夹爪
                lambda hw, msg: hw.send_gripper_mit_cmd(
                    msg.pos,
                    msg.vel,
                    msg.kp,
                    msg.kd,
                    msg.tau,
                ),
            ),
            (
                JointPosVelCmd,
                "cmd/pos_vel",
                # 夹爪位置-速度控制
                lambda hw, msg: hw.send_gripper_pos_vel_cmd(
                    msg.pos,
                    msg.vlim,
                ),
            ),
        )

        # ═══════════════════════════════════════════════════════════════
        # 批量创建订阅
        # ═══════════════════════════════════════════════════════════════

        # ---- 为每个关节创建订阅 ----
        # hardware.joint_names 是关节名列表，如 ["joint1", "joint2", ..., "joint6"]
        for joint_name in hardware.joint_names:
            for msg_type, label, command in joint_commands:
                self._subscribe(
                    msg_type,
                    # 话题名示例："/my_arm/joints/joint2/cmd/mit"
                    f"/{namespace}/joints/{joint_name}/{label}",
                    # 为每个 (关节名, 命令类型) 组合创建独立回调闭包
                    self._make_joint_callback(joint_name, label, command),
                    qos,
                )

        # ---- 如果硬件支持夹爪，为夹爪创建订阅 ----
        if hardware.has_gripper:
            for msg_type, label, command in gripper_commands:
                self._subscribe(
                    msg_type,
                    # 话题名示例："/my_arm/gripper/cmd/pos_vel"
                    f"/{namespace}/gripper/{label}",
                    self._make_gripper_callback(label, command),
                    qos,
                )

    # ══════════════════════════════════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════════════════════════════════

    def _subscribe(self, msg_type, topic: str, callback, qos: QoSProfile) -> None:
        """
        创建 ROS2 订阅并保存引用。

        **为什么保存到列表**：
          ROS2 的 create_subscription 返回一个订阅对象。
          如果该对象没有被任何变量持有，Python 的 GC 会回收它，
          导致订阅被销毁，消息不再触发回调。
          存储到 self._subscriptions 列表可确保其生命周期与本实例一致。

        **关键参数 callback_group**：
          使用 ReentrantCallbackGroup，允许回调函数并发执行（多线程安全），
          避免一个慢回调阻塞其他消息的处理。
        """
        self._subscriptions.append(
            self._node.create_subscription(
                msg_type,                           # 消息类型（JointMitCmd 或 JointPosVelCmd）
                topic,                              # 完整话题名
                callback,                           # 收到消息后的回调函数
                qos,                                # QoS 可靠性配置
                callback_group=self._node.reentrant_group,  # 可重入回调组（支持并发）
            )
        )

    def _make_joint_callback(self, joint_name: str, label: str, command) -> object:
        """
        工厂方法：为指定的 (关节名, 命令类型) 创建回调闭包。

        使用闭包模式而非 lambda 的原因：
          - 闭包内可以写多行逻辑（仲裁检查 / 异常处理 / 状态发布）
          - 每个回调绑定独立的 joint_name 和 label，避免循环变量迟绑定陷阱
          - 便于单元测试（可单独验证回调行为）

        Args:
            joint_name: 关节名称（如 "joint3"）
            label:      命令标签（"cmd/mit" 或 "cmd/pos_vel"）
            command:    可调用对象 → (hardware, joint_name, msg) -> None

        Returns:
            _callback(msg) 闭包，可直接用作 ROS2 回调
        """

        def _callback(msg) -> None:
            """
            关节命令回调 — 仲裁检查 → 执行 → 发布状态。

            流程：
              1. can_send_lowlevel? → 状态机允许发送?
                 - GRAVITY_COMP / SAFE_HOMING → 拒绝
                 - TRAJ_RUNNING → 根据仲裁模式决定
              2. 允许 → 调用 hardware.send_joint_xxx()
              3. 异常 → 记录警告日志（不抛异常，防止回调崩溃）
              4. finally → 无论成败，广播一次机械臂状态
            """
            # ----- 步骤 1: 仲裁权限检查 -----
            if not self._can_send_lowlevel(
                f"/joints/{joint_name}/{label}",
                allow_preempt=True,  # 关节指令允许抢占轨迹
            ):
                return  # 被拒绝，直接返回

            # ----- 步骤 2-3: 执行 + 异常处理 -----
            try:
                command(self._hardware, joint_name, msg)
            except Exception as exc:
                # 仅记录警告，不中断回调流程
                # 异常原因示例：关节名无效、硬件驱动异常、位置超限等
                self._node.get_logger().warn(
                    f"joint {label} failed for {joint_name}: {exc}"
                )
            finally:
                # ----- 步骤 4: 发布最新状态 -----
                # 确保下游（如 GUI / 监控节点）能感知机械臂当前状态
                self._node.publish_arm_status()

        return _callback

    def _make_gripper_callback(self, label: str, command) -> object:
        """
        工厂方法：为夹爪创建回调闭包。

        与 _make_joint_callback 的重要区别：
          - 无 joint_name 参数（夹爪是单体，不是多关节）
          - allow_preempt=False（夹爪指令不允许抢占轨迹，
            因为夹爪操作不应中断正在运行的机械臂运动）

        Args:
            label:   命令标签（"cmd/mit" 或 "cmd/pos_vel"）
            command: 可调用对象 → (hardware, msg) -> None

        Returns:
            _callback(msg) 闭包
        """

        def _callback(msg) -> None:
            """
            夹爪命令回调 — 仲裁检查 → 执行 → 发布状态。

            与关节回调的关键差异：
              - _can_send_lowlevel 的 allow_preempt=False，
                即轨迹运行时，夹爪指令总是被拒绝，
                确保夹爪不会意外打断机械臂运动轨迹。
            """
            # 夹爪指令不允许抢占轨迹，始终检查
            if not self._can_send_lowlevel(
                f"/gripper/{label}",
                allow_preempt=False,  # ← 与关节回调不同：不抢占
            ):
                return

            try:
                command(self._hardware, msg)
            except Exception as exc:
                self._node.get_logger().warn(f"gripper {label} failed: {exc}")
            finally:
                self._node.publish_arm_status()

        return _callback

    def _can_send_lowlevel(self, label: str, *, allow_preempt: bool) -> bool:
        """
        仲裁决策：判断当前是否允许向底层发送指令。

        这是整个模块的核心仲裁逻辑，确保多指令源（轨迹/直通/手动）
        不会同时控制同一个硬件。

        **状态机模型**：
          ┌─────────────┐
          │ GRAVITY_COMP │ ← 重力补偿中 → 拒绝所有直通指令
          │ SAFE_HOMING  │ ← 安全回零中 → 拒绝所有直通指令
          ├─────────────┤
          │ TRAJ_RUNNING │ ← 轨迹执行中 → 按仲裁模式决定
          ├─────────────┤
          │   IDLE 等    │ ← 空闲状态 → 允许所有直通指令
          └─────────────┘

        **抢占语义**：
          当状态为 TRAJ_RUNNING 且允许抢占时：
            1. 记录警告日志"正在抢占轨迹"
            2. 调用 hardware.stop_motion() 停止当前轨迹
            3. 返回 True，允许新指令执行
          这实现了「手动操作打断自动轨迹」的用户体验。

        Args:
            label:     操作标签（用于日志），如 "/joints/joint2/cmd/mit"
            allow_preempt: 是否允许抢占。True=关节指令可抢占，False=夹爪不可抢占

        Keyword Args:
            allow_preempt: 强制关键字参数（* 分隔），避免调用时位置参数误传

        Returns:
            True  → 允许发送
            False → 拒绝发送（已记录警告日志说明原因）
        """
        # 获取当前状态机状态（字符串）
        state = self._hardware.state_machine

        # ----- 场景 1: 安全状态 → 拒绝一切 -----
        # GRAVITY_COMP: 重力补偿模式，机械臂正在感知外力
        # SAFE_HOMING:  安全回零中，需要独占控制权
        if state in ("GRAVITY_COMP", "SAFE_HOMING"):
            self._node.get_logger().warn(
                f"rejecting {label} in state {state}"
            )
            return False

        # ----- 场景 2: 轨迹正在运行 -----
        if state == "TRAJ_RUNNING":
            # 拒绝条件：
            #   1. 全局仲裁模式是 "reject" → 不抢占
            #   2. 当前回调 allow_preempt=False（如夹爪）→ 不抢占
            if self._arbitration == "reject" or not allow_preempt:
                self._node.get_logger().warn(
                    f"rejecting {label} while trajectory is running"
                )
                return False

            # 抢占路径：记录日志 → 停止轨迹 → 返回 True 允许新指令
            self._node.get_logger().warn(
                f"preempting trajectory for {label}"
            )
            self._hardware.stop_motion()

        # ----- 场景 3: 其他状态（IDLE 等）→ 允许 -----
        return True
