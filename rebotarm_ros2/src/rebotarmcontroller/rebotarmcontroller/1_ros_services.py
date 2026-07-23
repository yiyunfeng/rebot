"""
ros_services 模块 — 机械臂 ROS2 Service 服务端
================================================

本模块将 HardwareManager 的各类操作暴露为 ROS2 Service 接口，
供外部节点（GUI、脚本、调试工具等）通过 service call 控制机械臂。

**提供的服务列表**：
  | 服务名                           | 类型             | 功能                       |
  |----------------------------------|------------------|----------------------------|
  | /{namespace}/enable              | std_srvs/Trigger | 使能机械臂                  |
  | /{namespace}/disable             | std_srvs/Trigger | 失能机械臂                  |
  | /{namespace}/safe_home           | std_srvs/Trigger | 安全回零                    |
  | /{namespace}/gravity_compensation/start | Trigger  | 启动重力补偿               |
  | /{namespace}/gravity_compensation/stop  | Trigger  | 停止重力补偿               |
  | /{namespace}/set_zero            | SetZero          | 设置关节零点               |
  | /{namespace}/move_to_pose_ik     | MoveToPoseIK     | IK 逆解移动到目标位姿       |
  | /{namespace}/gripper/set         | SetGripper       | 设置夹爪位置               |
  | /{namespace}/gripper/open        | GripperCommand   | 打开夹爪                   |
  | /{namespace}/gripper/close       | GripperCommand   | 闭合夹爪                   |
"""

from __future__ import annotations

from rebotarm_msgs.srv import (
    GripperCommand,    # 夹爪控制请求（位置 + 超时）
    MoveToPoseIK,      # IK 位姿移动请求 + 响应（含 q_solution）
    SetGripper,        # 设置夹爪位置
    SetZero,           # 关节零点设置
)
from std_srvs.srv import Trigger  # 标准触发服务（无请求数据，仅成功/消息响应）

from .conversions import pose_to_xyz_rpy  # Pose → xyz+rpy 转换


class ArmServices:
    """
    机械臂 Service 服务端 —— 包装 HardwareManager 为 ROS2 Service。

    设计要点：
      - 慢速操作（enable/disable/safe_home/零点）→ slow_group（互斥回调组，避免并发冲突）
      - 实时操作（IK 移动/夹爪控制）→ reentrant_group（允许并发执行）
      - 统一错误处理：_run() 模板方法包装 try/except
    """

    def __init__(self, node, hardware, namespace: str) -> None:
        """
        初始化所有 Service 服务端。

        Args:
            node:      ROS2 Node 引用
            hardware:  HardwareManager 实例
            namespace: 话题/服务命名空间前缀
        """
        self._node = node
        self._hardware = hardware

        # 服务定义元组：(请求类型, 服务名后缀, 处理函数, 回调组)
        services = (
            (Trigger, "enable",                              self.enable,                        node.slow_group),
            (Trigger, "disable",                             self.disable,                       node.slow_group),
            (Trigger, "safe_home",                           self.safe_home,                     node.slow_group),
            (Trigger, "gravity_compensation/start",          self.start_gravity_compensation,    node.slow_group),
            (Trigger, "gravity_compensation/stop",           self.stop_gravity_compensation,     node.slow_group),
            (SetZero, "set_zero",                            self.set_zero,                      node.slow_group),
            (MoveToPoseIK, "move_to_pose_ik",                self.move_to_pose_ik,               node.reentrant_group),
            (SetGripper, "gripper/set",                      self.set_gripper,                   node.reentrant_group),
            (GripperCommand, "gripper/open",                 self.open_gripper,                  node.slow_group),
            (GripperCommand, "gripper/close",                self.close_gripper,                 node.slow_group),
        )
        for srv_type, name, handler, group in services:
            node.create_service(
                srv_type,
                f"/{namespace}/{name}",  # 完整服务名，如 "/rebotarm/enable"
                handler,
                callback_group=group,
            )

    # ══════════════════════════════════════════════════════════════════
    # 通用模板方法
    # ══════════════════════════════════════════════════════════════════

    def _run(self, response, action, success_message: str, *, read_hardware=True):
        """
        统一的 service handler 模板。

        封装 try/except + 状态发布，使每个具体 handler 只需提供 action 即可：
          - 成功 → response.success=True, message=成功信息
          - 异常 → response.success=False, message=异常消息
          - finally → publish_arm_status() 广播最新状态

        Args:
            response:        服务响应对象（被原地修改后返回）
            action:          无参可调用对象，执行具体的硬件操作
            success_message: 成功时的消息文本
            read_hardware:   发布状态时是否从硬件重新读取（disable 时无需）
        """
        try:
            action()
            response.success = True
            response.message = success_message
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        self._node.publish_arm_status(read_hardware=read_hardware)
        return response

    # ══════════════════════════════════════════════════════════════════
    # 基础操作
    # ══════════════════════════════════════════════════════════════════

    def enable(self, _request, response):
        """使能机械臂（启用控制回路）。"""
        return self._run(response, self._hardware.enable, "enabled")

    def disable(self, _request, response):
        """
        失能机械臂。
        操作顺序：先停止重力补偿，再 disable 硬件。
        read_hardware=False：硬件已失能，无需再读取状态码。
        """
        def action():
            self._hardware.stop_gravity_compensation()
            self._hardware.disable()
        return self._run(response, action, "disabled", read_hardware=False)

    def safe_home(self, _request, response):
        """安全回零：机械臂回到预设的安全位置。"""
        return self._run(response, self._hardware.safe_home, "safe_home complete")

    # ══════════════════════════════════════════════════════════════════
    # 重力补偿
    # ══════════════════════════════════════════════════════════════════

    def start_gravity_compensation(self, _request, response):
        """启动重力补偿模式（机械臂进入零力拖动状态）。"""
        return self._run(
            response,
            self._hardware.start_gravity_compensation,
            "gravity compensation started",
        )

    def stop_gravity_compensation(self, _request, response):
        """停止重力补偿模式。"""
        return self._run(
            response,
            self._hardware.stop_gravity_compensation,
            "gravity compensation stopped",
        )

    # ══════════════════════════════════════════════════════════════════
    # 零点设置
    # ══════════════════════════════════════════════════════════════════

    def set_zero(self, request, response):
        """
        设置关节零点位置。
        先停止重力补偿，再调用 hardware.set_zero()。
        """
        def action():
            self._hardware.stop_gravity_compensation()
            if not self._hardware.set_zero(request.joint_name):
                raise RuntimeError("set_zero failed")
        return self._run(response, action, "set_zero complete")

    # ══════════════════════════════════════════════════════════════════
    # IK 位姿移动
    # ══════════════════════════════════════════════════════════════════

    def move_to_pose_ik(self, request, response):
        """
        通过逆运动学(IK)移动到目标位姿。
        异常时 hold_current_position() 保持当前位置防止机械臂掉落。
        返回 q_solution（IK 求解的关节角度解）供客户端参考。
        """
        try:
            self._hardware.stop_gravity_compensation()
            x, y, z, roll, pitch, yaw = pose_to_xyz_rpy(request.target_pose)
            ok, q_solution = self._hardware.move_to_pose_ik(x, y, z, roll, pitch, yaw)
            response.success = ok
            response.message = "IK target accepted" if ok else "IK failed"
            response.q_solution = q_solution
        except Exception as exc:
            self._hardware.hold_current_position()
            response.success = False
            response.message = str(exc)
            response.q_solution = []
        self._node.publish_arm_status()
        return response

    # ══════════════════════════════════════════════════════════════════
    # 夹爪控制
    # ══════════════════════════════════════════════════════════════════

    def set_gripper(self, request, response):
        """直接设置夹爪位置（不等待到达）。"""
        try:
            reached, reached_position = self._hardware.set_gripper_position(
                request.position,
            )
            response.success = bool(reached)
            response.reached_position = float(reached_position)
        except Exception as exc:
            response.success = False
            response.reached_position = 0.0
            self._node.get_logger().error(f"gripper set failed: {exc}")
        self._node.publish_arm_status()
        return response

    def open_gripper(self, request, response):
        """打开夹爪（使用硬件配置的 open_position）。"""
        return self._move_gripper(
            request, response, self._hardware.gripper_open_position, "open"
        )

    def close_gripper(self, request, response):
        """闭合夹爪（使用硬件配置的 close_position）。"""
        return self._move_gripper(
            request, response, self._hardware.gripper_close_position, "close"
        )

    def _move_gripper(self, request, response, default_target: float, label: str):
        """
        夹爪移动的通用实现。
        request.position == 0.0 时使用 default_target。
        request.timeout <= 0.0 时使用默认超时 3.0s。
        """
        try:
            target = (
                default_target if request.position == 0.0 else float(request.position)
            )
            success, position = self._hardware.set_gripper_position(
                target,
                timeout=request.timeout if request.timeout > 0.0 else 3.0,
            )
            response.success = bool(success)
            response.reached_position = float(position)
            response.message = (
                f"gripper {label} complete" if success else f"gripper {label} timeout"
            )
        except Exception as exc:
            response.success = False
            response.reached_position = 0.0
            response.message = str(exc)
        self._node.publish_arm_status()
        return response
