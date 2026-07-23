"""
draw_square —— 机械臂 TCP 沿矩形路径运动演示。

算法概要：
    1. 先将机械臂复位到 start_point（关节空间）
    2. 计算矩形的 4 个角点（笛卡尔空间）
    3. 依次对每个角点求解 IK → 关节运动规划 → 执行轨迹
    4. 最后回到第 1 个角点，完成闭合矩形

关键设计决策（防 wrap-joint 跳变）：
    连续关节（如 joint1、joint4 等 ±π 范围）在 IK 求解时可能给出与当前值
    相差近 2π 的解，导致轨迹规划器生成不必要的整圈旋转。本 demo 通过以下策略避免：
    - 对每个角点尝试多个 yaw 偏移量
    - 过滤掉关节变化超过 max_wrap_joint_delta 的解
    - 选择关节空间移动量最小的解（最小代价）
    - 将 IK 结果 wrap 到参考值附近（_wrap_joints）
"""

from __future__ import annotations

import sys
from math import pi

from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetMotionPlan
import rclpy
from std_msgs.msg import Header
from tf_transformations import quaternion_from_euler

from rebotarm_moveit_demos.demo_common import MoveItDemoBase


class DrawSquare(MoveItDemoBase):
    """控制机械臂 TCP 沿矩形四角运动，实现「画矩形」效果。

    运动序列：
        start_point → corner1 → corner2 → corner3 → corner4 → corner1
                  (reset)    (edge1)   (edge2)   (edge3)   (edge4)

    其中 start_point 是关节空间中的起始位姿，4 个 corner 是笛卡尔空间中
    矩形的角点，由 rectangle_center / rectangle_width / rectangle_height 定义。
    """

    def __init__(self) -> None:
        super().__init__("draw_square")

        # ---- 关节包裹（wrap）相关 ----
        # 连续关节（可无限旋转的关节）名称集合
        self.wrap_joint_names = {str(name) for name in self._param("wrap_joint_names")}
        # 连续关节允许的最大单步变化量 (rad)，超过则视为跳变（整圈翻转），丢弃该解
        self.max_wrap_joint_delta = float(self._param("max_wrap_joint_delta"))

        # ---- TCP 坐标帧 ----
        self.frame_id = str(self._param("frame_id"))
        self.tcp_link_name = str(self._param("tcp_link_name"))

        # ---- 运动目标参数 ----
        # 起始关节角（关节空间，复位目标）
        self.start_point = self._wrap_joints(
            [float(value) for value in self._param("start_point")]
        )
        # 矩形几何参数：中心坐标、宽、高（笛卡尔空间，单位 m）
        self.rectangle_center = [float(value) for value in self._param("rectangle_center")]
        self.rectangle_width = float(self._param("rectangle_width"))
        self.rectangle_height = float(self._param("rectangle_height"))

        # ---- TCP 姿态 ----
        # TCP 的基础 RPY 姿态 (rad)，各角点共用
        self.tcp_rpy = [float(value) for value in self._param("tcp_rpy")]
        # 候选 yaw 偏移列表 (rad)，每个角点会尝试所有偏移量，选最优解
        self.tcp_yaw_offsets = [float(value) for value in self._param("tcp_yaw_offsets")]

        # ---- MoveIt 规划服务客户端 ----
        self._planner = self.node.create_client(GetMotionPlan, "/plan_kinematic_path")

        # ---- 超时 & 碰撞检测 ----
        self.ik_timeout = float(self._param("ik_timeout"))
        self.result_timeout = float(self._param("result_timeout"))
        self.avoid_collisions = bool(self._param("avoid_collisions"))

    # ------------------------------------------------------------------
    #  主流程
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """执行完整的「画矩形」演示。

        Returns:
            True 全部成功，False 任意步骤失败。
        """
        # 等待所有必要的 ROS 服务就绪
        if not self._planner.wait_for_service(timeout_sec=30.0):
            self.node.get_logger().error(
                "MoveIt service /plan_kinematic_path is not available"
            )
            return False
        if not self.wait_for_ik_service():
            return False
        if not self.wait_for_execute_server():
            return False

        # Step 1: 从当前关节位置复位到 start_point
        current_joints = self._wrap_joints(self._current_joint_values())
        if not self._plan_to_joints("reset", current_joints, self.start_point):
            return False

        # Step 2: 移动到第一个角点（从 start_point 出发）
        points = self._rectangle_points()
        first_corner = self.corner_joint_target(points[0], self.start_point, "corner 1")
        if first_corner is None or not self._plan_to_joints(
            "corner 1",
            self.start_point,
            first_corner,
        ):
            return False

        # Step 3: 依次走过剩余 3 条边，最后回到 corner1 形成闭合矩形
        current_joints = first_corner
        for edge_index, end in enumerate(points[1:] + [points[0]], start=1):
            target = self.corner_joint_target(end, current_joints, f"corner {edge_index + 1}")
            if target is None:
                return False
            if not self._plan_to_joints(f"edge {edge_index}", current_joints, target):
                return False
            current_joints = target

        self.node.get_logger().info("rectangle draw demo finished")
        return True

    # ------------------------------------------------------------------
    #  矩形角点计算
    # ------------------------------------------------------------------

    def _rectangle_points(self) -> list[list[float]]:
        """根据矩形中心、宽、高计算 4 个角点的笛卡尔坐标。

        角点顺序（俯视逆时针）：
            点0: 左前   点1: 右前
            点3: 左后   点2: 右后

        Returns:
            4 个角点的 [x, y, z] 列表。
        """
        center = self.rectangle_center
        half_width = self.rectangle_width * 0.5
        half_height = self.rectangle_height * 0.5
        return [
            [center[0] - half_width, center[1] - half_height, center[2]],  # 左前
            [center[0] + half_width, center[1] - half_height, center[2]],  # 右前
            [center[0] + half_width, center[1] + half_height, center[2]],  # 右后
            [center[0] - half_width, center[1] + half_height, center[2]],  # 左后
        ]

    # ------------------------------------------------------------------
    #  TCP 位姿构造
    # ------------------------------------------------------------------

    def _waypoint(self, tcp_position: list[float], yaw_offset: float = 0.0) -> Pose:
        """根据 TCP 位置和 RPY 姿态构造 Pose 消息。

        Args:
            tcp_position: TCP 的 [x, y, z] 坐标 (m)。
            yaw_offset:   叠加在基础 yaw 上的额外偏转 (rad)。

        Returns:
            geometry_msgs/Pose 消息。
        """
        roll, pitch, yaw = self.tcp_rpy
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw + yaw_offset)
        return Pose(
            position=Point(x=tcp_position[0], y=tcp_position[1], z=tcp_position[2]),
            orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
        )

    # ------------------------------------------------------------------
    #  角点 IK 求解（带 yaw 搜索 & wrap 过滤）
    # ------------------------------------------------------------------

    def corner_joint_target(
        self,
        tcp_position: list[float],
        seed_values: list[float],
        label: str,
    ) -> list[float] | None:
        """对给定 TCP 位置求解最优关节目标。

        尝试 tcp_yaw_offsets 中的所有 yaw 偏移量，对每个偏移量调用 IK，
        然后过滤掉导致连续关节跳变的解，最终选择关节空间代价最小的解。

        Args:
            tcp_position: 目标 TCP 笛卡尔坐标 [x, y, z]。
            seed_values:  IK 种子关节值（通常为当前关节状态，帮助 IK 收敛到附近解）。
            label:        日志标签（如 "corner 1"），方便调试。

        Returns:
            最优关节目标值列表，无有效解时返回 None。
        """
        seed_values = self._wrap_joints(seed_values)
        self.node.get_logger().info(
            f"compute IK for {label}: "
            f"[{tcp_position[0]:.3f}, {tcp_position[1]:.3f}, {tcp_position[2]:.3f}]"
        )

        best = None
        best_yaw_offset = 0.0
        best_cost = float("inf")

        # 遍历所有候选 yaw 偏移，寻找最优 IK 解
        for yaw_offset in self.tcp_yaw_offsets:
            target = self._corner_joint_target(tcp_position, seed_values, label, yaw_offset)
            if target is None:
                continue
            # 检查连续关节是否发生了跳变（整圈翻转）
            if any(
                name in self.wrap_joint_names
                and abs(goal - start) > self.max_wrap_joint_delta
                for name, start, goal in zip(self.joint_names, seed_values, target)
            ):
                continue
            # 代价 = 关节空间总变化量，越小越好
            cost = sum(abs(goal - start) for start, goal in zip(seed_values, target))
            if cost < best_cost:
                best = target
                best_yaw_offset = yaw_offset
                best_cost = cost

        if best is None:
            self.node.get_logger().error(
                f"Failed to compute IK for {label} without wrapped-joint flip"
            )
            return None

        self.node.get_logger().info(
            f"{label} target yaw_offset={best_yaw_offset:.4f}: "
            f"{[round(value, 4) for value in best]}"
        )
        return best

    def _corner_joint_target(
        self,
        tcp_position: list[float],
        seed_values: list[float],
        label: str,
        yaw_offset: float,
    ) -> list[float] | None:
        """对单个 yaw_offset 调用 IK 服务，返回 wrap 后的关节值。

        Args:
            tcp_position: 目标 TCP 位置。
            seed_values:  IK 种子关节值。
            label:        日志标签。
            yaw_offset:   当前尝试的 yaw 偏移量。

        Returns:
            wrap 后的关节目标值列表，IK 失败则返回 None。
        """
        target = self.compute_ik_joint_target(
            PoseStamped(
                header=Header(frame_id=self.frame_id),
                pose=self._waypoint(tcp_position, yaw_offset),
            ),
            seed_values,
            self.tcp_link_name,
            self.ik_timeout,
            self.avoid_collisions,
            f"IK for {label} yaw_offset={yaw_offset:.4f}",
            warn_only=True,  # IK 失败仅 warning，不抛异常，由调用方处理
        )
        # IK 成功则将结果 wrap 到 seed 附近，避免连续关节的 ±2π 等效表示
        return None if target is None else self._wrap_joints(target, seed_values)

    # ------------------------------------------------------------------
    #  连续关节包裹（wrap joints）
    # ------------------------------------------------------------------

    def _wrap_joints(
        self,
        values: list[float],
        reference: list[float] | None = None,
    ) -> list[float]:
        """将连续关节值包裹到参考值附近，避免 ±2π 等效表示导致的跳变。

        对于标记为 wrap 的连续关节，IK/规划器可能返回与参考值相差 2π 的等效角度
        （例如 ref=3.0，IK 返回 -3.28，等效但不在同一周期内）。
        本函数将结果规范化到 [ref-π, ref+π] 范围内，使轨迹连续无跳变。

        Args:
            values:    需要包裹的关节值列表（与 self.joint_names 对齐）。
            reference: 参考关节值，默认全 0 → 包裹到 [-π, π]。

        Returns:
            包裹后的关节值列表。
        """
        result = []
        references = reference if reference is not None else [0.0] * len(values)
        for name, value, ref in zip(self.joint_names, values, references):
            # 非连续关节不处理，原样返回
            if name not in self.wrap_joint_names:
                result.append(value)
                continue
            # 将 value wrap 到 [ref-π, ref+π) 范围
            wrapped = ref + (value - ref + pi) % (2.0 * pi) - pi
            # 边界情况：如果结果超出 [-π, π]，回退到标准规范化
            if wrapped < -pi or wrapped > pi:
                wrapped = (value + pi) % (2.0 * pi) - pi
            # 处理浮点精度导致的 -π ↔ +π 边界歧义
            result.append(pi if wrapped == -pi and value > 0.0 else wrapped)
        return result

    # ------------------------------------------------------------------
    #  运动规划 & 执行
    # ------------------------------------------------------------------

    def _plan_to_joints(
        self,
        label: str,
        start_values: list[float],
        goal_values: list[float],
    ) -> bool:
        """调用 MoveIt 规划关节空间路径并执行轨迹。

        Args:
            label:        日志标签。
            start_values: 规划的起始关节值。
            goal_values:  规划的目标关节值。

        Returns:
            True 规划 & 执行成功，False 失败。
        """
        start_values = self._wrap_joints(start_values)
        goal_values = self._wrap_joints(goal_values)
        self.node.get_logger().info(f"move to {label}")

        # ---- 组装 MotionPlanRequest ----
        request = GetMotionPlan.Request()
        request.motion_plan_request.group_name = self.group_name
        request.motion_plan_request.pipeline_id = str(self._param("pipeline_id"))
        request.motion_plan_request.planner_id = str(self._param("planner_id"))
        request.motion_plan_request.allowed_planning_time = float(
            self._param("planning_time")
        )
        request.motion_plan_request.num_planning_attempts = 5  # 规划失败后重试次数
        request.motion_plan_request.max_velocity_scaling_factor = float(
            self._param("velocity_scaling")
        )
        request.motion_plan_request.max_acceleration_scaling_factor = float(
            self._param("acceleration_scaling")
        )

        # 设置起点：使用当前关节状态构造 RobotState
        request.motion_plan_request.start_state = self._joint_state(start_values)
        # 设置终点：关节约束（每个关节的目标值 ± tolerance）
        request.motion_plan_request.goal_constraints = [
            self._joint_constraints(goal_values)
        ]

        # ---- 异步调用规划服务 ----
        future = self._planner.call_async(request)
        if not self.wait(future, self.result_timeout):
            self.node.get_logger().error(
                f"MoveIt planner did not return within {self.result_timeout:.1f}s"
            )
            return False

        # ---- 检查规划结果 ----
        response = future.result()
        plan_response = response.motion_plan_response if response is not None else None
        if (
            plan_response is None
            or plan_response.error_code.val != MoveItErrorCodes.SUCCESS
        ):
            code = plan_response.error_code.val if plan_response is not None else "empty"
            self.node.get_logger().error(
                f"MoveIt planning failed with code {code}"
            )
            return False

        # ---- 执行轨迹 ----
        self.node.get_logger().info(f"MoveIt planned {label}")
        return self.execute_trajectory(plan_response.trajectory, self.result_timeout)

    def _joint_constraints(self, joint_values: list[float]) -> Constraints:
        """为给定的关节值构造 MoveIt Constraints 消息。

        每个关节约束包含目标位置和 ±tolerance 的上下容忍范围，
        weight=1.0 表示所有关节同等重要。

        Args:
            joint_values: 目标关节值列表（与 self.joint_names 对齐）。

        Returns:
            moveit_msgs/Constraints 消息。
        """
        tolerance = float(self._param("joint_tolerance"))
        return Constraints(
            joint_constraints=[
                JointConstraint(
                    joint_name=name,
                    position=value,
                    tolerance_above=tolerance,
                    tolerance_below=tolerance,
                    weight=1.0,
                )
                for name, value in zip(self.joint_names, joint_values)
            ]
        )

    def _current_joint_values(self) -> list[float]:
        """读取当前关节状态，回退到 start_point。"""
        return self.current_joint_values(list(self.start_point), "start_point")


# ================================================================
#  入口
# ================================================================


def main() -> None:
    """ROS2 节点入口：初始化 → 运行 demo → 清理退出。"""
    rclpy.init()
    demo = DrawSquare()
    try:
        ok = demo.run()
    except Exception as exc:
        demo.node.get_logger().error(str(exc))
        ok = False
    finally:
        demo.node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
