"""MoveIt 版 Pick & Place。

这个节点不直接算轨迹，而是像 RViz 一样向 /move_action 发送目标。
它同时维护 MoveIt planning scene：
  - 桌子是碰撞物体
  - 夹取前方块是世界碰撞物体
  - 夹取后方块 attach 到 gripper_tcp
  - 放置后方块重新回到世界碰撞物体
"""

from __future__ import annotations

import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from rebotarm_gazebo.cube_spawner import CubeSpawner
from rebotarm_gazebo.real_controller import RealController
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    OrientationConstraint,
    PlanningScene,
    PlanningSceneComponents,
    PositionConstraint,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ── 机械臂与夹爪关节名称 ──
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]  # 六轴机械臂关节名
GRIPPER_JOINTS = ["gripper_joint1", "gripper_joint2"]                      # 两指夹爪关节名

# ── 默认位姿 ──
HOME = [0.0, -0.05, -0.05, 0.0, 0.0, 0.0]  # Home 位姿（6个关节角度，弧度）

# ── 坐标系 ──
TASK_FRAME = "base_link"  # MoveIt 规划参考坐标系

# ── 允许碰撞的 link ──
TOUCH_LINKS = ["gripper_tcp", "gripper_link", "gripper_left", "gripper_right"]  # 夹爪相关 link，抓取时允许与方块接触

# Gazebo 中 robot spawn 的 world -> base_link 固定偏移。
# MoveIt 规划统一用 base_link；只有生成 Gazebo 方块时需要转成 world 坐标。
SIM_BASE_WORLD_X = 0.05   # base_link 相对 world 的 X 偏移
SIM_BASE_WORLD_Y = 0.0    # base_link 相对 world 的 Y 偏移
SIM_BASE_WORLD_Z = 0.265  # base_link 相对 world 的 Z 偏移（桌面高度）

# ── 桌子参数 ──
TABLE_X = 0.28           # 桌子中心 X（world 坐标）
TABLE_Y = 0.0            # 桌子中心 Y（world 坐标）
TABLE_Z = 0.0            # 桌子底面 Z（world 坐标）
TABLE_YAW = 1.5708       # 桌子偏航角（≈ π/2，旋转 90°）

# ── 桌面尺寸 ──
TABLE_TOP_LENGTH = 0.4     # 桌面长（X方向）
TABLE_TOP_WIDTH = 0.6      # 桌面宽（Y方向）
TABLE_TOP_THICKNESS = 0.03 # 桌面厚度（Z方向）
TABLE_TOP_Z = 0.260        # 桌面上表面 Z 坐标（world）

# ── 桌腿参数 ──
TABLE_LEG_RADIUS = 0.02     # 桌腿半径（圆柱体）
TABLE_LEG_HEIGHT = 0.245    # 桌腿高度
TABLE_LEG_X_OFFSET = 0.17   # 桌腿相对桌子中心的 X 偏移
TABLE_LEG_Y_OFFSET = 0.27   # 桌腿相对桌子中心的 Y 偏移


def make_pose(x: float, y: float, z: float, q: tuple[float, float, float, float]) -> Pose:
    """由位置坐标和四元数构造 Pose 消息"""
    return Pose(
        position=Point(x=float(x), y=float(y), z=float(z)),           # 平移分量
        orientation=Quaternion(
            x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])  # 旋转分量（四元数）
        ),
    )


def make_box(size_x: float, size_y: float, size_z: float) -> SolidPrimitive:
    """创建长方体几何基元，用于碰撞检测"""
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX              # 几何类型：盒子
    box.dimensions = [size_x, size_y, size_z]  # 盒子尺寸 [长(X), 宽(Y), 高(Z)]
    return box


def make_sphere(radius: float) -> SolidPrimitive:
    """创建球体几何基元，用于位置约束的容差区域"""
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE  # 几何类型：球体
    sphere.dimensions = [radius]         # 球体尺寸：半径（米）
    return sphere


def make_cylinder(radius: float, height: float) -> SolidPrimitive:
    """创建圆柱体几何基元，用于桌腿碰撞检测"""
    cylinder = SolidPrimitive()
    cylinder.type = SolidPrimitive.CYLINDER  # 几何类型：圆柱体
    cylinder.dimensions = [height, radius]   # 圆柱体尺寸：[高度, 半径]（单位：米）
    return cylinder


def yaw_pose(x: float, y: float, z: float, yaw: float) -> Pose:
    """由位置和偏航角构造 Pose（roll/pitch 为 0）"""
    return Pose(
        position=Point(x=float(x), y=float(y), z=float(z)),  # 平移分量
        orientation=Quaternion(
            z=float(math.sin(yaw / 2.0)),  # 绕 Z 轴旋转的虚部（半角公式）
            w=float(math.cos(yaw / 2.0)),  # 绕 Z 轴旋转的实部
        ),
    )


def gripper_traj(position: float) -> JointTrajectory:
    """生成夹爪轨迹消息：两个指节同步运动到同一位置"""
    msg = JointTrajectory()
    msg.joint_names = list(GRIPPER_JOINTS)                     # 两指节关节名
    point = JointTrajectoryPoint()
    point.positions = [position, position]                     # 两指节目标位置（等值 → 对称开合）
    point.time_from_start = Duration(sec=0, nanosec=500_000_000)  # 运动时间 0.5 秒
    msg.points = [point]
    return msg


class MoveItPickPlace(Node):
    """MoveIt 版 Pick & Place 节点：向 move_group 发目标而非直接计算轨迹"""

    def __init__(self) -> None:
        super().__init__("moveit_pick_place")  # 节点名

        # ── ROS2 参数声明（可通过 launch 文件覆盖） ──
        self.declare_parameter("mode", "sim")               # 运行模式：sim / hardware
        self.declare_parameter("namespace", "rebotarm")      # 机械臂命名空间
        self.declare_parameter("cube_x", 0.30)              # 方块 X 坐标（base_link 系）
        self.declare_parameter("cube_y", 0.15)              # 方块 Y 坐标
        self.declare_parameter("cube_z", 0.025)             # 方块 Z 坐标（底部）
        self.declare_parameter("place_x", 0.40)             # 放置目标 X 坐标
        self.declare_parameter("place_y", -0.10)            # 放置目标 Y 坐标
        self.declare_parameter("place_z", 0.025)            # 放置目标 Z 坐标
        self.declare_parameter("cube_size", 0.06)           # 方块边长（米）
        self.declare_parameter("pre_height", 0.12)          # 抓取/放置前的抬高高度（相对方块底部）
        self.declare_parameter("pick_height", 0.03)         # 抓取时的下降高度
        self.declare_parameter("gripper_open", 0.06)        # 夹爪张开宽度（模拟值）
        # gripper_close < 0 → 自动按 cube_size 算
        self.declare_parameter("gripper_close", -1.0)       # 夹爪闭合宽度（负值=自动计算）
        self.declare_parameter("max_gripper_width", 0.09)   # 夹爪最大开口宽度（硬件映射用）
        self.declare_parameter("hardware_open_gripper_position", -5.0)  # 硬件夹爪张开指令值（DM）
        self.declare_parameter("hardware_closed_gripper_position", 0.0) # 硬件夹爪闭合指令值（DM）
        self.declare_parameter("gripper_max_effort", 10.0)  # 夹爪最大力矩
        self.declare_parameter("velocity_scaling", 0.5)     # 全局速度倍率（0-1）
        self.declare_parameter("acceleration_scaling", 0.5) # 全局加速度倍率（0-1）
        self.declare_parameter("constrain_joint6", False)   # 是否限制 joint6 转动范围
        self.declare_parameter("joint6_goal_tolerance", 0.6)# joint6 约束容差（弧度）

        # ── 参数缓存 ──
        self.mode = str(self.get_parameter("mode").value).lower()  # 运行模式
        self.namespace = str(self.get_parameter("namespace").value).strip("/")  # 命名空间（去首尾斜杠）
        self.cube_size = float(self.get_parameter("cube_size").value)  # 方块边长
        self.gripper_open = float(self.get_parameter("gripper_open").value)  # 夹爪张开宽度
        gripper_close = float(self.get_parameter("gripper_close").value)
        if gripper_close < 0.0:
            gripper_close = self.cube_size / 2.0 + 0.001  # 自动：半边长 + 1mm 余量
        self.gripper_close = max(0.0, min(gripper_close, self.gripper_open))  # 闭合宽度（限制在 [0, open] 范围）

        # 机械臂规划、轨迹连续化和关节限位统一由 RealController 处理。
        self._controller = RealController(self)
        self._controller.velocity_scaling = float(
            self.get_parameter("velocity_scaling").value
        )
        self._controller.acceleration_scaling = float(
            self.get_parameter("acceleration_scaling").value
        )
        # 把方块、桌子等碰撞体注册到 MoveIt planning scene，规划器自动绕开
        self.scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")  # 场景更新 service
        self._get_scene = self.create_client(GetPlanningScene, "/get_planning_scene")  # 场景读取 service
        # ── 模式相关初始化 ──
        if self.mode == "sim":
            self.cube = CubeSpawner(self, size=self.cube_size)  # Gazebo 方块生成器
            self.gripper_pub = self.create_publisher(
                JointTrajectory, "/gripper_controller/joint_trajectory", 10  # 夹爪轨迹发布者
            )
            self.hardware_gripper = None  # 仿真模式无需硬件 gripper action
        else:
            self.cube = None                # 真机模式无需 Gazebo 方块
            self.gripper_pub = None         # 真机不用 topic 控制夹爪
            self.hardware_gripper = ActionClient(
                self, GripperCommand, f"/{self.namespace}/gripper/command"  # 真机夹爪 action
            )

    def run(self) -> bool:
        """执行完整的 Pick & Place 流程，返回 True 表示成功"""
        log = self.get_logger().info
        log(f"===== MoveIt Pick & Place 开始，mode={self.mode} =====")

        # ── 读取抓取/放置目标坐标 ──
        cube = self._point("cube")    # (x, y, z) 抓取点
        place = self._point("place")  # (x, y, z) 放置点

        # 末端朝下的四元数（夹爪垂直于桌面）: 绕 Y 轴 90°
        down_q = (0.0, 0.7071068, 0.0, 0.7071068)

        # ── 构造各阶段目标位姿 ──
        pick_above = make_pose(cube[0], cube[1], cube[2] + self._pre_height(), down_q)   # 抓取点上方
        pick_at = make_pose(cube[0], cube[1], cube[2] + self._pick_height(), down_q)      # 抓取点（下降到方块）
        place_above = make_pose(place[0], place[1], place[2] + self._pre_height(), down_q) # 放置点上方
        place_at = make_pose(place[0], place[1], place[2] + self._pick_height() + 0.02, down_q)  # 放置点（+2cm 确保放稳）

        # ── 等待所有 MoveIt 服务就绪 ──
        if not self.scene.wait_for_service(timeout_sec=20.0):
            self.get_logger().error("/apply_planning_scene 不可用")
            return False
        if not self._get_scene.wait_for_service(timeout_sec=20.0):
            self.get_logger().error("/get_planning_scene 不可用")
            return False

        # ── Step 1: 添加桌子和绿色方块到 MoveIt planning scene（规划器绕障用） ──
        log("Step 1: 添加桌子和绿色方块到 MoveIt planning scene")
        self._apply_scene(self._table_objects() + [self._cube_object(*cube)], [])
        if self.mode == "sim" and self.cube:
            self.cube.spawn(*self._base_to_world_point(cube))  # Gazebo 中生成方块
            self.cube.detach()  # 抵消 DetachableJoint Reset() 自动吸附

        # ── Step 2: 张开夹爪 ──
        log("Step 2: 张开夹爪")
        self._move_gripper(self.gripper_open)

        # ── Step 3: MoveIt 规划到方块上方 ──
        log("Step 3: MoveIt 规划到方块上方")
        if not self._go_pose(pick_above):
            return False

        # ── Step 4: 允许夹爪接触方块（ACM），规划下降到夹取位置 ──
        log("Step 4: 允许夹爪接触方块（ACM），规划下降到夹取位置")
        self._allow_cube_touch_collisions(True)   # 临时允许夹爪 link 与方块碰撞
        if not self._go_pose(pick_at):
            return False

        # ── Step 5: 闭合夹爪，方块 attach 到 gripper_tcp ──
        log("Step 5: 闭合夹爪，方块 attach 到 gripper_tcp")
        self._move_gripper(self.gripper_close)     # 闭合夹爪抓住方块
        self._attach_cube()                         # MoveIt 场景：方块 attach
        self._sim_attach_cube()                     # Gazebo 仿真：方块 attach
        self._allow_cube_touch_collisions(False)   # 恢复碰撞检测（方块已是手臂一部分）

        # ── Step 6: 抬升（带着方块） ──
        log("Step 6: 抬升")
        if not self._go_pose(pick_above):
            return False

        # ── Step 7: 移动到放置点上方 ──
        log("Step 7: 移动到放置点上方，夹爪水平")
        if not self._go_pose(place_above):
            return False

        # ── Step 8: 下降到放置点 ──
        log("Step 8: 下降到放置点")
        if not self._go_pose(place_at):
            return False

        # ── Step 9: 张开夹爪释放方块，先上升再 detach（避免方块挡路） ──
        log("Step 9: 张开夹爪，先上升再 detach（避免方块挡路）")
        self._move_gripper(self.gripper_open)
        self._sim_detach_cube()   # Gazebo 仿真：解绑方块
        if not self._go_pose(place_above):
            return False

        # ── Step 10: detach 方块，添加回世界碰撞场景 ──
        log("Step 10: detach 方块，添加回世界场景")
        self._detach_cube()  # MoveIt 场景：解绑方块
        self._apply_scene([self._cube_object(*place)], [])  # 方块加入放置位置

        # ── Step 11: 回到 Home 位姿 ──
        log("Step 11: MoveIt 关节目标回 Home")
        self._go_home()

        log("===== MoveIt Pick & Place 完成 =====")
        return True

    def _point(self, prefix: str) -> tuple[float, float, float]:
        """从参数读取坐标元组，如 _point("cube") → (cube_x, cube_y, cube_z)"""
        return (
            float(self.get_parameter(f"{prefix}_x").value),
            float(self.get_parameter(f"{prefix}_y").value),
            float(self.get_parameter(f"{prefix}_z").value),
        )

    def _pre_height(self) -> float:
        """抓取/放置前在上方的抬高高度"""
        return float(self.get_parameter("pre_height").value)

    def _pick_height(self) -> float:
        """下降到抓取目标的高度偏移"""
        return float(self.get_parameter("pick_height").value)

    @staticmethod
    def _base_to_world_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
        """base_link 坐标 → world 坐标：加上固定偏移（仅 Gazebo 仿真用）"""
        return (
            point[0] + SIM_BASE_WORLD_X,
            point[1] + SIM_BASE_WORLD_Y,
            point[2] + SIM_BASE_WORLD_Z,
        )

    @staticmethod
    def _world_to_base_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
        """world 坐标 → base_link 坐标：减去固定偏移"""
        return (
            point[0] - SIM_BASE_WORLD_X,
            point[1] - SIM_BASE_WORLD_Y,
            point[2] - SIM_BASE_WORLD_Z,
        )

    def _go_pose(self, pose: Pose) -> bool:
        """通过 TCP 位置+姿态约束让 MoveIt 规划到目标位姿"""
        current = self._controller.get_current_joints()
        constraints = Constraints()
        constraints.name = "gripper_tcp_pose"  # 约束组名称

        # ── 位置约束：TCP 在目标位置 1.5cm 球体内 ──
        pc = PositionConstraint()
        pc.header = Header(frame_id=TASK_FRAME)                # 参考坐标系：base_link
        pc.link_name = "gripper_tcp"                           # 约束的目标 link
        pc.weight = 1.0                                         # 约束权重（1.0 = 必须满足）
        pc.constraint_region = BoundingVolume(
            primitives=[make_sphere(0.015)],                    # 容差球体：半径 0.015m = 1.5cm
            primitive_poses=[Pose(
                position=pose.position,                         # 球心 = 目标位置
                orientation=Quaternion(w=1.0)                   # 球体无旋转
            )],
        )

        # ── 姿态约束：TCP 朝向与目标朝向偏差 < 0.2 rad ≈ 11.5° ──
        oc = OrientationConstraint()
        oc.header = Header(frame_id=TASK_FRAME)                # 参考坐标系：base_link
        oc.link_name = "gripper_tcp"                           # 约束的目标 link
        oc.orientation = pose.orientation                       # 目标姿态（四元数）
        oc.absolute_x_axis_tolerance = 0.2                     # X 轴角度容差（弧度）
        oc.absolute_y_axis_tolerance = 0.2                     # Y 轴角度容差（弧度）
        oc.absolute_z_axis_tolerance = 0.2                     # Z 轴角度容差（弧度）
        oc.weight = 1.0                                         # 约束权重

        constraints.position_constraints = [pc]                 # 添加位置约束
        constraints.orientation_constraints = [oc]              # 添加姿态约束

        # ── 可选 joint6 约束（默认关闭，容易卡规划器） ──
        if bool(self.get_parameter("constrain_joint6").value):
            tolerance = float(self.get_parameter("joint6_goal_tolerance").value)
            constraints.joint_constraints.append(
                JointConstraint(
                    joint_name="joint6",                        # 约束的关节名
                    position=current[5],                        # 保持当前 joint6 角度不变
                    tolerance_above=tolerance,                  # 允许正向偏差
                    tolerance_below=tolerance,                  # 允许负向偏差
                    weight=1.0,                                 # 约束权重
                )
            )
        return self._controller.moveit_execute(constraints)

    def _go_home(self) -> bool:
        """通过关节约束让 MoveIt 规划回到 Home 位姿（每个关节容差 ±0.02 rad）"""
        constraints = Constraints(name="home")
        constraints.joint_constraints = [
            JointConstraint(
                joint_name=name,            # 关节名
                position=pos,               # 目标角度（HOME 值）
                tolerance_above=0.02,       # 允许正向偏差（弧度）
                tolerance_below=0.02,       # 允许负向偏差（弧度）
                weight=1.0,                 # 约束权重
            )
            for name, pos in zip(ARM_JOINTS, HOME)  # 遍历所有六轴关节
        ]
        return self._controller.moveit_execute(constraints)

    def _allow_cube_touch_collisions(self, allowed: bool) -> bool:
        """允许/禁止夹爪 touch_links 与 green_cube 碰撞（避免 OMPL 在接近方块时规划失败）"""
        acm = self._current_acm()             # 读取当前 Allowed Collision Matrix
        if acm is None:
            return False
        for link in TOUCH_LINKS:
            self._set_acm_entry(acm, "green_cube", link, allowed)  # 逐对设置碰撞许可
        scene = PlanningScene(is_diff=True)
        scene.allowed_collision_matrix = acm  # 写回 ACM
        return self._apply_scene_msg(scene)

    def _current_acm(self) -> AllowedCollisionMatrix | None:
        """从 MoveIt 读取当前场景的 Allowed Collision Matrix"""
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX  # 只请求 ACM
        future = self._get_scene.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        res = future.result()
        if res is None:
            self.get_logger().error("Failed to read planning scene ACM")
            return None
        return res.scene.allowed_collision_matrix

    @staticmethod
    def _set_acm_entry(
        acm: AllowedCollisionMatrix,
        first: str,
        second: str,
        allowed: bool,
    ) -> None:
        """在 ACM 中设置一对物体的碰撞许可（双向）"""
        # 确保 first 和 second 都在 ACM 条目中注册
        for name in (first, second):
            if name in acm.entry_names:
                continue
            acm.entry_names.append(name)              # 添加新条目名
            for entry in acm.entry_values:
                entry.enabled.append(False)            # 已有条目末尾补 False
            acm.entry_values.append(
                AllowedCollisionEntry(enabled=[False] * len(acm.entry_names))  # 新增一行全 False
            )
        # 设置双向碰撞许可
        i = acm.entry_names.index(first)
        j = acm.entry_names.index(second)
        acm.entry_values[i].enabled[j] = allowed       # first → second
        acm.entry_values[j].enabled[i] = allowed       # second → first（对称）

    def _apply_scene_msg(self, scene: PlanningScene) -> bool:
        """发送 PlanningScene 到 MoveIt"""
        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self.scene.call_async(req)  # 异步调用 /apply_planning_scene
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        res = future.result()
        ok = bool(res and res.success)
        if not ok:
            self.get_logger().warn("apply planning scene failed")
        return ok

    def _apply_scene(
        self,
        objects: list[CollisionObject],
        attached: list[AttachedCollisionObject],
    ) -> bool:
        """批量添加/更新世界碰撞物体和 attached 物体到 planning scene"""
        scene = PlanningScene(is_diff=True)                    # 增量更新场景
        scene.world.collision_objects = objects                # 世界碰撞物体列表
        scene.robot_state.attached_collision_objects = attached  # attached 碰撞物体列表
        scene.robot_state.is_diff = True                       # 标记 robot_state 也是增量
        return self._apply_scene_msg(scene)

    def _cube_object(self, x: float, y: float, z: float, operation=CollisionObject.ADD) -> CollisionObject:
        """构造方块 CollisionObject（绿色正方体）"""
        return CollisionObject(
            id="green_cube",                                                      # 唯一标识
            header=Header(frame_id=TASK_FRAME),                                   # 参考坐标系
            primitives=[make_box(self.cube_size, self.cube_size, self.cube_size)],  # 正方体形状
            primitive_poses=[make_pose(x, y, z, (0.0, 0.0, 0.0, 1.0))],         # 位置（无旋转）
            operation=operation,                                                  # ADD / REMOVE
        )

    def _table_objects(self) -> list[CollisionObject]:
        """构造桌子的碰撞物体列表：1 个桌面 + 4 条桌腿"""
        # 桌面中心（world → base_link）
        table_center = self._world_to_base_point(
            (TABLE_X, TABLE_Y, TABLE_Z + TABLE_TOP_Z)  # world 坐标桌面中心
        )
        objects = [
            CollisionObject(
                id="gazebo_table_surface",                                        # 桌面唯一标识
                header=Header(frame_id=TASK_FRAME),
                primitives=[make_box(TABLE_TOP_LENGTH, TABLE_TOP_WIDTH, TABLE_TOP_THICKNESS)],  # 桌面长方体
                primitive_poses=[yaw_pose(*table_center, TABLE_YAW)],            # 桌面位姿（含偏航角）
                operation=CollisionObject.ADD,
            )
        ]

        # ── 四条桌腿（圆柱体） ──
        cos_yaw = math.cos(TABLE_YAW)
        sin_yaw = math.sin(TABLE_YAW)
        local = [
            (TABLE_LEG_X_OFFSET, TABLE_LEG_Y_OFFSET),   # 左前
            (TABLE_LEG_X_OFFSET, -TABLE_LEG_Y_OFFSET),  # 右前
            (-TABLE_LEG_X_OFFSET, -TABLE_LEG_Y_OFFSET), # 右后
            (-TABLE_LEG_X_OFFSET, TABLE_LEG_Y_OFFSET),  # 左后
        ]
        for index, (x, y) in enumerate(local, start=1):
            wx = TABLE_X + cos_yaw * x - sin_yaw * y   # world X（考虑桌子偏航旋转）
            wy = TABLE_Y + sin_yaw * x + cos_yaw * y   # world Y
            leg_center = self._world_to_base_point(
                (wx, wy, TABLE_Z + TABLE_LEG_HEIGHT / 2.0)  # world 坐标桌腿中心（Z 中位）
            )
            objects.append(
                CollisionObject(
                    id=f"gazebo_table_leg_{index}",                                # 桌腿唯一标识
                    header=Header(frame_id=TASK_FRAME),
                    primitives=[make_cylinder(TABLE_LEG_RADIUS, TABLE_LEG_HEIGHT)], # 圆柱体桌腿
                    primitive_poses=[yaw_pose(*leg_center, 0.0)],                  # 桌腿位姿（无旋转）
                    operation=CollisionObject.ADD,
                )
            )
        return objects

    def _attach_cube(self) -> None:
        """将方块从世界场景移除并 attach 到 gripper_tcp（MoveIt 场景）"""
        # 先从世界场景中移除方块
        remove = CollisionObject(id="green_cube", header=Header(frame_id=TASK_FRAME))
        remove.operation = CollisionObject.REMOVE

        # 再作为 attached 物体挂到 gripper_tcp
        attached_object = CollisionObject(
            id="green_cube",
            header=Header(frame_id="gripper_tcp"),                                  # 坐标系相对 gripper_tcp
            primitives=[make_box(self.cube_size, self.cube_size, self.cube_size)],  # 正方体
            primitive_poses=[make_pose(0.0, 0.0, -0.03, (0.0, 0.0, 0.0, 1.0))],   # 相对 TCP 向下 3cm
            operation=CollisionObject.ADD,
        )
        attached = AttachedCollisionObject(
            link_name="gripper_tcp",                                                # 附着目标 link
            object=attached_object,
            touch_links=["gripper_tcp", "gripper_link", "gripper_left", "gripper_right"],  # 触碰允许
        )
        if self._apply_scene([remove], [attached]):
            self.get_logger().info("green_cube attached to gripper_tcp in RViz planning scene")

    def _detach_cube(self) -> None:
        """将方块从 gripper_tcp 上 detach（MoveIt 场景）"""
        attached_object = CollisionObject(id="green_cube")
        attached_object.operation = CollisionObject.REMOVE                           # 删除 attached 物体
        attached = AttachedCollisionObject(link_name="gripper_tcp", object=attached_object)
        if self._apply_scene([], [attached]):
            self.get_logger().info("green_cube detached from gripper_tcp in RViz planning scene")

    def _move_gripper(self, position: float) -> None:
        """控制夹爪移动到目标宽度（模拟或真机模式）"""
        if self.mode == "sim":
            self.gripper_pub.publish(gripper_traj(position))  # 仿真：发布 JointTrajectory
            time.sleep(0.6)                                    # 等待 0.6s 完成运动
            return

        # ── 真机模式：通过 GripperCommand action ──
        if self.hardware_gripper is None or not self.hardware_gripper.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("真机 gripper action 不可用")
            return
        hardware_position = self._hardware_gripper_position(position)  # 模拟值 → 硬件值映射
        goal = GripperCommand.Goal()
        goal.command.position = hardware_position                              # 目标位置（硬件值）
        goal.command.max_effort = float(self.get_parameter("gripper_max_effort").value)  # 最大力矩
        future = self.hardware_gripper.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle and handle.accepted:
            result_future = handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)
        self.get_logger().info(
            f"gripper sim_width={position:.4f} hardware={hardware_position:.4f}"
        )

    def _hardware_gripper_position(self, sim_position: float) -> float:
        """将模拟夹爪宽度映射到硬件指令值（线性映射）"""
        max_width = float(self.get_parameter("max_gripper_width").value)  # 最大开口宽度
        ratio = 0.0 if max_width <= 0.0 else 2.0 * float(sim_position) / max_width  # 软硬件比值映射
        ratio = max(0.0, min(1.0, ratio))                                    # 限制在 [0, 1]
        open_position = float(self.get_parameter("hardware_open_gripper_position").value)    # 硬件张开值（DM: -5.0）
        closed_position = float(self.get_parameter("hardware_closed_gripper_position").value) # 硬件闭合值（DM: 0.0）
        return closed_position + (open_position - closed_position) * ratio    # 线性插值

    def _sim_attach_cube(self) -> None:
        """Gazebo 仿真中将方块 attach 到夹爪（仅 sim 模式生效）"""
        if self.mode != "sim" or self.cube is None:
            return
        self.cube.attach()

    def _sim_detach_cube(self) -> None:
        """Gazebo 仿真中将方块 detach（仅 sim 模式生效）"""
        if self.mode != "sim" or self.cube is None:
            return
        self.cube.detach()


def main(args=None) -> None:
    """ROS2 入口：初始化节点 → 执行流程 → 销毁退出"""
    rclpy.init(args=args)                     # 初始化 rclpy
    node = MoveItPickPlace()                  # 创建节点
    try:
        node.run()                            # 执行 Pick & Place 流程
    finally:
        node.destroy_node()                   # 销毁节点
        rclpy.shutdown()                      # 关闭 rclpy


if __name__ == "__main__":
    main()
