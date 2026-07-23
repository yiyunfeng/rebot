"""
向 MoveIt 规划场景中添加桌面碰撞物体。

用途：Gazebo 仿真中机械臂放在桌面上，MoveIt 规划轨迹时需要知道桌面的位置，
否则可能规划出穿过桌面的轨迹。这个节点向 MoveIt 发送桌面（面板 + 4条腿）
作为碰撞物体，让 MoveIt 在规划时自动避开。
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header

# ---------------------------------------------------------------------------
# 桌面尺寸常量（单位：米）
# ---------------------------------------------------------------------------
TABLE_X = 0.28         # 桌子在世界坐标系中的 X 位置
TABLE_Y = 0.0          # 桌子在世界坐标系中的 Y 位置
TABLE_Z = 0.0          # 桌子在地面上的 Z 位置
TABLE_YAW = 1.5708     # 桌子绕 Z 轴旋转（约 90 度）

# 桌面面板
TABLE_TOP_LENGTH = 0.4      # 面板长度
TABLE_TOP_WIDTH = 0.6       # 面板宽度
TABLE_TOP_THICKNESS = 0.03  # 面板厚度
TABLE_TOP_Z = 0.245         # 面板离地高度

# 桌腿（圆柱体）
TABLE_LEG_RADIUS = 0.02       # 桌腿半径
TABLE_LEG_HEIGHT = 0.245      # 桌腿高度
TABLE_LEG_X_OFFSET = 0.17     # 桌腿 X 方向偏移（相对桌面中心）
TABLE_LEG_Y_OFFSET = 0.27     # 桌腿 Y 方向偏移（相对桌面中心）

# 四条桌腿在桌面局部坐标系中的位置（矩形四角）
_LEG_LOCAL_POSITIONS = [
    ( TABLE_LEG_X_OFFSET,  TABLE_LEG_Y_OFFSET),
    ( TABLE_LEG_X_OFFSET, -TABLE_LEG_Y_OFFSET),
    (-TABLE_LEG_X_OFFSET, -TABLE_LEG_Y_OFFSET),
    (-TABLE_LEG_X_OFFSET,  TABLE_LEG_Y_OFFSET),
]


# ---------------------------------------------------------------------------
# 辅助函数：创建碰撞物体的形状和位姿
# ---------------------------------------------------------------------------

def _make_box(length: float, width: float, height: float) -> SolidPrimitive:
    """创建一个长方体形状，用于桌面面板。"""
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [length, width, height]
    return box


def _make_cylinder(radius: float, height: float) -> SolidPrimitive:
    """创建一个圆柱体形状，用于桌腿。"""
    cylinder = SolidPrimitive()
    cylinder.type = SolidPrimitive.CYLINDER
    # 注意：CYLINDER 的 dimensions 定义是 [高度, 半径]
    cylinder.dimensions = [height, radius]
    return cylinder


def _make_pose(x: float, y: float, z: float, yaw: float = 0.0) -> Pose:
    """创建位姿：位置 (x,y,z) + 绕 Z 轴旋转 yaw 弧度。

    绕 Z 轴的纯旋转用半角公式转换为四元数：
        qz = sin(yaw/2), qw = cos(yaw/2)
    """
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.z = math.sin(yaw / 2.0)
    pose.orientation.w = math.cos(yaw / 2.0)
    return pose


# ---------------------------------------------------------------------------
# 主节点：向 MoveIt 规划场景添加桌面碰撞物体
# ---------------------------------------------------------------------------

class PlanningSceneObjects(Node):
    """将桌面碰撞物体发布到 MoveIt 规划场景的 ROS 2 节点。

    工作流程：
        1. 等待 MoveIt 的 /apply_planning_scene 服务就绪
        2. 构造 PlanningScene 消息（is_diff=True 表示增量更新）
        3. 添加 1 个面板 + 4 条桌腿作为碰撞物体
        4. 发送请求并检查结果
    """

    def __init__(self) -> None:
        super().__init__("gazebo_planning_scene_objects")
        self.declare_parameter("apply_timeout_sec", 10.0)

        # 连接 MoveIt 的规划场景服务
        self._client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
        )

    def apply(self) -> bool:
        """向 MoveIt 发布桌面碰撞物体。

        Returns:
            True 表示成功添加，False 表示失败。
        """
        timeout = float(self.get_parameter("apply_timeout_sec").value)

        # 步骤 1：等待 MoveIt 服务可用
        if not self._client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                "MoveIt 服务 /apply_planning_scene 不可用，请确认 move_group 已启动"
            )
            return False

        # 步骤 2：构造请求（is_diff=True 意为增量更新场景）
        request = ApplyPlanningScene.Request()
        request.scene = PlanningScene(is_diff=True)
        request.scene.world.collision_objects.extend(self._build_table_objects())

        # 步骤 3：发送请求并等待响应
        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

        # 步骤 4：检查结果
        if future.result() is None:
            self.get_logger().error("添加桌面碰撞物体超时")
            return False
        if not future.result().success:
            self.get_logger().error("MoveIt 拒绝了桌面碰撞物体")
            return False

        self.get_logger().info("已成功将桌面碰撞物体添加到 MoveIt 规划场景")
        return True

    def _build_table_objects(self) -> list[CollisionObject]:
        """构建桌面碰撞物体列表（1个面板 + 4条桌腿）。

        Returns:
            CollisionObject 列表。
        """
        header = Header(frame_id="world")
        objects = []

        # --- 桌面面板 ---
        table_top = CollisionObject(
            id="gazebo_table_surface",
            header=header,
            primitives=[
                _make_box(TABLE_TOP_LENGTH, TABLE_TOP_WIDTH, TABLE_TOP_THICKNESS)
            ],
            primitive_poses=[
                _make_pose(TABLE_X, TABLE_Y, TABLE_Z + TABLE_TOP_Z, TABLE_YAW)
            ],
            operation=CollisionObject.ADD,
        )
        objects.append(table_top)

        # --- 4 条桌腿 ---
        # 桌腿在桌面局部坐标 → 世界坐标的旋转参数
        cos_yaw = math.cos(TABLE_YAW)
        sin_yaw = math.sin(TABLE_YAW)

        for i, (local_x, local_y) in enumerate(_LEG_LOCAL_POSITIONS, start=1):
            # 二维旋转：将局部偏移转到世界方向
            world_x = TABLE_X + cos_yaw * local_x - sin_yaw * local_y
            world_y = TABLE_Y + sin_yaw * local_x + cos_yaw * local_y
            # 桌腿 Z 坐标：地面开始 + 半高（因为圆柱体原点在几何中心）
            leg_z = TABLE_Z + TABLE_LEG_HEIGHT / 2.0

            leg = CollisionObject(
                id=f"gazebo_table_leg_{i}",
                header=header,
                primitives=[_make_cylinder(TABLE_LEG_RADIUS, TABLE_LEG_HEIGHT)],
                primitive_poses=[_make_pose(world_x, world_y, leg_z)],
                operation=CollisionObject.ADD,
            )
            objects.append(leg)

        return objects


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PlanningSceneObjects()
    try:
        success = node.apply()
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
