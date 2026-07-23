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
from cube_spawner import CubeSpawner
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PlanningScene,
    PositionConstraint,
)
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["gripper_joint1", "gripper_joint2"]
HOME = [0.0, -0.05, -0.05, 0.0, 0.0, 0.0]

TABLE_X = 0.28
TABLE_Y = 0.0
TABLE_Z = 0.0
TABLE_YAW = 1.5708
TABLE_TOP_LENGTH = 0.4
TABLE_TOP_WIDTH = 0.6
TABLE_TOP_THICKNESS = 0.03
TABLE_TOP_Z = 0.260
TABLE_LEG_RADIUS = 0.02
TABLE_LEG_HEIGHT = 0.245
TABLE_LEG_X_OFFSET = 0.17
TABLE_LEG_Y_OFFSET = 0.27


def make_pose(x: float, y: float, z: float, q: tuple[float, float, float, float]) -> Pose:
    return Pose(
        position=Point(x=float(x), y=float(y), z=float(z)),
        orientation=Quaternion(
            x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])
        ),
    )


def make_box(size_x: float, size_y: float, size_z: float) -> SolidPrimitive:
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [size_x, size_y, size_z]
    return box


def make_sphere(radius: float) -> SolidPrimitive:
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [radius]
    return sphere


def make_cylinder(radius: float, height: float) -> SolidPrimitive:
    cylinder = SolidPrimitive()
    cylinder.type = SolidPrimitive.CYLINDER
    cylinder.dimensions = [height, radius]
    return cylinder


def yaw_pose(x: float, y: float, z: float, yaw: float) -> Pose:
    return Pose(
        position=Point(x=float(x), y=float(y), z=float(z)),
        orientation=Quaternion(
            z=float(math.sin(yaw / 2.0)),
            w=float(math.cos(yaw / 2.0)),
        ),
    )


def gripper_traj(position: float, seconds: float = 0.5) -> JointTrajectory:
    msg = JointTrajectory()
    msg.joint_names = list(GRIPPER_JOINTS)
    point = JointTrajectoryPoint()
    point.positions = [position, position]
    sec = int(seconds)
    point.time_from_start = Duration(
        sec=sec, nanosec=int((seconds - sec) * 1_000_000_000)
    )
    msg.points = [point]
    return msg


class MoveItPickPlace(Node):
    def __init__(self) -> None:
        super().__init__("moveit_pick_place")

        self.declare_parameter("namespace", "rebotarm")
        self.declare_parameter("cube_x", 0.35)
        self.declare_parameter("cube_y", 0.15)
        self.declare_parameter("cube_z", 0.29)
        self.declare_parameter("place_x", 0.35)
        self.declare_parameter("place_y", -0.10)
        self.declare_parameter("place_z", 0.29)
        self.declare_parameter("cube_size", 0.06)
        self.declare_parameter("pre_height", 0.12)
        self.declare_parameter("pick_height", 0.00)
        self.declare_parameter("gripper_open", 0.06)
        self.declare_parameter("gripper_close", -1.0)
        self.declare_parameter("velocity_scaling", 0.5)
        self.declare_parameter("acceleration_scaling", 0.5)

        self.namespace = str(self.get_parameter("namespace").value).strip("/")
        self.cube_size = float(self.get_parameter("cube_size").value)
        self.gripper_open = float(self.get_parameter("gripper_open").value)
        gripper_close = float(self.get_parameter("gripper_close").value)
        if gripper_close < 0.0:
            gripper_close = self.cube_size / 2.0 + 0.001
        self.gripper_close = max(0.0, min(gripper_close, self.gripper_open))

        self.move_group = ActionClient(self, MoveGroup, "/move_action")
        self.scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")

        self.cube = CubeSpawner(self, size=self.cube_size)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, "/gripper_controller/joint_trajectory", 10
        )

    def run(self) -> bool:
        log = self.get_logger().info
        log("===== MoveIt Pick & Place 开始 =====")

        cube = self._point("cube")
        place = self._point("place")
        down_q = (0.0, 0.7071068, 0.0, 0.7071068)
        horizontal_q = (0.0, 0.0, 0.0, 1.0)

        pick_above = make_pose(cube[0], cube[1], cube[2] + self._pre_height(), down_q)
        pick_at = make_pose(cube[0], cube[1], cube[2] + self._pick_height(), down_q)
        place_above = make_pose(place[0], place[1], place[2] + self._pre_height(), horizontal_q)
        place_at = make_pose(place[0], place[1], place[2] + self._pick_height() + 0.02, horizontal_q)

        if not self.move_group.wait_for_server(timeout_sec=20.0):
            self.get_logger().error("/move_action 不可用，请确认 move_group 已启动")
            return False
        if not self.scene.wait_for_service(timeout_sec=20.0):
            self.get_logger().error("/apply_planning_scene 不可用")
            return False

        log("Step 1: 添加桌子和绿色方块到 MoveIt planning scene")
        self._apply_scene(self._table_objects() + [self._cube_object(*cube)], [])
        if self.cube:
            self.cube.spawn(*cube)

        log("Step 2: 张开夹爪")
        self._move_gripper(self.gripper_open)

        log("Step 3: MoveIt 规划到方块上方")
        if not self._go_pose(pick_above):
            return False

        log("Step 4: 下降到夹取位置")
        if not self._go_pose(pick_at):
            return False

        log("Step 5: 闭合夹爪，从 planning scene 移除方块（物理靠 grasp_fix）")
        self._move_gripper(self.gripper_close)
        self._apply_scene([CollisionObject(id="green_cube", operation=CollisionObject.REMOVE)], [])

        log("Step 6: 抬升")
        if not self._go_pose(pick_above):
            return False

        log("Step 7: 移动到放置点上方，夹爪水平")
        if not self._go_pose(place_above):
            return False

        log("Step 8: 下降到放置点")
        if not self._go_pose(place_at):
            return False

        log("Step 9: 张开夹爪，上升")
        self._move_gripper(self.gripper_open)
        if not self._go_pose(place_above):
            return False

        log("Step 10: 方块加回世界场景")
        self._apply_scene([self._cube_object(*place)], [])

        log("Step 11: MoveIt 关节目标回 Home")
        self._go_home()

        log("===== MoveIt Pick & Place 完成 =====")
        return True

    def _point(self, prefix: str) -> tuple[float, float, float]:
        return (
            float(self.get_parameter(f"{prefix}_x").value),
            float(self.get_parameter(f"{prefix}_y").value),
            float(self.get_parameter(f"{prefix}_z").value),
        )

    def _pre_height(self) -> float:
        return float(self.get_parameter("pre_height").value)

    def _pick_height(self) -> float:
        return float(self.get_parameter("pick_height").value)

    def _go_pose(self, pose: Pose) -> bool:
        constraints = Constraints()
        constraints.name = "gripper_tcp_pose"

        pc = PositionConstraint()
        pc.header = Header(frame_id="world")
        pc.link_name = "gripper_tcp"
        pc.weight = 1.0
        pc.constraint_region = BoundingVolume(
            primitives=[make_sphere(0.015)],
            primitive_poses=[Pose(position=pose.position, orientation=Quaternion(w=1.0))],
        )

        oc = OrientationConstraint()
        oc.header = Header(frame_id="world")
        oc.link_name = "gripper_tcp"
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.2
        oc.absolute_y_axis_tolerance = 0.2
        oc.absolute_z_axis_tolerance = 0.2
        oc.weight = 1.0

        constraints.position_constraints = [pc]
        constraints.orientation_constraints = [oc]
        return self._send_move_group([constraints])

    def _go_home(self) -> bool:
        constraints = Constraints(name="home")
        constraints.joint_constraints = [
            JointConstraint(
                joint_name=name,
                position=pos,
                tolerance_above=0.02,
                tolerance_below=0.02,
                weight=1.0,
            )
            for name, pos in zip(ARM_JOINTS, HOME)
        ]
        return self._send_move_group([constraints])

    def _send_move_group(self, constraints: list[Constraints]) -> bool:
        request = MotionPlanRequest()
        request.group_name = "arm"
        request.goal_constraints = constraints
        request.num_planning_attempts = 5
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = float(
            self.get_parameter("velocity_scaling").value
        )
        request.max_acceleration_scaling_factor = float(
            self.get_parameter("acceleration_scaling").value
        )

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options = PlanningOptions(plan_only=False, replan=True, replan_attempts=2)
        goal.planning_options.planning_scene_diff.is_diff = True

        future = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("MoveIt goal rejected")
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        ok = result.error_code.val == MoveItErrorCodes.SUCCESS
        self.get_logger().info(f"MoveIt result={result.error_code.val}, success={ok}")
        return ok

    def _apply_scene(
        self,
        objects: list[CollisionObject],
        attached: list[AttachedCollisionObject],
    ) -> bool:
        req = ApplyPlanningScene.Request()
        req.scene = PlanningScene(is_diff=True)
        req.scene.world.collision_objects = objects
        req.scene.robot_state.attached_collision_objects = attached
        req.scene.robot_state.is_diff = True
        future = self.scene.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        res = future.result()
        return bool(res and res.success)

    def _cube_object(self, x: float, y: float, z: float, operation=CollisionObject.ADD) -> CollisionObject:
        return CollisionObject(
            id="green_cube",
            header=Header(frame_id="world"),
            primitives=[make_box(self.cube_size, self.cube_size, self.cube_size)],
            primitive_poses=[make_pose(x, y, z, (0.0, 0.0, 0.0, 1.0))],
            operation=operation,
        )

    def _table_objects(self) -> list[CollisionObject]:
        objects = [
            CollisionObject(
                id="gazebo_table_surface",
                header=Header(frame_id="world"),
                primitives=[make_box(TABLE_TOP_LENGTH, TABLE_TOP_WIDTH, TABLE_TOP_THICKNESS)],
                primitive_poses=[yaw_pose(TABLE_X, TABLE_Y, TABLE_Z + TABLE_TOP_Z, TABLE_YAW)],
                operation=CollisionObject.ADD,
            )
        ]

        cos_yaw = math.cos(TABLE_YAW)
        sin_yaw = math.sin(TABLE_YAW)
        local = [
            (TABLE_LEG_X_OFFSET, TABLE_LEG_Y_OFFSET),
            (TABLE_LEG_X_OFFSET, -TABLE_LEG_Y_OFFSET),
            (-TABLE_LEG_X_OFFSET, -TABLE_LEG_Y_OFFSET),
            (-TABLE_LEG_X_OFFSET, TABLE_LEG_Y_OFFSET),
        ]
        for index, (x, y) in enumerate(local, start=1):
            wx = TABLE_X + cos_yaw * x - sin_yaw * y
            wy = TABLE_Y + sin_yaw * x + cos_yaw * y
            objects.append(
                CollisionObject(
                    id=f"gazebo_table_leg_{index}",
                    header=Header(frame_id="world"),
                    primitives=[make_cylinder(TABLE_LEG_RADIUS, TABLE_LEG_HEIGHT)],
                    primitive_poses=[yaw_pose(wx, wy, TABLE_Z + TABLE_LEG_HEIGHT / 2.0, 0.0)],
                    operation=CollisionObject.ADD,
                )
            )
        return objects

    def _attach_cube(self) -> None:
        remove = CollisionObject(id="green_cube", header=Header(frame_id="world"))
        remove.operation = CollisionObject.REMOVE

        attached_object = CollisionObject(
            id="green_cube",
            header=Header(frame_id="gripper_tcp"),
            primitives=[make_box(self.cube_size, self.cube_size, self.cube_size)],
            primitive_poses=[make_pose(0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 1.0))],
            operation=CollisionObject.ADD,
        )
        attached = AttachedCollisionObject(
            link_name="gripper_tcp",
            object=attached_object,
            touch_links=["gripper_tcp", "gripper_link", "gripper_left", "gripper_right"],
        )
        self._apply_scene([remove], [attached])

    def _detach_cube(self) -> None:
        attached_object = CollisionObject(id="green_cube")
        attached_object.operation = CollisionObject.REMOVE
        attached = AttachedCollisionObject(link_name="gripper_tcp", object=attached_object)
        self._apply_scene([], [attached])

    def _move_gripper(self, position: float) -> None:
        self.gripper_pub.publish(gripper_traj(position))
        time.sleep(0.7)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = MoveItPickPlace()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
