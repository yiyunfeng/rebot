"""Simple Gazebo pick and place demo.

1. spawn a green cube on the table
2. ask MoveIt /compute_ik for TCP poses (no collision avoidance)
3. send joint targets to ros2_control
4. close/open gripper; Gazebo grasp plugin handles the simulated grasp
"""

from __future__ import annotations

import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from cube_spawner import CubeSpawner
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from tf_transformations import quaternion_from_euler
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINTS = ["gripper_joint1", "gripper_joint2"]

TABLE_TOP_Z = 0.260  # 桌面板顶面 world z


def make_pose(x: float, y: float, z: float, rpy: tuple[float, float, float]) -> Pose:
    qx, qy, qz, qw = quaternion_from_euler(*rpy)
    return Pose(
        position=Point(x=float(x), y=float(y), z=float(z)),
        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )


def one_point_trajectory(
    joints: list[str], positions: list[float], seconds: float
) -> JointTrajectory:
    traj = JointTrajectory()
    traj.joint_names = list(joints)
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in positions]
    sec = int(seconds)
    point.time_from_start = Duration(
        sec=sec, nanosec=int((seconds - sec) * 1_000_000_000),
    )
    traj.points = [point]
    return traj


def nearest_angle(angle: float, reference: float) -> float:
    """返回最接近 reference 的 2*pi 等价角度。

    旋转关节的 q、q + 2*pi、q - 2*pi 表示的是同一个物理姿态。
    IK 可能返回任意一个等价值，但轨迹控制器会按数值差去运动。
    因此需要选离参考角最近的等价值，避免不必要的整圈旋转。
    """
    while angle - reference > math.pi:
        angle -= 2.0 * math.pi
    while angle - reference < -math.pi:
        angle += 2.0 * math.pi
    return angle


def nearest_joint_values(values: list[float], reference: list[float]) -> list[float]:
    return [
        nearest_angle(float(value), float(ref))
        for value, ref in zip(values, reference)
    ]


class SimplePickPlace(Node):
    def __init__(self) -> None:
        super().__init__("simple_pick_place")

        self.declare_parameter("cube_x", 0.35)
        self.declare_parameter("cube_y", 0.15)
        self.declare_parameter("place_x", 0.35)
        self.declare_parameter("place_y", -0.10)
        self.declare_parameter("cube_size", 0.06)

        self.declare_parameter("pre_height", 0.12)
        self.declare_parameter("pick_tcp_offset", 0.00)
        self.declare_parameter("place_tcp_offset", 0.00)
        self.declare_parameter("gripper_open", 0.06)
        # gripper_close < 0  → 自动按 cube_size 算
        self.declare_parameter("gripper_close", -1.0)
        self.declare_parameter("gripper_duration", 1.0)
        self.declare_parameter("move_duration", 1.5)
        self.declare_parameter("arm_result_timeout", 30.0)

        self.declare_parameter("tcp_rpy", [0.0, 1.5708, 0.0])
        self.declare_parameter("home_joints", [0.0, -0.05, -0.05, 0.0, 0.0, 0.0])

        self.declare_parameter("ik_link_name", "gripper_tcp")
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("avoid_collisions", False)

        self.arm_action = ActionClient(
            self, FollowJointTrajectory, "/rebotarm_controller/follow_joint_trajectory"
        )
        self.gripper_pub = self.create_publisher(
            JointTrajectory, "/gripper_controller/joint_trajectory", 10
        )
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, qos_profile_sensor_data
        )

        size = float(self.get_parameter("cube_size").value)
        self.cube = CubeSpawner(self, size=size)
        self.latest_joints: dict[str, float] = {}

    # ------------------------------------------------------------------
    def _joint_state_cb(self, msg: JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            self.latest_joints[name] = float(position)

    def _current_arm_joints(self) -> list[float]:
        home = [float(v) for v in self.get_parameter("home_joints").value]
        return [self.latest_joints.get(name, home[i]) for i, name in enumerate(ARM_JOINTS)]

    def _cube_center_z(self) -> float:
        size = float(self.get_parameter("cube_size").value)
        return TABLE_TOP_Z + size / 2.0

    # ------------------------------------------------------------------
    #  IK
    # ------------------------------------------------------------------
    def _solve_ik(self, pose: Pose, seed: list[float], label: str) -> list[float] | None:
        state = RobotState()
        state.joint_state.name = list(ARM_JOINTS)
        state.joint_state.position = [float(v) for v in seed]

        request = GetPositionIK.Request()
        request.ik_request.group_name = "arm"
        request.ik_request.robot_state = state
        request.ik_request.ik_link_name = str(self.get_parameter("ik_link_name").value)
        request.ik_request.avoid_collisions = bool(
            self.get_parameter("avoid_collisions").value
        )
        request.ik_request.pose_stamped = PoseStamped(
            header=Header(frame_id=str(self.get_parameter("frame_id").value)),
            pose=pose,
        )
        request.ik_request.timeout = Duration(sec=2)

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if not future.done() or future.result() is None:
            self.get_logger().error(f"{label}: IK service timeout")
            return None

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"{label}: IK failed, code={response.error_code.val}")
            return None

        joint_map = dict(
            zip(response.solution.joint_state.name, response.solution.joint_state.position)
        )
        joints = [float(joint_map[name]) for name in ARM_JOINTS]
        # 第一层处理：让 IK 结果相对求解 seed 连续，避免 IK 跳到另一个
        # 2*pi 等价分支。
        joints = nearest_joint_values(joints, seed)
        self.get_logger().info(f"{label}: {[round(v, 4) for v in joints]}")
        return joints

    # ------------------------------------------------------------------
    #  Arm / Gripper motion
    # ------------------------------------------------------------------
    def _move_arm(self, joints: list[float], label: str) -> bool:
        duration = float(self.get_parameter("move_duration").value)
        current = self._current_arm_joints()
        # 第二层处理：真正发送轨迹前，再让目标角相对 Gazebo 当前关节状态
        # 连续。这样即使仿真里的 joint6 / 腕部已经处在另一个数值圈，也不会
        # 因为目标角差值过大而整圈旋转。
        joints = nearest_joint_values(joints, current)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = one_point_trajectory(ARM_JOINTS, joints, duration)

        self.get_logger().info(f"move arm -> {label}")
        future = self.arm_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"{label}: arm goal rejected")
            return False

        timeout = float(self.get_parameter("arm_result_timeout").value)
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done():
            self.get_logger().error(f"{label}: arm timeout after {timeout:.1f}s")
            return False
        result = result_future.result()
        if result is None or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            code = result.result.error_code if result is not None else "none"
            self.get_logger().error(f"{label}: arm failed, code={code}")
            return False
        return True

    def _move_pose(self, pose: Pose, seed: list[float], label: str) -> tuple[bool, list[float]]:
        joints = self._solve_ik(pose, seed, label)
        if joints is None:
            return False, seed
        return self._move_arm(joints, label), joints

    def _move_gripper(self, position: float, label: str) -> None:
        gripper_open = float(self.get_parameter("gripper_open").value)
        duration = float(self.get_parameter("gripper_duration").value)
        position = max(0.0, min(float(position), gripper_open))
        traj = one_point_trajectory(GRIPPER_JOINTS, [position, position], duration)
        traj.points[0].velocities = [0.0, 0.0]
        self.gripper_pub.publish(traj)
        self.get_logger().info(f"{label}: gripper -> {position:.4f}")
        time.sleep(duration + 0.2)

    # ------------------------------------------------------------------
    #  Main
    # ------------------------------------------------------------------
    def run(self) -> bool:
        if not self.arm_action.wait_for_server(timeout_sec=20.0):
            self.get_logger().error("/rebotarm_controller/follow_joint_trajectory not ready")
            return False
        if not self.ik_client.wait_for_service(timeout_sec=20.0):
            self.get_logger().error("/compute_ik not ready")
            return False

        size = float(self.get_parameter("cube_size").value)
        cube_x = float(self.get_parameter("cube_x").value)
        cube_y = float(self.get_parameter("cube_y").value)
        place_x = float(self.get_parameter("place_x").value)
        place_y = float(self.get_parameter("place_y").value)
        pre_height = float(self.get_parameter("pre_height").value)
        pick_offset = float(self.get_parameter("pick_tcp_offset").value)
        place_offset = float(self.get_parameter("place_tcp_offset").value)
        rpy = tuple(float(v) for v in self.get_parameter("tcp_rpy").value)

        cube_z = self._cube_center_z()
        pick_z = cube_z + pick_offset
        place_z = cube_z + place_offset

        gripper_open = float(self.get_parameter("gripper_open").value)
        gripper_close = float(self.get_parameter("gripper_close").value)
        if gripper_close < 0.0:
            gripper_close = size / 2.0 + 0.001
        gripper_close = max(0.0, min(gripper_close, gripper_open))
        home = [float(v) for v in self.get_parameter("home_joints").value]

        self.get_logger().info("===== simple pick place =====")
        self.get_logger().info(
            f"cube=({cube_x:.3f},{cube_y:.3f},{cube_z:.3f}) "
            f"place=({place_x:.3f},{place_y:.3f},{place_z:.3f})"
        )
        self.get_logger().info(
            f"gripper open={gripper_open:.4f} close={gripper_close:.4f}"
        )

        # 1. spawn cube + open gripper
        self.cube.spawn(cube_x, cube_y, cube_z)
        self._move_gripper(gripper_open, "open")

        # 2. home
        seed = self._current_arm_joints()
        if not self._move_arm(home, "home"):
            return False
        seed = home

        # 3. pick approach + pick down
        pick_above = make_pose(cube_x, cube_y, pick_z + pre_height, rpy)
        pick_at = make_pose(cube_x, cube_y, pick_z, rpy)
        ok, seed = self._move_pose(pick_above, seed, "pick_above")
        if not ok:
            return False
        approach_joints = list(seed)
        ok, seed = self._move_pose(pick_at, seed, "pick_down")
        if not ok:
            return False

        # 4. close gripper
        self._move_gripper(gripper_close, "close")

        # 5. retract → place approach → place down
        ok, seed = self._move_pose(pick_above, approach_joints, "pick_up")

        place_above = make_pose(place_x, place_y, place_z + pre_height, rpy)
        place_at = make_pose(place_x, place_y, place_z, rpy)
        ok, seed = self._move_pose(place_above, seed, "place_above")
        if not ok:
            return False
        ok, seed = self._move_pose(place_at, seed, "place_down")
        if not ok:
            return False

        # 6. release + retreat + home
        self._move_gripper(gripper_open, "release")
        ok, seed = self._move_pose(place_above, seed, "place_up")
        if not ok:
            return False
        self._move_arm(home, "home")

        self.get_logger().info("===== simple pick place done =====")
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimplePickPlace()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
