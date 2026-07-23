"""Simple pick and place demo.

sim:
  - spawn a green cube on the table
  - ask MoveIt /compute_ik for TCP poses
  - send joint targets to Gazebo ros2_control
  - use DetachableJoint for stable simulated gripping

hardware:
  - use /rebotarm/move_to_pose_ik or /compute_ik for IK (see hw_ik_solver)
  - use /rebotarm/follow_joint_trajectory for arm motion
  - use /rebotarm/gripper/command action for gripper
"""

from __future__ import annotations

import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from control_msgs.action import GripperCommand as GripperAction
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rebotarm_gazebo.cube_spawner import CubeSpawner
from rebotarm_gazebo.real_controller import (
    ARM_JOINTS,
    GRIPPER_JOINTS,
    JOINT_LIMIT_EPS,
    JOINT_POSITION_LIMITS,
    build_single_point_trajectory as one_point_trajectory,
    normalize_joint_values,
)
from rebotarm_msgs.srv import MoveToPoseIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf_transformations import quaternion_from_euler
from trajectory_msgs.msg import JointTrajectory

TABLE_TOP_Z = 0.260  # 桌面板顶面 world z


def make_pose(x: float, y: float, z: float, rpy: tuple[float, float, float]) -> Pose:
    qx, qy, qz, qw = quaternion_from_euler(*rpy)
    return Pose(
        position=Point(x=float(x), y=float(y), z=float(z)),
        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
    )

class SimplePickPlace(Node):
    def __init__(self) -> None:
        super().__init__("simple_pick_place")

        self.declare_parameter("mode", "sim")
        self.declare_parameter("namespace", "rebotarm")
        self.declare_parameter("cube_x", 0.35)
        self.declare_parameter("cube_y", 0.15)
        self.declare_parameter("cube_z", -1.0)
        self.declare_parameter("place_x", 0.45)
        self.declare_parameter("place_y", -0.10)
        self.declare_parameter("place_z", -1.0)
        self.declare_parameter("cube_size", 0.06)

        self.declare_parameter("pre_height", 0.12)
        self.declare_parameter("pick_tcp_offset", 0.05)
        self.declare_parameter("place_tcp_offset", 0.05)
        self.declare_parameter("gripper_open", 0.06)
        self.declare_parameter("gripper_close", -1.0)
        self.declare_parameter("max_gripper_width", 0.09)
        self.declare_parameter("closed_gripper_position", 0.0)
        self.declare_parameter("hardware_open_gripper_position", -5.0)
        self.declare_parameter("hardware_closed_gripper_position", 0.0)
        self.declare_parameter("gripper_max_effort", 10.0)
        self.declare_parameter("move_duration", 3.0)
        self.declare_parameter("arm_result_timeout", 30.0)

        self.declare_parameter("tcp_rpy", [0.0, 1.5708, 0.0])
        self.declare_parameter("home_joints", [0.0, -0.05, -0.05, 0.0, 0.0, 0.0])

        self.declare_parameter("ik_link_name", "gripper_tcp")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("avoid_collisions", False)

        # 真机 IK: "sdk" → /move_to_pose_ik, "moveit" → /compute_ik
        self.declare_parameter("hw_ik_solver", "sdk")

        self.mode = str(self.get_parameter("mode").value).lower()
        self.namespace = str(self.get_parameter("namespace").value).strip("/")
        self.is_sim = self.mode == "sim"

        arm_action_name = (
            "/rebotarm_controller/follow_joint_trajectory"
            if self.is_sim
            else f"/{self.namespace}/follow_joint_trajectory"
        )
        joint_states_topic = "/joint_states" if self.is_sim else f"/{self.namespace}/joint_states"
        self.arm_action = ActionClient(self, FollowJointTrajectory, arm_action_name)
        self.create_subscription(
            JointState, joint_states_topic, self._joint_state_cb, qos_profile_sensor_data
        )

        if self.is_sim:
            self.gripper_pub = self.create_publisher(
                JointTrajectory, "/gripper_controller/joint_trajectory", 10
            )
            self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
            self.hw_ik_client = None
            self.gripper_client = None
            self.cube = CubeSpawner(self, size=float(self.get_parameter("cube_size").value))
        else:
            self.gripper_pub = None
            self.cube = None
            self.gripper_client = ActionClient(
                self, GripperAction, f"/{self.namespace}/gripper/command"
            )
            hw_ik = str(self.get_parameter("hw_ik_solver").value).lower()
            if hw_ik == "moveit":
                self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
                self.hw_ik_client = None
            else:
                self.ik_client = None
                self.hw_ik_client = self.create_client(
                    MoveToPoseIK, f"/{self.namespace}/move_to_pose_ik"
                )
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

    def _joints_within_limits(self, joints: list[float], label: str) -> bool:
        """检查最终关节目标是否仍在 DM URDF 定义的旋转角范围内。

        这里的输入已经经过 normalize_joint_values 归一化，因此检查的是准备
        发给真机控制器的实际目标角，而不是 IK 原始返回的等效角。
        """
        for name, value in zip(ARM_JOINTS, joints):
            lower, upper = JOINT_POSITION_LIMITS[name]
            if lower - JOINT_LIMIT_EPS <= float(value) <= upper + JOINT_LIMIT_EPS:
                continue
            self.get_logger().error(
                f"{label}: {name}={value:.4f} 超出关节角限制 "
                f"[{lower:.4f}, {upper:.4f}]"
            )
            return False
        return True

    # ------------------------------------------------------------------
    #  IK
    # ------------------------------------------------------------------
    def _solve_ik(self, pose: Pose, seed: list[float], label: str) -> list[float] | None:
        if self.ik_client is None:
            return None

        state = RobotState()
        state.joint_state.name = list(ARM_JOINTS)
        state.joint_state.position = [float(v) for v in seed]

        request = GetPositionIK.Request()
        request.ik_request.group_name = "arm"
        request.ik_request.robot_state = state
        request.ik_request.ik_link_name = str(self.get_parameter("ik_link_name").value)
        request.ik_request.avoid_collisions = bool(self.get_parameter("avoid_collisions").value)
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
        joints = normalize_joint_values(joints, seed)
        self.get_logger().info(f"{label}: {[round(v, 4) for v in joints]}")
        return joints

    def _solve_ik_hardware(self, pose: Pose, seed: list[float], label: str) -> list[float] | None:
        if self.hw_ik_client is None:
            return None
        if not self.hw_ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f"{label}: /move_to_pose_ik not ready")
            return None
        req = MoveToPoseIK.Request()
        req.target_pose = pose
        future = self.hw_ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        res = future.result()
        if res is None or not res.success:
            self.get_logger().error(f"{label}: SDK IK failed")
            return None
        joints = [float(v) for v in res.q_solution]
        joints = normalize_joint_values(joints, seed)
        self.get_logger().info(f"{label}: {[round(v, 4) for v in joints]}")
        return joints

    # ------------------------------------------------------------------
    #  Hardware helpers
    # ------------------------------------------------------------------
    def _enable_robot(self) -> None:
        client = self.create_client(Trigger, f"/{self.namespace}/enable")
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("enable service not ready")
            return
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        res = future.result()
        if res is not None and res.success:
            self.get_logger().info("robot enabled")
        else:
            msg = res.message if res is not None else "no response"
            self.get_logger().warn(f"enable failed: {msg}")

    # ------------------------------------------------------------------
    #  Arm / Gripper motion
    # ------------------------------------------------------------------
    def _move_arm(self, joints: list[float], label: str) -> bool:
        duration = float(self.get_parameter("move_duration").value)
        current = self._current_arm_joints()
        joints = normalize_joint_values(joints, current)
        if not self._joints_within_limits(joints, label):
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = one_point_trajectory(ARM_JOINTS, joints, duration)

        self.get_logger().info(
            f"move arm -> {label} | "
            f"current={[round(v, 4) for v in current]} | "
            f"target={[round(v, 4) for v in joints]} | "
            f"duration={duration:.1f}s"
        )
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
            message = result.result.error_string if result is not None else ""
            self.get_logger().error(f"{label}: arm failed, code={code}, message={message}")
            return False
        self.get_logger().info(f"{label}: arm done")
        return True

    def _move_pose(self, pose: Pose, seed: list[float], label: str) -> tuple[bool, list[float]]:
        if self.is_sim or self.ik_client is not None:
            joints = self._solve_ik(pose, seed, label)       # MoveIt /compute_ik
        else:
            joints = self._solve_ik_hardware(pose, seed, label)  # SDK /move_to_pose_ik
        if joints is None:
            return False, seed
        return self._move_arm(joints, label), joints

    def _move_gripper(self, position: float, label: str) -> bool:
        gripper_open = float(self.get_parameter("gripper_open").value)
        position = max(0.0, min(float(position), gripper_open))
        if not self.is_sim:
            return self._move_gripper_hardware(position, label)

        if self.gripper_pub is None:
            return False
        traj = one_point_trajectory(GRIPPER_JOINTS, [position, position], 0.6)
        self.gripper_pub.publish(traj)
        self.get_logger().info(f"{label}: gripper -> {position:.4f}")
        time.sleep(0.8)
        return True

    def _move_gripper_hardware(self, position: float, label: str) -> bool:
        if self.gripper_client is None:
            return False
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(f"{label}: /gripper/command not ready")
            return False

        hardware_position = self._hardware_gripper_position(position)
        goal = GripperAction.Goal()
        goal.command.position = hardware_position
        goal.command.max_effort = float(self.get_parameter("gripper_max_effort").value)
        self.get_logger().info(
            f"{label}: gripper sim_width={position:.4f} hardware={hardware_position:.4f}"
        )
        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(f"{label}: gripper goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().warn(f"{label}: gripper result timeout")
            return False
        result = result_future.result()
        r = result.result
        self.get_logger().info(
            f"{label}: gripper done | reached={r.reached_goal} stalled={r.stalled} "
            f"position={r.position:.4f} effort={r.effort:.4f}"
        )
        if not r.reached_goal:
            self.get_logger().warn(f"{label}: gripper did not reach target, stop task")
            return False
        return True

    def _hardware_gripper_position(self, sim_position: float) -> float:
        max_width = float(self.get_parameter("max_gripper_width").value)
        ratio = 0.0 if max_width <= 0.0 else 2.0 * float(sim_position) / max_width
        ratio = max(0.0, min(1.0, ratio))
        open_position = float(self.get_parameter("hardware_open_gripper_position").value)
        closed_position = float(self.get_parameter("hardware_closed_gripper_position").value)
        return closed_position + (open_position - closed_position) * ratio

    # ------------------------------------------------------------------
    #  Main
    # ------------------------------------------------------------------
    def run(self) -> bool:
        if self.is_sim:
            if not self.arm_action.wait_for_server(timeout_sec=20.0):
                self.get_logger().error("follow_joint_trajectory action not ready")
                return False
            if self.ik_client and not self.ik_client.wait_for_service(timeout_sec=20.0):
                self.get_logger().error("/compute_ik not ready")
                return False
        else:
            if not self.arm_action.wait_for_server(timeout_sec=20.0):
                self.get_logger().error("follow_joint_trajectory action not ready")
                return False
            ik_ready = (
                self.ik_client.wait_for_service(timeout_sec=20.0)
                if self.ik_client is not None
                else self.hw_ik_client.wait_for_service(timeout_sec=20.0)
            )
            if not ik_ready:
                ik_name = "/compute_ik" if self.ik_client is not None else "/move_to_pose_ik"
                self.get_logger().error(f"{ik_name} not ready")
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

        cube_z = float(self.get_parameter("cube_z").value)
        if cube_z < 0.0:
            cube_z = self._cube_center_z()
        base_place_z = float(self.get_parameter("place_z").value)
        if base_place_z < 0.0:
            base_place_z = cube_z
        pick_z = cube_z + pick_offset
        place_z = base_place_z + place_offset

        gripper_open = float(self.get_parameter("gripper_open").value)
        gripper_close = float(self.get_parameter("gripper_close").value)
        if gripper_close < 0.0:
            gripper_close = size / 2.0 + 0.001
        gripper_close = max(0.0, min(gripper_close, gripper_open))
        home = [float(v) for v in self.get_parameter("home_joints").value]

        self.get_logger().info(f"===== simple pick place ({self.mode}) =====")
        self.get_logger().info(
            f"cube=({cube_x:.3f},{cube_y:.3f},{cube_z:.3f}) "
            f"place=({place_x:.3f},{place_y:.3f},{place_z:.3f})"
        )
        self.get_logger().info(f"gripper open={gripper_open:.4f} close={gripper_close:.4f}")

        # 1. sim 生成方块；hardware 下默认方块已经由人工摆好
        if self.is_sim and self.cube is not None:
            self.cube.spawn(cube_x, cube_y, cube_z)
            self.cube.detach()

        # 2. hardware: 使能机械臂（夹爪进入 MIT 模式），否则 arm 轨迹报 -4
        if not self.is_sim:
            self._enable_robot()

        # 3. home。硬件模式先让 arm 进入稳定保持，再控制夹爪。
        seed = self._current_arm_joints()
        if not self._move_arm(home, "home"):
            return False
        seed = home
        if not self._move_gripper(gripper_open, "open"):
            return False

        # 3. pick approach + pick
        pick_above = make_pose(cube_x - 0.05, cube_y, pick_z - 0.265 + pre_height, rpy)
        pick_at = make_pose(cube_x - 0.05, cube_y, pick_z - 0.265, rpy)
        approach_joints = seed
        for label, pose in [("pick_above", pick_above), ("pick_down", pick_at)]:
            ok, seed = self._move_pose(pose, seed, label)
            if label == "pick_above":
                approach_joints = list(seed)
            if not ok:
                return False

        # 4. close gripper, then attach
        if not self._move_gripper(gripper_close, "close"):
            return False
        if self.is_sim and self.cube is not None:
            self.cube.attach()

        # 5. retract → place approach → place down
        place_above = make_pose(place_x - 0.05, place_y, place_z - 0.265 + pre_height, rpy)
        place_at = make_pose(place_x - 0.05, place_y, place_z - 0.265, rpy)
        seed = approach_joints
        for label, pose in [
            ("pick_up", pick_above),
            ("place_above", place_above),
            ("place_down", place_at),
        ]:
            ok, seed = self._move_pose(pose, seed, label)
            if not ok:
                return False

        # 6. release + rise + home
        if not self._move_gripper(gripper_open, "release"):
            return False
        if self.is_sim and self.cube is not None:
            self.cube.detach()
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
