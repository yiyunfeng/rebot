"""Gazebo camera grasp pipeline.

运行命令：
    cd /home/yyf/Desktop/pythonProject/rebot/rebotarm_ros2
    source /opt/ros/humble/setup.bash
    source install/setup.bash

    # 完整仿真抓取，通常由 launch 自动启动
    ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=grasp

    # 只生成仿真方块，不执行抓取；用于 mode:=vision
    ros2 run rebotarm_gazebo camera_grasp_sim --ros-args -p execute_grasp:=false

说明：
    本文件只负责 Gazebo 仿真抓取：
      1. 用 CubeSpawner 生成带 DetachableJoint 的绿色方块；
      2. 订阅相机检测得到的目标位姿；
      3. MoveIt 规划到目标上方、下降、夹爪闭合；
      4. 调用 CubeSpawner.attach()，让方块吸附到 gripper_link；
      5. 抬升后移动到 y 轴镜像位置放置；
      6. 回 home 并闭合夹爪。
"""

from __future__ import annotations

import math
import time

import rclpy
import tf2_geometry_msgs
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from rebotarm_gazebo.cube_spawner import CubeSpawner
from rebotarm_gazebo.real_controller import RealController, make_pose_from_rpy
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

GRIPPER_JOINTS = ["gripper_joint1", "gripper_joint2"]
GRIPPER_MAX_OPEN = 0.0715
# 方块绕竖直方向旋转 90/180 度不影响夹取，因此仿真会尝试这些等效 yaw。
# 选择 joint6 变化最小的一组 IK，避免放置到 -y 时腕部突然转半圈。
TOP_DOWN_YAW_CANDIDATES = (0.0, math.pi / 2.0, -math.pi / 2.0, math.pi, -math.pi)
MAX_JOINT6_DELTA = 3.0


def gripper_trajectory(position: float, seconds: float = 0.6) -> JointTrajectory:
    """构造 Gazebo 双指夹爪轨迹，两根手指同步开合。"""
    traj = JointTrajectory()
    traj.joint_names = list(GRIPPER_JOINTS)
    point = JointTrajectoryPoint()
    point.positions = [float(position), float(position)]
    sec = int(seconds)
    point.time_from_start = Duration(
        sec=sec,
        nanosec=int((seconds - sec) * 1_000_000_000),
    )
    traj.points = [point]
    return traj


class CameraGraspSim(Node):
    """仿真相机抓取节点，参数只保留 Gazebo 需要的内容。"""

    def __init__(self) -> None:
        super().__init__("camera_grasp_sim")

        self.declare_parameter("target_pose_topic", "/dabai_camera/target_pose")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("execute_grasp", True)
        self.declare_parameter("execute_once", True)
        self.declare_parameter("retry_cooldown", 2.0)

        # 抓取高度。目标 z 来自相机深度：
        #   pre_grasp_height      : 先移动到目标上方，避免从侧面扫过桌面物体；
        #   target_x/y_compensation: 视觉目标在桌面平面内的补偿；
        #   pick_z_offset          : 保留的安全余量，防止 TCP 直接压到桌面；
        #   pick_z_compensation    : 视觉/TF 高度偏高时的向下补偿。
        # HSV 和 OBB 都只是给出目标中心/轮廓，最终夹取位姿统一在这里修正。
        self.declare_parameter("pre_grasp_height", 0.12)
        self.declare_parameter("target_x_compensation", 0.01)
        self.declare_parameter("target_y_compensation", 0.003)
        self.declare_parameter("pick_z_offset", 0.02)
        self.declare_parameter("pick_z_compensation", 0.05)
        self.declare_parameter("lift_height", 0.12)
        self.declare_parameter("min_target_z", 0.01)
        self.declare_parameter("max_target_z", 0.45)

        # 仿真夹爪参数。gripper_close < 0 时按方块边长自动算半宽，避免直接闭死。
        self.declare_parameter("gripper_open", GRIPPER_MAX_OPEN)
        self.declare_parameter("gripper_close", -1.0)
        self.declare_parameter("gripper_margin", 0.001)
        self.declare_parameter("gripper_settle_time", 1.0)
        self.declare_parameter("gripper_topic", "/gripper_controller/joint_trajectory")

        # Gazebo 方块参数是 world 坐标。world->base_link 当前为 (0.05, 0, 0.265)，
        # 默认方块中心 z=0.285，表示 5cm 方块放在桌面顶面 z=0.260 上。
        self.declare_parameter("cube_name", "green_cube")
        self.declare_parameter("cube_size", 0.05)
        self.declare_parameter("cube_x", 0.30)
        self.declare_parameter("cube_y", 0.15)
        self.declare_parameter("cube_z", 0.285)

        self._has_executed = False
        self._is_running = False
        self._should_exit = False
        self._next_attempt_time = 0.0
        self._target_callback_group = ReentrantCallbackGroup()
        self._controller = RealController(self)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._gripper_pub = self.create_publisher(
            JointTrajectory,
            str(self.get_parameter("gripper_topic").value),
            10,
        )

        self._cube = CubeSpawner(
            self,
            size=float(self.get_parameter("cube_size").value),
            name=str(self.get_parameter("cube_name").value),
        )
        self._spawn_cube()

        if bool(self.get_parameter("execute_grasp").value):
            self.create_subscription(
                PoseStamped,
                str(self.get_parameter("target_pose_topic").value),
                self._target_cb,
                10,
                callback_group=self._target_callback_group,
            )

    def _spawn_cube(self) -> None:
        """生成相机能看到、后续也能被 attach 的同一个方块。"""
        x = float(self.get_parameter("cube_x").value)
        y = float(self.get_parameter("cube_y").value)
        z = float(self.get_parameter("cube_z").value)
        if self._cube.spawn(x, y, z):
            # DetachableJoint Reset 会自动吸附一次，生成后立刻释放，保持方块在桌面上。
            self._cube.detach()
            self.get_logger().info(f"vision cube ready at world=({x:.3f}, {y:.3f}, {z:.3f})")

    def _target_cb(self, msg: PoseStamped) -> None:
        """收到目标位姿后触发一次仿真抓取。"""
        if self._is_running:
            return
        if self._has_executed and bool(self.get_parameter("execute_once").value):
            return
        if time.monotonic() < self._next_attempt_time:
            return

        target = self._target_in_base(msg)
        if target is None or not self._target_is_safe(target):
            return

        self._is_running = True
        try:
            success = self._execute_pick(target)
            self._has_executed = success
            if success and bool(self.get_parameter("execute_once").value):
                # 抓取流程已经完整结束，继续让 executor 空转没有意义。
                # 这里主动退出，避免 MoveIt action client 在流程结束后继续
                # 挂在 wait set 中触发 rclpy 状态错误。
                self.get_logger().info("sim grasp sequence finished")
                self._should_exit = True
            elif not success:
                self._next_attempt_time = (
                    time.monotonic() + float(self.get_parameter("retry_cooldown").value)
                )
        finally:
            self._is_running = False

    @property
    def should_exit(self) -> bool:
        """execute_once 成功后通知 main 停止 spin。"""
        return self._should_exit

    def _target_in_base(self, msg: PoseStamped) -> PoseStamped | None:
        """把相机坐标系目标转换到 base_link。"""
        base_frame = str(self.get_parameter("base_frame").value)
        try:
            transform = self._tf_buffer.lookup_transform(
                base_frame,
                msg.header.frame_id,
                Time(),
                timeout=RclpyDuration(seconds=0.5),
            )
        except TransformException as exc:
            self.get_logger().warn(f"target TF unavailable: {exc}")
            return None
        return tf2_geometry_msgs.do_transform_pose_stamped(msg, transform)

    def _target_is_safe(self, target: PoseStamped) -> bool:
        """过滤明显错误的视觉高度。"""
        z = float(target.pose.position.z)
        min_z = float(self.get_parameter("min_target_z").value)
        max_z = float(self.get_parameter("max_target_z").value)
        if min_z <= z <= max_z:
            return True
        self.get_logger().warn(f"target z={z:.3f} outside [{min_z:.3f}, {max_z:.3f}]")
        return False

    def _execute_pick(self, target: PoseStamped) -> bool:
        """仿真抓取和放置。

        放置点不单独做参数：沿用抓取点 x 和 z，仅把 y 取反，表示桌面上
        关于机械臂 x 轴的对角位置。这样 HSV/OBB/SAM 的视觉输出只负责
        找到物体，放置策略固定在执行层，后面调试也更少改参数。
        """
        raw_x = float(target.pose.position.x)
        raw_y = float(target.pose.position.y)
        z = float(target.pose.position.z)
        x_comp = float(self.get_parameter("target_x_compensation").value)
        y_comp = float(self.get_parameter("target_y_compensation").value)
        x = raw_x + x_comp
        y = raw_y + y_comp
        place_y = -y
        pre_height = float(self.get_parameter("pre_grasp_height").value)
        pick_z_offset = float(self.get_parameter("pick_z_offset").value)
        pick_z_compensation = float(self.get_parameter("pick_z_compensation").value)
        lift_height = float(self.get_parameter("lift_height").value)
        pick_z = z + pick_z_offset - pick_z_compensation

        self.get_logger().info(
            f"sim pick target raw=({raw_x:.3f}, {raw_y:.3f}, {z:.3f}), "
            f"comp=({x_comp:.3f}, {y_comp:.3f}, -{pick_z_compensation:.3f}), "
            f"base=({x:.3f}, {y:.3f}, {z:.3f}); "
            f"above_z={z + pre_height:.3f}, pick_z={pick_z:.3f}, "
            f"pick_z_offset={pick_z_offset:.3f}, z_comp={pick_z_compensation:.3f}, "
            f"lift_z={z + lift_height:.3f}; place_y={place_y:.3f}"
        )

        self._cube.detach()
        if not self._open_gripper():
            return False
        if self._move_top_down(x, y, z + pre_height, "pick_above") is None:
            return False
        pick_at = self._move_top_down(x, y, pick_z, "pick_at")
        if pick_at is None:
            return False
        self._log_tcp_pose_compare("pick_at", pick_at)
        if not self._close_gripper():
            return False
        self._cube.attach()
        if self._move_top_down(x, y, z + lift_height, "lift") is None:
            return False
        if self._move_top_down(x, place_y, z + pre_height, "place_above") is None:
            return False
        if self._move_top_down(x, place_y, pick_z, "place_at") is None:
            return False

        # 先释放 Gazebo detachable joint，再张开夹爪，避免方块继续跟着 gripper_link 走。
        self._cube.detach()
        if not self._open_gripper():
            return False
        if self._move_top_down(x, place_y, z + pre_height, "place_lift") is None:
            return False
        if not self._controller.home():
            return False
        return self._close_gripper(position=0.0)

    def _move_top_down(self, x: float, y: float, z: float, label: str) -> Pose | None:
        """选择 joint6 最连续的竖直抓取姿态并执行。

        方块抓取只要求 TCP 朝下，绕竖直轴的 yaw 对夹取没有本质影响。
        如果固定 yaw，放置点从 +y 镜像到 -y 时，IK 可能选择 joint6 从
        -2.6 跳到 +2.6 的合法解，视觉上就是腕部大幅旋转。这里尝试几个
        等效 yaw，只执行 joint6 目标最接近当前角度的那一组。
        """
        current = self._controller.get_current_joints()
        best_pose: Pose | None = None
        best_joints: list[float] | None = None
        best_yaw = 0.0
        best_joint6_delta = math.inf
        best_total_delta = math.inf

        for yaw in TOP_DOWN_YAW_CANDIDATES:
            pose = self._top_down_pose(x, y, z, yaw)
            joints = self._controller.solve_ik(pose, seed=current)
            if joints is None:
                continue

            # joint6 是有上下限的真实关节，不能用 ±2π 等效差值判断；
            # 这里直接比较将要发送给控制器的目标角度差，才能发现大幅反转。
            joint6_delta = abs(float(joints[5]) - float(current[5]))
            total_delta = sum(abs(float(a) - float(b)) for a, b in zip(joints, current))
            if (joint6_delta, total_delta) < (best_joint6_delta, best_total_delta):
                best_pose = pose
                best_joints = joints
                best_yaw = yaw
                best_joint6_delta = joint6_delta
                best_total_delta = total_delta

        if best_pose is None or best_joints is None:
            self.get_logger().error(f"{label}: no IK solution for top-down yaw candidates")
            return None

        if best_joint6_delta > MAX_JOINT6_DELTA:
            self.get_logger().error(
                f"{label}: joint6 delta={best_joint6_delta:.3f} rad too large, "
                f"reject to avoid wrist spin"
            )
            return None

        self.get_logger().info(
            f"{label}: selected yaw={best_yaw:.3f}, "
            f"joint6_delta={best_joint6_delta:.3f}"
        )
        if not self._controller.moveit_to_joints(best_joints, label):
            return None
        return best_pose

    def _log_tcp_pose_compare(self, label: str, target: Pose) -> None:
        """打印目标 TCP 位姿和实际 TF 位姿，定位夹取高度/偏移误差。

        MoveIt 执行成功只说明控制器接受并完成轨迹；真正夹取时还要确认
        `gripper_tcp` 在 TF 中的位置是否和我们计算的 `pick_at` 一致。
        这里在闭爪前打印一次，方便对比视觉计算位姿、URDF TCP 偏置和
        Gazebo 执行后的实际末端位置。
        """
        base_frame = str(self.get_parameter("base_frame").value)
        try:
            transform = self._tf_buffer.lookup_transform(
                base_frame,
                "gripper_tcp",
                Time(),
                timeout=RclpyDuration(seconds=0.5),
            )
        except TransformException as exc:
            self.get_logger().warn(f"{label} TCP TF unavailable: {exc}")
            return

        actual = transform.transform
        dx = actual.translation.x - target.position.x
        dy = actual.translation.y - target.position.y
        dz = actual.translation.z - target.position.z
        self.get_logger().info(
            f"{label} target tcp pos=({target.position.x:.4f}, "
            f"{target.position.y:.4f}, {target.position.z:.4f}), "
            f"quat=({target.orientation.x:.4f}, {target.orientation.y:.4f}, "
            f"{target.orientation.z:.4f}, {target.orientation.w:.4f})"
        )
        self.get_logger().info(
            f"{label} actual tcp pos=({actual.translation.x:.4f}, "
            f"{actual.translation.y:.4f}, {actual.translation.z:.4f}), "
            f"quat=({actual.rotation.x:.4f}, {actual.rotation.y:.4f}, "
            f"{actual.rotation.z:.4f}, {actual.rotation.w:.4f}), "
            f"delta=({dx:.4f}, {dy:.4f}, {dz:.4f})"
        )

    @staticmethod
    def _top_down_pose(x: float, y: float, z: float, yaw: float = 0.0) -> Pose:
        """桌面垂直抓取姿态：沿用项目里 Y 轴约 90 度的 TCP 朝下定义。"""
        return make_pose_from_rpy(x, y, z, (0.0, 1.5708, float(yaw)))

    def _open_gripper(self) -> bool:
        """仿真夹爪张开到开爪关节位姿。"""
        position = float(self.get_parameter("gripper_open").value)
        return self._move_gripper(position, "open")

    def _close_gripper(self, position: float | None = None) -> bool:
        """仿真夹爪闭合。

        抓取时 position=None，按方块半宽闭合，避免手指穿过物体；
        回 home 后 position=0.0，表示真正收拢夹爪。
        """
        if position is None:
            position = float(self.get_parameter("gripper_close").value)
        if position < 0.0:
            size = float(self.get_parameter("cube_size").value)
            margin = float(self.get_parameter("gripper_margin").value)
            position = size / 2.0 + margin
        return self._move_gripper(position, "close")

    def _move_gripper(self, position: float, label: str) -> bool:
        """发送夹爪关节目标，并保留短暂等待让 Gazebo 控制器完成插值。"""
        position = max(0.0, min(float(position), GRIPPER_MAX_OPEN))
        self.get_logger().info(f"sim gripper {label} -> {position:.4f}")
        self._gripper_pub.publish(gripper_trajectory(position))
        time.sleep(float(self.get_parameter("gripper_settle_time").value))
        return True


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CameraGraspSim()
    if not bool(node.get_parameter("execute_grasp").value):
        node.destroy_node()
        rclpy.shutdown()
        return

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.should_exit:
            executor.spin_once(timeout_sec=0.1)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
