"""Hardware camera grasp pipeline.

运行命令：
    cd /home/yyf/Desktop/pythonProject/rebot/rebotarm_ros2
    source /opt/ros/humble/setup.bash
    source install/setup.bash

    # 真机：先启动硬件和 MoveIt，再启动视觉位姿桥接，最后启动本节点
    ros2 launch rebotarm_gazebo rebotarm.launch.py mode:=hardware
    ros2 run rebotarm_gazebo camera_grasp_hardware

    # 本节点启动后只缓存最新 /rebot_grasp/grasp_pose，不会自动抓取。
    # 确认 debug 图里的目标和姿态正确后，手动触发一次：
    ros2 service call /rebot_grasp/execute_grasp std_srvs/srv/Trigger "{}"

说明：
    本文件只负责真实机械臂抓取执行：
      1. 订阅 /rebot_grasp/grasp_pose；
      2. TF 转到 base_link；
      3. enable 真机；
      4. 张开夹爪，移动到物体上方，再下降；
      5. 闭合夹爪并抬升；
      6. 移动到 y 轴镜像位置放置，回 home 后闭合夹爪。

    真机没有 Gazebo attach/detach，也不生成方块。
"""

from __future__ import annotations

import time

import rclpy
import tf2_geometry_msgs
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from rebotarm_gazebo.real_controller import RealController, make_pose_from_rpy
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener


class CameraGraspHardware(Node):
    """真实机械臂相机抓取节点，参数只保留真机需要的内容。"""

    def __init__(self) -> None:
        super().__init__("camera_grasp_hardware")

        self.declare_parameter("target_pose_topic", "/rebot_grasp/grasp_pose")
        self.declare_parameter("trigger_service", "/rebot_grasp/execute_grasp")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("retry_cooldown", 2.0)
        self.declare_parameter("pre_grasp_height", 0.05)
        self.declare_parameter("lift_height", 0.05)
        self.declare_parameter("min_target_z", 0.01)
        self.declare_parameter("max_target_z", 0.45)

        self._is_running = False
        self._latest_target: PoseStamped | None = None
        self._next_attempt_time = 0.0
        self._target_callback_group = ReentrantCallbackGroup()
        self._controller = RealController(self)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("target_pose_topic").value),
            self._target_cb,
            10,
            callback_group=self._target_callback_group,
        )
        self.create_service(
            Trigger,
            str(self.get_parameter("trigger_service").value),
            self._execute_service_cb,
            callback_group=self._target_callback_group,
        )
        self.get_logger().info(
            "camera_grasp_hardware ready: waiting for /rebot_grasp/grasp_pose; "
            "trigger with: ros2 service call /rebot_grasp/execute_grasp "
            'std_srvs/srv/Trigger "{}"'
        )

    def _target_cb(self, msg: PoseStamped) -> None:
        """收到 rebot_grasp 输出的抓取位姿后缓存；按需触发执行。"""
        if self._is_running:
            return
        if time.monotonic() < self._next_attempt_time:
            return

        target = self._target_in_base(msg)
        if target is None or not self._target_is_safe(target):
            return

        self._latest_target = target

    def _execute_service_cb(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """手动触发一次抓取，使用最近一帧有效视觉位姿。"""
        if self._is_running:
            response.success = False
            response.message = "grasp is already running"
            return response
        if self._latest_target is None:
            response.success = False
            response.message = "no valid /rebot_grasp/grasp_pose received yet"
            return response

        response.success = self._run_execute(self._latest_target)
        response.message = "grasp finished" if response.success else "grasp failed"
        return response

    def _run_execute(self, target: PoseStamped) -> bool:
        """执行一次抓取，并更新状态。"""
        self._is_running = True
        try:
            success = self._execute_pick(target)
            if not success:
                self._next_attempt_time = (
                    time.monotonic() + float(self.get_parameter("retry_cooldown").value)
                )
            return success
        finally:
            self._is_running = False

    def _target_in_base(self, msg: PoseStamped) -> PoseStamped | None:
        """把 rebot_grasp 输出坐标系转换到 MoveIt 使用的 base_link。"""
        base_frame = str(self.get_parameter("base_frame").value)
        source_frame = msg.header.frame_id
        if source_frame == base_frame:
            self.get_logger().debug(
                f"received grasp pose already in {base_frame}: "
                f"({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})"
            )
            return msg

        try:
            transform = self._tf_buffer.lookup_transform(
                base_frame,
                source_frame,
                Time(),
                timeout=RclpyDuration(seconds=0.5),
            )
        except TransformException as exc:
            self.get_logger().warn(f"target TF unavailable: {exc}")
            return None
        target = tf2_geometry_msgs.do_transform_pose_stamped(msg, transform)
        self.get_logger().debug(
            f"converted grasp pose {source_frame} -> {target.header.frame_id}: "
            f"({target.pose.position.x:.3f}, {target.pose.position.y:.3f}, {target.pose.position.z:.3f})"
        )
        return target

    def _target_is_safe(self, target: PoseStamped) -> bool:
        """真机安全兜底：过滤明显异常的目标高度。"""
        z = float(target.pose.position.z)
        min_z = float(self.get_parameter("min_target_z").value)
        max_z = float(self.get_parameter("max_target_z").value)
        if min_z <= z <= max_z:
            return True
        self.get_logger().warn(f"target z={z:.3f} outside [{min_z:.3f}, {max_z:.3f}]")
        return False

    def _execute_pick(self, target: PoseStamped) -> bool:
        """真机抓取和放置。

        放置点沿用抓取点 x、高度和姿态，仅把 y 取反。这样不需要额外传
        place_x/place_y 参数，真实桌面调试时只要保证镜像位置没有障碍物。
        """
        raw_x = float(target.pose.position.x)
        raw_y = float(target.pose.position.y)
        z = float(target.pose.position.z)
        x = raw_x
        y = raw_y
        place_y = -y
        pre_height = float(self.get_parameter("pre_grasp_height").value)
        lift_height = float(self.get_parameter("lift_height").value)
        pick_z = z-0.02#加一点moveit的z补偿

        frame_id = target.header.frame_id or str(self.get_parameter("base_frame").value)

        self.get_logger().info(
            f"hardware pick target frame={frame_id}, "
            f"raw=({raw_x:.3f}, {raw_y:.3f}, {z:.3f}), "
            f"base=({x:.3f}, {y:.3f}, {z:.3f}); "
            f"above_z={z + pre_height:.3f}, pick_z={pick_z:.3f}, "
            f"lift_z={z + lift_height:.3f}; place_y={place_y:.3f}"
        )

        # HSV 只提供目标点，姿态固定为夹爪向下。
        orientation = make_pose_from_rpy(x, y, z, (0.0, 1.5708, 0.0)).orientation
        self.get_logger().info("HSV tcp rpy fixed to downward=(0.000, 1.5708, 0.000)")
        pose_plan = [
            ("pick_above", self._pose_with_orientation(x, y, z + pre_height, orientation)),
            ("pick_at", self._pose_with_orientation(x, y, pick_z, orientation)),
            ("lift", self._pose_with_orientation(x, y, z + lift_height, orientation)),
            ("place_above", self._pose_with_orientation(x, place_y, z + pre_height, orientation)),
            ("place_at", self._pose_with_orientation(x, place_y, pick_z, orientation)),
            ("place_lift", self._pose_with_orientation(x, place_y, z + lift_height, orientation)),
        ]
        joint_plan = []
        for label, pose in pose_plan:
            joints = self._solve_pose(pose, label)
            if joints is None:
                return False
            joint_plan.append((label, joints))

        self._controller.enable()
        if not self._controller.open_gripper():
            return False
        if not self._move_joints(joint_plan[0][1], "pick_above"):
            return False
        if not self._move_joints(joint_plan[1][1], "pick_at"):
            return False
        if not self._controller.close_gripper():
            return False
        if not self._move_joints(joint_plan[2][1], "lift"):
            return False
        if not self._move_joints(joint_plan[3][1], "place_above"):
            return False
        if not self._move_joints(joint_plan[4][1], "place_at"):
            return False
        if not self._controller.open_gripper():
            return False
        if not self._move_joints(joint_plan[5][1], "place_lift"):
            return False
        if not self._controller.home():
            return False
        return self._controller.close_gripper()

    def _solve_pose(self, pose: Pose, label: str) -> list[float] | None:
        """先做 IK 预检查，避免执行到一半才发现目标不可达。"""
        self.get_logger().info(
            f"{label} target pose=({pose.position.x:.3f}, "
            f"{pose.position.y:.3f}, {pose.position.z:.3f})"
        )
        joints = self._controller.solve_ik(pose)
        if joints is None:
            self.get_logger().error(f"{label}: IK failed")
        return joints

    def _move_joints(self, joints: list[float], label: str) -> bool:
        """按预检查得到的关节角执行 MoveIt 规划。"""
        return self._controller.moveit_to_joints(joints, label)

    @staticmethod
    def _pose_with_orientation(x: float, y: float, z: float, orientation: Quaternion) -> Pose:
        """使用固定 TCP 姿态，只调整执行层规划点的位置。"""
        return Pose(
            position=Point(x=float(x), y=float(y), z=float(z)),
            orientation=Quaternion(
                x=float(orientation.x),
                y=float(orientation.y),
                z=float(orientation.z),
                w=float(orientation.w),
            ),
        )

def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CameraGraspHardware()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
