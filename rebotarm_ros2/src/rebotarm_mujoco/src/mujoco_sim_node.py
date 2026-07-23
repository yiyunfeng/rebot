"""MuJoCo 仿真节点。

功能：
    1. 加载 reBot Arm B601 DM 的 MJCF 模型；
    2. 发布 /joint_states，供 robot_state_publisher 和 MoveIt 使用；
    3. 提供两个 FollowJointTrajectory action server：
       - /rebotarm_controller/follow_joint_trajectory
       - /gripper_controller/follow_joint_trajectory
    4. 可选启动 MuJoCo viewer，直接查看同一个仿真实例的运动。

设计取舍：
    MuJoCo 只负责物理仿真和摩擦接触，MoveIt 仍负责规划。
    夹取时依赖 finger_pad 与物体之间的摩擦和接触力，不做吸附。
"""

from __future__ import annotations

import os
import threading
import time
from typing import Iterable

from ament_index_python.packages import get_package_share_directory
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState


class MujocoSimNode(Node):
    """把 MuJoCo 关节控制包装成 MoveIt 能调用的轨迹 action。"""

    _MUJOCO_JOINT_ALIAS = {
        "gripper_joint1": "left_finger",
        "gripper_joint2": "right_finger",
    }
    _MUJOCO_ACTUATOR_ALIAS = {
        "gripper_joint1": "gripper",
        "gripper_joint2": "gripper",
    }
    _GRIPPER_ROS_MAX = 0.0715
    _GRIPPER_MUJOCO_MIN = 0.001
    _GRIPPER_MUJOCO_MAX = 0.05

    def __init__(self) -> None:
        super().__init__("mujoco_sim_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("rebotarm_mesh_dir", "")
        self.declare_parameter("orbbec_mesh_dir", "")
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("step_hz", 500.0)
        self.declare_parameter("control_rate", 100.0)
        self.declare_parameter("trajectory_time_scale", 0.45)
        self.declare_parameter("goal_tolerance", 0.003)
        self.declare_parameter("settle_timeout", 1.5)
        self.declare_parameter("use_viewer", True)
        self.declare_parameter("arm_action_name", "/rebotarm_controller/follow_joint_trajectory")
        self.declare_parameter("gripper_action_name", "/gripper_controller/follow_joint_trajectory")
        self.declare_parameter("arm_joint_names", ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])
        self.declare_parameter("gripper_joint_names", ["gripper_joint1", "gripper_joint2"])
        self.declare_parameter("home_positions", [0.0, -0.05, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not model_path:
            model_path = self._default_model_path()

        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                "未安装 MuJoCo Python 包。请在当前运行 ros2 的 Python 环境执行："
                "python3 -m pip install --user 'mujoco==3.2.7'"
            ) from exc

        self._mujoco = mujoco
        self._model = self._load_model(model_path)
        self._data = mujoco.MjData(self._model)
        self._lock = threading.RLock()
        self._callback_group = ReentrantCallbackGroup()
        self._viewer = None

        self._arm_joint_names = list(self.get_parameter("arm_joint_names").value)
        self._gripper_joint_names = list(self.get_parameter("gripper_joint_names").value)
        self._joint_names = self._arm_joint_names + self._gripper_joint_names
        self._qpos_addr = self._build_qpos_map(self._joint_names)
        self._actuator_id = self._build_actuator_map(self._joint_names)

        home_positions = list(self.get_parameter("home_positions").value)
        home_positions = self._targets_for_mujoco(self._joint_names, home_positions)
        self._set_joint_targets(self._joint_names, home_positions)
        self._write_qpos(self._joint_names, home_positions)
        mujoco.mj_forward(self._model, self._data)
        self._start_viewer_if_needed()

        self._joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        publish_rate = float(self.get_parameter("publish_rate").value)
        self._step_dt = 1.0 / float(self.get_parameter("step_hz").value)
        self._control_dt = 1.0 / float(self.get_parameter("control_rate").value)
        self._trajectory_time_scale = float(self.get_parameter("trajectory_time_scale").value)
        self._goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self._settle_timeout = float(self.get_parameter("settle_timeout").value)
        self._is_executing_trajectory = False
        self._timer = self.create_timer(
            1.0 / publish_rate,
            self._on_timer,
            callback_group=self._callback_group,
        )

        self._arm_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.get_parameter("arm_action_name").value,
            execute_callback=self._execute_trajectory,
            callback_group=self._callback_group,
        )
        self._gripper_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.get_parameter("gripper_action_name").value,
            execute_callback=self._execute_trajectory,
            callback_group=self._callback_group,
        )
        self.get_logger().info(f"MuJoCo model loaded: {model_path}")

    def _default_model_path(self) -> str:
        pkg_share = get_package_share_directory("rebotarm_mujoco")
        return os.path.join(pkg_share, "models", "rebotarm_dm.xml")

    def _load_model(self, model_path: str):
        """加载 MJCF，并把 ROS package mesh 目录展开成绝对路径。

        MuJoCo 不认识 ROS 的 package:// 路径；而 install 目录中的 MJCF 也不能
        依赖源码相对路径。因此模型 XML 中使用 ${REBOTARM_MESH_DIR} 和
        ${ORBBEC_MESH_DIR} 占位符，在这里按 launch 传入的 package share 路径替换。
        """
        with open(model_path, "r", encoding="utf-8") as file:
            xml = file.read()

        needs_rebotarm_mesh = "${REBOTARM_MESH_DIR}" in xml
        needs_orbbec_mesh = "${ORBBEC_MESH_DIR}" in xml
        if not needs_rebotarm_mesh and not needs_orbbec_mesh:
            return self._mujoco.MjModel.from_xml_path(model_path)

        rebotarm_mesh_dir = self.get_parameter("rebotarm_mesh_dir").get_parameter_value().string_value
        orbbec_mesh_dir = self.get_parameter("orbbec_mesh_dir").get_parameter_value().string_value
        if needs_rebotarm_mesh and not os.path.isdir(rebotarm_mesh_dir):
            raise RuntimeError(f"rebotarm mesh 目录不存在: {rebotarm_mesh_dir}")
        if needs_orbbec_mesh and not os.path.isdir(orbbec_mesh_dir):
            raise RuntimeError(f"orbbec mesh 目录不存在: {orbbec_mesh_dir}")

        xml = xml.replace("${REBOTARM_MESH_DIR}", rebotarm_mesh_dir)
        xml = xml.replace("${ORBBEC_MESH_DIR}", orbbec_mesh_dir)
        return self._mujoco.MjModel.from_xml_string(xml)

    def _start_viewer_if_needed(self) -> None:
        """按配置启动 MuJoCo GUI。

        viewer 使用当前节点里的 model/data，不会创建第二套仿真。
        MoveIt 发来的轨迹改变 data 后，viewer.sync() 会显示同一份状态。
        """
        if not bool(self.get_parameter("use_viewer").value):
            self.get_logger().info("MuJoCo viewer disabled by use_viewer=false")
            return

        try:
            import mujoco.viewer
        except ImportError as exc:
            raise RuntimeError(
                "已安装 mujoco 但无法导入 mujoco.viewer。请检查 MuJoCo Python 包是否完整。"
            ) from exc

        try:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        except Exception as exc:
            raise RuntimeError(
                "MuJoCo viewer 启动失败。请确认当前终端有图形界面权限，"
                "或在 config/mujoco_params.yaml 中设置 use_viewer: false。"
            ) from exc
        self.get_logger().info("MuJoCo viewer started")

    def _build_qpos_map(self, joint_names: Iterable[str]) -> dict[str, int]:
        """记录每个 ROS 关节名对应 MuJoCo qpos 下标。"""
        qpos_addr = {}
        for name in joint_names:
            mujoco_name = self._MUJOCO_JOINT_ALIAS.get(name, name)
            joint_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_JOINT, mujoco_name)
            if joint_id < 0:
                raise RuntimeError(f"MJCF 中找不到 joint: {name} -> {mujoco_name}")
            qpos_addr[name] = int(self._model.jnt_qposadr[joint_id])
        return qpos_addr

    def _build_actuator_map(self, joint_names: Iterable[str]) -> dict[str, int]:
        """记录每个关节对应的 position actuator；缺失时直接写 qpos。"""
        actuator_id = {}
        for name in joint_names:
            mujoco_name = self._MUJOCO_ACTUATOR_ALIAS.get(name, name)
            act_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco_name)
            if act_id >= 0:
                actuator_id[name] = int(act_id)
            else:
                self.get_logger().warn(f"MJCF 中找不到 actuator: {name} -> {mujoco_name}，该关节将直接写 qpos")
        return actuator_id

    def _on_timer(self) -> None:
        """定时推进仿真并发布关节状态。"""
        with self._lock:
            # 轨迹 action 执行时由 _move_segment 独占推进仿真。
            # 如果定时器同时 mj_step，同一段轨迹会被两套节拍交替推进，表现为抖动。
            if not self._is_executing_trajectory:
                self._mujoco.mj_step(self._model, self._data)
            self._publish_joint_states()
            self._sync_viewer()

    def _sync_viewer(self) -> None:
        """同步 GUI 画面。

        MuJoCo viewer 是被动显示器，仿真仍由本节点 step；
        这里仅把最新 data 同步到窗口。
        """
        if self._viewer is None:
            return
        if self._viewer.is_running():
            self._viewer.sync()
        else:
            self.get_logger().warn("MuJoCo viewer closed by user")
            self._viewer = None

    def _publish_joint_states(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._joint_names
        msg.position = [
            self._position_for_ros(name, float(self._data.qpos[self._qpos_addr[name]]))
            for name in self._joint_names
        ]
        msg.velocity = [0.0 for _ in self._joint_names]
        self._joint_pub.publish(msg)

    def _write_qpos(self, joint_names: Iterable[str], positions: Iterable[float]) -> None:
        for name, value in zip(joint_names, positions):
            self._data.qpos[self._qpos_addr[name]] = float(value)

    def _set_joint_targets(self, joint_names: Iterable[str], positions: Iterable[float]) -> None:
        """设置 MuJoCo position actuator 的目标值。

        对有 actuator 的关节写 ctrl，让 MuJoCo 通过力学积分到目标；
        没 actuator 的关节才直接写 qpos，避免模型配置遗漏时整条链不可控。
        """
        for name, value in zip(joint_names, positions):
            value = float(value)
            if name in self._actuator_id:
                self._data.ctrl[self._actuator_id[name]] = value
            else:
                self._data.qpos[self._qpos_addr[name]] = value

    def _target_for_mujoco(self, name: str, value: float) -> float:
        if name not in self._MUJOCO_JOINT_ALIAS:
            return float(value)
        value = max(0.0, min(self._GRIPPER_ROS_MAX, float(value)))
        ratio = value / self._GRIPPER_ROS_MAX
        return self._GRIPPER_MUJOCO_MIN + ratio * (self._GRIPPER_MUJOCO_MAX - self._GRIPPER_MUJOCO_MIN)

    def _targets_for_mujoco(self, joint_names: Iterable[str], positions: Iterable[float]) -> list[float]:
        return [self._target_for_mujoco(name, value) for name, value in zip(joint_names, positions)]

    def _position_for_ros(self, name: str, value: float) -> float:
        if name not in self._MUJOCO_JOINT_ALIAS:
            return float(value)
        ratio = (float(value) - self._GRIPPER_MUJOCO_MIN) / (self._GRIPPER_MUJOCO_MAX - self._GRIPPER_MUJOCO_MIN)
        ratio = max(0.0, min(1.0, ratio))
        return ratio * self._GRIPPER_ROS_MAX

    def _current_positions(self, joint_names: Iterable[str]) -> list[float]:
        return [float(self._data.qpos[self._qpos_addr[name]]) for name in joint_names]

    def _execute_trajectory(self, goal_handle):
        """执行 MoveIt 发来的 JointTrajectory。

        轨迹点使用线性插值下发给 MuJoCo position actuator。
        这里不额外做 IK 或路径规划，避免和 MoveIt 的职责重复。
        """
        trajectory = goal_handle.request.trajectory
        result = FollowJointTrajectory.Result()
        if not trajectory.points:
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "trajectory has no points"
            goal_handle.abort()
            return result

        joint_names = list(trajectory.joint_names)
        unknown = [name for name in joint_names if name not in self._qpos_addr]
        if unknown:
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            result.error_string = f"unknown joints: {unknown}"
            goal_handle.abort()
            return result

        last_time = 0.0
        final_target = None
        with self._lock:
            self._is_executing_trajectory = True
        try:
            for point in trajectory.points:
                target = self._targets_for_mujoco(joint_names, point.positions)
                final_target = target
                point_time = float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9
                duration = max((point_time - last_time) * self._trajectory_time_scale, self._control_dt)
                last_time = point_time
                if not self._move_segment(joint_names, target, duration):
                    result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    result.error_string = "goal canceled"
                    goal_handle.canceled()
                    return result
            if final_target is not None and not self._settle_to_target(joint_names, final_target):
                result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                result.error_string = "goal tolerance violated"
                goal_handle.abort()
                return result
        finally:
            with self._lock:
                self._is_executing_trajectory = False

        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = "ok"
        goal_handle.succeed()
        return result

    def _move_segment(self, joint_names: list[str], target: list[float], duration: float) -> bool:
        """按控制周期插值到下一个轨迹点。

        MuJoCo 物理步长保持 500 Hz，用于稳定接触和摩擦；
        控制命令不需要每个物理步都发布一次，否则 Python 循环和 GUI sync 会拖慢动作。
        因此这里按 control_rate 更新目标，中间一次推进多个 mj_step。
        """
        steps = max(1, int(duration / self._control_dt))
        physics_substeps = max(1, int(round(self._control_dt / self._step_dt)))
        with self._lock:
            start = self._current_positions(joint_names)

        for index in range(1, steps + 1):
            ratio = index / steps
            positions = [s + (t - s) * ratio for s, t in zip(start, target)]
            with self._lock:
                self._set_joint_targets(joint_names, positions)
                for _ in range(physics_substeps):
                    self._mujoco.mj_step(self._model, self._data)
                self._publish_joint_states()
                self._sync_viewer()
            time.sleep(self._control_dt)
        return True

    def _settle_to_target(self, joint_names: list[str], target: list[float]) -> bool:
        """保持最终控制目标，等 MuJoCo 真实关节状态进入容差后再向 MoveIt 回成功。

        position actuator 有物理惯性，写入 ctrl 后 qpos 不会瞬间等于目标。
        如果 action 过早返回 SUCCEEDED，MoveIt/RViz 会认为已经到位，但
        /joint_states 仍在途中，后续规划的起点和仿真真实状态就会不同步。
        """
        deadline = time.monotonic() + self._settle_timeout
        physics_substeps = max(1, int(round(self._control_dt / self._step_dt)))
        while time.monotonic() < deadline:
            with self._lock:
                self._set_joint_targets(joint_names, target)
                for _ in range(physics_substeps):
                    self._mujoco.mj_step(self._model, self._data)
                current = self._current_positions(joint_names)
                self._publish_joint_states()
                self._sync_viewer()
            max_error = max(abs(value - goal) for value, goal in zip(current, target))
            if max_error <= self._goal_tolerance:
                return True
            time.sleep(self._control_dt)
        self.get_logger().warn(
            f"trajectory final target not reached within {self._settle_timeout:.2f}s; "
            f"max_error={max_error:.4f}, tolerance={self._goal_tolerance:.4f}"
        )
        return False

    def destroy_node(self) -> bool:
        """关闭节点时同步关闭 MuJoCo GUI，避免窗口残留。"""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except KeyboardInterrupt:
                # Ctrl-C 关闭 launch 时 viewer 可能正在退出；这里吞掉二次中断，
                # 避免把正常停止打印成 Python 异常。
                pass
            self._viewer = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MujocoSimNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.remove_node(node)
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
