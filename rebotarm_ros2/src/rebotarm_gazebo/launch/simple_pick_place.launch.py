"""
方案1 启动文件 — 简化版夹取放置（TCP 笛卡尔位姿 + IK）。

mode:=sim:
  include gazebo.launch.py 并启用 move_group（提供 /compute_ik 服务），
  任务节点通过 IK 将 TCP 位姿转为关节角后执行。

mode:=hardware:
  include rebotarm_moveit_config/hardware.launch.py，可保留 RViz；
  任务节点直接调用真机 /rebotarm/move_to_pose 等接口。

用法：
    # 仿真
    ros2 launch rebotarm_gazebo simple_pick_place.launch.py mode:=sim use_rviz:=true \
      cube_x:=0.35 cube_y:=0.15 cube_z:=-1.0 \
      place_x:=0.20 place_y:=-0.15 place_z:=-1.0 cube_size:=0.06

    # 真机
    ros2 launch rebotarm_gazebo simple_pick_place.launch.py mode:=hardware use_rviz:=true \
      cube_x:=0.35 cube_y:=0.15 cube_z:=0.29 \
      place_x:=0.20 place_y:=-0.15 place_z:=0.29 cube_size:=0.06

    ros2 launch rebotarm_gazebo simple_pick_place.launch.py mode:=hardware hw_ik_solver:=moveit

"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _arg(name: str, default: str, desc: str) -> DeclareLaunchArgument:
    """快捷声明启动参数的辅助函数。"""
    return DeclareLaunchArgument(name, default_value=default, description=desc)


def _task_params() -> dict:
    """只传递用户真正关心的顶层参数，其余用节点内部默认值。"""
    return {
        "mode":      LaunchConfiguration("mode"),
        "namespace": LaunchConfiguration("namespace"),
        "cube_x":    LaunchConfiguration("cube_x"),
        "cube_y":    LaunchConfiguration("cube_y"),
        "cube_z":    LaunchConfiguration("cube_z"),
        "place_x":   LaunchConfiguration("place_x"),
        "place_y":   LaunchConfiguration("place_y"),
        "place_z":   LaunchConfiguration("place_z"),
        "cube_size": LaunchConfiguration("cube_size"),
        "pre_height": LaunchConfiguration("pre_height"),
        "pick_tcp_offset": LaunchConfiguration("pick_tcp_offset"),
        "place_tcp_offset": LaunchConfiguration("place_tcp_offset"),
        "max_gripper_width": LaunchConfiguration("max_gripper_width"),
        "gripper_open": LaunchConfiguration("gripper_open"),
        "closed_gripper_position": LaunchConfiguration("closed_gripper_position"),
        "gripper_close": LaunchConfiguration("gripper_close"),
        "hardware_open_gripper_position": LaunchConfiguration("hardware_open_gripper_position"),
        "hardware_closed_gripper_position": LaunchConfiguration("hardware_closed_gripper_position"),
        "gripper_max_effort": LaunchConfiguration("gripper_max_effort"),
        "arm_result_timeout":   LaunchConfiguration("arm_result_timeout"),
        "move_duration":       LaunchConfiguration("move_duration"),
        "hw_ik_solver":        LaunchConfiguration("hw_ik_solver"),
    }


def generate_launch_description() -> LaunchDescription:
    """生成 launch 描述：sim 启 Gazebo，hardware 启真机 MoveIt/RViz 环境。"""
    mode = LaunchConfiguration("mode")
    is_sim = PythonExpression(["'", mode, "' == 'sim'"])

    # Gazebo 仿真环境 + move_group（提供 /compute_ik）
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("rebotarm_gazebo"), "launch", "gazebo.launch.py"])
        ),
        condition=IfCondition(is_sim),
        launch_arguments={
            "start_moveit":    "true",
            "use_rviz":        LaunchConfiguration("use_rviz"),
            "use_gripper_gui": "false",
        }.items(),
    )

    # 真机 driver — 提供 follow_joint_trajectory / move_to_pose_ik / gripper/command 等 action
    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("rebotarm_bringup"), "launch", "driver.launch.py"])
        ),
        condition=UnlessCondition(is_sim),
        launch_arguments={
            "arm_namespace": LaunchConfiguration("namespace"),
        }.items(),
    )

    hardware = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("rebotarm_gazebo"), "launch", "hardware.launch.py"])
        ),
        condition=UnlessCondition(is_sim),
        launch_arguments={
            "use_rviz": LaunchConfiguration("use_rviz"),
            "arm_namespace": LaunchConfiguration("namespace"),
            "model": "dm",
        }.items(),
    )

    # 任务节点（延迟等 Gazebo / 真机接口就绪）
    params = _task_params()
    sim_task = TimerAction(
        period=18.0,
        actions=[Node(
            package="rebotarm_gazebo",
            executable="simple_pick_place",
            name="simple_pick_place",
            output="screen",
            parameters=[{**params, "use_sim_time": True}],
        )],
        condition=IfCondition(is_sim),
    )
    hw_task = TimerAction(
        period=10.0,
        actions=[Node(
            package="rebotarm_gazebo",
            executable="simple_pick_place",
            name="simple_pick_place",
            output="screen",
            parameters=[{**params, "use_sim_time": False}],
        )],
        condition=UnlessCondition(is_sim),
    )

    return LaunchDescription([
        _arg("mode",      "sim",      "sim 或 hardware"),
        _arg("namespace", "rebotarm", "真机命名空间"),
        _arg("use_rviz",  "false",    "是否启动 RViz"),
        _arg("cube_x",    "0.35",  "正方体拾取位置 X (m)，world 坐标系"),
        _arg("cube_y",    "0.15",  "正方体拾取位置 Y (m)，world 坐标系"),
        _arg("cube_z",    "-1.0",  "正方体中心 Z (m)，-1 表示按桌面高度自动计算"),
        _arg("place_x",   "0.40",  "正方体放置位置 X (m)，world 坐标系"),
        _arg("place_y",   "-0.15", "正方体放置位置 Y (m)，world 坐标系"),
        _arg("place_z",   "-1.0",  "放置位置中心 Z (m)，-1 表示等于 cube_z"),
        _arg("cube_size", "0.06",  "正方体边长 (m)"),
        _arg("pre_height", "0.10", "抓取/放置前的上方悬停高度 (m)"),
        _arg("pick_tcp_offset", "-0.03", "夹取时 TCP 在方块中心上方的 Z 偏移 (m)"),
        _arg("place_tcp_offset", "-0.03", "放置时 TCP 在方块中心上方的 Z 偏移 (m)"),
        _arg("max_gripper_width", "0.09", "夹爪最大总开口宽度 (m)，与 pick_place.yaml 一致"),
        _arg("gripper_open", "0.06", "夹爪打开位置 (m)，全开避免接近时碰撞方块"),
        _arg("closed_gripper_position", "0.0", "夹爪完全闭合位置 (m)"),
        _arg("gripper_close", "-1.0", "夹爪夹取位置 (m)，-1 表示按 pick_place 比例自动计算"),
        _arg("hardware_open_gripper_position", "-5.0", "DM 真机夹爪全开电机位置"),
        _arg("hardware_closed_gripper_position", "0.0", "真机夹爪闭合电机位置"),
        _arg("gripper_max_effort", "10.0", "真机夹爪最大力"),
        _arg("arm_result_timeout", "30.0", "等待每段机械臂 action 执行结果的最长时间 (s)"),
        _arg("move_duration", "3.0", "每段轨迹运动的时长 (s)，越小越快"),
        _arg("hw_ik_solver", "sdk", "真机 IK: sdk→/move_to_pose_ik, moveit→/compute_ik"),
        gazebo,
        driver,
        hardware,
        sim_task,
        hw_task,
    ])
