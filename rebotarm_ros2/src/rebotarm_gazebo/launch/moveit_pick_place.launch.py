"""
MoveIt 版 Pick & Place 启动文件。

用户只需指定正方体和放置位置的坐标，坐标系统一为 base_link。
Z 坐标、夹爪开合值、悬停高度等全部在节点内根据 cube_size 自动算。

用法：
    # 仿真
    ros2 launch rebotarm_gazebo moveit_pick_place.launch.py mode:=sim \
      cube_x:=0.30 cube_y:=0.15 cube_z:=0.025 \
      place_x:=0.15 place_y:=-0.15 place_z:=0.025 \
      cube_size:=0.06 pre_height:=0.12 pick_height:=0.03 use_rviz:=true

    # 真机
    ros2 launch rebotarm_gazebo moveit_pick_place.launch.py mode:=hardware \
      cube_x:=0.30 cube_y:=0.15 cube_z:=0.025 \
      place_x:=0.15 place_y:=-0.15 place_z:=0.025 \
      cube_size:=0.06 pre_height:=0.12 pick_height:=0.03 use_rviz:=true
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


def _arg(name, default, desc):
    return DeclareLaunchArgument(name, default_value=default, description=desc)


def _task_params():
    """精简到只传用户真正关心的参数。"""
    return {
        "mode":      LaunchConfiguration("mode"),
        "namespace": LaunchConfiguration("namespace"),
        "cube_x":    LaunchConfiguration("cube_x"),
        "cube_y":    LaunchConfiguration("cube_y"),
        "cube_z":    LaunchConfiguration("cube_z"),
        "place_x":   LaunchConfiguration("place_x"),
        "place_y":   LaunchConfiguration("place_y"),
        "place_z":   LaunchConfiguration("place_z"),
        "cube_size":           LaunchConfiguration("cube_size"),
        "pre_height":          LaunchConfiguration("pre_height"),
        "pick_height":         LaunchConfiguration("pick_height"),
        "gripper_open":        LaunchConfiguration("gripper_open"),
        "gripper_close":       LaunchConfiguration("gripper_close"),
        "max_gripper_width":    LaunchConfiguration("max_gripper_width"),
        "hardware_open_gripper_position": LaunchConfiguration("hardware_open_gripper_position"),
        "hardware_closed_gripper_position": LaunchConfiguration("hardware_closed_gripper_position"),
        "gripper_max_effort":   LaunchConfiguration("gripper_max_effort"),
        "velocity_scaling":    LaunchConfiguration("velocity_scaling"),
        "acceleration_scaling": LaunchConfiguration("acceleration_scaling"),
        "constrain_joint6":     LaunchConfiguration("constrain_joint6"),
        "joint6_goal_tolerance": LaunchConfiguration("joint6_goal_tolerance"),
    }


def generate_launch_description():
    mode = LaunchConfiguration("mode")
    is_sim = PythonExpression(["'", mode, "' == 'sim'"])

    sim_env = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("rebotarm_gazebo"), "launch", "gazebo.launch.py"])
        ),
        condition=IfCondition(is_sim),
        launch_arguments={
            "start_moveit": "true",
            "use_rviz": LaunchConfiguration("use_rviz"),
            "use_gripper_gui": "false",
        }.items(),
    )

    hw_env = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("rebotarm_bringup"), "launch", "driver.launch.py"])
        ),
        condition=UnlessCondition(is_sim),
        launch_arguments={
            "arm_namespace": LaunchConfiguration("namespace"),
        }.items(),
    )

    hw_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("rebotarm_gazebo"), "launch", "hardware.launch.py"])
        ),
        condition=UnlessCondition(is_sim),
        launch_arguments={
            "use_rviz": LaunchConfiguration("use_rviz"),
            "arm_namespace": LaunchConfiguration("namespace"),
        }.items(),
    )

    params = _task_params()

    sim_task = TimerAction(
        period=14.0,
        actions=[Node(
            package="rebotarm_gazebo", executable="moveit_pick_place",
            name="moveit_pick_place", output="screen",
            parameters=[{**params, "use_sim_time": True}],
        )],
        condition=IfCondition(is_sim),
    )

    hw_task = TimerAction(
        period=5.0,
        actions=[Node(
            package="rebotarm_gazebo", executable="moveit_pick_place",
            name="moveit_pick_place", output="screen",
            parameters=[{**params, "use_sim_time": False}],
        )],
        condition=UnlessCondition(is_sim),
    )

    return LaunchDescription([
        _arg("mode",      "sim",       "sim 或 hardware"),
        _arg("namespace", "rebotarm",  "真机命名空间"),
        _arg("use_rviz",  "false",     "是否启动 RViz"),
        _arg("cube_x",    "0.30",      "正方体 X (m)，base_link 坐标系"),
        _arg("cube_y",    "0.15",      "正方体 Y (m)，base_link 坐标系"),
        _arg("cube_z",    "0.025",     "正方体中心 Z (m)，base_link 坐标系"),
        _arg("place_x",   "0.40",      "放置位置 X (m)，base_link 坐标系"),
        _arg("place_y",   "-0.10",     "放置位置 Y (m)，base_link 坐标系"),
        _arg("place_z",   "0.025",     "放置位置中心 Z (m)，base_link 坐标系"),
        _arg("cube_size",           "0.06", "正方体边长 (m)"),
        _arg("pre_height",          "0.10", "抓取/放置前的上方悬停高度 (m)"),
        _arg("pick_height",         "0.00", "TCP 到方块中心上方的夹取高度 (m)"),
        _arg("gripper_open",        "0.06", "夹爪打开位置 (m)"),
        _arg("gripper_close",       "-1.0", "夹爪夹取位置 (m)，-1 表示按 cube_size 自动计算"),
        _arg("max_gripper_width",   "0.09", "夹爪最大总开口宽度 (m)，用于真机比例换算"),
        _arg("hardware_open_gripper_position", "-5.0", "DM 真机夹爪全开电机位置"),
        _arg("hardware_closed_gripper_position", "0.0", "真机夹爪闭合电机位置"),
        _arg("gripper_max_effort",  "10.0", "真机夹爪最大力"),
        _arg("velocity_scaling",    "0.5",  "最大速度比例 (0.1-1.0)"),
        _arg("acceleration_scaling", "0.5",  "最大加速度比例 (0.1-1.0)"),
        _arg("constrain_joint6",     "false", "规划末端位姿时约束 joint6，可能导致规划失败，默认关闭"),
        _arg("joint6_goal_tolerance", "0.6", "joint6 目标约束容差 (rad)"),

        sim_env, hw_env, hw_moveit, sim_task, hw_task,
    ])
