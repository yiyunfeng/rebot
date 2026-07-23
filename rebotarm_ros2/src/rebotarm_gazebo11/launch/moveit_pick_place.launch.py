"""
MoveIt 版 Pick & Place 启动文件。

用户只需指定正方体和放置位置的 XY 坐标。
Z 坐标、夹爪开合值、悬停高度等全部在节点内根据 cube_size 自动算。

用法：
    ros2 launch rebotarm_gazebo11 moveit_pick_place.launch.py
    ros2 launch rebotarm_gazebo11 moveit_pick_place.launch.py mode:=hardware cube_x:=0.3
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _arg(name, default, desc):
    return DeclareLaunchArgument(name, default_value=default, description=desc)


def _task_params():
    """精简到只传用户真正关心的参数。"""
    return {
        "cube_x":    LaunchConfiguration("cube_x"),
        "cube_y":    LaunchConfiguration("cube_y"),
        "place_x":   LaunchConfiguration("place_x"),
        "place_y":   LaunchConfiguration("place_y"),
        "cube_size":            LaunchConfiguration("cube_size"),
        "velocity_scaling":     LaunchConfiguration("velocity_scaling"),
        "acceleration_scaling": LaunchConfiguration("acceleration_scaling"),
    }


def generate_launch_description():
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("rebotarm_gazebo11"), "launch", "gazebo.launch.py"])
        ),
        launch_arguments={
            "use_rviz": LaunchConfiguration("use_rviz"),
        }.items(),
    )

    params = _task_params()
    task = TimerAction(
        period=14.0,
        actions=[Node(
            package="rebotarm_gazebo11", executable="moveit_pick_place",
            name="moveit_pick_place", output="screen",
            parameters=[{**params, "use_sim_time": True}],
        )],
    )

    return LaunchDescription([
        _arg("use_rviz",  "false",     "是否启动 RViz"),
        _arg("cube_x",    "0.35",      "正方体 X (m)"),
        _arg("cube_y",    "0.15",      "正方体 Y (m)"),
        _arg("place_x",   "0.35",      "放置位置 X (m)"),
        _arg("place_y",   "-0.10",     "放置位置 Y (m)"),
        _arg("cube_size",          "0.06", "正方体边长 (m)"),
        _arg("velocity_scaling",    "0.5",  "最大速度比例"),
        _arg("acceleration_scaling", "0.5",  "最大加速度比例"),
        gazebo, task,
    ])
