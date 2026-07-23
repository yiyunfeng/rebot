"""
简单夹取放置 — Gazebo Classic 11 + MoveIt IK。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _arg(name: str, default: str, desc: str) -> DeclareLaunchArgument:
    return DeclareLaunchArgument(name, default_value=default, description=desc)


def generate_launch_description() -> LaunchDescription:
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("rebotarm_gazebo11"), "launch", "gazebo.launch.py"])
        ),
        launch_arguments={},
    )

    task = TimerAction(
        period=18.0,
        actions=[Node(
            package="rebotarm_gazebo11",
            executable="simple_pick_place",
            name="simple_pick_place",
            output="screen",
            parameters=[{
                "cube_x": LaunchConfiguration("cube_x"),
                "cube_y": LaunchConfiguration("cube_y"),
                "place_x": LaunchConfiguration("place_x"),
                "place_y": LaunchConfiguration("place_y"),
                "cube_size": LaunchConfiguration("cube_size"),
                "use_sim_time": True,
            }],
        )],
    )

    return LaunchDescription([
        _arg("cube_x", "0.35", "拾取 X (m)"),
        _arg("cube_y", "0.15", "拾取 Y (m)"),
        _arg("place_x", "0.40", "放置 X (m)"),
        _arg("place_y", "-0.10", "放置 Y (m)"),
        _arg("cube_size", "0.06", "边长 (m)"),
        gazebo, task,
    ])
