"""
reBotArm Gazebo Classic 11 仿真启动文件 — 纯仿真模式。

启动 Gazebo Classic + MoveIt + RViz，仅用于仿真，不含硬件相关组件。

用法：
    ros2 launch rebotarm_gazebo11 rebotarm.launch.py
    ros2 launch rebotarm_gazebo11 rebotarm.launch.py use_rviz:=false use_gazebo_gui:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="true", description="Start RViz"),
        DeclareLaunchArgument("use_gazebo_gui", default_value="true", description="Start Gazebo Classic client"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare("rebotarm_gazebo11"), "launch", "gazebo.launch.py"])
            ),
            launch_arguments={
                "use_rviz": LaunchConfiguration("use_rviz"),
                "use_gazebo_gui": LaunchConfiguration("use_gazebo_gui"),
            }.items(),
        ),
    ])
