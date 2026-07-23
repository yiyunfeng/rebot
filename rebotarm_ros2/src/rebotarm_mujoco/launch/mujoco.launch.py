"""启动 MuJoCo + MoveIt 2。

运行：
    cd /home/yyf/Desktop/pythonProject/rebot/rebotarm_ros2
    source install/setup.bash
    ros2 launch rebotarm_mujoco mujoco.launch.py

说明：
    MuJoCo 节点提供 FollowJointTrajectory action server；
    MoveIt 仍然负责规划，执行时把轨迹发给 MuJoCo。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

from rebotarm_mujoco.moveit_launch_common import moveit_parameters


def generate_launch_description():
    pkg_share = get_package_share_directory("rebotarm_mujoco")
    rebotarm_bringup_share = get_package_share_directory("rebotarm_bringup")
    orbbec_description_share = get_package_share_directory("orbbec_description")
    mujoco_params = os.path.join(pkg_share, "config", "mujoco_params.yaml")
    model_path = os.path.join(pkg_share, "models", "rebotarm_dm.xml")
    rebotarm_mesh_dir = os.path.join(rebotarm_bringup_share, "description", "meshes_b601_gripper")
    orbbec_mesh_dir = os.path.join(orbbec_description_share, "meshes", "astra2")
    rviz_config = os.path.join(pkg_share, "rviz", "mujoco_moveit.rviz")
    use_rviz = LaunchConfiguration("use_rviz")
    use_viewer = LaunchConfiguration("use_viewer")

    moveit_config = (
        MoveItConfigsBuilder("rebotarm", package_name="rebotarm_mujoco")
        .robot_description(file_path="config/rebotarm.urdf.xacro")
        .robot_description_semantic(file_path="config/rebotarm.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="true", description="是否启动 RViz MoveIt 控制界面"),
        DeclareLaunchArgument("use_viewer", default_value="true", description="是否启动 MuJoCo GUI"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[moveit_config.robot_description],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="mujoco_world_to_base",
            output="log",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--roll", "0", "--pitch", "0", "--yaw", "0",
                "--frame-id", "world",
                "--child-frame-id", "base_link",
            ],
        ),
        Node(
            package="rebotarm_mujoco",
            executable="mujoco_sim_node",
            name="mujoco_sim_node",
            output="screen",
            parameters=[
                mujoco_params,
                {
                    "model_path": model_path,
                    "rebotarm_mesh_dir": rebotarm_mesh_dir,
                    "orbbec_mesh_dir": orbbec_mesh_dir,
                    "use_viewer": use_viewer,
                },
            ],
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[moveit_parameters(moveit_config)],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="log",
            arguments=["-d", rviz_config],
            condition=IfCondition(use_rviz),
            parameters=[moveit_parameters(moveit_config)],
        ),
    ])
