"""Gazebo Classic 11 launch — 纯仿真模式。

启动 Gazebo Classic + MoveIt + RViz，不含任何硬件相关组件。
"""

import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def _controller_spawner(controller_name) -> Node:
    return Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            controller_name,
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "120",
        ],
        output="screen",
    )


def generate_launch_description() -> LaunchDescription:
    gazebo_prefix = Path(get_package_prefix("rebotarm_gazebo11"))
    gazebo_share = Path(get_package_share_directory("rebotarm_gazebo11"))
    bringup_share = Path(get_package_share_directory("rebotarm_bringup"))

    world = LaunchConfiguration("world")
    use_rviz = LaunchConfiguration("use_rviz")
    use_gazebo_gui = LaunchConfiguration("use_gazebo_gui")

    default_world = str(gazebo_share / "worlds" / "arm_on_the_table.sdf")
    robot_xacro = str(gazebo_share / "config" / "rebotarm_gazebo.urdf.xacro")
    rviz_config = str(gazebo_share / "rviz" / "gazebo_moveit.rviz")
    arm_controller = "rebotarm_controller"
    gripper_controller = "gripper_controller"

    robot_description = Command(["xacro ", robot_xacro])

    moveit_params_fn = (
        SourceFileLoader(
            "rebotarm_gazebo11_moveit_launch_common",
            str(gazebo_share / "launch" / "moveit_launch_common.py"),
        )
        .load_module()
        .moveit_parameters
    )
    moveit_config = (
        MoveItConfigsBuilder("rebotarm", package_name="rebotarm_gazebo11")
        .robot_description(file_path="config/rebotarm.urdf.xacro")
        .robot_description_semantic(file_path="config/rebotarm.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    moveit_params = moveit_params_fn(moveit_config)
    moveit_params["robot_description"] = robot_description
    moveit_params["use_sim_time"] = True

    model_path = os.pathsep.join(
        [
            str(gazebo_share / "worlds"),
            str(bringup_share.parent),
            str(Path.home() / ".gazebo" / "models"),
            "/usr/share/gazebo-11/models",
        ]
    )
    plugin_path = os.pathsep.join(
        [p for p in [
            str(gazebo_share / "resource"),
            str(gazebo_prefix / "lib"),
            os.environ.get("GAZEBO_PLUGIN_PATH", ""),
            "/opt/ros/humble/lib",
        ] if p]
    )

    gzserver = ExecuteProcess(
        cmd=[
            "gzserver",
            "--verbose",
            "-s",
            "libgazebo_ros_init.so",
            "-s",
            "libgazebo_ros_factory.so",
            world,
        ],
        additional_env={"HOME": "/tmp", "GAZEBO_PLUGIN_PATH": plugin_path},
        output="screen",
    )
    gzclient = ExecuteProcess(
        cmd=["gzclient"],
        additional_env={"HOME": "/tmp", "GAZEBO_PLUGIN_PATH": plugin_path},
        condition=IfCondition(use_gazebo_gui),
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            {"use_sim_time": True, "publish_frequency": 30.0},
        ],
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=[
            "--x",
            "0.05",
            "--y",
            "0.0",
            "--z",
            "0.265",
            "--roll",
            "0.0",
            "--pitch",
            "0.0",
            "--yaw",
            "0.0",
            "--frame-id",
            "world",
            "--child-frame-id",
            "base_link",
        ],
        parameters=[{"use_sim_time": True}],
    )

    spawn_robot = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            (
                "ros2 run rebotarm_gazebo11 gazebo_robot_description "
                f"'{robot_xacro}' --format sdf --world-fixed "
                "| ros2 run gazebo_ros spawn_entity.py "
                "-entity rebotarm -stdin -x 0.05 -y 0.0 -z 0.265 -timeout 120.0"
            ),
        ],
        output="screen",
    )

    jsp_spawner = _controller_spawner("joint_state_broadcaster")
    arm_spawner = _controller_spawner(arm_controller)
    gripper_spawner = _controller_spawner(gripper_controller)

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_params],
    )
    planning_scene_objects = Node(
        package="rebotarm_gazebo11",
        executable="planning_scene_objects",
        name="gazebo_planning_scene_objects",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
        parameters=[moveit_params, {"use_sim_time": True}],
    )
    after_spawn = RegisterEventHandler(
        OnProcessExit(target_action=spawn_robot, on_exit=[jsp_spawner])
    )
    after_jsp = RegisterEventHandler(
        OnProcessExit(target_action=jsp_spawner, on_exit=[arm_spawner])
    )
    after_arm = RegisterEventHandler(
        OnProcessExit(target_action=arm_spawner, on_exit=[gripper_spawner])
    )
    # 控制器就绪 2s 后发 home + 夹爪张开，修正 Classic 初始关节状态
    init_home = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash", "-lc",
                    "ros2 topic pub --once /rebotarm_controller/joint_trajectory "
                    "trajectory_msgs/msg/JointTrajectory "
                    "'{joint_names: [joint1,joint2,joint3,joint4,joint5,joint6], "
                    "points: [{positions: [0.0,-0.05,-0.05,0.0,0.0,0.0], time_from_start: {sec: 3}}]}'"
                ],
                output="screen",
            ),
            ExecuteProcess(
                cmd=[
                    "bash", "-lc",
                    "ros2 topic pub --once /gripper_controller/joint_trajectory "
                    "trajectory_msgs/msg/JointTrajectory "
                    "'{joint_names: [gripper_joint1,gripper_joint2], "
                    "points: [{positions: [0.0,0.0], time_from_start: {sec: 1}}]}'"
                ],
                output="screen",
            ),
        ],
    )

    after_gripper = RegisterEventHandler(
        OnProcessExit(
            target_action=gripper_spawner,
            on_exit=[init_home, move_group, planning_scene_objects, rviz],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value=default_world, description="Gazebo Classic world file"),
            DeclareLaunchArgument("use_rviz", default_value="true", description="Start RViz"),
            DeclareLaunchArgument("use_gazebo_gui", default_value="true", description="Start Gazebo Classic client"),
            SetEnvironmentVariable("GAZEBO_MODEL_PATH", model_path),
            SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
            SetEnvironmentVariable("GAZEBO_LOG_PATH", "/tmp/gazebo"),
            SetEnvironmentVariable("GAZEBO_PLUGIN_PATH", plugin_path),
            gzserver,
            gzclient,
            robot_state_publisher,
            static_tf,
            spawn_robot,
            after_spawn,
            after_jsp,
            after_arm,
            after_gripper,
        ]
    )
