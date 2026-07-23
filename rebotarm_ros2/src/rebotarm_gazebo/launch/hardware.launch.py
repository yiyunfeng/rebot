"""
真机 MoveIt 启动文件 — 直接操作真实机械臂。

启动 move_group + RViz（可选），连接真实硬件进行运动规划。

用法：
    ros2 launch rebotarm_gazebo hardware.launch.py
   
"""

import os
from importlib.machinery import SourceFileLoader

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

moveit_parameters = SourceFileLoader(
    "moveit_launch_common",
    os.path.join(os.path.dirname(__file__), "moveit_launch_common.py"),
).load_module().moveit_parameters


def _declare_arg(name: str, default: str, description: str) -> DeclareLaunchArgument:
    """简化启动参数声明。"""
    return DeclareLaunchArgument(name, default_value=default, description=description)


def generate_launch_description():

    # --- 启动参数（可通过命令行覆盖） ---
    use_rviz = LaunchConfiguration("use_rviz")
    use_camera_rviz = LaunchConfiguration("use_camera_rviz")
    image_view_topic = LaunchConfiguration("image_view_topic")
    arm_namespace = LaunchConfiguration("arm_namespace")
    
    # rviz_config 只接收文件名，避免把绝对路径再次拼到 share/rviz 后面。
    rviz_config = PathJoinSubstitution(
        [
            FindPackageShare("rebotarm_gazebo"),
            "rviz",
            LaunchConfiguration("rviz_config"),
        ]
    )
    
    moveit_config = (
        MoveItConfigsBuilder("rebotarm", package_name="rebotarm_gazebo")
        .robot_description(
            file_path="config/rebotarm_gazebo_camera.urdf.xacro"
        )
        .robot_description_semantic(
            file_path="config/rebotarm.srdf"
        )
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .trajectory_execution(file_path="config/moveit_hardware_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    moveit_params = moveit_parameters(moveit_config)

    # 真机模式没有 Gazebo /clock，必须使用系统时间，否则 RobotState/TF 会出现时间警告。
    hardware_time = {"use_sim_time": False}

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_params, hardware_time],
        remappings=[("/joint_states", ["/",arm_namespace,"/joint_states"])],
    )
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
        parameters=[moveit_params, hardware_time],
        remappings=[("/joint_states", ["/", arm_namespace, "/joint_states"])],
    )

    # 真机相机图像单独开一个 rqt_image_view。主 RViz 保留 MoveIt/机械臂
    # 3D 模型，这个窗口只负责放大显示 DaBai DCW 的 RGB 图。
    camera_image_viewer = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="hardware_camera_rgb_viewer",
        output="screen",
        arguments=[image_view_topic],
        condition=IfCondition(use_camera_rviz),
    )
     
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "world",
            "--child-frame-id", "base_link",
        ],
        parameters=[hardware_time],
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description, hardware_time],
        remappings=[("/joint_states", ["/", arm_namespace, "/joint_states"])],
    )

    return LaunchDescription(
        [
        _declare_arg(
            "rviz_config",
            "gazebo_moveit.rviz",
            "RViz 配置文件",
        ),
        _declare_arg("use_rviz", "true", "是否启动 RViz"),
        _declare_arg("use_camera_rviz", "false", "是否额外启动真机相机大图窗口"),
        _declare_arg("image_view_topic", "/camera/color/image_raw", "rqt_image_view 显示的 RGB 图像话题"),
        _declare_arg("arm_namespace", "rebotarm", "命名空间"),
        static_tf_node,
        robot_state_publisher_node,
        move_group_node,
        rviz_node,
        camera_image_viewer,
        RegisterEventHandler(
            OnProcessExit(
                target_action=move_group_node,
                on_exit=[
                    EmitEvent(event=Shutdown(reason="move_group exited"))
                ],
            )
        ),
        ]
    )


    
