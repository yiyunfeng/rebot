"""
reBotArm 总启动文件 — 支持三种运行模式。

模式说明：
    sim              纯仿真：Gazebo + MoveIt（无需硬件）
    hardware         纯硬件：真实机械臂 + MoveIt（无需 Gazebo）
    twin             数字孪生：Gazebo 镜像硬件的运动

用法：
    ros2 launch rebotarm_gazebo rebotarm.launch.py mode:=sim
    ros2 launch rebotarm_gazebo rebotarm.launch.py mode:=hardware
    ros2 launch rebotarm_gazebo rebotarm.launch.py mode:=twin
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """生成包含三种模式的启动描述。

    mode 参数决定实际启动哪些组件。
    OpaqueFunction 在启动时被调用，可读取参数值动态返回不同的启动动作。
    """
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mode",
                default_value="sim",
                description="运行模式: sim, hardware, twin",
            ),
            DeclareLaunchArgument("model", default_value="dm"),
            DeclareLaunchArgument("channel", default_value=""),
            DeclareLaunchArgument("arm_namespace", default_value="rebotarm"),
            DeclareLaunchArgument("world", default_value=""),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_camera_rviz", default_value="false"),
            DeclareLaunchArgument("image_view_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("joint_state_rate", default_value="100.0"),
            DeclareLaunchArgument("cmd_arbitration", default_value="reject"),
            DeclareLaunchArgument("disable_after_safe_home", default_value="true"),
            DeclareLaunchArgument(
                "twin_start_gravity_compensation",
                default_value="true",
                description="twin 模式是否自动启动真机重力补偿",
            ),
            DeclareLaunchArgument(
                "twin_safe_home_on_exit",
                default_value="true",
                description="twin 重力补偿节点退出时是否调用 safe_home",
            ),
            DeclareLaunchArgument(
                "twin_disable_on_exit",
                default_value="true",
                description="twin 重力补偿节点退出时是否调用 disable",
            ),
            # OpaqueFunction 在启动时调用 _launch_setup，动态决定启动内容
            OpaqueFunction(function=_launch_setup),
        ]
    )


# ---------------------------------------------------------------------------
# 主启动逻辑：根据 mode 参数选择启动哪些组件
# ---------------------------------------------------------------------------


def _launch_setup(context, *args, **kwargs):
    """根据 mode 参数返回不同的启动动作列表。"""
    del args, kwargs

    mode = LaunchConfiguration("mode").perform(context).strip().lower()
    model = LaunchConfiguration("model").perform(context).strip().lower()
    if model not in ("", "dm"):
        raise ValueError(f"不支持的机械臂型号 model={model!r}，本项目仅保留 DM")

    gazebo_share = get_package_share_directory("rebotarm_gazebo")
    bringup_share = get_package_share_directory("rebotarm_bringup")

    arm_namespace = LaunchConfiguration("arm_namespace").perform(context).strip("/")
    use_rviz = LaunchConfiguration("use_rviz").perform(context).strip().lower()

    # 世界文件：用参数值或用默认路径
    world_value = LaunchConfiguration("world").perform(context).strip()
    world = (
        world_value
        if world_value
        else os.path.join(gazebo_share, "worlds", "arm_on_the_table.sdf")
    )

    # --- 三种模式 ---

    if mode == "sim":
        # 纯仿真：Gazebo + MoveIt（无需硬件驱动）
        return [
            _include(
                gazebo_share,
                "gazebo.launch.py",
                {
                    "world": world,
                    "use_rviz": LaunchConfiguration("use_rviz"),
                    "start_moveit": "true",
                    "arm_controller": "rebotarm_controller",
                    "gripper_controller": "gripper_controller",
                },
            ),
        ]

    if mode == "hardware":
        # 纯硬件：驱动 + MoveIt（连接真实机械臂，无需 Gazebo）
        return [
            _include_driver(bringup_share),
            _include_moveit(gazebo_share),
        ]

    if mode == "twin":
        # 数字孪生：真机进入重力补偿，可手动拖动；
        # joint_state_mirror 将真机 arm + gripper 状态镜像到 Gazebo。
        # 这里不启动硬件 MoveIt，避免 MoveIt/控制器向真机下位置轨迹导致难以拖动。
        actions = [
            _include_driver(bringup_share),
            _include(
                gazebo_share,
                "gazebo.launch.py",
                {
                    "world": world,
                    "use_rviz": "false",
                    "start_moveit": "false",
                    "arm_controller": "gazebo_rebotarm_controller",
                    "gripper_controller": "gazebo_gripper_controller",
                },
            ),
            _twin_gravity_node(arm_namespace),
            _mirror_node(arm_namespace),
        ]
        if use_rviz in ("true", "1", "yes", "on"):
            actions.append(_twin_rviz_node(gazebo_share))
        return actions

    raise ValueError(f"不支持的模式 mode={mode!r}，可选值: sim, hardware, twin")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _include(pkg_share: str, launch_file: str, args: dict):
    """引入另一个 launch 文件。

    Args:
        pkg_share: 包的 share 目录路径。
        launch_file: launch 文件名（如 "gazebo.launch.py"）。
        args: 传给子 launch 文件的参数字典。
    """
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, "launch", launch_file)),
        launch_arguments=args.items(),
    )


def _include_driver(bringup_share: str):
    """引入硬件驱动 launch 文件（连接真实机械臂）。"""
    return _include(
        bringup_share,
        "driver.launch.py",
        {
            "model": LaunchConfiguration("model"),
            "channel": LaunchConfiguration("channel"),
            "joint_state_rate": LaunchConfiguration("joint_state_rate"),
            "cmd_arbitration": LaunchConfiguration("cmd_arbitration"),
            "arm_namespace": LaunchConfiguration("arm_namespace"),
            "disable_after_safe_home": LaunchConfiguration("disable_after_safe_home"),
        },
    )


def _include_moveit(gazebo_share: str):
    """引入硬件 MoveIt launch 文件。"""
    return _include(
        gazebo_share,
        "hardware.launch.py",
        {
            "arm_namespace": LaunchConfiguration("arm_namespace"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "use_camera_rviz": LaunchConfiguration("use_camera_rviz"),
            "image_view_topic": LaunchConfiguration("image_view_topic"),
        },
    )


def _mirror_node(namespace: str) -> Node:
    """创建 joint_state_mirror 节点。

    订阅硬件关节状态 → 发布到 Gazebo 控制器，
    让 Gazebo 中机械臂跟随真实机械臂运动。
    """
    return Node(
        package="rebotarm_gazebo",
        executable="joint_state_mirror",
        name="joint_state_mirror",
        output="screen",
        parameters=[
            {
                "source_joint_states": f"/{namespace}/joint_states",
                "source_gripper_state": f"/{namespace}/gripper/state",
                "arm_command_topic": "/gazebo_rebotarm_controller/joint_trajectory",
                "gripper_command_topic": "/gazebo_gripper_controller/joint_trajectory",
                "gripper_joint_names": ["gripper_joint1", "gripper_joint2"],
                "publish_rate": 30.0,
                "point_duration": 0.08,
                "log_period": 3.0,
            }
        ],
    )


def _twin_gravity_node(namespace: str) -> Node:
    """创建 twin 专用重力补偿节点。

    该节点只启动/停止重力补偿，不会在退出时 safe_home。
    """
    return Node(
        package="rebotarm_gazebo",
        executable="twin_gravity_compensation",
        name="twin_gravity_compensation",
        output="screen",
        condition=IfCondition(LaunchConfiguration("twin_start_gravity_compensation")),
        parameters=[
            {
                "namespace": namespace,
                "safe_home_on_exit": LaunchConfiguration("twin_safe_home_on_exit"),
                "disable_on_exit": LaunchConfiguration("twin_disable_on_exit"),
            }
        ],
    )


def _twin_rviz_node(gazebo_share: str) -> Node:
    """创建不依赖 MoveIt 的 twin RViz。

    只显示 TF 和 RobotModel，用来观察真机状态是否镜像到仿真模型。
    """
    robot_xacro = os.path.join(gazebo_share, "config", "rebotarm_gazebo.urdf.xacro")
    robot_desc_cmd = Command(["xacro ", robot_xacro])
    return Node(
        package="rviz2",
        executable="rviz2",
        name="twin_rviz",
        output="screen",
        arguments=["-d", os.path.join(gazebo_share, "rviz", "twin.rviz")],
        parameters=[
            {
                "robot_description": robot_desc_cmd,
                "use_sim_time": True,
            }
        ],
    )
