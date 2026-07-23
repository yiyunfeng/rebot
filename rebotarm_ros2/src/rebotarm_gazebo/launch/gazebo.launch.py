"""
Gazebo 仿真启动文件。

用法：
    ros2 launch rebotarm_gazebo gazebo.launch.py
    ros2 launch rebotarm_gazebo gazebo.launch.py start_moveit:=false use_rviz:=false

启动的内容：
    1. Gazebo 仿真环境（世界文件）
    2. 机械臂模型（SDF 格式）
    3. ros2_control 控制器（arm + gripper）
    4. robot_state_publisher（发布 TF）
    5. MoveIt（move_group + RViz）
    6. clock bridge（Gazebo 时钟 → ROS 2）
    7. planning_scene_objects（桌面碰撞物体）
    8. static TF（world → base_link）
    9. gripper_slider_gui（夹爪滑条弹窗，独立于 MoveIt）

启动顺序（受 EventHandler 控制）：
    joint_state_broadcaster → arm_controller → gripper_controller
    → move_group, planning_scene_objects, rviz
"""

import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _declare_arg(name: str, default: str, description: str) -> DeclareLaunchArgument:
    """简化启动参数声明。"""
    return DeclareLaunchArgument(name, default_value=default, description=description)


def _controller_spawner(controller_name: str) -> Node:
    """创建一个 ros2_control 控制器 spawner 节点。

    spawner 负责向 controller_manager 请求加载并启动指定控制器。
    """
    return Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            controller_name,
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "60",
        ],
        output="screen",
    )


# ---------------------------------------------------------------------------
# 主函数：生成 LaunchDescription
# ---------------------------------------------------------------------------

def generate_launch_description() -> LaunchDescription:
    """生成 Gazebo 仿真的完整 LaunchDescription。"""

    # --- 路径 ---
    gazebo_share = get_package_share_directory("rebotarm_gazebo")
    bringup_share = get_package_share_directory("rebotarm_bringup")

    # --- 加载 MoveIt 启动参数的构建函数 ---
    moveit_params_fn = (
        SourceFileLoader(
            "moveit_launch_common",
            os.path.join(gazebo_share, "launch", "moveit_launch_common.py"),
        )
        .load_module()
        .moveit_parameters
    )

    # --- 启动参数（可通过命令行覆盖） ---
    world = LaunchConfiguration("world")
    use_rviz = LaunchConfiguration("use_rviz")
    robot_xacro = LaunchConfiguration("robot_xacro")
    rviz_config = LaunchConfiguration("rviz_config")
    start_moveit = LaunchConfiguration("start_moveit")
    use_gripper_gui = LaunchConfiguration("use_gripper_gui")
    arm_controller = LaunchConfiguration("arm_controller")
    gripper_controller = LaunchConfiguration("gripper_controller")

    # 默认值
    default_world = os.path.join(gazebo_share, "worlds", "arm_on_the_table.sdf")
    default_xacro = os.path.join(gazebo_share, "config", "rebotarm_gazebo.urdf.xacro")

    # --- 机器人描述命令 ---
    # ROS 和 Gazebo 都直接读取 xacro；Gazebo 版本额外固定底座。
    robot_desc_cmd = Command([
        "xacro ", robot_xacro, " load_ros2_control:=true",
    ])
    spawn_desc_cmd = Command([
        "xacro ", robot_xacro,
        " load_ros2_control:=true gazebo_world_fixed:=true",
    ])

    # --- MoveIt 配置 ---
    moveit_config = (
        MoveItConfigsBuilder("rebotarm", package_name="rebotarm_gazebo")
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
    moveit_params["robot_description"] = robot_desc_cmd
    moveit_params["use_sim_time"] = True

    # --- 环境变量：Gazebo 资源路径 ---
    env = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            os.path.join(gazebo_share, "worlds"), ":",
            str(Path(bringup_share).parent.resolve()),
        ],
    )

    # --- Gazebo 仿真 ---
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch", "gz_sim.launch.py",
            )
        ),
        launch_arguments={"gz_args": [world, " -v 4 -r"]}.items(),
    )

    # --- 生成机械臂模型到 Gazebo 中 ---
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-string", spawn_desc_cmd,
            "-x", "0.05", "-y", "0.0", "-z", "0.265",
            "-R", "0.0", "-P", "0.0", "-Y", "0.0",
            "-name", "rebotarm",
            "-allow_renaming", "false",
            # 机器人 SDF 很长，默认等待时间偏短时会先打印 timeout，
            # 但 Gazebo 后续仍可能创建成功；加长等待避免误判。
            "--timeout", "60000",
        ],
    )

    # --- Gazebo 时钟桥接（Gazebo 时钟 → ROS 2 /clock 话题） ---
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    # --- Robot State Publisher（发布 TF 变换树） ---
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            {"robot_description": robot_desc_cmd},
            {"use_sim_time": True, "publish_frequency": 30.0},
        ],
    )

    # --- 静态 TF：world → base_link ---
    # Gazebo 中机械臂被放置在 (0.05, 0, 0.265)，
    # 需要一个静态 TF 让 ROS 2 的 TF 树和 Gazebo 的位姿保持一致
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=[
            "--x", "0.05", "--y", "0.0", "--z", "0.265",
            "--roll", "0.0", "--pitch", "0.0", "--yaw", "0.0",
            "--frame-id", "world",
            "--child-frame-id", "base_link",
        ],
        parameters=[{"use_sim_time": True}],
    )

    # --- MoveIt move_group ---
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        condition=IfCondition(start_moveit),
        parameters=[moveit_params],
    )

    # --- 桌面碰撞物体（向 MoveIt 场景添加桌面，用于避障规划） ---
    planning_scene_objects = Node(
        package="rebotarm_gazebo",
        executable="planning_scene_objects",
        name="gazebo_planning_scene_objects",
        output="screen",
        condition=IfCondition(start_moveit),
        parameters=[{"use_sim_time": True}],
    )

    # --- RViz 可视化 ---
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
        parameters=[moveit_params, {"use_sim_time": True}],
    )

    # --- 夹爪滑条弹窗（备用） ---
    # 默认关闭。需要绕过 MoveIt 手动调夹爪时，启动参数加 use_gripper_gui:=true。
    gripper_gui = Node(
        package="rebotarm_gazebo",
        executable="gripper_slider_gui",
        name="gripper_slider_gui",
        output="screen",
        condition=IfCondition(use_gripper_gui),
        parameters=[{
            "use_sim_time": True,
            "command_topic": ["/", gripper_controller, "/joint_trajectory"],
            "joint_names": ["gripper_joint1", "gripper_joint2"],
            "min_position": 0.0,
            "max_position": 0.0715,
            "initial_position": 0.0,
            "motion_duration": 0.5,
        }],
    )

    # --- ros2_control 控制器 spawner ---
    jsp_spawner = _controller_spawner("joint_state_broadcaster")
    arm_spawner = _controller_spawner(arm_controller)
    gripper_spawner = _controller_spawner(gripper_controller)

    # --- 按顺序启动控制器 ---
    # 链条：jsp → arm → gripper → MoveIt/RViz
    # 每一步等前一步的进程退出（即控制器加载完成）后再启动下一步
    after_jsp = RegisterEventHandler(
        OnProcessExit(target_action=jsp_spawner, on_exit=[arm_spawner])
    )
    after_arm = RegisterEventHandler(
        OnProcessExit(target_action=arm_spawner, on_exit=[gripper_spawner])
    )
    after_gripper = RegisterEventHandler(
        OnProcessExit(
            target_action=gripper_spawner,
            on_exit=[move_group, planning_scene_objects, rviz, gripper_gui],
        )
    )

    # --- 组装完整启动描述 ---
    return LaunchDescription([
        _declare_arg("world", default_world, "Gazebo 世界 SDF 文件"),
        _declare_arg("robot_xacro", default_xacro, "Gazebo 专用的机器人 xacro"),
        _declare_arg(
            "rviz_config",
            os.path.join(gazebo_share, "rviz", "gazebo_moveit.rviz"),
            "RViz 配置文件",
        ),
        _declare_arg("use_rviz", "true", "是否启动 RViz"),
        _declare_arg("start_moveit", "true", "是否启动 move_group"),
        _declare_arg("use_gripper_gui", "false", "是否启动夹爪滑条弹窗"),
        _declare_arg("arm_controller", "rebotarm_controller", "Arm 控制器名称"),
        _declare_arg("gripper_controller", "gripper_controller", "Gripper 控制器名称"),
        env,
        gazebo,
        spawn_robot,
        clock_bridge,
        robot_state_publisher,
        static_tf,
        jsp_spawner,
        after_jsp,
        after_arm,
        after_gripper,
    ])
