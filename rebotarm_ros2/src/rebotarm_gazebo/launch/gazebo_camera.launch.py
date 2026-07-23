"""
带 DaBai DCW 腕部相机的 Gazebo 仿真启动文件。

用法：
    cd /home/yyf/Desktop/pythonProject/rebot/rebotarm_ros2
    source /opt/ros/humble/setup.bash
    source install/setup.bash

    # 仿真完整抓取链路：Gazebo + MoveIt + RViz + 相机桥接 + 检测 + 抓取
    ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=grasp

    # 真机 HSV 夹取链路：硬件 + 真机 Orbbec + HSV 检测 + 抓取执行节点
    ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=hardware_hsv
    ros2 service call /rebot_grasp/execute_grasp std_srvs/srv/Trigger "{}"

    # MoveIt 手动调试 + 仿真相机和检测结果，不执行抓取
    # 会启动两个 RViz：
    #   1) MoveIt 操作用 RViz
    #   2) 相机图像/点云/检测结果显示用 RViz
    ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=vision

    # 只启动 Gazebo、机器人控制器、TF、相机话题桥接
    ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=base

    # 默认等价于 launch 文件当前默认 mode
    ros2 launch rebotarm_gazebo gazebo_camera.launch.py

仿真视觉后端：
    仿真只使用 HSV 检测 Gazebo 绿色方块，不加载 AI 模型检测。
    真机视觉流程不在本 launch 中处理，保持 rebot_grasp / hardware 流程不变。

启动的内容：
    1. Gazebo 仿真环境（世界文件）
    2. 机械臂模型（SDF 格式）
    3. ros2_control 控制器（arm + gripper）
    4. robot_state_publisher（发布 TF）
    5. DaBai DCW 相机 image / depth / point cloud / camera_info 桥接
    6. MoveIt（按 mode 启动 move_group + RViz）
    7. clock bridge（Gazebo 时钟 → ROS 2）
    8. planning_scene_objects（桌面碰撞物体）
    9. static TF（world → base_link）
    10. gripper_slider_gui（夹爪滑条弹窗，独立于 MoveIt）

mode:
    legacy:       兼容旧参数 use_rviz / start_moveit / start_detector / start_grasp_pipeline
    base:         只启动 Gazebo、机器人控制器、TF、相机话题桥接
    vision:       base + MoveIt + RViz + OpenCV 目标检测，不执行抓取
    full:         vision 的兼容别名
    grasp:        vision + 视觉抓取 pipeline
    hardware_hsv: 真机 + Orbbec + HSV 检测 + 手动触发抓取

启动顺序（受 EventHandler 控制）：
    joint_state_broadcaster → arm_controller → gripper_controller
    → mode 对应的 MoveIt / RViz / 检测 / 抓取节点
"""

import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import (
    IfCondition,
    LaunchConfigurationEquals,
    LaunchConfigurationNotEquals,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _declare_arg(name: str, default: str, description: str) -> DeclareLaunchArgument:
    """简化启动参数声明。"""
    return DeclareLaunchArgument(name, default_value=default, description=description)


def _controller_spawner(controller_name: str, condition=None) -> Node:
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
        condition=condition,
    )


def _mode_condition(mode, legacy_flag, enabled_modes) -> PythonExpression:
    """根据 mode 生成启动条件，并保留 legacy 下旧布尔参数的兼容行为。"""
    return PythonExpression([
        "('", mode, "' == 'legacy' and '", legacy_flag,
        "'.lower() in ['true', '1', 'yes', 'on']) or ('",
        mode, "' in ", repr(list(enabled_modes)), ")",
    ])


# ---------------------------------------------------------------------------
# 主函数：生成 LaunchDescription
# ---------------------------------------------------------------------------

def generate_launch_description() -> LaunchDescription:
    """生成 Gazebo 仿真的完整 LaunchDescription。"""

    # --- 路径 ---
    gazebo_share = get_package_share_directory("rebotarm_gazebo")
    bringup_share = get_package_share_directory("rebotarm_bringup")
    orbbec_share = get_package_share_directory("orbbec_description")

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
    mode = LaunchConfiguration("mode")
    world = LaunchConfiguration("world")
    use_rviz = LaunchConfiguration("use_rviz")
    robot_xacro = LaunchConfiguration("robot_xacro")
    rviz_config = LaunchConfiguration("rviz_config")
    image_view_topic = LaunchConfiguration("image_view_topic")
    start_moveit = LaunchConfiguration("start_moveit")
    start_detector = LaunchConfiguration("start_detector")
    start_grasp_pipeline = LaunchConfiguration("start_grasp_pipeline")
    use_gripper_gui = LaunchConfiguration("use_gripper_gui")
    arm_controller = LaunchConfiguration("arm_controller")
    gripper_controller = LaunchConfiguration("gripper_controller")

    # 默认值
    default_world = os.path.join(gazebo_share, "worlds", "arm_on_the_table.sdf")
    default_xacro = os.path.join(gazebo_share, "config", "rebotarm_gazebo_camera.urdf.xacro")
    detector_config = os.path.join(gazebo_share, "config", "camera_object_detector.yaml")
    joint_pose_config = os.path.join(gazebo_share, "config", "joint_pose_presets.yaml")
    default_rviz = os.path.join(gazebo_share, "rviz", "gazebo_camera.rviz")
    source_worlds = str(Path(__file__).resolve().parents[1] / "worlds")

    # --- mode 映射 ---
    # legacy 继续读取旧参数；其他 mode 用一个参数决定常用启动组合。
    start_moveit_condition = _mode_condition(mode, start_moveit, ["vision", "full", "grasp"])
    use_rviz_condition = _mode_condition(mode, use_rviz, ["full", "vision", "grasp"])
    start_detector_condition = _mode_condition(mode, start_detector, ["full", "vision", "grasp"])
    start_grasp_pipeline_condition = _mode_condition(mode, start_grasp_pipeline, ["grasp"])
    start_table_view_condition = _mode_condition(mode, "false", ["full", "vision", "grasp"])
    delayed_grasp_pipeline_condition = PythonExpression(["'", mode, "' == 'grasp'"])
    sim_condition = LaunchConfigurationNotEquals("mode", "hardware_hsv")
    hardware_hsv_condition = LaunchConfigurationEquals("mode", "hardware_hsv")
    legacy_grasp_pipeline_condition = PythonExpression([
        "('", mode, "' == 'legacy' and '", start_grasp_pipeline,
        "'.lower() in ['true', '1', 'yes', 'on'])",
    ])

    # --- 机器人描述命令 ---
    # ROS 和 Gazebo 都直接读取 xacro；Gazebo 版本额外固定底座。
    robot_desc_cmd = Command([
        "xacro ", robot_xacro, " load_ros2_control:=true",
    ])
    spawn_desc_cmd = Command([
        "xacro ", robot_xacro,
        " load_ros2_control:=true gazebo_world_fixed:=true",
    ])

    # --- MoveIt 配置（仅用于 Gazebo 相机仿真） ---
    # 真机 hardware 模式不会加载本文件；真机由 rebotarm.launch.py include
    # hardware.launch.py，并在 hardware.launch.py 中使用
    # config/moveit_hardware_controllers.yaml。
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
            source_worlds, ":",
            str(Path(orbbec_share).parent.resolve()), ":",
            str(Path(bringup_share).parent.resolve()),
        ],
        condition=sim_condition,
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
        condition=sim_condition,
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
            # 带相机和 ros2_control 的 SDF 更长，默认等待时间可能先报
            # create timeout；加长等待，让 Gazebo 有足够时间返回结果。
            "--timeout", "60000",
        ],
        condition=sim_condition,
    )

    # --- Gazebo 时钟桥接（Gazebo 时钟 → ROS 2 /clock 话题） ---
    # 把 Gazebo Sim 里的 topic
    # 桥接成 ROS 2 topic
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
        parameters=[{"use_sim_time": False}],
        condition=sim_condition,
    )

    # --- RGB-D 相机桥接（Gazebo Sim → ROS 2） ---
    camera_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/dabai_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/dabai_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/dabai_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/dabai_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=sim_condition,
    )

    camera_object_detector = Node(
        package="rebotarm_gazebo",
        executable="camera_object_detector",
        name="camera_object_detector",
        output="screen",
        condition=IfCondition(start_detector_condition),
        parameters=[
            detector_config,
            {
                "use_sim_time": True,
                "color_image_topic": "/dabai_camera/image",
                "depth_image_topic": "/dabai_camera/depth_image",
                "camera_info_topic": "/dabai_camera/camera_info",
                "target_pose_topic": "/dabai_camera/target_pose",
                "debug_image_topic": "/dabai_camera/debug_image",
                "camera_frame": "dabai_camera_optical_frame",
            },
        ],
    )

    camera_grasp_sim = Node(
        package="rebotarm_gazebo",
        executable="camera_grasp_sim",
        name="camera_grasp_sim",
        output="screen",
        condition=IfCondition(delayed_grasp_pipeline_condition),
        parameters=[{
            "use_sim_time": True,
            "execute_grasp": True,
            "target_pose_topic": "/dabai_camera/target_pose",
            "base_frame": "base_link",
            "gripper_topic": ["/", gripper_controller, "/joint_trajectory"],
            "cube_name": "green_cube",
            "cube_size": 0.05,
        }],
    )

    # --- 真机 HSV 链路 ---
    # 这个 mode 不启动 Gazebo，不走 rebot_grasp AI adapter。
    # camera_object_detector 直接订阅真机 Orbbec 图像，把 HSV 检测点发布到
    # /rebot_grasp/grasp_pose；camera_grasp_hardware 等待手动 service 触发执行。
    hardware_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_share, "launch", "rebotarm.launch.py")
        ),
        launch_arguments={
            "mode": "hardware",
            "use_rviz": "true",
            "use_camera_rviz": "false",
            "image_view_topic": "/camera/color/image_raw",
        }.items(),
    )

    hardware_orbbec = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("orbbec_camera"),
                "launch",
                "dabai_dcw.launch.py",
            ])
        ),
        launch_arguments={
            "publish_tf": "false",
            "color_fps": "30",
            "depth_fps": "30",
            "use_rviz": "false",
        }.items(),
    )

    hardware_hsv_detector = Node(
        package="rebotarm_gazebo",
        executable="camera_object_detector",
        name="camera_object_detector",
        output="screen",
        parameters=[
            detector_config,
            {
                "use_sim_time": False,
                "color_image_topic": "/camera/color/image_raw",
                "depth_image_topic": "/camera/depth/image_raw",
                "camera_info_topic": "/camera/color/camera_info",
                "target_pose_topic": "/rebot_grasp/grasp_pose",
                "debug_image_topic": "/rebot_grasp/debug_image",
                "camera_frame": "dabai_camera_optical_frame",
            },
        ],
    )

    hardware_grasp_executor = Node(
        package="rebotarm_gazebo",
        executable="camera_grasp_hardware",
        name="camera_grasp_hardware",
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    hardware_rgb_viewer = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="hardware_hsv_rgb_viewer",
        output="screen",
        arguments=["/camera/color/image_raw"],
    )

    hardware_debug_viewer = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="hardware_hsv_debug_viewer",
        output="screen",
        arguments=["/rebot_grasp/debug_image"],
    )

    # legacy 模式保持旧行为：start_grasp_pipeline:=true 时直接启动抓取节点。
    # 新的 mode:=grasp 则在 table_view_commander 执行完成后再启动，避免机械臂
    # 尚未移动到桌面观察姿态时，pipeline 就开始向 MoveIt 发送抓取规划请求。
    camera_grasp_pipeline_legacy = Node(
        package="rebotarm_gazebo",
        executable="camera_grasp_sim",
        name="camera_grasp_sim",
        output="screen",
        condition=IfCondition(legacy_grasp_pipeline_condition),
        parameters=[{
            "use_sim_time": True,
            "execute_grasp": True,
            "target_pose_topic": "/dabai_camera/target_pose",
            "base_frame": "base_link",
            "gripper_topic": ["/", gripper_controller, "/joint_trajectory"],
            "cube_name": "green_cube",
            "cube_size": 0.05,
        }],
    )

    # --- 视觉模式方块 ---
    # vision 模式不执行抓取，但仍需要桌面上有一个可见方块供相机检测。
    # 这里复用仿真抓取节点，让它只生成方块后退出，不订阅目标、不移动机械臂。
    camera_grasp_sim_cube_only = Node(
        package="rebotarm_gazebo",
        executable="camera_grasp_sim",
        name="camera_grasp_sim_cube_only",
        output="screen",
        condition=IfCondition(PythonExpression(["'", mode, "' in ['vision', 'full']"])),
        parameters=[{
            "use_sim_time": True,
            "execute_grasp": False,
            "cube_name": "green_cube",
            "cube_size": 0.05,
            "cube_x": 0.30,
            "cube_y": 0.15,
            "cube_z": 0.285,
        }],
    )

    # --- 桌面观察姿态 ---
    # home 姿态下腕部相机基本水平，看不到桌面。vision/full/grasp 模式启动后，
    # 先把仿真机械臂移动到 joint_pose_presets.yaml 里的 table_view 姿态，
    # 再让相机检测持续输出结果。真机不在本 launch 中自动移动，避免误动作。
    table_view_commander = Node(
        package="rebotarm_gazebo",
        executable="joint_pose_commander",
        name="joint_pose_commander",
        output="screen",
        condition=IfCondition(start_table_view_condition),
        parameters=[
            joint_pose_config,
            {
                "use_sim_time": True,
                "command_action": ["/", arm_controller, "/follow_joint_trajectory"],
                "enable_before_move": False,
            },
        ],
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
        condition=sim_condition,
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
        condition=sim_condition,
    )

    # --- MoveIt move_group ---
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        condition=IfCondition(start_moveit_condition),
        parameters=[moveit_params],
    )

    # --- 桌面碰撞物体（向 MoveIt 场景添加桌面，用于避障规划） ---
    planning_scene_objects = Node(
        package="rebotarm_gazebo",
        executable="planning_scene_objects",
        name="gazebo_planning_scene_objects",
        output="screen",
        condition=IfCondition(start_moveit_condition),
        parameters=[{"use_sim_time": True}],
    )

    # --- RViz 1：MoveIt 规划操作 ---
    # 注意：当前环境中 moveit_rviz_plugin/MotionPlanning 与
    # rviz_default_plugins/Image 放在同一个 RViz 进程会触发 RViz -11 崩溃。
    # 所以 MoveIt 操作 RViz 只保留 MoveIt 相关显示，不加载 Image 显示。
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="gazebo_camera_moveit_rviz",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz_condition),
        parameters=[moveit_params, {"use_sim_time": True}],
    )

    # --- 独立大图窗口：相机 RGB 图像 ---
    # RViz 的 Image display 更偏“显示项”，不适合做大图查看器；这里改用
    # rqt_image_view，让 RGB 图像独立成一个大窗口。主 RViz 继续保留 MoveIt
    # 和机械臂 3D 模型，不再被图像视图占用。
    camera_image_viewer = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="gazebo_camera_rgb_viewer",
        output="screen",
        arguments=[image_view_topic],
        condition=IfCondition(use_rviz_condition),
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
    jsp_spawner = _controller_spawner("joint_state_broadcaster", sim_condition)
    arm_spawner = _controller_spawner(arm_controller, sim_condition)
    gripper_spawner = _controller_spawner(gripper_controller, sim_condition)

    # --- 按顺序启动控制器 ---
    # 链条：jsp → arm → gripper → MoveIt/RViz/table_view
    # grasp 模式下，抓取节点不再靠固定延时启动，而是等 table_view_commander
    # 执行完成后再启动。这样腕部相机已经朝向桌面，检测结果更稳定。
    # 每一步等前一步的进程退出（即控制器加载完成）后再启动下一步
    after_jsp = RegisterEventHandler(
        OnProcessExit(target_action=jsp_spawner, on_exit=[arm_spawner]),
        condition=sim_condition,
    )
    after_arm = RegisterEventHandler(
        OnProcessExit(target_action=arm_spawner, on_exit=[gripper_spawner]),
        condition=sim_condition,
    )
    after_gripper = RegisterEventHandler(
        OnProcessExit(
            target_action=gripper_spawner,
            on_exit=[
                move_group,
                planning_scene_objects,
                rviz,
                camera_image_viewer,
                gripper_gui,
                camera_grasp_sim_cube_only,
                table_view_commander,
            ],
        ),
        condition=sim_condition,
    )
    after_table_view = RegisterEventHandler(
        OnProcessExit(
            target_action=table_view_commander,
            on_exit=[TimerAction(period=1.0, actions=[camera_grasp_sim])],
        ),
        condition=sim_condition,
    )

    def _runtime_actions(context, *args, **kwargs):
        """按 mode 返回实际启动动作，避免真机 HSV 和 Gazebo 仿真混跑。"""
        del args, kwargs
        mode_value = LaunchConfiguration("mode").perform(context).strip().lower()
        if mode_value == "hardware_hsv":
            return [
                hardware_stack,
                hardware_orbbec,
                hardware_hsv_detector,
                hardware_grasp_executor,
                hardware_rgb_viewer,
                hardware_debug_viewer,
            ]
        return [
            env,
            gazebo,
            spawn_robot,
            clock_bridge,
            camera_bridge,
            camera_object_detector,
            camera_grasp_pipeline_legacy,
            robot_state_publisher,
            static_tf,
            jsp_spawner,
            after_jsp,
            after_arm,
            after_gripper,
            after_table_view,
        ]

    # --- 组装完整启动描述 ---
    return LaunchDescription([
        _declare_arg(
            "mode",
            "legacy",
            "启动模式：legacy/base/vision/full/grasp/hardware_hsv；hardware_hsv 为真机 Orbbec HSV 夹取",
        ),
        _declare_arg("world", default_world, "Gazebo 世界 SDF 文件"),
        _declare_arg("robot_xacro", default_xacro, "Gazebo 专用的机器人 xacro"),
        _declare_arg(
            "rviz_config",
            default_rviz,
            "MoveIt 操作用 RViz 配置文件，不加载 Image 显示，避免 RViz 插件冲突",
        ),
        _declare_arg("image_view_topic", "/dabai_camera/image", "rqt_image_view 显示的 RGB 图像话题"),
        _declare_arg("use_rviz", "true", "legacy 模式下是否启动 RViz"),
        _declare_arg("start_moveit", "true", "legacy 模式下是否启动 move_group"),
        _declare_arg("start_detector", "true", "legacy 模式下是否启动腕部相机 OpenCV 目标检测"),
        _declare_arg("start_grasp_pipeline", "false", "legacy 模式下是否按视觉目标执行抓取动作"),
        _declare_arg("use_gripper_gui", "false", "是否启动夹爪滑条弹窗"),
        _declare_arg("arm_controller", "rebotarm_controller", "Arm 控制器名称"),
        _declare_arg("gripper_controller", "gripper_controller", "Gripper 控制器名称"),
        OpaqueFunction(function=_runtime_actions),
    ])
