# =============================================================================
# ROS2 Launch 文件：rebotarm 机械臂的完整启动脚本
# 功能：启动机械臂控制器、robot_state_publisher 和可选的 RViz2 可视化
# =============================================================================

# ---- 导入 launch 核心模块 ----
from launch import LaunchDescription  # 启动描述对象，用于组织所有启动动作

from launch.actions import (
    DeclareLaunchArgument,  # 声明可配置的启动参数（命令行或 YAML 传入）
    Shutdown,               # 优雅关闭整个 launch 系统
)
from launch.conditions import IfCondition  # 条件判断，根据参数决定是否启动某个节点

# ---- 导入 launch 替换（substitutions）模块 ----
# substitutions 提供在运行时动态构造字符串值的能力
from launch.substitutions import (
    Command,              # 执行 shell 命令并将输出作为参数值（例如 cat 一个文件）
    LaunchConfiguration,  # 获取用户在启动时传入的 launch 参数值
    PathJoinSubstitution, # 跨平台的路径拼接（自动处理 / 和 \）
)

# ---- 导入 launch_ros 模块 ----
from launch_ros.actions import Node  # 启动一个 ROS2 节点
from launch_ros.parameter_descriptions import ParameterValue  # 包装参数值，可指定类型
from launch_ros.substitutions import FindPackageShare  # 查找 ROS2 包的 share 目录路径


# =============================================================================
# generate_launch_description() 是 launch 文件的入口函数
# 每次启动时都会被调用，必须返回一个 LaunchDescription 对象
# =============================================================================
def generate_launch_description():
    # ---- 获取 rebotarm_bringup 包的 share 目录绝对路径 ----
    bringup_share = FindPackageShare("rebotarm_bringup")

    # ---- 声明所有可配置参数的占位符 ----
    # LaunchConfiguration 不会立即求值，而是在运行时才读取用户传入的值
    hardware_config = LaunchConfiguration("hardware_config")  # 硬件配置文件路径
    model = LaunchConfiguration("model")                      # 机械臂型号；本项目仅保留 DM
    channel = LaunchConfiguration("channel")                  # 通信通道（如 CAN 通道名）
    joint_state_rate = LaunchConfiguration("joint_state_rate")  # 关节状态发布频率（Hz）
    cmd_arbitration = LaunchConfiguration("cmd_arbitration")  # 命令仲裁策略（reject / queue）
    arm_namespace = LaunchConfiguration("arm_namespace")      # 机械臂话题命名空间
    use_rviz = LaunchConfiguration("use_rviz")                # 是否启动 RViz2 可视化
    frame_id = LaunchConfiguration("frame_id")                # 机器人基座的 TF 坐标系名称
    ee_frame_id = LaunchConfiguration("ee_frame_id")          # 末端执行器的 TF 坐标系名称
    disable_after_safe_home = LaunchConfiguration("disable_after_safe_home")  # 安全回零后是否禁用电机

    # ---- DM URDF 模型文件 ----
    # 本仓库后续只维护 DM 机械臂，URDF 固定使用 DevArm 模型。
    urdf_file = PathJoinSubstitution(
        [
            bringup_share,     # 包 share 目录
            "description",     # 描述文件子目录
            "urdf",            # URDF 文件子目录
            "reBot-DevArm_fixend.urdf",
        ]
    )

    # ---- 拼接 RViz2 配置文件路径 ----
    rviz_config = PathJoinSubstitution([bringup_share, "rviz", "rebotarm.rviz"])

    # ---- 读取 URDF 文件内容并封装为参数值 ----
    # Command("cat ...") 会在运行时执行 cat 命令读取 URDF 文件内容
    # ParameterValue 将其包装为字符串类型的 ROS 参数
    robot_description = ParameterValue(Command(["cat ", urdf_file]), value_type=str)

    # ---- 构建并返回完整的启动描述 ----
    return LaunchDescription(
        [
            # ================================================================
            # 1. 声明启动参数（DeclareLaunchArgument）
            #    这些参数可在命令行通过 key:=value 传入，未传入则使用 default_value
            # ================================================================

            # 硬件配置文件路径（默认指向 rebotarm_hardware.yaml）
            DeclareLaunchArgument(
                "hardware_config",
                default_value=PathJoinSubstitution(
                    [bringup_share, "config", "rebotarm_hardware.yaml"]
                ),
            ),
            # 机械臂型号（空字符串或 dm 表示 DM；RS 配置已移除）
            DeclareLaunchArgument("model", default_value=""),
            # 通信通道名称（空字符串表示使用默认通道）
            DeclareLaunchArgument("channel", default_value=""),
            # 关节状态发布频率，默认 100Hz
            DeclareLaunchArgument("joint_state_rate", default_value="100.0"),
            # 命令仲裁策略：reject=拒绝新命令, queue=排队等待
            DeclareLaunchArgument("cmd_arbitration", default_value="reject"),
            # 机械臂话题命名空间，默认为 "rebotarm"
            DeclareLaunchArgument("arm_namespace", default_value="rebotarm"),
            # 是否启动 RViz2，默认不启动
            DeclareLaunchArgument("use_rviz", default_value="false"),
            # 基座 TF 坐标系名称
            DeclareLaunchArgument("frame_id", default_value="base_link"),
            # 末端执行器 TF 坐标系名称
            DeclareLaunchArgument("ee_frame_id", default_value="end_link"),
            # 安全回零后是否禁用电机，默认启用
            DeclareLaunchArgument("disable_after_safe_home", default_value="true"),

            # ================================================================
            # 2. 启动 reBotArmController 控制器节点
            #    这是机械臂的核心控制节点，负责电机驱动、运动控制和状态反馈
            # ================================================================
            Node(
                package="rebotarmcontroller",            # 所属 ROS2 包名
                executable="reBotArmController",         # 可执行文件名
                name="reBotArmController",               # 节点名称（用于日志和 ros2 node list）
                output="screen",                         # 将 stdout/stderr 输出到终端
                on_exit=Shutdown(reason="reBotArmController exited"),  # 控制器退出时关闭整个 launch
                parameters=[                            # 传入的参数列表
                    {
                        "hardware_config": hardware_config,           # 硬件配置 YAML 路径
                        "model": model,                               # 机械臂型号
                        "channel": channel,                           # 通信通道
                        "joint_state_rate": joint_state_rate,         # 关节状态频率
                        "cmd_arbitration": cmd_arbitration,           # 命令仲裁策略
                        "arm_namespace": arm_namespace,               # 话题命名空间
                        "frame_id": frame_id,                         # 基座坐标系
                        "ee_frame_id": ee_frame_id,                   # 末端坐标系
                        "disable_after_safe_home": ParameterValue(    # 布尔类型参数
                            disable_after_safe_home,                  # 安全回零后禁用电机
                            value_type=bool,                          # 显式指定为 bool 类型
                        ),
                    }
                ],
            ),

            # ================================================================
            # 3. 启动 robot_state_publisher 节点
            #    读取 URDF 中的 robot_description 参数，发布机器人的 TF 变换树
            # ================================================================
            Node(
                package="robot_state_publisher",         # 所属 ROS2 包名
                executable="robot_state_publisher",      # 可执行文件名
                name="robot_state_publisher",            # 节点名称
                output="screen",                         # 输出到终端
                parameters=[{"robot_description": robot_description}],  # 传入 URDF 描述
                remappings=[                            # 话题重映射
                    # 将默认的 /joint_states 话题重映射到命名空间下的 /<namespace>/joint_states
                    ("/joint_states", ["/", arm_namespace, "/joint_states"]),
                ],
            ),

            # ================================================================
            # 4. 启动 RViz2 可视化节点（可选，仅当 use_rviz=true 时启动）
            # ================================================================
            Node(
                package="rviz2",                        # 所属 ROS2 包名
                executable="rviz2",                     # 可执行文件名
                name="rviz2",                           # 节点名称
                output="screen",                        # 输出到终端
                arguments=["-d", rviz_config],          # 命令行参数：-d 指定 RViz 配置文件路径
                condition=IfCondition(use_rviz),        # 条件启动：仅当 use_rviz 为 true 时启动
            ),
        ]
    )
