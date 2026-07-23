"""
Orbbec DaBai DCW 真机相机 ROS2 驱动启动文件。

用法：
    cd /home/yyf/Desktop/pythonProject/rebot/rebotarm_ros2
    source /opt/ros/humble/setup.bash
    source install/setup.bash

    # USB2 稳定模式：默认 640x360@5fps，发布 RGB、Depth、CameraInfo、PointCloud。
    ros2 launch orbbec_camera dabai_dcw.launch.py

    # USB3 或带宽足够时，可手动提高帧率。
    ros2 launch orbbec_camera dabai_dcw.launch.py color_fps:=10 depth_fps:=10

    # 打开 RViz 查看彩色图、深度图和点云。
    ros2 launch orbbec_camera dabai_dcw.launch.py use_rviz:=true

输出话题：
    /camera/color/image_raw
    /camera/color/camera_info
    /camera/depth/image_raw
    /camera/depth/camera_info
    /camera/depth/points

注意：
    这里只负责启动真实 DaBai DCW 相机。
    仿真相机命令写在 rebotarm_gazebo/launch/gazebo_camera.launch.py。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import PushRosNamespace
from launch.actions import GroupAction
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import os


def generate_launch_description():
    # DaBai DCW 在当前设备上以 USB2.0 连接时，带宽比 USB3 低很多。
    # 这里的默认参数优先保证稳定启动：
    #   1. 固定 product_id=0x0659，明确选择 DaBai DCW；
    #   2. RGB/Depth 都使用 640x360@5fps，降低 USB2 带宽压力；
    #   3. 保留普通点云，关闭彩色点云，避免额外同步和彩色对齐开销。
    # 如果后续换到 USB3，可以在命令行覆盖 color_fps/depth_fps 到 10 或 30。
    args = [
        DeclareLaunchArgument('camera_name', default_value='camera'),
        DeclareLaunchArgument('depth_registration', default_value='false'),
        DeclareLaunchArgument('serial_number', default_value=''),
        DeclareLaunchArgument('usb_port', default_value=''),
        DeclareLaunchArgument('device_num', default_value='1'),
        DeclareLaunchArgument('vendor_id', default_value='0x2bc5'),
        DeclareLaunchArgument('product_id', default_value='0x0659'),
        DeclareLaunchArgument('enable_point_cloud', default_value='true'),
        DeclareLaunchArgument('enable_colored_point_cloud', default_value='false'),
        DeclareLaunchArgument('cloud_frame_id', default_value=''),
        DeclareLaunchArgument('point_cloud_qos', default_value='default'),
        DeclareLaunchArgument('connection_delay', default_value='100'),
        DeclareLaunchArgument('color_width', default_value='640'),
        DeclareLaunchArgument('color_height', default_value='360'),
        DeclareLaunchArgument('color_fps', default_value='5'),
        DeclareLaunchArgument('color_format', default_value='MJPG'),
        DeclareLaunchArgument('enable_color', default_value='true'),
        DeclareLaunchArgument('flip_color', default_value='false'),
        DeclareLaunchArgument('color_qos', default_value='default'),
        DeclareLaunchArgument('color_camera_info_qos', default_value='default'),
        DeclareLaunchArgument('enable_color_auto_exposure', default_value='true'),
        DeclareLaunchArgument('enable_color_auto_exposure_priority', default_value='false'),
        DeclareLaunchArgument('color_exposure', default_value='-1'),
        DeclareLaunchArgument('color_gain', default_value='-1'),
        DeclareLaunchArgument('enable_color_auto_white_balance', default_value='true'),
        DeclareLaunchArgument('color_white_balance', default_value='-1'),
        DeclareLaunchArgument('depth_width', default_value='640'),
        DeclareLaunchArgument('depth_height', default_value='360'),
        DeclareLaunchArgument('depth_fps', default_value='5'),
        DeclareLaunchArgument('depth_format', default_value='Y11'),
        DeclareLaunchArgument('enable_depth', default_value='true'),
        DeclareLaunchArgument('flip_depth', default_value='false'),
        DeclareLaunchArgument('min_depth_limit', default_value='0'),
        DeclareLaunchArgument('max_depth_limit', default_value='0'),
        DeclareLaunchArgument('depth_qos', default_value='default'),
        DeclareLaunchArgument('depth_camera_info_qos', default_value='default'),
        DeclareLaunchArgument('ir_width', default_value='640'),
        DeclareLaunchArgument('ir_height', default_value='480'),
        DeclareLaunchArgument('ir_fps', default_value='10'),
        DeclareLaunchArgument('ir_format', default_value='Y10'),
        DeclareLaunchArgument('enable_ir', default_value='false'),
        DeclareLaunchArgument('flip_ir', default_value='false'),
        DeclareLaunchArgument('ir_qos', default_value='default'),
        DeclareLaunchArgument('ir_camera_info_qos', default_value='default'),
        DeclareLaunchArgument('enable_ir_auto_exposure', default_value='true'),
        DeclareLaunchArgument('ir_exposure', default_value='-1'),
        DeclareLaunchArgument('ir_gain', default_value='-1'),
        DeclareLaunchArgument('publish_tf', default_value='true'),
        DeclareLaunchArgument('tf_publish_rate', default_value='0.0'),
        DeclareLaunchArgument('ir_info_url', default_value=''),
        DeclareLaunchArgument('color_info_url', default_value=''),
        DeclareLaunchArgument('log_level', default_value='none'),
        DeclareLaunchArgument('enable_publish_extrinsic', default_value='false'),
        DeclareLaunchArgument('enable_d2c_viewer', default_value='false'),
        DeclareLaunchArgument('enable_ldp', default_value='true'),
        DeclareLaunchArgument('enable_soft_filter', default_value='true'),
        DeclareLaunchArgument('soft_filter_max_diff', default_value='-1'),
        DeclareLaunchArgument('soft_filter_speckle_size', default_value='-1'),
        DeclareLaunchArgument('ordered_pc', default_value='false'),
        DeclareLaunchArgument('enable_depth_scale', default_value='true'),
        DeclareLaunchArgument('align_mode', default_value='HW'),
        DeclareLaunchArgument('laser_energy_level', default_value='-1'),
        DeclareLaunchArgument('enable_heartbeat', default_value='false'),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='是否启动 RViz，用于查看 color/depth/pointcloud。',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution(
                [FindPackageShare('orbbec_camera'), 'rviz', 'pointcloud.rviz']
            ),
            description='RViz 配置文件路径，默认显示彩色图、深度图和点云。',
        ),
    ]

    # Node configuration
    # 只有相机驱动参数传给 OBCameraNodeDriver；RViz 参数是启动层参数，
    # 如果一起传给驱动会变成无意义的 ROS 参数，后续排查配置时容易混淆。
    camera_arg_names = [
        arg.name for arg in args if arg.name not in ('use_rviz', 'rviz_config')
    ]
    parameters = [{name: LaunchConfiguration(name)} for name in camera_arg_names]

    # RViz 只在 use_rviz:=true 时启动。默认关闭是为了让无显示器/SSH 终端
    # 也能稳定启动相机驱动；需要看图像时手动打开即可。
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='orbbec_dabai_dcw_rviz',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen',
    )
    # get  ROS_DISTRO
    ros_distro = os.environ["ROS_DISTRO"]
    if ros_distro == "foxy":
        return LaunchDescription(
            args
            + [
                Node(
                    package="orbbec_camera",
                    executable="orbbec_camera_node",
                    name="ob_camera_node",
                    namespace=LaunchConfiguration("camera_name"),
                    parameters=parameters,
                    output="screen",
                ),
                rviz_node,
            ]
        )
    # Define the ComposableNode
    else:
        # Define the ComposableNode
        compose_node = ComposableNode(
            package="orbbec_camera",
            plugin="orbbec_camera::OBCameraNodeDriver",
            name=LaunchConfiguration("camera_name"),
            namespace="",
            parameters=parameters,
        )
        # Define the ComposableNodeContainer
        container = ComposableNodeContainer(
            name="camera_container",
            namespace="",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                compose_node,
            ],
            output="screen",
        )
        # Launch description
        ld = LaunchDescription(
            args
            + [
                GroupAction(
                    [PushRosNamespace(LaunchConfiguration("camera_name")), container]
                ),
                rviz_node,
            ]
        )
        return ld
