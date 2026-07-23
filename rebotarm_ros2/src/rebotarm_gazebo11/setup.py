"""
rebotarm_gazebo11 包的安装配置文件。

定义包的元数据、数据文件、以及四个可执行入口点。
"""

from glob import glob
import os

from setuptools import setup

package_name = "rebotarm_gazebo11"


def _collect_files(directory: str) -> list[str]:
    """递归收集目录下所有文件路径，用于 data_files 安装。"""
    return [
        path
        for path in glob(os.path.join(directory, "**", "*"), recursive=True)
        if os.path.isfile(path)
    ]


setup(
    name=package_name,
    version="0.1.0",
    package_dir={package_name: "src"},
    packages=[package_name],

    # data_files: 安装到 share/<package>/ 的非 Python 文件
    # 格式: [(目标目录, [源文件列表]), ...]
    data_files=[
        # ament 索引（ROS 2 包发现机制）
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        # 包顶层文件
        (f"share/{package_name}", ["package.xml", "README.md", ".setup_assistant"]),
        # launch / config / rviz / worlds 等子目录
        (f"share/{package_name}/launch", _collect_files("launch")),
        (f"share/{package_name}/config", _collect_files("config")),
        (f"share/{package_name}/rviz", _collect_files("rviz")),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
        (f"share/{package_name}/worlds/ground_plane", _collect_files("worlds/ground_plane")),
        (f"share/{package_name}/worlds/sun", _collect_files("worlds/sun")),
        (f"share/{package_name}/resource", glob("resource/*.so")),
        (f"share/{package_name}/worlds/table", _collect_files("worlds/table")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yyf",
    maintainer_email="yyf@todo.todo",
    description="Gazebo simulation launch and ros2_control integration for reBotArm.",
    license="Apache-2.0",
    extras_require={"test": []},

    # entry_points: 命令行工具 → Python 函数的映射
    # ros2 run rebotarm_gazebo11 <name> 即调用对应的 main 函数
    entry_points={
        "console_scripts": [
            # 生成 Gazebo 机器人描述（URDF 或 SDF 格式）
            "gazebo_robot_description = rebotarm_gazebo11.gazebo_robot_description:main",
            # Gazebo 夹爪弹窗滑条（直接发布 gripper joint trajectory）
            "gripper_slider_gui = rebotarm_gazebo11.gripper_slider_gui:main",
            # 硬件关节状态 → Gazebo 镜像（twin / gazebo_to_hardware 模式）
            "joint_state_mirror = rebotarm_gazebo11.joint_state_mirror:main",
            # 向 MoveIt 添加桌面碰撞物体（避障规划用）
            "planning_scene_objects = rebotarm_gazebo11.planning_scene_objects:main",
            # MoveIt 轨迹 → 硬件中继（gazebo_to_hardware 模式）
            "trajectory_relay = rebotarm_gazebo11.trajectory_relay:main",
            # 方案1：简化版夹取放置（无 MoveIt，直接关节控制）
            "simple_pick_place = rebotarm_gazebo11.simple_pick_place:main",
            # 方案2：MoveIt 版夹取放置（带碰撞检测和运动规划）
            "moveit_pick_place = rebotarm_gazebo11.moveit_pick_place:main",
        ],
    },
)
