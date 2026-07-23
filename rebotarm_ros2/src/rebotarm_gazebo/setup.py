"""
rebotarm_gazebo 包的安装配置文件。

定义包的元数据、数据文件、以及四个可执行入口点。
"""

from glob import glob
import os

from setuptools import setup

package_name = "rebotarm_gazebo"


def _collect_files(directory: str) -> list[str]:
    """递归收集目录下所有文件路径，用于 data_files 安装。"""
    return [
        path
        for path in glob(os.path.join(directory, "**", "*"), recursive=True)
        # data_files 只安装源码资源；跳过 Python 编译缓存，避免 symlink install 失效。
        if os.path.isfile(path)
        and "__pycache__" not in path.split(os.sep)
        and not path.endswith((".pyc", ".pyo"))
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
        (f"share/{package_name}/worlds/table", _collect_files("worlds/table")),
        (f"share/{package_name}/worlds/green_cube", _collect_files("worlds/green_cube")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yyf",
    maintainer_email="yyf@todo.todo",
    description="Gazebo simulation launch and ros2_control integration for reBotArm.",
    license="Apache-2.0",
    extras_require={"test": []},

    # entry_points: 命令行工具 → Python 函数的映射
    # ros2 run rebotarm_gazebo <name> 即调用对应的 main 函数
    entry_points={
        "console_scripts": [
            # Gazebo 夹爪弹窗滑条（直接发布 gripper joint trajectory）
            "gripper_slider_gui = rebotarm_gazebo.gripper_slider_gui:main",
            # 硬件关节状态 → Gazebo 镜像（twin 模式）
            "joint_state_mirror = rebotarm_gazebo.joint_state_mirror:main",
            # 向 MoveIt 添加桌面碰撞物体（避障规划用）
            "planning_scene_objects = rebotarm_gazebo.planning_scene_objects:main",
            # twin 模式：启动真机重力补偿，不 safe_home
            "twin_gravity_compensation = rebotarm_gazebo.twin_gravity_compensation:main",
            # 腕部 RGB-D 相机 OpenCV 目标检测
            "camera_object_detector = rebotarm_gazebo.camera_object_detector:main",
            # 旧入口：自动转到 camera_grasp_sim / camera_grasp_hardware
            "camera_grasp_pipeline = rebotarm_gazebo.camera_grasp_pipeline:main",
            # 仿真相机目标 → Gazebo MoveIt 抓取流程（含方块生成和吸附）
            "camera_grasp_sim = rebotarm_gazebo.camera_grasp_sim:main",
            # 真机相机目标 → 真实机械臂抓取流程
            "camera_grasp_hardware = rebotarm_gazebo.camera_grasp_hardware:main",
            # 命名关节姿态执行器：如 table_view 桌面观察姿态
            "joint_pose_commander = rebotarm_gazebo.joint_pose_commander:main",
            # 方案1：简化版夹取放置（无 MoveIt，直接关节控制）
            "simple_pick_place = rebotarm_gazebo.simple_pick_place:main",
            # 方案2：MoveIt 版夹取放置（带碰撞检测和运动规划）
            "moveit_pick_place = rebotarm_gazebo.moveit_pick_place:main",
        ],
    },
)
