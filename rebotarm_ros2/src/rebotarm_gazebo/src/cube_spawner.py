"""桌面正方体工具：生成方块，并用 Gazebo Sim DetachableJoint 吸附/释放。"""
from __future__ import annotations

import os
import subprocess
import time

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
import xacro

WORLD_NAME = "arm_and_table"


class CubeSpawner:
    """在 Gazebo 中管理正方体。

    方块 SDF 内置 DetachableJoint：
    - close gripper 后调用 attach()
    - open gripper 后调用 detach()

    这样不需要任务节点自己反复 set_pose 同步方块位置。
    """

    def __init__(self, node: Node, size: float = 0.06, name: str = "") -> None:
        self._node = node
        self._logger = node.get_logger()
        self._size = float(size)
        # name 为空时沿用旧逻辑：每次 spawn 生成一个唯一方块名。
        # 视觉抓取需要“相机看到的方块”和“吸附的方块”是同一个实体，
        # 因此会传入固定名 green_cube，让独立 spawner 和 pipeline 共享话题。
        self._cube_name = str(name).strip()

    # ------------------------------------------------------------------
    def spawn(self, x: float, y: float, z: float) -> bool:
        """生成正方体。"""
        if not self._cube_name:
            self._cube_name = f"green_cube_{int(time.time() * 1000)}"
        mass = 0.05
        inertia = mass * self._size * self._size / 6.0
        model_path = os.path.join(
            get_package_share_directory("rebotarm_gazebo"),
            "worlds",
            "green_cube",
            "model.sdf.xacro",
        )
        sdf = xacro.process_file(
            model_path,
            mappings={
                "name": self._cube_name,
                "size": str(self._size),
                "mass": str(mass),
                "inertia": str(inertia),
            },
        ).toxml()
        try:
            subprocess.run(
                ["ros2", "run", "ros_gz_sim", "create",
                 "-world", WORLD_NAME, "-string", sdf,
                 "-name", self._cube_name,
                 "-x", str(x), "-y", str(y), "-z", str(z)],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            self._logger.error(f"立方体生成失败: {e.stderr}")
            return False
        self._logger.info(f"{self._cube_name} 生成于 ({x:.3f}, {y:.3f}, {z:.3f})")
        return True

    def attach(self) -> bool:
        """通过 DetachableJoint 固定方块和夹爪。"""
        if not self._cube_name:
            return False
        ok = self._publish_empty(f"/{self._cube_name}/attach")
        if not ok:
            return False
        self._logger.info(f"DetachableJoint attach {self._cube_name}")
        return True

    def detach(self) -> bool:
        """通过 DetachableJoint 释放方块。"""
        ok = self._publish_empty(f"/{self._cube_name}/detach")
        if not ok:
            return False
        self._logger.info(f"DetachableJoint detach {self._cube_name}")
        return True

    def _publish_empty(self, topic: str) -> bool:
        """向 Gazebo Transport 发布 Empty 消息。"""
        try:
            result = subprocess.run(
                [
                    "ign", "topic",
                    "-t", topic,
                    "-m", "ignition.msgs.Empty",
                    "-p", "unused: true",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except subprocess.TimeoutExpired:
            self._logger.warn(f"发布 {topic} 超时")
            return False
        if result.returncode != 0:
            self._logger.warn(f"发布 {topic} 失败: {result.stderr.strip()}")
            return False
        return True
