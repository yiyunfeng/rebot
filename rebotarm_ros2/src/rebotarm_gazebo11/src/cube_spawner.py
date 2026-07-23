"""桌面正方体工具 — Gazebo Classic spawn / delete。"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time

from rclpy.node import Node

CUBE_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{inertia}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{inertia}</iyy><iyz>0</iyz>
          <izz>{inertia}</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
        <material>
          <ambient>0 0.8 0 1</ambient>
          <diffuse>0 0.8 0 1</diffuse>
        </material>
      </visual>
      <!-- 碰撞缩到 90%，手指闭合时能碰到，搬运时留间隙不重叠 -->
      <collision name="collision">
          <geometry><box><size>{size} {size} {size}</size></box></geometry>
        <surface>
          <friction>
            <ode>
              <mu>1.5</mu>
              <mu2>1.5</mu2>
              <slip1>0</slip1>
              <slip2>0</slip2>
            </ode>
          </friction>
          <contact>
            <ode>
              <kp>1e3</kp>
              <kd>10</kd>
            </ode>
          </contact>
        </surface>
      </collision>
    </link>
  </model>
</sdf>"""


class CubeSpawner:
    """Gazebo Classic 正方体管理：spawn / remove。"""

    def __init__(self, node: Node, size: float = 0.06) -> None:
        self._node = node
        self._logger = node.get_logger()
        self._size = float(size)
        self._cube_name = ""

    # ------------------------------------------------------------------
    def spawn(self, x: float, y: float, z: float) -> bool:
        """生成正方体（先删旧的）。"""
        self.remove()

        self._cube_name = f"green_cube_{int(time.time() * 1000)}"
        mass = 0.10
        inertia = mass * self._size * self._size / 6.0
        sdf = CUBE_SDF.format(
            name=self._cube_name, size=self._size, mass=mass, inertia=inertia,
            # collision_size=self._size * 0.9,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sdf", delete=False, prefix="cube_"
        ) as f:
            f.write(sdf)
            sdf_path = f.name

        try:
            subprocess.run(
                ["ros2", "run", "gazebo_ros", "spawn_entity.py",
                 "-entity", self._cube_name,
                 "-file", sdf_path,
                 "-x", str(x), "-y", str(y), "-z", str(z),
                 "-reference_frame", "world"],
                check=True, capture_output=True, text=True, timeout=10,
            )
        except subprocess.CalledProcessError as e:
            self._logger.error(f"立方体生成失败: {e.stderr}")
            os.unlink(sdf_path)
            return False

        os.unlink(sdf_path)
        self._logger.info(f"{self._cube_name} 生成于 ({x:.3f}, {y:.3f}, {z:.3f})")
        return True

    def remove(self) -> None:
        """删除正方体。"""
        if not self._cube_name:
            return
        try:
            subprocess.run(
                ["ros2", "service", "call", "/gazebo/delete_entity",
                 "gazebo_msgs/srv/DeleteEntity",
                 f"{{name: {self._cube_name}}}"],
                check=False, capture_output=True, text=True, timeout=2,
            )
        except subprocess.TimeoutExpired:
            self._logger.warn(f"删除 {self._cube_name} 超时")
        except Exception:
            pass
