"""
conversions 模块 — 位姿与坐标转换工具
=====================================

本模块提供 ROS2 Pose 消息 与 欧拉角/正运动学 之间的转换函数。

**两种转换方向**：
  1. Pose → (x, y, z, roll, pitch, yaw)：从 ROS 位姿消息提取笛卡尔坐标+RPY欧拉角
  2. FK 输出 → Pose：将正运动学计算的位置矩阵+旋转矩阵封装为 ROS Pose 消息
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Pose                          # ROS 标准位姿消息（position + orientation）
from tf_transformations import euler_from_quaternion, quaternion_from_matrix  # TF 四元数/欧拉角转换


def pose_to_xyz_rpy(pose: Pose) -> tuple[float, float, float, float, float, float]:
    """
    将 ROS Pose 消息解包为笛卡尔坐标 + RPY 欧拉角。

    用途：从 MoveToPose 等 action 目标中提取数值，供逆运动学(IK)或轨迹规划使用。

    Args:
        pose: ROS geometry_msgs/Pose 消息，包含 position(xyz) 和 orientation(四元数)

    Returns:
        一个 6 元组 (x, y, z, roll, pitch, yaw)，单位均为米/弧度:
          - x, y, z          → 笛卡尔空间位置
          - roll, pitch, yaw → RPY 欧拉角（固定轴 X-Y-Z 旋转）
    """
    # 从 Pose.orientation 字段提取四元数 [x, y, z, w]
    quat = [
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]
    # 四元数 → RPY 欧拉角转换（tf_transformations 库函数）
    roll, pitch, yaw = euler_from_quaternion(quat)
    return (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(roll),
        float(pitch),
        float(yaw),
    )


def fk_to_pose(position: np.ndarray, rotation: np.ndarray) -> Pose:
    """
    将正运动学(FK)输出的位置向量+旋转矩阵封装为 ROS Pose 消息。

    用途：在 move_to_pose 等操作完成后，将 FK 结果发布回客户端。

    处理流程：
      1. 用 3×3 旋转矩阵构造 4×4 齐次变换矩阵
      2. 通过 tf_transformations 从矩阵提取四元数
      3. 将位置和四元数填入 Pose 消息

    Args:
        position: 3 元素 numpy 数组 [x, y, z]，单位米
        rotation: 3×3 numpy 旋转矩阵

    Returns:
        填充好 position 和 orientation 的 geometry_msgs/Pose 消息
    """
    # 构造 4×4 齐次变换矩阵：左上 3×3 为旋转，其余为单位矩阵默认值
    mat = np.eye(4)
    mat[:3, :3] = rotation
    # 从齐次变换矩阵提取四元数
    quat = quaternion_from_matrix(mat)

    # 组装 Pose 消息
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose
