"""香蕉抓取并返回 ready 姿态所需的奖励和成功判据。"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def return_home_with_object(
    env,
    minimum_height: float,
    home_std: float,
    maximum_object_ee_distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """物体被夹持并抬起后，按六轴关节接近默认 ready 姿态的程度给奖励。"""

    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    joint_error = torch.abs(
        robot.data.joint_pos[:, robot_cfg.joint_ids]
        - robot.data.default_joint_pos[:, robot_cfg.joint_ids]
    ).mean(dim=1)
    home_score = 1.0 - torch.tanh(joint_error / home_std)
    object_ee_distance = torch.linalg.vector_norm(
        object_asset.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1
    )
    has_lifted_object = (object_asset.data.root_pos_w[:, 2] > minimum_height) & (
        object_ee_distance < maximum_object_ee_distance
    )
    return has_lifted_object.float() * home_score


def grasp_return_success(
    env,
    minimum_height: float,
    home_joint_tolerance: float,
    maximum_object_ee_distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """物体仍在夹爪附近且六轴回到 ready 姿态时返回 1。

    只检查机械臂六轴，不要求夹爪回到初始张开位置：返回途中必须保持夹紧。
    物体与末端距离约束可排除把香蕉抛起后空手回到 ready 的假成功。
    """

    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    joint_error = torch.abs(
        robot.data.joint_pos[:, robot_cfg.joint_ids]
        - robot.data.default_joint_pos[:, robot_cfg.joint_ids]
    ).amax(dim=1)
    object_ee_distance = torch.linalg.vector_norm(
        object_asset.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1
    )
    return (
        (object_asset.data.root_pos_w[:, 2] > minimum_height)
        & (joint_error < home_joint_tolerance)
        & (object_ee_distance < maximum_object_ee_distance)
    ).float()
