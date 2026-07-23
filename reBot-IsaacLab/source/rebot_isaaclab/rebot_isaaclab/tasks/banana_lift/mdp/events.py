"""腕部 RGB-D 相机的 sim2real 域随机化事件。

这些事件在环境 reset 时修改 USD/传感器参数，不参与策略前向计算，也不会改变
动作空间。训练时让策略反复看到轻微不同的标定和光照，降低对单一仿真外观的依赖。
"""

from __future__ import annotations

import math

import torch
from pxr import Gf

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils


class randomize_wrist_camera(ManagerTermBase):
    """围绕标称标定随机相机外参和焦距。

    继承 ManagerTermBase 是因为需要在初始化时保存“未扰动的标称值”。如果每次
    reset 都读取当前相机姿态再叠加噪声，误差会逐回合累积，最终偏离机械臂。
    """

    def __init__(self, cfg, env):
        """解析相机实体并保存不会随 reset 漂移的标称外参和内参。"""

        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.camera = env.scene.sensors[sensor_cfg.name]
        # 这里不能访问 camera.data：该 property 会请求尚未生成的首帧 RTX buffer。
        # _data 中的 pose/K 已由 TiledCamera._create_buffers 初始化，可以安全复制。
        # clone() 很重要，否则后续 set_* 可能连“默认值”引用也一起修改。
        self.default_pos_w = self.camera._data.pos_w.clone()
        self.default_quat_w_ros = self.camera._data.quat_w_ros.clone()
        self.default_intrinsics = self.camera._data.intrinsic_matrices.clone()

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        sensor_cfg: SceneEntityCfg,
        position_error_m: float,
        rotation_error_deg: float,
        focal_scale_range: tuple[float, float],
    ) -> None:
        """只对 ``env_ids`` 对应相机实例重新采样位姿误差和焦距比例。"""

        # camera 已在 __init__ 缓存；保留这两个形参只是为了符合 EventManager 调用协议。
        del env, sensor_cfg
        if env_ids is None:
            # env_ids=None 表示此次事件应用到该 tiled camera 的全部实例。
            env_ids = torch.arange(self.camera.num_instances, device=self.camera.device)

        count = len(env_ids)
        # 平移误差分别沿世界 XYZ 均匀采样，范围单位为米。
        pos_noise = torch.empty(count, 3, device=self.camera.device).uniform_(
            -position_error_m, position_error_m
        )
        # 配置对用户使用角度更直观，四元数计算前统一转换为弧度。
        angle_limit = math.radians(rotation_error_deg)
        angle_noise = torch.empty(count, 3, device=self.camera.device).uniform_(-angle_limit, angle_limit)
        rotation_noise = math_utils.quat_from_euler_xyz(
            angle_noise[:, 0], angle_noise[:, 1], angle_noise[:, 2]
        )
        # 右乘噪声表示在标称相机姿态上叠加小旋转；四元数采用 Isaac Lab 的
        # (w, x, y, z) 存储，并在 set_world_poses 中明确指定 ROS 相机约定。
        orientations = math_utils.quat_mul(self.default_quat_w_ros[env_ids], rotation_noise)
        self.camera.set_world_poses(
            positions=self.default_pos_w[env_ids] + pos_noise,
            orientations=orientations,
            env_ids=env_ids,
            convention="ros",
        )

        # 只缩放 K 矩阵中的 fx、fy，主点 cx、cy 保持不变。这近似模拟焦距/装配
        # 误差，同时不引入与当前相机模型不一致的图像裁剪偏移。
        intrinsics = self.default_intrinsics[env_ids].clone()
        focal_scale = torch.empty(count, device=self.camera.device).uniform_(*focal_scale_range)
        intrinsics[:, 0, 0] *= focal_scale
        intrinsics[:, 1, 1] *= focal_scale
        self.camera.set_intrinsic_matrices(intrinsics, env_ids=env_ids)


def randomize_dome_light(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    intensity_range: tuple[float, float],
    color_range: tuple[float, float],
) -> None:
    """为当前 reset 批次随机一次共享 DomeLight 的强度和 RGB 色温。"""
    light = env.scene[asset_cfg.name]
    # DomeLight 是全局 prim，不存在逐环境独立属性；因此 env_ids 在这里不参与索引。
    del env_ids
    prim = light.prims[0]
    # 这里使用 CPU 标量即可，因为属性最终通过 USD API 写入，而不是进入 GPU 仿真张量。
    intensity = torch.empty(1).uniform_(*intensity_range).item()
    color = torch.empty(3).uniform_(*color_range).tolist()
    prim.GetAttribute("inputs:intensity").Set(intensity)
    prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*color))
