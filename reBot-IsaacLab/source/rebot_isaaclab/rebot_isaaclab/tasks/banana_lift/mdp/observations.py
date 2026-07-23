"""RGB-D PPO 使用的项目特有观测函数。"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCamera


def flattened_rgbd(
    env,
    sensor_cfg: SceneEntityCfg,
    depth_limit_m: float = 1.5,
) -> torch.Tensor:
    """返回形状为 ``(num_envs, H*W*4)`` 的归一化 RGB-D。

    RGB 被缩放到 0~1 后减去每张图的通道均值，减少仿真整体亮度变化带来的
    偏移。深度中的 NaN/Inf 会被替换并裁剪到 ``depth_limit_m``，再归一化到
    0~1。四通道排列为 RGBD，展平前保持 Isaac Lab 的 NHWC 顺序。
    """
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]

    # clone/float 保证后续归一化不会原地修改相机 annotator 的共享输出缓存。
    rgb = camera.data.output["rgb"][..., :3].float().clone() / 255.0
    rgb -= rgb.mean(dim=(1, 2), keepdim=True)

    depth = camera.data.output["distance_to_image_plane"].float().clone()
    depth = torch.nan_to_num(depth, nan=depth_limit_m, posinf=depth_limit_m, neginf=0.0)
    depth = depth.clamp_(0.0, depth_limit_m).div_(depth_limit_m)

    rgbd = torch.cat((rgb, depth), dim=-1)
    return rgbd.reshape(env.num_envs, -1)
