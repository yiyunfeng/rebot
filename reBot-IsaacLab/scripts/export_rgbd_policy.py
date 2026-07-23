#!/usr/bin/env python3
"""把最新 RGB-D PPO checkpoint 导出为真机端更容易加载的 TorchScript。

训练 checkpoint 里包含 optimizer、iteration 等 PPO 继续训练信息；真机部署只
需要确定性 Actor，即 ``observation -> action``。本脚本重建同结构网络，加载
``model_state_dict``，再用 ``torch.jit.trace`` 导出一个独立 ``.pt`` 文件。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import torch

from rebot_isaaclab.rgbd_actor_critic import RgbdActorCritic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl" / "rebot_banana_grasp_return_rgbd"
DEFAULT_EXPORT_PATH = PROJECT_ROOT / "exported" / "rgbd_policy_latest.pt"
TASK_NAME = "grasp_return_ready"

# 观测排列由 banana_lift_vision_env_cfg.py 固定：
# 7 关节位置 + 7 关节速度 + 7 上一动作 + 64*64*4 RGB-D。
PROPRIO_SIZE = 21
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 4
NUM_ACTIONS = 7
OBS_DIM = PROPRIO_SIZE + IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS


class DeterministicActor(torch.nn.Module):
    """只导出确定性 Actor，真机端不需要 Critic、采样分布或 PPO 接口。"""

    def __init__(self, policy: RgbdActorCritic) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """输入训练同布局观测，输出 7 维动作均值。"""

        # 导出时固定观测长度，真实运行前由调用方保证输入布局正确。
        proprio = self.policy.proprio_norm(observations[:, : self.policy.proprio_size])
        image = observations[:, self.policy.proprio_size :].reshape(
            -1,
            self.policy.image_height,
            self.policy.image_width,
            self.policy.image_channels,
        )
        image = image.permute(0, 3, 1, 2).contiguous()
        features = torch.cat((proprio, self.policy.cnn(image)), dim=-1)
        return self.policy.actor(features)


def latest_checkpoint() -> Path:
    """按目录修改时间选择最新 ``model_*.pt``，避免手动复制 checkpoint 路径。"""

    checkpoints = sorted(LOG_ROOT.glob("*/model_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"没有找到 RGB-D checkpoint: {LOG_ROOT}")
    return checkpoints[-1]


def main() -> None:
    """加载 checkpoint，导出 TorchScript，并写一份同名 metadata JSON。"""

    checkpoint_env = os.environ.get("REBOT_RGBD_CHECKPOINT")
    checkpoint = Path(checkpoint_env) if checkpoint_env else latest_checkpoint()
    export_path = Path(os.environ.get("REBOT_RGBD_EXPORT_PATH", DEFAULT_EXPORT_PATH))
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # checkpoint 来自本机训练目录；weights_only=True 避免反序列化无关 Python 对象。
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = RgbdActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=NUM_ACTIONS,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        image_channels=IMAGE_CHANNELS,
    )
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    actor = DeterministicActor(model).eval()

    # 真机端输入必须与训练观测一致：前 21 维本体状态，后面是展平 RGB-D。
    example_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    traced = torch.jit.trace(actor, example_obs)
    traced.save(str(export_path))

    metadata = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "task": TASK_NAME,
        "checkpoint": str(checkpoint),
        "torchscript": str(export_path),
        "observation_dim": OBS_DIM,
        "proprio_size": PROPRIO_SIZE,
        "image_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
        "num_actions": NUM_ACTIONS,
        "observation_layout": "joint_pos7, joint_vel7, last_action7, flattened_rgbd64x64x4",
    }
    metadata_path = export_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[RGBD Export] checkpoint: {checkpoint}")
    print(f"[RGBD Export] torchscript: {export_path}")
    print(f"[RGBD Export] metadata: {metadata_path}")


if __name__ == "__main__":
    main()
