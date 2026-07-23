#!/usr/bin/env python3
"""用 teacher 数据对 RGB-D Actor 做行为克隆预训练。

BC 只负责给策略一个“会往香蕉走、会闭爪、会抬起并返回 ready”的初始动作模式，不替代
后续 PPO。PPO 仍然负责在真实接触、摩擦、随机光照和相机误差下继续优化成功率。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rebot_isaaclab.rgbd_actor_critic import RgbdActorCritic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEACHER_DATASET_PATH = PROJECT_ROOT / "data" / "rgbd_isaacsim_teacher_latest.pt"
OUTPUT_PATH = PROJECT_ROOT / "exported" / "rgbd_bc_policy.pt"

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 4
PROPRIO_SIZE = 21
OBS_DIM = PROPRIO_SIZE + IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS
ACTION_DIM = 7
EXPECTED_TASK = "grasp_return_ready"

EPOCHS = int(os.environ.get("REBOT_BC_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("REBOT_BC_BATCH_SIZE", "128"))
LEARNING_RATE = float(os.environ.get("REBOT_BC_LR", "3e-4"))


def main() -> None:
    """读取 teacher 数据，训练 Actor，并保存可供 PPO 初始化的权重。"""

    if not TEACHER_DATASET_PATH.exists():
        raise FileNotFoundError(f"没有 IsaacSim teacher 数据，请先运行 ./run_collect_teacher.sh: {TEACHER_DATASET_PATH}")
    dataset_path = TEACHER_DATASET_PATH

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if data.get("task") != EXPECTED_TASK:
        raise ValueError(
            f"teacher 任务不匹配: {data.get('task')!r} != {EXPECTED_TASK!r}；"
            "请重新运行 ./run.sh collect-teacher，旧数据不含返回 ready 阶段"
        )
    observations = data["observations"].float()
    actions = data["actions"].float().clamp(-1.0, 1.0)

    if observations.shape[1] != OBS_DIM:
        raise ValueError(f"观测维度不匹配: {observations.shape[1]} != {OBS_DIM}")
    if actions.shape[1] != ACTION_DIM:
        raise ValueError(f"动作维度不匹配: {actions.shape[1]} != {ACTION_DIM}")

    dataset = TensorDataset(observations, actions)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    policy = RgbdActorCritic(
        num_actor_obs=OBS_DIM,
        num_critic_obs=OBS_DIM,
        num_actions=ACTION_DIM,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        image_channels=IMAGE_CHANNELS,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print(f"[BC] dataset: {dataset_path}")
    print(f"[BC] teacher_type={data.get('teacher_type', 'isaacsim_traditional_grasp')}")
    print(f"[BC] samples={len(dataset)}, epochs={EPOCHS}, batch_size={BATCH_SIZE}, device={device}")

    epoch_losses: list[float] = []
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_samples = 0
        for obs_batch, action_batch in loader:
            obs_batch = obs_batch.to(device, non_blocking=True)
            action_batch = action_batch.to(device, non_blocking=True)

            pred = policy.act_inference(obs_batch)
            loss = loss_fn(pred, action_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * obs_batch.shape[0]
            total_samples += obs_batch.shape[0]

        epoch_mse = total_loss / total_samples
        epoch_losses.append(epoch_mse)
        print(f"[BC] epoch={epoch + 1}/{EPOCHS}, mse={epoch_mse:.6f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "infos": {
                "source_dataset": str(dataset_path),
                "teacher_type": data.get("teacher_type", "isaacsim_traditional_grasp"),
                "task": EXPECTED_TASK,
                "samples": len(dataset),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epoch_mse": epoch_losses,
                "final_mse": epoch_losses[-1] if epoch_losses else None,
                "action_mean": actions.mean(dim=0).tolist(),
                "action_std": actions.std(dim=0, unbiased=False).tolist(),
                "teacher_collected_plans": data.get("collected_plans"),
                "teacher_samples_per_plan_mean": data.get("samples_per_plan_mean"),
                "teacher_trajectory_steps": data.get("trajectory_steps"),
                "note": "BC initialization for RGB-D PPO; not a final deployment policy.",
            },
        },
        OUTPUT_PATH,
    )
    print(f"[BC] 保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
