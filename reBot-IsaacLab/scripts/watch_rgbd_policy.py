#!/usr/bin/env python3
"""打开 Isaac Sim GUI，播放最新 RGB-D 策略抓香蕉的过程。

这个脚本只做可视化 rollout，不训练、不保存模型、不连接真机。它会加载最新
RGB-D checkpoint，用确定性动作在单个环境里跑若干个 episode，方便直接观察
机械臂到底是在接近香蕉、乱撞，还是已经学到闭爪、抬升并返回 ready 姿态。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher


# headless=False 会打开 Isaac Sim 窗口；enable_cameras=True 才能读取腕部 RGB-D。
app = AppLauncher(headless=False, enable_cameras=True).app

import gymnasium as gym  # noqa: E402
import rsl_rl.runners.on_policy_runner as runner_module  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import rebot_isaaclab.tasks  # noqa: F401,E402 - 注册 Gym 任务
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rebot_isaaclab.rgbd_env_wrapper import StableRslRlVecEnvWrapper, initial_reset  # noqa: E402
from rebot_isaaclab.rgbd_actor_critic import RgbdActorCritic  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK = "Isaac-Rebot-Banana-Lift-RGBD-v0"
EPISODES = int(os.environ.get("REBOT_RGBD_WATCH_EPISODES", "3"))
SLEEP_S = float(os.environ.get("REBOT_RGBD_WATCH_SLEEP", "0.02"))
HOLD_READY_STEPS = int(os.environ.get("REBOT_RGBD_WATCH_HOLD_READY_STEPS", "80"))
ZERO_ACTION_ONLY = os.environ.get("REBOT_RGBD_WATCH_ZERO_ACTION", "0").lower() in {"1", "true", "yes"}


def detach_camera(env) -> None:
    """退出前主动释放 Replicator annotator，避免 Isaac Sim 4.5 相机析构报错。"""

    camera = env.unwrapped.scene.sensors.get("wrist_camera")
    if camera is None:
        return
    for annotator in camera._annotators.values():
        annotator.detach(camera.render_product_paths)
    camera._annotators.clear()


def main() -> None:
    """加载最新策略，在 GUI 中运行若干个 episode。"""

    log_root = PROJECT_ROOT / "logs" / "rsl_rl" / "rebot_banana_grasp_return_rgbd"
    checkpoint = Path(os.environ.get("REBOT_RGBD_CHECKPOINT") or get_checkpoint_path(
        str(log_root), run_dir=".*", checkpoint="model_.*.pt"
    ))

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env_cfg.seed = 42
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = 42
    runner_module.RgbdActorCritic = RgbdActorCritic

    print("[RGBD Watch] creating env ...", flush=True)
    gym_env = gym.make(TASK, cfg=env_cfg)
    print("[RGBD Watch] initial reset ...", flush=True)
    initial_reset(gym_env.unwrapped)
    env = StableRslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs, _ = env.get_observations()

        print(f"[RGBD Watch] checkpoint: {checkpoint}")
        print(f"[RGBD Watch] episodes={EPISODES}")

        # 先用零动作保持 ready pose 一小段时间。这样 GUI 里能直观看出：
        # - 如果这里就下垂/乱摆，问题在机械臂 drive 或 reset 后的控制目标；
        # - 如果这里稳定，后面才乱摆，问题就是当前策略输出还没学好。
        zero_actions = torch.zeros((env.num_envs, env.num_actions), device=env.unwrapped.device)
        if HOLD_READY_STEPS > 0:
            print(f"[RGBD Watch] hold ready with zero action: {HOLD_READY_STEPS} steps")
        for _ in range(HOLD_READY_STEPS):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(zero_actions)
            app.update()
            if SLEEP_S > 0:
                time.sleep(SLEEP_S)

        if ZERO_ACTION_ONLY:
            print("[RGBD Watch] zero-action only mode; close GUI or Ctrl+C to stop")
            while app.is_running():
                with torch.inference_mode():
                    obs, _, dones, _ = env.step(zero_actions)
                app.update()
                if SLEEP_S > 0:
                    time.sleep(SLEEP_S)
            return

        print("[RGBD Watch] start policy rollout")
        completed = 0
        while completed < EPISODES and app.is_running():
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            app.update()
            if SLEEP_S > 0:
                time.sleep(SLEEP_S)
            if dones.any():
                completed += int(dones.sum().item())
                print(f"[RGBD Watch] completed episodes: {completed}/{EPISODES}")
    finally:
        detach_camera(env)
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
