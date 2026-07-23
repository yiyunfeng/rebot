#!/usr/bin/env python3
"""评估最新端到端 RGB-D PPO checkpoint 的抓取返回成功率。"""

import json
import os
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


# 评估仍需要读取腕部相机，因此必须启用 RTX 相机扩展。
app = AppLauncher(headless=True, enable_cameras=True).app

import gymnasium as gym  # noqa: E402
import rsl_rl.runners.on_policy_runner as runner_module  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import rebot_isaaclab.tasks  # noqa: F401,E402 - 注册 RGB-D Gym 任务
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rebot_isaaclab.metrics import wilson_interval  # noqa: E402
from rebot_isaaclab.rgbd_env_wrapper import StableRslRlVecEnvWrapper, initial_reset  # noqa: E402
from rebot_isaaclab.rgbd_actor_critic import RgbdActorCritic  # noqa: E402
from rebot_isaaclab.tasks.banana_lift.banana_lift_env_cfg import (  # noqa: E402
    HOME_JOINT_TOLERANCE,
    MAX_OBJECT_EE_DISTANCE,
    SUCCESS_HEIGHT,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK = "Isaac-Rebot-Banana-Lift-RGBD-v0"
NUM_ENVS = int(os.environ.get("REBOT_RGBD_EVAL_NUM_ENVS", "4"))
TARGET_EPISODES = int(os.environ.get("REBOT_RGBD_EVAL_EPISODES", "32"))
PROGRESS_EVERY_STEPS = int(os.environ.get("REBOT_RGBD_EVAL_PROGRESS_STEPS", "50"))


def main() -> None:
    """加载最新视觉模型，累计 episode 成功标志并保存 JSON。"""
    log_root = PROJECT_ROOT / "logs" / "rsl_rl" / "rebot_banana_grasp_return_rgbd"
    checkpoint = Path(get_checkpoint_path(str(log_root), run_dir=".*", checkpoint="model_.*.pt"))

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=NUM_ENVS)
    env_cfg.seed = 42
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = 42

    # 与训练入口相同：只向 RSL-RL runner 注册项目策略类，不修改安装目录。
    runner_module.RgbdActorCritic = RgbdActorCritic

    print(f"[RGBD Evaluation] checkpoint: {checkpoint}", flush=True)
    print(f"[RGBD Evaluation] envs={NUM_ENVS}, target_episodes={TARGET_EPISODES}", flush=True)
    print("[RGBD Evaluation] creating env ...", flush=True)
    gym_env = gym.make(TASK, cfg=env_cfg)
    print("[RGBD Evaluation] initial reset ...", flush=True)
    initial_reset(gym_env.unwrapped)
    print("[RGBD Evaluation] initial reset done", flush=True)
    env = StableRslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs, _ = env.get_observations()
        print("[RGBD Evaluation] policy loaded; evaluating ...", flush=True)

        episode_success = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.unwrapped.device)
        successes = 0
        completed = 0
        steps = 0

        while completed < TARGET_EPISODES:
            with torch.inference_mode():
                obs, _, dones, extras = env.step(policy(obs))
                # wrapper 在自动 reset 前复制 success，避免读取到新 episode 初始状态。
                success_now = extras["task_success"]
                episode_success |= success_now
            steps += 1

            if steps % PROGRESS_EVERY_STEPS == 0:
                print(
                    f"[RGBD Evaluation] steps={steps}, completed={completed}/{TARGET_EPISODES}, "
                    f"successes={successes}",
                    flush=True,
                )

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() == 0:
                continue
            done_ids = done_ids[: TARGET_EPISODES - completed]
            successes += int(episode_success[done_ids].sum().item())
            completed += int(done_ids.numel())
            episode_success[done_ids] = False
            print(
                f"[RGBD Evaluation] completed={completed}/{TARGET_EPISODES}, successes={successes}",
                flush=True,
            )

        ci_low, ci_high = wilson_interval(successes, completed)
        result = {
            "task": TASK,
            "checkpoint": str(checkpoint),
            "seed": 42,
            "episodes": completed,
            "successes": successes,
            "success_rate": successes / completed,
            "success_rate_95ci": [ci_low, ci_high],
            "success_height_m": SUCCESS_HEIGHT,
            "home_joint_tolerance_rad": HOME_JOINT_TOLERANCE,
            "maximum_object_ee_distance_m": MAX_OBJECT_EE_DISTANCE,
        }
        output_dir = PROJECT_ROOT / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_rgbd_{datetime.now():%Y%m%d_%H%M%S}.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[RGBD Evaluation] 保存: {output_path}")
    finally:
        # 与训练入口一致：先主动 detach Replicator annotator，避免 Isaac Sim 4.5
        # 在 TiledCamera.__del__ 阶段访问已经失效的 weakref callback。
        camera = env.unwrapped.scene.sensors.get("wrist_camera")
        if camera is not None:
            for annotator in camera._annotators.values():
                annotator.detach(camera.render_product_paths)
            camera._annotators.clear()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
