#!/usr/bin/env python3
"""使用项目内轻量 CNN 训练端到端 RGB-D PPO 策略。"""

import os
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


# 正式训练默认 headless；需要观察并行环境时设置 REBOT_RGBD_GUI=1，仍复用
# 同一个训练入口和 PPO 配置，不额外维护 GUI 训练脚本。
SHOW_GUI = os.environ.get("REBOT_RGBD_GUI", "0").lower() in {"1", "true", "yes"}
# enable_cameras=True 会加载 RTX/Replicator 相机扩展；缺少该项相机无法初始化。
app = AppLauncher(headless=not SHOW_GUI, enable_cameras=True).app

import gymnasium as gym  # noqa: E402
import rsl_rl.runners.on_policy_runner as runner_module  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import rebot_isaaclab.tasks  # noqa: F401,E402 - 启动 Kit 后注册 Gym 任务
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rebot_isaaclab.rgbd_env_wrapper import StableRslRlVecEnvWrapper, initial_reset, project_env_step  # noqa: E402
from rebot_isaaclab.rgbd_actor_critic import RgbdActorCritic  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK = "Isaac-Rebot-Banana-Lift-RGBD-v0"
NUM_ENVS = int(os.environ.get("REBOT_RGBD_NUM_ENVS", "8"))
MAX_ITERATIONS = int(os.environ.get("REBOT_RGBD_ITERATIONS", "200"))
RESUME = os.environ.get("REBOT_RGBD_RESUME", "0").lower() in {"1", "true", "yes"}
CHECKPOINT = os.environ.get("REBOT_RGBD_CHECKPOINT")
BC_CHECKPOINT = PROJECT_ROOT / "exported" / "rgbd_bc_policy.pt"
EXPECTED_BC_TASK = "grasp_return_ready"
WARMUP_STEPS = 3
DEBUG_RESET = os.environ.get("REBOT_RGBD_DEBUG_RESET", "0").lower() in {"1", "true", "yes"}
DEBUG_WARMUP = os.environ.get("REBOT_RGBD_DEBUG_WARMUP", "0").lower() in {"1", "true", "yes"}
DEBUG_STEP_LIMIT = int(os.environ.get("REBOT_RGBD_DEBUG_STEPS", "0"))


def latest_checkpoint(log_root: Path) -> Path:
    """选择最新 RGB-D checkpoint，用于接着上次训练继续优化。"""

    checkpoints = sorted(log_root.glob("*/model_*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"没有找到可恢复的 RGB-D checkpoint: {log_root}")
    return checkpoints[-1]


def main() -> None:
    """创建 RGB-D 环境，注册自定义网络并执行 PPO 训练。"""
    print("[RGBD Train] parse env/agent config", flush=True)
    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=NUM_ENVS)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    env_cfg.seed = agent_cfg.seed
    agent_cfg.max_iterations = MAX_ITERATIONS

    log_dir = PROJECT_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"[RGBD Train] 日志与 checkpoint: {log_dir}", flush=True)
    print(
        f"[RGBD Train] num_envs={NUM_ENVS}, iterations={MAX_ITERATIONS}, gui={SHOW_GUI}",
        flush=True,
    )

    # RSL-RL 2.3.3 用 eval(class_name) 在 runner 模块内寻找策略类。把项目类注册到
    # 该模块命名空间即可复用官方 Runner，无需修改 site-packages 或复制 Runner。
    runner_module.RgbdActorCritic = RgbdActorCritic

    print("[RGBD Train] creating Isaac Lab gym env ...", flush=True)
    gym_env = gym.make(TASK, cfg=env_cfg)
    print("[RGBD Train] gym env ready; initial reset ...", flush=True)
    reset_start = time.perf_counter()
    initial_reset(gym_env.unwrapped, verbose=DEBUG_RESET)
    print(f"[RGBD Train] initial reset done: {time.perf_counter() - reset_start:.2f}s", flush=True)
    print("[RGBD Train] wrapping for RSL-RL ...", flush=True)
    env = StableRslRlVecEnvWrapper(
        gym_env,
        clip_actions=agent_cfg.clip_actions,
        debug_step_limit=DEBUG_STEP_LIMIT,
    )
    print("[RGBD Train] RSL-RL wrapper ready", flush=True)
    try:
        # TiledCamera 的首帧渲染最容易让人误以为训练卡死。
        # 正式创建 runner 前先走几步零动作：如果这里很慢，问题就是 RGB-D 渲染/驱动/并行数；
        # 如果这里正常但 runner.learn 后很慢，问题才在 PPO 采样或优化阶段。
        print(f"[RGBD Train] warmup {WARMUP_STEPS} zero-action steps ...", flush=True)
        actions = torch.zeros((env.num_envs, env.num_actions), device=agent_cfg.device)
        warmup_start = time.perf_counter()
        for step_index in range(WARMUP_STEPS):
            step_start = time.perf_counter()
            project_env_step(env.unwrapped, actions, verbose=DEBUG_WARMUP, label=f"warmup_{step_index + 1}")
            print(
                f"[RGBD Train] warmup step {step_index + 1}/{WARMUP_STEPS}: "
                f"{time.perf_counter() - step_start:.2f}s",
                flush=True,
            )
        print(f"[RGBD Train] warmup total: {time.perf_counter() - warmup_start:.2f}s", flush=True)

        print("[RGBD Train] creating PPO runner ...", flush=True)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        print("[RGBD Train] PPO runner ready", flush=True)
        if RESUME or CHECKPOINT:
            checkpoint = Path(CHECKPOINT) if CHECKPOINT else latest_checkpoint(log_dir.parent)
            runner.load(str(checkpoint))
            print(f"[RGBD Train] resume checkpoint: {checkpoint}", flush=True)
            print(f"[RGBD Train] start iteration: {runner.current_learning_iteration}", flush=True)
        elif BC_CHECKPOINT.exists():
            # 行为克隆只提供 actor/critic 网络初始权重，不包含 PPO optimizer 状态；
            # 因此这里直接加载 policy.state_dict，而不是调用 runner.load()。
            bc_state = torch.load(BC_CHECKPOINT, map_location=agent_cfg.device, weights_only=False)
            bc_task = bc_state.get("infos", {}).get("task")
            if bc_task == EXPECTED_BC_TASK:
                runner.alg.policy.load_state_dict(bc_state["model_state_dict"])
                print(f"[RGBD Train] BC 初始化: {BC_CHECKPOINT}", flush=True)
            else:
                print(
                    f"[RGBD Train] 跳过旧 BC：task={bc_task!r}，需要 {EXPECTED_BC_TASK!r}；"
                    "请重新运行 collect-teacher 和 train-bc",
                    flush=True,
                )
        else:
            print(f"[RGBD Train] 未找到 BC 初始化权重，从随机策略开始: {BC_CHECKPOINT}", flush=True)
        print("[RGBD Train] start PPO learning; RSL-RL will print after each completed iteration", flush=True)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        # Isaac Sim 4.5 的 TiledCamera 在退出时会通过 __del__ 再 detach 一次
        # Replicator annotator；如果 SimulationManager callback 已经失效，进程会在
        # env.close() 中 abort。训练已经结束后先主动 detach 并清空 annotator，
        # 让后续析构只释放 PhysX/Kit 资源，不再碰已经失效的相机回调。
        camera = env.unwrapped.scene.sensors.get("wrist_camera")
        if camera is not None:
            for annotator in camera._annotators.values():
                annotator.detach(camera.render_product_paths)
            camera._annotators.clear()
        # 先释放 TiledCamera/PhysX 环境，再由最外层 finally 关闭 Kit。
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
