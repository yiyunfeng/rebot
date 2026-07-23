"""RGB-D 训练/评估共用的 Isaac Lab 环境 wrapper。

Isaac Lab 官方 ``RslRlVecEnvWrapper`` 会在构造函数里隐式调用一次
``env.reset()``。在本项目的多环境 RGB-D 任务中，这个隐藏 reset 曾经卡在
关节 reset、RTX 纹理等待或相机相关路径上。这里把 reset 和 step 展开成项目
内可控流程，训练、评估和 GUI 播放使用同一条稳定路径。
"""

from __future__ import annotations

import time

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def timed_call(name: str, func, verbose: bool = False):
    """执行一个步骤；verbose=True 时打印耗时，方便定位 Isaac Sim 卡点。"""

    if verbose:
        print(f"[RGBD Trace] {name} ...", flush=True)
    start = time.perf_counter()
    result = func()
    if verbose:
        print(f"[RGBD Trace] {name} done: {time.perf_counter() - start:.2f}s", flush=True)
    return result


def apply_reset_events(env, env_ids: torch.Tensor, verbose: bool = False) -> None:
    """逐个执行 reset event，避免官方 ``event_manager.apply`` 卡住时无日志。"""

    event_manager = env.event_manager
    if "reset" not in event_manager.available_modes:
        if verbose:
            print("[RGBD Reset] no reset events", flush=True)
        return

    global_step_count = env._sim_step_counter // env.cfg.decimation
    names = event_manager._mode_term_names["reset"]
    cfgs = event_manager._mode_term_cfgs["reset"]
    if verbose:
        print(f"[RGBD Reset] reset events: {names}", flush=True)

    for index, (name, term_cfg) in enumerate(zip(names, cfgs)):
        def run_event():
            min_step_count = term_cfg.min_step_count_between_reset
            if min_step_count == 0:
                event_manager._reset_term_last_triggered_step_id[index][env_ids] = global_step_count
                event_manager._reset_term_last_triggered_once[index][env_ids] = True
                term_cfg.func(env, env_ids, **(term_cfg.params or {}))
                return

            last_step = event_manager._reset_term_last_triggered_step_id[index][env_ids]
            triggered_once = event_manager._reset_term_last_triggered_once[index][env_ids]
            valid_trigger = (global_step_count - last_step) >= min_step_count
            valid_trigger |= (last_step == 0) & ~triggered_once
            valid_env_ids = env_ids[valid_trigger]
            if len(valid_env_ids) == 0:
                return

            event_manager._reset_term_last_triggered_once[index][valid_env_ids] = True
            event_manager._reset_term_last_triggered_step_id[index][valid_env_ids] = global_step_count
            term_cfg.func(env, valid_env_ids, **(term_cfg.params or {}))

        timed_call(f"event.reset.{name}", run_event, verbose)


def initial_reset(env, verbose: bool = False) -> None:
    """初始 reset：使用项目内展开流程，避开官方 reset 的隐藏阻塞点。"""

    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
    if verbose:
        print(f"[RGBD Reset] env_ids={env_ids.detach().cpu().tolist()}", flush=True)

    timed_call("recorder.pre_reset", lambda: env.recorder_manager.record_pre_reset(env_ids), verbose)
    timed_call("curriculum.compute", lambda: env.curriculum_manager.compute(env_ids=env_ids), verbose)
    timed_call("scene.reset", lambda: env.scene.reset(env_ids), verbose)
    apply_reset_events(env, env_ids, verbose)

    env.extras["log"] = dict()
    for name, manager in (
        ("observation_manager.reset", env.observation_manager),
        ("action_manager.reset", env.action_manager),
        ("reward_manager.reset", env.reward_manager),
        ("curriculum_manager.reset", env.curriculum_manager),
        ("command_manager.reset", env.command_manager),
        ("event_manager.reset", env.event_manager),
        ("termination_manager.reset", env.termination_manager),
        ("recorder_manager.reset", env.recorder_manager),
    ):
        env.extras["log"].update(timed_call(name, lambda manager=manager: manager.reset(env_ids), verbose))

    env.episode_length_buf[env_ids] = 0
    timed_call("scene.write_data_to_sim", env.scene.write_data_to_sim, verbose)
    timed_call("sim.forward", env.sim.forward, verbose)
    timed_call("recorder.post_reset", lambda: env.recorder_manager.record_post_reset(env_ids), verbose)
    env.obs_buf = timed_call(
        "observation_manager.compute",
        lambda: env.observation_manager.compute(update_history=True),
        verbose,
    )


def project_env_step(env, actions: torch.Tensor, verbose: bool = False, label: str = "step"):
    """展开版 ``env.step``，返回格式与 Isaac Lab ManagerBasedRLEnv.step 一致。"""

    if verbose:
        print(f"[RGBD Step] {label}", flush=True)
    timed_call("action_manager.process_action", lambda: env.action_manager.process_action(actions.to(env.device)), verbose)
    timed_call("recorder.pre_step", env.recorder_manager.record_pre_step, verbose)

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    if verbose:
        print(f"[RGBD Step] is_rendering={is_rendering}, decimation={env.cfg.decimation}", flush=True)

    for substep in range(env.cfg.decimation):
        env._sim_step_counter += 1
        prefix = f"physics[{substep + 1}/{env.cfg.decimation}]"
        timed_call(f"{prefix}.action_manager.apply_action", env.action_manager.apply_action, verbose)
        timed_call(f"{prefix}.scene.write_data_to_sim", env.scene.write_data_to_sim, verbose)
        timed_call(f"{prefix}.sim.step", lambda: env.sim.step(render=False), verbose)
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            timed_call(f"{prefix}.sim.render", env.sim.render, verbose)
        timed_call(f"{prefix}.scene.update", lambda: env.scene.update(dt=env.physics_dt), verbose)

    env.episode_length_buf += 1
    env.common_step_counter += 1
    env.reset_buf = timed_call("termination_manager.compute", env.termination_manager.compute, verbose)
    env.reset_terminated = env.termination_manager.terminated
    env.reset_time_outs = env.termination_manager.time_outs
    env.reward_buf = timed_call("reward_manager.compute", lambda: env.reward_manager.compute(dt=env.step_dt), verbose)
    # 成功状态必须在自动 reset 前复制出来；评估端若在 step() 返回后重新读取
    # scene，会看到新 episode 的初始状态，从而把真实成功误计为失败。
    if "success" in env.reward_manager.active_terms:
        success_cfg = env.reward_manager.get_term_cfg("success")
        env.extras["task_success"] = success_cfg.func(env, **success_cfg.params).bool().clone()

    if len(env.recorder_manager.active_terms) > 0:
        env.obs_buf = timed_call("observation_manager.compute.for_recorder", env.observation_manager.compute, verbose)
        timed_call("recorder.post_step", env.recorder_manager.record_post_step, verbose)

    reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if verbose:
        print(f"[RGBD Step] reset_env_ids={reset_env_ids.detach().cpu().tolist()}", flush=True)
    if len(reset_env_ids) > 0:
        timed_call("recorder.pre_reset.step", lambda: env.recorder_manager.record_pre_reset(reset_env_ids), verbose)
        timed_call("reset_idx.step", lambda: env._reset_idx(reset_env_ids), verbose)
        timed_call("scene.write_data_to_sim.after_step_reset", env.scene.write_data_to_sim, verbose)
        timed_call("sim.forward.after_step_reset", env.sim.forward, verbose)
        if env.sim.has_rtx_sensors() and env.cfg.rerender_on_reset:
            timed_call("sim.render.after_step_reset", env.sim.render, verbose)
        timed_call("recorder.post_reset.step", lambda: env.recorder_manager.record_post_reset(reset_env_ids), verbose)

    timed_call("command_manager.compute", lambda: env.command_manager.compute(dt=env.step_dt), verbose)
    if "interval" in env.event_manager.available_modes:
        timed_call("event_manager.apply.interval", lambda: env.event_manager.apply(mode="interval", dt=env.step_dt), verbose)
    env.obs_buf = timed_call(
        "observation_manager.compute.final",
        lambda: env.observation_manager.compute(update_history=True),
        verbose,
    )
    return env.obs_buf, env.reward_buf, env.reset_terminated, env.reset_time_outs, env.extras


class StableRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """不隐式 reset，并使用项目内 step 路径的 RSL-RL wrapper。"""

    def __init__(self, env, clip_actions: float | None = None, debug_step_limit: int = 0) -> None:
        self.env = env
        self.clip_actions = clip_actions
        self._debug_step_count = 0
        self._debug_step_limit = debug_step_limit

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])

        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0

        self._modify_action_space()

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self._debug_step_count += 1
        verbose = self._debug_step_count <= self._debug_step_limit
        obs_dict, rewards, terminated, truncated, extras = project_env_step(
            self.unwrapped,
            actions.to(self.device),
            verbose=verbose,
            label=f"step_{self._debug_step_count}",
        )
        dones = (terminated | truncated).to(dtype=torch.long)
        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return obs_dict["policy"], rewards, dones, extras
