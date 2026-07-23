"""将 reBot 香蕉任务注册到 Gymnasium。

注册后，训练/评估脚本只需使用固定 TASK id，Isaac Lab 会从 entry point 找到
对应环境配置和 PPO 配置。任务名称属于外部接口，修改时需同步所有调用脚本。
"""

import gymnasium as gym

from . import agents


gym.register(
    # 64×64 RGB-D 正式训练任务，使用项目内 CNN Actor-Critic。
    id="Isaac-Rebot-Banana-Lift-RGBD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.banana_lift_vision_env_cfg:RebotBananaLiftVisionTrainEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_rgbd_ppo_cfg:RebotBananaRgbdPPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)
