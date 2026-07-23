"""导出香蕉任务使用的 RSL-RL PPO 配置。"""

from .rsl_rl_rgbd_ppo_cfg import RebotBananaRgbdPPORunnerCfg

# Gym 注册字符串通过该包定位配置类，名称应保持稳定。
__all__ = ["RebotBananaRgbdPPORunnerCfg"]
