"""端到端 RGB-D 策略使用的 RSL-RL PPO 配置。"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RgbdActorCriticCfg:
    """传给项目内 ``RgbdActorCritic`` 的最小网络配置。"""

    class_name: str = "RgbdActorCritic"
    image_height: int = 64
    image_width: int = 64
    image_channels: int = 4
    cnn_output_dim: int = 128
    hidden_dims: list[int] = [256, 128]
    init_noise_std: float = 0.6


@configclass
class RebotBananaRgbdPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RGB-D 主线 PPO 参数。"""

    # 图像 rollout 占用显存较大，减少每轮步数；16 环境时每轮采集 256 个样本。
    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 50
    # 与旧“随机目标跟踪”任务分目录，防止误恢复语义不同的 checkpoint。
    experiment_name = "rebot_banana_grasp_return_rgbd"
    # 图像已单独归一化，不能再为全部像素维护巨大的运行均值/方差。
    empirical_normalization = False
    policy: RgbdActorCriticCfg = RgbdActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
