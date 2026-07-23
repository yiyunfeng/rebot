"""供 RSL-RL PPO 使用的轻量 RGB-D Actor-Critic 网络。

本机 RSL-RL 2.3.3 只接受形状为 ``(N, D)`` 的观测 Tensor，不能直接接收
``{"rgb": ..., "depth": ...}``。环境因此把 RGB-D 展平后放在观测末尾；本类
在网络内部恢复为 ``NCHW`` 图像并送入 CNN，CNN 参数与策略一起由 PPO 更新。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class RgbdActorCritic(nn.Module):
    """用共享 CNN 编码 RGB-D，再分别预测动作均值和状态价值。"""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        image_height: int = 64,
        image_width: int = 64,
        image_channels: int = 4,
        cnn_output_dim: int = 128,
        hidden_dims: tuple[int, int] | list[int] = (256, 128),
        init_noise_std: float = 0.6,
        **_: object,
    ) -> None:
        """构建网络并根据总观测长度自动推导本体状态长度。

        Args:
            num_actor_obs: Actor 接收的总观测长度。
            num_critic_obs: Critic 接收的总观测长度；当前必须与 Actor 相同。
            num_actions: 策略输出动作数量。
            image_height: 展平前图像高度。
            image_width: 展平前图像宽度。
            image_channels: RGB 三通道加深度一通道，共 4 通道。
            cnn_output_dim: CNN 输出的视觉特征长度。
            hidden_dims: Actor/Critic MLP 的两层宽度。
            init_noise_std: PPO 高斯动作分布的初始标准差。
        """
        super().__init__()

        if num_actor_obs != num_critic_obs:
            raise ValueError("当前 RGB-D 策略要求 Actor 与 Critic 使用相同观测")
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims 必须正好包含两层")

        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.image_size = image_height * image_width * image_channels
        self.proprio_size = num_actor_obs - self.image_size
        if self.proprio_size <= 0:
            raise ValueError(
                f"观测长度 {num_actor_obs} 小于 RGB-D 长度 {self.image_size}，"
                "请核对相机分辨率和观测排列"
            )

        # 三层 stride=2 卷积把 64×64 图像降到 8×8，再用自适应池化固定输出大小。
        # 不依赖 torchvision，避免额外模型和预训练权重下载。
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, cnn_output_dim),
            nn.ELU(),
        )

        # 关节位置、速度和上一动作的量纲不同，LayerNorm 只处理本体状态；
        # RGB-D 已在环境观测函数中归一化，不再使用 RSL-RL 的全向量 normalizer。
        self.proprio_norm = nn.LayerNorm(self.proprio_size)
        fused_size = self.proprio_size + cnn_output_dim
        self.actor = self._make_mlp(fused_size, hidden_dims, num_actions)
        self.critic = self._make_mlp(fused_size, hidden_dims, 1)

        # PPO 用独立可学习标准差构造连续动作高斯分布。
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _make_mlp(input_dim: int, hidden_dims: tuple[int, int] | list[int], output_dim: int) -> nn.Sequential:
        """创建 Actor/Critic 共用结构但参数独立的两层 MLP。"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ELU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def _features(self, observations: torch.Tensor) -> torch.Tensor:
        """拆分二维观测，将末尾 RGB-D 恢复成 NCHW 并融合视觉/本体特征。"""
        if observations.ndim != 2 or observations.shape[1] != self.proprio_size + self.image_size:
            raise ValueError(
                f"RGB-D 策略收到错误观测形状 {tuple(observations.shape)}，"
                f"期望 (N, {self.proprio_size + self.image_size})"
            )

        proprio = self.proprio_norm(observations[:, : self.proprio_size])
        image = observations[:, self.proprio_size :].reshape(
            -1, self.image_height, self.image_width, self.image_channels
        )
        # Isaac Lab 相机输出为 NHWC，PyTorch Conv2d 要求 NCHW。
        image = image.permute(0, 3, 1, 2).contiguous()
        return torch.cat((proprio, self.cnn(image)), dim=-1)

    def update_distribution(self, observations: torch.Tensor) -> None:
        """根据当前观测更新 PPO 采样使用的高斯动作分布。"""
        mean = self.actor(self._features(observations))
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, observations: torch.Tensor, **_: object) -> torch.Tensor:
        """按当前高斯策略采样动作，用于训练阶段探索。"""
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """直接返回动作均值，用于无随机探索的评估和部署。"""
        return self.actor(self._features(observations))

    def evaluate(self, observations: torch.Tensor, **_: object) -> torch.Tensor:
        """计算 Critic 对当前观测的状态价值。"""
        return self.critic(self._features(observations))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """返回每个样本所有动作维度对数概率之和。"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        """当前动作分布均值，供 PPO 计算 KL 散度。"""
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """当前动作分布标准差，供 PPO 记录和计算 KL 散度。"""
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """返回各动作维度熵之和，供 PPO 熵奖励使用。"""
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """无循环状态，保留空实现以满足 RSL-RL Policy 接口。"""
        del dones

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """默认前向等同确定性 Actor，便于导出和独立检查。"""
        return self.act_inference(observations)

    def load_state_dict(self, state_dict, strict: bool = True) -> bool:
        """加载 checkpoint，并按 RSL-RL 约定返回可继续训练标志。"""
        super().load_state_dict(state_dict, strict=strict)
        return True
