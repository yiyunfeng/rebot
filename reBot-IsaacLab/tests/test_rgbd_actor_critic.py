"""不启动 Isaac Sim，检查 RGB-D 策略的张量形状和梯度。"""

import pytest
import torch

from rebot_isaaclab.rgbd_actor_critic import RgbdActorCritic


IMAGE_SIZE = 64 * 64 * 4
PROPRIO_SIZE = 21
ACTION_SIZE = 7


def test_rgbd_actor_critic_shapes_and_gradients():
    """Actor、Critic 输出形状正确，且损失能够反向传播到 CNN。"""
    policy = RgbdActorCritic(
        num_actor_obs=PROPRIO_SIZE + IMAGE_SIZE,
        num_critic_obs=PROPRIO_SIZE + IMAGE_SIZE,
        num_actions=ACTION_SIZE,
    )
    observations = torch.randn(2, PROPRIO_SIZE + IMAGE_SIZE)

    actions = policy.act(observations)
    values = policy.evaluate(observations)
    loss = actions.mean() + values.mean()
    loss.backward()

    assert actions.shape == (2, ACTION_SIZE)
    assert values.shape == (2, 1)
    assert policy.cnn[0].weight.grad is not None


def test_rgbd_actor_critic_rejects_wrong_observation_size():
    """观测长度与配置分辨率不一致时给出明确错误，而不是在卷积层中崩溃。"""
    policy = RgbdActorCritic(
        num_actor_obs=PROPRIO_SIZE + IMAGE_SIZE,
        num_critic_obs=PROPRIO_SIZE + IMAGE_SIZE,
        num_actions=ACTION_SIZE,
    )

    with pytest.raises(ValueError, match="错误观测形状"):
        policy.act_inference(torch.randn(2, PROPRIO_SIZE + IMAGE_SIZE - 1))


def test_rgbd_actor_critic_requires_same_actor_and_critic_observation():
    """当前共享编码器不接受不同长度的非对称 Actor/Critic 观测。"""
    with pytest.raises(ValueError, match="相同观测"):
        RgbdActorCritic(
            num_actor_obs=PROPRIO_SIZE + IMAGE_SIZE,
            num_critic_obs=PROPRIO_SIZE + IMAGE_SIZE + 1,
            num_actions=ACTION_SIZE,
        )
