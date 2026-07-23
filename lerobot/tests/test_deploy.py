from lerobot_robot_rebot.config import JOINT_NAMES
from lerobot_robot_rebot.scripts.deploy import _ordered_state


def test_dataset_action_names_match_raw_observation_keys() -> None:
    """部署日志必须直接使用 joint1.pos，不能拼成 joint1.pos.pos。"""
    observation = {f"{name}.pos": float(index) for index, name in enumerate(JOINT_NAMES)}
    action_names = list(observation)

    state = _ordered_state(observation, action_names)

    assert state == [float(index) for index in range(7)]
