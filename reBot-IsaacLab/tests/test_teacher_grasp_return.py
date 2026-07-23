"""teacher 只采集抓取并返回 ready，不包含 IsaacSim 后续放置阶段。"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


# ROS Humble 也安装了名为 ``scripts`` 的顶层包，显式加入项目脚本目录以免误导入。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import collect_isaacsim_teacher as teacher  # noqa: E402


def _stage(name: str, arm_offset: float, gripper: float) -> dict:
    return {
        "name": name,
        "arm": [arm_offset, 0.0, 0.0, 0.0, 0.0, 0.0],
        "gripper_m_per_finger": gripper,
    }


def _plan() -> dict:
    return {
        "timestamp": 1.0,
        "tcp_rotation": np.eye(3).tolist(),
        "pregrasp_position_m": [0.25, 0.0, 0.10],
        "grasp_position_m": [0.30, 0.0, 0.10],
        "stages": [
            _stage("open", 0.00, 0.030),
            _stage("pregrasp", 0.10, 0.030),
            _stage("grasp", 0.20, 0.030),
            _stage("close", 0.20, 0.001),
            _stage("retreat", 0.10, 0.001),
            _stage("return", 0.00, 0.001),
            # 上游 IsaacSim 可以继续放置，但 collector 必须忽略这些阶段。
            _stage("place", 0.30, 0.001),
            _stage("release", 0.30, 0.030),
        ],
    }


def test_teacher_ends_at_return_and_ignores_place(monkeypatch):
    def fake_stage_pose(stage: dict):
        return np.array([stage["arm"][0], 0.0, 0.10]), np.eye(3)

    monkeypatch.setattr(teacher, "stage_pose", fake_stage_pose)
    image = torch.zeros(teacher.IMAGE_HEIGHT * teacher.IMAGE_WIDTH * 4)

    observations, actions, labels = teacher.samples_from_plan(_plan(), image)

    expected_samples = (
        teacher.APPROACH_STEPS
        + teacher.INSERT_STEPS
        + teacher.CLOSE_STEPS
        + teacher.RETREAT_STEPS
        + teacher.RETURN_STEPS
    )
    assert len(observations) == len(actions) == len(labels) == expected_samples
    assert labels[-1]["segment"] == "retreat->return"
    assert all("place" not in label["segment"] for label in labels)
    assert actions[-1][0, 6].item() == -1.0
    # 第一个 open/ready 样本的六轴相对关节位置应为 0，而不是绝对关节角。
    assert torch.allclose(observations[0][0, :6], torch.zeros(6))


def test_teacher_requires_return_stage(monkeypatch):
    monkeypatch.setattr(teacher, "stage_pose", lambda _stage: (np.zeros(3), np.eye(3)))
    plan = _plan()
    plan["stages"] = [stage for stage in plan["stages"] if stage["name"] != "return"]

    with pytest.raises(ValueError, match="return"):
        teacher.samples_from_plan(plan, torch.zeros(teacher.IMAGE_HEIGHT * teacher.IMAGE_WIDTH * 4))
