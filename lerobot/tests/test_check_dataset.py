import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_robot_rebot.camera import depth_mm_to_model_image
from lerobot_robot_rebot.config import JOINT_NAMES
from lerobot_robot_rebot.scripts.check_dataset import check_dataset
from lerobot_robot_rebot.scripts.record import ReBotDragRecorder, dataset_features


class FakeValidRobot:
    """生成满足关节限制且深度映射正确的两帧测试数据。"""

    def __init__(self) -> None:
        self.depth_mm = np.array([[0, 200, 500], [800, 1200, 2000]], dtype=np.uint16)

    def get_observation(self) -> dict:
        state = np.array([-0.1, -1.0, -1.0, 0.2, 0.0, 0.0, -1.0], dtype=np.float32)
        observation = {
            f"{name}.pos": float(value)
            for name, value in zip(JOINT_NAMES, state, strict=True)
        }
        observation["main_rgb"] = np.zeros((2, 3, 3), dtype=np.uint8)
        observation["main_depth"] = depth_mm_to_model_image(self.depth_mm, 150, 2000)
        return observation

    def get_depth_mm(self) -> np.ndarray:
        return self.depth_mm.copy()


def test_checker_streams_valid_parquet_and_raw_depth(tmp_path) -> None:
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/test_rebot_check",
        root=root,
        fps=30,
        robot_type="rebot_b601",
        features=dataset_features(height=2, width=3),
        use_videos=False,
    )
    recorder = ReBotDragRecorder(FakeValidRobot(), dataset, "测试抓取", fps=30)
    recorder.start_episode()
    recorder.capture()
    recorder.capture()
    recorder.finish_episode()
    dataset.finalize()

    report = check_dataset(root, "local/test_rebot_check", 150, 2000, 0.3)

    assert report.ok
    assert report.episodes == 1
    assert report.frames == 2
    assert report.warnings == []
