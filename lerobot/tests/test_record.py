import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

from datasets import config as datasets_config
from lerobot_robot_rebot.config import JOINT_NAMES
from lerobot_robot_rebot.scripts.record import ReBotDragRecorder, dataset_features


class FakeRecordingRobot:
    def __init__(self) -> None:
        self.frame = 0
        self.depth_mm = np.array([[0, 200, 500], [800, 1200, 2000]], dtype=np.uint16)

    def get_observation(self) -> dict:
        state = np.arange(7, dtype=np.float32) * 0.01 + self.frame * 0.1
        observation = {
            f"{name}.pos": float(value)
            for name, value in zip(JOINT_NAMES, state, strict=True)
        }
        observation["main_rgb"] = np.full((2, 3, 3), self.frame, dtype=np.uint8)
        observation["main_depth"] = np.full((2, 3, 3), self.frame + 1, dtype=np.uint8)
        self.frame += 1
        return observation

    def get_depth_mm(self) -> np.ndarray:
        return self.depth_mm.copy()


def test_recorder_saves_two_frames_and_raw_depth(tmp_path) -> None:
    dataset = LeRobotDataset.create(
        repo_id="local/test_rebot",
        root=tmp_path / "dataset",
        fps=30,
        robot_type="rebot_b601",
        features=dataset_features(height=2, width=3),
        use_videos=False,
    )
    recorder = ReBotDragRecorder(FakeRecordingRobot(), dataset, "测试抓取", fps=30)

    recorder.start_episode()
    recorder.capture()
    recorder.capture()
    recorder.finish_episode()
    dataset.finalize()

    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 2
    raw_paths = sorted((dataset.root / "raw_depth" / "episode_000000").glob("frame_*.png"))
    assert len(raw_paths) == 2
    np.testing.assert_array_equal(np.asarray(Image.open(raw_paths[0]), dtype=np.uint16), recorder.robot.depth_mm)


def test_recorder_archives_stale_raw_depth_directory(tmp_path) -> None:
    dataset = LeRobotDataset.create(
        repo_id="local/test_rebot_stale",
        root=tmp_path / "dataset",
        fps=30,
        robot_type="rebot_b601",
        features=dataset_features(height=2, width=3),
        use_videos=False,
    )
    stale_depth = dataset.root / "raw_depth" / "episode_000000"
    stale_depth.mkdir(parents=True)
    (stale_depth / "partial.png").write_bytes(b"partial")
    recorder = ReBotDragRecorder(FakeRecordingRobot(), dataset, "测试抓取", fps=30)

    recorder.start_episode()

    archived = list((dataset.root / "discarded").glob("stale_episode_000000_*/depth_mm"))
    assert len(archived) == 1
    assert (archived[0] / "partial.png").read_bytes() == b"partial"
    assert stale_depth.is_dir()


def test_recorder_resumes_and_saves_next_episode(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", tmp_path / "hf_cache")
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/test_rebot_resume",
        root=root,
        fps=30,
        robot_type="rebot_b601",
        features=dataset_features(height=2, width=3),
        use_videos=False,
    )
    first = ReBotDragRecorder(FakeRecordingRobot(), dataset, "测试抓取", fps=30)
    first.start_episode()
    first.capture()
    first.capture()
    first.finish_episode()
    dataset.finalize()

    resumed_dataset = LeRobotDataset(
        repo_id="local/test_rebot_resume",
        root=root,
        video_backend="pyav",
    )
    assert resumed_dataset.episode_buffer is None
    resumed_dataset.start_image_writer(num_threads=1)
    second = ReBotDragRecorder(FakeRecordingRobot(), resumed_dataset, "测试抓取", fps=30)
    second.start_episode()
    second.capture()
    second.capture()
    second.finish_episode()
    resumed_dataset.finalize()

    assert resumed_dataset.meta.total_episodes == 2
    assert resumed_dataset.meta.total_frames == 4
    assert (root / "raw_depth" / "episode_000001" / "frame_000001.png").is_file()
