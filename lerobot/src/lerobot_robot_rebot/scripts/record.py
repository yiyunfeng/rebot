"""B601 MIT 拖动示教与单相机 RGB-D 数据录制。"""

from __future__ import annotations

import argparse
import json
import queue
import select
import sys
import termios
import threading
import time
import tty
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
from PIL import Image

from ..camera import ReBotRGBDConfig
from ..config import JOINT_NAMES, ReBotB601Config
from ..robot import ReBotB601


@dataclass
class PendingFrame:
    observation: dict
    depth_mm: np.ndarray


class RawKeyboard:
    """后台线程只读取按键；除 Esc 外，控制命令都由主循环串行执行。"""

    def __init__(self, emergency_callback) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("拖动示教必须在交互式终端运行")
        self.events: queue.Queue[str] = queue.Queue()
        self._emergency_callback = emergency_callback
        self._running = False
        self._thread: threading.Thread | None = None
        self._old_settings = None

    def start(self) -> None:
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, name="rebot-keyboard", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while self._running:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            key = sys.stdin.read(1).lower()
            if key == "\x1b":
                # Esc 不等待下一帧，立即停止发送线程并失能电机。
                self._emergency_callback()
                self.events.put("esc")
                return
            self.events.put(key)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)


class ReBotDragRecorder:
    """用 observation_t 和下一帧实测位置 q_(t+1) 形成监督样本。"""

    def __init__(self, robot: ReBotB601, dataset: LeRobotDataset, task: str, fps: int) -> None:
        self.robot = robot
        self.dataset = dataset
        self.task = task
        self.fps = fps
        self.is_recording = False
        self._pending: PendingFrame | None = None
        self._raw_depth_dir: Path | None = None

    def start_episode(self) -> None:
        if self.is_recording:
            return
        # LeRobotDataset 从磁盘续录时不会预建 episode_buffer；自建数据集才会。
        # 在第一次 Space 时初始化，保证 capture、discard 都能安全访问。
        if self.dataset.episode_buffer is None:
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()
        episode_index = self.dataset.meta.total_episodes
        self._raw_depth_dir = self.dataset.root / "raw_depth" / f"episode_{episode_index:06d}"
        if self._raw_depth_dir.exists():
            # 上次可能在创建目录后、进入录制状态前异常退出。保留残留并让本次续录继续。
            stale_dir = (
                self.dataset.root
                / "discarded"
                / f"stale_episode_{episode_index:06d}_{uuid.uuid4().hex[:8]}"
            )
            stale_dir.mkdir(parents=True, exist_ok=False)
            self._raw_depth_dir.rename(stale_dir / "depth_mm")
            (stale_dir / "reason.json").write_text(
                json.dumps(
                    {"reason": "stale_raw_depth_before_start", "time": time.time()},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[录制] 已归档同名残留目录到 {stale_dir}")
        self._raw_depth_dir.mkdir(parents=True, exist_ok=False)
        self._pending = None
        self.is_recording = True
        print(f"[录制] episode {episode_index} 开始；再次按 Space 保存")

    def capture(self) -> None:
        if not self.is_recording:
            return
        observation = self.robot.get_observation()
        depth_mm = self.robot.get_depth_mm()
        state = self._state_from_observation(observation)

        if self._pending is not None:
            self._commit(self._pending, action=state)
        self._pending = PendingFrame(observation, depth_mm)

    def finish_episode(self) -> None:
        if not self.is_recording or self._pending is None:
            raise RuntimeError("当前 episode 没有可保存的帧")
        final_state = self._state_from_observation(self._pending.observation)
        self._commit(self._pending, action=final_state)
        self._pending = None

        episode_index = self.dataset.meta.total_episodes
        # 使用原版 LeRobot API；图像已嵌入 Parquet，uint16 毫米深度另存 sidecar。
        self.dataset.save_episode()
        self._write_raw_metadata(episode_index)
        self._raw_depth_dir = None
        self.is_recording = False
        print(f"[录制] episode {episode_index} 已保存")

    def discard_episode(self, reason: str = "redo") -> None:
        """将未完成尝试归档到 discarded，不直接删除采集文件。"""

        if not self.is_recording:
            print("[录制] 当前没有正在录制的 episode")
            return
        self.dataset._wait_image_writer()
        attempt_dir = self.dataset.root / "discarded" / f"{reason}_{uuid.uuid4().hex[:8]}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        episode_index = self.dataset.episode_buffer["episode_index"]

        for camera_key in self.dataset.meta.image_keys:
            image_dir = self.dataset._get_image_file_dir(episode_index, camera_key)
            if image_dir.is_dir():
                image_dir.rename(attempt_dir / camera_key.replace(".", "_"))
        if self._raw_depth_dir is not None and self._raw_depth_dir.is_dir():
            self._raw_depth_dir.rename(attempt_dir / "depth_mm")

        (attempt_dir / "reason.json").write_text(
            json.dumps({"reason": reason, "time": time.time()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.dataset.clear_episode_buffer(delete_images=False)
        self._pending = None
        self._raw_depth_dir = None
        self.is_recording = False
        print(f"[录制] 本次数据已归档到 {attempt_dir}")

    def _commit(self, pending: PendingFrame, action: np.ndarray) -> None:
        frame_index = self.dataset.episode_buffer["size"]
        state = self._state_from_observation(pending.observation)
        self.dataset.add_frame(
            {
                "observation.state": state.astype(np.float32),
                "observation.images.main_rgb": pending.observation["main_rgb"],
                "observation.images.main_depth": pending.observation["main_depth"],
                "action": action.astype(np.float32),
                "task": self.task,
            }
        )
        if self._raw_depth_dir is None:
            raise RuntimeError("raw depth 目录尚未创建")
        Image.fromarray(pending.depth_mm).save(self._raw_depth_dir / f"frame_{frame_index:06d}.png")

    @staticmethod
    def _state_from_observation(observation: dict) -> np.ndarray:
        return np.asarray([observation[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float64)

    def _write_raw_metadata(self, episode_index: int) -> None:
        if self._raw_depth_dir is None:
            return
        metadata = {
            "episode_index": episode_index,
            "unit": "millimeter",
            "dtype": "uint16",
            "fps": self.fps,
            "alignment": "depth_to_color",
            "model_depth_key": "observation.images.main_depth",
        }
        (self._raw_depth_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def dataset_features(height: int, width: int) -> dict:
    state_names = [f"{name}.pos" for name in JOINT_NAMES]
    image_feature = {
        "dtype": "image",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }
    return {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": state_names},
        "action": {"dtype": "float32", "shape": (7,), "names": state_names},
        "observation.images.main_rgb": image_feature.copy(),
        "observation.images.main_depth": image_feature.copy(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="本地 LeRobotDataset 目录")
    parser.add_argument("--repo-id", default="local/rebot_b601_rgbd")
    parser.add_argument("--task", required=True, help="任务文本，例如：抓起红色方块并放入盒中")
    parser.add_argument("--hardware-yaml", type=Path)
    parser.add_argument("--sdk-path", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--min-depth-mm", type=int, default=150)
    parser.add_argument("--max-depth-mm", type=int, default=2000)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="向已有且至少包含一个 episode 的数据集续录")
    parser.add_argument(
        "--confirm-hardware-safe",
        action="store_true",
        help="确认型号、通信、限位、净空和物理急停均已检查",
    )
    args = parser.parse_args()
    if not args.confirm_hardware_safe:
        parser.error("真机录制必须显式传入 --confirm-hardware-safe")
    if args.resume:
        info_path = args.root / "meta" / "info.json"
        if not info_path.is_file():
            parser.error(f"续录目录缺少 meta/info.json: {args.root}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if int(info.get("total_episodes", 0)) <= 0:
            parser.error("--resume 需要至少一个已保存 episode；零 episode 归档请改用新目录")
    elif args.root.exists() and any(args.root.iterdir()):
        parser.error(f"为避免覆盖数据，--root 必须为空目录或不存在: {args.root}")
    if args.fps <= 0 or args.num_episodes <= 0:
        parser.error("--fps 和 --num-episodes 必须为正数")
    return args


def main() -> None:
    args = parse_args()
    robot = ReBotB601(
        ReBotB601Config(
            id="b601_drag",
            operating_mode="teach",
            sdk_path=args.sdk_path,
            hardware_yaml=args.hardware_yaml,
            camera=ReBotRGBDConfig(
                width=args.width,
                height=args.height,
                fps=args.fps,
                min_depth_mm=args.min_depth_mm,
                max_depth_mm=args.max_depth_mm,
            ),
        )
    )
    features = dataset_features(args.height, args.width)
    if args.resume:
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.root,
            video_backend="pyav",
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, args.fps, features)
        if dataset.meta.total_episodes >= args.num_episodes:
            raise ValueError(
                f"数据集已有 {dataset.meta.total_episodes} 个 episode，"
                f"--num-episodes 必须更大"
            )
        dataset.start_image_writer(num_threads=4)
        print(
            f"[续录] 已有 {dataset.meta.total_episodes} 个 episode，"
            f"将录制到总计 {args.num_episodes} 个"
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=args.root,
            fps=args.fps,
            robot_type=robot.name,
            features=features,
            use_videos=False,
            image_writer_threads=4,
        )
    recorder = ReBotDragRecorder(robot, dataset, args.task, args.fps)
    keyboard = RawKeyboard(robot.emergency_stop)

    print("安全确认：通信通道、关节与夹爪限位、工作区净空和物理急停必须有效。")
    print("按键：Space 开始/保存，O 打开，C 限力闭合，S 保持，R 归档重录，Esc 急停退出。")
    stop_requested = False
    try:
        robot.connect()
        keyboard.start()
        while not stop_requested and dataset.meta.total_episodes < args.num_episodes:
            loop_start = time.perf_counter()
            while not keyboard.events.empty():
                key = keyboard.events.get_nowait()
                if key == "esc":
                    stop_requested = True
                elif key == " ":
                    if recorder.is_recording:
                        recorder.finish_episode()
                    else:
                        recorder.start_episode()
                elif key == "o":
                    robot.open_gripper()
                    print("[夹爪] 打开")
                elif key == "c":
                    robot.close_gripper()
                    print("[夹爪] 限力闭合")
                elif key == "s":
                    robot.hold_gripper()
                    print("[夹爪] 保持当前位置")
                elif key == "r":
                    recorder.discard_episode()

            if recorder.is_recording and not stop_requested:
                recorder.capture()
            time.sleep(max(1 / args.fps - (time.perf_counter() - loop_start), 0.0))
    except BaseException:
        robot.emergency_stop()
        raise
    finally:
        keyboard.stop()
        if recorder.is_recording:
            recorder.discard_episode(reason="aborted")
        dataset.finalize()
        robot.disconnect()


if __name__ == "__main__":
    main()
