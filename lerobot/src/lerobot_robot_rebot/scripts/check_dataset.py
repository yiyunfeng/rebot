"""校验 B601 RGB-D 数据集的关节、动作和原始毫米深度。"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from ..camera import depth_mm_to_model_image
from ..config import DEFAULT_JOINT_LIMITS, JOINT_NAMES


@dataclass
class CheckReport:
    episodes: int = 0
    frames: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_dataset_index(root: Path) -> tuple[dict, list[dict]]:
    """只读取小体积元数据，不让 Hugging Face 解码整套 RGB-D 图像。"""
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episode_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"找不到 episode 元数据: {root / 'meta' / 'episodes'}")

    columns = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    episodes: list[dict] = []
    for path in episode_files:
        episodes.extend(pq.read_table(path, columns=columns).to_pylist())
    episodes.sort(key=lambda item: int(item["episode_index"]))
    return info, episodes


def _decode_stored_image(value: dict, root: Path) -> np.ndarray:
    """解码 Parquet 中的一张图；每次仅保留当前帧，控制峰值内存。"""
    image_bytes = value.get("bytes")
    if image_bytes is not None:
        source = io.BytesIO(image_bytes)
    else:
        relative_path = value.get("path")
        if not relative_path:
            raise ValueError("图像记录同时缺少 bytes 和 path")
        source = root / relative_path
    with Image.open(source) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def check_dataset(
    root: Path,
    repo_id: str,
    min_depth_mm: int,
    max_depth_mm: int,
    max_action_step: float,
) -> CheckReport:
    del repo_id  # 本地质检直接读取标准 LeRobot v3 文件，不需要访问 Hub。
    info, episodes = _load_dataset_index(root)
    report = CheckReport(int(info["total_episodes"]), int(info["total_frames"]))
    if len(episodes) != report.episodes:
        report.errors.append(f"episode 元数据={len(episodes)}，info.json={report.episodes}")

    data_path_pattern = str(info["data_path"])
    checked_frames = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        raw_paths = sorted((root / "raw_depth" / f"episode_{episode_index:06d}").glob("frame_*.png"))
        expected_frames = end - start
        can_check_depth = len(raw_paths) == expected_frames
        if not can_check_depth:
            report.errors.append(
                f"episode {episode_index}: raw depth={len(raw_paths)}，dataset frames={expected_frames}"
            )

        data_path = root / data_path_pattern.format(
            chunk_index=int(episode["data/chunk_index"]),
            file_index=int(episode["data/file_index"]),
        )
        if not data_path.is_file():
            report.errors.append(f"episode {episode_index}: 数据文件不存在: {data_path}")
            continue

        frame_index = 0
        columns = [
            "observation.state",
            "action",
            "observation.images.main_depth",
            "episode_index",
        ]
        for batch in pq.ParquetFile(data_path).iter_batches(batch_size=128, columns=columns):
            for item in batch.to_pylist():
                # 一个 Parquet 文件可能容纳多条 episode，只检查当前元数据对应的行。
                if int(item["episode_index"]) != episode_index:
                    continue
                global_index = start + frame_index
                state = np.asarray(item["observation.state"], dtype=np.float32).reshape(-1)
                action = np.asarray(item["action"], dtype=np.float32).reshape(-1)
                if state.shape != (7,) or action.shape != (7,):
                    report.errors.append(f"frame {global_index}: state/action 维度不是 7")
                else:
                    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
                        report.errors.append(f"frame {global_index}: state/action 含 NaN 或 Inf")
                    for joint_index, name in enumerate(JOINT_NAMES):
                        lower, upper = DEFAULT_JOINT_LIMITS[name]
                        if not lower <= state[joint_index] <= upper:
                            report.errors.append(
                                f"frame {global_index}: {name} state={state[joint_index]:.4f} 超限"
                            )
                        if not lower <= action[joint_index] <= upper:
                            report.errors.append(
                                f"frame {global_index}: {name} action={action[joint_index]:.4f} 超限"
                            )
                    largest_step = float(np.max(np.abs(action - state)))
                    if largest_step > max_action_step:
                        report.warnings.append(
                            f"frame {global_index}: |action-state| 最大 {largest_step:.4f} rad，"
                            f"超过 {max_action_step}"
                        )

                if can_check_depth:
                    with Image.open(raw_paths[frame_index]) as raw_image:
                        depth_mm = np.asarray(raw_image, dtype=np.uint16).copy()
                    stored_model = _decode_stored_image(
                        item["observation.images.main_depth"], root
                    )
                    expected_model = depth_mm_to_model_image(
                        depth_mm, min_depth_mm, max_depth_mm
                    )
                    if stored_model.shape != expected_model.shape or not np.array_equal(
                        stored_model, expected_model
                    ):
                        report.errors.append(
                            f"episode {episode_index} frame {frame_index}: "
                            "模型深度与 raw depth 映射不一致"
                        )
                frame_index += 1

        checked_frames += frame_index
        if frame_index != expected_frames:
            report.errors.append(
                f"episode {episode_index}: Parquet frames={frame_index}，"
                f"episode metadata={expected_frames}"
            )

    if checked_frames != report.frames:
        report.errors.append(f"已检查 frames={checked_frames}，info.json={report.frames}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/rebot_b601_rgbd")
    parser.add_argument("--min-depth-mm", type=int, default=150)
    parser.add_argument("--max-depth-mm", type=int, default=2000)
    parser.add_argument("--max-action-step", type=float, default=0.3)
    parser.add_argument("--report", type=Path, help="可选的新 JSON 报告路径")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"数据集目录不存在: {args.root}")
    if args.report is not None and args.report.exists():
        parser.error(f"为避免覆盖已有报告，目标必须不存在: {args.report}")
    return args


def main() -> None:
    args = parse_args()
    report = check_dataset(
        args.root,
        args.repo_id,
        args.min_depth_mm,
        args.max_depth_mm,
        args.max_action_step,
    )
    payload = asdict(report) | {"ok": report.ok}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
