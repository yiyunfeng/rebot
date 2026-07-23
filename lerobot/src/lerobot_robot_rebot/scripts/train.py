"""用本地 B601 RGB-D 数据训练 ACT/DP，或微调 π0/π0.5。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

POLICIES = ("act", "diffusion", "pi0", "pi05")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_train_command(args: argparse.Namespace) -> list[str]:
    output_dir = args.output_dir or project_root() / "outputs" / args.policy
    steps = args.steps or (30_000 if args.policy in {"pi0", "pi05"} else 100_000)
    batch_size = args.batch_size or (1 if args.policy in {"pi0", "pi05"} else 8)
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={args.dataset_root.resolve()}",
        "--dataset.video_backend=pyav",
        f"--output_dir={output_dir.resolve()}",
        f"--job_name=rebot_b601_{args.policy}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--num_workers={args.num_workers}",
        "--wandb.enable=false",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
    ]

    if args.policy == "act":
        command.extend(["--policy.type=act", "--policy.chunk_size=50", "--policy.n_action_steps=25"])
    elif args.policy == "diffusion":
        command.extend(
            [
                "--policy.type=diffusion",
                "--policy.horizon=16",
                "--policy.n_action_steps=8",
                "--policy.n_obs_steps=2",
            ]
        )
    else:
        model_dir = args.models_dir / f"{args.policy}_base"
        if not (model_dir / "config.json").is_file() or not any(model_dir.glob("*.safetensors")):
            raise FileNotFoundError(
                f"缺少 {args.policy} 本地基础模型: {model_dir}；先运行 rebot-download-models"
            )
        rename_map = {
            "observation.images.main_rgb": "observation.images.base_0_rgb",
            "observation.images.main_depth": "observation.images.left_wrist_0_rgb",
        }
        command.extend(
            [
                f"--policy.path={model_dir.resolve()}",
                f"--rename_map={json.dumps(rename_map, separators=(',', ':'))}",
                "--policy.gradient_checkpointing=true",
                f"--policy.train_expert_only={str(not args.pi_full_finetune).lower()}",
            ]
        )
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/rebot_b601_rgbd")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models-dir", type=Path, default=project_root() / "models")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--pi-full-finetune", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dataset_root.is_dir():
        parser.error(f"数据集目录不存在: {args.dataset_root}")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps 必须为正数")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size 必须为正数")
    return args


def main() -> None:
    args = parse_args()
    command = build_train_command(args)
    print("训练命令：")
    print(" ".join(command))
    if args.dry_run:
        return
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("当前 rebot_lerobot 环境检测不到 CUDA；先修复 NVIDIA 驱动/容器映射")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
