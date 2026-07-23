"""下载 π0/π0.5 官方基础模型，供离线微调和部署。"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_IDS = {"pi0": "lerobot/pi0_base", "pi05": "lerobot/pi05_base"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi0", "pi05", "all"), default="all")
    parser.add_argument("--models-dir", type=Path, default=project_root() / "models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = MODEL_IDS if args.model == "all" else {args.model: MODEL_IDS[args.model]}
    for name, repo_id in selected.items():
        target = args.models_dir / f"{name}_base"
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"为避免混合或覆盖权重，目标目录必须为空或不存在: {target}")
        print(f"[下载] {repo_id} -> {target}")
        snapshot_download(repo_id=repo_id, local_dir=target, repo_type="model")
        if not (target / "config.json").is_file() or not any(target.glob("*.safetensors")):
            raise RuntimeError(f"{repo_id} 下载不完整")
        print(f"[完成] {name} 基础模型已可离线使用")


if __name__ == "__main__":
    main()
