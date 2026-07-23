#!/usr/bin/env python3
"""真机 RGB-D 神经网络策略 dry-run。

默认只做三件事：
1. 打开真实 RGB-D 相机；
2. 按训练时同样布局构造 16405 维观测；
3. 加载导出的 TorchScript 策略并打印 7 维动作。

它不会连接机械臂，也不会发送任何运动命令。真正让机械臂运动前，必须另外接入
限位、速度限制、工作空间裁剪、碰撞检查、watchdog 和人工急停确认。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
GRASP_ROOT = REPO_ROOT / "rebot_grasp"
POLICY_PATH = PROJECT_ROOT / "exported" / "rgbd_policy_latest.pt"
CONFIG_PATH = GRASP_ROOT / "config" / "default.yaml"

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 4
PROPRIO_SIZE = 21
OBS_DIM = PROPRIO_SIZE + IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS
DEPTH_LIMIT_M = 1.5
sys.path.insert(0, str(GRASP_ROOT))

from drivers.camera import make_camera  # noqa: E402


def flatten_real_rgbd(color_bgr: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
    """把真机 RGB-D 转成训练同布局的 64x64x4 展平图像。"""

    color_small = cv2.resize(color_bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth_mm, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

    rgb = color_small[..., ::-1].astype(np.float32) / 255.0
    rgb -= rgb.mean(axis=(0, 1), keepdims=True)

    depth_m = depth_small.astype(np.float32) / 1000.0
    depth_m = np.nan_to_num(depth_m, nan=DEPTH_LIMIT_M, posinf=DEPTH_LIMIT_M, neginf=0.0)
    depth = np.clip(depth_m, 0.0, DEPTH_LIMIT_M)[..., None] / DEPTH_LIMIT_M
    return np.concatenate([rgb, depth], axis=-1).reshape(-1).astype(np.float32)


def main() -> None:
    """循环读取真机相机并打印网络动作，作为 sim2real 部署前检查。"""

    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"没有导出策略，请先运行 ./run_export_rgbd.sh: {POLICY_PATH}")

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    camera = make_camera(cfg)
    policy = torch.jit.load(str(POLICY_PATH), map_location="cpu").eval()
    last_action = np.zeros(7, dtype=np.float32)

    print("[RealPolicy] DRY-RUN only: no robot command will be sent")
    print(f"[RealPolicy] policy={POLICY_PATH}")
    try:
        camera.open()
        camera.warm_up(10)
        while True:
            color_bgr, depth_mm = camera.get_frame()
            if color_bgr is None or depth_mm is None:
                time.sleep(0.02)
                continue

            # 真机执行版需要填入真实 joint_pos7/joint_vel7。dry-run 没连接机械臂，
            # 所以前 14 维先置零，只保留 last_action，验证相机和网络链路。
            proprio = np.zeros(PROPRIO_SIZE, dtype=np.float32)
            proprio[14:21] = last_action
            rgbd = flatten_real_rgbd(color_bgr, depth_mm)
            obs = torch.from_numpy(np.concatenate([proprio, rgbd])[None, :])
            if obs.shape[1] != OBS_DIM:
                raise RuntimeError(f"观测维度错误: {obs.shape[1]} != {OBS_DIM}")

            with torch.inference_mode():
                action = policy(obs).cpu().numpy()[0].astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
            last_action = action
            print(
                "[RealPolicy] action "
                f"dxyz={np.round(action[:3], 3).tolist()} "
                f"drpy={np.round(action[3:6], 3).tolist()} "
                f"gripper={action[6]:+.3f}",
                end="\r",
                flush=True,
            )
            time.sleep(0.10)
    except KeyboardInterrupt:
        print("\n[RealPolicy] stopped")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
