#!/usr/bin/env python3
"""真机 RGB-D 神经网络策略安全执行入口。

默认不会连接机械臂。只有同时满足下面两个条件才会发送真实运动：

1. 环境变量 ``REBOT_REAL_POLICY_ENABLE=1``；
2. 启动后人工输入 ``RUN POLICY``。

该脚本把策略输出的 7 维动作解释为：

- ``action[0:3]``：TCP 在 base 坐标系下的小平移，最大每步 2 cm；
- ``action[3:6]``：TCP RPY 小角度增量，最大每步 0.10 rad；
- ``action[6]``：夹爪，正数打开，负数力控闭合。

它不是绕过传统抓取器的安全控制，而是给训练好的策略提供一条可审查的真机
执行通道：限速、限位、人工确认和急停检查必须保留。
"""

from __future__ import annotations

import os
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
PROPRIO_SIZE = 21
OBS_DIM = PROPRIO_SIZE + IMAGE_HEIGHT * IMAGE_WIDTH * 4
DEPTH_LIMIT_M = 1.5

# 与 IsaacLab 环境动作尺度保持一致。
MAX_TRANSLATION_M = 0.02
MAX_ROTATION_RAD = 0.10
CONTROL_DT_S = 0.25

# 保守 workspace。真正上真机前可根据桌面、夹具和相机视野再收紧。
WORKSPACE_X = (0.10, 0.45)
WORKSPACE_Y = (-0.25, 0.25)
WORKSPACE_Z = (0.04, 0.35)

sys.path.insert(0, str(GRASP_ROOT))

from drivers.camera import make_camera  # noqa: E402
from drivers.robot.grasp_driver import (  # noqa: E402
    GraspDriver,
    ensure_rebot_sdk_in_syspath,
    selected_arm_config,
    selected_hardware_yaml,
)


def flatten_real_rgbd(color_bgr: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
    """把真机 RGB-D 转成训练同布局的 64×64×4 展平图像。"""

    color_small = cv2.resize(color_bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth_mm, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)

    rgb = color_small[..., ::-1].astype(np.float32) / 255.0
    rgb -= rgb.mean(axis=(0, 1), keepdims=True)
    depth_m = depth_small.astype(np.float32) / 1000.0
    depth_m = np.nan_to_num(depth_m, nan=DEPTH_LIMIT_M, posinf=DEPTH_LIMIT_M, neginf=0.0)
    depth = np.clip(depth_m, 0.0, DEPTH_LIMIT_M)[..., None] / DEPTH_LIMIT_M
    return np.concatenate([rgb, depth], axis=-1).reshape(-1).astype(np.float32)


def matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    """把旋转矩阵转为 XYZ/RPY 欧拉角。"""

    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if sy > 1e-6:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def wait_motion(controller, duration: float) -> None:
    """等待 SDK 轨迹线程结束，超时立即报错。"""

    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + 1.0)
        if thread.is_alive():
            raise TimeoutError(f"robot trajectory exceeded {duration + 1.0:.1f}s")
    else:
        time.sleep(duration)


def read_proprio(arm, grasp_driver: GraspDriver, last_action: np.ndarray, cfg: dict) -> np.ndarray:
    """读取 7 维关节位置、7 维速度和上一动作，组成训练同布局本体观测。"""

    state = arm.get_state(request_feedback=False)
    q_arm = np.asarray(state[0], dtype=np.float32)[:6]
    qd_arm = np.asarray(state[1], dtype=np.float32)[:6] if len(state) > 1 else np.zeros(6, dtype=np.float32)

    gripper_pos, gripper_vel, _ = grasp_driver.get_gripper_state()
    angle_open = float(cfg["robot"]["gripper"]["dm"]["angle_open"])
    gripper_m = np.clip(abs(gripper_pos) / max(angle_open, 1e-6) * GraspDriver.MAX_DISTANCE_M, 0.0, GraspDriver.MAX_DISTANCE_M)
    gripper_vel_m = np.clip(abs(gripper_vel) / max(angle_open, 1e-6) * GraspDriver.MAX_DISTANCE_M, 0.0, GraspDriver.MAX_DISTANCE_M)

    joint_pos = np.concatenate([q_arm, np.array([gripper_m], dtype=np.float32)])
    joint_vel = np.concatenate([qd_arm, np.array([gripper_vel_m], dtype=np.float32)])
    return np.concatenate([joint_pos, joint_vel, last_action.astype(np.float32)]).astype(np.float32)


def clip_workspace(xyz: np.ndarray) -> np.ndarray:
    """把策略目标限制在保守工作空间内。"""

    return np.array(
        [
            np.clip(xyz[0], *WORKSPACE_X),
            np.clip(xyz[1], *WORKSPACE_Y),
            np.clip(xyz[2], *WORKSPACE_Z),
        ],
        dtype=np.float64,
    )


def main() -> int:
    """加载策略、相机和机械臂，循环执行安全裁剪后的网络动作。"""

    if os.environ.get("REBOT_REAL_POLICY_ENABLE") != "1":
        print("[RealPolicyExec] blocked: set REBOT_REAL_POLICY_ENABLE=1 only after safety checks")
        return 0
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"没有导出策略，请先运行 ./run_export_rgbd.sh: {POLICY_PATH}")

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    # SDK/Pinocchio 在部分环境中可能受 NumPy ABI 影响；必须放在安全开关之后，
    # 确保未显式开启真机执行时不会仅因 import 就触发底层库崩溃。
    ensure_rebot_sdk_in_syspath(cfg["robot"].get("repo_root"))
    from reBotArm_control_py.actuator import RebotArm
    from reBotArm_control_py.controllers import RebotArmEndPose

    selected = selected_arm_config(cfg["robot"].get("repo_root"))
    hardware_yaml = selected_hardware_yaml(cfg["robot"].get("repo_root"))
    hardware_cfg = yaml.safe_load(hardware_yaml.read_text(encoding="utf-8"))
    channel = str(hardware_cfg["channel"])
    if selected.arm_type != "dm" or not channel.startswith("/dev/tty"):
        raise RuntimeError(f"只允许 B601-DM 串口配置，当前 arm={selected.arm_type}, channel={channel}")
    if not Path(channel).exists():
        raise FileNotFoundError(f"DM serial channel does not exist: {channel}")

    print(f"[Safety] model=B601-DM channel={channel}")
    print(f"[Safety] workspace x={WORKSPACE_X}, y={WORKSPACE_Y}, z={WORKSPACE_Z}")
    print("[Safety] clear workspace, confirm joint/gripper limits, keep emergency stop ready")
    if input("Type RUN POLICY after checking workspace and E-stop: ").strip() != "RUN POLICY":
        print("[RealPolicyExec] cancelled before connecting hardware")
        return 0

    camera = make_camera(cfg)
    arm = None
    controller = None
    grasp_driver: GraspDriver | None = None
    policy = torch.jit.load(str(POLICY_PATH), map_location="cpu").eval()
    last_action = np.zeros(7, dtype=np.float32)
    gripper_is_open = False
    max_steps = int(os.environ.get("REBOT_REAL_POLICY_STEPS", "120"))

    try:
        camera.open()
        camera.warm_up(10)
        arm = RebotArm()
        controller = RebotArmEndPose(arm, arm_control_mode=selected.controller_mode)
        grasp_driver = GraspDriver(
            arm,
            controller,
            gripper_config=cfg["robot"].get("gripper"),
            repo_root=cfg["robot"].get("repo_root"),
        )
        grasp_driver.start()

        print(f"[RealPolicyExec] policy={POLICY_PATH}")
        for step in range(max_steps):
            color_bgr, depth_mm = camera.get_frame()
            if color_bgr is None or depth_mm is None:
                time.sleep(0.02)
                continue

            proprio = read_proprio(arm, grasp_driver, last_action, cfg)
            rgbd = flatten_real_rgbd(color_bgr, depth_mm)
            obs = torch.from_numpy(np.concatenate([proprio, rgbd])[None, :])
            if obs.shape[1] != OBS_DIM:
                raise RuntimeError(f"观测维度错误: {obs.shape[1]} != {OBS_DIM}")

            with torch.inference_mode():
                action = policy(obs).cpu().numpy()[0].astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
            last_action = action

            T_tcp = grasp_driver.get_tcp_pose()
            xyz = T_tcp[:3, 3] + action[:3].astype(np.float64) * MAX_TRANSLATION_M
            xyz = clip_workspace(xyz)
            rpy = np.asarray(matrix_to_rpy(T_tcp[:3, :3]), dtype=np.float64)
            rpy += action[3:6].astype(np.float64) * MAX_ROTATION_RAD

            print(
                f"[RealPolicyExec] step={step + 1}/{max_steps} "
                f"target_xyz={np.round(xyz, 3).tolist()} "
                f"action={np.round(action, 3).tolist()}"
            )
            if not controller.move_to_traj(*xyz.tolist(), *rpy.tolist(), duration=CONTROL_DT_S):
                raise RuntimeError("policy target IK/trajectory failed")
            wait_motion(controller, CONTROL_DT_S)

            if action[6] > 0.4 and not gripper_is_open:
                grasp_driver.open_gripper(0.045, timeout=2.0)
                gripper_is_open = True
            elif action[6] < -0.4 and gripper_is_open:
                grasp_driver.grasp(timeout=3.0)
                gripper_is_open = False
        print("[RealPolicyExec] max steps reached")
    finally:
        print("[RealPolicyExec] stopping and disconnecting")
        try:
            if grasp_driver is not None:
                grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[RealPolicyExec] gripper cleanup failed: {exc}")
        try:
            if controller is not None and controller._running:
                controller.end()
            elif arm is not None:
                arm.disconnect()
        except Exception as exc:
            print(f"[RealPolicyExec] robot cleanup failed: {exc}")
        try:
            camera.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[RealPolicyExec] interrupted")
        raise SystemExit(130)
