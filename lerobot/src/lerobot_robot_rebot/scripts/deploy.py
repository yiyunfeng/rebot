"""在 B601 真机上低速运行本地 ACT/DP checkpoint。"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

from ..camera import ReBotRGBDConfig
from ..config import ReBotB601Config
from ..robot import ReBotB601
from .record import RawKeyboard


def _ordered_state(observation: dict, action_names: list[str]) -> list[float]:
    """按训练动作顺序读取同名关节状态。"""
    return [float(observation[name]) for name in action_names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True, help="训练数据集，用于读取 feature/stats")
    parser.add_argument("--repo-id", default="local/rebot_b601_rgbd")
    parser.add_argument("--task", required=True)
    parser.add_argument("--hardware-yaml", type=Path)
    parser.add_argument("--sdk-path", type=Path)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--fps", type=int, default=10, help="策略闭环频率，初次真机建议 5~10 Hz")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--max-relative-target", type=float, default=0.04)
    parser.add_argument("--arm-velocity", type=float, default=0.3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--min-depth-mm", type=int, default=150)
    parser.add_argument("--max-depth-mm", type=int, default=2000)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--confirm-hardware-safe", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint.is_dir() or not (args.checkpoint / "config.json").is_file():
        parser.error(f"checkpoint 无效: {args.checkpoint}")
    if not args.dataset_root.is_dir():
        parser.error(f"训练数据集不存在: {args.dataset_root}")
    if not args.confirm_hardware_safe:
        parser.error("真机部署必须显式传入 --confirm-hardware-safe")
    if args.fps <= 0 or args.duration <= 0:
        parser.error("--fps 和 --duration 必须为正数")
    if not 0 < args.max_relative_target <= 0.08:
        parser.error("--max-relative-target 必须位于 (0, 0.08]")
    if not 0 < args.arm_velocity <= 1.0:
        parser.error("--arm-velocity 必须位于 (0, 1.0] rad/s")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境检测不到 CUDA；请先修复 NVIDIA 驱动，或仅为调试改用 --device cpu")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    metadata = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.dataset_root)
    policy_config = PreTrainedConfig.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        cli_overrides=[f"--device={args.device}", f"--use_amp={str(args.use_amp).lower()}"],
    )
    policy_config.pretrained_path = args.checkpoint
    policy = make_policy(policy_config, ds_meta=metadata)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    policy.eval()
    policy.reset()

    # 首次 CUDA 推理会初始化算子，可能耗时数秒。用与数据集同尺寸的空观测预热，
    # 且放在连接机械臂之前，确保这段初始化绝不会产生真机动作。
    warmup_observation = dict.fromkeys(
        metadata.features["observation.state"]["names"], 0.0
    )
    for feature_name, feature in metadata.features.items():
        if feature_name.startswith("observation.images."):
            camera_name = feature_name.removeprefix("observation.images.")
            warmup_observation[camera_name] = np.zeros(feature["shape"], dtype=np.uint8)
    warmup_frame = build_inference_frame(
        warmup_observation,
        device=device,
        ds_features=metadata.features,
        task=args.task,
        robot_type=ReBotB601.name,
    )
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda")
        if device.type == "cuda" and args.use_amp
        else nullcontext(),
    ):
        postprocessor(policy.select_action(preprocessor(warmup_frame)))
    policy.reset()
    print("[部署] 模型预热完成，尚未连接机械臂")

    robot = ReBotB601(
        ReBotB601Config(
            id="b601_deploy",
            operating_mode="deploy",
            sdk_path=args.sdk_path,
            hardware_yaml=args.hardware_yaml,
            camera=ReBotRGBDConfig(
                width=args.width,
                height=args.height,
                fps=max(args.fps, 30),
                min_depth_mm=args.min_depth_mm,
                max_depth_mm=args.max_depth_mm,
            ),
            max_relative_target=args.max_relative_target,
            arm_velocity_limits=(args.arm_velocity,) * 6,
        )
    )
    keyboard = RawKeyboard(robot.emergency_stop)

    print("即将进入 POS_VEL 真机部署；Esc 立即急停，Ctrl+C 正常停止。")
    print(f"安全参数：{args.fps} Hz，单步≤{args.max_relative_target} rad，速度≤{args.arm_velocity} rad/s")
    try:
        robot.connect()
        keyboard.start()
        print(
            "机械臂已使能并保持当前位置。确认工作区安全后按 Space 启动策略；"
            "Q 正常结束，Esc 立即急停。"
        )
        while True:
            event = keyboard.events.get()
            if event == "esc":
                raise RuntimeError("用户按 Esc 触发急停")
            if event == "q":
                print("[部署] 用户按 Q，未启动策略并正常退出")
                return
            if event == " ":
                break

        # 等待期间没有调用策略；启动前重置动作队列，确保从当前观测开始推理。
        policy.reset()
        print(f"[部署] ACT/DP 策略已启动，将运行 {args.duration:.1f}s")
        start = time.perf_counter()
        step = 0
        overruns = 0
        while time.perf_counter() - start < args.duration:
            loop_start = time.perf_counter()
            if not keyboard.events.empty():
                event = keyboard.events.get_nowait()
                if event == "esc":
                    raise RuntimeError("用户按 Esc 触发急停")
                if event == "q":
                    print("[部署] 用户按 Q，正常结束当前任务")
                    break

            observation = robot.get_observation()
            inference_frame = build_inference_frame(
                observation,
                device=device,
                ds_features=metadata.features,
                task=args.task,
                robot_type=robot.name,
            )
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda")
                if device.type == "cuda" and args.use_amp
                else nullcontext(),
            ):
                action_tensor = policy.select_action(preprocessor(inference_frame))
                action_tensor = postprocessor(action_tensor)
            requested = make_robot_action(action_tensor, metadata.features)
            applied = robot.send_action(requested)
            inference_time = time.perf_counter() - loop_start
            if inference_time > 1 / args.fps:
                overruns += 1

            if step % args.fps == 0:
                max_clip = max(abs(requested[name] - applied[name]) for name in requested)
                # action names 已经是 joint1.pos 形式，不能再次追加 .pos。
                state = _ordered_state(observation, metadata.features["action"]["names"])
                print(
                    f"[部署] t={time.perf_counter() - start:5.1f}s "
                    f"推理={inference_time * 1000:5.1f}ms 最大限幅={max_clip:.4f}rad "
                    f"state={[round(value, 3) for value in state]}"
                )
            step += 1
            time.sleep(max(1 / args.fps - inference_time, 0.0))
        print(f"[部署] 正常结束：steps={step}，周期超时={overruns}")
    except BaseException:
        robot.emergency_stop()
        raise
    finally:
        keyboard.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
