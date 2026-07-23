"""移动到指定 TCP 位姿，方便人工测量真实位置。

用途：
  用于让真机移动到一个指定 TCP 位姿，并可保持使能停在该位置。
  真实位置由人工用尺子测量，本脚本不录入测量值、不计算误差。
  本脚本只控制机械臂 arm 组，不打开相机、不加载 YOLO/SAM/GraspNet、不控制夹爪。

运行示例：
  cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp

  # 移动到 config/default.yaml 的 robot.ready_pose
  ./scripts/run_move_pose_error_test.sh --use-ready

  # 手动指定 TCP 位姿，单位：米 / 弧度
  ./scripts/run_move_pose_error_test.sh --x 0.20 --y 0.00 --z 0.20 --roll 0.0 --pitch 0.7 --yaw 0.0

  # 交互模式：启动一次后连续输入多个位姿，输入 q 退出 加了补偿
  ./scripts/run_move_pose_error_test.sh --interactive
   
  #不加补偿
  ./scripts/run_move_pose_error_test.sh --interactive --raw

安全说明：
  运行前确认工作空间清空、急停可用、串口设备正确。
  正常退出时脚本会先回 home，再断开并失能。
  如果需要保持在目标位姿观察，请显式加 --keep-enabled。该模式下按 Ctrl+C 不会
  自动回 home 或失能，观察结束后请再运行 run_home_disable.sh。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.robot.grasp_driver import selected_arm_config  # noqa: E402
from reBotArm_control_py.actuator import RebotArm  # noqa: E402
from reBotArm_control_py.controllers import RebotArmEndPose  # noqa: E402
from utils.transforms import apply_execution_compensation_to_pose  # noqa: E402

PoseTarget = tuple[float, float, float, float, float, float, float]


def parse_args() -> argparse.Namespace:
    """解析目标 TCP 位姿、补偿选项和退出时的安全回零参数。"""
    # 单次模式从 --x/--y/--z 读取目标；--interactive 则复用一次硬件连接连续输入。
    parser = argparse.ArgumentParser(description="Move real reBotArm TCP to a target pose")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--use-ready", action="store_true", help="使用 config/default.yaml 的 robot.ready_pose")
    parser.add_argument("--interactive", action="store_true", help="交互模式：启动一次后连续输入多个 TCP 位姿")
    parser.add_argument("--x", type=float, help="目标 TCP x，单位米")
    parser.add_argument("--y", type=float, help="目标 TCP y，单位米")
    parser.add_argument("--z", type=float, help="目标 TCP z，单位米")
    parser.add_argument("--roll", type=float, default=0.0, help="目标 TCP roll，单位弧度")
    parser.add_argument("--pitch", type=float, default=0.7, help="目标 TCP pitch，单位弧度")
    parser.add_argument("--yaw", type=float, default=0.0, help="目标 TCP yaw，单位弧度")
    parser.add_argument("--duration", type=float, default=None, help="运动时长，单位秒；默认读取 ready_pose.duration 或 3.0")
    parser.add_argument("--settle", type=float, default=0.8, help="运动结束后的稳定等待时间，单位秒")
    parser.add_argument("--keep-enabled", action="store_true", help="测完后保持使能和控制循环，便于观察实际位置")
    parser.add_argument("--apply-compensation", dest="apply_compensation", action="store_true", help="按 default.yaml 的 execution_compensation_* 修正后再发送")
    parser.add_argument("--raw", dest="apply_compensation", action="store_false", help="不使用 execution_compensation_*，直接发送输入位姿")
    parser.set_defaults(apply_compensation=False)
    parser.add_argument("--home-max-vel", type=float, default=0.35, help="正常退出回 home 的最大关节速度")
    parser.add_argument("--home-timeout", type=float, default=25.0, help="正常退出回 home 的超时时间，单位秒")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    """读取 YAML 配置，并要求顶层为字典。"""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} 顶层必须是 YAML mapping")
    return data


def ready_target(args: argparse.Namespace, cfg: dict[str, Any]) -> PoseTarget:
    """从配置读取 ready_pose，生成目标位姿。"""
    ready = cfg.get("robot", {}).get("ready_pose", {})
    return (
        float(ready.get("x", 0.25)),
        float(ready.get("y", 0.0)),
        float(ready.get("z", 0.35)),
        float(ready.get("roll", 0.0)),
        float(ready.get("pitch", 1.2)),
        float(ready.get("yaw", 0.0)),
        float(args.duration if args.duration is not None else ready.get("duration", 3.0)),
    )


def target_from_args(args: argparse.Namespace, cfg: dict[str, Any]) -> PoseTarget:
    """从命令行或 ready_pose 生成目标位姿。"""
    if args.use_ready:
        return ready_target(args, cfg)

    # 位置三项必须同时提供；姿态和时长已有命令行默认值。
    missing = [name for name in ("x", "y", "z") if getattr(args, name) is None]
    if missing:
        raise ValueError(f"缺少目标位置参数: {', '.join('--' + name for name in missing)}，或使用 --use-ready")
    return (
        float(args.x),
        float(args.y),
        float(args.z),
        float(args.roll),
        float(args.pitch),
        float(args.yaw),
        float(args.duration if args.duration is not None else 3.0),
    )


def wait_motion(controller: RebotArmEndPose, duration: float, settle: float) -> None:
    """等待 SDK 轨迹线程结束，再额外等待机械臂稳定。"""
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + settle + 2.0)
    else:
        time.sleep(duration)
    time.sleep(max(0.0, settle))


def command_target(target: PoseTarget, cfg: dict[str, Any], apply_compensation: bool) -> PoseTarget:
    """生成实际发给机械臂的位姿。

    --apply-compensation 时只修正 x/y/z，roll/pitch/yaw 和 duration 保持不变。
    这里复用真实抓取同一套 execution_compensation_*，便于验证补偿后落点。
    """
    if not apply_compensation:
        return target

    # 补偿函数只接收 6D 位姿，因此先暂存第七项 duration，处理后再拼回。
    x, y, z, roll, pitch, yaw, duration = target
    x2, y2, z2, roll2, pitch2, yaw2 = apply_execution_compensation_to_pose(
        (x, y, z, roll, pitch, yaw),
        cfg,
    )
    return (x2, y2, z2, roll2, pitch2, yaw2, duration)


def move_to_target(
    controller: RebotArmEndPose,
    target: PoseTarget,
    cfg: dict[str, Any],
    settle: float,
    apply_compensation: bool,
) -> bool:
    """移动到一个 TCP 位姿；成功返回 True。"""
    # target 是用户输入的原始值；command 是应用可选补偿后真正下发的值。
    command = command_target(target, cfg, apply_compensation)
    x0, y0, z0, roll0, pitch0, yaw0, _ = target
    x, y, z, roll, pitch, yaw, duration = command
    if apply_compensation:
        print(
            f"[MoveTest] target raw xyz=({x0:+.4f},{y0:+.4f},{z0:+.4f}) "
            f"rpy=({roll0:+.4f},{pitch0:+.4f},{yaw0:+.4f})"
        )
        print(
            f"[MoveTest] compensation delta xyz=({x - x0:+.4f},{y - y0:+.4f},{z - z0:+.4f})"
        )
    print(
        f"[MoveTest] move_to xyz=({x:+.4f},{y:+.4f},{z:+.4f}) "
        f"rpy=({roll:+.4f},{pitch:+.4f},{yaw:+.4f}), duration={duration:.2f}s"
    )
    # move_to_traj 返回 False 通常表示 IK 或轨迹规划未得到可执行结果，此时不等待。
    ok = controller.move_to_traj(x, y, z, roll, pitch, yaw, duration=duration)
    if not ok:
        print("[MoveTest] move_to_traj failed: IK or trajectory planning failed")
        return False
    wait_motion(controller, duration, settle)
    print("[MoveTest] reached target pose command; measure the real TCP position manually.")
    return True


def parse_interactive_target(line: str, args: argparse.Namespace, cfg: dict[str, Any]) -> PoseTarget | None:
    """解析交互输入。

    支持：
      - ready：移动到 config/default.yaml 的 robot.ready_pose；
      - x y z roll pitch yaw：六个数，duration 用 --duration 或 3.0；
      - x y z roll pitch yaw duration：七个数，最后一个是本次运动时长；
      - q / quit / exit：退出。
    """
    text = line.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"q", "quit", "exit"}:
        raise EOFError
    if lowered == "ready":
        return ready_target(args, cfg)

    # 同时接受空格和逗号分隔，最终都整理成六个或七个字段。
    parts = text.replace(",", " ").split()
    if len(parts) not in {6, 7}:
        print("[MoveTest] 输入格式: x y z roll pitch yaw [duration]，或 ready，或 q")
        return None
    try:
        values = [float(part) for part in parts]
    except ValueError:
        print("[MoveTest] 输入包含非数字，请重新输入")
        return None
    duration = values[6] if len(values) == 7 else float(args.duration if args.duration is not None else 3.0)
    return (values[0], values[1], values[2], values[3], values[4], values[5], duration)


def interactive_loop(controller: RebotArmEndPose, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """连续输入多个 TCP 位姿，复用同一个 SDK 连接。"""
    print("\n[MoveTest] interactive mode")
    print("[MoveTest] 输入: x y z roll pitch yaw [duration]")
    print("[MoveTest] 也可以输入 ready；输入 q 回 home 后断开失能。")
    # 每轮只解析和执行一个目标；空行或格式错误会直接进入下一轮。
    while True:
        try:
            line = input("pose> ")
            target = parse_interactive_target(line, args, cfg)
        except EOFError:
            print("[MoveTest] exit interactive mode")
            return
        if target is None:
            continue
        move_to_target(controller, target, cfg, float(args.settle), bool(args.apply_compensation))


def main() -> int:
    """连接真机并执行单次或交互式 TCP 目标位置误差测试。"""
    args = parse_args()
    cfg = load_config(args.config)
    target = None if args.interactive else target_from_args(args, cfg)

    selected = selected_arm_config(cfg.get("robot", {}).get("repo_root"))
    arm = RebotArm()
    controller = RebotArmEndPose(arm, arm_control_mode=selected.controller_mode)

    # 这里只测 arm 末端位姿，不需要切换/使能夹爪，避免额外串口写寄存器影响测试。
    controller._has_gripper = False
    skip_home_disconnect = False

    try:
        print(f"[MoveTest] controller mode={selected.controller_mode}")
        controller.start()

        if args.interactive:
            interactive_loop(controller, args, cfg)
            return 0

        if target is None or not move_to_target(controller, target, cfg, float(args.settle), bool(args.apply_compensation)):
            return 2

        if args.keep_enabled:
            print("\n[MoveTest] keep-enabled=true，保持控制循环和使能，机械臂会停在目标位姿。")
            print("[MoveTest] 量完后按 Ctrl+C 只退出脚本，不自动回 home 或失能。")
            print("[MoveTest] 观察结束后请运行 ./scripts/run_home_disable.sh。")
            while True:
                time.sleep(1.0)
        return 0
    except KeyboardInterrupt:
        print("\n[MoveTest] interrupted")
        if args.keep_enabled and getattr(controller, "_running", False):
            # 人工量尺时，用户需要机械臂停在目标位姿继续保持。
            # 因此 keep-enabled 模式下不在 Ctrl+C 后回 home 或断开失能。
            controller._running = False
            skip_home_disconnect = True
            print("[MoveTest] keep-enabled 模式：不自动断开/失能，机械臂保持当前位置。")
            print("[MoveTest] 量完后运行 ./scripts/run_home_disable.sh。")
        return 130
    finally:
        if getattr(controller, "_running", False) and not skip_home_disconnect:
            print("[MoveTest] safe_home before exit")
            controller.safe_home(max_vel=float(args.home_max_vel), timeout=float(args.home_timeout))
            print("[MoveTest] disconnect and disable")
            arm.disconnect()
            controller._running = False


if __name__ == "__main__":
    raise SystemExit(main())
