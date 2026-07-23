#!/usr/bin/env python3
"""检查 ArUco 在相机坐标系下的静态稳定性。

用法：
    cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp
    ./scripts/run_check_aruco_static.sh

说明：
    默认先移动到 config/default.yaml 的 robot.ready_pose，让相机能看到 ArUco；
    到位后机械臂保持静止，只采样 marker2camera 位姿。
    如需只开相机、不移动机械臂，运行时加 --no-move-ready。
    如果这里抖动很小，而手眼 marker2base 残差很大，问题就更偏向 FK/模型/TCP。
"""

from __future__ import annotations

import argparse
from collections import Counter
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.aruco_pose import ArUcoDetector  # noqa: E402
from drivers.camera import make_camera  # noqa: E402
from drivers.robot.grasp_driver import ensure_rebot_sdk_in_syspath, selected_arm_config  # noqa: E402
from utils.camera_utils import load_config  # noqa: E402
from utils.transforms import rotation_matrix_to_euler_zyx  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析采样数量、相机预热和是否移动待机位等参数。"""
    parser = argparse.ArgumentParser(description="Check static ArUco pose stability")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--samples", type=int, default=80, help="采集到多少个有效 ArUco 位姿后自动统计")
    parser.add_argument("--warmup", type=int, default=10, help="相机预热帧数")
    parser.add_argument("--repo-root", default=None, help="reBotArm_control_py 仓库路径；默认读取配置")
    parser.add_argument("--move-ready", dest="move_ready", action="store_true", help="启动后先移动到 robot.ready_pose")
    parser.add_argument("--no-move-ready", dest="move_ready", action="store_false", help="不移动机械臂，只打开相机检查")
    parser.set_defaults(move_ready=False)
    return parser.parse_args()


def ready_pose_from_config(cfg: dict) -> tuple[float, float, float, float, float, float, float]:
    """从配置读取 ``x/y/z/rpy/duration``，缺失字段使用保守默认值。"""
    ready = (cfg.get("robot") or {}).get("ready_pose") or {}
    return (
        float(ready.get("x", 0.2)),
        float(ready.get("y", 0.0)),
        float(ready.get("z", 0.2)),
        float(ready.get("roll", 0.0)),
        float(ready.get("pitch", 0.7)),
        float(ready.get("yaw", 0.0)),
        float(ready.get("duration", 3.0)),
    )


def wait_motion_done(controller, duration: float) -> None:
    """等待轨迹线程完成，并额外留出机械结构停止抖动的时间。"""
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + 2.0)
    else:
        time.sleep(duration)
    time.sleep(0.8)


def move_to_ready_pose(cfg: dict, repo_root: str | None):
    """先移动到拍摄 ArUco 的 ready_pose；不控制夹爪。"""
    robot_cfg = cfg.get("robot") or {}
    ensure_rebot_sdk_in_syspath(repo_root or robot_cfg.get("repo_root"))

    from reBotArm_control_py.actuator import RebotArm
    from reBotArm_control_py.controllers import RebotArmEndPose

    x, y, z, roll, pitch, yaw, duration = ready_pose_from_config(cfg)
    selected = selected_arm_config(repo_root or robot_cfg.get("repo_root"))
    arm = RebotArm()
    controller = RebotArmEndPose(arm, arm_control_mode=selected.controller_mode)
    # 此工具只移动六轴机械臂，不初始化夹爪，减少不必要的硬件写操作。
    controller._has_gripper = False

    print(f"[Robot] start controller mode={selected.controller_mode}")
    controller.start()
    print(
        "[Robot] move to ready_pose "
        f"xyz=({x:+.3f},{y:+.3f},{z:+.3f}) "
        f"rpy=({roll:+.3f},{pitch:+.3f},{yaw:+.3f}), duration={duration:.1f}s"
    )
    ok = controller.move_to_traj(x, y, z, roll, pitch, yaw, duration=duration)
    if not ok:
        raise RuntimeError("move_to_traj ready_pose failed")
    wait_motion_done(controller, duration)
    print("[Robot] ready_pose reached; start static ArUco check")
    return arm, controller


def print_stats(poses: list[np.ndarray]) -> None:
    """统计静止标记在相机系中的位置和姿态抖动。"""
    if not poses:
        print("[ArucoStatic] no valid marker pose")
        return

    # poses 形状为 (样本数, 4, 4)，平移在 [:3, 3]，旋转在 [:3, :3]。
    Ts = np.asarray(poses, dtype=np.float64)
    xyz = Ts[:, :3, 3]
    rpy = np.asarray([rotation_matrix_to_euler_zyx(T[:3, :3]) for T in Ts], dtype=np.float64)
    xyz_mean = np.mean(xyz, axis=0)
    xyz_median = np.median(xyz, axis=0)
    rpy_mean = np.mean(rpy, axis=0)
    xyz_err = xyz - xyz_mean
    rpy_err = rpy - rpy_mean
    # MAD 使用中位数，受偶发错误检测的影响比标准差小。
    xyz_mad = np.median(np.abs(xyz - xyz_median), axis=0)

    print(f"\n[ArucoStatic] valid samples={len(poses)}")
    print(
        "[ArucoStatic] camera xyz mean: "
        f"x={xyz_mean[0]:+.4f}, y={xyz_mean[1]:+.4f}, z={xyz_mean[2]:+.4f} m"
    )
    print(
        "[ArucoStatic] camera xyz std: "
        f"x={np.std(xyz[:, 0]) * 1000.0:.2f}mm, "
        f"y={np.std(xyz[:, 1]) * 1000.0:.2f}mm, "
        f"z={np.std(xyz[:, 2]) * 1000.0:.2f}mm, "
        f"norm_rms={np.sqrt(np.mean(np.sum(xyz_err * xyz_err, axis=1))) * 1000.0:.2f}mm"
    )
    print(
        "[ArucoStatic] camera xyz median/mad: "
        f"median=({xyz_median[0]:+.4f},{xyz_median[1]:+.4f},{xyz_median[2]:+.4f}) m, "
        f"mad=({xyz_mad[0] * 1000.0:.2f},{xyz_mad[1] * 1000.0:.2f},{xyz_mad[2] * 1000.0:.2f}) mm"
    )
    print(
        "[ArucoStatic] camera xyz min/max/span: "
        f"x=({np.min(xyz[:,0]):+.4f},{np.max(xyz[:,0]):+.4f}, span={(np.ptp(xyz[:,0]) * 1000.0):.2f}mm), "
        f"y=({np.min(xyz[:,1]):+.4f},{np.max(xyz[:,1]):+.4f}, span={(np.ptp(xyz[:,1]) * 1000.0):.2f}mm), "
        f"z=({np.min(xyz[:,2]):+.4f},{np.max(xyz[:,2]):+.4f}, span={(np.ptp(xyz[:,2]) * 1000.0):.2f}mm)"
    )
    print(
        "[ArucoStatic] camera rpy mean: "
        f"roll={rpy_mean[0]:+.4f}, pitch={rpy_mean[1]:+.4f}, yaw={rpy_mean[2]:+.4f} rad"
    )
    print(
        "[ArucoStatic] camera rpy std: "
        f"roll={np.std(rpy_err[:, 0]):.5f}, "
        f"pitch={np.std(rpy_err[:, 1]):.5f}, "
        f"yaw={np.std(rpy_err[:, 2]):.5f} rad"
    )


def duplicate_id_text(poses) -> str:
    """返回同一帧内重复 ArUco ID 的说明；无重复时返回空字符串。"""
    counts = Counter(pose.id for pose in poses)
    duplicates = {mid: count for mid, count in counts.items() if count > 1}
    if not duplicates:
        return ""
    return ", ".join(f"id={mid} x{count}" for mid, count in sorted(duplicates.items()))


def main() -> int:
    """保持机械臂静止采集多帧 ArUco 位姿，并输出稳定性统计。"""
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)
    aruco_cfg = cfg.get("calibration", {}).get("aruco", {})
    # marker_length_m 必须是打印标记黑色外框的真实边长，错误会直接造成距离比例错误。
    detector = ArUcoDetector(
        marker_length_m=float(aruco_cfg.get("marker_length_m", 0.03)),
        aruco_dict_id=int(aruco_cfg.get("dict_id", 0)),
        target_marker_id=aruco_cfg.get("target_marker_id", 0),
    )

    arm = None
    controller = None
    if args.move_ready:
        arm, controller = move_to_ready_pose(cfg, args.repo_root)

    cam = make_camera(cfg)
    window = "aruco_static_check"
    poses: list[np.ndarray] = []
    last_print = 0.0
    duplicate_warned = False

    try:
        print("[ArucoStatic] open camera; keep robot still during sampling")
        cam.open()
        cam.warm_up(max(0, int(args.warmup)))
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("[Keys] Q/Esc=quit early")

        while len(poses) < int(args.samples):
            color, _depth = cam.get_frame()
            if color is None:
                continue

            # 先检查同一帧是否出现重复 ID；重复时无法判断是哪张物理标记，不采样。
            all_poses = detector.detect_all(color, cam.K, cam.D)
            dup_text = duplicate_id_text(all_poses)
            pose = None if dup_text else detector.detect(color, cam.K, cam.D)
            view = detector.draw_detected(color, cam.K, cam.D)
            if dup_text:
                cv2.putText(
                    view,
                    f"duplicate ArUco IDs: {dup_text}",
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                view,
                f"valid samples: {len(poses)}/{args.samples}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, view)

            if dup_text and not duplicate_warned:
                print(
                    "[ArucoStatic] detected duplicate marker IDs in one frame: "
                    f"{dup_text}. 同 ID 的多个物理 ArUco 不能混着做手眼标定；"
                    "请遮住其它同 ID 标记，或改用单张 10cm ArUco，或制作不同 ID 且布局已知的 board。"
                )
                duplicate_warned = True

            if pose is not None:
                # 只保存有效且无重复 ID 的 marker -> camera 位姿。
                poses.append(pose.T_marker2cam)
                now = time.monotonic()
                if now - last_print > 0.5:
                    t = pose.T_marker2cam[:3, 3]
                    print(
                        f"[ArucoStatic] sample {len(poses):03d}/{args.samples}: "
                        f"camera xyz=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f})"
                    )
                    last_print = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cam.close()
        if controller is not None and getattr(controller, "_running", False):
            print("[Robot] safe_home before exit")
            controller.safe_home(max_vel=0.35, timeout=25.0)
            print("[Robot] disconnect and disable")
            arm.disconnect()
            controller._running = False
        cv2.destroyAllWindows()

    print_stats(poses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
