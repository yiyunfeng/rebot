#!/usr/bin/env python3
"""读取 Real 感知候选并控制 B601-DM 完成一次人工确认的抓取放置循环。

感知进程只负责相机、YOLO 和 SAM；本进程独占机械臂串口。机械臂停在 ready
姿态时才接受新候选，随后执行：预抓取、力控夹取、抬起、返回 ready、竖直
放回原位置、再次返回 ready。每一轮开始前都需要操作者按 Enter 确认。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
GRASP_ROOT = REPO_ROOT / "rebot_grasp"
CONFIG_PATH = GRASP_ROOT / "config/default.yaml"
CANDIDATE_PATH = Path("/tmp/rebot_grasp_candidate.json")
READY_STABILIZE_SECONDS = 1.0
sys.path.insert(0, str(GRASP_ROOT))

from drivers.robot.grasp_driver import (  # noqa: E402
    GraspDriver,
    ensure_rebot_sdk_in_syspath,
    selected_arm_config,
    selected_hardware_yaml,
)
from utils.camera_utils import (  # noqa: E402
    compose_cam_to_base_transform,
    load_hand_eye,
)
from utils.transforms import (  # noqa: E402
    apply_execution_compensation_to_pose,
    mat4_to_pose6d,
    transform_grasp_pose_to_base,
)

# SDK 必须先加入 sys.path，之后才能导入真机控制类。
ensure_rebot_sdk_in_syspath()
from reBotArm_control_py.actuator import RebotArm  # noqa: E402
from reBotArm_control_py.controllers import RebotArmEndPose  # noqa: E402


def _wait_motion(controller: RebotArmEndPose, duration: float) -> None:
    """等待 SDK 的轨迹发送线程结束，并留出少量机械稳定时间。"""
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + 2.6)
        if thread.is_alive():
            raise TimeoutError(f"robot trajectory exceeded {duration + 2.6:.1f}s")
    else:
        time.sleep(duration + 0.6)


def _move_pose(
    controller: RebotArmEndPose,
    pose: tuple[float, ...],
    duration: float,
    name: str,
) -> None:
    """规划并执行一个 TCP 轨迹；IK 失败时立即终止本轮而不是继续运动。"""
    x, y, z, roll, pitch, yaw = pose
    print(
        f"[Real] {name} xyz=({x:+.3f},{y:+.3f},{z:+.3f}) "
        f"rpy=({roll:+.3f},{pitch:+.3f},{yaw:+.3f})"
    )
    if not controller.move_to_traj(x, y, z, roll, pitch, yaw, duration=duration):
        raise RuntimeError(f"{name} IK/trajectory failed")
    _wait_motion(controller, duration)


def _move_ready(controller: RebotArmEndPose, ready_cfg: dict) -> None:
    """移动到相机能够观察桌面的统一 ready_pose。"""
    duration = float(ready_cfg["duration"])
    pose = (
        float(ready_cfg["x"]),
        float(ready_cfg["y"]),
        float(ready_cfg["z"]),
        float(ready_cfg.get("roll", 0.0)),
        float(ready_cfg["pitch"]),
        float(ready_cfg.get("yaw", 0.0)),
    )
    _move_pose(controller, pose, duration, "ready")


def _read_new_candidate(after_timestamp: float) -> dict | None:
    """读取 ready 之后产生的最新 Real 候选，忽略旧文件和 Sim 候选。"""
    if not CANDIDATE_PATH.is_file():
        return None
    try:
        candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
        timestamp = float(candidate["timestamp"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    if candidate.get("source") != "real" or timestamp <= after_timestamp:
        return None
    return candidate


def _candidate_to_poses(
    candidate: dict,
    grasp_driver: GraspDriver,
    T_hand_eye: np.ndarray,
    cfg: dict,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """将相机候选转换为 base 下的抓取和竖直放置位姿。"""
    position_cam = np.asarray(candidate["position_m"], dtype=np.float64)
    tcp_rotation_cam = np.asarray(candidate["tcp_rotation"], dtype=np.float64)
    if position_cam.shape != (3,) or tcp_rotation_cam.shape != (3, 3):
        raise ValueError("candidate position/tcp_rotation shape is invalid")
    if not np.all(np.isfinite(position_cam)) or not np.all(np.isfinite(tcp_rotation_cam)):
        raise ValueError("candidate contains non-finite values")

    # 候选是在机械臂静止于 ready 时拍摄的；此刻读取 FK，与 eye-in-hand 标定
    # 组合成 camera -> base，之后才允许机械臂离开 ready。
    T_cam2base = compose_cam_to_base_transform(
        grasp_driver.get_tcp_pose(),
        T_hand_eye,
        cfg,
    )
    grasp_cfg = cfg["grasp_pipeline"]["grasp"]
    pregrasp_offset = float(grasp_cfg["pregrasp_offset_m"])
    insertion_depth = float(grasp_cfg["insertion_depth_m"])
    grasp_pose, pregrasp_pose = transform_grasp_pose_to_base(
        position_cam,
        tcp_rotation_cam,
        T_cam2base,
        pregrasp_offset,
        insertion_depth,
    )

    # 放置使用同一个物体位置，但强制 TCP +X 沿 base -Z，形成竖直下放。
    object_position_base = T_cam2base[:3, :3] @ position_cam + T_cam2base[:3, 3]
    tcp_rotation_base = T_cam2base[:3, :3] @ tcp_rotation_cam
    vertical_x = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    vertical_y = tcp_rotation_base[:, 1].copy()
    vertical_y[2] = 0.0
    if np.linalg.norm(vertical_y) < 1e-6:
        vertical_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    vertical_y /= np.linalg.norm(vertical_y)
    vertical_rotation = np.column_stack([vertical_x, vertical_y, np.cross(vertical_x, vertical_y)])

    place_position = object_position_base + vertical_x * insertion_depth
    place_pregrasp_position = place_position - vertical_x * pregrasp_offset
    T_place = np.eye(4, dtype=np.float64)
    T_place[:3, :3] = vertical_rotation
    T_place[:3, 3] = place_position
    T_place_pregrasp = T_place.copy()
    T_place_pregrasp[:3, 3] = place_pregrasp_position

    # 真机固有落点误差只修正最终下发位姿，不修改视觉和手眼标定数据。
    poses = tuple(
        apply_execution_compensation_to_pose(pose, cfg)
        for pose in (
            grasp_pose,
            pregrasp_pose,
            mat4_to_pose6d(T_place),
            mat4_to_pose6d(T_place_pregrasp),
        )
    )
    min_z = float(grasp_cfg.get("min_base_z_m", 0.0))
    for name, pose in zip(("grasp", "pregrasp", "place", "place_pregrasp"), poses):
        if not np.all(np.isfinite(pose)) or pose[2] < min_z:
            raise ValueError(f"unsafe {name} pose: z={pose[2]:.4f}m, minimum={min_z:.4f}m")
    return poses


def _execute_cycle(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    candidate: dict,
    poses: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]],
    ready_cfg: dict,
) -> bool:
    """执行一轮抓取、返回、竖直放回和再次返回。"""
    grasp_pose, pregrasp_pose, place_pose, place_pregrasp_pose = poses
    jaw_width = float(candidate["jaw_width_m"])
    if not np.isfinite(jaw_width) or not 0.0 < jaw_width <= GraspDriver.MAX_DISTANCE_M:
        raise ValueError(f"invalid candidate jaw_width_m={jaw_width}")
    open_width = float(np.clip(jaw_width + 0.006, 0.0, GraspDriver.MAX_DISTANCE_M))

    grasp_driver.open_gripper(open_width)
    _move_pose(controller, pregrasp_pose, 2.0, "pregrasp")
    _move_pose(controller, grasp_pose, 1.5, "grasp")

    print("[Real] close gripper with force control")
    grasp_ok = grasp_driver.grasp()
    _move_pose(controller, pregrasp_pose, 1.5, "retreat")
    _move_ready(controller, ready_cfg)
    if not grasp_ok:
        print("[Real] empty grasp; skip place")
        grasp_driver.open_gripper(open_width)
        return False

    _move_pose(controller, place_pregrasp_pose, 2.0, "place_pregrasp")
    _move_pose(controller, place_pose, 1.5, "place")
    print("[Real] release object")
    grasp_driver.open_gripper(open_width)
    _move_pose(controller, place_pregrasp_pose, 1.5, "place_retreat")
    _move_ready(controller, ready_cfg)
    return True


def main() -> int:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    robot_cfg = cfg["robot"]
    ready_cfg = robot_cfg["ready_pose"]
    camera_type = str(cfg["camera"]["type"]).lower()
    T_hand_eye, hand_eye_mode = load_hand_eye(GRASP_ROOT, camera_type)
    if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
        raise RuntimeError(f"missing eye_in_hand calibration for camera.type={camera_type}")
    if T_hand_eye.shape != (4, 4) or not np.all(np.isfinite(T_hand_eye)):
        raise RuntimeError("hand_eye.npz contains an invalid T_result")

    selected = selected_arm_config(robot_cfg.get("repo_root"))
    hardware_yaml = selected_hardware_yaml(robot_cfg.get("repo_root"))
    hardware_cfg = yaml.safe_load(hardware_yaml.read_text(encoding="utf-8"))
    channel = str(hardware_cfg["channel"])
    if selected.arm_type != "dm":
        raise RuntimeError(f"real grasp only supports B601-DM, got {selected.arm_type}")
    if not channel.startswith("/dev/tty"):
        raise RuntimeError(f"B601-DM requires a serial channel, got {channel}")
    if not Path(channel).exists():
        raise FileNotFoundError(f"DM serial channel does not exist: {channel}")

    print(f"[Safety] model=B601-DM channel={channel}")
    print(f"[Safety] hand-eye={camera_type}/hand_eye.npz mode={hand_eye_mode}")
    print("[Safety] no collision planner: clear the workspace and keep emergency stop ready")
    if os.environ.get("REBOT_REAL_CONFIRMED") != "1":
        confirmation = input("Type RUN REAL after checking limits, workspace and E-stop: ").strip()
        if confirmation != "RUN REAL":
            print("[Real] cancelled before connecting hardware")
            return 0

    rebotarm: RebotArm | None = None
    controller: RebotArmEndPose | None = None
    grasp_driver: GraspDriver | None = None
    try:
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
        grasp_driver = GraspDriver(
            rebotarm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        grasp_driver.start()
        _move_ready(controller, ready_cfg)
        # 丢弃移动期间及刚到位时的候选，等待腕部相机在 ready 稳定后重新积累三帧。
        ready_after = time.time() + READY_STABILIZE_SECONDS

        while True:
            candidate = _read_new_candidate(ready_after)
            if candidate is None:
                print("[Real] waiting for a stable candidate at ready pose", end="\r", flush=True)
                time.sleep(0.1)
                continue

            print(
                f"\n[Real] candidate={candidate['class_name']} "
                f"conf={float(candidate['confidence']):.2f} "
                f"camera_xyz={np.round(candidate['position_m'], 4).tolist()}"
            )
            command = input("Press Enter to execute this cycle, or type q to stop: ").strip().lower()
            if command == "q":
                break

            # 操作者确认期间可能产生了更新的稳定候选；优先采用最新一份。
            candidate = _read_new_candidate(ready_after) or candidate
            poses = _candidate_to_poses(candidate, grasp_driver, T_hand_eye, cfg)
            _execute_cycle(controller, grasp_driver, candidate, poses, ready_cfg)
            ready_after = time.time() + READY_STABILIZE_SECONDS
    finally:
        print("\n[Real] stopping: release gripper, safe home and disconnect")
        try:
            if grasp_driver is not None and controller is not None and controller._running:
                grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[Real] gripper cleanup failed: {exc}")
        try:
            if controller is not None and controller._running:
                controller.end()
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Real] robot cleanup failed: {exc}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[Real] interrupted")
        raise SystemExit(130)
