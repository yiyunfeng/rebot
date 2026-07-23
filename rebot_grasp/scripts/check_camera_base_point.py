#!/usr/bin/env python3
"""检查相机点云/手眼变换到 base 坐标的误差。

用途：
  1. 固定机械臂，不发送任何运动命令；
  2. 打开 RGB-D 相机画面；
  3. 鼠标左键点击桌面上的固定点；
  4. 打印该点在 camera 坐标系和 base 坐标系下的位置。

用法：
    cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp
    ./scripts/run_check_camera_base_point.sh

操作：
    鼠标左键：打印点击点坐标
    F：拟合已点击桌面点的 base 平面
    C：清空已点击点
    R：刷新当前 TCP 位姿
    Q/Esc：退出

说明：
  run_check_camera_base_point.sh 默认只读当前姿态，不发送任何运动命令；
  如确实需要先移动到 config/default.yaml 的 robot.ready_pose，运行时加 --move-ready。
  点选检查过程只读取相机和当前关节状态，不开合夹爪。
  输出的 base 坐标用于和尺子测得的桌面实际坐标对比，判断手眼/深度误差。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drivers.camera import make_camera  # noqa: E402
from drivers.robot.grasp_driver import ensure_rebot_sdk_in_syspath, selected_arm_config  # noqa: E402
from utils.camera_utils import (  # noqa: E402
    compose_cam_to_base_transform,
    load_config,
    load_hand_eye,
)
from utils.ordinary_grasp import get_depth_mm  # noqa: E402


def calibration_path(cam_type: str) -> Path:
    """按相机类型生成该设备默认的手眼标定文件路径。"""
    return PROJECT_ROOT / "config" / "calibration" / str(cam_type).lower() / "hand_eye.npz"


@dataclass
class ClickState:
    """记录最近一次鼠标点击。"""

    u: int = -1
    v: int = -1
    pending: bool = False


def parse_args() -> argparse.Namespace:
    """解析配置、深度采样范围和机械臂是否先移动待机位。"""
    parser = argparse.ArgumentParser(description="Check RGB-D point transformed to robot base")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--roi-size", type=int, default=7, help="点击点周围取深度的 ROI 尺寸")
    parser.add_argument("--repo-root", default=None, help="reBotArm_control_py 仓库路径；默认读取配置")
    parser.add_argument("--move-ready", dest="move_ready", action="store_true", help="启动后先移动到 robot.ready_pose")
    parser.add_argument("--no-move-ready", dest="move_ready", action="store_false", help="不移动机械臂，只读取当前位姿")
    parser.set_defaults(move_ready=False)
    return parser.parse_args()


def backproject_pixel(u: int, v: int, z_m: float, K: np.ndarray) -> np.ndarray:
    """把像素点和深度反投影到相机坐标系。"""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (float(u) - cx) * z_m / fx
    y = (float(v) - cy) * z_m / fy
    return np.array([x, y, z_m], dtype=np.float64)


class TcpReader:
    """只读当前 TCP 位姿，不启动 SDK 控制循环。

    这里不使用 GraspDriver.start()，因为 start() 会配置控制模式并接管夹爪；
    本工具只需要当前关节角和 FK，避免对真实机械臂产生额外动作。
    """

    def __init__(self, cfg: dict, repo_root: Optional[str], arm: Any = None, connected: bool = False) -> None:
        """准备 FK 模型；可以复用已有连接，也可以自行建立只读连接。"""
        robot_cfg = cfg.get("robot") or {}
        repo = ensure_rebot_sdk_in_syspath(repo_root or robot_cfg.get("repo_root"))

        from reBotArm_control_py.kinematics import compute_fk, load_robot_model, pad_q_for_model

        # 没有外部 arm 时由 TcpReader 自己创建连接，并在 close() 中负责断开。
        if arm is None:
            from reBotArm_control_py.actuator import RebotArm

            self._arm = RebotArm()
            self._owns_connection = True
        else:
            self._arm = arm
            self._owns_connection = False
        self._compute_fk = compute_fk
        self._pad_q_for_model = pad_q_for_model
        self._model = load_robot_model()
        self._repo = repo
        self._connected = connected

    def connect(self) -> None:
        """连接硬件串口/CAN，只读关节反馈，不使能电机。"""
        if self._connected:
            return
        self._arm.connect()
        self._connected = True

    def close(self) -> None:
        """仅关闭由本对象自己建立的连接，避免误断开外部控制器。"""
        if self._connected and self._owns_connection:
            try:
                self._arm.disconnect()
            finally:
                self._connected = False

    def get_tcp_pose(self) -> np.ndarray:
        """读取当前关节角，计算 TCP->base 的 4x4 位姿矩阵。"""
        arm_group = self._arm.groups.get("arm")
        if arm_group is None:
            raise RuntimeError("hardware config missing groups.arm")
        n = arm_group.num_joints
        q_arm = self._arm.get_state(request_feedback=False)[0][:n]
        q = self._pad_q_for_model(self._model, q_arm, n)
        pos, rot, _ = self._compute_fk(self._model, q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T


def mouse_callback(event, x, y, _flags, param) -> None:
    """把鼠标左键位置标记为待处理，坐标计算留给主循环完成。"""
    state: ClickState = param
    if event == cv2.EVENT_LBUTTONDOWN:
        state.u = int(x)
        state.v = int(y)
        state.pending = True


def transform_camera_point(T_cam2base: np.ndarray, point_cam: np.ndarray) -> np.ndarray:
    """给相机三维点补齐齐次坐标 1，再变换到机器人基座坐标系。"""
    # 三维点补 1 后才能让 4×4 矩阵同时应用旋转和平移。
    point_h = np.array([point_cam[0], point_cam[1], point_cam[2], 1.0], dtype=np.float64)
    return (T_cam2base @ point_h)[:3]


def print_plane_fit(points: list[np.ndarray], label: str) -> None:
    """拟合桌面点 z = a*x + b*y + c，判断高度误差主要沿哪个方向变化。"""
    if len(points) < 3:
        print(f"[Plane] {label}: need at least 3 clicked points, got {len(points)}")
        return

    pts = np.asarray(points, dtype=np.float64)
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    a, b, c = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]
    z_pred = A @ np.array([a, b, c], dtype=np.float64)
    residual = pts[:, 2] - z_pred
    z_span = float(np.max(pts[:, 2]) - np.min(pts[:, 2]))
    normal = np.array([-a, -b, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tilt_deg = np.degrees(np.arctan(np.hypot(a, b)))
    # 小角度下，z=a*x+b*y 对应法向约为绕 base X 轴 +b、绕 base Y 轴 -a。
    roll_x_deg = np.degrees(np.arctan(b))
    pitch_y_deg = np.degrees(np.arctan(-a))

    print(f"\n[Plane] {label}, n={len(points)}")
    print(f"  z = {a:+.6f} * x + {b:+.6f} * y + {c:+.6f}")
    print(f"  dz/dx: {a * 100.0:+.2f} mm per 10cm x")
    print(f"  dz/dy: {b * 100.0:+.2f} mm per 10cm y")
    print(
        "  plane normal in base: "
        f"[{normal[0]:+.4f}, {normal[1]:+.4f}, {normal[2]:+.4f}], tilt={tilt_deg:.2f} deg"
    )
    print(
        "  approx correction to flatten table: "
        f"base roll_x={-roll_x_deg:+.2f} deg, base pitch_y={-pitch_y_deg:+.2f} deg"
    )
    print(f"  clicked z span: {z_span * 1000.0:.2f} mm")
    print(f"  residual std: {float(np.std(residual)) * 1000.0:.2f} mm")


def ready_pose_from_config(cfg: dict) -> tuple[float, float, float, float, float, float, float]:
    """读取拍摄桌面的预备位姿，单位：米 / 弧度 / 秒。"""
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


def wait_motion_done(controller: Any, duration: float) -> None:
    """等待 SDK 轨迹线程结束，再短暂停稳。"""
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + 2.0)
    else:
        time.sleep(duration)
    time.sleep(0.8)


def move_to_ready_pose(cfg: dict, repo_root: Optional[str]) -> tuple[Any, Any]:
    """先把真机移动到 ready_pose，让相机能拍到桌面。

    这里只控制 arm 组，不切换夹爪、不打开/闭合夹爪。
    返回同一个 arm/controller，后续读取 TCP 和退出回 home 时继续复用。
    """
    # 先把 SDK 路径加入 sys.path，再延迟导入硬件类，便于该脚本独立运行。
    robot_cfg = cfg.get("robot") or {}
    ensure_rebot_sdk_in_syspath(repo_root or robot_cfg.get("repo_root"))

    from reBotArm_control_py.actuator import RebotArm
    from reBotArm_control_py.controllers import RebotArmEndPose

    x, y, z, roll, pitch, yaw, duration = ready_pose_from_config(cfg)
    selected = selected_arm_config(repo_root or robot_cfg.get("repo_root"))
    arm = RebotArm()
    controller = RebotArmEndPose(arm, arm_control_mode=selected.controller_mode)
    # 点位检查不需要夹爪，关闭控制器内置夹爪操作，避免额外硬件动作。
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
    print("[Robot] ready_pose reached; opening camera check window")
    return arm, controller


def print_point(
    u: int,
    v: int,
    point_cam: np.ndarray,
    point_base_raw: np.ndarray,
    point_base_comp: np.ndarray,
    point_base_inv: np.ndarray,
) -> None:
    """打印点击点坐标，raw/comp 用于区分是否叠加 hand_eye_compensation_m。"""
    # 同时输出三条变换结果，便于直接判断误差来自补偿方向还是手眼矩阵方向。
    print("\n[Point]")
    print(f"  pixel: u={u}, v={v}")
    print(
        "  camera xyz: "
        f"x={point_cam[0]:+.4f}, y={point_cam[1]:+.4f}, z={point_cam[2]:+.4f} m"
    )
    print(
        "  base xyz raw hand-eye: "
        f"x={point_base_raw[0]:+.4f}, y={point_base_raw[1]:+.4f}, z={point_base_raw[2]:+.4f} m"
    )
    print(
        "  base xyz with hand_eye_compensation_m: "
        f"x={point_base_comp[0]:+.4f}, y={point_base_comp[1]:+.4f}, z={point_base_comp[2]:+.4f} m"
    )
    print(
        "  base xyz if hand_eye inverted: "
        f"x={point_base_inv[0]:+.4f}, y={point_base_inv[1]:+.4f}, z={point_base_inv[2]:+.4f} m"
    )
    print("  compare this base xyz with ruler/table measurement.")


def main() -> int:
    """显示 RGB-D 画面，并把用户点击的深度点转换到 base 坐标系。"""
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)
    cam_type = str((cfg.get("camera") or {}).get("type", "")).lower()
    # 此工具检查完整坐标链，所以手眼文件缺失或模式不对时不能继续。
    T_hand_eye, mode = load_hand_eye(PROJECT_ROOT, cam_type)
    if T_hand_eye is None:
        print(f"[ERROR] hand_eye.npz not found for camera type: {cam_type}")
        return 2
    if mode != "eye_in_hand":
        print(f"[ERROR] expected eye_in_hand calibration, got: {mode}")
        return 2
    hand_eye_path = calibration_path(cam_type)
    hand_eye_data = np.load(str(hand_eye_path), allow_pickle=False)
    has_samples = "samples_T_gripper2base" in hand_eye_data.files
    print(f"[Init] hand_eye file: {hand_eye_path}")
    print(f"[Init] hand_eye samples saved: {has_samples}")

    arm = None
    controller = None
    if args.move_ready:
        arm, controller = move_to_ready_pose(cfg, args.repo_root)

    tcp_reader = TcpReader(cfg, args.repo_root, arm=arm, connected=arm is not None)
    cam = make_camera(cfg)
    click = ClickState()
    # 三组点分别记录原始手眼、补偿后手眼和手眼矩阵取逆的对照结果。
    raw_points: list[np.ndarray] = []
    comp_points: list[np.ndarray] = []
    inv_points: list[np.ndarray] = []
    window = "camera_base_point_check"

    try:
        print("[Init] connect robot feedback only, no motion command will be sent")
        tcp_reader.connect()
        T_tcp2base = tcp_reader.get_tcp_pose()
        print(f"[Init] current TCP base xyz={T_tcp2base[:3, 3].tolist()}")

        print("[Init] open RGB-D camera")
        cam.open()
        cam.warm_up(10)
        K = cam.K.astype(np.float64)

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse_callback, click)
        print("[Keys] Left click=add point  F=fit plane  C=clear points  R=refresh TCP  Q/Esc=quit")

        while True:
            color, depth = cam.get_frame()
            if color is None or depth is None:
                continue

            view = color.copy()
            cv2.putText(
                view,
                "Left click: print base xyz | R: refresh TCP | Q: quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if click.u >= 0 and click.v >= 0:
                cv2.drawMarker(
                    view,
                    (click.u, click.v),
                    (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=18,
                    thickness=2,
                )
            cv2.imshow(window, view)

            if click.pending:
                click.pending = False
                z_mm = get_depth_mm(depth, click.u, click.v, roi_size=max(1, int(args.roi_size)))
                if z_mm <= 0.0 or not np.isfinite(z_mm):
                    print(f"[Point] invalid depth at pixel ({click.u}, {click.v})")
                    continue

                # 点击时刷新一次 TCP，避免机械臂姿态变化后仍用旧 T_tcp2base。
                T_tcp2base = tcp_reader.get_tcp_pose()
                point_cam = backproject_pixel(click.u, click.v, z_mm / 1000.0, K)
                T_cam2base_raw = T_tcp2base @ T_hand_eye
                T_cam2base_comp = compose_cam_to_base_transform(T_tcp2base, T_hand_eye, cfg)
                T_cam2base_inv = T_tcp2base @ np.linalg.inv(T_hand_eye)
                point_base_raw = transform_camera_point(T_cam2base_raw, point_cam)
                point_base_comp = transform_camera_point(T_cam2base_comp, point_cam)
                point_base_inv = transform_camera_point(T_cam2base_inv, point_cam)
                print_point(
                    click.u,
                    click.v,
                    point_cam,
                    point_base_raw,
                    point_base_comp,
                    point_base_inv,
                )
                raw_points.append(point_base_raw)
                comp_points.append(point_base_comp)
                inv_points.append(point_base_inv)
                print(f"  saved table samples: {len(raw_points)}")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("f"), ord("F")):
                print_plane_fit(raw_points, "raw hand-eye")
                print_plane_fit(comp_points, "with hand_eye_compensation_m")
                print_plane_fit(inv_points, "if hand_eye inverted")
            if key in (ord("c"), ord("C")):
                raw_points.clear()
                comp_points.clear()
                inv_points.clear()
                print("[Plane] cleared clicked points")
            if key in (ord("r"), ord("R")):
                T_tcp2base = tcp_reader.get_tcp_pose()
                print(f"[TCP] refreshed base xyz={T_tcp2base[:3, 3].tolist()}")

    finally:
        cam.close()
        tcp_reader.close()
        if controller is not None and getattr(controller, "_running", False):
            print("[Robot] safe_home before exit")
            controller.safe_home(max_vel=0.35, timeout=25.0)
            print("[Robot] disconnect and disable")
            arm.disconnect()
            controller._running = False
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
