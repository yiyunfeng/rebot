"""真机相机抓取主入口。

配置固定读取 config/default.yaml。
G 抓取，R 恢复预览，Q/Esc 退出。
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
# 允许直接执行 ``python scripts/main.py`` 时仍能导入项目内的 drivers 和 utils。
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.camera import make_camera
from drivers.robot.grasp_driver import GraspDriver, selected_arm_config
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from utils.camera_utils import compose_cam_to_base_transform, load_config, load_hand_eye
from utils.ordinary_grasp import (
    GraspPose,
    draw_grasp,
    estimate_grasps,
    select_best_grasp,
)
from utils.sam_utils import load_sam_refiner, render_sam_masks_window
from utils.transforms import (
    apply_execution_compensation_to_pose,
    canonicalize_parallel_gripper_tcp_rotation,
    rotation_matrix_to_euler_zyx,
    transform_grasp_pose_to_base,
)
from utils.yolo_utils import load_yolo

CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def _dispatch_graspnet_main() -> int:
    """动态加载 GraspNet 入口，避免普通模式启动时导入其重型依赖。"""
    # GraspNet 流程独立维护，main.py 只做入口分发。
    grasp_path = PROJECT_ROOT / "scripts" / "grasp.py"
    spec = importlib.util.spec_from_file_location("rebot_grasp_grasp_main", grasp_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 GraspNet 入口: {grasp_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


def _load_config() -> tuple[dict[str, Any], str]:
    """读取固定配置文件，并返回完整配置和标准化后的感知模式名。"""
    cfg = load_config(CONFIG_PATH)
    # strip/lower 让 YAML 中的 " SAM "、"Sam" 等写法都能统一识别为 "sam"。
    backend = str(cfg.get("perception", {}).get("backend", "obb")).strip().lower()
    return cfg, backend


def _wait_motion(controller: RebotArmEndPose, duration: float, extra: float = 0.6) -> None:
    """等待异步轨迹发送结束；无法取得线程时按预计时长保守等待。"""
    # RebotArmEndPose 轨迹发送在线程里执行；没有线程时退回固定等待。
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + extra + 2.0)
    else:
        time.sleep(duration + extra)


def _move_ready(controller: RebotArmEndPose, ready_cfg: dict[str, Any]) -> None:
    """把 TCP 移到配置中的统一待机位，并等待动作完成。"""
    # ready_pose 是抓取前后统一停靠位，来自 config/default.yaml。
    duration = float(ready_cfg.get("duration", 3.0))
    controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.25)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.35)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 1.2)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    _wait_motion(controller, duration)


def _cam_to_base(T_hand_eye: np.ndarray, grasp_driver: GraspDriver, cfg: dict[str, Any]) -> np.ndarray:
    """用当前 TCP 位姿和手眼标定，计算此刻的 camera -> base 变换。"""
    # eye-in-hand: 当前 TCP 位姿和手眼矩阵共同决定 camera -> base。
    # 相机装在末端并随机械臂运动，所以不能只在程序启动时计算一次。
    return compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)


def _execute_grasp(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
) -> bool:
    """按固定顺序执行一次普通抓取，返回夹爪是否检测到物体。"""
    # 动作顺序保持直线：开爪 -> 预抓取 -> 抓取 -> 闭爪 -> 回 ready -> 释放。
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d

    print(f"[Grasp] pregrasp  xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})  rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Grasp] grasp     xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})  rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")

    print("[Grasp] Open gripper")
    grasp_driver.open_gripper()

    print("[Grasp] Move to pregrasp")
    if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[Grasp] Pregrasp IK failed")
        return False
    _wait_motion(controller, 2.0)

    print("[Grasp] Move to grasp")
    if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[Grasp] Grasp IK failed")
        return False
    _wait_motion(controller, 1.5)

    print("[Grasp] Closing")
    ok = grasp_driver.grasp()
    print("[Grasp] Holding object" if ok else "[Grasp] Empty grasp")

    print("[Grasp] Return ready")
    _move_ready(controller, ready_cfg)

    print("[Grasp] Release at ready")
    grasp_driver.release_gripper()
    return ok


def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    best: Optional[GraspPose],
    status_text: str,
) -> np.ndarray:
    """在图像副本上绘制全部候选、当前最佳候选和运行状态。"""
    # 只负责画图，不参与抓取计算。
    display = image.copy()
    for grasp in grasps:
        draw_grasp(display, grasp)

    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    if best is not None:
        x_m, y_m, z_m = best.position.tolist()
        best_text = (
            f"best={best.class_name} conf={best.conf:.2f} "
            f"xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f}) jaw={best.jaw_width_m * 100:.1f}cm"
        )
        cv2.putText(
            display,
            best_text,
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 140),
            2,
        )
    return display


def _print_best_grasp(grasp: GraspPose) -> None:
    """打印最佳候选在相机坐标系中的位置、尺寸和姿态，供调试核对。"""
    # 相机坐标系下的原始抓取结果，便于和 base 坐标输出对照。
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(grasp.tcp_rotation)
    print("\n[G] Best grasp:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.3f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  jaw_width_m={grasp.jaw_width_m:.4f} object_length_m={grasp.object_length_m:.4f}")
    print(f"  position_xyz={grasp.position.tolist()}")
    print(f"  grasp_rpy={rotation_matrix_to_euler_zyx(grasp.rotation).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def main() -> int:
    """启动所选感知模式、相机和机械臂，并处理预览与按键抓取。"""
    # 加载 default.yaml，并根据 perception.backend 选择三种模式之一。
    cfg, backend = _load_config()
    print(f"[Mode] perception.backend={backend}")
    if backend == "graspnet":
        # GraspNet 的模型、候选筛选和可视化流程不同，交给独立入口处理。
        return _dispatch_graspnet_main()
    if backend not in {"obb", "sam"}:
        raise ValueError(f"不支持的 perception.backend={backend!r}")

    # ready_pose 是每次抓取前后的统一安全停靠位；配置缺失时使用默认值。
    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )
    # make_camera 根据 camera.type 创建 Orbbec 或 RealSense 的统一驱动对象。
    cam_cfg = cfg.get("camera", {})
    print(f"=== Camera: {cam_cfg.get('type')} ===")
    cam = make_camera(cfg)

    # 实时推理不是每帧执行；这些变量缓存上一次结果，供中间帧继续显示。
    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    last_sam_display: Optional[np.ndarray] = None
    # frozen=True 表示按 G 后冻结抓取快照；按 R 才恢复实时预览。
    frozen = False
    last_display: Optional[np.ndarray] = None
    # frame_index 控制隔多少帧做一次模型推理；fps_* 只用于计算显示帧率。
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    window_name = "Main - Ordinary Grasp"
    sam_window_name = "SAM - Mask Contours"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys]  G=grasp  R=resume  Q/ESC=quit\n")

    # 先设为 None，确保初始化中途失败时 finally 仍可判断哪些资源需要释放。
    controller: Optional[RebotArmEndPose] = None
    rebotarm: Optional[RebotArm] = None
    grasp_driver: Optional[GraspDriver] = None
    T_hand_eye: Optional[np.ndarray] = None
    yolo_opts: dict[str, Any] = {}
    sam_refiner = None
    robot_ready = False

    try:
        # 1. 相机和标定。
        cam.open()
        cam.warm_up(15)
        # K 是 3×3 相机内参矩阵，用于把像素和深度反投影为相机坐标。
        K = cam.K.astype(np.float32)

        cam_type = str(cam_cfg.get("type", "")).lower()
        # eye-in-hand 标定文件保存 camera -> gripper；缺失时仍允许预览，但禁止真机抓取。
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            print("[WARN] Hand-eye calibration unavailable; grasp execution disabled")
            T_hand_eye = None

        # 2. 感知模型。
        yolo_cfg = cfg.get("yolo", {})
        gp_cfg = cfg.get("grasp_pipeline", {})
        grasp_cfg = gp_cfg.get("grasp", {})

        # 这些参数只影响抓取点计算和推理频率，统一从 YAML 读取。
        model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
        pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
        insertion_depth_m = float(grasp_cfg.get("insertion_depth_m", 0.0))
        depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))
        infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))

        print(f"=== Load YOLO: {model_name} ===")
        model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)
        cfg_for_sam = cfg
        if str(cfg.get("perception", {}).get("backend", "")).lower() == "sam":
            # 选择 sam 模式就强制启用精分割；复制字典避免修改原始 cfg。
            sam_cfg = dict(cfg.get("sam", {}))
            sam_cfg["enabled"] = True
            cfg_for_sam = dict(cfg)
            cfg_for_sam["sam"] = sam_cfg
        sam_refiner = load_sam_refiner(cfg_for_sam, project_root=PROJECT_ROOT)
        if sam_refiner is not None:
            print("[SAM] enabled: YOLO box -> SAM mask refinement")
            cv2.namedWindow(sam_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(sam_window_name, 960, 540)

        # 3. 机械臂和夹爪。
        print("=== Init robot ===")
        # selected_arm_config 根据硬件 YAML 决定控制模式，防止给错误型号发送指令。
        selected = selected_arm_config(robot_cfg.get("repo_root"))
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
        grasp_driver = GraspDriver(
            rebotarm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        grasp_driver.start()
        robot_ready = True
        print(f"[Robot] mode: {selected.controller_mode}")

        print("[Robot] Move ready")
        _move_ready(controller, ready_cfg)

        # 4. 实时预览；按 G 时抓当前帧执行一次抓取。
        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                # 驱动短暂丢帧时等待下一帧，不把空数组送进模型。
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                # 每秒更新一次平均 FPS，避免每帧计算造成数字剧烈跳动。
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            if not frozen and (frame_index % infer_every == 0 or not last_results):
                # model.predict 返回 list[Results]；单张图通常只有 results[0]。
                last_results = model.predict(
                    color_bgr,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                # SAM 模式用 YOLO 框生成更精确的 mask；OBB 模式这里得到 None。
                sam_masks = sam_refiner.refine_results(last_results, color_bgr) if sam_refiner is not None else None
                if sam_refiner is not None:
                    last_sam_display = render_sam_masks_window(color_bgr, sam_masks, "SAM LIVE")
                # 把二维检测、深度图和内参组合成相机坐标系下的三维抓取候选。
                last_grasps = estimate_grasps(
                    last_results,
                    depth_mm,
                    K,
                    depth_quantile=depth_quantile,
                    mask_overrides=sam_masks,
                )

            status = f"{'FROZEN' if frozen else 'LIVE'} {fps_value:.1f}fps | G=grasp R=resume Q=quit"
            # 多个目标时先排除无效姿态，再选择 YOLO 置信度最高的候选用于显示。
            best_live = select_best_grasp(last_grasps)
            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            else:
                display = _render_display(color_bgr, last_grasps, best_live, status)

            cv2.imshow(window_name, display)
            if sam_refiner is not None and last_sam_display is not None:
                cv2.imshow(sam_window_name, last_sam_display)
            key = cv2.waitKey(1) & 0xFF
            # & 0xFF 保留低 8 位，使不同平台返回的按键码保持一致。
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                continue

            if key in (ord("g"), ord("G")):
                # 抓取时重新取一帧并重新推理，避免使用预览缓存中的旧目标位置。
                print("\n[G] Capture and estimate grasp")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] Frame capture failed")
                    continue

                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                # 快照必须重新分割和估计，不能复用实时预览中可能已经过期的 mask。
                snap_sam_masks = sam_refiner.refine_results(snap_results, snap_color) if sam_refiner is not None else None
                if sam_refiner is not None:
                    last_sam_display = render_sam_masks_window(snap_color, snap_sam_masks, "SAM SNAPSHOT")
                    cv2.imshow(sam_window_name, last_sam_display)
                    cv2.waitKey(1)
                    #处理sam识别的mask，计算物体位置，夹取的角度
                snap_grasps = estimate_grasps(
                    snap_results,
                    snap_depth,
                    K,
                    depth_quantile=depth_quantile,
                    mask_overrides=snap_sam_masks,
                )
                # 当前规则：从有效候选中选择检测置信度 conf 最高的物体。
                best = select_best_grasp(snap_grasps)
                if best is None:
                    print("[G] No valid grasp")
                    continue

                _print_best_grasp(best)

                snap_display = _render_display(snap_color, snap_grasps, best, "SNAPSHOT")
                frozen = True
                last_display = snap_display
                last_results = snap_results
                last_grasps = snap_grasps

                if T_hand_eye is None:
                    print("[G] Hand-eye calibration unavailable")
                    continue

                # 先打印未补偿的 base_link 结果，再只对最终发给机械臂的目标做执行补偿。
                # 此刻重新读取 TCP，因为 eye-in-hand 相机的 base 位姿随机械臂姿态改变。
                T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, cfg)
                object6d, _ = transform_grasp_pose_to_base(
                    best.position,
                    best.tcp_rotation,
                    T_cam2base,
                    0.0,
                    0.0,
                )
                print(
                    "[Base object] "
                    f"xyz=({object6d[0]:+.3f},{object6d[1]:+.3f},{object6d[2]:+.3f}) "
                    f"rpy=({object6d[3]:+.3f},{object6d[4]:+.3f},{object6d[5]:+.3f})"
                )
                # grasp6d 是最终夹取位姿；pre6d 沿工具接近轴后退一段距离，防止横向撞物。
                grasp6d, pre6d = transform_grasp_pose_to_base(
                    best.position,
                    best.tcp_rotation,
                    T_cam2base,
                    pregrasp_offset_m,
                    insertion_depth_m,
                )
                print(
                    "[Base raw] pregrasp "
                    f"xyz=({pre6d[0]:+.3f},{pre6d[1]:+.3f},{pre6d[2]:+.3f}) "
                    f"rpy=({pre6d[3]:+.3f},{pre6d[4]:+.3f},{pre6d[5]:+.3f})"
                )
                print(
                    "[Base raw] grasp    "
                    f"xyz=({grasp6d[0]:+.3f},{grasp6d[1]:+.3f},{grasp6d[2]:+.3f}) "
                    f"rpy=({grasp6d[3]:+.3f},{grasp6d[4]:+.3f},{grasp6d[5]:+.3f})"
                )
                # 标定/结构误差补偿只作用于最终执行目标，不修改前面打印的原始计算结果。
                grasp6d = apply_execution_compensation_to_pose(grasp6d, cfg)
                pre6d = apply_execution_compensation_to_pose(pre6d, cfg)
                _execute_grasp(
                    controller,
                    grasp_driver,
                    grasp6d,
                    pre6d,
                    ready_cfg,
                )

    finally:
        # 无论正常退出还是 Ctrl+C，都尽量释放夹爪、回 home 并断开。
        print("\n[Exit] Release gripper and home")
        try:
            if robot_ready and grasp_driver is not None and controller is not None and getattr(controller, "_running", False):
                grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            if controller is not None and getattr(controller, "_running", False):
                controller.end()
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            cam.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
