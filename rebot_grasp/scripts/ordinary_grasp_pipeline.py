"""普通抓取感知调试：相机预览、YOLO/SAM、OBB 抓取姿态显示。"""

import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import select
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (PROJECT_ROOT,):
    # 支持直接从 scripts/ 运行时导入项目内模块。
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from drivers.camera import make_camera
from utils.camera_utils import load_config
from utils.ordinary_grasp import draw_grasp, estimate_grasps, get_depth_mm, select_best_grasp
from utils.sam_utils import load_sam_refiner
from utils.transforms import canonicalize_parallel_gripper_tcp_rotation, rotation_matrix_to_euler_zyx


clicked_point = {"u": -1, "v": -1}


def mouse_callback(event, x, y, flags, param):
    """记录鼠标左键像素，主循环会用对应深度反投影三维坐标。"""
    del flags, param
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point["u"] = x
        clicked_point["v"] = y
        print(f"[Test] Clicked pixel: (u={x}, v={y})")


def print_best_grasp(grasp) -> None:
    """打印当前最佳候选在相机坐标系中的位置、尺寸和姿态。"""
    # 相机坐标系下的抓取结果，用来和主抓取程序输出对照。
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(grasp.tcp_rotation)
    print("\n[G] Best grasp:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.3f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  jaw_width_m={grasp.jaw_width_m:.4f} object_length_m={grasp.object_length_m:.4f}")
    print(f"  position_xyz={grasp.position.tolist()}")
    print(f"  grasp_rpy={rotation_matrix_to_euler_zyx(grasp.rotation).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def read_terminal_key() -> str:
    """非阻塞读取终端命令；没有输入或 stdin 不是终端时返回空字符串。"""
    if not sys.stdin.isatty():
        return ""
    # timeout=0 表示立即返回，不让终端输入阻塞相机预览循环。
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return ""
    return sys.stdin.readline().strip().lower()


def main():
    """启动仅感知调试流程：相机取帧、YOLO/SAM 推理和抓取姿态显示。"""
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 检测阈值、推理间隔、SAM 开关和深度分位数均从 default.yaml 读取。
    cfg = load_config(config_path)
    cam_type = str(cfg.get("camera", {}).get("type", "")).lower()
    yolo_cfg = cfg.get("yolo", {})
    det_cfg = cfg.get("detection", {})
    grasp_cfg = cfg.get("grasp_pipeline", {}).get("grasp", {})
    infer_every = max(1, int(cfg.get("grasp_pipeline", {}).get("infer_every_live", 3)))

    model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
    device = yolo_cfg.get("device", "cpu")
    use_world = bool(yolo_cfg.get("use_world", False))
    custom_classes = list(yolo_cfg.get("custom_classes", ["cup"]))
    conf_thres = float(det_cfg.get("conf_threshold", 0.25))
    iou_thres = float(det_cfg.get("iou_threshold", 0.45))
    depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))

    print("=== Init YOLO ===")
    model_path = models_dir / model_name
    print(f"Load model: {model_path}")
    model = YOLO(str(model_path))
    # YOLOE/World 支持运行时指定文本类别；普通封闭类别模型不能调用 set_classes。
    if use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower()):
        model.set_classes(custom_classes)
        print(f"Open-vocabulary classes: {custom_classes}")
    sam_refiner = load_sam_refiner(cfg, project_root=PROJECT_ROOT)
    if sam_refiner is not None:
        print("[SAM] enabled: YOLO box -> SAM mask refinement")

    print(f"\n=== Init camera: {cam_type} ===")
    cam = make_camera(cfg)
    cam.open()
    cam.warm_up(10)
    # K 中 fx/fy 是焦距，cx/cy 是主点，用于把点击像素反投影到相机三维坐标。
    K = cam.K.astype("float32")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    print(f"[Camera] fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    window_name = f"Ordinary Grasp Test ({cam_type})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)
    print("\n[Keys] Left click=sample depth  G=print best  Q=quit")
    print("[Terminal] You can also type g + Enter to print best grasp")
    print(f"[Perf] YOLO/SAM inference every {infer_every} frame(s)")

    frame_index = 0
    last_grasps = []
    try:
        while True:
            frame_index += 1
            color_image, depth_mm = cam.get_frame()
            if color_image is None or depth_mm is None:
                continue

            # YOLO/SAM 比相机取帧慢，中间帧复用上次抓取结果。
            if frame_index == 1 or frame_index % infer_every == 0:
                results = model.predict(
                    color_image,
                    verbose=False,
                    device=device,
                    conf=conf_thres,
                    iou=iou_thres,
                )

                # SAM 关闭时传入 None，estimate_grasps 会回退到 YOLO 自带 mask/检测框。
                sam_masks = sam_refiner.refine_results(results, color_image) if sam_refiner is not None else None
                last_grasps = estimate_grasps(
                    results,
                    depth_mm,
                    K,
                    depth_quantile=depth_quantile,
                    mask_overrides=sam_masks,
                )

            grasps = last_grasps
            for grasp in grasps:
                draw_grasp(color_image, grasp, show_pose_text=False)

            # 多物体时显示有效候选中 YOLO 置信度最高的一个。
            best = select_best_grasp(grasps)
            if best is not None:
                best_text = (
                    f"best={best.class_name} conf={best.conf:.2f} "
                    f"jaw={best.jaw_width_m * 100:.1f}cm  press G for pose"
                )
                cv2.putText(
                    color_image,
                    best_text,
                    (10, color_image.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (120, 255, 140),
                    2,
                )

            cu, cv_y = clicked_point["u"], clicked_point["v"]
            if cu >= 0 and cv_y >= 0:
                # 点击取点只用于检查深度和内参，不影响真正的抓取候选。
                z_mm = get_depth_mm(depth_mm, cu, cv_y, 5)
                if z_mm > 0:
                    z_m = z_mm / 1000.0
                    x_m = (cu - cx) * z_m / fx
                    y_m = (cv_y - cy) * z_m / fy
                    cv2.drawMarker(color_image, (cu, cv_y), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
                    label = f"TEST X:{x_m:.3f} Y:{y_m:.3f} Z:{z_m:.3f} m"
                    cv2.rectangle(color_image, (cu + 5, cv_y - 25), (cu + 320, cv_y + 5), (0, 0, 0), -1)
                    cv2.putText(
                        color_image,
                        label,
                        (cu + 8, cv_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 255),
                        2,
                    )

            title = f"{cam_type.upper()} | {model_name} | ordinary grasp"
            cv2.putText(color_image, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.imshow(window_name, color_image)

            key = cv2.waitKeyEx(1)
            terminal_key = read_terminal_key()
            should_print_best = key in (ord("g"), ord("G")) or terminal_key == "g"

            if key in (ord("q"), ord("Q"), 27) or terminal_key in ("q", "quit", "exit"):
                break
            if should_print_best:
                if best is None:
                    print("[G] No valid grasp yet. Move the object into view or wait for the next inference frame.")
                else:
                    print_best_grasp(best)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
