"""
GraspNet 实时相机预览与抓取推理 Demo。

功能：
  通过 RGB-D 相机实时预览，按 G/空格键触发 GraspNet 对当前帧进行
  全场景抓取姿态推理，并在 Open3D 窗口中可视化抓取候选。

按键说明:
  G / 空格: 对当前 RGB-D 帧运行 GraspNet 推理，冻结画面显示结果.
  R: 恢复实时预览（解冻画面）.
  Q / Esc: 退出程序.

用法:
    cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp

    # DaBai DCW 默认从 config/default.yaml 读取 camera.type=orbbec_dabai_dcw
    ./scripts/run_graspnet_camera_demo.sh

    # 显式指定 DaBai DCW
    ./scripts/run_graspnet_camera_demo.sh --camera-type orbbec_dabai_dcw --open3d-grasps final

    # 其他相机调试示例
    ./scripts/run_graspnet_camera_demo.sh --camera-type orbbec_gemini2

注意:
  这个脚本只运行 GraspNet 视觉推理和 Open3D 可视化，不控制机械臂。
  请通过 run_graspnet_camera_demo.sh 启动，避免 ROS2 PYTHONPATH 污染直接相机环境。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 设置 matplotlib 临时目录和 Qt 字体目录
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

# 项目根目录和 GraspNet SDK 根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"


def _prepare_imports() -> None:
    """配置 Python 搜索路径：将项目根目录和 GraspNet SDK 子目录加入 sys.path。

    GraspNet SDK 包含多个子模块 (models, dataset, utils, pointnet2, graspnetAPI)，
    需要将它们全部加入搜索路径才能正常导入。
    """
    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    # GraspNet SDK 子目录列表
    graspnet_paths = [
        GRASPNET_ROOT,
        *(GRASPNET_ROOT / subdir for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI")),
    ]
    for path in reversed(graspnet_paths):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(1, path_str)


# 在导入 GraspNet 相关模块之前，先配置好搜索路径
_prepare_imports()

import utils.graspnet_utils as graspnet_utils  # noqa: E402
from drivers.camera import make_camera  # noqa: E402
from utils.camera_utils import configure_camera, load_config  # noqa: E402
from utils.yolo_utils import detect_objects, load_yolo  # noqa: E402


def valid_depth_count(depth_mm: np.ndarray, min_depth: float, max_depth: float) -> int:
    """统计 GraspNet 深度范围内的有效深度像素数量。

    GraspNet 输入点云只使用 min_depth/max_depth 之间的深度。如果这里为 0，
    后面构建点云时一定会报 "No valid depth pixels"。
    """
    if depth_mm is None:
        return 0
    depth = depth_mm.astype(np.uint16, copy=False)
    min_mm = int(max(0.0, min_depth) * 1000.0)
    max_mm = int(max_depth * 1000.0)
    return int(((depth > min_mm) & (depth < max_mm)).sum())


def capture_valid_graspnet_frame(cam, min_depth: float, max_depth: float, retries: int = 12):
    """为 GraspNet 抓一帧有有效深度的 RGB-D。

    DaBai DCW 不支持硬件帧同步，按键触发瞬间偶尔会拿到深度全 0 或缺失的帧。
    这里在推理前做一次轻量重试，只用于视觉推理，不控制机械臂。
    """
    # 重试期间保存有效深度最多的一帧；即使一直没有完全有效帧，也能返回最佳诊断数据。
    best_color = None
    best_depth = None
    best_count = 0
    for _ in range(max(1, int(retries))):
        color, depth = cam.get_frame()
        if color is None or depth is None:
            continue
        count = valid_depth_count(depth, min_depth, max_depth)
        if count > best_count:
            best_color, best_depth, best_count = color, depth, count
        # GraspNet 至少需要一个处于深度范围内的像素，满足后立即停止重试。
        if count > 0:
            return color, depth, count
    return best_color, best_depth, best_count


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    返回：
        包含所有命令行参数值的 Namespace 对象.
    """
    # 这里只提供调试时确实需要覆盖的相机和可视化参数，其余继续读取 YAML。
    parser = argparse.ArgumentParser(description="Minimal live full-scene GraspNet camera demo")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "default.yaml"))
    parser.add_argument("--camera-type", choices=("orbbec_dabai_dcw", "orbbec_gemini2", "realsense_d435i", "realsense_d405"), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--open3d-grasps",
        choices=("final", "bbox", "pre-bbox"),
        default="final",
        help="Open3D 中显示的抓取集：final（最终抓取）, bbox（框内抓取）, 或 pre-bbox（粗抓取）",
    )
    return parser.parse_args()


def main() -> None:
    """主循环：实时相机预览 + GraspNet 按需推理。

    完整流程：
      1. 加载配置，构建 GraspNet 网络和 YOLO 目标检测模型.
      2. 打开 RGB-D 相机，预热并读取相机内参.
      3. 实时预览模式下，YOLO 跳帧检测目标（每隔 infer_every 帧）.
      4. 按 G/空格键：冻结当前画面，对当前 RGB-D 帧运行 GraspNet 推理.
         推理流程：点云生成 -> 抓取候选生成 -> 碰撞检测 -> NMS 排序.
      5. 推理完成后在 2D 图像上标注最佳抓取 + 在 Open3D 窗口中可视化 3D 抓取.
      6. 按 R 键恢复实时预览，按 Q/Esc 退出.
    """
    args = parse_args()
    # 加载并合并相机配置参数
    cfg = configure_camera(load_config(args.config), args)
    # 读取 GraspNet 相关配置
    graspnet_cfg = cfg.get("graspnet", {})
    checkpoint = graspnet_utils.resolve_checkpoint_path(graspnet_cfg.get("checkpoint", "checkpoint-rs.tar"))
    num_point = int(graspnet_cfg.get("num_point", 20000))
    collision_thresh = float(graspnet_cfg.get("collision_thresh", 0.01))
    min_depth = float(graspnet_cfg.get("min_depth", 0.05))
    max_depth = float(graspnet_cfg.get("max_depth", 2.0))
    top_k = int(graspnet_cfg.get("top_k", 50))
    target_class = graspnet_cfg.get("target_class")
    target_margin_px = int(graspnet_cfg.get("target_margin_px", 12))
    target_expand_ratio = float(graspnet_cfg.get("target_expand_ratio", 1.0))

    # 构建 GraspNet 网络模型
    net = graspnet_utils.build_net(str(checkpoint), graspnet_utils.DEFAULT_NUM_VIEW)
    # 加载 YOLO 目标检测模型（用于在 2D 图像上标注目标框）
    yolo_model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)
    # GraspNet demo 的实时预览主要用于取当前帧，YOLO 框只是辅助选目标。
    # 30fps 相机下若每 3 帧跑一次 YOLO，容易让 SDK 队列堆满并丢旧帧；
    # 因此这里把实时 YOLO 更新频率限制到至少每 15 帧一次。按 G 推理时
    # infer_frame 仍会对当前帧重新做目标筛选，不影响最终 GraspNet 结果。
    yolo_opts["infer_every"] = max(int(yolo_opts.get("infer_every", 3)), 15)
    # 创建相机实例
    cam = make_camera(cfg)
    # Open3D 3D 可视化窗口（按需创建）
    vis: Optional[graspnet_utils.Open3DGraspWindow] = None
    status = "warming up camera..."
    target_status = "target detector warming up..."
    frozen = False                        # 是否冻结画面（推理完成后冻结，直到按 R 恢复）
    last_display: Optional[np.ndarray] = None  # 推理完成的最后一帧（冻结时显示）
    last_live_display: Optional[np.ndarray] = None  # 实时预览最近一帧；丢帧时复用，避免窗口黑屏闪烁
    last_detections = []                  # 最近一次 YOLO 检测结果
    selected_target = None                # 当前选中的目标框
    frame_index = 0
    dropped_frames = 0
    window_name = "GraspNet Live Camera"

    print(
        "Using camera: "
        f"{cfg['camera']['type']} "
        f"{cfg['camera'].get('color_width')}x{cfg['camera'].get('color_height')}@{cfg['camera'].get('fps')}"
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(cfg["camera"].get("color_width", 1280)), int(cfg["camera"].get("color_height", 720)))

    try:
        # 打开相机并预热（丢弃前若干帧以稳定曝光和白平衡）
        cam.open()
        cam.warm_up(graspnet_utils.DEFAULT_WARMUP_FRAMES)
        print("Camera intrinsics:")
        print(cam.K)
        print("Press G or SPACE to infer current frame. Press Q or ESC to quit.")
        print(f"YOLO bbox post-filter mode: target={target_class or 'best detection'}")
        print(f"Live YOLO update interval: every {yolo_opts['infer_every']} frames")

        while True:
            # 获取对齐后的彩色图像和深度图像
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                dropped_frames += 1
                # DaBai DCW 不支持硬件 frame sync，真机上偶尔拿不到完整 RGB-D 对。
                # 这里不要刷新黑图，否则窗口会“一闪一闪”；有上一帧时继续显示上一帧，
                # 只在左上角提示正在等下一组有效 RGB-D 帧。
                if frozen and last_display is not None:
                    display = graspnet_utils.draw_status(
                        last_display.copy(),
                        f"waiting for RGB-D frame... dropped={dropped_frames}",
                        target_status,
                        frozen=True,
                    )
                elif last_live_display is not None:
                    display = graspnet_utils.draw_status(
                        last_live_display.copy(),
                        f"waiting for RGB-D frame... dropped={dropped_frames}",
                        target_status,
                    )
                else:
                    width = int(cfg["camera"].get("color_width", 640))
                    height = int(cfg["camera"].get("color_height", 360))
                    display = graspnet_utils.draw_status(
                        np.zeros((height, width, 3), dtype=np.uint8),
                        f"waiting for RGB-D frame... dropped={dropped_frames}",
                        target_status,
                    )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue

            dropped_frames = 0
            frame_index += 1
            # 非冻结模式：跳帧运行 YOLO 检测（每隔 infer_every 帧检测一次，避免每帧推理导致卡顿）
            if not frozen and yolo_model is not None and (frame_index == 1 or frame_index % int(yolo_opts["infer_every"]) == 0):
                try:
                    _, last_detections = detect_objects(yolo_model, color_bgr, yolo_opts)
                    # 根据 target_class 选择最佳目标框
                    selected_target = graspnet_utils.select_target(last_detections, target_class)
                    target_status = graspnet_utils.target_status_text(selected_target, last_detections, target_class)
                except Exception as exc:
                    last_detections = []
                    selected_target = None
                    target_status = f"YOLO failed: {exc}"

            # 根据冻结状态选择显示内容
            if frozen and last_display is not None:
                # 冻结模式：显示上次推理的结果帧
                display = graspnet_utils.draw_status(last_display, status, target_status, frozen=True)
            else:
                # 实时模式：叠加 YOLO 检测框
                display_base = color_bgr
                if yolo_model is not None:
                    display_base = graspnet_utils.draw_detections_overlay(color_bgr, last_detections, selected_target, target_class)
                display = graspnet_utils.draw_status(display_base, status, target_status)
                # 保存真正有效的实时帧。后续如果相机临时丢帧，继续显示这张图，
                # 画面不会被空图覆盖。
                last_live_display = display.copy()
            cv2.imshow(window_name, display)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                # R: 解冻，恢复实时预览
                frozen = False
                last_display = None
                status = "live preview"
                continue
            if key in (ord("g"), ord("G"), ord(" ")):
                # G/空格: 对当前帧运行 GraspNet 抓取推理
                try:
                    infer_color, infer_depth, valid_count = capture_valid_graspnet_frame(
                        cam,
                        min_depth,
                        max_depth,
                    )
                    if infer_color is None or infer_depth is None or valid_count <= 0:
                        raise RuntimeError(
                            "No valid RGB-D frame for GraspNet "
                            f"(valid_depth_pixels={valid_count}, range={min_depth:.3f}-{max_depth:.3f}m)"
                        )
                    print(
                        "[G] Capture and run GraspNet "
                        f"(valid_depth_pixels={valid_count}, range={min_depth:.3f}-{max_depth:.3f}m)"
                    )
                    result = graspnet_utils.infer_frame(
                        net,
                        infer_color,
                        infer_depth,
                        cam.K,
                        num_point=num_point,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        collision_thresh=collision_thresh,
                        voxel_size=graspnet_utils.DEFAULT_VOXEL_SIZE,
                        yolo_model=yolo_model,
                        yolo_opts=yolo_opts,
                        target_class=target_class,
                        target_margin_px=target_margin_px,
                        target_expand_ratio=target_expand_ratio,
                    )
                    status = result.status
                    target_status = result.target_status
                    last_detections = result.detections
                    selected_target = result.selected_target
                    print(status)
                    # 在 2D 图像上绘制检测框和最佳抓取 2D 投影
                    display_base = graspnet_utils.draw_detections_overlay(
                        infer_color,
                        last_detections,
                        selected_target,
                        target_class,
                    )
                    vis_grasps = graspnet_utils.visualization_grasps(result, args.open3d_grasps)
                    graspnet_utils.draw_grasp_projections(display_base, vis_grasps, cam.K, top_k=top_k)
                    graspnet_utils.draw_best_grasp_projection(display_base, result.best, cam.K)
                    last_display = display_base
                    frozen = True  # 推理完成后冻结画面

                    # 在 Open3D 窗口中可视化 3D 抓取候选
                    if vis is None and len(vis_grasps) > 0:
                        vis = graspnet_utils.Open3DGraspWindow("GraspNet Grasps", top_k)
                    if vis is not None:
                        vis.update(result.o3d_cloud, vis_grasps)
                        print(f"Open3D {args.open3d_grasps} candidates={len(vis_grasps)}")
                except Exception as exc:
                    status = f"inference failed: {exc}"
                    print(status)

            # 检查 Open3D 可视化窗口是否被关闭
            if vis is not None and not vis.poll():
                vis.close()
                vis = None
            # 检查 OpenCV 窗口是否被关闭
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # 清理资源
        cam.close()
        if vis is not None:
            vis.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
