"""
3D 目标检测预览工具。

功能：
  使用 YOLO 模型检测目标，并通过深度图像反投影计算目标在相机坐标系下的 3D 坐标。
  支持鼠标左键点击采样任意像素点的深度值并显示其 3D 世界坐标。

按键说明:
  鼠标左键点击: 采样点击像素处的深度值，回投影为 3D 坐标并显示.
  Q / Esc: 退出程序.

用法:
    python scripts/object_detection.py
"""

import os
# 设置 Qt 字体目录，避免 OpenCV 中文显示问题
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import sys
import cv2
from pathlib import Path
from ultralytics import YOLO

# 将项目根目录添加到 Python 搜索路径，确保可以导入项目内模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.camera import make_camera
from utils.camera_utils import load_config
from utils.ordinary_grasp import get_depth_mm

# ==========================================
# 鼠标交互状态
# ==========================================
# 记录用户最近一次左键点击的像素坐标 (u, v)
clicked_point = {"u": -1, "v": -1}

def mouse_callback(event, x, y, flags, param):
    """鼠标回调：记录左键点击的像素坐标。

    参数：
        event: OpenCV 鼠标事件类型.
        x: 像素横坐标 u.
        y: 像素纵坐标 v.
        flags: 鼠标事件标志（未使用）.
        param: 用户自定义参数（未使用）.
    """
    # 回调只记录像素，不在 GUI 回调中读取深度；主循环会使用同一帧深度完成计算。
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point["u"] = x
        clicked_point["v"] = y
        print(f"[Test] Clicked pixel: (u={x}, v={y})")

# ==========================================
# 主函数
# ==========================================
def main():
    """主循环：初始化 YOLO 模型和相机，进入实时检测与 3D 坐标显示循环。

    流程：
      1. 加载配置文件，初始化 YOLO 模型（支持开放式词汇检测）.
      2. 打开 RGB-D 相机，获取内参矩阵 K（fx, fy, cx, cy）.
      3. 每帧运行 YOLO 推理，对检测到的目标框中心点进行深度反投影，
         得到目标在相机坐标系下的 (X, Y, Z) 坐标（单位：米）.
      4. 响应鼠标左键点击，采样点击像素的深度并回投影显示.
    """
    config_path  = PROJECT_ROOT / "config" / "default.yaml"
    models_dir   = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 加载 YAML 配置文件
    cfg = load_config(config_path)

    # 读取相机类型
    cam_type = cfg.get("camera", {}).get("type", "").lower()

    # 读取 YOLO 相关配置参数
    yolo_cfg     = cfg.get("yolo", {})
    model_name   = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
    device       = yolo_cfg.get("device", "cpu")
    use_world    = yolo_cfg.get("use_world", False)
    custom_classes = yolo_cfg.get("custom_classes", ["person", "cup", "cell phone"])

    model_path = models_dir / model_name

    print(f"=== Init YOLO ===")
    print(f"Load model: {model_path}")
    # 加载 YOLO 模型
    model = YOLO(str(model_path))

    # 判断是否为开放式词汇模型：支持通过 set_classes 动态设置检测类别
    is_open_vocab = use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower())
    if is_open_vocab:
        print(f"Enable open vocabulary: {len(custom_classes)} classes")
        model.set_classes(custom_classes)

    print(f"YOLO ready on {device.upper()}")
    if "26" in model_name:
        print(f"[Info] YOLO26 end-to-end inference enabled.")

    # 初始化相机
    print(f"\n=== Init camera: {cam_type} ===")
    try:
        cam = make_camera(cfg)
    except ValueError as e:
        print(f"\n[Fatal] {e}")
        sys.exit(1)

    try:
        cam.open()
    except RuntimeError as e:
        print(f"\n[Fatal] {e}")
        sys.exit(1)

    # 读取相机内参矩阵 K（3x3），提取焦距和主点坐标
    K  = cam.K
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    print(f"[Camera] {cam_type} (fx:{fx:.2f}, cx:{cx:.2f})")

    print("\n[Keys] Left click=sample point  Q=quit")
    window_name = f"Unified 3D Vision ({cam_type})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # 注册鼠标回调以支持点击采样
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            # 获取对齐后的彩色图像 (BGR) 和深度图像 (毫米)
            color_image, depth_mm = cam.get_frame()
            if color_image is None:
                continue

            # YOLO 目标检测推理
            results = model.predict(color_image, verbose=False, device=device)

            # 遍历每个检测结果
            for r in results:
                # 遍历每个检测框
                for box in r.boxes:
                    # 解析检测框的像素坐标 (x1, y1) - 左上角, (x2, y2) - 右下角
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    # 类别 ID 和置信度
                    cls_id, conf = int(box.cls[0]), float(box.conf[0])
                    class_name = model.names[cls_id]

                    # 取检测框中心点作为目标像素坐标
                    u, v = (x1 + x2) // 2, (y1 + y2) // 2
                    x_m, y_m, z_m = 0.0, 0.0, 0.0
                    valid_depth = False

                    if depth_mm is not None:
                        # 在中心点周围 5x5 邻域内取有效深度值
                        # 深度反投影公式: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
                        z_mm = get_depth_mm(depth_mm, u, v, 5)
                        if z_mm > 0:
                            z_m = z_mm / 1000.0   # 毫米 -> 米
                            x_m = (u - cx) * z_m / fx
                            y_m = (v - cy) * z_m / fy
                            valid_depth = True

                    if valid_depth:
                        # 有有效深度：绘制绿色框 + 红色中心点 + 3D 坐标标签
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)
                        text_label  = f"{class_name} {conf:.2f}"
                        coord_label = f"X:{x_m:.2f} Y:{y_m:.2f} Z:{z_m:.2f} (m)"
                        # 绘制标签背景（黑色矩形）
                        bg_w = max(len(text_label), len(coord_label)) * 10
                        cv2.rectangle(color_image, (x1, y1 - 40), (x1 + bg_w, y1), (0, 0, 0), -1)
                        cv2.putText(color_image, text_label,  (x1 + 5, y1 - 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),   2)
                        cv2.putText(color_image, coord_label, (x1 + 5, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # 无有效深度：绘制橙色框 + "No Depth" 提示
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(color_image, f"{class_name} (No Depth)",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # 鼠标点击点的深度采样与 3D 坐标回投影
            cu, cv_y = clicked_point["u"], clicked_point["v"]
            if cu != -1 and cv_y != -1 and depth_mm is not None:
                h_dm, w_dm = depth_mm.shape
                # 检查点击坐标是否在深度图像范围内
                if 0 <= cu < w_dm and 0 <= cv_y < h_dm:
                    # 点击像素处采样深度值（毫米）
                    cz_mm = get_depth_mm(depth_mm, cu, cv_y)
                    if cz_mm > 0:
                        # 深度反投影：从像素坐标 (u, v) + 深度 Z 计算 3D 坐标 (X, Y, Z)
                        cz_m = cz_mm / 1000.0
                        cx_m = (cu - cx) * cz_m / fx
                        cy_m = (cv_y - cy) * cz_m / fy
                        # 在点击位置绘制十字标记和 3D 坐标
                        cv2.drawMarker(color_image, (cu, cv_y), (255, 0, 255),
                                       cv2.MARKER_CROSS, 20, 2)
                        test_label = f"TEST -> X:{cx_m:.3f} Y:{cy_m:.3f} Z:{cz_m:.3f} m"
                        cv2.rectangle(color_image, (cu + 5, cv_y - 25),
                                      (cu + 320, cv_y + 5), (0, 0, 0), -1)
                        cv2.putText(color_image, test_label, (cu + 10, cv_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # 窗口顶部显示相机类型和模型名称
            cv2.putText(color_image, f"{cam_type.upper()} | {model_name.upper()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, color_image)

            # 按键处理：Q 或 Esc 退出
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            # 窗口被关闭时退出循环
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        # 清理资源：关闭相机，销毁所有 OpenCV 窗口
        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
