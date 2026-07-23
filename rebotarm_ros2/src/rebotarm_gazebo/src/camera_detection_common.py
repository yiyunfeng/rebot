"""Common data structures and mask utilities for camera detection.

公共数据结构和后处理工具放在这里，HSV 检测只负责产生 mask/轮廓，
深度过滤、中心点和角度计算统一在这里完成。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class ObjectDetection:
    """视觉检测结果的统一内部格式。

    contour/mask 用于调试图和深度取样；u/v/depth 用于反投影 3D 点。
    angle_deg 是图像平面主轴角度，当前抓取节点暂未消费，但 debug 图会显示出来。
    position_xyz / orientation_xyzw 保留为扩展字段。当前 Gazebo 仿真只用 HSV，
    因此保持 None，由 ROS 节点继续用像素+深度反投影。
    """

    u: int                      # 目标中心点在图像中的像素 x 坐标（列）
    v: int                      # 目标中心点在图像中的像素 y 坐标（行）
    depth: float                # 目标区域的深度中位数（米），用于反投影 3D 位置
    contour: np.ndarray         # 最大外轮廓的多边形点集，shape (N, 1, 2)，用于 OBB 计算
    mask: np.ndarray            # 目标区域的二值 mask，shape (H, W)，值为 0/255
    angle_deg: float            # 目标在图像平面的主轴角度（度），长边相对 x 轴的夹角
    label: str = ""             # 语义标签，如 "hsv_target"
    confidence: float = 0.0     # 检测置信度 [0, 1]，HSV 固定 1.0
    backend: str = "hsv"        # 检测后端名称，便于下游区分 "hsv" / "yolo" / "sam"
    position_xyz: tuple[float, float, float] | None = None        # 相机坐标系 3D 位置（扩展字段）
    orientation_xyzw: tuple[float, float, float, float] | None = None  # 四元数姿态（扩展字段）


def detection_from_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    contour: np.ndarray | None,
    backend: str,
    label: str,
    confidence: float,
    param: Callable[[str], Any],
) -> ObjectDetection | None:
    """从二值 mask 计算中心、深度中位数和主轴角度。

    HSV 检测走这里，统一处理深度无效值、最小面积和深度范围。

    Args:
        mask: 目标区域的二值图像，值域可为 0/1、0/255 或 0.0~1.0，shape (H, W)
        depth: 对齐后的深度图（米），shape (H, W)，dtype float32
        contour: 目标外轮廓；传 None 时从 mask 自动提取
        backend: 检测后端名称
        label: 语义标签
        confidence: 检测置信度
        param: 参数回调函数，接收参数名返回对应值（如 min_area、min_depth 等）

    Returns:
        ObjectDetection | None: 检测结果，mask 内无有效深度或面积过小时返回 None
    """
    # ---- 尺寸对齐：RGB 与深度图分辨率不同时，resize mask 到深度图尺寸 ----
    if mask.shape[:2] != depth.shape[:2]:
        # INTER_NEAREST 保持二值边界，避免线性插值产生灰色过渡带
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    # ---- 二值化归一：将不同来源的 mask 统一为 0/255 uint8 ----
    # HSV 输出 0/255，SAM 输出 0/1，YOLO seg 输出 0.0~1.0 float，统一后 mask > 0 才可靠
    mask = (mask > 0).astype(np.uint8) * 255

    # ---- 轮廓提取：调用方未传轮廓时，从 mask 中自动提取 ----
    if contour is None:
        # RETR_EXTERNAL: 只取最外层轮廓，忽略孔洞内轮廓
        # CHAIN_APPROX_SIMPLE: 压缩水平/垂直/对角线段，只保留端点，减少内存占用
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None  # mask 全黑，无有效目标区域
        # 取面积最大的轮廓作为目标（假设画面中只有一个目标物体）
        contour = max(contours, key=cv2.contourArea)

    # ---- 面积过滤：排除噪声产生的过小区域 ----
    area = cv2.contourArea(contour)
    if area < float(param("min_area")):
        return None

    # ---- 中心点计算：用图像矩求轮廓质心 ----
    moments = cv2.moments(contour)
    # m00 是轮廓面积（像素数），为零说明轮廓退化，无法计算质心
    if moments["m00"] <= 0.0:
        return None
    # 质心公式：u = m10 / m00, v = m01 / m00
    u = int(moments["m10"] / moments["m00"])
    v = int(moments["m01"] / moments["m00"])

    # ---- 深度采样：取 mask 覆盖区域内所有像素的深度值 ----
    # 不直接取中心点深度：中心点可能落在空洞/边缘上（深度为 0），mask 内中位数更稳
    valid = depth[mask > 0]
    min_depth = float(param("min_depth"))  # 最小有效深度（米），排除近处干扰
    max_depth = float(param("max_depth"))  # 最大有效深度（米），排除远处背景
    # 只保留有限数值和合理工作距离内的点，排除 NaN/Inf/0 和远处背景
    valid = valid[np.isfinite(valid) & (valid >= min_depth) & (valid <= max_depth)]
    if valid.size == 0:
        return None  # mask 区域内无有效深度，无法反投影 3D 位置

    # ---- 组装检测结果 ----
    return ObjectDetection(
        u=u,
        v=v,
        depth=float(np.median(valid)),  # 深度中位数，比均值更抗离群点（如边缘噪点）
        contour=contour,
        mask=mask,
        angle_deg=contour_angle_deg(contour),
        label=label,
        confidence=confidence,
        backend=backend,
    )


def scale_contour(contour: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    """把彩色图上的 contour 缩放到深度图尺寸，处理 RGB/Depth 分辨率不同的情况。

    Args:
        contour: 轮廓点集，shape (N, 1, 2)，像素坐标
        src_hw: 源图像尺寸 (H, W)
        dst_hw: 目标图像尺寸 (H, W)

    Returns:
        缩放后的轮廓点集，shape (N, 1, 2)，dtype int32
    """
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    # 先转 float32 做乘法缩放，避免整数除法截断
    scaled = contour.astype(np.float32).copy()
    # x 坐标按宽度比例缩放
    scaled[:, :, 0] *= float(dst_w) / max(1.0, float(src_w))
    # y 坐标按高度比例缩放
    scaled[:, :, 1] *= float(dst_h) / max(1.0, float(src_h))
    # 四舍五入回整数像素坐标
    return np.round(scaled).astype(np.int32)


def contour_angle_deg(contour: np.ndarray) -> float:
    """用最小外接矩形估计目标在图像平面的主轴角度。

    Args:
        contour: 轮廓点集，shape (N, 1, 2)

    Returns:
        主轴角度（度），长边相对图像 x 轴的夹角
    """
    # cv2.minAreaRect 返回 ((cx, cy), (w, h), angle)
    (_, _), (w, h), angle = cv2.minAreaRect(contour)
    # OpenCV 的 angle 定义和矩形长短边有关：w>=h 时 angle 在 [-90,0]；
    # w<h 时换成 h>=w，angle 需 +90° 才能统一为"长边相对 x 轴"的角度
    if w < h:
        angle += 90.0
    return float(angle)
