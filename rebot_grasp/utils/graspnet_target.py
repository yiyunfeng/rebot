"""GraspNet 目标 mask 的纯 NumPy/OpenCV 几何辅助函数。"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_binary_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """把 SAM 输出规范成与 RGB 图像同尺寸的 uint8 二值 mask。"""
    mask_array = np.asarray(mask)
    mask_array = np.squeeze(mask_array)
    if mask_array.ndim != 2:
        raise ValueError(
            f"SAM mask must be 2D after squeeze, got shape={mask_array.shape}"
        )

    height, width = image_shape
    if mask_array.shape != (height, width):
        mask_array = cv2.resize(
            mask_array, (width, height), interpolation=cv2.INTER_NEAREST
        )
    return (mask_array > 0).astype(np.uint8)


def projected_points_in_mask(
    translations: np.ndarray,
    K: np.ndarray,
    target_mask: np.ndarray,
    *,
    margin_px: int = 0,
) -> np.ndarray:
    """判断相机坐标系三维点的像素投影是否位于目标 mask 内。

    ``margin_px`` 大于 0 时先膨胀 mask，用于容忍 SAM 边界和深度/RGB 对齐的
    少量像素误差；0 表示严格按 SAM 原始区域判断。
    """
    points = np.asarray(translations, dtype=np.float64)
    intrinsics = np.asarray(K, dtype=np.float64)
    mask = np.asarray(target_mask)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"translations must have shape (N, 3), got {points.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {intrinsics.shape}")
    if mask.ndim != 2:
        raise ValueError(f"target_mask must be 2D, got {mask.shape}")

    mask_u8 = (mask > 0).astype(np.uint8)
    margin_px = max(0, int(margin_px))
    if margin_px:
        diameter = margin_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        mask_u8 = cv2.dilate(mask_u8, kernel)

    keep = np.zeros(len(points), dtype=bool)
    if len(points) == 0 or not np.any(mask_u8):
        return keep

    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return keep

    valid_points = points[valid_indices]
    valid_z = valid_points[:, 2]
    u = np.rint(
        intrinsics[0, 0] * valid_points[:, 0] / valid_z + intrinsics[0, 2]
    ).astype(np.int64)
    v = np.rint(
        intrinsics[1, 1] * valid_points[:, 1] / valid_z + intrinsics[1, 2]
    ).astype(np.int64)

    height, width = mask_u8.shape
    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    in_image_indices = valid_indices[in_image]
    keep[in_image_indices] = mask_u8[v[in_image], u[in_image]] > 0
    return keep
