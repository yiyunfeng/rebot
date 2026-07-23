"""感知模块共用的小型辅助函数。"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def tensor_to_numpy(value: Any) -> Optional[np.ndarray]:
    """把 NumPy、PyTorch 或类似 tensor 的值统一转换为 CPU NumPy 数组。

    ``detach`` 用于脱离梯度图，``cpu`` 用于把 GPU 数据移回内存；普通数组
    则直接返回，避免不必要的复制。
    """
    # 按“无需处理 -> PyTorch 处理 -> 通用转换”的顺序兼容多种输入类型。
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def class_name(names: Any, cls_id: int) -> str:
    """兼容字典和列表形式的类别表；查不到时退回类别编号。"""
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    try:
        return str(names[cls_id])
    except Exception:
        return str(cls_id)


def clip_bbox(values: np.ndarray, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """整理并裁剪检测框，保证 ``(x1, y1, x2, y2)`` 位于图像范围内。"""
    h, w = image_shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in values[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    x2 = int(np.clip(x2, 0, max(0, w - 1)))
    y2 = int(np.clip(y2, 0, max(0, h - 1)))
    return x1, y1, x2, y2


def detection_count(result: Any) -> int:
    """返回一个 Ultralytics 结果中的目标数，优先读取 OBB，再读取普通 boxes。"""
    for attr in ("obb", "boxes"):
        container = getattr(result, attr, None)
        if container is None:
            continue
        try:
            count = len(container)
        except Exception:
            continue
        if count > 0:
            return count
    return 0
