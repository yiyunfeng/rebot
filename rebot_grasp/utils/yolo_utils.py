"""YOLO 模型加载与原始检测结果解析的辅助函数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# 尝试相对导入，支持包内和直接脚本运行两种方式
try:
    from .common_utils import class_name, clip_bbox, detection_count, tensor_to_numpy
except ImportError:
    from common_utils import class_name, clip_bbox, detection_count, tensor_to_numpy


@dataclass
class YoloDetection:
    """单个 YOLO 检测结果的数据结构。

    字段含义：
        result_index:   YOLO 返回的 results 列表中的索引（一个 result 对应一张图）
        detection_index: 当前 result 中第几个检测框（0-based）
        class_name:      检测到的类别名称（如 "apple", "cup"）
        conf:            置信度分数，范围 [0.0, 1.0]
        bbox_xyxy:       边界框坐标，格式 (x1, y1, x2, y2)，单位为像素
        mask:            实例分割 mask，形状 (H, W) 的 uint8 二值图（1=前景）
    """
    result_index: int
    detection_index: int
    class_name: str
    conf: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray


def resolve_yolo_model_path(model_name: str, project_root: Path) -> Path:
    """解析 YOLO 模型文件的绝对路径。

    路径解析逻辑（按优先级）：
        1. 如果 model_name 是绝对路径，直接返回。
        2. 如果 model_name 包含多级目录（如 "sdk/models/yolo.pt"），
           相对于 project_root 解析。
        3. 否则，默认从 project_root/models/ 目录下查找。

    参数：
        model_name:   模型路径字符串（可以是绝对路径、相对多层路径或单纯文件名）
        project_root: 项目根目录

    返回：
        解析后的模型文件绝对路径
    """
    # 展开 ~ 为用户主目录
    model_path = Path(str(model_name)).expanduser()
    # 情况 1: 已经是绝对路径，直接返回
    if model_path.is_absolute():
        return model_path
    # 情况 2: 包含目录分隔符（如 "subdir/model.pt"），相对于 project_root 解析
    if len(model_path.parts) > 1:
        return project_root / model_path
    # 情况 3: 纯文件名，默认在 project_root/models/ 下查找
    return project_root / "models" / model_path


def load_yolo(
    cfg: dict[str, Any],
    *,
    project_root: Path,
    no_yolo: bool = False,
    model_override: Optional[str] = None,
    device_override: Optional[str] = None,
    conf_override: Optional[float] = None,
    iou_override: Optional[float] = None,
    infer_every_override: Optional[int] = None,
    extra_classes: Optional[list[str]] = None,
) -> tuple[Optional[Any], dict[str, Any]]:
    """加载 YOLO 模型，合并配置文件与命令行覆盖参数。

    参数合并优先级（从高到低）：
        命令行覆盖参数 > 配置文件 yolo/detection 节 > 硬编码默认值

    如果 no_yolo=True 则跳过加载，直接返回 (None, opts)。

    参数：
        cfg:                  完整的配置字典
        project_root:         项目根目录（用于解析模型路径）
        no_yolo:              是否禁用 YOLO
        model_override:       命令行覆盖的模型名称
        device_override:      命令行覆盖的设备（"cpu" / "cuda:0"）
        conf_override:        命令行覆盖的置信度阈值
        iou_override:         命令行覆盖的 IOU 阈值（NMS 用）
        infer_every_override: 命令行覆盖的推理间隔（每隔多少帧推理一次）
        extra_classes:        额外追加的类别列表（用于 open-vocabulary 模型）

    返回：
        (model, yolo_opts) 的元组：
        - model:     YOLO 模型实例，no_yolo 时为 None
        - yolo_opts: 合并后的 YOLO 运行参数字典
    """
    # 从配置中提取 grasp_pipeline 节
    gp_cfg = cfg.get("grasp_pipeline", {})
    # 初始化 yolo_opts，enabled 控制是否启用 YOLO
    yolo_opts: dict[str, Any] = {
        "enabled": not no_yolo,
        # infer_every: 命令行覆盖 > 配置的 infer_every_live > 默认值 3
        "infer_every": max(1, int(infer_every_override or gp_cfg.get("infer_every_live", 3))),
    }
    # 如果禁用 YOLO，直接返回空值
    if no_yolo:
        return None, yolo_opts

    # 仅在需要时导入 ultralytics，避免加载未使用的依赖
    from ultralytics import YOLO

    # 读取配置各节
    yolo_cfg = cfg.get("yolo", {})
    det_cfg = cfg.get("detection", {})
    # 模型名称：命令行覆盖 > 配置 > 默认 yoloe-26s-seg.pt
    model_name = str(model_override or yolo_cfg.get("model_name", "yoloe-26s-seg.pt"))
    model_path = resolve_yolo_model_path(model_name, project_root)
    # 设备：命令行覆盖 > 配置 > 默认 cpu
    device = device_override or yolo_cfg.get("device", "cpu")
    # 置信度阈值：命令行覆盖 > 配置 > 默认 0.25
    conf = float(conf_override if conf_override is not None else det_cfg.get("conf_threshold", 0.25))
    # IOU 阈值（NMS）：命令行覆盖 > 配置 > 默认 0.45
    iou = float(iou_override if iou_override is not None else det_cfg.get("iou_threshold", 0.45))
    # 自定义类别列表（用于 open-vocabulary 检测）
    custom_classes = list(yolo_cfg.get("custom_classes", []))
    # 合并 extra_classes，去重追加
    for extra_class in extra_classes or []:
        if extra_class and extra_class not in custom_classes:
            custom_classes.append(extra_class)
    # 是否启用 open-vocabulary（YOLOE-World 模式）
    use_world = bool(yolo_cfg.get("use_world", True))

    print(f"Loading YOLO target detector: {model_path}")
    # 加载 YOLO 模型
    model = YOLO(str(model_path))
    # 如果是 open-vocabulary 模型且指定了自定义类别，则设置类别列表
    if use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower()) and custom_classes:
        model.set_classes(custom_classes)
        print(f"YOLO open-vocabulary classes: {custom_classes}")

    # 将最终使用的参数写回 yolo_opts
    yolo_opts.update(
        {
            "model_name": model_name,
            "device": device,
            "conf": conf,
            "iou": iou,
            "custom_classes": custom_classes,
        }
    )
    return model, yolo_opts


def detection_mask(result: Any, index: int, image_shape: tuple[int, int], bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """获取检测结果的实例分割 mask，若无则回退到 bbox 矩形 mask。

    mask 生成策略（按优先级）：
        1. 如果 YOLO 结果包含实例分割 mask（result.masks.data），提取对应 index 的 mask。
        2. 如果 mask 尺寸与图像不一致，resize 到图像尺寸。
        3. 如果 mask 不存在或提取失败，回退：以 bbox 为界生成矩形二值 mask。

    参数：
        result:      单个 YOLO 推理结果（ultralytics Results 对象）
        index:       检测结果在 results 中的索引
        image_shape: 原始图像尺寸 (H, W)
        bbox_xyxy:   边界框坐标 (x1, y1, x2, y2)，作为 mask 回退的备选

    返回：
        uint8 二值 mask 数组，形状 (H, W)，1 表示前景，0 表示背景
    """
    h, w = image_shape
    # 尝试获取实例分割 mask
    masks = getattr(result, "masks", None)
    data = getattr(masks, "data", None)
    if data is not None:
        try:
            if len(data) > index:
                # 将 tensor 转为 numpy 数组
                mask = tensor_to_numpy(data[index])
                if mask is not None:
                    mask = np.asarray(mask, dtype=np.float32)
                    # 如果 mask 尺寸与图像不一致，resize 对齐
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    # 二值化：>0.5 为前景
                    return (mask > 0.5).astype(np.uint8)
        except Exception:
            # 实例分割 mask 提取失败，静默跳过，回退到 bbox mask
            pass

    # 回退策略：将 bbox 区域设为前景
    x1, y1, x2, y2 = bbox_xyxy
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return mask


def obb_points(result: Any, index: int, image_shape: tuple[int, int]) -> Optional[np.ndarray]:
    """从 YOLO OBB（旋转边界框）结果中提取 4 个角点坐标。

    OBB 点坐标解析流程：
        1. 从 result.obb 中尝试读取 xyxyxyxy（像素坐标）或 xyxyxyxyn（归一化坐标）。
        2. 如果坐标为归一化值（值域 [0, 1]），乘以图像宽高转为像素坐标。
        3. 将一维 8 元素数组 reshape 为 (4, 2) 形状。

    参数：
        result:      单个 YOLO 推理结果
        index:       检测结果索引
        image_shape: 原始图像尺寸 (H, W)

    返回：
        shape 为 (4, 2) 的 float32 数组（4 个角点的 (x, y) 像素坐标），
        无 OBB 结果时返回 None
    """
    # 检查结果是否包含 OBB
    obb = getattr(result, "obb", None)
    if obb is None:
        return None

    # 按优先级尝试两种属性名：像素坐标 > 归一化坐标
    points = None
    for attr in ("xyxyxyxy", "xyxyxyxyn"):
        values = getattr(obb, attr, None)
        if values is None:
            continue
        try:
            points = tensor_to_numpy(values[index])
        except Exception:
            points = None
        if points is not None:
            break

    if points is None:
        return None

    points = np.asarray(points, dtype=np.float32)
    # 处理 batch 维度：若为 (1, 4, 2) 则 squeeze 为 (4, 2)
    if points.ndim == 3 and points.shape[0] == 1:
        points = points[0]
    # 处理一维 8 元素数组：reshape 为 (4, 2)
    if points.ndim == 1 and points.size == 8:
        points = points.reshape(4, 2)
    if points.shape != (4, 2):
        return None
    # 判断是否为归一化坐标（值域 [0, 1.5] 内均视为归一化）
    if float(np.max(np.abs(points))) <= 1.5:
        h, w = image_shape
        # 归一化坐标 -> 像素坐标
        points = points * np.array([w, h], dtype=np.float32)
    return points.astype(np.float32)


def obb_detection_meta(result: Any, index: int, image_shape: tuple[int, int]) -> tuple[str, float, tuple[int, int, int, int]]:
    """从 OBB（旋转框）检测结果中提取元信息。

    从 result.obb 中读取 class_id、confidence，并计算 axis-aligned 边界框：
    - 优先使用 obb.xyxy（OBB 的外接矩形）
    - 若无 xyxy，则从 4 个角点计算 min/max 得到外接矩形

    参数：
        result:      YOLO 推理结果
        index:       检测结果索引
        image_shape: 图像尺寸 (H, W)

    返回：
        (类别名, 置信度, bbox_xyxy) 的元组
    """
    # 获取类别名映射表
    names = getattr(result, "names", {})
    obb = getattr(result, "obb", None)
    if obb is None:
        raise ValueError("YOLO result has no OBB detections")

    # 读取类别 ID 和置信度（tensor -> numpy -> 标量）
    cls_row = tensor_to_numpy(getattr(obb, "cls")[index])
    conf_row = tensor_to_numpy(getattr(obb, "conf")[index])
    cls_id = int(np.asarray(cls_row).reshape(-1)[0])
    conf = float(np.asarray(conf_row).reshape(-1)[0])

    # 优先使用 OBB 自带的 axis-aligned xyxy
    xyxy = getattr(obb, "xyxy", None)
    if xyxy is not None:
        bbox = clip_bbox(np.asarray(tensor_to_numpy(xyxy[index])).reshape(-1), image_shape)
    else:
        # 从 4 个角点的 OBB 多边形计算外接矩形
        points = obb_points(result, index, image_shape)
        if points is None:
            raise ValueError("YOLO OBB result has neither xyxy nor polygon points")
        # 取 min/max 构成外接矩形
        bbox = clip_bbox(np.concatenate([points.min(axis=0), points.max(axis=0)]), image_shape)
    return class_name(names, cls_id), conf, bbox


def box_detection_meta(result: Any, index: int, image_shape: tuple[int, int]) -> tuple[str, float, tuple[int, int, int, int]]:
    """从标准水平框检测结果中提取元信息。

    从 result.boxes 中读取 class_id、confidence、xyxy 边界框。

    参数：
        result:      YOLO 推理结果
        index:       检测结果索引
        image_shape: 图像尺寸 (H, W)

    返回：
        (类别名, 置信度, bbox_xyxy) 的元组
    """
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise ValueError("YOLO result has no box detections")
    box = boxes[index]
    # 读取 xyxy 坐标
    xyxy = tensor_to_numpy(box.xyxy[0]).reshape(-1)
    # 读取类别 ID 和置信度
    cls_id = int(tensor_to_numpy(box.cls[0]).reshape(-1)[0])
    conf = float(tensor_to_numpy(box.conf[0]).reshape(-1)[0])
    # 裁剪到图像边界
    return class_name(names, cls_id), conf, clip_bbox(xyxy, image_shape)


def detection_meta(result: Any, index: int, image_shape: tuple[int, int]) -> tuple[str, float, tuple[int, int, int, int]]:
    """统一的检测结果元信息提取入口。

    自动判断检测类型：如果有 OBB 结果则走 OBB 路径，否则走标准水平框路径。

    参数：
        result:      YOLO 推理结果
        index:       检测结果索引
        image_shape: 图像尺寸 (H, W)

    返回：
        (类别名, 置信度, bbox_xyxy) 的元组
    """
    # 根据是否有 OBB 结果选择不同的解析路径
    if getattr(result, "obb", None) is not None:
        return obb_detection_meta(result, index, image_shape)
    return box_detection_meta(result, index, image_shape)


def detection_polygon_mask(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """根据多边形角点生成二值 mask。

    使用 cv2.fillPoly 填充多边形区域为前景。

    参数：
        points:      多边形角点坐标，shape (N, 2)，像素坐标
        image_shape: mask 尺寸 (H, W)

    返回：
        uint8 二值 mask，多边形内部为 1，外部为 0
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    # 四舍五入转为整数像素坐标，填充多边形内部
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask


def collect_detections(results: list[Any], image_shape: tuple[int, int]) -> list[YoloDetection]:
    """批量收集所有 YOLO 推理结果中的检测目标。

    批量收集流程：
        1. 遍历 result 列表（每张图一个 result）。
        2. 对每个 result，遍历其内部的 detection_index。
        3. 提取元信息（类别、置信度、bbox）和 mask。
        4. mask 生成策略：优先使用 OBB 多边形 mask，无 OBB 时回退到实例分割
           mask 或 bbox 矩形 mask。

    参数：
        results:     YOLO model.predict() 返回的结果列表
        image_shape: 原始图像尺寸 (H, W)

    返回：
        YoloDetection 对象的列表，按检测顺序排列
    """
    detections: list[YoloDetection] = []
    for result_index, result in enumerate(results):
        # 遍历当前 result 中的每个检测框
        for detection_index in range(detection_count(result)):
            try:
                # 提取元信息
                name, conf, bbox = detection_meta(result, detection_index, image_shape)
                # 尝试获取 OBB 多边形角点
                points = obb_points(result, detection_index, image_shape)
                # mask 策略：有 OBB 用它填充多边形；否则从实例分割 mask 或 bbox 回退
                mask = detection_polygon_mask(points, image_shape) if points is not None else detection_mask(result, detection_index, image_shape, bbox)
            except Exception:
                # 单个检测框解析失败不影响其他检测框
                continue
            detections.append(YoloDetection(result_index, detection_index, name, conf, bbox, mask))
    return detections


def detect_objects(
    model: Any,
    color_bgr: np.ndarray,
    yolo_opts: dict[str, Any],
) -> tuple[list[Any], list[YoloDetection]]:
    """执行完整的 YOLO 目标检测流程。

    完整检测流程：
        1. 调用 model.predict() 执行推理。
        2. 使用 yolo_opts 中的参数（device, conf, iou）控制推理行为。
        3. 调用 collect_detections() 解析原始结果为 YoloDetection 对象列表。

    参数：
        model:     YOLO 模型实例
        color_bgr: 输入图像（BGR 格式，OpenCV 标准）
        yolo_opts: YOLO 运行参数字典

    返回：
        (results, detections) 的元组：
        - results:     model.predict() 的原始返回结果列表
        - detections:  解析后的 YoloDetection 对象列表
    """
    # 执行 YOLO 推理
    results = model.predict(
        color_bgr,
        verbose=False,
        device=yolo_opts.get("device", "cpu"),
        conf=float(yolo_opts.get("conf", 0.25)),
        iou=float(yolo_opts.get("iou", 0.45)),
    )
    # 解析原始结果为结构化检测列表
    detections = collect_detections(results, color_bgr.shape[:2])
    return results, detections
