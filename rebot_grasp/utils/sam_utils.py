"""使用可选的 SAM 对 YOLO 检测结果进行 mask 精细化。

SAM 只做“按提示框精分割”，不负责识别类别。这里复用 YOLO 的检测框作为 prompt，
再把 SAM mask 传给 ordinary_grasp.py，用更干净的轮廓和深度区域计算抓取点。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    from .common_utils import detection_count
    from .yolo_utils import detection_meta
except ImportError:
    from common_utils import detection_count
    from yolo_utils import detection_meta


def resolve_sam_model_path(model_name: str, project_root: Path) -> Path:
    """按项目约定解析 SAM 权重路径。"""
    # expanduser() 会把 ~/models/sam.pt 中的 ~ 展开成用户目录。
    model_path = Path(str(model_name)).expanduser()

    # 绝对路径已经明确指向文件，不再拼接项目目录。
    if model_path.is_absolute():
        return model_path

    # 带目录的相对路径（如 weights/sam.pt）相对于项目根目录解析；
    # 只有文件名（如 sam_b.pt）则按项目约定放到 models/ 下查找。
    if len(model_path.parts) > 1:
        return project_root / model_path
    return project_root / "models" / model_path


class SamMaskRefiner:
    """YOLO box -> SAM mask 的轻量封装。

    predictor 懒加载并复用，避免每帧重复初始化模型。SAM 依赖较重，所以只有配置
    `sam.enabled: true` 时才会创建这个对象。
    """

    def __init__(self, model_path: Path, *, conf: float = 0.01, device: Optional[str] = None) -> None:
        """保存模型配置；SAM 会在第一次调用 ``refine_results`` 时才加载。"""
        self._model_path = Path(model_path)
        self._conf = float(conf)
        self._device = device
        self._predictor: Any | None = None

    def refine_results(self, results: list[Any], color_bgr: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        """对 YOLO 结果中的每个检测框生成 SAM mask。

        处理过程：YOLO 提供目标框 -> 目标框作为 SAM 提示 -> SAM 返回目标的像素级 mask。
        返回值 key 为 `(result_index, detection_index)`，用于把每张 mask 准确对应回
        YOLO 的某一批结果和其中的某一个检测目标。

        单个目标处理失败时只跳过该目标，不影响其他目标；调用方会自动回退到
        YOLO 自带的 mask 或检测框。
        """
        # 第一次调用时加载 SAM，之后复用同一个 predictor，避免每帧重复加载模型。
        predictor = self._load_predictor()

        # OpenCV 图像是 BGR，而 SAM 按 RGB 读取颜色，所以推理前必须转换通道顺序。
        image_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # 一张图只设置一次。后面循环时只更换 YOLO 检测框，不重复处理整张图。
        predictor.set_image(image_rgb)

        # 保存“YOLO 检测目标 -> SAM mask”的对应关系。
        # 例如 (0, 2) 表示 results[0] 中第 3 个检测目标的 mask。
        overrides: dict[tuple[int, int], np.ndarray] = {}
        image_shape = color_bgr.shape[:2]

        # results 可能包含多张图的结果，因此先遍历图像结果，再遍历其中的每个目标。
        for result_index, result in enumerate(results):
            for detection_index in range(detection_count(result)):
                try:
                    # 取出当前目标的 YOLO 检测框。类别和置信度在这里不需要，所以用 _ 忽略。
                    _, _, bbox = detection_meta(result, detection_index, image_shape)

                    # 把检测框作为 SAM 的 box prompt，让 SAM 只分割这个框对应的物体。
                    # predictor 要求 bboxes 是批量格式，因此单个框也要写成 [bbox]。
                    sam_results = predictor(bboxes=[list(map(float, bbox))])

                    # SAM 可能返回多层结果；这里取当前框的第一张 mask，并对齐原图尺寸。
                    mask = self._first_mask(sam_results, image_shape)
                except Exception:
                    # 当前目标失败就不覆盖它，后续抓取逻辑会使用 YOLO mask 或 bbox。
                    mask = None
                if mask is not None:
                    overrides[(result_index, detection_index)] = mask
        return overrides

    def _load_predictor(self) -> Any:
        """懒加载 Ultralytics SAM Predictor。"""
        # _predictor 初始为 None。模型只在第一次真正分割时创建，
        # 后续帧直接复用，避免反复读取权重和占用新的显存。
        if self._predictor is None:
            from ultralytics.models.sam import Predictor as SAMPredictor  # type: ignore[import-not-found]

            # overrides 是 Ultralytics Predictor 的启动配置，不是检测结果的 overrides。
            overrides: dict[str, Any] = {
                "task": "segment",
                "mode": "predict",
                "model": str(self._model_path),
                "conf": self._conf,
                "save": False,
                "verbose": False,
            }

            # device 为空时让 Ultralytics 自己选择；配置了 cpu/cuda 时才显式传入。
            if self._device:
                overrides["device"] = self._device

            self._predictor = SAMPredictor(overrides=overrides)
            print(f"Loading SAM mask refiner: {self._model_path}")
        return self._predictor

    @staticmethod
    def _first_mask(results: Any, image_shape: tuple[int, int]) -> np.ndarray | None:
        """从 SAM 输出中取第一张二值 mask，并 resize 到彩色图尺寸。"""
        # SAM 没有返回结果或没有 masks，说明这个提示框没有得到有效分割。
        if not results or getattr(results[0], "masks", None) is None:
            return None

        # Ultralytics 把 mask tensor 存在 results[0].masks.data 中，
        # 其常见形状为 (mask数量, 高, 宽)。当前每次只输入一个框，所以取 data[0]。
        data = getattr(results[0].masks, "data", None)
        if data is None or len(data) == 0:
            return None

        # 推理结果可能在 GPU 上，先移到 CPU 再转成 NumPy，供 OpenCV 后续处理。
        mask = data[0].cpu().numpy()
        h, w = image_shape

        # 某些模型按内部推理尺寸输出 mask，需要恢复到原彩色图的 (高, 宽)。
        # mask 是离散区域，使用最近邻插值可避免边缘产生无意义的灰度过渡。
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # SAM 输出是 0~1 的概率/置信值；大于 0.5 记为前景，最终统一成 uint8 的 0/1 mask。
        return (mask > 0.5).astype(np.uint8)


def load_sam_refiner(
    cfg: dict[str, Any],
    *,
    project_root: Path,
    device_override: Optional[str] = None,
) -> SamMaskRefiner | None:
    """从配置创建 SAM refiner；未启用时返回 None。"""
    # SAM 是可选步骤。关闭时直接返回 None，主流程会继续使用 YOLO mask 或 bbox。
    sam_cfg = cfg.get("sam", {})
    if not bool(sam_cfg.get("enabled", False)):
        return None

    # 模型、阈值和设备都从 YAML 读取，避免写死在推理代码中。
    model_name = str(sam_cfg.get("model_name", "sam_b.pt"))
    model_path = resolve_sam_model_path(model_name, project_root)
    conf = float(sam_cfg.get("conf_threshold", 0.01))

    # 命令行/调用方临时指定的设备优先级高于 YAML 配置。
    device = device_override or sam_cfg.get("device")
    return SamMaskRefiner(model_path, conf=conf, device=device)


def draw_sam_masks_overlay(
    image: np.ndarray,
    masks: Optional[dict[tuple[int, int], np.ndarray]],
    *,
    alpha: float = 0.32,
) -> None:
    """把 SAM mask 半透明叠加到图像上，并画出外轮廓。

    SAM mask 是像素级结果，单独看抓取线很难判断分割是否正确；这里把 mask
    直接画回 RGB 图像，便于确认“YOLO 框内真正被 SAM 抠出来的区域”。
    """
    if not masks:
        return

    # overlay 专门保存彩色填充层，image 保留原图并直接绘制轮廓；
    # 函数结尾再把两者混合，得到半透明 mask 效果。
    overlay = image.copy()
    colors = [
        (0, 220, 255),   # 黄
        (255, 120, 80),  # 蓝橙
        (120, 255, 120), # 绿
        (255, 120, 255), # 紫
    ]
    for index, ((_, detection_index), mask) in enumerate(masks.items()):
        if mask is None:
            continue

        # 无论输入是 bool、0/1 还是概率值，都统一为只含 0 和 1 的 uint8 图。
        mask_u8 = (mask > 0).astype(np.uint8)

        # 防止外部传入的 mask 尺寸与当前显示图不同；最近邻缩放保持二值边界。
        if mask_u8.shape != image.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 全零 mask 没有任何物体像素，既不填色也不查找轮廓。
        if not np.any(mask_u8):
            continue

        # 颜色循环使用，目标数量超过颜色表时从头复用。
        color = colors[index % len(colors)]
        overlay[mask_u8 > 0] = color

        # RETR_EXTERNAL 只取物体最外层边界，避免内部孔洞产生大量干扰线。
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)

        # 在最大轮廓附近标出 SAM mask 序号，方便和检测结果对应。
        if contours:
            # 多个不连通区域中用面积最大的区域放标签，通常它才是主体。
            contour = max(contours, key=cv2.contourArea)
            m = cv2.moments(contour)

            # m00 是轮廓面积；接近 0 时不能作为分母计算质心。
            if m["m00"] > 1e-6:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                cv2.putText(
                    image,
                    f"SAM#{detection_index}",
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

    # alpha 控制 mask 颜色强度，结果直接写回 image，调用方无需接收新数组。
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def render_sam_masks_window(
    image: np.ndarray,
    masks: Optional[dict[tuple[int, int], np.ndarray]],
    status_text: str = "SAM mask contours",
) -> np.ndarray:
    """生成单独 SAM 调试窗口图像。

    主抓取窗口负责显示抓取线；这个窗口只看 SAM 分割是否正确：
      - 左侧保留原 RGB 图像亮度，便于确认物体位置；
      - SAM 区域用半透明颜色覆盖；
      - 外轮廓用粗线描出。
    """
    # copy() 保证调试绘制不会污染主窗口正在使用的原始图像。
    debug = image.copy()
    if masks:
        draw_sam_masks_overlay(debug, masks, alpha=0.38)
    else:
        # 没有可显示的 mask 时明确提示，便于区分“没有目标”和“窗口卡住”。
        cv2.putText(
            debug,
            "No SAM mask",
            (10, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    # 顶部始终显示 LIVE/SNAPSHOT 等状态，说明这张调试图来自哪个阶段。
    cv2.putText(
        debug,
        status_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return debug
