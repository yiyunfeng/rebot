"""基于 OBB（有向包围盒）/ 最小面积矩形的通用抓取姿态估计工具。

本模块实现了从 YOLO 检测结果（OBB 或标准检测框）到六自由度抓取姿态的完整流程：
1. 从检测结果中获取目标的旋转矩形（rect_points）
2. 在深度图中采样目标区域的深度值
3. 通过相机内参反投影到 3D 空间，计算抓取坐标系（grip_axis, open_axis, approach）
4. 生成符合 reBot 机械臂 TCP 位姿约定的 rotation 矩阵
5. 同时计算抓取宽度（jaw_width）和物体长度，用于夹爪规划
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

try:
    from .common_utils import detection_count
    from .transforms import grasp_axes_to_rebot_tcp_rotation
    from .yolo_utils import detection_meta, obb_points
except ImportError:
    from common_utils import detection_count
    from transforms import grasp_axes_to_rebot_tcp_rotation
    from yolo_utils import detection_meta, obb_points


@dataclass
class GraspPose:
    """单个目标的抓取姿态数据结构。

    包含从 2D 检测结果推导出的 3D 抓取位姿、抓取宽度和物体尺寸等信息。
    如果抓取估计失败（如深度无效、轴计算失败），rejected_reason 字段会记录原因。
    """

    # ---- 检测元信息 ----
    class_name: str           # 目标类别名称，如 "banana"、"cup"
    conf: float               # 检测置信度 [0.0, 1.0]
    bbox_xyxy: tuple[int, int, int, int]   # 检测框坐标 (x1, y1, x2, y2)，像素坐标

    # ---- 图像空间信息 ----
    center_px: tuple[int, int]            # 抓取中心点在图像中的像素坐标 (u, v)
    rect_points: np.ndarray               # 旋转矩形的 4 个角点，shape (4, 2)，像素坐标
    short_edge_points: np.ndarray         # 短边两端的 2 个点，shape (2, 2)，像素坐标，用于可视化抓取线

    # ---- 3D 空间信息（估计失败时为 None） ----
    position: Optional[np.ndarray]        # 抓取点在相机坐标系下的 3D 位置 [x, y, z] (米)，z 为深度方向
    rotation: Optional[np.ndarray]        # 3x3 抓取旋转矩阵，列向量为 [grip_axis, open_axis, approach]
    tcp_rotation: Optional[np.ndarray]    # 转换为 reBot TCP 坐标系的 3x3 旋转矩阵

    # ---- 抓取尺寸 ----
    jaw_width_m: float      # 预估的抓取宽度（米），即短边对应的 3D 长度，用于设定夹爪开度
    object_length_m: float  # 预估的物体长度（米），即长边对应的 3D 长度
    angle_deg: float        # 抓取短边在图像平面中的角度（度），0° = 水平，90° = 垂直

    # ---- 状态信息 ----
    valid_depth_pixels: int               # 有效深度像素数，用于判断深度数据质量
    rejected_reason: Optional[str] = None # 如果抓取估计被拒绝，记录原因（None 表示成功）

    @property
    def is_valid(self) -> bool:
        """抓取姿态是否有效。

        有效的条件：未被拒绝（rejected_reason 为 None），且 position 和 rotation 均已成功计算。
        """
        return self.rejected_reason is None and self.position is not None and self.rotation is not None


def get_depth_mm(depth_map: np.ndarray, u: int, v: int, roi_size: int = 5) -> float:
    """从深度图的局部窗口中采样中位数深度值。

    以像素坐标 (u, v) 为中心，取 roi_size × roi_size 窗口内所有有效深度值（>0）的中位数。
    使用中位数而非均值，可有效抵抗深度图中的噪声和离群值（如深度边缘、反射噪点）。

    参数：
        depth_map: 深度图，单位毫米，shape (H, W)，0 表示无效深度
        u: 目标像素的 x 坐标（列）
        v: 目标像素的 y 坐标（行）
        roi_size: 采样窗口的边长（像素），应为奇数

    返回：
        窗口内有效深度值的中位数（毫米）；如果没有有效深度，返回 0.0
    """
    h, w = depth_map.shape
    half = roi_size // 2
    # 计算裁剪边界，确保不越界
    x1, x2 = max(0, u - half), min(w, u + half + 1)
    y1, y2 = max(0, v - half), min(h, v + half + 1)
    roi = depth_map[y1:y2, x1:x2]
    # 过滤掉无效深度（值为 0 的像素，通常是缺失或超出量程的区域）
    valid = roi[roi > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def estimate_grasps(
    results: list[Any],
    depth_mm: np.ndarray,
    K: np.ndarray,
    depth_quantile: float = 0.75,
    mask_overrides: Optional[dict[tuple[int, int], np.ndarray]] = None,
) -> list[GraspPose]:
    """对一批 YOLO 检测结果进行批量抓取姿态估计。

    遍历每帧的每个检测目标，调用 estimate_grasp() 进行单目标抓取估计。
    支持通过 mask_overrides 为特定目标指定额外的分割 mask（如 SAM 精修结果）。

    参数：
        results: YOLO 检测结果列表，每个元素为一帧的检测输出
        depth_mm: 深度图（毫米），shape (H, W)
        K: 相机内参矩阵，3x3
        depth_quantile: 深度分位数阈值，用于选择抓取深度（0.75 = 取深度值的第 75 百分位作为抓取平面的深度）
        mask_overrides: 可选的 mask 覆盖字典，key 为 (result_index, detection_index)，
                        value 为对应的二值 mask（与深度图同尺寸）

    返回：
        抓取姿态列表，每个检测目标对应一个 GraspPose
    """
    # 没有外部 mask 时使用空字典，后面统一通过 get() 取得 None。
    grasps: list[GraspPose] = []
    mask_overrides = mask_overrides or {}
    # results 的第一层是一张图，第二层是该图中的每个检测目标。
    for result_index, result in enumerate(results):
        for index in range(detection_count(result)):
            grasps.append(
                estimate_grasp(
                    result,
                    index,
                    depth_mm,
                    K,
                    depth_quantile=depth_quantile,
                    mask_override=mask_overrides.get((result_index, index)),
                )
            )
    return grasps


def select_best_grasp(grasps: list[GraspPose]) -> Optional[GraspPose]:
    """从多个抓取候选中选择最佳抓取。

    筛选策略：仅保留有效的抓取姿态（is_valid == True），然后按置信度（conf）降序选择最高的。
    这个策略简单但有效：YOLO 检测置信度通常与目标定位准确度正相关。

    参数：
        grasps: 抓取姿态候选列表

    返回：
        置信度最高的有效抓取姿态；如果没有有效抓取，返回 None
    """
    # 先去掉无深度、坐标轴失败等不可执行候选，再比较检测置信度。
    valid = [grasp for grasp in grasps if grasp.is_valid]
    if not valid:
        return None
    return max(valid, key=lambda grasp: grasp.conf)


def draw_grasp(image: np.ndarray, grasp: GraspPose, *, show_pose_text: bool = True) -> None:
    """在图像上绘制抓取姿态的可视化标注。

    绘制内容包括：
    1. 检测框（矩形边框，有效=绿色，无效=橙色）
    2. 旋转矩形（黄色多边形，表示目标的 OBB/最小面积矩形）
    3. 抓取短边线（白色线段，表示夹爪开合方向）
    4. 抓取中心点（红色实心圆）
    5. 文本信息（类别、置信度、夹爪开度、3D 位置、角度等）
    6. 文本背景（黑色半透明矩形，确保文字可读）

    参数：
        image: 要绘制的图像（原地修改），BGR 格式
        grasp: 要可视化的抓取姿态
        show_pose_text: 是否显示 3D 位置信息（True: 显示 XYZ 坐标；False: 显示像素坐标）
    """
    x1, y1, x2, y2 = grasp.bbox_xyxy
    # 有效抓取用绿色边框，无效用橙色
    color = (0, 255, 0) if grasp.is_valid else (0, 165, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # 绘制旋转矩形（黄色多边形）
    rect_pts = np.round(grasp.rect_points).astype(np.int32)
    cv2.polylines(image, [rect_pts], True, (255, 200, 0), 2, cv2.LINE_AA)

    # 绘制短边抓取线（白色线段，表示夹爪开合方向）
    p0, p1 = np.round(grasp.short_edge_points).astype(np.int32)
    cv2.line(image, tuple(p0), tuple(p1), (255, 255, 255), 3, cv2.LINE_AA)

    # 绘制抓取中心点（红色实心圆，半径 5 像素）
    cv2.circle(image, grasp.center_px, 5, (0, 0, 255), -1)

    # 构建文本信息
    if grasp.is_valid:
        x_m, y_m, z_m = grasp.position.tolist()
        line1 = f"{grasp.class_name} {grasp.conf:.2f} jaw={grasp.jaw_width_m * 100:.1f}cm"
        if show_pose_text:
            line2 = f"X:{x_m:.3f} Y:{y_m:.3f} Z:{z_m:.3f} ang:{grasp.angle_deg:.1f}"
        else:
            line2 = f"center={grasp.center_px} ang:{grasp.angle_deg:.1f}"
    else:
        line1 = f"{grasp.class_name} {grasp.conf:.2f}"
        line2 = grasp.rejected_reason or "invalid"

    # 绘制文本背景（黑色半透明矩形）
    bg_w = max(len(line1), len(line2)) * 10
    cv2.rectangle(image, (x1, y1 - 42), (x1 + bg_w, y1), (0, 0, 0), -1)
    cv2.putText(image, line1, (x1 + 4, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(image, line2, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def estimate_grasp(
    result: Any,
    index: int,
    depth_mm: np.ndarray,
    K: np.ndarray,
    depth_quantile: float = 0.75,
    mask_override: Optional[np.ndarray] = None,
) -> GraspPose:
    """对单个检测目标进行完整的抓取姿态估计。

    算法流程：
    1. 提取检测元信息（类别、置信度、检测框）
    2. 获取目标的旋转矩形 rect_points（OBB → SAM mask → YOLO mask → bbox 回退）
    3. 获取深度采样区域 mask
    4. 找到最短边（短边方向 = 夹爪开合方向），并尝试通过 mask 精炼抓取线和中心点
    5. 从深度图中采样目标区域深度值，用分位数确定抓取平面深度 z_m
    6. 反投影中心点到 3D 空间 → 得到抓取点位置 position
    7. 计算抓取坐标系三轴：
       - approach: 接近方向（指向相机）= normalize(-position)
       - open_axis: 夹爪开合方向，由短边方向在像素空间的方向反投影到 3D，再正交化
       - grip_axis: 抓取方向 = normalize(cross(open_axis, approach))，与开合方向和接近方向都垂直
    8. 确保 open_axis 在 x 方向为正（指向机械臂基座右侧）
    9. 通过施密特正交化保证三轴严格正交（按 grip → open → approach 顺序重正交化）
    10. 生成对 reBot TCP 坐标系友好的 rotation 矩阵
    11. 计算抓取宽度（jaw_width）和物体长度（object_length）

    注意：在第 9 步中，正交化顺序是：先确保 grip_axis = normalize(cross(open, approach))，
    再确保 open_axis = normalize(cross(approach, grip_axis))。这样做保证 grip_axis 与 approach
    严格正交，open_axis 与 approach 严格正交，从而满足 SO(3) 的要求。

    参数：
        result: 单帧 YOLO 检测结果
        index: 该帧中目标检测框的索引
        depth_mm: 深度图（毫米），shape (H, W)
        K: 相机内参矩阵，3x3
        depth_quantile: 深度分位数，0.75 表示取区域内第 75 百分位的深度值作为抓取深度
        mask_override: 可选的外部 mask（如 SAM 生成），覆盖检测结果的 mask

    返回：
        抓取姿态数据，如果某步骤失败则返回带有 rejected_reason 的无效 GraspPose
    """
    # ========== 步骤 1: 提取检测元信息 ==========
    class_name, conf, bbox_xyxy = detection_meta(result, index, depth_mm.shape)

    # ========== 步骤 2: 获取旋转矩形 ==========
    # 多种回退策略：OBB → SAM mask → YOLO seg mask → bbox 角点
    rect_points = _rect_points(result, index, depth_mm.shape, bbox_xyxy, mask_override)

    # 旋转矩形的几何中心作为初始抓取中心
    center = rect_points.mean(axis=0).astype(np.float32)

    # ========== 步骤 3: 获取深度采样的 mask 区域 ==========
    mask = _depth_mask(result, index, depth_mm.shape, rect_points, mask_override)

    # ========== 步骤 4: 确定抓取方向（短边）和抓取线 ==========
    # 找到旋转矩形的最短边，其方向 = 夹爪开合方向（open direction）
    short_vec_uv, short_len_px = _short_edge(rect_points)
    # 归一化短边方向向量
    short_dir_uv = _normalize(short_vec_uv)
    # 计算所有边长，找到最长边
    edge_lengths = [float(np.linalg.norm(rect_points[(i + 1) % 4] - rect_points[i])) for i in range(4)]
    long_len_px = max(edge_lengths)
    grasp_span_px = short_len_px  # 初始抓取跨度 = 短边像素长度
    # 根据中心和短边向量构造抓取线段（中心 ± 半短边）
    short_edge_points = _line_from_center(center, short_vec_uv)

    # ========== 步骤 4b: 通过 mask 精炼抓取线和中心点 ==========
    # 对于弯曲/不对称物体（如香蕉），OBB 的中心和短边可能偏离实际最佳抓取位置。
    # 精炼步骤在物体纵轴的中位数切片上计算 mask 的真实宽度，
    # 用 mask 中心替代 OBB 中心，得到更可靠的抓取点。
    if short_dir_uv is not None:
        refined = _refine_grasp_line_from_mask(mask, center, short_dir_uv, long_len_px)
        if refined is not None:
            center, short_edge_points, grasp_span_px = refined

    # ========== 步骤 5: 从深度图中采样目标深度 ==========
    center_px = (int(round(float(center[0]))), int(round(float(center[1]))))
    # 取 mask 区域内所有有效深度值
    depth_values = depth_mm[mask > 0]
    depth_values = depth_values[depth_values > 0]
    # 如果 mask 区域没有有效深度，尝试用中心点周围的小窗口采样作为回退
    if len(depth_values) == 0:
        center_depth = get_depth_mm(depth_mm, center_px[0], center_px[1], 5)
        if center_depth > 0:
            depth_values = np.array([center_depth], dtype=np.float32)

    # 如果没有有效深度或短边方向无效，直接返回失败的 GraspPose
    if len(depth_values) == 0 or short_dir_uv is None:
        return GraspPose(
            class_name=class_name,
            conf=conf,
            bbox_xyxy=bbox_xyxy,
            center_px=center_px,
            position=None,
            rotation=None,
            tcp_rotation=None,
            jaw_width_m=0.0,
            object_length_m=0.0,
            angle_deg=0.0,
            rect_points=rect_points,
            short_edge_points=short_edge_points,
            valid_depth_pixels=int(len(depth_values)),
            rejected_reason="no_valid_depth_or_rect",
        )

    # 用分位数确定抓取平面的深度（米），取 depth_quantile 分位避免深度图中的离群值
    depth_quantile = float(np.clip(depth_quantile, 0.0, 1.0))
    z_m = float(np.quantile(depth_values, depth_quantile) / 1000.0)

    # ========== 步骤 6: 反投影中心点到 3D 空间 ==========
    # 利用针孔相机模型：X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
    position = _backproject(float(center[0]), float(center[1]), z_m, K)

    # ========== 步骤 7: 计算抓取坐标系三轴 ==========
    # approach: 接近方向，指向相机原点（即从物体指向相机）
    approach = _normalize(-position)
    if approach is None:
        approach = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # open_axis: 夹爪开合方向（短边方向在 3D 空间的表示）
    # 先将像素空间的短边方向向量转换为 3D 向量（利用针孔模型的比例关系）
    open_axis = _pixel_vec_to_3d(short_dir_uv, z_m, K)
    # 去掉 open_axis 中与 approach 平行的分量，使其与 approach 正交（Gram-Schmidt 第一步）
    open_axis = open_axis - float(np.dot(open_axis, approach)) * approach
    open_axis = _normalize(open_axis)
    if open_axis is None:
        return GraspPose(
            class_name=class_name,
            conf=conf,
            bbox_xyxy=bbox_xyxy,
            center_px=center_px,
            position=None,
            rotation=None,
            tcp_rotation=None,
            jaw_width_m=0.0,
            object_length_m=0.0,
            angle_deg=0.0,
            rect_points=rect_points,
            short_edge_points=short_edge_points,
            valid_depth_pixels=int(len(depth_values)),
            rejected_reason="open_axis_failed",
        )

    # ========== 步骤 8: 统一 open_axis 方向 ==========
    # 确保 open_axis 在相机 x 方向为正（指向机械臂基座右侧），保证抓取姿态的一致性
    if open_axis[0] < 0:
        open_axis = -open_axis

    # ========== 步骤 9: 正交化抓取坐标系三轴 ==========
    # 按 grip → open → approach 顺序重新正交化，确保 rotation 矩阵是有效的 SO(3) 旋转矩阵
    # grip_axis = normalize(cross(open_axis, approach)) — 抓取方向，由右手定则确定
    grip_axis = _normalize(np.cross(open_axis, approach))
    # open_axis = normalize(cross(approach, grip_axis)) — 重新计算，确保与 grip 和 approach 都正交
    open_axis = _normalize(np.cross(approach, grip_axis))
    if grip_axis is None or open_axis is None:
        return GraspPose(
            class_name=class_name,
            conf=conf,
            bbox_xyxy=bbox_xyxy,
            center_px=center_px,
            position=None,
            rotation=None,
            tcp_rotation=None,
            jaw_width_m=0.0,
            object_length_m=0.0,
            angle_deg=0.0,
            rect_points=rect_points,
            short_edge_points=short_edge_points,
            valid_depth_pixels=int(len(depth_values)),
            rejected_reason="grasp_axis_failed",
        )

    # 组装通用抓取旋转矩阵，列为 [grip_axis, open_axis, approach]
    rotation = np.column_stack([grip_axis, open_axis, approach]).astype(np.float32)

    # ========== 步骤 10: 转换为 reBot TCP 坐标系 ==========
    # 将通用抓取旋转矩阵转换为 reBot 机械臂 TCP 末端执行器约定的坐标系方向
    tcp_rotation = grasp_axes_to_rebot_tcp_rotation(rotation[:, 0], rotation[:, 1], rotation[:, 2]).astype(np.float32)

    # ========== 步骤 11: 计算抓取尺寸 ==========
    # 夹爪开度：将像素空间的短边跨度转换为 3D 距离（米）
    jaw_width_m = float(np.linalg.norm(_pixel_vec_to_3d(short_dir_uv * grasp_span_px, z_m, K)))
    # 物体长度：将像素空间的长边转换为 3D 距离（米）
    object_length_m = float(np.linalg.norm(_pixel_vec_to_3d(short_dir_uv * long_len_px, z_m, K)))
    # 短边在图像平面中的角度（度），用于调试和可视化
    angle_deg = float(np.degrees(np.arctan2(short_dir_uv[1], short_dir_uv[0])))

    return GraspPose(
        class_name=class_name,
        conf=conf,
        bbox_xyxy=bbox_xyxy,
        center_px=center_px,
        position=position,
        rotation=rotation,
        tcp_rotation=tcp_rotation,
        jaw_width_m=jaw_width_m,
        object_length_m=object_length_m,
        angle_deg=angle_deg,
        rect_points=rect_points,
        short_edge_points=short_edge_points,
        valid_depth_pixels=int(len(depth_values)),
    )


def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    """将向量归一化为单位向量。

    如果向量的模长小于 1e-8（接近零向量），返回 None 以避免除零错误。

    参数：
        vec: 输入向量，任意维度的 numpy 数组

    返回：
        归一化后的向量（dtype=float32）；如果 vec 是零向量，返回 None
    """
    # 单位向量只保留方向；接近零的向量没有可靠方向，返回 None 让上层拒绝候选。
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return (vec / norm).astype(np.float32)


def _line_from_center(center: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """以 center 为中点、vec 为总长度/方向的线段两端点。

    返回 [center - 0.5*vec, center + 0.5*vec]，即从 center 向两端各延伸 vec 的一半长度。

    参数：
        center: 线段中点坐标，shape (2,)
        vec: 线段的总向量（方向和总长度），shape (2,)

    返回：
        线段两端点坐标，shape (2, 2)，dtype=float32
    """
    return np.stack([center - 0.5 * vec, center + 0.5 * vec], axis=0).astype(np.float32)


def _refine_grasp_line_from_mask(
    mask: np.ndarray,
    center: np.ndarray,
    short_dir_uv: np.ndarray,
    long_len_px: float,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """利用 mask 的中心横截面精炼短轴抓取点和抓取线。

    短轴方向仍然来自 OBB/最小面积矩形。我们仅用 mask 在物体纵向中位数切片上的实际厚度来替换
    抓取中心点。这对于弯曲或不对称形状（如香蕉）更可靠，因为 OBB 的几何中心可能不在物体表面上。

    算法步骤：
    1. 提取 mask 中所有前景像素点
    2. 将像素点投影到抓取方向上（grip_dir = 与短边方向垂直的纵轴方向），
       和开合方向上（open_coord = 沿短边方向的坐标）
    3. 找到 grip 方向的中位数位置（物体纵轴的中位数切片）
    4. 在该切片附近取一个带状区域（band），宽度为长边长度的 4%~8%
    5. 在带状区域内统计开合方向上的像素分布，取第 5 百分位到第 95 百分位的范围作为抓取跨度
    6. 用 mask 的 open 方向中心替换 OBB 中心，得到精炼后的抓取线和跨度

    参数：
        mask: 二值分割 mask，shape (H, W)，值为 0 或 >0
        center: OBB 的几何中心，shape (2,)，像素坐标
        short_dir_uv: 短边方向（开合方向）单位向量，shape (2,)，像素坐标
        long_len_px: 长边长度（像素）

    返回：
        (精炼中心, 精炼短边端点, 精炼抓取跨度) 三元组；如果 mask 像素不足或跨度太小，返回 None
    """
    # 提取 mask 中所有前景像素的坐标
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 32:  # 像素太少，无法可靠精炼
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)

    # grip_dir_uv: 纵轴方向（与短边方向垂直的抓取方向）
    # 2D 旋转 90°：(dx, dy) → (-dy, dx)
    grip_dir_uv = np.array([-short_dir_uv[1], short_dir_uv[0]], dtype=np.float32)

    # 将所有像素点投影到以 OBB 中心为原点的局部坐标系
    # rel: 每个像素点相对于 OBB 中心的偏移向量
    rel = points - center.reshape(1, 2)
    # grip_coord: 每个像素点在纵轴方向上的投影坐标（标量内积）
    grip_coord = rel @ grip_dir_uv
    # open_coord: 每个像素点在短边方向（开合方向）上的投影坐标
    open_coord = rel @ short_dir_uv

    # 取纵轴方向上的中位数位置作为纵轴中心切片
    grip_center = float(np.median(grip_coord))

    # 在纵轴中心切片附近取一个带状区域（band）
    # 初始带宽 = 长边长度的 4%，限制在 [2, 12] 像素范围内
    band_half_width_px = float(np.clip(long_len_px * 0.04, 2.0, 12.0))
    band_mask = np.abs(grip_coord - grip_center) <= band_half_width_px
    # 如果带状区域像素太少，放宽带宽到 8%（限制在 [4, 18] 像素）
    if int(np.count_nonzero(band_mask)) < 24:
        band_half_width_px = float(np.clip(long_len_px * 0.08, 4.0, 18.0))
        band_mask = np.abs(grip_coord - grip_center) <= band_half_width_px
    # 仍然太少，放弃精炼
    if int(np.count_nonzero(band_mask)) < 24:
        return None

    # 在带状区域内，用 open 方向上的像素分布确定抓取宽度
    band_open = open_coord[band_mask]
    # 取第 5 百分位和第 95 百分位作为抓取跨度的两端（去除了极端离群像素）
    open_min = float(np.percentile(band_open, 5.0))
    open_max = float(np.percentile(band_open, 95.0))
    grasp_span_px = open_max - open_min
    if grasp_span_px < 2.0:  # 跨度太小，不可靠
        return None

    # 计算 open 方向上的真实中心（mask 中心，而非 OBB 中心）
    open_center = 0.5 * (open_min + open_max)

    # 精炼后的中心 = OBB 中心 + 纵轴方向偏移 + 开合方向偏移
    refined_center = center + grip_center * grip_dir_uv + open_center * short_dir_uv
    # 根据精炼后的中心和跨度重新构造短边线段
    short_edge_points = _line_from_center(refined_center, short_dir_uv * grasp_span_px)
    return refined_center.astype(np.float32), short_edge_points, float(grasp_span_px)


def _rect_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """从二值 mask 中提取最小面积旋转矩形的 4 个角点。

    算法：找到 mask 的最大外轮廓 → 计算其最小面积外接旋转矩形（cv2.minAreaRect）→ 提取 4 个角点。

    参数：
        mask: 二值 mask，shape (H, W)，dtype 为 uint8，0 = 背景，非 0 = 前景

    返回：
        旋转矩形的 4 个角点，shape (4, 2)，dtype=float32；如果轮廓不足 3 个点则返回 None
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # 取面积最大的轮廓
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:  # 至少需要 3 个点才能构成有效矩形
        return None
    # cv2.minAreaRect 返回 (中心(x,y), (宽, 高), 旋转角度)
    rect = cv2.minAreaRect(contour.astype(np.float32))
    # cv2.boxPoints 将 minAreaRect 的结果转换为 4 个角点
    return cv2.boxPoints(rect).astype(np.float32)


def _rect_points(
    result: Any,
    index: int,
    image_shape: tuple[int, int],
    bbox_xyxy: tuple[int, int, int, int],
    mask_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """获取目标旋转矩形的 4 个角点，带多重回退策略。

    获取优先级：
    1. OBB 角点（最精确，来自 YOLO-OBB 模型的有向包围盒）
    2. SAM mask override → 最小面积矩形（来自外部 SAM 精修的分割 mask）
    3. YOLO 实例分割 mask → 最小面积矩形（来自 YOLO-seg 的 mask 输出）
    4. 检测框 bbox 的 4 个角点（最粗糙的兜底方案：轴对齐矩形）

    参数：
        result: YOLO 检测结果
        index: 检测框索引
        image_shape: 图像尺寸 (H, W)，用于 resize mask
        bbox_xyxy: 检测框坐标 (x1, y1, x2, y2)
        mask_override: 可选的外部 mask（如 SAM 输出）

    返回：
        旋转矩形的 4 个角点，shape (4, 2)，dtype=float32，像素坐标
    """
    # 策略 1: 优先使用 OBB 角点（来自旋转检测头）
    points = obb_points(result, index, image_shape)
    if points is not None:
        return points

    # 策略 2: 如果有 SAM mask override，从中提取最小面积矩形
    if mask_override is not None:
        rect = _rect_from_mask(mask_override.astype(np.uint8))
        if rect is not None:
            return rect

    # 策略 3: 使用 YOLO 实例分割 mask 提取旋转矩形
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is not None and boxes is not None and len(masks.data) == len(boxes):
        mask = masks.data[index].cpu().numpy()
        # YOLO seg mask 通常比原图小，需要 resize 到原始尺寸
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        rect = _rect_from_mask((mask > 0.5).astype(np.uint8))
        if rect is not None:
            return rect

    # 策略 4: 兜底方案 — 使用轴对齐的 bbox 角点
    x1, y1, x2, y2 = bbox_xyxy
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _depth_mask(
    result: Any,
    index: int,
    image_shape: tuple[int, int],
    rect_points: np.ndarray,
    mask_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """获取深度采样的二值 mask 区域。

    深度采样区域用于从深度图中提取目标的深度值。获取优先级：
    1. 外部 mask override（如 SAM 精修 mask）— 最精确
    2. YOLO 实例分割 mask — 来自模型的分割输出
    3. 旋转矩形填充的 mask — 用 rect_points 构造的凸多边形区域

    参数：
        result: YOLO 检测结果
        index: 检测框索引
        image_shape: 图像尺寸 (H, W)
        rect_points: 旋转矩形的 4 个角点
        mask_override: 外部覆盖 mask

    返回：
        二值 mask，shape (H, W)，dtype=uint8，1 表示有效采样区域
    """
    # 策略 1: 使用外部 SAM mask
    if mask_override is not None:
        mask = mask_override.astype(np.uint8)
        if mask.shape != image_shape:
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)

    # 策略 2: 使用 YOLO 实例分割 mask
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is not None and boxes is not None and len(masks.data) == len(boxes):
        mask = masks.data[index].cpu().numpy()
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0.5).astype(np.uint8)

    # 策略 3: 兜底 — 用旋转矩形构造凸多边形 mask
    polygon = np.round(rect_points).astype(np.int32)
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def _short_edge(rect_points: np.ndarray) -> tuple[np.ndarray, float]:
    """遍历矩形的 4 条边，找到长度最短的那条边的向量和长度。

    矩形的 4 条边按顺序为：0→1, 1→2, 2→3, 3→0（封闭多边形）。

    参数：
        rect_points: 矩形的 4 个角点，shape (4, 2)

    返回：
        (最短边的向量, 最短边的长度) 元组
    """
    # 初始化为第一条边的向量和长度
    best_vec = rect_points[1] - rect_points[0]
    best_len = float(np.linalg.norm(best_vec))
    for i in range(4):
        p0 = rect_points[i]
        p1 = rect_points[(i + 1) % 4]  # 循环索引，第 4 条边连接 p3→p0
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length < best_len:
            best_vec = vec
            best_len = length
    return best_vec.astype(np.float32), best_len


def _backproject(u: float, v: float, z_m: float, K: np.ndarray) -> np.ndarray:
    """针孔相机模型反投影：将像素坐标和深度值转换为 3D 相机坐标。

    针孔相机投影公式（正向）：u = fx * X/Z + cx, v = fy * Y/Z + cy
    反投影公式（逆向）：X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = z_m

    其中：
    - fx, fy: 相机焦距（像素单位）
    - cx, cy: 主点坐标（像素单位），通常为图像中心
    - z_m: 深度值（米）
    - (u, v): 像素坐标
    - (X, Y, Z): 相机坐标系下的 3D 坐标（米）

    参数：
        u: 像素 x 坐标（列）
        v: 像素 y 坐标（行）
        z_m: 深度值（米）
        K: 相机内参矩阵，3x3，格式 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    返回：
        3D 坐标 [X, Y, Z]（米），相机坐标系，dtype=float32
    """
    # K 的对角线给出像素焦距，第三列给出光轴与图像平面的交点。
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u - cx) * z_m / fx
    y = (v - cy) * z_m / fy
    return np.array([x, y, z_m], dtype=np.float32)


def _pixel_vec_to_3d(vec_uv: np.ndarray, z_m: float, K: np.ndarray) -> np.ndarray:
    """将像素空间的向量转换为 3D 空间的向量。

    利用针孔相机模型的比例关系进行转换：
    像素偏移 dx 在 3D 空间对应的 X 偏移 = dx * Z / fx
    像素偏移 dy 在 3D 空间对应的 Y 偏移 = dy * Z / fy
    Z 偏移 = 0（因为像素向量不包含深度方向的分量）

    例如，已知在深度 Z = z_m 处，像素偏移为 (dx, dy) 像素，
    则 3D 空间中的实际偏移为 (dx * z_m / fx, dy * z_m / fy, 0)。

    注意：这是一个线性近似，假设物体表面在该区域是平坦且平行于像平面的。
    对于倾斜表面，这个近似会引入一定误差，但通常在实际抓取中可接受。

    参数：
        vec_uv: 像素空间向量 (dx, dy)，单位像素
        z_m: 深度值（米），即 Z 坐标
        K: 相机内参矩阵，3x3

    返回：
        3D 空间向量，shape (3,)，dtype=float32，Z 分量为 0
    """
    # 焦距理论上应大于零；下限 1e-6 仅防止损坏的内参造成除零崩溃。
    fx, fy = max(float(K[0, 0]), 1e-6), max(float(K[1, 1]), 1e-6)
    return np.array([float(vec_uv[0]) * z_m / fx, float(vec_uv[1]) * z_m / fy, 0.0], dtype=np.float32)
