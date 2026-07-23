"""GraspNet 辅助函数：架设在相机帧与 YOLO 检测之上的抓取推理层。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import open3d as o3d
import torch

# 设置 matplotlib 配置目录为 /tmp，避免权限问题
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# 设置 QT 字体目录（用于 Open3D 可视化窗口中的文字渲染）
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

# ============================================================
# 常量定义
# ============================================================

# 项目根目录（从当前文件向上两级：utils/ -> rebot_grasp/）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# GraspNet 基线模型 SDK 根目录
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"
# GraspNet 默认视角采样数（用于抓取候选生成时的角度离散化）
DEFAULT_NUM_VIEW = 300
# 碰撞检测的体素尺寸（米），越小检测越精细但越慢
DEFAULT_VOXEL_SIZE = 0.01
# 系统预热帧数：前 N 帧深度图不稳定，跳过不做推理
DEFAULT_WARMUP_FRAMES = 20
# Open3D 可视化时绕 X 轴翻转的变换矩阵（Y、Z 取反），
# 用于让点云在 Open3D 窗口中以更自然的姿态显示
DISPLAY_FLIP_X = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def prepare_graspnet_imports(graspnet_root: Path = GRASPNET_ROOT) -> None:
    """将 GraspNet SDK 子模块目录注入 sys.path，确保 import 路径可用。

    路径注入策略：
        将 graspnet-baseline 下的四个子目录（models, dataset, utils, pointnet2, graspnetAPI）
        以及根目录本身插入 sys.path 头部，确保后续 import 能正确找到这些包。

    注意：该函数在模块加载时即被调用（见下方 prepare_graspnet_imports()），
    因此导入 graspnet_utils 模块后 GraspNet 相关包即可直接 import。

    参数：
        graspnet_root: GraspNet SDK 根目录路径
    """
    # 将子目录逐个插入 sys.path
    for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI"):
        path = str(graspnet_root / subdir)
        if path not in sys.path:
            sys.path.insert(0, path)
    # 确保根目录也在 sys.path 中
    root = str(graspnet_root)
    if root not in sys.path:
        sys.path.insert(0, root)


# 模块加载时立即注入路径，确保后续 import 可用
prepare_graspnet_imports()

# 尝试相对导入 YoloDetection（支持包内和直接脚本运行两种方式）
try:
    from .yolo_utils import YoloDetection
except ImportError:
    from yolo_utils import YoloDetection

# GraspNet 相关导入（依赖上方 prepare_graspnet_imports 注入的路径）
# noqa: E402 忽略 "module level import not at top of file" 警告
from collision_detector import ModelFreeCollisionDetector  # noqa: E402
from data_utils import CameraInfo, create_point_cloud_from_depth_image  # noqa: E402
from graspnet import GraspNet, pred_decode  # noqa: E402
from graspnetAPI import Grasp, GraspGroup  # noqa: E402

# 坐标变换辅助函数
try:
    from .transforms import graspnet_rotation_to_rebot_tcp_rotation, transform_grasp_pose_to_base_with_retreat
except ImportError:
    from transforms import graspnet_rotation_to_rebot_tcp_rotation, transform_grasp_pose_to_base_with_retreat


@dataclass
class GraspNetFrameResult:
    """单帧 GraspNet 推理的完整结果数据。

    字段含义：
        grasps:          最终抓取集合（经过 bbox 过滤 + 宽度过滤后的抓取候选）
        pre_bbox_grasps: bbox 过滤前的原始抓取集合（来自全场景推理）
        bbox_grasps:     bbox 过滤后、宽度过滤前的抓取集合
        best:            最优抓取位姿（经过 NMS + 分数排序后的第一名）
        status:          状态描述字符串（用于 UI 显示）
        target_status:   目标检测状态字符串（YOLO 检测结果摘要）
        detections:      当前帧的 YOLO 检测结果列表
        selected_target: 选中的目标检测（用于抓取的目标对象）
        o3d_cloud:       Open3D 格式的点云（用于可视化）
        raw_cloud:       原始点云数组，shape (N, 3)（用于碰撞检测）
    """
    grasps: GraspGroup
    pre_bbox_grasps: GraspGroup
    bbox_grasps: GraspGroup
    best: Optional[Grasp]
    status: str
    target_status: str
    detections: list[YoloDetection]
    selected_target: Optional[YoloDetection]
    o3d_cloud: o3d.geometry.PointCloud
    raw_cloud: np.ndarray


class Open3DGraspWindow:
    """Open3D 可视化窗口封装：用于实时显示点云和抓取候选。

    窗口管理流程：
        1. __init__: 创建 Open3D Visualizer 窗口（1280x720）
        2. update:   清空旧几何体 -> 添加新点云 -> NMS 去重抓取 -> 取 top_k -> 渲染
        3. poll:     处理窗口事件并刷新渲染
        4. close:    销毁窗口

    每次 update 会完全替换窗口内容，不支持增量更新。
    """

    def __init__(self, title: str, top_k: int) -> None:
        """初始化 Open3D 抓取可视化窗口。

        参数：
            title: 窗口标题
            top_k: 最多显示的抓取候选数量（按分数排序取前 k 个）
        """
        self._top_k = top_k
        self._vis = o3d.visualization.Visualizer()
        # 创建窗口，失败则销毁并抛异常
        if not self._vis.create_window(title, width=1280, height=720):
            self._vis.destroy_window()
            raise RuntimeError("Open3D visualizer window could not be created")
        self._geometries = []  # 当前显示的几何体列表
        self._initialized = False  # 首次渲染标志（用于设置初始视角）

    def update(self, cloud: o3d.geometry.PointCloud, grasps: GraspGroup) -> None:
        """更新窗口显示：清空旧内容，渲染新的点云和抓取候选。

        渲染流程：
            1. 移除所有已有几何体。
            2. 复制点云并应用 DISPLAY_FLIP_X 变换（调整显示朝向）。
            3. 对抓取候选执行 NMS（非极大值抑制）去重，按分数排序取 top_k。
            4. 将点云和抓取几何体添加到可视化窗口。

        参数：
            cloud:  待显示的 Open3D 点云
            grasps: 待显示的抓取候选集合
        """
        # 清空所有旧几何体
        for geom in self._geometries:
            self._vis.remove_geometry(geom, reset_bounding_box=False)
        self._geometries = []

        # 复制点云并翻转 X 轴，使朝向更直观
        cloud_vis = o3d.geometry.PointCloud(cloud)
        cloud_vis.transform(DISPLAY_FLIP_X)
        geometries = [cloud_vis]

        # 处理抓取候选：NMS 去重 -> 按分数降序 -> 取 top_k
        if len(grasps) > 0:
            # 深拷贝抓取集合，避免修改原数据
            grasps_vis = GraspGroup(grasps.grasp_group_array.copy())
            try:
                # NMS（非极大值抑制）：去除重叠度过高的重复抓取
                grasps_vis = grasps_vis.nms()
            except Exception as exc:
                print(f"Grasp NMS skipped: {exc}")
            # 按分数降序排列（分数越高越好）
            grasps_vis.sort_by_score()
            # 只保留前 top_k 个最优抓取用于显示
            grasps_vis = grasps_vis[: self._top_k]
            # 对齐显示朝向
            grasps_vis.transform(DISPLAY_FLIP_X)
            # 将抓取几何体（夹爪模型）添加到场景
            geometries.extend(grasps_vis.to_open3d_geometry_list())

        # 添加所有几何体到窗口
        for geom in geometries:
            # 首次渲染时重置包围盒以设置初始视角
            self._vis.add_geometry(geom, reset_bounding_box=not self._initialized)
        self._geometries = geometries
        self._initialized = True
        # 刷新渲染
        self.poll()

    def poll(self) -> bool:
        """轮询窗口事件并刷新渲染。

        返回：
            True 表示窗口仍在运行，False 表示窗口已被关闭
        """
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def close(self) -> None:
        """销毁可视化窗口，释放资源。"""
        self._vis.destroy_window()


def copy_grasp_group(grasps: GraspGroup) -> GraspGroup:
    """深拷贝抓取集合（复制内部数组）。

    参数：
        grasps: 原始抓取集合

    返回：
        具有相同数据的新的 GraspGroup 实例
    """
    return GraspGroup(grasps.grasp_group_array.copy())


def visualization_grasps(result: GraspNetFrameResult, mode: str) -> GraspGroup:
    """根据模式返回用于 Open3D 可视化的抓取集合。

    支持三种模式：
    - "pre-bbox": 返回 bbox 过滤前的全场景抓取
    - "bbox":     返回 bbox 过滤后、宽度过滤前的抓取
    - 其他/默认:  返回最终抓取集合（全部过滤后）

    参数：
        result: 单帧 GraspNet 推理结果
        mode:   可视化模式字符串

    返回：
        对应当前模式的抓取集合
    """
    # 三个集合对应过滤链的不同阶段，用于观察候选在哪一步被删掉。
    if mode == "pre-bbox":
        return result.pre_bbox_grasps
    if mode == "bbox":
        return result.bbox_grasps
    return result.grasps


def resolve_checkpoint_path(
    checkpoint: str,
    *,
    project_root: Path = PROJECT_ROOT,
    graspnet_root: Path = GRASPNET_ROOT,
) -> Path:
    """解析 GraspNet 模型 checkpoint 文件的绝对路径。

    路径解析逻辑（与 resolve_yolo_model_path 类似）：
        1. 如果是绝对路径，直接返回。
        2. 如果包含多级目录，相对于 project_root 解析。
        3. 否则，默认在 graspnet_root/checkpoints/ 下查找。

    参数：
        checkpoint:    checkpoint 文件路径字符串
        project_root:  项目根目录
        graspnet_root: GraspNet SDK 根目录

    返回：
        checkpoint 文件的绝对路径
    """
    # 与 YOLO/SAM 保持同一规则：绝对路径、项目相对路径、默认模型目录。
    checkpoint_path = Path(str(checkpoint)).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path
    if len(checkpoint_path.parts) > 1:
        return project_root / checkpoint_path
    return graspnet_root / "checkpoints" / checkpoint_path


def build_net(checkpoint_path: str | Path, num_view: int = DEFAULT_NUM_VIEW) -> GraspNet:
    """构建并加载预训练权重的 GraspNet 网络模型。

    网络结构说明：
        - input_feature_dim=0: 仅使用点云坐标（无额外特征）
        - num_view:            抓取角度离散化的视角数（默认 300）
        - num_angle=12:        绕抓取轴的旋转角度离散化数
        - num_depth=4:         抓取深度的离散化层数
        - cylinder_radius=0.05: 夹爪圆柱体碰撞检测半径（5cm）
        - hmin=-0.02:          抓取高度下限（-2cm，允许部分嵌入物体表面）
        - hmax_list:           抓取高度列表（1cm/2cm/3cm/4cm），对应不同闭合深度

    注意：GraspNet 的 pointnet2 算子需要 CUDA，不支持 CPU 推理。

    参数：
        checkpoint_path: checkpoint 文件路径（.tar 格式）
        num_view:        视角采样数，默认 300

    返回：
        已加载权重并设置为 eval 模式的 GraspNet 模型

    异常：
        RuntimeError: CUDA 不可用时抛出
    """
    # 解析 checkpoint 路径
    checkpoint_path = resolve_checkpoint_path(str(checkpoint_path))
    if not torch.cuda.is_available():
        raise RuntimeError("GraspNet pointnet2 operators require CUDA, but torch.cuda is unavailable.")

    # 构建 GraspNet 网络
    net = GraspNet(
        input_feature_dim=0,       # 仅点云坐标
        num_view=num_view,         # 视角采样数
        num_angle=12,              # 旋转角度离散化
        num_depth=4,               # 深度离散化
        cylinder_radius=0.05,      # 夹爪半径（用于碰撞检测）
        hmin=-0.02,                # 最低抓取高度
        hmax_list=[0.01, 0.02, 0.03, 0.04],  # 抓取高度层
        is_training=False,         # 推理模式
    )
    # 移至 GPU
    device = torch.device("cuda:0")
    net.to(device)

    # 加载预训练权重
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    # 设置为评估模式（禁用 dropout、batch norm 等）
    net.eval()
    print(f"Loaded checkpoint {checkpoint_path} (epoch: {checkpoint['epoch']})")
    return net


def build_end_points(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    num_point: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[dict, o3d.geometry.PointCloud, np.ndarray]:
    """从 RGB-D 图像构建 GraspNet 输入格式的点云。

    构建流程：
        1. **尺寸对齐**：如果深度图与颜色图尺寸不一致，resize 对齐。
        2. **深度裁剪**：根据 min_depth_m / max_depth_m 生成深度 mask，过滤无效点。
        3. **点云生成**：使用相机内参 K 将深度图反投影为 3D 点云。
        4. **点云采样**：从有效点中随机采样 num_point 个点，不足时允许重复采样。
        5. **构建输入字典**：组织为 GraspNet 期望的 end_points 格式。

    参数：
        color_bgr:   BGR 格式颜色图像 (H, W, 3)
        depth_mm:    深度图像 (H, W)，单位为毫米
        K:           3x3 相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        num_point:   采样点数（输入 GraspNet 的点数）
        min_depth_m: 有效深度最小值（米），小于此值的点被过滤
        max_depth_m: 有效深度最大值（米），大于此值的点被过滤

    返回：
        (end_points, o3d_cloud, raw_cloud) 的元组：
        - end_points: GraspNet 输入字典 {"point_clouds": tensor, "cloud_colors": array}
        - o3d_cloud:  Open3D 格式的点云（用于可视化）
        - raw_cloud:  原始点云数组 (N, 3)，dtype=float32（用于碰撞检测）

    异常：
        RuntimeError: 深度范围内无有效像素时抛出
    """
    # 步骤 1: 尺寸对齐 —— 如果深度图与颜色图尺寸不一致，resize 深度图
    if color_bgr.shape[:2] != depth_mm.shape[:2]:
        depth_mm = cv2.resize(depth_mm, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 步骤 2: 深度裁剪 —— 将米转为毫米，生成有效深度 mask
    min_mm = int(max(0.0, min_depth_m) * 1000.0)
    max_mm = int(max_depth_m * 1000.0)
    depth = depth_mm.astype(np.uint16, copy=False)
    mask = (depth > min_mm) & (depth < max_mm)
    if int(mask.sum()) == 0:
        raise RuntimeError("No valid depth pixels in the configured depth range.")

    # 步骤 3: 颜色预处理 —— BGR -> RGB，归一化到 [0, 1]
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 步骤 4: 通过相机内参将深度图反投影为 3D 点云
    h, w = depth.shape
    camera = CameraInfo(w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), 1000.0)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    # 只保留深度有效范围内的点
    cloud_masked = cloud[mask]
    color_masked = color_rgb[mask]

    # 步骤 5: 点云采样 —— 随机采样 num_point 个点
    if len(cloud_masked) >= num_point:
        # 有效点足够时，无放回采样
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        # 有效点不足时，先取全部点，再用有放回采样补齐
        idxs = np.concatenate(
            [
                np.arange(len(cloud_masked)),
                np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True),
            ],
            axis=0,
        )

    # 步骤 6: 构建 GraspNet 输入字典
    end_points = {
        # 点云坐标：转为 float32，添加 batch 维度 (1, N, 3)，移动到 GPU
        "point_clouds": torch.from_numpy(cloud_masked[idxs].astype(np.float32)[np.newaxis]).cuda(non_blocking=True),
        # 颜色：保留原始 RGB 值
        "cloud_colors": color_masked[idxs],
    }

    # 步骤 7: 构建 Open3D 点云（用于可视化，使用全部有效点）
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    o3d_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    return end_points, o3d_cloud, cloud_masked


def infer_grasps(
    net: GraspNet,
    end_points: dict,
    raw_cloud: np.ndarray,
    collision_thresh: float,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
) -> tuple[GraspGroup, dict[str, int]]:
    """执行 GraspNet 完整推理流程：前向推理 -> 解码 -> 碰撞检测过滤。

    完整推理流程：
        1. **前向推理**：网络输出抓取预测（无梯度模式）。
        2. **解码（pred_decode）**：将网络输出解码为具体抓取位姿（GraspGroup）。
        3. **碰撞检测过滤**：使用 ModelFreeCollisionDetector 剔除与场景存在
           碰撞的抓取候选。

    注意：
        collision_thresh > 0 时才执行碰撞检测。设为 0 则跳过（用于调试）。

    参数：
        net:              GraspNet 网络模型
        end_points:       输入点云字典
        raw_cloud:        原始点云 (N, 3)，用于碰撞检测的参考几何
        collision_thresh: 碰撞阈值：抓取与点云的最小允许距离（米）
        voxel_size:       碰撞检测体素尺寸（米）

    返回：
        (grasps, counts) 的元组：
        - grasps:  碰撞过滤后的抓取集合
        - counts:  推理统计字典 {"decoded": N, "pre_collision": N,
                   "collision_removed": N, "final": N}
    """
    # 步骤 1: 前向推理（禁用梯度计算以节省显存）
    with torch.no_grad():
        end_points = net(end_points)
        # 步骤 2: 解码网络输出为抓取位姿
        grasp_preds = pred_decode(end_points)

    # 将解码结果转为 GraspGroup 对象
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    decoded_count = len(gg)

    # 步骤 3: 碰撞检测过滤
    collision_removed = 0
    if len(gg) > 0 and collision_thresh > 0:
        # 创建无模型碰撞检测器（基于体素化点云的碰撞检测）
        detector = ModelFreeCollisionDetector(raw_cloud, voxel_size=voxel_size)
        # detect 返回碰撞 mask：True 表示存在碰撞
        collision_mask = detector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        collision_removed = int(np.count_nonzero(collision_mask))
        # 保留无碰撞的抓取（~ 取反）
        gg = gg[~collision_mask]

    return gg, {
        "decoded": decoded_count,
        "pre_collision": decoded_count,
        "collision_removed": collision_removed,
        "final": len(gg),
    }


def filter_grasps_by_bbox(
    grasps: GraspGroup,
    bbox_xyxy: tuple[int, int, int, int],
    K: np.ndarray,
    *,
    margin_px: int = 0,
    expand_ratio: float = 1.0,
    image_shape: Optional[tuple[int, int]] = None,
) -> GraspGroup:
    """根据图像空间的边界框过滤抓取候选。

    过滤策略：
        1. 将每个抓取的三维平移向量通过相机内参投影到图像平面。
        2. 检查投影点 (u, v) 是否落在 bbox 范围内。
        3. 支持 margin（像素扩展）和 expand_ratio（比例扩展）来扩大/缩小过滤区域。

    算法步骤：
        - 提取所有抓取的 translation (x, y, z)
        - 计算投影：u = fx * x/z + cx,  v = fy * y/z + cy
        - 无效深度点（z <= 0）标记为 NaN，自动过滤
        - 以 bbox 中心为中心，按 expand_ratio 缩放宽高
        - 应用 margin_px 外扩
        - 裁剪到图像边界

    参数：
        grasps:        抓取候选集合
        bbox_xyxy:     目标边界框 (x1, y1, x2, y2)，像素坐标
        K:             3x3 相机内参矩阵
        margin_px:     边界框外扩像素数（默认 0）
        expand_ratio:  边界框扩展比例（>1 扩大，<1 缩小，默认 1.0 不缩放）
        image_shape:   图像尺寸 (H, W)，用于裁剪边界框到图像范围

    返回：
        过滤后的抓取集合（只保留投影点落在扩展 bbox 内的抓取）
    """
    if len(grasps) == 0:
        return grasps

    # 提取所有抓取的平移向量
    translations = np.asarray(grasps.translations, dtype=np.float64)
    z = translations[:, 2]
    # 有效深度判定：z > 1e-6 防止除以零
    valid_z = z > 1e-6
    # 初始化投影坐标数组，无效深度的点设为 NaN（后续过滤时自动排除）
    u = np.full(len(grasps), np.nan, dtype=np.float64)
    v = np.full(len(grasps), np.nan, dtype=np.float64)
    # 计算相机投影：u = fx * x / z + cx
    u[valid_z] = float(K[0, 0]) * translations[valid_z, 0] / z[valid_z] + float(K[0, 2])
    # v = fy * y / z + cy
    v[valid_z] = float(K[1, 1]) * translations[valid_z, 1] / z[valid_z] + float(K[1, 2])

    # 解析原始 bbox
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    # 计算中心点
    expand_ratio = max(1.0, float(expand_ratio))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    # 按比例扩展宽高
    half_w = 0.5 * max(1.0, x2 - x1) * expand_ratio
    half_h = 0.5 * max(1.0, y2 - y1) * expand_ratio
    # 计算扩展后的边界，并应用 margin
    x1 = int(round(cx - half_w)) - int(margin_px)
    y1 = int(round(cy - half_h)) - int(margin_px)
    x2 = int(round(cx + half_w)) + int(margin_px)
    y2 = int(round(cy + half_h)) + int(margin_px)
    # 裁剪到图像边界
    if image_shape is not None:
        h, w = image_shape
        x1 = int(np.clip(x1, 0, max(0, w - 1)))
        x2 = int(np.clip(x2, 0, max(0, w - 1)))
        y1 = int(np.clip(y1, 0, max(0, h - 1)))
        y2 = int(np.clip(y2, 0, max(0, h - 1)))

    # 过滤条件：有效深度 AND 投影点落在扩展 bbox 内
    keep = valid_z & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return grasps[keep]


def filter_grasps_by_width(grasps: GraspGroup, max_width_m: Optional[float]) -> GraspGroup:
    """根据夹爪张开宽度过滤抓取候选。

    只保留抓取宽度（grasp.width）不超过 max_width_m 的候选。
    这用于排除夹爪无法容纳的过大物体抓取。

    参数：
        grasps:       抓取候选集合
        max_width_m:  最大允许的抓取宽度（米）

    返回：
        过滤后的抓取集合
    """
    # 未配置上限或集合为空时不创建新集合，直接保持输入结果。
    if max_width_m is None or len(grasps) == 0:
        return grasps
    # widths 与候选逐项对应，布尔索引只保留真实夹爪能够张开的宽度。
    return grasps[np.asarray(grasps.widths, dtype=np.float64) <= float(max_width_m)]


def select_best_grasp(grasps: GraspGroup) -> Optional[Grasp]:
    """从抓取候选中选择最优抓取：NMS 去重 -> 按分数排序 -> 取第一名。

    选择策略：
        1. 深拷贝抓取集合，避免修改原数据。
        2. 执行 NMS（非极大值抑制）去除空间上重叠的重复抓取。
        3. 按分数降序排列（分数越高表示抓取质量越高）。
        4. 返回分数最高的抓取。

    参数：
        grasps: 抓取候选集合

    返回：
        最优抓取对象，集合为空时返回 None
    """
    if len(grasps) == 0:
        return None
    # 深拷贝抓取集合
    ranked = GraspGroup(grasps.grasp_group_array.copy())
    try:
        # NMS 去重：消除空间上高度重叠的抓取
        ranked = ranked.nms()
    except Exception as exc:
        print(f"[WARN] GraspNet NMS skipped: {exc}")
    # 按分数降序排列
    ranked.sort_by_score()
    return ranked[0] if len(ranked) > 0 else None


def select_target(detections: list[YoloDetection], target_class: Optional[str]) -> Optional[YoloDetection]:
    """从 YOLO 检测结果中选择目标抓取对象。

    目标匹配策略（不区分大小写）：
        1. 如果未指定 target_class，返回置信度最高的检测。
        2. 如果指定了 target_class：
           a. 先精确匹配类别名（casefold 比较）。
           b. 若无精确匹配，退而求其次匹配包含关系（target_class 是类别名的子串）。
           c. 精确匹配优先于包含匹配。
        3. 在候选列表中返回置信度最高的。

    注意：该函数不会检查类别名的有效性，仅做文本匹配。

    参数：
        detections:   YOLO 检测结果列表
        target_class: 目标类别名（如 "apple", "cup"），为 None 时选择置信度最高者

    返回：
        选中的目标检测对象，未找到则返回 None
    """
    if not detections:
        return None
    candidates = detections
    if target_class:
        # 将目标类别名转小写用于不区分大小写比较
        target_norm = target_class.casefold()
        # 精确匹配：类别名完全等于目标
        exact = [target for target in detections if target.class_name.casefold() == target_norm]
        # 包含匹配：目标类别名是检测类别名的子串
        contains = [target for target in detections if target_norm in target.class_name.casefold()]
        # 精确匹配优先，无精确匹配时使用包含匹配
        candidates = exact or contains
    if not candidates:
        return None
    # 在候选列表中返回置信度最高的
    return max(candidates, key=lambda target: target.conf)


def selected_target_text(selected: Optional[YoloDetection], target_class: Optional[str]) -> str:
    """生成选中目标的简短描述文本。

    参数：
        selected:     选中的目标检测对象
        target_class: 目标类别名

    返回：
        格式化文本，如 "target=apple 0.95" 或 "target=cup not found"
    """
    if selected is None:
        return f"target={target_class or 'best'} not found"
    return f"target={selected.class_name} {selected.conf:.2f}"


def target_status_text(selected: Optional[YoloDetection], detections: list[YoloDetection], target_class: Optional[str]) -> str:
    """生成目标检测状态的完整文本（含检测总数）。

    参数：
        selected:     选中的目标检测对象
        detections:   全部检测结果列表
        target_class: 目标类别名

    返回：
        状态文本，如 "target=apple 0.95 detections=3"
    """
    # 按“已选中 -> 指定类别未找到 -> 场景无目标”区分三种状态，方便窗口直接显示原因。
    if selected is not None:
        return f"target={selected.class_name} {selected.conf:.2f} detections={len(detections)}"
    if target_class:
        return f"target={target_class} not found detections={len(detections)}"
    return f"target not found detections={len(detections)}"


def draw_detections_overlay(
    frame: np.ndarray,
    detections: list[YoloDetection],
    selected: Optional[YoloDetection],
    target_class: Optional[str],
) -> np.ndarray:
    """在图像帧上绘制检测框叠加层。

    绘制规则：
        - 普通检测：橙色边框 (0, 185, 255)，线宽 2
        - 选中目标：绿色边框 (0, 255, 80)，线宽 3，标签前加 "TARGET " 前缀
        - 每个检测框上方显示类别名和置信度
        - 标签下方有黑色半透明背景以确保可读性

    参数：
        frame:        原始 BGR 图像帧
        detections:   检测结果列表
        selected:     被选中的目标检测（会高亮显示）
        target_class: 目标类别名

    返回：
        绘制了检测叠加层的新图像帧（不修改原图）
    """
    # 复制帧以避免修改原图
    display = frame.copy()
    selected_key = None
    # 记录选中目标的唯一标识 (result_index, detection_index)
    if selected is not None:
        selected_key = (selected.result_index, selected.detection_index)
    for target in detections:
        # 判断当前检测是否为选中目标
        is_selected = selected_key == (target.result_index, target.detection_index)
        # 颜色：选中为绿色，否则为橙色
        color = (0, 255, 80) if is_selected else (0, 185, 255)
        thickness = 3 if is_selected else 2
        x1, y1, x2, y2 = target.bbox_xyxy
        # 绘制矩形框
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        # 构造标签文本
        label = f"{target.class_name} {target.conf:.2f}"
        if target_class and is_selected:
            label = f"TARGET {label}"
        # 绘制标签背景（黑色填充矩形）
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - label_size[1] - 8)
        cv2.rectangle(display, (x1, bg_y1), (x1 + label_size[0] + 8, y1), (0, 0, 0), -1)
        # 绘制标签文字
        cv2.putText(display, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return display


def draw_best_grasp_projection(display: np.ndarray, grasp: Optional[Grasp], K: np.ndarray) -> None:
    """在图像帧上绘制最优抓取位姿的投影点。

    通过相机内参将三维抓取中心投影到二维图像平面，
    绘制十字标记和分数/宽度信息。

    注意：该函数直接在 display 数组上原地绘制，不返回新数组。

    参数：
        display: 待绘制的图像帧（原地修改）
        grasp:   最优抓取对象
        K:       3x3 相机内参矩阵
    """
    if grasp is None:
        return
    x, y, z = [float(v) for v in grasp.translation]
    if z <= 1e-6:
        return
    # 相机投影：u = fx * x / z + cx
    u = int(round(float(K[0, 0]) * x / z + float(K[0, 2])))
    v = int(round(float(K[1, 1]) * y / z + float(K[1, 2])))
    # 确保投影点在图像范围内
    if 0 <= u < display.shape[1] and 0 <= v < display.shape[0]:
        # 绘制红色十字标记
        cv2.drawMarker(display, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        # 显示分数和抓取宽度信息
        label = f"best score={grasp.score:.2f} width={grasp.width * 100:.1f}cm"
        cv2.putText(display, label, (u + 10, max(24, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def draw_grasp_projections(
    display: np.ndarray,
    grasps: GraspGroup,
    K: np.ndarray,
    *,
    top_k: int = 30,
) -> int:
    """把 GraspNet 候选抓取投影到 RGB 图像上。

    Open3D 能看 3D 夹爪姿态，但调试桌面抓取时经常需要直接在相机画面里
    检查候选是否落在目标物体上。这里按 score 排序取前 top_k 个候选：
      - 绿色短线：候选抓取宽度方向，长度对应 GraspNet 给出的 width；
      - 绿色圆点：抓取中心；
      - 数字：候选排名，方便和 Open3D 里的 top_k 对照。

    该函数只做可视化，不改变 GraspGroup，也不参与抓取选择。
    """
    if grasps is None or len(grasps) == 0:
        return 0

    draw_grasps = GraspGroup(grasps.grasp_group_array.copy())
    try:
        draw_grasps = draw_grasps.nms()
    except Exception as exc:
        print(f"[WARN] GraspNet projection NMS skipped: {exc}")
    draw_grasps.sort_by_score()
    draw_grasps = draw_grasps[: max(1, int(top_k))]

    h, w = display.shape[:2]
    drawn = 0
    for rank in range(len(draw_grasps)):
        grasp = draw_grasps[rank]
        center = np.asarray(grasp.translation, dtype=np.float64).reshape(3)
        if center[2] <= 1e-6:
            continue

        # GraspNet rotation_matrix 的第 2 列作为夹爪开合方向；正负方向等价。
        open_axis = np.asarray(grasp.rotation_matrix, dtype=np.float64)[:, 1]
        half_width = max(0.01, float(grasp.width) * 0.5)
        p0 = center - open_axis * half_width
        p1 = center + open_axis * half_width

        projected = []
        for point in (p0, center, p1):
            if point[2] <= 1e-6:
                break
            u = int(round(float(K[0, 0]) * point[0] / point[2] + float(K[0, 2])))
            v = int(round(float(K[1, 1]) * point[1] / point[2] + float(K[1, 2])))
            projected.append((u, v))
        if len(projected) != 3:
            continue

        (u0, v0), (uc, vc), (u1, v1) = projected
        if not (0 <= uc < w and 0 <= vc < h):
            continue

        color = (70, 255, 120)
        cv2.line(display, (u0, v0), (u1, v1), color, 2, cv2.LINE_AA)
        cv2.circle(display, (uc, vc), 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            display,
            str(rank + 1),
            (uc + 4, max(14, vc - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        drawn += 1

    if drawn:
        cv2.putText(
            display,
            f"GraspNet candidates: {drawn}",
            (10, max(24, h - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (70, 255, 120),
            2,
            cv2.LINE_AA,
        )
    return drawn


def grasp_to_base_poses(
    grasp: Grasp,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """将相机坐标系下的抓取位姿转换为机器人基座坐标系下的位姿集合。

    变换链包含三个关键位姿：
        1. 抓取位姿（grasp pose）：夹爪闭合抓取物体的位置
        2. 预抓取位姿（pre-grasp pose）：抓取前上方偏移位置，用于安全接近
        3. 回退位姿（retreat pose）：抓取后回退到的安全位置

    该函数内部调用 transform_grasp_pose_to_base_with_retreat 完成完整变换链：
        - 首先将抓取旋转矩阵从 GraspNet 坐标系转为 reBot TCP 坐标系
        - 然后应用 T_cam2base 变换到机器人基座坐标系
        - 最后生成预抓取和回退位姿

    参数：
        grasp:             GraspNet 抓取对象
        T_cam2base:        相机 -> 基座的 4x4 齐次变换矩阵
        pregrasp_offset_m: 预抓取位置沿抓取方向的后退距离（米），正数沿抓取方向后退
        retreat_offset_m:  抓取后向上回退的距离（米）
        insertion_depth_m: 插入深度（米），控制夹爪沿抓取方向多插入多少

    返回：
        (grasp_pose, pre_grasp_pose, retreat_pose) 的三元组，
        每个位姿为 6 元素元组 (x, y, z, roll, pitch, yaw)，单位为米和弧度
    """
    # GraspNet translation 在相机系中，rotation_matrix 使用 GraspNet 夹爪轴约定；
    # 先转换 TCP 方向，再由共享函数完成 camera -> base 和前后偏移。
    return transform_grasp_pose_to_base_with_retreat(
        np.asarray(grasp.translation, dtype=np.float64),
        graspnet_rotation_to_rebot_tcp_rotation(grasp.rotation_matrix),
        T_cam2base,
        pregrasp_offset_m,
        retreat_offset_m,
        insertion_depth_m,
    )


def draw_status(
    frame: np.ndarray,
    status: str,
    target_status: str = "",
    frozen: bool = False,
    title: str = "GraspNet Full-Scene Demo",
) -> np.ndarray:
    """在图像帧上绘制状态信息叠加层。

    显示内容包括：
        - 标题栏（应用名称）
        - 快捷键提示
        - YOLO 目标检测状态
        - GraspNet 推理状态
        - 冻结状态标识（[FROZEN] 标签）

    参数：
        frame:         原始图像帧
        status:        GraspNet 推理状态文本
        target_status: 目标检测状态文本
        frozen:        是否处于冻结状态（暂停更新）
        title:         窗口标题

    返回：
        绘制了状态信息的新图像帧（不修改原图）
    """
    display = frame.copy()
    lines = [
        title,
        "G/SPACE: infer   R: resume   Q/ESC: quit",
    ]
    if target_status:
        lines.append(target_status)
    lines.append(status)
    y = 28
    # 逐行绘制文字（带黑色描边提高可读性）
    for line in lines:
        # 先画黑色描边（粗体，作为背景）
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        # 再画白色文字（细体，作为前景）
        cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    # 冻结状态标识
    if frozen:
        cv2.putText(display, "[FROZEN]", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
    return display


def infer_frame(
    net: GraspNet,
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    *,
    num_point: int,
    min_depth: float,
    max_depth: float,
    collision_thresh: float,
    voxel_size: float = DEFAULT_VOXEL_SIZE,
    yolo_model: Optional[Any] = None,
    yolo_opts: Optional[dict[str, Any]] = None,
    target_class: Optional[str] = None,
    target_margin_px: int = 0,
    target_expand_ratio: float = 1.0,
    max_grasp_width_m: Optional[float] = None,
) -> GraspNetFrameResult:
    """执行完整的单帧 GraspNet 推理流程。

    完整帧推理流程（按顺序执行）：
        1. **YOLO 目标检测**（可选）：
           - 如果提供了 yolo_model，运行 detect_objects 获取检测结果。
           - 调用 select_target 从中选择目标对象。
           - 如果指定了 target_class 但未找到匹配目标，跳过抓取推理。

        2. **点云构建**：
           - 从 RGB-D 图像和相机内参构建 GraspNet 输入点云。

        3. **抓取推理**：
           - 前向推理 + 解码 + 碰撞检测（infer_grasps）。

        4. **Bbox 过滤**（仅当有 YOLO 目标时）：
           - 使用 filter_grasps_by_bbox 将抓取限制在目标周围。

        5. **宽度过滤**：
           - 使用 filter_grasps_by_width 排除夹爪无法容纳的抓取。

        6. **最优选择**：
           - 使用 select_best_grasp 选出 NMS+排序后的最优抓取。

        7. **组装结果**：
           - 将各阶段结果组装为 GraspNetFrameResult 返回。

    参数：
        net:                GraspNet 网络模型
        color_bgr:          BGR 颜色图像
        depth_mm:           深度图像（毫米）
        K:                  3x3 相机内参矩阵
        num_point:          采样点数
        min_depth:          最小有效深度（米）
        max_depth:          最大有效深度（米）
        collision_thresh:   碰撞检测阈值
        voxel_size:         体素尺寸
        yolo_model:         YOLO 模型实例（可选）
        yolo_opts:          YOLO 运行参数
        target_class:       目标类别名
        target_margin_px:   bbox 过滤的像素外扩
        target_expand_ratio: bbox 扩展比例
        max_grasp_width_m:  最大允许抓取宽度（米）

    返回：
        包含完整推理结果的 GraspNetFrameResult 对象
    """
    # 尝试相对导入 detect_objects
    try:
        from .yolo_utils import detect_objects
    except ImportError:
        from yolo_utils import detect_objects

    # 初始化变量
    detections: list[YoloDetection] = []
    selected_target: Optional[YoloDetection] = None
    target_label = "full scene"

    # 步骤 1: YOLO 目标检测（可选）
    if yolo_model is not None:
        _, detections = detect_objects(yolo_model, color_bgr, yolo_opts or {})
        selected_target = select_target(detections, target_class)
        # 如果指定了目标类别但未找到，跳过抓取推理
        if selected_target is None:
            target_status = target_status_text(selected_target, detections, target_class)
            # 构造空的返回结果
            empty = GraspGroup()
            empty_cloud = o3d.geometry.PointCloud()
            return GraspNetFrameResult(
                grasps=empty,
                pre_bbox_grasps=empty,
                bbox_grasps=empty,
                best=None,
                status=f"inference skipped: {target_status}",
                target_status=target_status,
                detections=detections,
                selected_target=None,
                o3d_cloud=empty_cloud,
                raw_cloud=np.empty((0, 3), dtype=np.float32),
            )
        target_label = f"{selected_target.class_name} {selected_target.conf:.2f}"

    # 步骤 2-3: 点云构建 + 抓取推理
    tic = time.time()
    end_points, o3d_cloud, raw_cloud = build_end_points(color_bgr, depth_mm, K, num_point, min_depth, max_depth)
    grasps, counts = infer_grasps(net, end_points, raw_cloud, collision_thresh, voxel_size)

    # 保存 bbox 过滤前的全场景抓取（用于可视化比较）
    pre_bbox_grasps = copy_grasp_group(grasps)

    # 步骤 4: Bbox 过滤（仅当有 YOLO 目标时）
    if selected_target is not None:
        grasps = filter_grasps_by_bbox(
            grasps,
            selected_target.bbox_xyxy,
            K,
            margin_px=target_margin_px,
            expand_ratio=target_expand_ratio,
            image_shape=color_bgr.shape[:2],
        )

    # 保存 bbox 过滤后的抓取（用于可视化比较）
    bbox_grasps = copy_grasp_group(grasps)

    # 步骤 5: 宽度过滤
    grasps = filter_grasps_by_width(grasps, max_grasp_width_m)

    # 步骤 6: 最优抓取选择（NMS + 分数排序）
    best = select_best_grasp(grasps)

    # 计时
    elapsed = time.time() - tic

    # 步骤 7: 生成状态描述文本
    if yolo_model is None:
        status = f"grasps={len(grasps)} decoded={counts['decoded']} inference={elapsed:.2f}s"
        target_status = "YOLO disabled: full-scene GraspNet"
    else:
        status = (
            f"{target_label} grasps={len(grasps)}/{len(bbox_grasps)}/{len(pre_bbox_grasps)} decoded={counts['decoded']} "
            f"collide={counts['collision_removed']}/{counts['pre_collision']} inference={elapsed:.2f}s"
        )
        target_status = target_status_text(selected_target, detections, target_class)

    # 组装并返回完整结果
    return GraspNetFrameResult(
        grasps=grasps,
        pre_bbox_grasps=pre_bbox_grasps,
        bbox_grasps=bbox_grasps,
        best=best,
        status=status,
        target_status=target_status,
        detections=detections,
        selected_target=selected_target,
        o3d_cloud=o3d_cloud,
        raw_cloud=raw_cloud,
    )
