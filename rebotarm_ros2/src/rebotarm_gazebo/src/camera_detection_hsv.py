"""
基于 HSV 颜色阈值的传统目标检测器。

用途：在 Gazebo 仿真环境或简单桌面场景中，通过 HSV 颜色空间阈值分割 + 轮廓检测来定位目标物体。
默认配置面向绿色方块/物体，不依赖任何深度学习模型（无需 PyTorch/YOLO），适合基础链路调试和快速原型验证。

工作原理：
  1. 将相机输入的 BGR 图像转换为 HSV 颜色空间
  2. 使用预设的 HSV 上下界阈值生成二值 mask
  3. 通过形态学操作（开运算 + 闭运算）去除噪声、填补孔洞
  4. 寻找最大外轮廓，生成目标 mask
  5. 调用 public 后处理函数 `detection_from_mask` 计算 3D 中心点坐标
"""

from __future__ import annotations  # 启用延迟类型注解求值，支持前向引用

from collections.abc import Callable  # 参数回调函数的类型注解
from typing import Any  # 任意类型占位

import cv2  # OpenCV，用于图像处理、颜色空间转换、形态学操作、轮廓检测
import numpy as np  # 数组运算，mask 和 HSV 阈值的底层数据结构

# 从本包的公共模块导入检测结果数据类和 mask 后处理函数
from rebotarm_gazebo.camera_detection_common import ObjectDetection, detection_from_mask


class HsvColorDetector:
    """传统 HSV 颜色阈值 + 轮廓检测器。

    这个后端不依赖深度学习模型，默认用于 Gazebo 绿色方块和基础链路调试。

    使用方式：
        detector = HsvColorDetector(param_callback)
        result = detector.detect(bgr_image, depth_image)
        # result 为 ObjectDetection 或 None（未检测到目标时）
    """

    def __init__(self, param: Callable[[str], Any]) -> None:
        """初始化检测器。

        Args:
            param: 参数回调函数，接收参数名（如 "hsv_lower"、"hsv_upper"）返回对应值。
                   通常来自 ROS2 节点参数系统或配置字典，允许运行时动态调整 HSV 阈值。
        """
        self._param = param  # 保存参数回调，每次 detect() 调用时动态读取最新阈值

    def detect(self, bgr: np.ndarray, depth: np.ndarray) -> ObjectDetection | None:
        """对一帧 BGR 图像 + 深度图执行 HSV 阈值检测。

        完整处理流程：
          1. BGR → HSV 颜色空间转换
          2. 按上下界阈值生成二值 mask（目标区域为 255，背景为 0）
          3. 形态学开运算去噪 + 闭运算填孔
          4. 寻找外轮廓，取面积最大者作为目标
          5. 生成目标 mask 并调用 detection_from_mask 计算 3D 位姿

        Args:
            bgr: 相机输入的 BGR 彩色图像，shape (H, W, 3)，dtype uint8。
            depth: 对齐后的深度图，shape (H, W)，dtype float32，单位米。

        Returns:
            ObjectDetection | None: 检测结果（含 3D 中心点、标签、置信度等），
                                    未找到目标时返回 None。
        """
        # 第一步：BGR → HSV 颜色空间转换
        # HSV（Hue 色调 / Saturation 饱和度 / Value 明度）对光照变化比 RGB 更鲁棒，
        # 因为色调通道将颜色信息与亮度分离，同一颜色的物体在不同光照下色调值保持相对稳定。
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # 第二步：从参数系统读取 HSV 上下界阈值
        # lower/upper 分别是 [H_min, S_min, V_min] 和 [H_max, S_max, V_max] 的列表，
        # 默认配置面向绿色物体（H 约 35-85 范围）。
        lower = np.array(self._param("hsv_lower"), dtype=np.uint8)
        upper = np.array(self._param("hsv_upper"), dtype=np.uint8)

        # cv2.inRange: 逐像素判断是否满足 lower <= pixel <= upper（三通道同时满足），
        # 满足的像素置为 255（白），不满足的置为 0（黑），输出单通道二值图像。
        mask = cv2.inRange(hsv, lower, upper)

        # 第三步：形态学操作去噪和填补
        # 开运算（先腐蚀后膨胀）：去除孤立的小白点噪声（椒盐噪声）。
        # 卷积核 (5, 5) 表示 5x5 像素的矩形结构元素，控制去噪的力度。
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # 闭运算（先膨胀后腐蚀）：填补目标区域内部的小黑洞/孔洞，
        # 使轮廓更完整连续，便于后续轮廓检测。
        # 卷积核 (7, 7) 比开运算略大，确保内部孔洞被充分填充。
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # 第四步：寻找外轮廓
        # RETR_EXTERNAL: 只检索最外层轮廓，忽略嵌套的内部轮廓，减少干扰。
        # CHAIN_APPROX_SIMPLE: 压缩水平/垂直/对角线段，只保留端点，节省内存。
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 没有找到任何轮廓 → 画面中没有匹配颜色的目标
        if not contours:
            return None

        # 取面积最大的轮廓作为检测目标（假设画面中只有一个目标物体）
        contour = max(contours, key=cv2.contourArea)

        # 第五步：生成仅含最大目标的二值 mask
        # 创建与深度图尺寸相同的全黑图像，然后仅用白色填充最大轮廓区域。
        object_mask = np.zeros(depth.shape[:2], dtype=np.uint8)  # shape = (H, W)
        cv2.drawContours(object_mask, [contour], -1,  # -1 表示绘制 all 轮廓（此处仅一个）
                          255,  # 填充颜色：白色 (255)
                          thickness=cv2.FILLED)  # 填充轮廓内部而非仅描边

        # 第六步：调用公共后处理函数，计算 3D 空间位姿
        # detection_from_mask 利用深度图和相机内参，将 2D mask 投影到 3D 相机坐标系，
        # 返回包含中心点坐标、OBB 包围盒、标签、置信度等信息的 ObjectDetection 结构。
        return detection_from_mask(
            mask=object_mask,        # 目标区域的二值 mask
            depth=depth,             # 对齐的深度图，用于提取目标区域的 3D 深度值
            contour=contour,         # 轮廓多边形，用于计算 OBB（旋转最小包围矩形）
            backend="hsv",           # 标记检测后端类型，便于下游区分不同检测方法
            label="hsv_target",      # 检测目标的语义标签
            confidence=1.0,          # 传统方法无置信度概念，固定为 1.0
            param=self._param,       # 传递参数回调，后处理中可能需要相机内参等参数
        )
