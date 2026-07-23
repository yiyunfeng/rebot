"""Orbbec DaBai DCW 相机驱动。

OrbbecDaBaiDCW 类：
  open() / close(): 管理 pyorbbecsdk Pipeline 的生命周期.
  get_frame(): 返回深度对齐到彩色图像的 BGR 和深度毫米帧.
  K / D: 从 SDK 读取的相机内参矩阵和畸变系数.

打开流程概览：
  1. _preload_orbbec_sdk_libs(): 预加载 Orbbec 原生 shared library，
     解决 SDK 编译时 RUNPATH 与当前运行环境路径不一致的问题.
  2. 导入 pyorbbecsdk 并抑制 SDK 初始化期间的 stderr 噪声.
  3. 创建 Pipeline 实例，选择 RGB 和 Depth 流配置.
  4. 启用 D2C (Depth-to-Color) 硬件对齐，深度图像映射到 RGB 网格.
  5. 从 SDK 读取工厂标定（相机内参 + 畸变系数）.
"""

from __future__ import annotations

import os
import ctypes
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import CameraDriver, CameraFrameError


class OrbbecDaBaiDCW(CameraDriver):
    """Orbbec DaBai DCW RGB-D 相机驱动，基于 pyorbbecsdk。

    DaBai DCW 是一款低成本 RGB-D 相机，通过 USB 连接，常用于眼在手 (eye-in-hand)
    抓取场景。本驱动封装了：
      - SDK shared library 预加载
      - D2C (深度到彩色) 对齐
      - 工厂标定读取与回退策略
      - 多种像素格式到 BGR 的转换
      - 深度原始值到毫米的转换
    """

    def __init__(
        self,
        color_width: int = 640,
        color_height: int = 480,
        fps: int = 30,
        calib_dir: Optional[str] = None,
        depth_width: Optional[int] = None,
        depth_height: Optional[int] = None,
    ) -> None:
        """保存期望的流参数；真正连接设备和读取内参在 :meth:`open` 中完成。"""
        # RGB 彩色流参数
        self._cw = int(color_width)
        self._ch = int(color_height)
        # Depth 深度流参数（默认与彩色流相同）
        self._dw = int(depth_width or color_width)
        self._dh = int(depth_height or color_height)
        self._fps = int(fps)
        # 本地标定目录（可选，优先使用本地 OpenCV 标定结果）
        self._calib_dir = Path(calib_dir) if calib_dir else None

        # SDK Pipeline 实例
        self._pipeline = None
        # 深度缩放因子：原始值 * depth_scale_mm = 毫米
        self._depth_scale_mm: float = 1.0
        # 相机内参矩阵 K (3x3) 和畸变系数 D
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._aruco = None
        self._reset_frame_failures()

    # ==========================================
    # 生命周期：打开/关闭
    # ==========================================

    def open(self) -> None:
        """打开 DaBai DCW 的彩色和深度流。

        完整打开流程：
          1. 预加载 Orbbec SDK 原生 shared library.
          2. 导入 pyorbbecsdk 并抑制初始化噪声.
          3. 创建 Pipeline，选择合适的分辨率/格式配置.
          4. 启用 D2C 深度对齐（优先硬件对齐，回退软件对齐）.
          5. 从 SDK 读取工厂标定内参和畸变系数.

        异常：
            RuntimeError: 相机未连接或无权限访问时抛出.
        """
        # 步骤 1: 预加载 SDK 原生库，解决 RUNPATH 不匹配问题
        self._preload_orbbec_sdk_libs()
        try:
            from pyorbbecsdk import (
                Config,
                Context,
                OBAlignMode,
                OBFormat,
                OBSensorType,
                Pipeline,
            )
        except ImportError as e:
            raise RuntimeError(
                "pyorbbecsdk is not installed in the active environment.\n"
                "  Activate conda env: conda activate rebotarm\n"
                "  Then install the local SDK: cd sdk/pyorbbecsdk && pip install -e ."
            ) from e

        # Orbbec 原生库在启动时会向 stderr 输出大量调试信息。
        # 重定向 stderr 到 /dev/null 以保持终端清洁，同时保留真实的 Python 异常输出。
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)

        try:
            # 设置 SDK 日志级别为 FATAL，进一步减少噪声
            try:
                from pyorbbecsdk import OBLogSeverity
                Context().set_logger_severity(OBLogSeverity.FATAL)
            except Exception:
                pass

            # 步骤 2: 创建 Pipeline（相机连接入口）
            try:
                self._pipeline = Pipeline()
            except Exception as e:
                raise RuntimeError(
                    f"Orbbec DaBai DCW not found: {e}\n"
                    "  Check USB connection, udev rules, and whether another process owns the camera.\n"
                    "  Permission quick fix: sudo chmod a+rw /dev/bus/usb/*/*"
                ) from e

            # 验证设备型号是否为 DaBai
            self._warn_if_unexpected_device()
            cfg = Config()

            # 步骤 3: 选择彩色和深度流配置（分辨率 + 像素格式 + 帧率）
            color_profile = self._select_color_profile(OBSensorType, OBFormat)
            depth_profile = self._select_depth_profile(OBSensorType, OBFormat)
            cfg.enable_stream(color_profile)
            cfg.enable_stream(depth_profile)

            # 步骤 4: DaBai DCW 用于眼在手 RGB-D 相机，深度必须映射到
            # RGB 图像网格上，下游 OpenCV 抓取代码才能按 RGB 像素索引深度值。
            # 优先尝试硬件对齐 (HW_MODE)，失败则回退到软件对齐 (SW_MODE).
            self._start_with_alignment(cfg, OBAlignMode)

            # 可选：启用帧同步（如果支持）
            try:
                self._pipeline.enable_frame_sync()
            except Exception:
                pass

            self._reset_frame_failures()
            # 步骤 5: 从 SDK 读取工厂标定（相机内参 + 畸变）
            self._read_sdk_calibration()
            print(
                "[OrbbecDaBaiDCW] Ready "
                f"(color={self._cw}x{self._ch}@{self._fps}, "
                f"depth={self._dw}x{self._dh}@{self._fps})"
            )
        finally:
            # 恢复 stderr
            os.dup2(saved, 2)
            os.close(saved)

    def close(self) -> None:
        """关闭相机 Pipeline，停止流传输。"""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    # ==========================================
    # 帧获取
    # ==========================================

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取一帧对齐后的 RGB 和深度图像。

        返回:
            (color_bgr, depth_mm):
              - color_bgr: uint8 BGR 彩色图像 (H, W, 3).
              - depth_mm: uint16 深度图像 (H, W)，单位为毫米.

        流程:
          1. 等待下一组同步帧（超时 500ms）.
          2. 分离彩色帧和深度帧.
          3. 彩色帧按原始像素格式转换为 BGR.
          4. 深度帧原始值 * depth_scale 转换为毫米.
          5. 如果 D2C 对齐后深度图尺寸与彩色图不一致，
             用最近邻插值将深度图缩放到彩色图尺寸.
        """
        if self._pipeline is None:
            return None, None
        try:
            from pyorbbecsdk import OBFormat
            # 等待下一组帧，超时 500ms
            frames = self._pipeline.wait_for_frames(500)
            if frames is None:
                self._record_frame_failure("wait_for_frames timeout")
                return None, None

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                self._record_frame_failure("missing color or depth frame")
                return None, None

            # 格式转换：彩色 -> BGR，深度 -> 毫米
            color_bgr = self._color_frame_to_bgr(color_frame, OBFormat)
            depth_mm = self._depth_frame_to_mm(depth_frame)
            if color_bgr is None or depth_mm is None:
                self._record_frame_failure("failed to decode color or depth frame")
                return color_bgr, depth_mm

            # D2C 对齐后深度图尺寸应与彩色图一致，但某些 SDK/配置组合
            # 仍返回原始深度流尺寸。这里做最终尺寸匹配：用最近邻插值
            # 将深度图缩放至彩色图分辨率，确保按 RGB 像素索引深度正确.
            if depth_mm.shape[:2] != color_bgr.shape[:2]:
                depth_mm = cv2.resize(
                    depth_mm,
                    (color_bgr.shape[1], color_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            self._reset_frame_failures()
            return color_bgr, depth_mm
        except CameraFrameError:
            raise
        except Exception as exc:
            self._record_frame_failure(str(exc))
            return None, None

    # ==========================================
    # 内参属性
    # ==========================================

    @property
    def K(self) -> np.ndarray:
        """相机内参矩阵 K (3x3)，格式: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]."""
        if self._K is None:
            raise RuntimeError("Camera is not open")
        return self._K

    @property
    def D(self) -> np.ndarray:
        """畸变系数向量 (1, N)，包含 k1, k2, p1, p2, k3 等."""
        if self._D is None:
            raise RuntimeError("Camera is not open")
        return self._D

    # ==========================================
    # 内部实现
    # ==========================================

    def _warn_if_unexpected_device(self) -> None:
        """检查设备名称是否包含 "dabai"，不匹配时打印警告。"""
        try:
            info = self._pipeline.get_device().get_device_info()
            name = str(info.get_name())
            serial = str(info.get_serial_number())
            firmware = str(info.get_firmware_version())
            print(f"[OrbbecDaBaiDCW] Device: {name}, serial={serial}, firmware={firmware}")
            lowered = name.lower()
            if "dabai" not in lowered and "da bai" not in lowered:
                print(f"[OrbbecDaBaiDCW] WARN: expected DaBai DCW, got {name!r}")
        except Exception:
            pass

    def _preload_orbbec_sdk_libs(self) -> None:
        """预加载 Orbbec SDK 原生 shared library。

        背景：
          pyorbbecsdk 编译时在另一台机器上设置了 RUNPATH，指向那个机器
          的 SDK 安装路径。在当前项目 checkout 目录下运行时，动态链接器
          找不到这些 .so 文件。

        解决方案：
          在导入 pyorbbecsdk 之前，用 ctypes.CDLL 按绝对路径预加载
          三个关键原生库到全局符号表 (RTLD_GLOBAL)：
            - libdepthengine.so.2.0  (深度引擎)
            - libob_usb.so           (USB 设备通信)
            - libOrbbecSDK.so.1.10   (Orbbec SDK 核心)

        RTLD_GLOBAL 标志确保这些符号对后续 import pyorbbecsdk 可见。
        """
        # 原生库只在项目内置 SDK 目录存在时预加载；系统安装版不走这条路径。
        sdk_lib_dir = Path(__file__).resolve().parents[2] / "sdk" / "pyorbbecsdk" / "install" / "lib"
        if not sdk_lib_dir.exists():
            return
        # 依赖顺序从底层深度/USB 到 SDK 主库，后加载的库可以找到前面的全局符号。
        for lib_name in ("libdepthengine.so.2.0", "libob_usb.so", "libOrbbecSDK.so.1.10"):
            lib_path = sdk_lib_dir / lib_name
            if lib_path.exists():
                try:
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

    def _select_color_profile(self, OBSensorType, OBFormat):
        """选择彩色流配置：按优先级尝试格式列表，匹配指定分辨率和帧率。

        像素格式优先级: YUYV > BGRA > RGB > BGR > MJPG.
        DaBai DCW 通常提供压缩 MJPG 和原始 YUV/RGB 格式.
        实测当前 USB2 + pyorbbecsdk 组合下 MJPG 会返回彩色帧但 OpenCV
        解码失败，因此优先使用未压缩格式，保证 OpenCV 后续处理稳定.

        所有格式都尝试失败后，使用 SDK 默认配置.
        """
        profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        # 按优先级尝试各像素格式
        for fmt_name in ("YUYV", "BGRA", "RGB", "BGR", "UYVY", "MJPG"):
            fmt = getattr(OBFormat, fmt_name, None)
            if fmt is None:
                continue
            try:
                profile = profiles.get_video_stream_profile(self._cw, self._ch, fmt, self._fps)
                # 更新实际分辨率（SDK 可能返回与请求不同的值）
                self._cw = int(profile.get_width())
                self._ch = int(profile.get_height())
                self._fps = int(profile.get_fps())
                return profile
            except Exception:
                pass
        # 回退：使用默认彩色流配置
        profile = profiles.get_default_video_stream_profile()
        self._cw = int(profile.get_width())
        self._ch = int(profile.get_height())
        self._fps = int(profile.get_fps())
        return profile

    def _select_depth_profile(self, OBSensorType, OBFormat):
        """选择深度流配置：Y16 格式（16 位无符号整数，值 = 深度单位 * depth_scale）。"""
        profiles = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            profile = profiles.get_video_stream_profile(self._dw, self._dh, OBFormat.Y16, self._fps)
        except Exception:
            profile = profiles.get_default_video_stream_profile()
        self._dw = int(profile.get_width())
        self._dh = int(profile.get_height())
        return profile

    def _start_with_alignment(self, cfg, OBAlignMode) -> None:
        """启用 D2C（深度到彩色）对齐并启动 Pipeline。

        优先尝试硬件对齐 (HW_MODE)，失败则回退到软件对齐 (SW_MODE)。
        如果两种对齐模式都失败，抛出 RuntimeError.

        深度到彩色对齐的含义：
          将深度图像的每个像素重新映射到彩色图像的像素网格上，
          使得 depth[u, v] 和 color[u, v] 对应场景中的同一点。
        """
        errors: list[str] = []
        for mode_name in ("HW_MODE", "SW_MODE"):
            # 硬件对齐速度快且占用 CPU 少；设备不支持时再尝试软件对齐。
            mode = getattr(OBAlignMode, mode_name)
            try:
                cfg.set_align_mode(mode)
                self._pipeline.start(cfg)
                print(f"[OrbbecDaBaiDCW] D2C alignment: {mode_name}")
                return
            except Exception as exc:
                errors.append(f"{mode_name}: {exc}")
        raise RuntimeError(
            "Failed to start DaBai DCW with depth-to-color alignment:\n  "
            + "\n  ".join(errors)
        )

    def _read_sdk_calibration(self) -> None:
        """从 SDK 读取工厂标定的相机内参和畸变系数。

        流程：
          1. 通过 pipeline.get_camera_param() 读取当前流的标定参数.
          2. 如果内参为空（fx<=0 或 width<=0），则从设备标定列表中选择
             最匹配当前 RGB 流分辨率的一条工厂标定.
          3. 读取畸变系数（优先本地 OpenCV 标定文件，回退 SDK 工厂值）.
        """
        param = self._pipeline.get_camera_param()
        intr = param.rgb_intrinsic
        # 某些 DaBai DCW 固件/SDK 组合：活动 Pipeline 返回空标定，
        # 但设备确实提供有效工厂标定列表。检测零矩阵并从列表中选取.
        if float(intr.fx) <= 0.0 or int(intr.width) <= 0:
            param = self._select_calibration_param()
            intr = param.rgb_intrinsic
        # 构建内参矩阵 K (3x3)
        self._K = np.array([
            [intr.fx, 0.0,     intr.cx],
            [0.0,     intr.fy, intr.cy],
            [0.0,     0.0,     1.0    ],
        ], dtype=np.float64)
        self._D = self._load_distortion_from_sdk(param)

    def _select_calibration_param(self):
        """从设备工厂标定列表中选择与当前 RGB 流分辨率最匹配的一条。

        遍历设备所有标定条目，计算每条标定的 RGB 分辨率与当前流分辨率
        的曼哈顿距离 (|w - cw| + |h - ch|)，选择距离最小且 fx>0 的条目.
        """
        device = self._pipeline.get_device()
        params = device.get_calibration_camera_param_list()
        count = params.get_count() if hasattr(params, "get_count") else len(params)
        if count <= 0:
            raise RuntimeError("DaBai DCW calibration list is empty")

        best_param = params.get_camera_param(0)
        best_score = float("inf")
        for index in range(count):
            param = params.get_camera_param(index)
            intr = param.rgb_intrinsic
            # 抓取代码只需要 RGB 相机模型做像素到射线的计算，
            # 所以选择 RGB 分辨率匹配当前流的标定条目.
            score = abs(int(intr.width) - self._cw) + abs(int(intr.height) - self._ch)
            if score < best_score and float(intr.fx) > 0.0:
                best_param = param
                best_score = score
        return best_param

    def _load_distortion_from_sdk(self, camera_param) -> np.ndarray:
        """读取畸变系数：优先本地 OpenCV 标定文件，否则使用 SDK 工厂值。

        策略：
          1. 如果配置了 calib_dir 且其中存在 intrinsics.npz，
             加载其中的 dist_coeffs 作为畸变系数.
          2. 回退到 SDK 工厂标定中的 rgb_distortion 字段.
        """
        if self._calib_dir is not None:
            npz_path = self._calib_dir / "intrinsics.npz"
            if npz_path.exists():
                try:
                    # 本地 OpenCV 标定更贴合当前镜头；读取失败再安全回退工厂参数。
                    data = np.load(str(npz_path))
                    return data["dist_coeffs"].reshape(1, -1).astype(np.float64)
                except Exception:
                    pass

        # SDK 将径向 k1~k6 和切向 p1/p2 分开存储，按 OpenCV 顺序组装。
        dist = camera_param.rgb_distortion
        coeffs = [
            getattr(dist, "k1", 0.0),
            getattr(dist, "k2", 0.0),
            getattr(dist, "p1", 0.0),
            getattr(dist, "p2", 0.0),
            getattr(dist, "k3", 0.0),
            getattr(dist, "k4", 0.0),
            getattr(dist, "k5", 0.0),
            getattr(dist, "k6", 0.0),
        ]
        return np.asarray(coeffs, dtype=np.float64).reshape(1, -1)

    def _color_frame_to_bgr(self, frame, OBFormat) -> Optional[np.ndarray]:
        """将 SDK 彩色帧转换为 OpenCV 兼容的 BGR uint8 图像。

        支持的像素格式：
          - MJPG:      JPEG 压缩 -> cv2.imdecode 解码
          - RGB888/RGB: RGB -> cv2.COLOR_RGB2BGR 转换
          - BGR888/BGR: 直接作为 BGR 返回（无需转换）
          - BGRA:      BGRA -> cv2.COLOR_BGRA2BGR 丢弃 Alpha 通道
          - YUYV:      YUYV YUV 4:2:2 -> cv2.COLOR_YUV2BGR_YUYV
          - UYVY:      UYVY YUV 4:2:2 -> cv2.COLOR_YUV2BGR_UYVY
          - NV12:      YUV 4:2:0 半平面 -> cv2.COLOR_YUV2BGR_NV12
          - NV21:      YUV 4:2:0 半平面 -> cv2.COLOR_YUV2BGR_NV21

        注意：
          pyorbbecsdk 返回的 frame data 可能是非连续 numpy buffer。
          OpenCV 的 imdecode/cvtColor 对内存连续性敏感，
          所以这里先通过 np.ascontiguousarray 统一成连续 uint8 数组.
        """
        width, height = frame.get_width(), frame.get_height()
        # 当前 pyorbbecsdk 在 DaBai DCW 上偶尔会让 get_data() 返回 stride=0 的
        # ndarray，直接 np.asarray 会把整帧误读成同一个字节。这里从原始指针
        # 拷贝 data_size 字节，确保 OpenCV 看到的是真实连续 buffer。
        raw = self._frame_to_uint8(frame)
        fmt = frame.get_format()
        try:
            if fmt == getattr(OBFormat, "MJPG", None):
                return cv2.imdecode(raw.reshape(-1), cv2.IMREAD_COLOR)
            if fmt in (getattr(OBFormat, "RGB888", None), getattr(OBFormat, "RGB", None)):
                return cv2.cvtColor(raw.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
            if fmt in (getattr(OBFormat, "BGR888", None), getattr(OBFormat, "BGR", None)):
                return raw.reshape(height, width, 3)
            if fmt == getattr(OBFormat, "BGRA", None):
                return cv2.cvtColor(raw.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
            if fmt == getattr(OBFormat, "YUYV", None):
                return cv2.cvtColor(raw.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUYV)
            if fmt == getattr(OBFormat, "UYVY", None):
                return cv2.cvtColor(raw.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY)
            if fmt == getattr(OBFormat, "NV12", None):
                return cv2.cvtColor(raw.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
            if fmt == getattr(OBFormat, "NV21", None):
                return cv2.cvtColor(raw.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV21)
        except Exception:
            return None
        return None

    def _depth_frame_to_mm(self, frame) -> Optional[np.ndarray]:
        """将 SDK 深度帧转换为 uint16 毫米值图像。

        转换流程：
          1. 将原始字节 buffer 转换为连续 uint8 数组.
          2. 按 little-endian uint16 解读（Y16 格式）.
          3. 获取 depth_scale（单位毫米），乘以原始值得到深度毫米值.
          4. 裁切到 [0, 65535] 范围并转为 uint16.

        注意：
          DaBai DCW 的 Y16 深度帧在 pyorbbecsdk 中通常以 uint8 buffer 暴露。
          先把原始字节连续化，再按 little-endian uint16 解释，避免
          np.frombuffer 在非 C-contiguous ndarray 上抛错。
        """
        width, height = frame.get_width(), frame.get_height()
        try:
            # 与官方示例一致，深度按 Y16 解释为 uint16；区别是这里先从
            # 原始指针拷贝，避免 get_data() 的 stride=0 异常 view。
            raw = self._frame_to_uint8(frame)
            depth_raw = np.frombuffer(raw.tobytes(), dtype=np.uint16).reshape(height, width)
            try:
                self._depth_scale_mm = float(frame.get_depth_scale())
            except Exception:
                pass
            # 原始值 * depth_scale_mm -> 毫米，四舍五入后裁切到 uint16 范围
            return np.clip(
                np.rint(depth_raw.astype(np.float32) * self._depth_scale_mm),
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
        except Exception:
            return None

    def _frame_to_uint8(self, frame) -> np.ndarray:
        """从 Orbbec Frame 原始指针拷贝连续 uint8 buffer。

        不直接使用 frame.get_data() 的原因：
          在当前 DaBai DCW + pyorbbecsdk 组合下，get_data() 有时返回
          strides=(0,) 的 ndarray，表现为整帧每个字节都相同。用
          get_data_pointer() + get_data_size() 拷贝可以绕过这个绑定层问题。
        """
        try:
            # PyCapsule 内保存 C 层 frame buffer 指针；先声明参数/返回类型再取地址。
            capsule = frame.get_data_pointer()
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"frame_data_pointer")
            # string_at 会复制指定字节数，得到生命周期独立且连续的 Python bytes。
            data = ctypes.string_at(ptr, int(frame.get_data_size()))
            return np.frombuffer(data, dtype=np.uint8)
        except Exception:
            # SDK 版本不提供指针接口时，退回官方 get_data()，并强制整理为连续数组。
            return np.ascontiguousarray(np.asanyarray(frame.get_data()), dtype=np.uint8)
