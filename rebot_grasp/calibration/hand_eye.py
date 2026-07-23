"""手眼标定 — 基于 OpenCV calibrateHandEye。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np


class CalibMode(Enum):
    """相机安装方式；当前项目只实现眼在手（相机随末端运动）。"""

    EYE_IN_HAND = "eye_in_hand"   # 相机在末端，随末端运动


_METHOD_MAP = {
    "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
    "PARK":       cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class CalibResult:
    """一次手眼标定的矩阵结果及其来源信息。"""

    T_result: np.ndarray    # (4, 4) 手眼变换矩阵
    mode: str               # CalibMode 枚举对应的字符串值
    n_samples: int
    method: str


@dataclass
class _Sample:
    """同一采样时刻的机械臂末端位姿和标记板位姿。"""

    T_gripper2base: np.ndarray   # (4, 4)
    T_marker2cam:   np.ndarray   # (4, 4)


class HandEyeCalibrator:
    """
    手眼标定器。

    Eye-in-Hand 模式：
        求解 T_cam2gripper，使得
            T_marker2base = T_gripper2base @ T_cam2gripper @ T_marker2cam
        在所有姿态下恒成立。

    使用方法：
        calib = HandEyeCalibrator(CalibMode.EYE_IN_HAND)
        calib.add_sample(T_gripper2base, T_marker2cam)
        ...
        result = calib.calibrate()
        HandEyeCalibrator.save(result, "hand_eye.npz")
    """

    def __init__(
        self,
        mode: CalibMode = CalibMode.EYE_IN_HAND,
        method: str = "TSAI",
    ) -> None:
        """创建标定器并记录求解模式；样本通过 :meth:`add_sample` 逐个加入。"""
        if mode != CalibMode.EYE_IN_HAND:
            raise ValueError("Only eye-in-hand calibration is supported")
        self._mode = mode
        self._method = method.upper()
        self._samples: List[_Sample] = []

    @property
    def n_samples(self) -> int:
        """返回当前已经保存的成对位姿数量。"""
        return len(self._samples)

    def add_sample(
        self,
        T_gripper2base: np.ndarray,
        T_marker2cam: np.ndarray,
    ) -> None:
        """
        添加一个标定样本。

        参数：
            T_gripper2base: (4,4) 末端到基座的变换（正运动学 FK 输出）
            T_marker2cam:   (4,4) 标记到相机的变换（ArUco 检测输出）
        """
        # 两个矩阵必须来自同一个采样时刻，否则机械臂姿态与相机观测无法组成方程。
        # 统一转成 float64，满足 OpenCV 标定接口的数值精度要求。
        self._samples.append(_Sample(
            T_gripper2base=np.asarray(T_gripper2base, dtype=np.float64),
            T_marker2cam=np.asarray(T_marker2cam, dtype=np.float64),
        ))

    def calibrate(self, min_samples: int = 5) -> CalibResult:
        """
        计算手眼变换。

        参数：
            min_samples: 最少样本数（< 此值会抛出异常）

        返回：
            CalibResult，T_result 即手眼变换矩阵
        """
        if self.n_samples < min_samples:
            raise ValueError(
                f"样本不足：{self.n_samples} < {min_samples}，请继续采集"
            )

        # self._method 是配置中的算法名称，例如 "PARK"。
        # OpenCV 不接收这个字符串，所以要先转换成对应的整数常量，
        # 例如 "PARK" -> cv2.CALIB_HAND_EYE_PARK。名称无效时默认使用 TSAI。
        cv_method = _METHOD_MAP.get(self._method, cv2.CALIB_HAND_EYE_TSAI)

        # OpenCV 接口：R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
        # OpenCV 不接收 4×4 矩阵列表，需要把每个样本拆成 3×3 旋转和 3×1 平移。
        R_g2b = [s.T_gripper2base[:3, :3] for s in self._samples]
        t_g2b = [s.T_gripper2base[:3,  3].reshape(3, 1) for s in self._samples]
        R_t2c = [s.T_marker2cam[:3, :3] for s in self._samples]
        t_t2c = [s.T_marker2cam[:3,  3].reshape(3, 1) for s in self._samples]

        # 根据多组“末端到基座”和“标记板到相机”的位姿，求解相机到末端的变换。
        # OpenCV 返回：
        #   R_c2g: (3, 3) 旋转矩阵
        #   t_c2g: (3, 1) 平移向量
        # 它们满足：末端坐标 = R_c2g @ 相机坐标 + t_c2g。
        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_g2b,
            t_g2b,
            R_t2c,
            t_t2c,
            # 指定使用哪一种手眼标定求解算法，例如 PARK 或 TSAI。
            method=cv_method,
        )

        # 把旋转和平移合成一个 4×4 齐次变换矩阵 T_cam2gripper：
        # [ R_c2g  t_c2g ]
        # [   0       1   ]
        # 之后可用 T_cam2gripper @ 相机齐次坐标，得到末端坐标系中的坐标。
        T_cam2gripper = np.eye(4, dtype=np.float64)
        T_cam2gripper[:3, :3] = R_c2g
        T_cam2gripper[:3, 3] = t_c2g.flatten()

        # 除变换矩阵外，一并返回标定模式、样本数量和求解方法，便于保存和检查。
        return CalibResult(
            T_result=T_cam2gripper,
            mode=self._mode.value,
            n_samples=self.n_samples,
            method=self._method,
        )

    @staticmethod
    def save(result: CalibResult, path: Union[str, Path]) -> None:
        """保存标定结果为 .npz 文件。"""
        path = Path(path)
        # 标定目录可能尚未创建，保存前递归建立；exist_ok 避免目录已存在时报错。
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            T_result=result.T_result,
            mode=np.array([result.mode]),
            n_samples=np.array([result.n_samples]),
            method=np.array([result.method]),
        )

    @staticmethod
    def load(path: Union[str, Path]) -> CalibResult:
        """从 .npz 文件加载标定结果。"""
        # 文件只包含 NumPy 数组，不允许 pickle，避免加载标定文件时执行任意对象代码。
        data = np.load(str(path), allow_pickle=False)
        return CalibResult(
            T_result=data["T_result"],
            mode=str(data["mode"][0]),
            n_samples=int(data["n_samples"][0]),
            method=str(data["method"][0]) if "method" in data else "TSAI",
        )
