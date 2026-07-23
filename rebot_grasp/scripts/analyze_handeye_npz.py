#!/usr/bin/env python3
"""离线分析 hand_eye.npz 的样本一致性。

用法：
    cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp
    python scripts/analyze_handeye_npz.py

说明：
    新版 collect_handeye_eih.py 会把每个样本的 T_gripper2base 和
    T_marker2cam 一起保存到 hand_eye.npz。这个脚本用这些样本重新计算
    marker 在 base 下的位置残差：

        T_marker2base = T_gripper2base @ T_cam2gripper @ T_marker2cam

    ArUco 标记板固定在桌面上，所以所有样本的 marker base 位置应该接近一致。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def parse_args() -> argparse.Namespace:
    """解析待分析的手眼标定 ``.npz`` 路径。"""
    parser = argparse.ArgumentParser(description="Analyze saved hand-eye calibration samples")
    parser.add_argument(
        "--path",
        default="config/calibration/orbbec_dabai_dcw/hand_eye.npz",
        help="hand_eye.npz 路径，默认使用 DaBai DCW 标定结果",
    )
    return parser.parse_args()


def solve_handeye(T_g2b: np.ndarray, T_m2c: np.ndarray, method: int) -> np.ndarray:
    """用保存的成对位姿重新求解 camera -> gripper 齐次变换。"""
    R_g2b = [T[:3, :3] for T in T_g2b]
    t_g2b = [T[:3, 3].reshape(3, 1) for T in T_g2b]
    R_t2c = [T[:3, :3] for T in T_m2c]
    t_t2c = [T[:3, 3].reshape(3, 1) for T in T_m2c]
    R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_c2g
    T[:3, 3] = t_c2g.reshape(3)
    return T


def residual_stats(T: np.ndarray, T_g2b: np.ndarray, T_m2c: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """计算每个样本的 marker -> base 位置及其相对均值的毫米级残差。"""
    # 每组样本都把同一个固定标记变换到 base；理想情况下这些矩阵的平移应完全相同。
    T_m2b = np.asarray([g @ T @ m for g, m in zip(T_g2b, T_m2c)], dtype=np.float64)
    xyz = T_m2b[:, :3, 3]
    mean = np.mean(xyz, axis=0)
    # 以所有位置的均值作为参考点，统计每个样本偏离参考点的欧氏距离。
    err = xyz - mean
    err_norm = np.linalg.norm(err, axis=1)
    stats = {
        "mean_x": float(mean[0]),
        "mean_y": float(mean[1]),
        "mean_z": float(mean[2]),
        "rms_mm": float(np.sqrt(np.mean(err_norm * err_norm)) * 1000.0),
        "max_mm": float(np.max(err_norm) * 1000.0),
        "z_span_mm": float((np.max(xyz[:, 2]) - np.min(xyz[:, 2])) * 1000.0),
        "z_std_mm": float(np.std(xyz[:, 2]) * 1000.0),
    }
    return xyz, stats


def print_stats(prefix: str, stats: dict[str, float]) -> None:
    """用统一格式打印 RMS、最大误差和 Z 方向离散程度。"""
    print(
        f"{prefix} "
        f"rms={stats['rms_mm']:.2f}mm, "
        f"max={stats['max_mm']:.2f}mm, "
        f"z_span={stats['z_span_mm']:.2f}mm, "
        f"z_std={stats['z_std_mm']:.2f}mm"
    )


def main() -> int:
    """加载标定文件，对比不同算法，并列出残差最大的样本。"""
    args = parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        print(f"[ERROR] not found: {path}")
        return 2

    # 标定文件只应包含数值数组，禁用 pickle 可避免载入任意 Python 对象。
    data = np.load(str(path), allow_pickle=False)
    print(f"[HandEye] file: {path}")
    print(f"[HandEye] keys: {', '.join(data.files)}")
    print(f"[HandEye] mode: {str(data['mode'][0]) if 'mode' in data else 'unknown'}")
    print(f"[HandEye] method: {str(data['method'][0]) if 'method' in data else 'unknown'}")
    print(f"[HandEye] n_samples: {int(data['n_samples'][0]) if 'n_samples' in data else 'unknown'}")
    print(f"[HandEye] T_result:\n{data['T_result']}")

    # 旧版文件可能只有最终矩阵，没有原始样本；这种文件无法重新计算残差。
    required = {"samples_T_gripper2base", "samples_T_marker2cam"}
    if not required.issubset(set(data.files)):
        print("[HandEye] no saved samples in this npz; rerun collect_handeye_eih.py first.")
        return 1

    T = data["T_result"].astype(np.float64)
    T_g2b = data["samples_T_gripper2base"].astype(np.float64)
    T_m2c = data["samples_T_marker2cam"].astype(np.float64)
    xyz, stats = residual_stats(T, T_g2b, T_m2c)
    mean = np.array([stats["mean_x"], stats["mean_y"], stats["mean_z"]], dtype=np.float64)
    err_norm = np.linalg.norm(xyz - mean, axis=1)

    print(
        "[MarkerBase] mean xyz: "
        f"x={mean[0]:+.4f}, y={mean[1]:+.4f}, z={mean[2]:+.4f} m"
    )
    print_stats("[MarkerBase] residual:", stats)

    # 对同一组样本依次运行五种 OpenCV 算法，便于比较哪一种在当前数据上更稳定。
    print("[MethodCompare] all samples:")
    for name, method in METHODS.items():
        try:
            T_candidate = solve_handeye(T_g2b, T_m2c, method)
            _, method_stats = residual_stats(T_candidate, T_g2b, T_m2c)
            print_stats(f"  {name:9s}", method_stats)
        except Exception as exc:
            print(f"  {name:9s} failed: {exc}")

    if len(T_g2b) >= 10:
        # 样本足够多时再剔除残差最大的 25%，观察结果是否被少数坏样本拖偏。
        keep_count = max(6, int(round(len(T_g2b) * 0.75)))
        keep_idx = np.argsort(err_norm)[:keep_count]
        print(f"[MethodCompare] best {keep_count}/{len(T_g2b)} samples by saved-result residual:")
        for name, method in METHODS.items():
            try:
                T_candidate = solve_handeye(T_g2b[keep_idx], T_m2c[keep_idx], method)
                _, method_stats = residual_stats(T_candidate, T_g2b[keep_idx], T_m2c[keep_idx])
                print_stats(f"  {name:9s}", method_stats)
            except Exception as exc:
                print(f"  {name:9s} failed: {exc}")

    # argsort 从小到大排列，因此取最后五项并反转，按误差从大到小输出。
    worst = np.argsort(err_norm)[-5:][::-1]
    print("[MarkerBase] worst samples:")
    for idx in worst:
        print(
            f"  #{idx + 1:02d}: "
            f"xyz=({xyz[idx,0]:+.4f},{xyz[idx,1]:+.4f},{xyz[idx,2]:+.4f}) "
            f"err={err_norm[idx] * 1000.0:.2f}mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
