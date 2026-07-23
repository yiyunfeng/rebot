"""
眼在手 (Eye-in-Hand) 手眼标定数据采集与求解工具 (Gemini2 + reBotArm)。

相机固定在机械臂末端执行器上，ArUco 标记板固定在工作台面上。
通过采集多组机械臂 TCP 位姿与对应的 ArUco 标记板在相机坐标系下的位姿，
使用 Tsai/等手眼标定方法求解相机到夹爪（camera2gripper）的变换矩阵。

模式:
  自动模式（默认）: 机械臂遍历 50 个预定义标定位姿，在每个位姿处停留
                    检测 ArUco 标记板。检测到标记板稳定后自动采集样本，
                    超时无标记板则跳过该位姿。全部完成或按 c/q 后自动求解.
  手动模式（--manual）: 重力补偿模式，用户用手拖拽机械臂到目标位姿。
                        松手后机械臂自动锁住位置（速度接近零时锁定），
                        按回车键手动采集当前样本。

设置:
  相机固定在末端执行器上（眼在手）.
  ArUco 标记板固定在工作台面上.

用法:
    python scripts/collect_handeye_eih.py             # 自动模式
    python scripts/collect_handeye_eih.py --manual     # 手动重力补偿模式
"""

import os
import sys
import threading
import argparse
import queue
import time
import cv2
import numpy as np
from pathlib import Path

# 将项目根目录添加到 Python 搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

from drivers.camera import make_camera
from drivers.robot.grasp_driver import GraspDriver, selected_arm_config
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from calibration.hand_eye import CalibMode, HandEyeCalibrator
from utils.camera_utils import load_config
from utils.transforms import rotation_matrix_to_euler_zyx


# ==========================================
# 预定义标定位姿列表（笛卡尔空间，单位：米，弧度）.
# 每项: (x, y, z, roll, pitch, yaw).
#
# 位姿设计原则：
#   - pitch > 0 表示末端朝下指向 ArUco 标记板.
#   - 先用小范围、多次数采样，降低机械臂自身误差和关节回弹对结果的影响.
#   - 避免过大的 roll/yaw、过远的 x/y 伸展；10cm ArUco 尽量留在画面中心区域.
#   - 共 72 个位姿，覆盖 x/y/z 和 roll/pitch/yaw，但每次移动幅度较小.
# ==========================================
CALIB_POSES_XYZ = [
    # -- 中心小范围：先保证重复性 --
    (0.27, -0.06, 0.30, -0.15, 0.65, -0.25),
    (0.27,  0.00, 0.30,  0.00, 0.65,  0.00),
    (0.27,  0.06, 0.30,  0.15, 0.65,  0.25),
    (0.29, -0.06, 0.30,  0.15, 0.65, -0.20),
    (0.29,  0.00, 0.30, -0.10, 0.65,  0.00),
    (0.29,  0.06, 0.30, -0.15, 0.65,  0.20),

    # -- x 方向小步长：避免一次前后移动太大 --
    (0.24, -0.04, 0.31, -0.10, 0.62, -0.15),
    (0.24,  0.04, 0.31,  0.10, 0.62,  0.15),
    (0.31, -0.04, 0.29,  0.10, 0.68, -0.15),
    (0.31,  0.04, 0.29, -0.10, 0.68,  0.15),
    (0.33, -0.03, 0.30,  0.12, 0.66, -0.18),
    (0.33,  0.03, 0.30, -0.12, 0.66,  0.18),

    # -- y 方向小步长：左右不超过 12cm --
    (0.28, -0.12, 0.31, -0.20, 0.60, -0.30),
    (0.28, -0.09, 0.30,  0.10, 0.66, -0.25),
    (0.28,  0.09, 0.30, -0.10, 0.66,  0.25),
    (0.28,  0.12, 0.31,  0.20, 0.60,  0.30),
    (0.30, -0.12, 0.29,  0.18, 0.70, -0.25),
    (0.30,  0.12, 0.29, -0.18, 0.70,  0.25),

    # -- z 方向小步长：高低变化保留，但不去极限 --
    (0.26, -0.05, 0.34, -0.18, 0.50, -0.20),
    (0.26,  0.05, 0.34,  0.18, 0.50,  0.20),
    (0.30, -0.05, 0.34,  0.12, 0.52, -0.18),
    (0.30,  0.05, 0.34, -0.12, 0.52,  0.18),
    (0.28, -0.04, 0.27,  0.10, 0.76, -0.15),
    (0.28,  0.04, 0.27, -0.10, 0.76,  0.15),

    # -- 姿态覆盖：roll/yaw 小角度变化，避免大幅扭转 --
    (0.27, -0.03, 0.32, -0.30, 0.58,  0.00),
    (0.27,  0.03, 0.32,  0.30, 0.58,  0.00),
    (0.29, -0.03, 0.32,  0.00, 0.58, -0.35),
    (0.29,  0.03, 0.32,  0.00, 0.58,  0.35),
    (0.27, -0.03, 0.28, -0.25, 0.74, -0.20),
    (0.27,  0.03, 0.28,  0.25, 0.74,  0.20),
    (0.30, -0.03, 0.28,  0.20, 0.74, -0.25),
    (0.30,  0.03, 0.28, -0.20, 0.74,  0.25),

    # -- 收尾重复中心附近位姿，用来压低整体一致性误差 --
    (0.28, -0.04, 0.30, -0.12, 0.66, -0.12),
    (0.28,  0.04, 0.30,  0.12, 0.66,  0.12),
    (0.29, -0.04, 0.31,  0.12, 0.62, -0.12),
    (0.29,  0.04, 0.31, -0.12, 0.62,  0.12),

    # -- 补充可见性位姿：同样小幅度，增加扫到 ArUco 的概率 --
    (0.25, -0.08, 0.30, -0.16, 0.66, -0.30),
    (0.25,  0.00, 0.30,  0.00, 0.66,  0.00),
    (0.25,  0.08, 0.30,  0.16, 0.66,  0.30),
    (0.32, -0.08, 0.30,  0.16, 0.66, -0.30),
    (0.32,  0.00, 0.30,  0.00, 0.66,  0.00),
    (0.32,  0.08, 0.30, -0.16, 0.66,  0.30),
    (0.27, -0.10, 0.32, -0.22, 0.56, -0.32),
    (0.27,  0.10, 0.32,  0.22, 0.56,  0.32),
    (0.30, -0.10, 0.32,  0.22, 0.56, -0.32),
    (0.30,  0.10, 0.32, -0.22, 0.56,  0.32),
    (0.27, -0.08, 0.28, -0.18, 0.74, -0.28),
    (0.27,  0.08, 0.28,  0.18, 0.74,  0.28),
    (0.30, -0.08, 0.28,  0.18, 0.74, -0.28),
    (0.30,  0.08, 0.28, -0.18, 0.74,  0.28),
    (0.28, -0.02, 0.33, -0.24, 0.54,  0.18),
    (0.28,  0.02, 0.33,  0.24, 0.54, -0.18),

    # -- 继续加密中心视野：提高有效采样数，仍不扩大运动范围 --
    (0.26, -0.06, 0.29, -0.08, 0.70, -0.18),
    (0.26,  0.00, 0.29,  0.08, 0.70,  0.00),
    (0.26,  0.06, 0.29,  0.08, 0.70,  0.18),
    (0.31, -0.06, 0.31,  0.08, 0.62, -0.18),
    (0.31,  0.00, 0.31, -0.08, 0.62,  0.00),
    (0.31,  0.06, 0.31, -0.08, 0.62,  0.18),
    (0.27, -0.11, 0.30, -0.12, 0.68, -0.22),
    (0.27,  0.11, 0.30,  0.12, 0.68,  0.22),
    (0.30, -0.11, 0.30,  0.12, 0.68, -0.22),
    (0.30,  0.11, 0.30, -0.12, 0.68,  0.22),
    (0.28, -0.07, 0.33, -0.20, 0.55, -0.10),
    (0.28,  0.07, 0.33,  0.20, 0.55,  0.10),
    (0.29, -0.07, 0.27,  0.16, 0.78, -0.10),
    (0.29,  0.07, 0.27, -0.16, 0.78,  0.10),
    (0.25, -0.03, 0.32, -0.18, 0.58, -0.26),
    (0.25,  0.03, 0.32,  0.18, 0.58,  0.26),
    (0.32, -0.03, 0.28,  0.18, 0.72, -0.26),
    (0.32,  0.03, 0.28, -0.18, 0.72,  0.26),
    (0.28, -0.01, 0.30, -0.06, 0.66, -0.06),
    (0.28,  0.01, 0.30,  0.06, 0.66,  0.06),
]

# 自动模式参数
AUTO_MOVE_DURATION_S = 3.0          # 移动到目标位姿的运动时长
AUTO_SETTLE_EXTRA_S = 2.0           # 运动到达后再等 2 秒，降低关节回弹/视觉抖动影响
AUTO_MARKER_TIMEOUT_S = 4.0         # 稳定等待后，最多再等 ArUco 出现的时间
AUTO_MARKER_STABLE_FRAMES = 15      # 需要连续稳定检测多少帧后才采集
AUTO_MARKER_MAX_TRANS_STD_M = 0.002 # 稳定窗口内 ArUco 平移标准差阈值
AUTO_MARKER_MAX_RPY_STD_RAD = 0.01  # 稳定窗口内 ArUco 姿态标准差阈值
OUTLIER_KEEP_RATIO = 0.75           # 求解前保留残差最小的样本比例
MIN_CALIB_SAMPLES = 5                # 最少需要的标定样本数（低于此值不求解）


def make_input_thread(line_queue: queue.Queue) -> threading.Thread:
    """创建终端输入监听线程。

    该线程独立运行，将用户的终端输入行放入队列。
    主循环从队列中非阻塞读取，实现终端命令与 OpenCV 窗口事件并行处理.

    参数：
        line_queue: 用于传递输入行的线程安全队列.

    返回：
        已启动的 daemon 线程对象.
    """
    def _loop():
        """阻塞等待终端输入，并把每一行安全地交给主线程处理。"""
        while True:
            try:
                line_queue.put(input())
            except EOFError:
                line_queue.put(None)
                break
            except KeyboardInterrupt:
                line_queue.put(None)
                break
    # daemon=True 使主程序退出时不会被仍在等待 input() 的线程卡住。
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def save_handeye_with_samples(result, calibrator: HandEyeCalibrator, path: Path) -> dict[str, float]:
    """保存手眼结果和原始样本，并返回 marker 在 base 下的一致性残差。

    对 eye-in-hand 来说，同一个 ArUco 板固定在桌面上，所以每个样本算出的
    T_marker2base = T_gripper2base @ T_cam2gripper @ T_marker2cam
    应该接近同一个位置。残差越大，说明手眼解、样本姿态或 ArUco 位姿越不可靠。
    """
    # 把对象列表整理成 (N, 4, 4) 数组，既便于矩阵批量计算，也便于保存到 npz。
    samples = list(calibrator._samples)
    T_g2b = np.asarray([s.T_gripper2base for s in samples], dtype=np.float64)
    T_m2c = np.asarray([s.T_marker2cam for s in samples], dtype=np.float64)
    T_m2b = np.asarray([g @ result.T_result @ m for g, m in zip(T_g2b, T_m2c)], dtype=np.float64)

    # 固定标记在 base 中只应有一个位置；各样本位置相对均值的偏差就是一致性残差。
    marker_xyz = T_m2b[:, :3, 3]
    mean_xyz = np.mean(marker_xyz, axis=0)
    err = marker_xyz - mean_xyz
    err_norm = np.linalg.norm(err, axis=1)
    stats = {
        "mean_x": float(mean_xyz[0]),
        "mean_y": float(mean_xyz[1]),
        "mean_z": float(mean_xyz[2]),
        "rms_mm": float(np.sqrt(np.mean(err_norm * err_norm)) * 1000.0),
        "max_mm": float(np.max(err_norm) * 1000.0),
        "z_span_mm": float((np.max(marker_xyz[:, 2]) - np.min(marker_xyz[:, 2])) * 1000.0),
        "z_std_mm": float(np.std(marker_xyz[:, 2]) * 1000.0),
    }

    # 同时保存最终矩阵和原始样本，后续可以离线重算不同算法，不必重新操作真机。
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        T_result=result.T_result,
        mode=np.array([result.mode]),
        n_samples=np.array([result.n_samples]),
        method=np.array([result.method]),
        samples_T_gripper2base=T_g2b,
        samples_T_marker2cam=T_m2c,
        samples_T_marker2base=T_m2b,
        marker_base_mean_xyz_m=mean_xyz,
        marker_base_error_norm_m=err_norm,
    )
    return stats


def marker_base_residual_stats(result, samples) -> tuple[dict[str, float], np.ndarray]:
    """计算固定 ArUco 在 base 下的一致性残差。"""
    # 变换链：marker -> camera -> gripper -> base。
    T_g2b = np.asarray([s.T_gripper2base for s in samples], dtype=np.float64)
    T_m2c = np.asarray([s.T_marker2cam for s in samples], dtype=np.float64)
    T_m2b = np.asarray([g @ result.T_result @ m for g, m in zip(T_g2b, T_m2c)], dtype=np.float64)
    marker_xyz = T_m2b[:, :3, 3]
    mean_xyz = np.mean(marker_xyz, axis=0)
    err = marker_xyz - mean_xyz
    err_norm = np.linalg.norm(err, axis=1)
    stats = {
        "mean_x": float(mean_xyz[0]),
        "mean_y": float(mean_xyz[1]),
        "mean_z": float(mean_xyz[2]),
        "rms_mm": float(np.sqrt(np.mean(err_norm * err_norm)) * 1000.0),
        "max_mm": float(np.max(err_norm) * 1000.0),
        "z_span_mm": float((np.max(marker_xyz[:, 2]) - np.min(marker_xyz[:, 2])) * 1000.0),
        "z_std_mm": float(np.std(marker_xyz[:, 2]) * 1000.0),
    }
    return stats, err_norm


def filtered_calibrator_by_residual(
    result,
    calibrator: HandEyeCalibrator,
    keep_ratio: float,
) -> tuple[HandEyeCalibrator, dict[str, float]]:
    """按 marker-base 残差剔除最差样本，返回新的标定器。"""
    samples = list(calibrator._samples)
    stats, err_norm = marker_base_residual_stats(result, samples)
    # 即使 keep_ratio 很小，也至少保留 OpenCV 求解所需的最少样本数。
    keep_n = max(MIN_CALIB_SAMPLES, int(round(len(samples) * keep_ratio)))
    keep_n = min(len(samples), keep_n)
    # argsort 从残差小到大排列，只把最稳定的 keep_n 组样本放进新标定器。
    keep_idx = np.argsort(err_norm)[:keep_n]

    filtered = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method=result.method)
    for idx in keep_idx:
        sample = samples[int(idx)]
        filtered.add_sample(sample.T_gripper2base, sample.T_marker2cam)

    stats["kept"] = float(keep_n)
    stats["total"] = float(len(samples))
    stats["reject_max_mm"] = float(np.max(err_norm[np.argsort(err_norm)[keep_n:]]) * 1000.0) if keep_n < len(samples) else 0.0
    return filtered, stats


def marker_window_stats(window: list[np.ndarray]) -> tuple[float, float]:
    """返回 ArUco 稳定窗口内最大平移标准差和最大 RPY 标准差。"""
    if not window:
        return float("inf"), float("inf")
    Ts = np.asarray(window, dtype=np.float64)
    trans_std = float(np.max(np.std(Ts[:, :3, 3], axis=0)))
    rpy = np.asarray([rotation_matrix_to_euler_zyx(T[:3, :3]) for T in Ts], dtype=np.float64)
    rpy_std = float(np.max(np.std(rpy, axis=0)))
    return trans_std, rpy_std


def marker_window_is_stable(window: list[np.ndarray]) -> tuple[bool, float, float]:
    """判断最近若干帧 ArUco 位姿是否足够稳定。"""
    if len(window) < AUTO_MARKER_STABLE_FRAMES:
        trans_std, rpy_std = marker_window_stats(window)
        return False, trans_std, rpy_std
    trans_std, rpy_std = marker_window_stats(window[-AUTO_MARKER_STABLE_FRAMES:])
    stable = (
        trans_std <= AUTO_MARKER_MAX_TRANS_STD_M
        and rpy_std <= AUTO_MARKER_MAX_RPY_STD_RAD
    )
    return stable, trans_std, rpy_std


# ==========================================
# 重力补偿控制器（手动模式专用）
# ==========================================
class GravityCompController:
    """MIT 模式 + 末端速度锁定，用于手拖示教位姿采集。

    参考实现：reBotArm_control_py/example/10_gravity_compensation_lock.py

    工作原理：
      1. 机械臂运行在 MIT 模式下，控制器计算每个关节的重力补偿力矩.
      2. 用户用手拖拽机械臂时，关节速度非零.
      3. 控制器持续监测末端执行器的线速度 (v) 和角速度 (w).
      4. 当用户松手时，末端速度降低到阈值以下，控制器将
         当前关节角锁定为新的目标位置，实现原地锁定.
      5. 锁定时加入积分误差补偿以抵消残余重力.

    速度锁定检测：
      - 线速度阈值 V_THRESH = 0.04 m/s
      - 角速度阈值 W_THRESH = 0.08 rad/s
      - 通过 Pinocchio 的雅可比矩阵将关节速度映射到末端速度.
    """

    KP = 8.0      # MIT 位置比例增益
    KD = 1.5      # MIT 速度阻尼增益
    V_THRESH  = 0.04   # 末端线速度锁定阈值 (m/s)
    W_THRESH  = 0.08   # 末端角速度锁定阈值 (rad/s)

    def __init__(self, arm: RebotArm) -> None:
        """加载动力学模型并初始化手拖示教所需的控制状态。"""
        # 导入 Pinocchio 动力学和运动学函数
        from reBotArm_control_py.dynamics import compute_generalized_gravity
        from reBotArm_control_py.kinematics import (
            load_robot_model, get_end_effector_frame_id,
        )
        from reBotArm_control_py.kinematics.robot_model import pad_q_for_model
        import pinocchio as pin

        self._compute_gravity = compute_generalized_gravity
        self._pad_q_for_model = pad_q_for_model
        self._pin = pin

        self._arm = arm
        if not self._arm.has_gripper:
            raise ValueError(
                "Hardware config is missing groups.gripper. "
                "Enable the gripper group in the selected hardware YAML under "
                "reBotArm_control_py/config."
            )
        # 加载机器人模型和末端执行器坐标系 ID
        self._model = load_robot_model()
        self._data  = self._model.createData()
        self._ee_id = get_end_effector_frame_id(self._model)

        self._n = None              # 关节数，在连接后设置
        self._q_target = None       # MIT 目标关节角
        self._integral = None       # 重力补偿积分误差
        self._gc_running = threading.Event()  # 控制回路运行标志
        self._io_lock = threading.RLock()     # IO 操作互斥锁

    def start(self) -> None:
        """连接机械臂，切换到 MIT 模式，启动重力补偿控制回路。

        启动步骤：
          1. 连接机械臂并获取当前关节角.
          2. 配置 arm 组为 MIT 模式（kp=8.0, kd=1.5）.
          3. 配置 gripper 组为 MIT 模式（保持当前位置）.
          4. 以当前关节角初始化目标位置和积分误差.
          5. 启动控制回路 (_worker).
        """
        self._arm.connect()
        print("[GravityComp] Connected")

        # arm 组设置为 MIT 模式
        self._arm.arm.mode_mit(
            kp=np.full(self._arm.arm.num_joints, self.KP),
            kd=np.full(self._arm.arm.num_joints, self.KD),
        )
        # gripper 组设置为 MIT 模式
        self._arm.gripper.mode_mit()
        self._arm.enable_all()
        print("[GravityComp] Enabled")

        n = self._arm.arm.num_joints
        self._n = n
        # 等待所有电机反馈就绪
        self._wait_state_valid()
        with self._io_lock:
            q0 = self._arm.get_state()[0][:n]
        # 初始化目标位置为当前关节角（避免上电跳变）
        self._q_target = q0.copy()
        # 积分误差归零
        self._integral = np.zeros(n)

        print(f"[GravityComp] MIT mode, kp={self.KP} kd={self.KD}. Move the arm by hand.")

        self._gc_running.set()
        self._arm.start_control_loop(self._worker, rate=self._arm.rate)

    def _wait_state_valid(self, timeout: float = 2.0) -> None:
        """等待所有电机的反馈数据就绪。

        循环检查每个电机的 get_state() 是否返回非 None，
        确保完全获取到电机状态后再开始控制.

        异常：
            RuntimeError: 超时后仍未就绪.
        """
        # 控制回路启动前必须确认每个电机都有反馈，否则初始目标可能包含无效值。
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            with self._io_lock:
                self._arm.get_state()
                # all(...) 只有在所有 arm/gripper 电机状态都非 None 时才为 True。
                ok = all(
                    m.get_state() is not None
                    for m in self._arm._motor_map.values()
                )
            if ok:
                return
            time.sleep(0.02)
        raise RuntimeError("[GravityComp] Arm feedback is not ready")

    def safe_home(self) -> None:
        """安全停止重力补偿，用 SDK 控制器回 Home 位，然后断开。

        流程：
          1. 停止重力补偿控制回路.
          2. 用 RebotArmEndPose 控制器接管控制权.
          3. 以当前位置为起点，执行 safe_home 回到安全位姿.
          4. 断开机械臂连接.
        """
        self._gc_running.clear()
        try:
            print("[GravityComp] Homing...")
            self._arm.stop_control_loop()
            with self._io_lock:
                q_now = self._arm.get_state()[0][: self._n]
                g_now = self._arm.gripper.get_positions()

            # 创建 SDK 控制器，以当前位置初始化并执行回 Home
            ctrl = RebotArmEndPose(
                self._arm,
                arm_control_mode="mit",
                use_gravity_ff=True,
            )
            ctrl.set_gripper_target(float(g_now[0]) if g_now.size else 0.0)
            ctrl._q_target[:] = q_now
            ctrl.start()
            ctrl.safe_home()
            self._arm.stop_control_loop()
        except Exception as e:
            print(f"[GravityComp] Homing failed: {e}")
        try:
            self._arm.disconnect()
        except Exception:
            pass
        print("[GravityComp] Disconnected")

    def _worker(self, r, dt: float) -> None:
        """重力补偿控制回路 worker（每个控制周期调用一次）。

        控制逻辑：
          1. 读取当前关节角和关节速度.
          2. 计算重力补偿力矩 tau_g.
          3. 计算末端执行器速度 v (通过雅可比矩阵 J @ qd).
          4. 速度锁定判断：
             - 末端速度 > V_THRESH/W_THRESH -> 用户正在拖拽 -> 更新目标位置.
             - 末端速度 < 阈值 -> 用户松手 -> 锁定当前目标位置.
          5. 发送 MIT 控制指令：
             pos=q_target（当前锁定位置）, kp=KP, kd=KD, tau=tau_g + 积分.

        积分误差的作用：
          当机械臂锁定后，残余重力可能导致轻微偏移。积分项累积位置误差
          来补偿残余重力，使机械臂在锁定位置保持稳定.
        """
        if not self._gc_running.is_set():
            return

        pin = self._pin
        model, data, ee_id = self._model, self._data, self._ee_id
        n = self._n
        KP, KD = self.KP, self.KD

        try:
            with self._io_lock:
                q_all, qd_all, _ = r.get_state()
            # 提取 arm 关节的关节角和速度
            q = q_all[:n]
            qd = qd_all[:n]
            # 填充为完整模型关节向量（包括夹爪关节）后计算重力
            q_model = self._pad_q_for_model(model, q, n)
            tau_g = self._compute_gravity(model=model, q=q_model)[:n]

            # 位置误差积分（补偿残余重力）
            q_err = self._q_target - q
            self._integral += q_err
            np.clip(self._integral, -0.5, 0.5, out=self._integral)

            # 通过雅可比矩阵计算末端执行器速度
            # J @ qd -> v (6维: [vx, vy, vz, wx, wy, wz])
            qd_model = np.zeros(model.nv)
            qd_model[: min(model.nv, n)] = qd[: min(model.nv, n)]
            pin.computeJointJacobians(model, data, q_model)
            pin.updateFramePlacements(model, data)
            J = pin.getFrameJacobian(model, data, ee_id, pin.ReferenceFrame.WORLD)
            v = J @ qd_model

            # 速度锁定判断：末端速度高于阈值表示用户正在拖拽
            if (np.linalg.norm(v[:3]) > self.V_THRESH or
                    np.linalg.norm(v[3:]) > self.W_THRESH):
                # 用户正在拖拽：更新目标位置为当前位置
                self._q_target = q.copy()
                # 衰减积分以防止累积过大
                self._integral *= 0.9

            # 发送 MIT 控制指令:
            #   pos = 目标关节角, kp = 位置增益, kd = 阻尼增益,
            #   tau = 重力补偿 + 积分误差
            with self._io_lock:
                r.arm.send_mit(
                    pos=self._q_target,
                    vel=np.zeros(n),
                    kp=np.full(n, KP),
                    kd=np.full(n, KD),
                    tau=tau_g + self._integral,
                )
                # 夹爪保持当前位置
                r.gripper.send_mit(r.gripper.get_positions())
        except Exception:
            pass


# ==========================================
# 主流程
# ==========================================
def main():
    """眼在手标定数据采集与求解的主流程。

    自动模式状态机（tick_auto）:
      idle -> settling (运动+稳定等待) -> searching (搜索ArUco) -> capture (采集) -> next (下一个位姿) -> ...
      超时无ArUco时跳过当前位姿进入next.

    手动模式：
      用户拖拽机械臂（重力补偿），松手后机械臂自动锁定。
      按回车键采集当前样本，按 c/q 求解并退出.

    总体流程：
      1. 加载配置，初始化相机和 ArUco 检测.
      2. 初始化手眼标定器 (HandEyeCalibrator).
      3. 连接机械臂（自动模式启动运动控制，手动模式启动重力补偿）.
      4. 主循环：读取相机帧 -> ArUco 检测 -> 自动/手动采集 -> 可视化显示.
      5. 退出时自动求解手眼标定并保存到 hand_eye.npz.
    """
    parser = argparse.ArgumentParser(description="Eye-in-hand calibration data collection")
    parser.add_argument("--manual", action="store_true",
                        help="手动模式：重力补偿，手拖机械臂后按回车采集")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg  = load_config(root / "config" / "default.yaml")

    # 读取相机和标定配置
    cam_type   = cfg["camera"]["type"]
    calib_dir  = root / "config" / "calibration" / cam_type
    aruco_cfg  = cfg["calibration"]["aruco"]
    he_method  = cfg["calibration"].get("hand_eye_method", "TSAI")
    save_path  = calib_dir / "hand_eye.npz"

    # 初始化相机
    cam = make_camera(cfg)
    cam.setup_aruco(
        marker_length_m=aruco_cfg["marker_length_m"],
        dict_id=aruco_cfg.get("dict_id", 0),
        target_marker_id=aruco_cfg.get("target_marker_id"),
    )

    # 初始化手眼标定器
    calibrator = HandEyeCalibrator(CalibMode.EYE_IN_HAND, method=he_method)

    # 机械臂相关变量
    mode_str = "manual (gravity compensation)" if args.manual else f"auto ({len(CALIB_POSES_XYZ)} preset poses)"
    gc_ctrl: GravityCompController | None = None       # 重力补偿控制器（手动模式）
    rebotarm: RebotArm | None = None                    # 机械臂实例
    controller: RebotArmEndPose | None = None           # SDK 末端控制器
    grasp_driver: GraspDriver | None = None             # 夹爪驱动
    auto_controller_mode: str | None = None             # 自动模式控制器模式
    auto_use_gravity_ff = False                         # 是否使用重力前馈
    # 自动模式状态字典
    auto = {
        "enabled": not args.manual,
        "idx": 0,                    # 当前位姿索引
        "pose_idx": None,            # 正在执行的位姿索引
        "phase": "idle",             # 状态机阶段：idle | settling | searching | capture | done
        "settle_until": 0.0,         # 稳定阶段截止时间
        "timeout_at": 0.0,           # ArUco 搜索超时时间
        "stable_frames": 0,          # 连续稳定检测 ArUco 的帧数
        "marker_window": [],         # 最近若干帧 ArUco 位姿，用于判断抖动
        "status": "waiting to start",
        "finished": False,           # 是否完成全部位姿
    }
    result_saved = False

    print(f"\n=== Eye-in-Hand Calibration ===")
    print(f"Camera: {cam_type}  |  Mode: {mode_str}  |  Solver: {he_method}")
    print(f"ArUco size: {aruco_cfg['marker_length_m']*100:.0f}cm  |  Output: {save_path}")
    print()

    # 先打开相机，确保相机正常后再连接机械臂
    try:
        cam.open()
        print("Warming up camera...", end="", flush=True)
        cam.warm_up(20)
        print(" ready\n")
    except Exception as e:
        try:
            cam.close()
        except Exception:
            pass
        print(f"[Camera] Initialization failed: {e}")
        sys.exit(1)

    # 连接机械臂
    robot_cfg = cfg.get("robot", {})
    try:
        rebotarm = RebotArm()
        if args.manual:
            # 手动模式：初始化末端控制器和重力补偿控制器
            controller = RebotArmEndPose(rebotarm, arm_control_mode="mit", use_gravity_ff=True)
            grasp_driver = GraspDriver(
                rebotarm,
                controller,
                gripper_config=robot_cfg.get("gripper"),
                repo_root=robot_cfg.get("repo_root"),
            )
            gc_ctrl = GravityCompController(rebotarm)
            gc_ctrl.start()
            print("[Robot] Manual mode ready. Move the arm by hand, then press Enter to capture.")
        else:
            # 自动模式：初始化末端控制器和夹爪驱动，启动自动遍历
            selected = selected_arm_config(robot_cfg.get("repo_root"))
            auto_controller_mode = selected.controller_mode
            auto_use_gravity_ff = auto_controller_mode == "mit"
            controller = RebotArmEndPose(rebotarm, arm_control_mode=auto_controller_mode)
            grasp_driver = GraspDriver(
                rebotarm,
                controller,
                gripper_config=robot_cfg.get("gripper"),
                repo_root=robot_cfg.get("repo_root"),
            )
            grasp_driver.start()
            print(
                f"[Robot] Auto mode ready, control mode: {selected.controller_mode}. "
                f"{len(CALIB_POSES_XYZ)} preset poses will be traversed."
            )
    except Exception as e:
        try:
            cam.close()
        except Exception:
            pass
        print(f"[Robot] Connection failed: {e}")
        sys.exit(1)

    if args.manual:
        print("[Controls] Enter/G/Space=capture  s/c/q=finish and solve  pos=print current TCP pose")
    else:
        print("[Controls] Auto traversal and capture  c/q=stop and solve  pos=print current TCP pose")
    print()

    latest_pose = None
    line_queue: queue.Queue | None = None
    if sys.stdin.isatty():
        line_queue = queue.Queue()
        make_input_thread(line_queue)
    else:
        print("[Hint] Non-interactive terminal detected; terminal commands are disabled")

    def _print_fk() -> None:
        """打印当前 TCP 位姿的正运动学计算结果。"""
        try:
            T = grasp_driver.get_tcp_pose()
            t = T[:3, 3]
            R = T[:3, :3]
            _r, _p, _y = rotation_matrix_to_euler_zyx(R)
            print(f"  FK: x={t[0]:+.3f} y={t[1]:+.3f} z={t[2]:+.3f} m"
                  f"  rpy=[{_r:+.2f} {_p:+.2f} {_y:+.2f}] rad")
        except Exception as e:
            print(f"  [Error] {e}")

    def capture_sample(cur, source: str) -> bool:
        """采集一个标定样本。

        样本组成：
          - T_g2b (gripper to base): 末端执行器在世界坐标系中的位姿（正运动学计算）.
          - T_marker2cam: ArUco 标记板在相机坐标系中的位姿（ArUco 检测）.

        参数：
            cur: ArUco 检测结果，None 表示未检测到标记板.
            source: 样本来源描述（如 "auto pose 5/50" 或 "manual capture"）.

        返回：
            True 如果成功添加样本，False 如果失败.
        """
        if cur is None:
            print("  [Skip] Marker is not visible; adjust the pose and try again")
            return False

        print(f"\n[Sample {calibrator.n_samples + 1}] {source}")
        print(f"  ArUco: x={cur.T_marker2cam[0,3]:.3f} "
              f"y={cur.T_marker2cam[1,3]:.3f} "
              f"z={cur.T_marker2cam[2,3]:.3f} m")
        try:
            # 通过正运动学获取末端位姿 T_g2b
            T_g2b = grasp_driver.get_tcp_pose()
            t = T_g2b[:3, 3]
            print(f"  End effector (FK): x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
            # 添加样本到手眼标定器
            calibrator.add_sample(T_g2b, cur.T_marker2cam)
            print(f"  [OK] Recorded, total samples: {calibrator.n_samples}"
                  + ("  will solve automatically on finish" if calibrator.n_samples >= 15 else ""))
            return True
        except Exception as e:
            print(f"  [Error] Failed to read TCP pose: {e}")
            return False

    def compute_and_save(reason: str) -> bool:
        """求解手眼标定并保存结果到 hand_eye.npz。

        使用 Tsai/等手眼标定方法求解 T_cam2gripper：
          AX = XB 形式的方程，其中：
           - A: 机械臂末端在两个位姿之间的相对运动.
           - B: 相机（观察到标记板）在两个位姿之间的相对运动.
           - X: 待求解的相机到夹爪变换矩阵 T_cam2gripper.

        参数：
            reason: 求解原因（如 "normal finish", "user interrupted"）.

        返回：
            True 如果成功求解并保存，False 如果失败.
        """
        nonlocal result_saved
        print(f"\n[Finish] {reason}")
        if calibrator.n_samples < MIN_CALIB_SAMPLES:
            print(f"[Result] Not enough samples ({calibrator.n_samples} < {MIN_CALIB_SAMPLES}); calibration was not solved")
            if save_path.exists():
                print("[Result] Existing hand_eye.npz was not updated")
            return False

        print(f"[Result] Solving with {calibrator.n_samples} samples...")
        try:
            # 先用全样本求一次，利用固定 ArUco 在 base 下的一致性残差找离群点。
            all_result = calibrator.calibrate(min_samples=MIN_CALIB_SAMPLES)
            all_stats, _ = marker_base_residual_stats(all_result, list(calibrator._samples))
            print(
                "[Result] All samples residual: "
                f"rms={all_stats['rms_mm']:.2f}mm "
                f"max={all_stats['max_mm']:.2f}mm "
                f"z_span={all_stats['z_span_mm']:.2f}mm "
                f"z_std={all_stats['z_std_mm']:.2f}mm"
            )

            solve_calibrator = calibrator
            if calibrator.n_samples > MIN_CALIB_SAMPLES:
                filtered, filter_stats = filtered_calibrator_by_residual(
                    all_result,
                    calibrator,
                    OUTLIER_KEEP_RATIO,
                )
                if filtered.n_samples < calibrator.n_samples:
                    print(
                        "[Result] Outlier filter: "
                        f"keep {int(filter_stats['kept'])}/{int(filter_stats['total'])} "
                        f"samples by marker-base residual"
                    )
                    solve_calibrator = filtered

            # 用过滤后的样本重新求解并保存，npz 中保存的样本也是最终参与求解的样本。
            result = solve_calibrator.calibrate(min_samples=MIN_CALIB_SAMPLES)
            residual = save_handeye_with_samples(result, solve_calibrator, save_path)
            t = result.T_result[:3, 3]
            R = result.T_result[:3, :3]
            print(f"[Result] T_cam2gripper translation: x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
            print(f"[Result] Rotation matrix:\n{R}")
            print(
                "[Result] Marker in base residual: "
                f"rms={residual['rms_mm']:.2f}mm "
                f"max={residual['max_mm']:.2f}mm "
                f"z_span={residual['z_span_mm']:.2f}mm "
                f"z_std={residual['z_std_mm']:.2f}mm"
            )
            print(
                "[Result] Marker base mean xyz: "
                f"x={residual['mean_x']:+.4f} "
                f"y={residual['mean_y']:+.4f} "
                f"z={residual['mean_z']:+.4f} m"
            )
            print(f"[Result] [OK] Saved to {save_path}")
            if calibrator.n_samples < 15:
                print("[Result] Tip: fewer than 15 samples; collect more samples for better accuracy")
            result_saved = True
            return True
        except Exception as e:
            print(f"[Result] [Error] Solve failed: {e}")
            return False

    def start_next_auto_pose() -> bool:
        """自动模式：移动到下一个预定义标定位姿。

        调用 SDK 控制器的 move_to_traj 执行笛卡尔空间轨迹规划。
        如果当前位姿无逆运动学解（IK 失败），自动跳过并递增索引.

        返回：
            True 如果所有位姿已遍历完成，False 如果正在执行当前位姿.
        """
        if not auto["enabled"] or controller is None:
            return False

        total = len(CALIB_POSES_XYZ)
        while auto["idx"] < total:
            idx = auto["idx"]
            x, y, z, roll, pitch, yaw = CALIB_POSES_XYZ[idx]
            print(f"\n[Auto] Pose {idx+1}/{total}: "
                  f"pos=({x:.2f},{y:.2f},{z:.2f}) rpy=({roll:.2f},{pitch:.2f},{yaw:.2f})")
            # 执行笛卡尔轨迹运动
            ok = controller.move_to_traj(x, y, z, roll=roll, pitch=pitch, yaw=yaw, duration=AUTO_MOVE_DURATION_S)
            if ok:
                now = time.monotonic()
                auto["pose_idx"] = idx
                auto["phase"] = "settling"                                # 进入稳定等待阶段
                auto["settle_until"] = now + AUTO_MOVE_DURATION_S + AUTO_SETTLE_EXTRA_S  # 运动+额外稳定时间
                auto["timeout_at"] = auto["settle_until"] + AUTO_MARKER_TIMEOUT_S         # ArUco 超时时间
                auto["stable_frames"] = 0
                auto["marker_window"] = []
                auto["status"] = f"pose {idx+1}/{total} moving"
                return False

            # IK 无解，跳过当前位姿
            print(f"[Auto] Pose {idx+1}/{total} has no IK solution, skipping")
            auto["idx"] += 1

        # 全部位姿遍历完成
        auto["phase"] = "done"
        auto["finished"] = True
        auto["status"] = "all poses completed"
        print("\n[Auto] All preset poses completed")
        return True

    def tick_auto(cur) -> bool:
        """自动模式状态机 tick（每帧调用一次）。

        状态机阶段：
          idle -> start_next_auto_pose() 移动到下一个位姿.
          settling -> 等待运动 + 额外稳定时间结束 -> 进入 searching.
          searching -> 等待连续稳定检测 ArUco -> capture_sample -> idle (next).
          searching -> 超时无 ArUco -> idle (skip).

        参数：
            cur: 当前帧的 ArUco 检测结果.

        返回：
            True 如果自动遍历已完成.
        """
        if not auto["enabled"] or auto["finished"]:
            return auto["finished"]

        if auto["phase"] == "idle":
            return start_next_auto_pose()

        pose_idx = auto["pose_idx"]
        total = len(CALIB_POSES_XYZ)
        now = time.monotonic()

        if auto["phase"] == "settling":
            # 稳定等待阶段：运动到位后还需要额外稳定时间
            remain = auto["settle_until"] - now
            if remain > 0.0:
                auto["stable_frames"] = 0
                auto["marker_window"] = []
                auto["status"] = f"pose {pose_idx+1}/{total} moving/settling {remain:.1f}s"
                return False
            # 稳定时间到，进入搜索 ArUco 阶段
            auto["phase"] = "searching"

        # 搜索 ArUco 阶段
        if cur is not None:
            # 检测到 ArUco，先放入稳定窗口；只有窗口内位姿抖动足够小才采样。
            auto["marker_window"].append(cur.T_marker2cam.copy())
            if len(auto["marker_window"]) > AUTO_MARKER_STABLE_FRAMES:
                auto["marker_window"] = auto["marker_window"][-AUTO_MARKER_STABLE_FRAMES:]
            auto["stable_frames"] = len(auto["marker_window"])
            stable, trans_std, rpy_std = marker_window_is_stable(auto["marker_window"])
            remain = max(0.0, auto["timeout_at"] - now)
            auto["status"] = (
                f"pose {pose_idx+1}/{total} marker stable "
                f"{auto['stable_frames']}/{AUTO_MARKER_STABLE_FRAMES}  "
                f"std={trans_std*1000.0:.1f}mm/{rpy_std:.3f}rad  remaining {remain:.1f}s"
            )
            if stable:
                # 连续稳定检测足够帧数 -> 采集样本
                capture_sample(cur, f"auto pose {pose_idx+1}/{total}")
                auto["idx"] += 1
                auto["phase"] = "idle"
                auto["stable_frames"] = 0
                auto["marker_window"] = []
                return start_next_auto_pose()  # 立即开始下一个位姿
        else:
            # 未检测到 ArUco，重置稳定帧计数
            auto["stable_frames"] = 0
            auto["marker_window"] = []
            remain = max(0.0, auto["timeout_at"] - now)
            auto["status"] = f"pose {pose_idx+1}/{total} waiting for ArUco {remain:.1f}s"

        # 超时：跳过当前位姿
        if now >= auto["timeout_at"]:
            print(f"[Auto] Pose {pose_idx+1}/{total} timed out without ArUco, skipping")
            auto["idx"] += 1
            auto["phase"] = "idle"
            auto["stable_frames"] = 0
            auto["marker_window"] = []
            return start_next_auto_pose()

        return False

    def safe_home_and_disconnect() -> None:
        """停止当前控制回路，用新 SDK 控制器回 Home，然后断开连接。"""
        if rebotarm is None or auto_controller_mode is None:
            return
        try:
            print("[Robot] Homing and disconnecting...")
            rebotarm.stop_control_loop()
            q_now = rebotarm.get_state()[0][: rebotarm.arm.num_joints]
            g_now = rebotarm.gripper.get_positions() if rebotarm.has_gripper else np.array([])

            # 新建控制器，以当前位置为起点执行回 Home
            ctrl = RebotArmEndPose(
                rebotarm,
                arm_control_mode=auto_controller_mode,
                use_gravity_ff=auto_use_gravity_ff,
            )
            ctrl.set_gripper_target(float(g_now[0]) if g_now.size else 0.0)
            ctrl._q_target[:] = q_now
            ctrl.start()
            ctrl.safe_home()
            rebotarm.stop_control_loop()
        except Exception as e:
            print(f"[Robot] Homing failed: {e}")
        try:
            rebotarm.disconnect()
        except Exception:
            pass

    def handle_line(raw: str) -> bool:
        """处理终端输入的一行命令。

        支持的命令：
          - s / c / q: 停止采集并求解.
          - pos: 打印当前 TCP 位姿（正运动学）.
          - 空行 / g（手动模式）: 采集当前样本.
          - 其他: 显示帮助信息.

        返回：
            True 如果需要退出主循环.
        """
        if raw is None:
            print("\n[Interrupt] Terminal input closed; stopping and trying to solve")
            return True

        line = raw.strip().lower()

        if line in {"q", "c", "s"}:
            return True

        if line == "pos":
            _print_fk()
            return False

        if args.manual and line in {"", "g"}:
            # 空行 + 手动模式 = 采集样本
            capture_sample(latest_pose, "manual capture")
            return False

        if line:
            if args.manual:
                print("  Manual commands: Enter/G/Space=capture  s/c/q=finish and solve  pos=print current TCP pose")
            else:
                print("  Auto commands: c/q=finish and solve  pos=print current TCP pose")
        return False

    # ==========================================
    # 主循环
    # ==========================================
    WIN = "Eye-in-Hand Calibration  (operate in terminal)"
    finish_reason = "normal finish"
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

        while True:
            # 获取相机帧并检测 ArUco
            bgr, _ = cam.get_frame()
            if bgr is not None:
                pose = cam.detect_aruco(bgr)
                latest_pose = pose
                # 自动模式状态机 tick
                if tick_auto(pose):
                    finish_reason = "auto traversal completed"
                # 绘制 ArUco 检测可视化
                vis  = cam.draw_aruco(bgr)
                n    = calibrator.n_samples

                # OSD 辅助函数：在图像上叠加文字
                def osd(text, y, color=(220, 220, 220)):
                    """在当前预览帧的指定纵坐标叠加一行状态文字。"""
                    cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 1, cv2.LINE_AA)

                # 显示 ArUco 检测状态
                if pose:
                    if args.manual:
                        osd(f"[ID={pose.id}] z={pose.T_marker2cam[2,3]:.3f}m  "
                            f"samples:{n}  G/Space/Enter=capture  S/C/Q=finish",
                            28, (80, 220, 80))
                    else:
                        osd(f"[ID={pose.id}] z={pose.T_marker2cam[2,3]:.3f}m  samples:{n}",
                            28, (80, 220, 80))
                else:
                    if args.manual:
                        osd(f"No marker  samples:{n}  move arm to see marker",
                            28, (80, 80, 220))
                    else:
                        osd(f"No marker  samples:{n}",
                            28, (80, 80, 220))

                # 模式指示
                if args.manual:
                    osd("MANUAL: G/Space/Enter=capture  S/C/Q=finish", 50, (180, 180, 60))
                else:
                    osd(f"AUTO: {auto['status']}", 50, (180, 180, 60))

                # 样本进度条（0/15 -> 15/15）
                filled = min(n, 15) * (400 // 15)
                cv2.rectangle(vis, (10, 70), (10 + filled, 82), (0, 200, 100), -1)
                cv2.rectangle(vis, (10, 70), (410, 82), (160, 160, 160), 1)
                cv2.putText(vis, f"{n}/15", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                mode_label = "MANUAL(GravComp)" if args.manual else "AUTO"
                cv2.putText(vis, mode_label, (vis.shape[1] - 200, vis.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1)

                cv2.imshow(WIN, vis)

            # OpenCV 窗口按键处理。手动模式下允许直接在图像窗口里采样，
            # 避免焦点在窗口时终端 Enter/G 没有反应。
            key = cv2.waitKey(30) & 0xFF
            if args.manual and key in (ord("g"), ord("G"), ord(" "), 13, 10):
                capture_sample(latest_pose, "manual window capture")
            elif key in (ord("q"), ord("Q"), ord("c"), ord("C"), ord("s"), ord("S"), 27):
                finish_reason = "window exit"
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                finish_reason = "window closed"
                break

            # 终端输入处理（非阻塞方式从队列读取）
            try:
                if line_queue is not None and handle_line(line_queue.get_nowait()):
                    finish_reason = "user interrupted"
                    break
            except queue.Empty:
                pass

            # 自动模式下，所有位姿完成则退出主循环
            if auto["finished"]:
                break

    except KeyboardInterrupt:
        finish_reason = "Ctrl+C interrupt"
        print("\n[Ctrl+C] Stopping and trying to solve")

    finally:
        # 清理资源
        cv2.destroyAllWindows()
        cam.close()
        if gc_ctrl is not None:
            # 手动模式：重力补偿回 Home
            gc_ctrl.safe_home()
        elif controller is not None:
            # 自动模式：SDK 控制器回 Home
            safe_home_and_disconnect()
        # 求解并保存手眼标定结果
        compute_and_save(finish_reason)

    print(f"\nDone, total samples: {calibrator.n_samples}.")
    if calibrator.n_samples > 0 and not result_saved:
        print("Tip: hand_eye.npz was not generated; collect more samples and try again.")


if __name__ == "__main__":
    main()
    os._exit(0)  # 强制退出，避免 OpenCV 窗口残留
