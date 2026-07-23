#!/usr/bin/env python3
"""真机笛卡尔阻抗控制（单文件实验脚本）。

控制律：
    wrench = K * pose_error
    tau_feedforward = J(q).T @ wrench + gravity(q)
    tau_motor_pd = Kp * (q_target - q) - Kd * qd

脚本启动时的末端位姿就是目标位姿。移动机械臂后，刚度项将末端拉回目标，
电机内部 MIT 位置/速度环使用同一启动关节目标，克服 Python 外环
难以处理的电机静摩擦；外环仍用笛卡尔恢复力保持末端回位特性。

本脚本不会进行碰撞检测。首次真机测试必须清空工作空间、确认急停可用，
力矩上限固定在本文件中，不需要通过命令行重复输入。

运行：
    python3 scripts/cartesian_damping_control.py

程序持续运行，按 Ctrl+C 时失能并断开。
"""

from __future__ import annotations

import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pinocchio as pin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "sdk" / "reBotArm_control_py"
# 优先使用项目内固定版本的控制 SDK，避免误加载系统里另一个同名包。
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from reBotArm_control_py.actuator import RebotArm  # noqa: E402
from reBotArm_control_py.dynamics import compute_generalized_gravity  # noqa: E402
from reBotArm_control_py.kinematics import (  # noqa: E402
    get_end_effector_frame_id,
    load_robot_model,
    pad_q_for_model,
)

# 控制和安全参数集中在这里，便于真机实验前逐项审查。
# 控制状态日志的输出间隔，单位秒；0.2 s 表示每秒最多打印 5 次。
# 只影响终端输出频率，不改变 YAML 请求的控制频率。
LOG_INTERVAL_S = 0.2

# 任一关节绝对速度超过该值时停止控制，单位 rad/s。
MAX_JOINT_SPEED_RAD_S = 2.0

# 检查 URDF 关节角限位时允许的反馈容差，单位 rad；
# 0.02 rad 约等于 1.15°，只用于吸收反馈噪声，不扩大真实运动范围。
JOINT_LIMIT_TOLERANCE_RAD = 0.02

# [J1, J2, J3, J4, J5, J6] 的总力矩估计上限，单位 N*m。
# 前馈与电机内部 MIT 位置/速度力矩合成后，按这六个值限幅。
TORQUE_LIMITS_NM = np.array([3.0, 9.0, 9.0, 3.0, 1.0, 1.0])

# 笛卡尔刚度 K=[Kx, Ky, Kz, KRx, KRy, KRz]：
# 前三项单位 N/m，决定末端偏离目标位置后产生多大的返回力；
# 后三项单位 N*m/rad，决定末端偏离目标姿态后产生多大的返回力矩。
CARTESIAN_STIFFNESS = np.array([30.0, 30.0, 30.0, 1.0, 1.0, 1.0])

# 笛卡尔末端合力/合力矩上限。
MAX_CARTESIAN_FORCE_N = 4.0

# 对笛卡尔返回力矩 [Mx, My, Mz] 的三维合力矩限幅。
MAX_CARTESIAN_MOMENT_NM = 0.5

# 电机本地位置/速度增益，基准值来自项目的重力补偿锁定样例。
# 实际下发的 Kp 会按当前误差动态降低，保证 PD 估计力矩不超上限。
MIT_KP = np.full(6, 8.0, dtype=np.float64)

# 电机 MIT 速度增益。vel=0 时由电机就地产生 -Kd*qd 阻尼力矩；
# 与项目重力补偿锁定样例一致。
MIT_KD = np.full(6, 1.0, dtype=np.float64)


def clamp_norm(values: np.ndarray, limit: float) -> np.ndarray:
    """按向量模长限幅，保持原方向。"""
    norm = float(np.linalg.norm(values))
    if norm <= limit or norm == 0.0:
        return values
    return values * (limit / norm)


class CartesianImpedanceController:
    """将末端位姿误差转换为关节恢复力矩。"""

    def __init__(
        self,
        robot: RebotArm,
        model: pin.Model,
    ) -> None:
        """绑定真机和 Pinocchio 模型，初始化限幅、目标状态与停止标志。

        这里仅准备控制器，不发送力矩；``initialize_target`` 保存启动位姿后，
        由主程序把本对象注册为控制回调。
        """
        self.robot = robot
        self.arm = robot.arm
        self.n = self.arm.num_joints
        self.torque_limits = TORQUE_LIMITS_NM.copy()
        self.model = model
        self.data = model.createData()
        self.ee_frame_id = get_end_effector_frame_id(model)
        self.zero = np.zeros(self.n, dtype=np.float64)
        # 电机本地位置/速度阻抗的基准增益。
        self.motor_kp = MIT_KP.copy()
        self.motor_kd = MIT_KD.copy()
        self.target_pose: pin.SE3 | None = None
        self.target_q: np.ndarray | None = None
        self.error: Exception | None = None
        self.stop_event = threading.Event()
        self._last_log_time = 0.0
        self._cycles_since_log = 0

        # 当前控制律和力矩参数都按 B601 六轴定义，不能直接套到其他自由度模型。
        if self.n != 6 or model.nq < self.n or model.nv < self.n:
            raise ValueError(
                f"当前脚本要求 6 轴机械臂，实际 arm={self.n}, "
                f"model.nq={model.nq}, model.nv={model.nv}"
            )

    def initialize_target(self, q: np.ndarray) -> None:
        """将启动时末端位姿保存为笛卡尔阻抗目标。"""
        # SDK 只返回 arm 的六个关节角；Pinocchio 模型可能还包含夹爪自由度，
        # 因此先补齐为模型要求的 nq，再更新所有 frame 的世界坐标位姿。
        q_model = pad_q_for_model(self.model, q, self.n)
        pin.forwardKinematics(self.model, self.data, q_model)
        pin.updateFramePlacements(self.model, self.data)
        self.target_pose = self.data.oMf[self.ee_frame_id].copy()
        self.target_q = q.copy()

    def _check_state(self, q: np.ndarray, qd: np.ndarray) -> None:
        """校验反馈是否允许继续下发力矩。

        任一反馈非有限、关节速度过大或关节越过 URDF 限位时立即抛错，
        由控制回调切换为零力矩，并通知主线程执行失能和断开。
        """
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            raise RuntimeError("关节反馈包含 NaN/Inf，停止控制")

        # 速度阈值是独立于控制律的最后一道保护，防止反馈异常或低阻尼发散。
        speeding = np.flatnonzero(np.abs(qd) > MAX_JOINT_SPEED_RAD_S)
        if speeding.size:
            details = ", ".join(
                f"{self.arm.joint_names[index]}={qd[index]:+.3f} rad/s"
                for index in speeding
            )
            raise RuntimeError(
                f"关节速度超限（{details}），阈值为 "
                f"{MAX_JOINT_SPEED_RAD_S:.3f} rad/s；"
                f"完整 qd={np.array2string(qd, precision=3)}，停止控制"
            )

        # 容差只吸收编码器噪声和模型边界误差，不用于扩大真实可运动范围。
        lower = self.model.lowerPositionLimit[: self.n] - JOINT_LIMIT_TOLERANCE_RAD
        upper = self.model.upperPositionLimit[: self.n] + JOINT_LIMIT_TOLERANCE_RAD
        is_outside_limits = (q < lower) | (q > upper)
        # flatnonzero 返回一维数组中“非零”元素的下标；对布尔数组来说，
        # 就是所有 True 的位置，因此这里得到越限关节的索引，例如 [1, 4]。
        invalid = np.flatnonzero(is_outside_limits)
        if invalid.size:
            names = [self.arm.joint_names[index] for index in invalid]
            raise RuntimeError(f"关节越过 URDF 限位: {names}，停止控制")

    def _check_gravity_headroom(self, tau_gravity: np.ndarray) -> np.ndarray:
        """确认逐关节上限足以补偿当前姿态重力，并返回所需最小上限。"""
        # 在模型重力力矩上增加 15% 和 0.2N·m 余量，给动态扰动和建模误差留空间。
        required = np.abs(tau_gravity) * 1.15 + 0.2
        insufficient = np.flatnonzero(self.torque_limits < required)
        if insufficient.size:
            details = ", ".join(
                f"{self.arm.joint_names[index]}: 设置 "
                f"{self.torque_limits[index]:.2f}, 至少需要 {required[index]:.2f} N*m"
                for index in insufficient
            )
            raise RuntimeError(
                f"当前姿态的力矩上限不足以留出重力补偿余量（{details}）；"
                "请先支撑机械臂并调整姿态，不要超过电机额定上限"
            )
        return required

    def check_startup_gravity(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """在电机切换模式和使能之前检查启动姿态的重力力矩。"""
        q_model = pad_q_for_model(self.model, q, self.n)
        tau_gravity = compute_generalized_gravity(
            model=self.model,
            q=q_model,
            data=self.data,
        )[: self.n]
        required = self._check_gravity_headroom(tau_gravity)
        return tau_gravity, required

    def _compute_command(
        self,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """根据当前关节状态计算力矩命令。

        返回：
            tau_feedforward: 发给 MIT 的六轴前馈力矩。
            pose_error: 世界对齐坐标系中的末端位置/旋转误差。
            wrench: 限幅后的笛卡尔力/力矩，用于日志与排查。
            tau_gravity: 模型重力补偿力矩。
            tau_task: 笛卡尔恢复力映射到关节的力矩。
            kp_effective: 按力矩上限动态缩小后的 MIT Kp。
            kd_effective: 按力矩上限动态缩小后的 MIT Kd。
            tau_motor_pd: 电机内部位置/速度力矩估计值。
            tau_total_estimate: 前馈与电机 PD 合成后的总力矩估计值。
        """
        # q_model 使用完整 Pinocchio 模型维度；前 n 项对应真实 arm。
        q_model = pad_q_for_model(self.model, q, self.n)

        # 同一周期内先更新运动学、Jacobian 和 frame 位姿，保证三者对应同一 q。
        pin.forwardKinematics(self.model, self.data, q_model)
        pin.computeJointJacobians(self.model, self.data, q_model)
        pin.updateFramePlacements(self.model, self.data)

        current_pose = self.data.oMf[self.ee_frame_id]
        # LOCAL_WORLD_ALIGNED 使平移和旋转分量均沿世界坐标轴表达，
        # 从而能直接与下方世界系 pose_error 做逐轴刚度计算。
        jacobian = pin.getFrameJacobian(
            self.model,
            self.data,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        if self.target_pose is None or self.target_q is None:
            raise RuntimeError("尚未初始化启动目标位姿/关节角")
        pose_error = np.zeros(6, dtype=np.float64)
        # 平移误差和 SO(3) 对数映射得到的旋转误差都在世界对齐坐标系表达。
        pose_error[:3] = self.target_pose.translation - current_pose.translation
        pose_error[3:] = pin.log3(
            self.target_pose.rotation @ current_pose.rotation.T
        )

        # 笛卡尔刚度负责末端空间的柔性回位。
        wrench = CARTESIAN_STIFFNESS * pose_error
        # 力和力矩分别按三维模长限幅，避免逐轴限幅改变空间方向。
        wrench[:3] = clamp_norm(wrench[:3], MAX_CARTESIAN_FORCE_N)
        wrench[3:] = clamp_norm(wrench[3:], MAX_CARTESIAN_MOMENT_NM)

        # Jacobian 转置把末端 wrench 映射为关节任务力矩：tau = J^T F。
        # 只取真实六轴对应的列，模型里的其他自由度不发送给硬件。
        tau_task = jacobian[:, : self.n].T @ wrench
        # 重力补偿由完整动力学模型计算，再提取真实 arm 的六项。
        tau_gravity = compute_generalized_gravity(
            model=self.model,
            q=q_model,
            data=self.data,
        )[: self.n]
        self._check_gravity_headroom(tau_gravity)
        # 电机本地 PD 直接锁定启动关节角。增益按当前误差/速度动态缩小，
        # 使 PD 自身的估计力矩始终不超逐关节上限。
        q_error = self.target_q - q
        abs_qd = np.abs(qd)
        kd_limits = np.full(self.n, np.inf, dtype=np.float64)
        np.divide(
            self.torque_limits,
            abs_qd,
            out=kd_limits,
            where=abs_qd > 1e-9,
        )
        kd_effective = np.minimum(self.motor_kd, kd_limits)
        tau_damping = -kd_effective * qd

        position_headroom = np.maximum(
            self.torque_limits - np.abs(tau_damping),
            0.0,
        )
        abs_q_error = np.abs(q_error)
        kp_limits = np.full(self.n, np.inf, dtype=np.float64)
        np.divide(
            position_headroom,
            abs_q_error,
            out=kp_limits,
            where=abs_q_error > 1e-9,
        )
        kp_effective = np.minimum(self.motor_kp, kp_limits)
        tau_position = kp_effective * q_error
        tau_motor_pd = tau_position + tau_damping

        # 先限幅希望总力矩，再反算前馈 tau；这样电机内部叠加 PD 后
        # 仍不超 TORQUE_LIMITS_NM。
        tau_total_desired = tau_gravity + tau_task + tau_motor_pd
        # np.clip(x, lower, upper) 会把 x 的每个元素分别限制在上下界内：
        # 小于下界的变成下界，大于上界的变成上界，范围内保持不变。
        # 这里的上下界是六元素数组，因此 J1-J6 按各自的力矩上限独立限幅。
        tau_total_limited = np.clip(
            tau_total_desired,
            -self.torque_limits,
            self.torque_limits,
        )
        # 扣除电机内部 PD 后得到 MIT 前馈字段；再 clip 一次，
        # 是为了确保这个实际下发值本身也没有超出各关节范围。
        tau_feedforward = np.clip(
            tau_total_limited - tau_motor_pd,
            -self.torque_limits,
            self.torque_limits,
        )
        tau_total_estimate = tau_feedforward + tau_motor_pd
        return (
            tau_feedforward,
            pose_error,
            wrench,
            tau_gravity,
            tau_task,
            kp_effective,
            kd_effective,
            tau_motor_pd,
            tau_total_estimate,
        )

    def _send_mit_checked(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        tau: np.ndarray,
    ) -> None:
        """逐关节发送 MIT 命令，禁止 SDK 静默吞掉通信异常。"""
        # 不调用会吞异常的批量封装，逐电机发送并保留具体失败关节名称。
        for index, joint_name in enumerate(self.arm.joint_names):
            try:
                self.robot._motor_map[joint_name].send_mit(
                    float(pos[index]),
                    float(vel[index]),
                    float(kp[index]),
                    float(kd[index]),
                    float(tau[index]),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{joint_name} MIT 下发失败: {type(exc).__name__}: {exc}"
                ) from exc

    def __call__(self, _robot: RebotArm, _dt: float) -> None:
        """SDK 实时线程的单周期控制回调。

        回调只读取 SDK 已缓存反馈、计算力矩并发送 MIT 命令；连接管理、
        用户交互和最终失能均由主线程负责，避免阻塞固定频率控制线程。
        """
        if self.stop_event.is_set():
            return

        try:
            # MIT 指令会返回反馈，因此控制周期中读取缓存，避免重复阻塞串口。
            q = self.arm.get_positions(request_feedback=False)
            qd = self.arm.get_velocities(request_feedback=False)
            self._check_state(q, qd)
            (
                tau_feedforward,
                pose_error,
                wrench,
                tau_gravity,
                tau_task,
                kp_effective,
                kd_effective,
                tau_motor_pd,
                tau_total_estimate,
            ) = self._compute_command(q, qd)

            self._send_mit_checked(
                # 固定启动关节目标由电机本地 PD 执行，动态增益已计入力矩限幅。
                pos=self.target_q,
                vel=self.zero,
                kp=kp_effective,
                kd=kd_effective,
                tau=tau_feedforward,
            )

            # 诊断日志限制为 5 Hz，同时统计主机实际完成的控制频率。
            now = time.monotonic()
            self._cycles_since_log += 1
            if self._last_log_time == 0.0:
                self._last_log_time = now
                self._cycles_since_log = 0
            elif now - self._last_log_time >= LOG_INTERVAL_S:
                elapsed = now - self._last_log_time
                actual_rate = self._cycles_since_log / elapsed
                self._last_log_time = now
                self._cycles_since_log = 0
                print(
                    "mode=cartesian_impedance "
                    f"rate={actual_rate:.1f}Hz "
                    f"|e_xyz|={np.linalg.norm(pose_error[:3]):.4f} m "
                    f"|e_rot|={np.linalg.norm(pose_error[3:]):.4f} rad "
                    f"|force|={np.linalg.norm(wrench[:3]):.3f} N "
                    f"q={np.array2string(q, precision=3)} "
                    f"qd={np.array2string(qd, precision=3)} "
                    f"tau_g={np.array2string(tau_gravity, precision=3)} "
                    f"tau_task={np.array2string(tau_task, precision=3)} "
                    f"tau_pd={np.array2string(tau_motor_pd, precision=3)} "
                    f"tau_est={np.array2string(tau_total_estimate, precision=3)}"
                )
        except Exception as exc:
            # 控制线程不能直接完成 disconnect；先记录首个错误并尽力发送零力矩，
            # 主线程检测 error 后进入 finally，停止循环并失能全部电机。
            self.error = exc
            self.stop_event.set()
            try:
                self.arm.send_mit(
                    pos=self.zero,
                    vel=self.zero,
                    kp=self.zero,
                    kd=self.motor_kd,
                    tau=self.zero,
                )
            except Exception:
                # 通信已经失效时零力矩发送也可能失败，最终仍由主线程 disable_all。
                pass


def report_exception(stage: str, exc: Exception) -> None:
    """输出失败阶段、异常类型和完整调用栈。"""
    print(
        f"失败阶段 [{stage}] {type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    traceback.print_exception(
        type(exc),
        exc,
        exc.__traceback__,
        file=sys.stderr,
    )


def print_configuration(robot: RebotArm, model: pin.Model) -> None:
    """打印实际将使用的硬件通道、电机型号和动力学模型，供人工复核。"""
    joints = robot.groups.get("arm")
    models = [cfg.model for cfg in joints._jcfgs] if joints is not None else []
    vendors = [cfg.vendor for cfg in joints._jcfgs] if joints is not None else []
    print(f"hardware_yaml: {robot.hardware_yaml}")
    print(f"channel:       {robot._channel}")
    print(f"motor models:  {models}")
    print(f"vendors:       {vendors}")
    print(f"model:         nq={model.nq}, nv={model.nv}")
    print(f"end frame:     {model.frames[get_end_effector_frame_id(model)].name}")
    print("mode:          cartesian_impedance")
    print(f"torque limits: {TORQUE_LIMITS_NM.tolist()} N*m")


def read_initial_arm_state(
    robot: RebotArm, timeout: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """等待六个 arm 电机都返回有效反馈，禁止用缺失反馈的零值启动。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # 主动请求一轮总线反馈，再逐电机检查状态是否已经写入 SDK 缓存。
        robot.arm.get_positions(request_feedback=True)
        states = [robot._motor_map[name].get_state() for name in robot.arm.joint_names]
        if all(state is not None for state in states):
            q = np.array([state.pos for state in states], dtype=np.float64)
            qd = np.array([state.vel for state in states], dtype=np.float64)
            return q, qd
        time.sleep(0.05)
    raise RuntimeError("2 秒内未收到全部 arm 电机反馈，保持失能")


def main() -> int:
    """完成配置审查、持续控制以及任何退出路径下的失能。"""
    try:
        # 所有不需要硬件运动的检查都放在 connect() 之前。
        robot = RebotArm()
        model = load_robot_model()
        print_configuration(robot, model)
    except Exception as exc:
        report_exception("配置与参数检查", exc)
        return 2

    # 当前 MIT 模式切换和安全验证只覆盖 Damiao，拒绝混用其他 vendor。
    if any(cfg.vendor != "damiao" for cfg in robot.arm._jcfgs):
        print("安全检查失败：该脚本当前只验证了 Damiao 电机。", file=sys.stderr)
        return 2

    print("\n开始真机控制；请确认工作空间清空、急停可用、机械臂状态正常。")

    # 这些状态由 finally 使用，确保连接或初始化中途失败也能正确收尾。
    controller: CartesianImpedanceController | None = None
    connected = False
    stop_requested = threading.Event()
    phase = "准备连接"

    def request_stop(_signum=None, _frame=None) -> None:
        """信号处理器只置位，真正停机由主线程按统一顺序完成。"""
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        phase = "连接硬件"
        robot.connect()
        connected = True
        phase = "清除使能状态"
        # 连接后先清除可能遗留的使能状态，再读取并验证真实起始反馈。
        robot.disable_all()

        phase = "读取初始关节反馈"
        q_start, qd_start = read_initial_arm_state(robot)
        phase = "创建控制器"
        controller = CartesianImpedanceController(
            robot=robot,
            model=model,
        )
        phase = "检查初始关节状态"
        controller._check_state(q_start, qd_start)

        # 切换模式时机械臂仍处于失能状态，因此不能在切换前锁定目标。
        phase = "切换 arm 到 MIT 模式"
        if not robot.arm.mode_mit(
            kp=controller.motor_kp,
            kd=controller.motor_kd,
        ):
            raise RuntimeError("部分 arm 电机切换 MIT 模式失败")

        # 模式切换完成后重新读取姿态，将这一刻的末端位姿作为固定目标。
        # 这样不会因失能切换期间的位置变化而在一启动就产生冲击。
        phase = "读取控制目标姿态"
        q_target, qd_target = read_initial_arm_state(robot)
        controller._check_state(q_target, qd_target)
        controller.initialize_target(q_target)
        phase = "检查目标姿态重力力矩"
        tau_gravity, required_torque = controller.check_startup_gravity(q_target)
        print(
            "目标姿态模型重力力矩: "
            f"{np.array2string(tau_gravity, precision=3)} N*m"
        )
        print(
            "含安全余量的最小上限: "
            f"{np.array2string(required_torque, precision=3)} N*m"
        )
        initial_command = controller._compute_command(q_target, qd_target)
        tau_initial = initial_command[0]
        kp_initial = initial_command[5]
        kd_initial = initial_command[6]

        # JointGroup.enable() 会调用 vendor controller.enable_all()，可能连夹爪
        # 一起使能；这里逐个使能 arm，并立即发送当前姿态的重力前馈，
        # 避免某个关节已使能却要等全部关节完成后才收到第一条命令。
        for index, joint_name in enumerate(robot.arm.joint_names):
            phase = f"使能 {joint_name}"
            try:
                motor = robot._motor_map[joint_name]
                motor.enable()
                motor.send_mit(
                    float(q_target[index]),
                    0.0,
                    float(kp_initial[index]),
                    float(kd_initial[index]),
                    float(tau_initial[index]),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{joint_name} 使能/初始命令失败: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        # controller 实现了 __call__(robot, dt)，因此可作为回调函数传入。
        # 与项目样例一样使用硬件 YAML 的控制频率（当前配置为 500 Hz）。
        phase = "启动控制线程"
        robot.start_control_loop(controller, rate=robot.rate)
        print("控制已启动；按 Ctrl+C 可提前失能。")

        phase = "实时控制循环"
        while not stop_requested.is_set():
            # 控制线程通过 controller.error 把异常转交主线程统一处理。
            if controller.error is not None:
                raise controller.error
            time.sleep(0.02)
        print("停止原因：收到 Ctrl+C/SIGTERM。")
    except Exception as exc:
        report_exception(phase, exc)
        return 3
    finally:
        # 收尾顺序：阻止新命令 → 停控制线程 → 失能 → 断开通信。
        if controller is not None:
            controller.stop_event.set()
        cleanup_failed = False
        if connected:
            cleanup_actions = (
                ("停止控制线程", robot.stop_control_loop),
                ("失能全部电机", robot.disable_all),
                ("断开硬件连接", robot.disconnect),
            )
            for cleanup_stage, action in cleanup_actions:
                try:
                    action()
                except Exception as exc:
                    cleanup_failed = True
                    report_exception(f"清理/{cleanup_stage}", exc)
        if cleanup_failed:
            print("清理过程存在错误，无法确认机械臂已完全失能/断开。", file=sys.stderr)
        elif connected:
            print("机械臂已失能并断开。")
        else:
            print("硬件连接未建立，无需执行失能/断开。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
