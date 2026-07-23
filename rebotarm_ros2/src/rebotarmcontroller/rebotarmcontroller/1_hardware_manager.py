"""
hardware_manager 模块 — 硬件抽象层 (HAL)
========================================

本模块是控制器与 reBotArm SDK 之间的适配层，封装所有底层硬件操作。
上层模块（Service/Action/MotorPassthrough）不直接调用 SDK API，
而是通过 HardwareManager 方法间接访问，实现硬件接口与业务逻辑的解耦。

**职责边界**：
  - 硬件生命周期：connect / enable / disable / shutdown / safe_home
  - 实时控制：关节位置/速度/力矩指令、轨迹跟踪、末端位姿控制
  - 状态管理：状态机切换、关节/夹爪状态读取、错误码管理
  - 重力补偿：基于动力学模型的前馈力矩补偿
  - 线程安全：所有指令通过 _locked 装饰器串行化

**状态机模型**：
  IDLE ──→ TRAJ_RUNNING ──→ IDLE          (轨迹执行)
  IDLE ──→ GRAVITY_COMP ──→ IDLE          (重力补偿)
  IDLE ──→ SAFE_HOMING ──→ IDLE           (安全回零)
  IDLE ──→ LOWLEVEL_STREAMING ──→ IDLE    (直通指令流)

  GRAVITY_COMP / SAFE_HOMING 期间拒绝一切其他指令
"""

from __future__ import annotations

import functools
import threading
import time

import numpy as np

from .conversions import fk_to_pose
from .hardware_config import resolve_hardware_config

# 夹爪目标位置容差（弧度）
_GRIPPER_GOAL_TOLERANCE_RAD = 0.12
# 夹爪闭合时的电机位置
_GRIPPER_CLOSED_POSITION = 0.0


# ══════════════════════════════════════════════════════════════════════
# 线程安全装饰器
# ══════════════════════════════════════════════════════════════════════

def _locked(method):
    """
    方法级可重入锁装饰器。

    使用 RLock（可重入锁）而非普通 Lock 的原因：
      - HardwareManager 内部方法可能互相调用（如 safe_home 内调 set_gripper_position）
      - RLock 允许同一线程多次获取锁，避免死锁
      - 确保所有对 SDK 状态的修改操作是线程安全的
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._cmd_lock:
            return method(self, *args, **kwargs)
    return wrapper


class HardwareManager:
    """ROS-facing adapter for the new grouped reBotArm SDK."""

    def __init__(
        self, hardware_config: str | None = None, model: str = "", channel: str = "",
    ) -> None:
        """
        初始化硬件管理器。

        加载顺序：
          1. 解析硬件配置（YAML → _runtime 字典）
          2. 延迟导入 SDK 模块（必须在 sys.path 配置完成后）
          3. 初始化 RebotArm 实例 + 机械臂/夹爪分组
          4. 加载运动学/动力学模型用于重力补偿
          5. 缓存运行时增益参数
        """
        # ----- 线程锁 -----
        self._cmd_lock = threading.RLock()

        # ----- 配置加载 -----
        hardware_config_path, hardware_data = resolve_hardware_config(
            hardware_config, model, channel,
        )

        # ----- 延迟导入 SDK（必须在 sys.path 配置后进行）-----
        from reBotArm_control_py.actuator import RebotArm
        from reBotArm_control_py.controllers import RebotArmEndPose
        from reBotArm_control_py.dynamics import compute_generalized_gravity
        from reBotArm_control_py.kinematics import compute_fk, load_robot_model, pad_q_for_model

        # 缓存运动学函数引用
        self._compute_fk = compute_fk
        self._pad_q_for_model = pad_q_for_model

        # ----- 运行时配置提取 -----
        runtime_config = hardware_data["_runtime"]
        control_runtime = runtime_config["control"]
        self._arm_control_mode = control_runtime["arm_control_mode"]  # "posvel" / "mit"

        # ----- SDK 实例初始化 -----
        self._robot = RebotArm(hw_yaml=str(hardware_config_path))
        self._arm_group = self._robot.groups.get("arm")
        if self._arm_group is None:
            raise ValueError("hardware config must define groups.arm")
        self._gripper_group = self._robot.groups.get("gripper")

        # 替换 SDK 默认 get_state → 只返回机械臂关节状态（截取到实际关节数）
        self._robot_get_state = self._robot.get_state
        self._robot.get_state = self._get_arm_state

        # MIT 增益（多关节向量）
        self._arm_mit_kp = np.array(control_runtime["mit_kp"], dtype=np.float64)
        self._arm_mit_kd = np.array(control_runtime["mit_kd"], dtype=np.float64)

        # 末端位姿控制器（处理轨迹/IK/末端保持）
        self._endpos_ctrl = RebotArmEndPose(self._robot, arm_control_mode=self._arm_control_mode)

        # ----- 夹爪初始化 -----
        self._gripper_name = (
            self._gripper_group.joint_names[0]
            if self.has_gripper and self._gripper_group.joint_names
            else ""
        )
        gripper_limits = hardware_data.get("gripper", {}).get("position_limits", {})
        self.gripper_open_position = float(gripper_limits.get("open", 0.0))
        self.gripper_close_position = float(gripper_limits.get("close", 0.0))
        self._gripper_target_position: float | None = None

        # ----- 重力补偿初始化 -----
        self._gc_model = load_robot_model()
        self._gc_data = self._gc_model.createData()
        self._gc_compute_generalized_gravity = compute_generalized_gravity
        gc_runtime = runtime_config["gravity_compensation"]
        self._gravity_comp_kp = np.array(gc_runtime["kp"], dtype=np.float64)
        self._gravity_comp_kd = np.array(gc_runtime["kd"], dtype=np.float64)
        self._gravity_comp_joint_direction = np.array(gc_runtime["joint_direction"], dtype=np.float64)
        self._gravity_comp_tau_scale = np.array(gc_runtime["tau_scale"], dtype=np.float64)

        # ----- 运行时状态 -----
        self._connected = False
        self._enabled = False
        self._control_output_enabled = False
        self._state_machine = "IDLE"
        self._error_codes: list[str] = []
        self._gravity_comp_active = False
        self._gravity_comp_q_last: np.ndarray | None = None
        self._homing_thread: int | None = None

    # ═══════════════════════════════════════════════════════════════
    # 只读属性
    # ═══════════════════════════════════════════════════════════════

    @property
    def joint_names(self) -> list[str]:
        return list(self._arm_group.joint_names)

    @property
    def mode(self) -> str:
        return str(self._arm_group.mode)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def control_loop_active(self) -> bool:
        return bool(self._robot.control_loop_active)

    @property
    def has_gripper(self) -> bool:
        return bool(self._robot.has_gripper)

    @property
    def state_machine(self) -> str:
        return self._state_machine

    @property
    def error_codes(self) -> list[str]:
        return list(self._error_codes)

    @_locked
    def set_state_machine(self, state: str) -> None:
        """
        切换状态机状态（线程安全）。
        允许的值：IDLE / TRAJ_RUNNING / LOWLEVEL_STREAMING / GRAVITY_COMP / SAFE_HOMING
        """
        if state not in ("IDLE", "TRAJ_RUNNING", "LOWLEVEL_STREAMING",
                         "GRAVITY_COMP", "SAFE_HOMING"):
            raise ValueError(f"unsupported state machine value: {state}")
        self._state_machine = state

    # ═══════════════════════════════════════════════════════════════
    # 连接生命周期
    # ═══════════════════════════════════════════════════════════════

    @_locked
    def connect(self) -> None:
        """
        建立硬件连接并启动控制回路。
        异常处理：连接失败时清理所有中间状态，确保不残留半连接。
        """
        if self._connected:
            return
        try:
            self._robot.connect()
            if self.has_gripper:
                self._gripper_target_position = self.get_gripper_state()[0]
            self._start_endpos_loop()
            self._connected = True
            self._enabled = True
        except Exception:
            self._control_output_enabled = False
            self._endpos_ctrl._running = False
            try:
                self._robot.stop_control_loop()
                self._robot.disconnect()
            finally:
                self._connected = False
                self._enabled = False
            raise

    def shutdown(self, disable_after_safe_home: bool = True) -> None:
        """安全关机：safe_home → stop_control_loop → [disable] → disconnect。"""
        if not self._connected:
            return
        try:
            self.safe_home()
            with self._cmd_lock:
                self._robot.stop_control_loop()
            if disable_after_safe_home:
                self.disable()
            with self._cmd_lock:
                self._robot.disconnect()
        finally:
            self._connected = False
            self._enabled = False
            self._control_output_enabled = False
            self.set_state_machine("IDLE")

    # ═══════════════════════════════════════════════════════════════
    # 关节状态
    # ═══════════════════════════════════════════════════════════════

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._get_arm_state()

    def _get_arm_state(self, request_feedback: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从 SDK 读取状态，截取到实际关节数（排除虚拟关节）。"""
        pos, vel, torq = self._robot_get_state(request_feedback=request_feedback)
        n = len(self.joint_names)
        return pos[:n], vel[:n], torq[:n]

    def get_joint_positions(self, request: bool = False) -> np.ndarray:
        return self._arm_group.get_positions(request_feedback=request)

    def get_joint_velocities(self, request: bool = False) -> np.ndarray:
        return self._arm_group.get_velocities(request_feedback=request)

    # ═══════════════════════════════════════════════════════════════
    # 关节控制
    # ═══════════════════════════════════════════════════════════════

    @_locked
    def hold_current_position(self) -> np.ndarray:
        """保持当前位置：将控制器目标设为当前位置 + 零速度。"""
        current = self.get_joint_positions(request=True).copy()
        if self._state_machine != "SAFE_HOMING":
            self._endpos_ctrl._q_target[:] = current
            self._endpos_ctrl._qd_target[:] = 0.0
        return current

    @_locked
    def set_joint_position_target(self, positions) -> None:
        """设置关节位置目标（用于轨迹跟踪）。SAFE_HOMING 时拒绝。"""
        if self._state_machine == "SAFE_HOMING":
            raise RuntimeError("rejecting joint target during safe home")
        target = np.asarray(positions, dtype=np.float64).reshape(-1)
        if len(target) != len(self.joint_names):
            raise ValueError(f"expected {len(self.joint_names)} joint targets, got {len(target)}")
        self.stop_motion()
        self._endpos_ctrl._q_target[:] = target
        self._endpos_ctrl._qd_target[:] = 0.0

    @_locked
    def start_endpos_control(self) -> None:
        """启动或恢复末端位置控制回路。"""
        if self._gravity_comp_active:
            raise RuntimeError("stop gravity compensation before starting endpos control")
        if self._state_machine in ("SAFE_HOMING", "TRAJ_RUNNING"):
            raise RuntimeError(f"rejecting endpos control in state {self._state_machine}")
        if self.control_loop_active:
            self.set_state_machine("IDLE")
            return
        self._robot.stop_control_loop()
        self._start_endpos_loop()
        self._enabled = True
        self.set_state_machine("IDLE")

    def enable(self) -> None:
        self.start_endpos_control()

    @_locked
    def disable(self) -> None:
        """
        失能机械臂：停止重力补偿 → 关闭输出 → 停止运动 → 停止回路 → 失能所有电机。
        """
        if self._gravity_comp_active:
            raise RuntimeError("stop gravity compensation before disable")
        self._control_output_enabled = False
        self.stop_motion()
        self._endpos_ctrl._running = False
        self._robot.stop_control_loop()
        self._robot.disable_all()
        self._enabled = False
        self.set_state_machine("IDLE")

    # ═══════════════════════════════════════════════════════════════
    # 安全回零
    # ═══════════════════════════════════════════════════════════════

    def safe_home(self) -> None:
        """
        安全回零 —— 将机械臂移动到预设的安全位置。

        若正在重力补偿中（MIT 模式）→ 切换回末端控制，恢复 MIT 增益
                        否则 → 停止重力补偿
        否则 → 启动末端位置控制
        闭合夹爪 → 执行 safe_home 轨迹
        _homing_thread 记录当前线程 ID，用于在 _begin_gripper_command 中
        允许 homing 线程操作夹爪（其他线程调用夹爪指令将被拒绝）。
        """
        with self._cmd_lock:
            self.stop_motion()
            self.set_state_machine("IDLE")
            if self._gravity_comp_active:
                if self._arm_control_mode == "mit" and self.control_loop_active:
                    hold_target = (
                        self._gravity_comp_q_last.copy()
                        if self._gravity_comp_q_last is not None
                        else self._arm_group.get_positions(request_feedback=False).copy()
                    )
                    self._arm_group._mit_kp = self._arm_mit_kp.copy()
                    self._arm_group._mit_kd = self._arm_mit_kd.copy()
                    self._endpos_ctrl._q_target[:] = hold_target
                    self._endpos_ctrl._qd_target[:] = 0.0
                    self._endpos_ctrl._running = True
                    self._control_output_enabled = True
                    self._send_endpos_hold_once()
                    self._robot._ctrl_fn = self._endpos_loop_cb
                    self._gravity_comp_active = False
                    self._gravity_comp_q_last = None
                else:
                    self.stop_gravity_compensation()
            else:
                self.start_endpos_control()
            self.set_state_machine("SAFE_HOMING")
            self._homing_thread = threading.get_ident()
        try:
            if self.has_gripper:
                self.set_gripper_position(_GRIPPER_CLOSED_POSITION)
            self._endpos_ctrl.safe_home()
        finally:
            self._homing_thread = None
            self.set_state_machine("IDLE")

    # ═══════════════════════════════════════════════════════════════
    # 零点设置
    # ═══════════════════════════════════════════════════════════════

    @_locked
    def set_zero(self, joint_name: str = "") -> bool:
        """设置关节零点。joint_name 为空→全部，非空→单关节。"""
        self.stop_motion()
        self._robot.stop_control_loop()
        self._endpos_ctrl._running = False
        if joint_name:
            if joint_name not in self._robot._motor_map:
                raise KeyError(f"unknown joint: {joint_name}")
            self._set_zero_single(joint_name)
        else:
            self._robot.set_zero()
        self._enabled = False
        self.set_state_machine("IDLE")
        return True

    # ═══════════════════════════════════════════════════════════════
    # MIT / Pos-Vel 直通指令
    # ═══════════════════════════════════════════════════════════════

    @_locked
    def send_joint_mit_cmd(
        self, joint_name: str, pos: float, vel: float,
        kp: float, kd: float, tau: float,
    ) -> None:
        """
        向单个关节发送 MIT 控制指令。

        MIT 公式：τ = Kp×(q_des - q) + Kd×(qd_des - qd) + τ_ff

        获取当前全部关节位置 → 构造全零速度/零前馈/默认增益 →
        替换目标关节的 5 个参数 → send_mit → LOWLEVEL_STREAMING。
        """
        index = self._joint_index(joint_name)
        self._begin_lowlevel_streaming("mit")
        q = self._arm_group.get_positions(request_feedback=True)
        target_pos = np.array(q, dtype=np.float64, copy=True)
        target_vel = np.zeros(len(self.joint_names), dtype=np.float64)
        target_tau = np.zeros(len(self.joint_names), dtype=np.float64)
        target_kp = np.array(self._arm_mit_kp, dtype=np.float64, copy=True)
        target_kd = np.array(self._arm_mit_kd, dtype=np.float64, copy=True)
        target_pos[index] = float(pos)
        target_vel[index] = float(vel)
        target_kp[index] = float(kp)
        target_kd[index] = float(kd)
        target_tau[index] = float(tau)
        self._arm_group.send_mit(target_pos, vel=target_vel, kp=target_kp,
                                 kd=target_kd, tau=target_tau)
        self.set_state_machine("LOWLEVEL_STREAMING")

    @_locked
    def send_joint_pos_vel_cmd(self, joint_name: str, pos: float, vlim: float) -> None:
        """向单个关节发送位置-速度控制指令。"""
        index = self._joint_index(joint_name)
        self._begin_lowlevel_streaming("pos_vel")
        q = self._arm_group.get_positions(request_feedback=True)
        target_pos = np.array(q, dtype=np.float64, copy=True)
        target_vlim = np.array(getattr(self._arm_group, "_pv_vlim"), dtype=np.float64, copy=True)
        target_pos[index] = float(pos)
        target_vlim[index] = float(vlim)
        self._arm_group.send_pos_vel(target_pos, vlim=target_vlim)
        self.set_state_machine("LOWLEVEL_STREAMING")

    # ═══════════════════════════════════════════════════════════════
    # 末端位姿
    # ═══════════════════════════════════════════════════════════════

    def current_pose(self):
        """通过 FK 计算当前末端位姿 → ROS Pose 消息。"""
        q, _, _ = self.get_joint_state()
        q_padded = self._pad_q_for_model(self._gc_model, q, len(self.joint_names))
        position, rotation, _ = self._compute_fk(self._gc_model, q_padded)
        return fk_to_pose(position, rotation)

    def _require_idle(self, what: str) -> None:
        """前置检查：状态必须是 IDLE，否则抛出 RuntimeError。"""
        state = self._state_machine
        if state in ("TRAJ_RUNNING", "GRAVITY_COMP", "SAFE_HOMING"):
            raise RuntimeError(f"rejecting {what} in state {state}")

    # ═══════════════════════════════════════════════════════════════
    # 轨迹控制
    # ═══════════════════════════════════════════════════════════════

    @_locked
    def begin_trajectory_stream(self) -> None:
        """开始轨迹流 → TRAJ_RUNNING。"""
        self._require_idle("trajectory stream")
        self.start_endpos_control()
        self.set_state_machine("TRAJ_RUNNING")

    def move_to_pose_traj(self, x, y, z, roll, pitch, yaw, duration: float) -> bool:
        """笛卡尔轨迹规划。成功返回 True。"""
        self.begin_trajectory_stream()
        ok = False
        try:
            ok = bool(self._endpos_ctrl.move_to_traj(x, y, z, roll, pitch, yaw, duration))
        finally:
            if not ok and self._state_machine == "TRAJ_RUNNING":
                self.set_state_machine("IDLE")
        return ok

    @_locked
    def move_to_pose_ik(self, x, y, z, roll, pitch, yaw) -> tuple[bool, list[float]]:
        """IK 求解并移动到目标位姿。返回 (成功, 关节解)。"""
        self._require_idle("IK target")
        self.start_endpos_control()
        ok = self._endpos_ctrl.move_to_ik(x, y, z, roll, pitch, yaw)
        return bool(ok), [float(v) for v in self._endpos_ctrl._q_target]

    def get_joint_status_codes(self) -> list[int]:
        """获取每关节的驱动器状态码。异常关节填 0。"""
        codes: list[int] = []
        for name in self.joint_names:
            try:
                st = self._robot._motor_map[name].get_state()
                codes.append(int(st.status_code if st is not None else 0))
            except Exception:
                codes.append(0)
        return codes

    # ═══════════════════════════════════════════════════════════════
    # 重力补偿
    # ═══════════════════════════════════════════════════════════════

    @_locked
    def start_gravity_compensation(self) -> None:
        """
        启动重力补偿模式 —— 机械臂进入「零力拖动」状态。

        原理：
          1. 切换到 MIT 模式，使用低增益（Kp≈0, Kd=small）
          2. 通过动力学模型实时计算重力前馈 τ_gravity
          3. send_mit(q, vel=0, kp≈0, kd=small, tau=τ_gravity)
          4. 用户只需克服微小阻尼即可拖动

        流程：停止回路 → 保存位置 → 切换 MIT 模式 → disable → 等待 100ms →
              enable → 启动重力补偿控制回路 → GRAVITY_COMP
        """
        if self._state_machine in ("TRAJ_RUNNING", "SAFE_HOMING"):
            raise RuntimeError(f"rejecting gravity compensation in state {self._state_machine}")
        self.stop_gravity_compensation()
        if not self._enabled:
            self._arm_group.enable()
            if self.has_gripper:
                self._gripper_group.enable()
            self._enabled = True
        self.stop_motion()
        self._robot.stop_control_loop()
        self._endpos_ctrl._running = False
        self._gravity_comp_q_last = self._arm_group.get_positions(request_feedback=True).copy()
        self._arm_group.mode_mit(kp=self._gravity_comp_kp, kd=self._gravity_comp_kd)
        self._robot.disable_all()
        time.sleep(0.1)  # 等待电机完全停止
        self._robot.enable_all()
        self._enabled = True
        self._gravity_comp_active = True
        arm_rate = float(getattr(self._robot, "_rate", 500.0))
        self._gravity_comp_tick(self._robot, 1.0 / arm_rate)
        self._robot.start_control_loop(self._gravity_comp_tick, rate=arm_rate)
        self.set_state_machine("GRAVITY_COMP")

    @_locked
    def stop_gravity_compensation(self) -> None:
        """停止重力补偿，恢复末端位置保持。"""
        if not self._gravity_comp_active:
            return
        hold_target = (
            self._gravity_comp_q_last.copy()
            if self._gravity_comp_q_last is not None else None
        )
        self._robot.stop_control_loop()
        self._gravity_comp_active = False
        self._gravity_comp_q_last = None
        if self._enabled:
            self._start_endpos_hold(target=hold_target)
        self.set_state_machine("IDLE")

    def gravity_compensation_active(self) -> bool:
        return self._gravity_comp_active

    @staticmethod
    def _angles_near_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        就近原则角度对齐：delta = values - reference，wrap 到 [-π, π]，
        返回 reference + delta_wrapped。避免 2π 跳变。
        """
        delta = values - reference
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        return reference + delta

    def _read_gravity_comp_positions(
        self, *, request: bool = False, reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """读取位置，可选对齐参考值避免 2π 跳变。"""
        q = self._arm_group.get_positions(request_feedback=request)
        ref = reference if reference is not None else self._gravity_comp_q_last
        if ref is not None:
            q = self._angles_near_reference(q, ref)
        self._gravity_comp_q_last = np.array(q, dtype=np.float64, copy=True)
        return self._gravity_comp_q_last.copy()

    def _gravity_comp_tick(self, _robot, dt: float) -> None:
        """
        重力补偿控制回路回调（~500Hz）。

        每 tick：
          1. 尝试获取锁（非阻塞），失败跳过
          2. 读取位置（就近原则）
          3. 应用 joint_direction → pad_q_for_model → 计算 τ_model = G(q)
          4. 映射 τ_motor = τ_model × joint_direction × tau_scale
          5. send_mit(q, vel=0, kp=gc_kp, kd=gc_kd, tau=τ_motor)
        """
        del dt
        if not self._cmd_lock.acquire(blocking=False):
            return
        try:
            if not self._gravity_comp_active:
                return
            q = self._read_gravity_comp_positions(request=True)
            q_for_model = q * self._gravity_comp_joint_direction
            q_model = self._pad_q_for_model(self._gc_model, q_for_model, len(self.joint_names))
            tau_model = self._gc_compute_generalized_gravity(
                self._gc_model, q_model, self._gc_data,
            )[: len(self.joint_names)]
            tau_motor = tau_model * self._gravity_comp_joint_direction * self._gravity_comp_tau_scale
            self._arm_group.send_mit(
                q, vel=np.zeros(len(self.joint_names)),
                kp=self._gravity_comp_kp, kd=self._gravity_comp_kd, tau=tau_motor,
            )
        finally:
            self._cmd_lock.release()

    # ═══════════════════════════════════════════════════════════════
    # 夹爪
    # ═══════════════════════════════════════════════════════════════

    @_locked
    def set_gripper_target(self, position: float) -> None:
        """设置夹爪目标位置，发送 MIT 指令。"""
        self._begin_gripper_command(allow_endpos=True)
        target = float(position)
        self._endpos_ctrl.set_gripper_target(target)
        self._gripper_group.send_mit(
            np.array([target], dtype=np.float64),
            kp=getattr(self._gripper_group, "_mit_kp"),
            kd=getattr(self._gripper_group, "_mit_kd"),
        )
        self._gripper_target_position = target

    def wait_gripper_target(self, timeout: float = 3.0) -> bool:
        """阻塞等待夹爪到达（20Hz 轮询）。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.gripper_reached_target():
                return True
            time.sleep(0.02)
        return False

    def set_gripper_position(self, position: float, timeout: float = 3.0) -> tuple[bool, float]:
        """设置并等待夹爪到达。返回 (到达成功, 最终位置)。"""
        self.set_gripper_target(position)
        reached = self.wait_gripper_target(timeout)
        return reached, self.get_gripper_state()[0]

    def get_gripper_state(self) -> tuple[float, float, float, int]:
        """获取夹爪状态 (pos, vel, torque, status_code)。无夹爪时返回全 0。"""
        if not self.has_gripper or not self._gripper_name:
            return 0.0, 0.0, 0.0, 0
        pos = float(self._gripper_group.get_positions()[0])
        vel = float(self._gripper_group.get_velocities(request_feedback=False)[0])
        status, torque = 0, 0.0
        try:
            st = self._robot._motor_map[self._gripper_name].get_state()
            if st is not None:
                torque = float(st.torq)
                status = int(st.status_code)
        except Exception:
            status = 0
        return float(pos), float(vel), float(torque), status

    def gripper_reached_target(self) -> bool:
        """检查夹爪是否到达目标（容差 0.12 rad）。"""
        if self._gripper_target_position is None:
            return True
        pos = self.get_gripper_state()[0]
        return abs(pos - self._gripper_target_position) < _GRIPPER_GOAL_TOLERANCE_RAD

    @_locked
    def send_gripper_mit_cmd(self, pos: float, vel: float, kp: float, kd: float, tau: float) -> None:
        """夹爪 MIT 直通指令。"""
        self._begin_gripper_command()
        self._begin_gripper_lowlevel("mit")
        self._gripper_group.send_mit(
            np.array([float(pos)], dtype=np.float64),
            vel=np.array([float(vel)], dtype=np.float64),
            kp=np.array([float(kp)], dtype=np.float64),
            kd=np.array([float(kd)], dtype=np.float64),
            tau=np.array([float(tau)], dtype=np.float64),
        )
        self._gripper_target_position = None

    @_locked
    def send_gripper_pos_vel_cmd(self, pos: float, vlim: float) -> None:
        """夹爪 Pos-Vel 直通指令。"""
        self._begin_gripper_command()
        self._begin_gripper_lowlevel("pos_vel")
        self._gripper_group.send_pos_vel(
            np.array([float(pos)], dtype=np.float64),
            vlim=np.array([float(vlim)], dtype=np.float64),
        )
        self._gripper_target_position = None

    def _begin_gripper_command(self, *, allow_endpos: bool = False) -> None:
        """
        夹爪指令前置检查。

        拒绝条件：未使能 / 重力补偿中 / safe_home(非 homing 线程) /
                  轨迹运行中 / 夹爪未初始化。
        允许 homing 线程在 safe_home 期间操作夹爪以闭合。
        """
        if not self._enabled:
            raise RuntimeError("rejecting gripper command while arm is disabled")
        if self._gravity_comp_active or self.state_machine == "GRAVITY_COMP":
            raise RuntimeError("rejecting gripper command during gravity compensation")
        if self.state_machine == "SAFE_HOMING" and threading.get_ident() != self._homing_thread:
            raise RuntimeError("rejecting gripper command during safe home")
        if self.state_machine == "TRAJ_RUNNING":
            raise RuntimeError("rejecting gripper command while trajectory is running")
        if not self.has_gripper or not self._gripper_name:
            raise RuntimeError("gripper is not initialized")
        if not allow_endpos and self.control_loop_active:
            self.stop_motion()
            self._robot.stop_control_loop()
            self._endpos_ctrl._running = False

    # ═══════════════════════════════════════════════════════════════
    # 内部辅助
    # ═══════════════════════════════════════════════════════════════

    def _begin_lowlevel_streaming(self, required_mode: str) -> None:
        """
        准备进入直通指令流模式。

        拒绝条件：未使能 / 重力补偿中 / safe_home 中
        轨迹运行中 → 停止轨迹后继续
        准备：停止回路 → 切换模式（如需）→ LOWLEVEL_STREAMING
        """
        if not self._enabled:
            raise RuntimeError("rejecting low-level command while arm is disabled")
        if self._gravity_comp_active or self.state_machine == "GRAVITY_COMP":
            raise RuntimeError("rejecting low-level command during gravity compensation")
        if self.state_machine == "SAFE_HOMING":
            raise RuntimeError("rejecting low-level command during safe home")
        if self.state_machine == "TRAJ_RUNNING":
            self.stop_motion()
        self._robot.stop_control_loop()
        self._endpos_ctrl._running = False
        if required_mode != self.mode:
            self._enter_mode(self._arm_group, required_mode, "arm",
                             kp=self._arm_mit_kp, kd=self._arm_mit_kd)
        self.set_state_machine("LOWLEVEL_STREAMING")

    def _begin_gripper_lowlevel(self, required_mode: str) -> None:
        """切换夹爪组到指定模式。"""
        self._enter_mode(self._gripper_group, required_mode, "gripper")
        self.set_state_machine("LOWLEVEL_STREAMING")

    @staticmethod
    def _enter_mode(group, required_mode: str, label: str, **mit_gains) -> None:
        """将 group 切换到指定的控制模式。'mit' → mode_mit / 'pos_vel' → mode_pos_vel。"""
        if required_mode == "mit":
            ok = group.mode_mit(**mit_gains)
        elif required_mode == "pos_vel":
            ok = group.mode_pos_vel()
        else:
            raise ValueError(f"unsupported low-level mode: {required_mode}")
        if not ok:
            raise RuntimeError(f"{label} did not enter {required_mode} mode")

    def _start_endpos_hold(self, target: np.ndarray | None = None) -> None:
        """若控制回路未运行，启动末端位置保持。"""
        if self.control_loop_active:
            return
        self._start_endpos_loop(target)

    def _start_endpos_loop(self, target: np.ndarray | None = None) -> None:
        """
        启动末端位置控制回路。
        配置 groups → 设目标（None=hold current）→ enable 输出 → start_control_loop。
        """
        self._configure_groups_for_endpos()
        if target is None:
            self.hold_current_position()
        else:
            self._endpos_ctrl._q_target[:] = np.asarray(target, dtype=np.float64)
            self._endpos_ctrl._qd_target[:] = 0.0
        self._control_output_enabled = True
        self._robot.start_control_loop(self._endpos_loop_cb)
        self._endpos_ctrl._running = True

    def _configure_groups_for_endpos(self) -> None:
        """为末端控制配置 arm(mode → enable) 和 gripper(mit → enable)。"""
        if self._arm_control_mode == "mit":
            self._arm_group.mode_mit(kp=self._arm_mit_kp, kd=self._arm_mit_kd)
        else:
            self._arm_group.mode_pos_vel()
        self._arm_group.enable()
        if self.has_gripper:
            if not self._gripper_group.mode_mit():
                raise RuntimeError("gripper did not enter mit mode")
            self._gripper_group.enable()

    @_locked
    def stop_motion(self) -> None:
        """停止末端控制器运动线程（发送信号 + join 5s 超时）。"""
        self._endpos_ctrl._stop_send.set()
        if self._endpos_ctrl._send_thread is not None:
            self._endpos_ctrl._send_thread.join(timeout=5.0)
        self._endpos_ctrl._moving = False
        self._endpos_ctrl._stop_send.clear()

    def motion_active(self) -> bool:
        return bool(self._endpos_ctrl._moving)

    def _endpos_loop_cb(self, robot, dt: float) -> None:
        """
        末端位置控制回路回调。
        非阻塞获取锁（锁被占用则跳过本 tick）。
        _control_output_enabled=False 时跳过（disable 期间）。
        """
        del dt
        if not self._cmd_lock.acquire(blocking=False):
            return
        try:
            if not self._control_output_enabled:
                return
            self._endpos_ctrl._loop_cb(robot, 0.0)
        finally:
            self._cmd_lock.release()

    def _send_endpos_hold_once(self) -> None:
        """发送一次末端保持指令（safe_home 过渡用）。"""
        if self._arm_control_mode == "mit":
            self._arm_group.send_mit(
                self._endpos_ctrl._q_target, vel=self._endpos_ctrl._qd_target,
                kp=getattr(self._arm_group, "_mit_kp"),
                kd=getattr(self._arm_group, "_mit_kd"),
            )
        else:
            self._arm_group.send_pos_vel(
                self._endpos_ctrl._q_target,
                vlim=getattr(self._arm_group, "_pv_vlim"),
            )

    def _joint_index(self, joint_name: str) -> int:
        """关节名 → 索引。不存在抛出 KeyError。"""
        try:
            return self.joint_names.index(joint_name)
        except ValueError as exc:
            raise KeyError(f"unknown joint: {joint_name}") from exc

    def _set_zero_single(self, joint_name: str) -> None:
        """
        设置单关节零点。
        disable(等300ms) → 轮询等待状态码==0(最多200次，10秒) → set_zero_position。
        """
        self._robot.disable_all()
        time.sleep(0.3)
        motor = self._robot._motor_map[joint_name]
        ctrl = None
        for joint in self._robot._all_joints:
            if joint.name == joint_name:
                ctrl = self._robot._ctrl_map[str(joint.vendor)]
                break
        if ctrl is None:
            raise KeyError(f"unknown joint: {joint_name}")
        for _ in range(200):
            try:
                motor.request_feedback()
                ctrl.poll_feedback_once()
            except Exception:
                pass
            st = motor.get_state()
            if st is not None and st.status_code == 0:
                break
            time.sleep(0.05)
        motor.set_zero_position()
