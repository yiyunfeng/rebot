"""机械臂视觉抓取的夹爪驱动辅助模块。

SDK 自身负责机械臂连接、模式切换、笛卡尔规划和重力补偿控制回路。
本模块仅提供视觉抓取工作流所需的夹爪和末端位姿辅助功能。

selected_arm_config():
  读取 SDK 硬件 YAML 配置，选择对应的控制器模式.

GraspDriver 类:
  夹爪驱动，实现了夹爪状态机和力控闭合逻辑.

  核心 API：
    start(): 启动 SDK 控制回路，附加夹爪状态轮询.
    open_gripper(distance_m): 打开夹爪到指定间距.
    grasp(force): 力控闭合夹爪，检测物体接触（失速检测）.
    release_gripper(): 先打开再闭合到初始位置.
    get_gripper_state(): 返回缓存的夹爪位置、速度和力矩.
    get_tcp_pose(): 返回当前末端 TCP 位姿的 4x4 齐次矩阵.

  夹爪状态机（gripper_tick）:
    IDLE     -> POSITION: 用户调用 open_gripper/grasp 后，目标位置驱动.
    POSITION -> IDLE:     到达目标位置后自动停止.
    POSITION -> CLOSING:  grasp() 设置力控闭合模式.
    CLOSING  -> HOLDING:  夹爪移动超过启动距离且速度低于失速阈值 -> 抓住物体.
    CLOSING  -> POSITION: 夹爪接近硬限位 -> 空抓（无物体）.
    HOLDING  -> (保持):   维持当前抓取力.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


# 项目根目录（rebot_grasp）
_CAMERAWS_ROOT = Path(__file__).resolve().parents[2]
_REBOT_REPO_NAME = "reBotArm_control_py"
# 默认 SDK 仓库路径
_DEFAULT_REBOT_REPO = _CAMERAWS_ROOT / "sdk" / _REBOT_REPO_NAME

# 夹爪最大打开距离（米）
GRIPPER_MAX_DISTANCE_M = 0.09


@dataclass(frozen=True)
class SelectedArmConfig:
    """选定的机械臂硬件类型和对应的 SDK 控制器模式。

    使用 frozen=True 的 dataclass 保证不可变性。
    """
    arm_type: str
    """机械臂硬件类型，当前仅支持 'dm' (大秒电机)."""
    controller_mode: str
    """SDK 控制器模式，DM 型号固定为 'posvel'."""


def _is_rebot_repo_root(path: Path) -> bool:
    """判断给定路径是否为 reBotArm_control_py 仓库根目录。

    检查条件：
      1. 路径是一个存在的目录.
      2. 包含 actuator/rebotarm.py 文件.
      3. 包含 config/rebotarm.yaml 配置文件.
    """
    pkg = path / _REBOT_REPO_NAME
    return (
        path.is_dir()
        and (pkg / "actuator" / "rebotarm.py").is_file()
        and (path / "config" / "rebotarm.yaml").is_file()
    )


def find_rebot_repo_root(hint: Optional[str] = None) -> Path:
    """查找 reBotArm_control_py 仓库根目录的绝对路径。

    查找逻辑：
      1. 如果提供了 hint，以其为入口；否则使用默认路径.
      2. 相对路径相对于项目根目录解析；绝对路径直接使用.
      3. 通过 _is_rebot_repo_root 验证路径有效性.

    参数：
        hint: 用户指定的仓库路径提示，可以为 None 使用默认路径.

    返回：
        验证通过的仓库根目录绝对路径.

    异常：
        FileNotFoundError: 路径无效或不是有效的仓库根目录.
    """
    # 用户路径优先；未提供时使用 rebot_grasp/sdk 下的项目默认 SDK。
    repo = Path(hint).expanduser() if hint else _DEFAULT_REBOT_REPO
    if not repo.is_absolute():
        repo = (_CAMERAWS_ROOT / repo).resolve()
    else:
        repo = repo.resolve()
    if _is_rebot_repo_root(repo):
        return repo
    raise FileNotFoundError(f"reBotArm_control_py repo not found: {repo}")


def ensure_rebot_sdk_in_syspath(hint: Optional[str] = None) -> Path:
    """确保 reBotArm_control_py 仓库在 Python 搜索路径中。

    参数：
        hint: 用户指定的仓库路径提示.

    返回：
        仓库根目录路径.
    """
    repo = find_rebot_repo_root(hint)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def _read_yaml(path: Path) -> dict:
    """安全读取 YAML 文件，验证顶层为字典结构。

    参数：
        path: YAML 文件路径.

    返回：
        解析后的字典.

    异常：
        ValueError: 顶层不是字典时抛出.
    """
    # safe_load 只解析普通 YAML 数据，不构造任意 Python 对象。
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping")
    return data


def selected_hardware_yaml(repo_root: Optional[str] = None) -> Path:
    """读取硬件 YAML 配置文件的路径。

    从 rebotarm.yaml 中的 hardware_yaml 字段读取硬件配置文件路径，
    支持相对路径（相对于 config 目录）和绝对路径.

    参数：
        repo_root: 仓库根目录路径.

    返回：
        硬件 YAML 文件的绝对路径.

    异常：
        ValueError: hardware_yaml 字段缺失.
        FileNotFoundError: 硬件配置文件不存在.
    """
    # rebotarm.yaml 是总配置，它的 hardware_yaml 字段再指向具体型号配置。
    repo = find_rebot_repo_root(repo_root)
    config_dir = repo / "config"
    global_cfg = _read_yaml(config_dir / "rebotarm.yaml")
    hw_yaml = global_cfg.get("hardware_yaml")
    if not hw_yaml:
        raise ValueError(f"{config_dir / 'rebotarm.yaml'} missing hardware_yaml")

    # 相对 hardware_yaml 以 SDK 的 config/ 为基准，绝对路径保持原样。
    hw_path = Path(str(hw_yaml))
    if not hw_path.is_absolute():
        hw_path = config_dir / hw_path
    hw_path = hw_path.resolve()
    if not hw_path.is_file():
        raise FileNotFoundError(f"Hardware config not found: {hw_path}")
    return hw_path


def selected_arm_config(repo_root: Optional[str] = None) -> SelectedArmConfig:
    """返回当前选定的机械臂类型和匹配的 SDK 控制器模式。

    选择逻辑：
      从硬件文件名推断机械臂型号（例如 xxx_dm.yaml -> DM 型号）。
      当前项目仅支持 DM 型号，非 DM 配置会直接报错。
      这样确保在发出任何电机指令之前就排除不兼容的硬件.

    参数：
        repo_root: 仓库根目录路径.

    返回：
        SelectedArmConfig 对象，包含 arm_type 和 controller_mode.

    异常：
        ValueError: 非 DM 硬件配置时抛出.
    """
    # 当前 SDK 用文件名后缀标识电机类型，例如 rebotarm_dm.yaml。
    hw_path = selected_hardware_yaml(repo_root)
    stem = hw_path.stem.lower()
    if stem.endswith("_dm") or stem == "dm":
        return SelectedArmConfig(arm_type="dm", controller_mode="posvel")
    raise ValueError(f"Only DM hardware config is supported; got: {hw_path}")


class GraspDriver:
    """夹爪驱动，封装夹爪状态机和力控闭合抓取逻辑。

    夹爪状态机共有四个状态：
      IDLE:     空闲，等待指令.
      POSITION: 位置控制，正在移动到目标位置.
      CLOSING:  力控闭合，夹爪正在闭合，监测物体接触.
      HOLDING:  抓取保持，持续施加保持力矩.

    状态转换图:
      IDLE -> POSITION: open_gripper() 或 grasp() 触发.
      POSITION -> IDLE: 到达目标位置.
      CLOSING -> HOLDING: 检测到物体接触（失速检测）.
      CLOSING -> POSITION: 到达硬限位（空抓，无物体）.
    """

    MAX_DISTANCE_M = GRIPPER_MAX_DISTANCE_M

    # 夹爪状态枚举
    _STATE_IDLE = "idle"
    _STATE_POSITION = "position"
    _STATE_CLOSING = "closing"
    _STATE_HOLDING = "holding"

    def __init__(
        self,
        arm: Any,
        controller: Any,
        gripper_config: Optional[dict] = None,
        repo_root: Optional[str] = None,
    ) -> None:
        """绑定机械臂控制器，并从硬件配置生成夹爪方向、限位和力矩参数。

        构造函数只准备状态，不会立即驱动电机；调用 :meth:`start` 后控制循环
        才开始运行。方向符号来自硬件配置，不能简单假定“角度增大就是开爪”。
        """
        self._arm = arm
        self._controller = controller
        # 机械臂关节组（arm 组）和夹爪组（gripper 组）
        self._arm_group = arm.groups.get("arm")
        self._gripper_group = arm.groups.get("gripper")
        if self._arm_group is None:
            raise ValueError("Hardware config missing groups.arm")
        if self._gripper_group is None or not arm.has_gripper:
            raise ValueError("Hardware config missing groups.gripper")
        gripper_jcfgs = getattr(self._gripper_group, "_jcfgs", [])
        if not gripper_jcfgs:
            raise ValueError("groups.gripper has no joints")
        # 夹爪电机名称
        self._gripper_name = gripper_jcfgs[0].name
        self._gripper_motor: Any = None

        # 导入正运动学计算函数
        from reBotArm_control_py.kinematics import compute_fk, load_robot_model, pad_q_for_model

        self._compute_fk = compute_fk
        self._pad_q_for_model = pad_q_for_model
        self._model = load_robot_model()
        self._n = self._arm_group.num_joints

        # 读取硬件配置，初始化夹爪运动参数
        selected = selected_arm_config(repo_root)
        defaults = {
            "dm": {
                "angle_open": 5.0,           # 夹爪全开角度
                "counterclockwise": True,    # 开爪方向（逆时针）
                "tau_max": 1.5,              # 最大力矩上限
                "close_torque": 1.0,         # 闭合力矩
                "default_force": 0.30,       # 默认抓取力
            },
        }[selected.arm_type]
        # 合并用户配置与默认值
        gcfg = {**defaults, **((gripper_config or {}).get(selected.arm_type) or {})}
        # 运动方向符号：counterclockwise=True 时为 +1, False 时为 -1
        motion_sign = 1.0 if bool(gcfg.get("counterclockwise")) else -1.0
        # 开爪目标角度（绝对值 * 方向符号）
        self._angle_open = -motion_sign * abs(float(gcfg["angle_open"]))
        self._tau_max = abs(float(gcfg["tau_max"]))
        self._open_sign = 1.0 if self._angle_open >= 0.0 else -1.0
        self._close_sign = motion_sign
        # 闭合力矩 = 方向符号 * 力矩绝对值
        self._close_torque = self._close_sign * abs(float(gcfg["close_torque"]))
        self._default_force = self._close_sign * abs(float(gcfg["default_force"]))
        # 开爪软限位：全开角度的 98%
        self._open_soft_limit = 0.98 * self._angle_open
        self._open_lo = min(self._open_soft_limit, 0.0)
        self._open_hi = max(self._open_soft_limit, 0.0)
        # 硬限位角度：用于检测夹爪是否闭合到物理极限（空抓）
        self._hard_stop_angle = self._open_sign * 0.05
        # 位置到达容忍度
        self._arrive_tol = 0.12
        # 位置控制的 PD 增益
        self._kp_move = 5.0
        self._kd_move = 1.0
        # 力控闭合的阻尼增益
        self._kd_close = 0.5
        # 失速检测速度阈值：夹爪速度低于此值认为已抓住物体
        self._stall_vel = 0.05
        # 启动距离：夹爪移动超过此距离后才开始检测失速
        self._startup_dist = 0.30
        # 状态锁：保护多线程访问夹爪状态
        self._state_lock = threading.Lock()
        self._state = self._STATE_IDLE
        self._target_pos = 0.0
        self._start_pos = 0.0          # grasp() 调用时的起始位置
        self._contact_pos = 0.0        # 接触位置
        self._hold_torque = self._default_force  # 保持力矩
        self._position_reached = True
        self._grasp_result: Optional[bool] = None  # None=进行中, True=成功, False=空抓
        self._last_gripper_state: Optional[tuple[float, float, float]] = None

    def start(self) -> None:
        """启动 SDK 机械臂控制器，并让本驱动接管夹爪控制。

        启动步骤：
          1. 连接机械臂并配置 arm 组为 pos_vel 模式.
          2. 配置 gripper 组为 MIT 模式（力控能力）.
          3. 将当前关节角作为初始目标值，避免上电跳变.
          4. 启动 SDK 控制回路，将 gripper_tick 挂入回路中.
        """
        if getattr(self._controller, "_running", False):
            return

        # 禁用 SDK 内置夹爪控制，由本驱动接管
        self._controller._has_gripper = False
        self._arm.connect()
        self._gripper_motor = self._gripper_group._mm[self._gripper_name]
        if self._arm_group:
            # arm 组：pos_vel 模式（位置-速度控制）
            if self._controller._arm_control_mode == "mit":
                self._arm_group.mode_mit(
                    kp=self._arm_group._mit_kp,
                    kd=self._arm_group._mit_kd,
                )
            else:
                self._arm_group.mode_pos_vel()
            self._arm_group.enable()

        # gripper 组：MIT 模式（支持力矩前馈，用于力控抓取）
        self._gripper_group.mode_mit()
        self._gripper_group.enable()
        # 初始化 arm 目标位置为当前位置
        self._prime_arm_target()
        # 初始化夹爪状态
        self._prime_gripper_state()
        # 启动 SDK 控制回路，gripper_tick 挂入 loop_cb
        self._arm.start_control_loop(self._loop_cb)
        self._controller._running = True

    def _loop_cb(self, r: Any, dt: float) -> None:
        """SDK 控制回路回调：先执行控制器回调，再执行夹爪状态机轮询。"""
        self._controller._loop_cb(r, dt)
        self.gripper_tick(dt)

    def _ensure_running(self) -> None:
        """确保 GraspDriver 已启动，否则抛出异常。"""
        if not getattr(self._controller, "_running", False):
            raise RuntimeError("GraspDriver is not started; call grasp_driver.start() first")

    def _send_gripper_mit(
        self,
        pos: float,
        vel: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        tau: float = 0.0,
    ) -> None:
        """发送 MIT 控制指令到夹爪电机。

        MIT 控制模式 = 位置PD + 速度阻尼 + 力矩前馈:
          tau_total = kp*(pos - actual_pos) + kd*(vel - actual_vel) + tau_ff

        参数：
            pos: 目标位置.
            vel: 目标速度.
            kp: 位置比例增益.
            kd: 速度阻尼增益.
            tau: 力矩前馈（用于抓取保持力矩补偿）.
        """
        # 下发前同时限制位置和力矩，防止上层参数越过夹爪机械范围。
        pos_cmd = float(np.clip(pos, self._open_lo, self._open_hi))
        tau_cmd = float(np.clip(tau, -self._tau_max, self._tau_max))

        # 夹爪组只有一个电机，所以每个 MIT 参数都包装成长度为 1 的数组。
        self._gripper_group.send_mit(
            np.array([pos_cmd], dtype=np.float64),
            vel=np.array([vel], dtype=np.float64),
            kp=np.array([kp], dtype=np.float64),
            kd=np.array([kd], dtype=np.float64),
            tau=np.array([tau_cmd], dtype=np.float64),
        )

    def _prime_arm_target(self) -> None:
        """将 arm 目标位置初始化为当前位置，避免上电跳变。"""
        q_now = self._arm.get_state()[0][: self._n]
        self._controller._q_target[:] = q_now
        self._controller._qd_target[:] = 0.0

    def _prime_gripper_state(self, timeout: float = 1.0) -> None:
        """初始化夹爪状态：从电机读取当前位置作为初始目标位置。

        循环请求电机反馈直到获取到有效状态或超时。
        将 target_pos 和 contact_pos 设为当前位置，状态设为 IDLE.
        """
        # 轮询而不是固定 sleep：反馈一到即可继续，最迟在 timeout 后结束等待。
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._gripper_group._request_feedback()
            state = self._read_gripper_state_cached()
            if state is not None:
                # 状态锁保证控制线程不会在这些相关字段更新一半时读取到不一致状态。
                with self._state_lock:
                    self._target_pos = state[0]
                    self._contact_pos = state[0]
                    self._state = self._STATE_IDLE
                    self._position_reached = True
                    self._grasp_result = None
                return
            time.sleep(0.02)

    def _read_gripper_state_cached(self) -> Optional[tuple[float, float, float]]:
        """读取夹爪电机状态，失败时返回上次缓存值。

        返回：
            (position, velocity, torque) 或 None（无缓存时）.
        """
        if self._gripper_motor is None:
            return self._last_gripper_state
        st = self._gripper_motor.get_state()
        if st is None:
            return self._last_gripper_state
        self._last_gripper_state = (float(st.pos), float(st.vel), float(st.torq))
        return self._last_gripper_state

    def _wait_gripper_state(self, timeout: float = 1.0) -> tuple[float, float, float]:
        """阻塞等待直到获取到有效的夹爪状态。

        异常：
            RuntimeError: 超时后仍未获取到状态.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._read_gripper_state_cached()
            if state is not None:
                return state
            time.sleep(0.02)
        raise RuntimeError("Gripper feedback is not ready")

    def get_gripper_state(self) -> tuple[float, float, float]:
        """获取当前夹爪状态。

        返回：
            (position, velocity, torque) 三元组.

        异常：
            RuntimeError: 无缓存状态时抛出.
        """
        state = self._read_gripper_state_cached()
        if state is None:
            raise RuntimeError("Gripper feedback is not ready")
        return state

    def _set_position_target(self, target: float) -> None:
        """设置夹爪目标位置，将状态切换到 POSITION。

        参数：
            target: 目标位置（角度）.
        """
        with self._state_lock:
            self._target_pos = float(target)
            self._state = self._STATE_POSITION
            self._position_reached = False
            self._grasp_result = None

    def _wait_until(self, predicate, timeout: float) -> bool:
        """轮询等待直到谓词为 True 或超时。

        参数：
            predicate: 无参谓词函数.
            timeout: 超时时间（秒）.

        返回：
            True 如果谓词在超时前变为 True，否则 False.
        """
        # monotonic() 不受系统时间校准影响，适合计算硬件操作超时。
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        # 截止瞬间再检查一次，避免条件刚变为 True 却被循环边界判成失败。
        return predicate()

    def _position_done(self) -> bool:
        """检查位置移动是否完成。"""
        with self._state_lock:
            return self._position_reached

    def _grasp_done(self) -> bool:
        """检查抓取操作是否完成（无论成功或失败）。"""
        with self._state_lock:
            return self._grasp_result is not None

    def gripper_tick(self, dt: float = 0.0) -> None:
        """夹爪状态机轮询（每个控制周期调用一次）。

        这是 GraspDriver 的核心逻辑，在每个控制周期执行一次。根据当前状态
        决定发送何种 MIT 控制指令：

        状态转换逻辑：
          IDLE:     不发送指令.
          POSITION: 发送位置 PD 控制指令，到达目标后标记 _position_reached.
          CLOSING:  发送力矩控制指令（零位置 + 阻尼 + 闭合力矩），
                    监测失速检测条件：
                      - 夹爪移动超过启动距离 且 到达硬限位 -> 空抓（_grasp_result=False）.
                      - 夹爪移动超过启动距离 且 速度低于失速阈值 -> 抓住物体（_grasp_result=True）.
          HOLDING:  发送位置 + 保持力矩，维持抓取状态.

        失速检测原理：
          当夹爪在力控下闭合时：
          1. 如果空爪（无物体）：夹爪会持续运动直到碰到硬限位 -> 位置接近硬限位.
          2. 如果抓到了物体：夹爪碰到物体后会减速（速度 < stall_vel）-> 失速.

        启动距离保护（_startup_dist）：
          闭合开始阶段，夹爪可能还在加速，速度不稳定。只有移动超过
          startup_dist 距离后，才开始判断失速，避免误判。
        """
        del dt
        pos_vel_torq = self._read_gripper_state_cached()
        with self._state_lock:
            state = self._state
            target = self._target_pos
            command: Optional[tuple[float, float, float, float, float]] = None

            if state == self._STATE_POSITION:
                # 位置控制模式：PD 控制 + 零力矩前馈
                command = (target, 0.0, self._kp_move, self._kd_move, 0.0)
                # 到达目标位置判定
                if pos_vel_torq is not None and abs(pos_vel_torq[0] - target) < self._arrive_tol:
                    self._position_reached = True

            elif state == self._STATE_CLOSING:
                # 力控闭合模式：零位置目标 + 阻尼 + 闭合力矩
                # MIT 指令 = kd*(0 - actual_vel) + close_torque
                command = (0.0, 0.0, 0.0, self._kd_close, self._close_torque)
                if pos_vel_torq is not None:
                    pos, vel, _ = pos_vel_torq
                    self._contact_pos = pos
                    moved = abs(pos - self._start_pos) >= self._startup_dist
                    # 检查是否到达硬限位（夹爪完全闭合到物理极限）
                    at_hard_stop = self._open_sign * pos < self._open_sign * self._hard_stop_angle
                    if moved and at_hard_stop:
                        # 空抓：夹爪到达硬限位，没有物体
                        self._target_pos = 0.0
                        self._state = self._STATE_POSITION
                        self._position_reached = False
                        self._grasp_result = False
                        command = (0.0, 0.0, self._kp_move, self._kd_move, 0.0)
                    elif moved and abs(vel) < self._stall_vel:
                        # 抓住物体：速度低于失速阈值
                        self._target_pos = pos
                        self._state = self._STATE_HOLDING
                        self._grasp_result = True
                        command = (pos, 0.0, self._kp_move, self._kd_move, self._hold_torque)

            elif state == self._STATE_HOLDING:
                # 保持模式：维持当前位置 + 保持力矩
                command = (self._target_pos, 0.0, self._kp_move, self._kd_move, self._hold_torque)

        # 发送 MIT 控制指令
        if command is not None:
            pos, vel, kp, kd, tau = command
            self._send_gripper_mit(pos, vel=vel, kp=kp, kd=kd, tau=tau)

    def open_gripper(self, distance_m: float = GRIPPER_MAX_DISTANCE_M, timeout: float = 3.0) -> None:
        """打开夹爪到指定间距。

        将目标间距（米）映射为夹爪角度：
          夹爪角度 = (distance_m / MAX_DISTANCE_M) * angle_open

        参数：
            distance_m: 目标夹爪间距（米），范围 [0, MAX_DISTANCE_M].
            timeout: 超时时间（秒）.
        """
        self._ensure_running()
        d = float(np.clip(distance_m, 0.0, self.MAX_DISTANCE_M))
        # 距离 -> 角度映射
        raw_target = (d / self.MAX_DISTANCE_M) * self._angle_open
        target = float(np.clip(raw_target, self._open_lo, self._open_hi))

        self._set_position_target(target)
        self._wait_until(self._position_done, timeout)

    def grasp(self, force: Optional[float] = None, timeout: float = 5.0) -> bool:
        """力控闭合抓取，检测物体接触。

        抓取流程：
          1. 记录当前位置作为起始位置.
          2. 设置保持力矩（默认或用户指定）.
          3. 状态切换为 CLOSING，开始力控闭合.
          4. 等待失速检测结果（grasp_result）.
          5. 超时后若未完成，强制回退到当前位置并返回 False.

        力控闭合原理：
          发送零位置 + 阻尼 + 闭合力矩指令。夹爪持续闭合，当碰到物体时
          阻力增加导致速度下降。速度低于 stall_vel 判定为抓住物体。

        参数：
            force: 抓取力（力矩），None 时使用默认值 default_force.
            timeout: 超时时间（秒）.

        返回：
            True 如果成功抓住物体，False 如果空抓或超时.
        """
        self._ensure_running()
        start_pos, _, _ = self._wait_gripper_state()
        # 计算保持力矩（带符号）
        hold_torque = self._close_sign * float(
            np.clip(abs(force if force is not None else self._default_force), 0.05, self._tau_max)
        )
        with self._state_lock:
            self._start_pos = start_pos
            self._contact_pos = start_pos
            self._target_pos = 0.0
            self._hold_torque = hold_torque
            self._state = self._STATE_CLOSING
            self._position_reached = False
            self._grasp_result = None

        # 等待抓取完成或超时
        if not self._wait_until(self._grasp_done, timeout):
            # 超时处理：回退到当前位置
            with self._state_lock:
                if self._grasp_result is None:
                    self._target_pos = self._contact_pos
                    self._state = self._STATE_POSITION
                    self._position_reached = False
                    self._grasp_result = False

        with self._state_lock:
            return bool(self._grasp_result)

    def release_gripper(self, timeout: float = 4.0) -> None:
        """释放夹爪：先完全打开，再闭合到初始位置（0）.

        参数：
            timeout: 超时时间（秒）.
        """
        self._ensure_running()
        self.open_gripper(timeout=min(2.0, timeout))
        self._set_position_target(0.0)
        self._wait_until(self._position_done, timeout)

    def get_tcp_pose(self) -> np.ndarray:
        """通过正运动学计算当前末端 TCP 位姿。

        计算流程：
          1. 读取当前 arm 关节角 q_arm.
          2. 将 q_arm 填充为模型完整关节向量.
          3. 调用 Pinocchio 正运动学计算末端位置和旋转矩阵.
          4. 组装为 4x4 齐次变换矩阵 T = [[R, p], [0, 1]].

        返回：
            4x4 齐次变换矩阵，表示末端执行器在世界坐标系中的位姿.
        """
        q_arm = self._arm.get_state(request_feedback=False)[0][: self._n]
        # 将 arm 关节角填充为完整模型关节向量（包括夹爪关节等）
        q = self._pad_q_for_model(self._model, q_arm, self._n)
        # Pinocchio 正运动学：给定关节角 q，计算末端位置和旋转矩阵
        pos, rot, _ = self._compute_fk(self._model, q)
        # 组装为 4x4 齐次变换矩阵
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T
