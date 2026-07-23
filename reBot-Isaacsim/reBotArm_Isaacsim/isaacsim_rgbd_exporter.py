#!/usr/bin/env python3
"""运行 DM Isaac 场景、导出腕部 RGB-D，并循环执行稳定的仿真抓取放置计划。

Isaac Sim 必须使用官方 ``python.sh``，而 YOLO/SAM 运行在 ``rebotarm_gpu``。
两个进程通过 ``/tmp/rebot_sim_rgbd.npz`` 交换最新帧，避免混用 Python 环境。

工作流程：
1. 加载 DM 仿真场景（含机械臂 + 桌面 + 抓取物体）
2. 移动到观测姿态，开始按固定频率导出腕部相机 RGB-D 帧
3. 感知端读取 RGB-D → 检测物体 → 生成抓取计划 JSON → 写回 /tmp
4. 本进程检测到新计划后执行抓取放置循环，完成后写回执行结果
5. 循环直到 Ctrl+C 或仿真窗口关闭
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# 复用已有的 DM 仿真场景创建与关节控制逻辑
from isaacsim_joint_receiver import DMSimulation


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 该路径来自 rebotarm_dm_with_camera.usd，Camera Prim 会随 gripper_base 运动。
CAMERA_PATH = "/World/base_link/gripper_base/dabai_dcw_camera"

# 与感知端交换数据的文件路径
DEFAULT_OUTPUT = Path("/tmp/rebot_sim_rgbd.npz")           # RGB-D 帧（原子写入）
DEFAULT_PLAN = Path("/tmp/rebot_sim_grasp_plan.json")      # 抓取计划（感知端写，本进程执行后覆写）
DEFAULT_TEACHER_TRAJECTORY = Path("/tmp/rebot_sim_teacher_trajectory.npz")

# Teacher 轨迹收集参数
TEACHER_IMAGE_SIZE = 64                                    # 缩略图分辨率
TEACHER_STAGES = {"pregrasp", "grasp", "close", "retreat"} # 需要采样的阶段

# 全局运行标志，由信号处理函数控制优雅退出
_running = True


# ---------------------------------------------------------------------------
# 信号处理
# ---------------------------------------------------------------------------

def _stop(_signum, _frame) -> None:
    """收到 Ctrl+C/SIGTERM 后让主循环完成当前帧并正常关闭 Isaac Sim。"""
    global _running
    print(f"[RGBD] received signal {signal.Signals(_signum).name}; shutting down")
    _running = False


# ---------------------------------------------------------------------------
# 帧导出
# ---------------------------------------------------------------------------

def _write_frame(
    path: Path,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    T_camera_to_world: np.ndarray,
) -> None:
    """原子替换一整帧，防止感知进程读到写了一半的 NPZ 文件。

    采用「写临时文件 → os.replace 原子重命名」策略，确保读端不会看到
    不完整的帧数据。

    Isaac 深度单位为米；现有 ``rebot_grasp`` 统一接收毫米深度图，因此在这里
    转为 uint16 毫米。无效或过远深度会在转换前被限制到可表示范围。
    """
    # 写入临时文件，避免覆盖过程中被读取
    temp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp_path,
        color_bgr=color_bgr,                                                    # BGR 色彩 (H, W, 3) uint8
        depth_mm=np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16),        # 米→毫米，uint16 范围裁剪
        K=np.asarray(K, dtype=np.float32),                                       # 相机内参 3×3
        T_camera_to_world=np.asarray(T_camera_to_world, dtype=np.float64),      # 相机到世界 4×4 变换
        timestamp=np.array(time.time(), dtype=np.float64),                      # Unix 时间戳
    )
    # 原子替换：os.replace 在同一文件系统上是原子的
    os.replace(temp_path, path)


# ---------------------------------------------------------------------------
# 图像缩放
# ---------------------------------------------------------------------------

def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """用 NumPy 最近邻缩放，避免 Isaac Sim 环境额外依赖 OpenCV。

    Isaac Sim 自带的 Python 环境通常不包含 OpenCV，而 conda 环境的
    OpenCV 无法在此使用。纯 NumPy 实现避免了跨环境依赖问题。
    """
    # 在原始图像坐标上均匀采样，四舍五入到最近像素索引
    rows = np.linspace(0, image.shape[0] - 1, height).round().astype(np.int64)
    cols = np.linspace(0, image.shape[1] - 1, width).round().astype(np.int64)
    # rows[:, None] → (H, 1), cols[None, :] → (1, W)，广播索引 → (H, W)
    return np.ascontiguousarray(image[rows[:, None], cols[None, :]])


# ---------------------------------------------------------------------------
# Teacher 轨迹保存
# ---------------------------------------------------------------------------

def _write_teacher_trajectory(path: Path, trajectory: dict, plan_timestamp: float) -> None:
    """原子保存一条成功抓取的同步 RGB-D/关节轨迹。

    轨迹包含每个采样步骤的：彩色图、深度图、关节位置/速度/目标、
    相机到世界变换、阶段名称标签。用于后续训练 behavioral cloning 等策略。
    """
    # 将列表堆叠为批量数组，统一类型以节省存储
    temp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp_path,
        color_bgr=np.stack(trajectory["color_bgr"]).astype(np.uint8),
        depth_mm=np.stack(trajectory["depth_mm"]).astype(np.uint16),
        joint_positions=np.stack(trajectory["joint_positions"]).astype(np.float32),
        joint_velocities=np.stack(trajectory["joint_velocities"]).astype(np.float32),
        joint_targets=np.stack(trajectory["joint_targets"]).astype(np.float32),
        camera_to_world=np.stack(trajectory["camera_to_world"]).astype(np.float64),
        stage_names=np.asarray(trajectory["stage_names"], dtype="U16"),
        timestamps=np.asarray(trajectory["timestamps"], dtype=np.float64),
        plan_timestamp=np.asarray(plan_timestamp, dtype=np.float64),
        K=np.asarray(trajectory["K"], dtype=np.float32),
    )
    os.replace(temp_path, path)


# ---------------------------------------------------------------------------
# 运动控制
# ---------------------------------------------------------------------------

def _move_target(
    simulation: DMSimulation,
    goal: np.ndarray,
    duration_s: float,
    stage_name: str = "",
    sample_callback=None,
) -> None:
    """在指定仿真时间内用 smoothstep 曲线插值到目标，避免 position drive 瞬间跳变。

    直接跳到目标位置会导致：
    - 物理引擎产生极大的瞬时速度，物体飞溅
    - 不符合真机运动学约束

    使用 smoothstep (Hermite 三次插值) 保证位置、速度在起点和终点连续。
    """
    # 记录起始目标，用于插值
    start = simulation.target.copy()

    # 从仿真配置读取物理和渲染频率
    physics_hz = int(simulation.config["simulation"]["physics_hz"])
    rendering_hz = int(simulation.config["simulation"]["rendering_hz"])
    render_every = max(1, round(physics_hz / rendering_hz))  # 每 N 步渲染一次

    # 总步数 = 持续时间 × 物理频率
    steps = max(1, round(float(duration_s) * physics_hz))

    for step in range(steps):
        # 检查运行状态，支持中途取消
        if not _running or not simulation.app.is_running():
            return

        # smoothstep: t²(3 - 2t)，保证起点导数为 0（无突变速度），终点导数为 0（无过冲）
        alpha = (step + 1) / steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        # 线性插值位置
        simulation.target[:] = start + alpha * (goal - start)
        simulation._apply_target()  # 将目标写入底层关节控制器

        # 物理步进；只在 render_every 步时触发渲染以节省 GPU
        simulation.world.step(render=(step % render_every == 0))

        # 回调：用于 teacher 轨迹采样等
        if sample_callback is not None:
            sample_callback(stage_name, step, steps)


# ---------------------------------------------------------------------------
# 夹爪状态读取
# ---------------------------------------------------------------------------

def _read_gripper_position_m(simulation: DMSimulation) -> float:
    """读取受控 left_finger 的实际滑块位置（单位：米/每指）。

    joint_indices 最后一个是 left_finger 的关节索引；
    返回滑块的当前实际位置（非目标位置）。
    """
    actual = simulation.robot.get_joint_positions(joint_indices=simulation.joint_indices)
    return float(np.asarray(actual, dtype=np.float64)[-1])


def _read_gripper_effort(simulation: DMSimulation) -> float | None:
    """尽量读取夹爪 effort/force；当前 Isaac API 不支持时返回 None。

    不同 Isaac Sim 版本暴露的 articulation 力反馈接口名字不完全一致，所以这里
    只做保守尝试：能读到就用于成功判断，读不到则只依赖夹爪闭合度。

    尝试的方法名（按保守程度排序）：
    - get_measured_joint_efforts  — 最新版 Isaac Sim
    - get_joint_efforts           — 旧版 API
    - get_measured_joint_forces  — 某些中间版本
    """
    for method_name in ("get_measured_joint_efforts", "get_joint_efforts", "get_measured_joint_forces"):
        method = getattr(simulation.robot, method_name, None)
        if not callable(method):
            continue

        # 尝试带 joint_indices 参数调用，不存在则回退到无参调用
        try:
            values = method(joint_indices=simulation.joint_indices)
        except TypeError:
            try:
                values = method()
            except Exception:
                continue
        except Exception:
            continue

        effort = np.asarray(values, dtype=np.float64)
        if effort.size == 0:
            continue  # 空数组，跳过

        # 提取夹爪关节的 effort（最后一个关节）
        if effort.ndim == 1:
            return float(abs(effort.reshape(-1)[-1]))
        return float(np.linalg.norm(effort.reshape(-1, effort.shape[-1])[-1]))

    return None  # 所有方法都失败，回退到位置残差估算


# ---------------------------------------------------------------------------
# 物体状态读取
# ---------------------------------------------------------------------------

def _read_grasp_object_z(simulation: DMSimulation) -> float:
    """读取香蕉根 Prim 的世界 Z 高度，用于判断 retreat 后是否真的带起物体。

    通过 USD 的 ComputeLocalToWorldTransform 获取物体在世界坐标系中的
    平移分量，提取 Z 轴高度。
    """
    from pxr import Usd, UsdGeom

    prim = simulation.world.stage.GetPrimAtPath("/World/GraspObject")
    if not prim.IsValid():
        raise RuntimeError("抓取物体 Prim 不存在: /World/GraspObject")

    # 计算 Prim 的局部到世界变换矩阵
    world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # 提取平移分量的 Z 值
    return float(world_transform.ExtractTranslation()[2])


# ---------------------------------------------------------------------------
# 抓取计划执行
# ---------------------------------------------------------------------------

def _execute_plan(simulation: DMSimulation, plan: dict, sample_callback=None) -> dict:
    """执行抓取、返回初始姿态及竖直放置；物体移动依赖 PhysX 接触摩擦。

    计划结构：11 个阶段按固定顺序执行
    ┌─────────────────┬──────────────────────────────────────────────┐
    │ 阶段             │ 动作                                         │
    ├─────────────────┼──────────────────────────────────────────────┤
    │ open             │ 张开夹爪                                     │
    │ pregrasp         │ 移动到预夹取姿态（物体上方）                    │
    │ grasp            │ 向下移动到夹取姿态                            │
    │ close            │ 闭合夹爪 → 判断是否夹到物体                    │
    │ retreat          │ 上提 → 判断物体是否跟随（验证抓取成功）          │
    │ return           │ 返回初始 ready 姿态                           │
    │ place_pregrasp   │ 移动到放置点上方                              │
    │ place            │ 下降到放置高度                                │
    │ release           │ 张开夹爪释放物体                             │
    │ place_retreat    │ 上提离开                                     │
    │ finish_return    │ 返回初始姿态，准备下一次检测                    │
    └─────────────────┴──────────────────────────────────────────────┘

    抓取成功判断（三项条件同时满足）：
    1. block：夹爪闭合后被物体挡住，未能到达目标位置（残余 > 阈值）
    2. effort：夹爪电机力矩 > 阈值（或位置残差 × stiffness 估算）
    3. lift：retreat 后物体 Z 高度上升 > 阈值
    """
    # 验证计划阶段的名称和顺序
    expected_names = [
        "open",
        "pregrasp",
        "grasp",
        "close",
        "retreat",
        "return",
        "place_pregrasp",
        "place",
        "release",
        "place_retreat",
        "finish_return",
    ]
    stages = plan.get("stages")
    if not isinstance(stages, list) or [stage.get("name") for stage in stages] != expected_names:
        raise ValueError(f"抓取计划必须按顺序包含 {expected_names}")

    # 读取关节限位和夹爪配置
    arm_limits = simulation.config["arm"]["joints"]
    gripper_config = simulation.config["gripper"]

    # 各阶段持续时间（秒），由经验调优
    durations = {
        "open": 0.8,
        "pregrasp": 2.0,
        "grasp": 1.5,
        "close": 1.0,
        "retreat": 1.5,
        "return": float(simulation.config["sim2real"]["ready_duration_s"]),
        "place_pregrasp": 2.0,
        "place": 1.5,
        "release": 1.0,
        "place_retreat": 1.5,
        "finish_return": float(simulation.config["sim2real"]["ready_duration_s"]),
    }

    # 抓取成功的判断阈值（从配置文件读取）
    min_block_m = float(gripper_config.get("grasp_success_min_block_m", 0.003))
    min_effort = float(gripper_config.get("grasp_success_min_effort", 0.2))
    min_lift_m = float(gripper_config.get("grasp_success_min_lift_m", 0.015))

    # 建立阶段名 → 阶段数据的快速索引
    stage_by_name = {stage["name"]: stage for stage in stages}
    max_grasp_attempts = 3  # 最大重试次数

    # 抓取状态变量（在内部函数中通过 nonlocal 更新）
    grasp_attempts = 0
    grasp_success = False
    block_ok = False
    effort_ok = False
    lift_ok = False
    gripper_actual_after_close = float("nan")
    gripper_block_m = 0.0
    gripper_effort = None
    gripper_effort_source = "unavailable"
    object_z_before_attempt = float("nan")
    object_z_after_retreat = float("nan")
    object_lift_m = 0.0

    # ------------------------------------------------------------------
    # 内部辅助函数
    # ------------------------------------------------------------------

    def run_stage(stage_name: str) -> float:
        """执行单个抓取阶段，返回该阶段的夹爪目标值（米/每指）。

        包含关节限位校验、目标下发、运动执行、实际位置日志。
        """
        stage = stage_by_name[stage_name]
        arm = np.asarray(stage.get("arm"), dtype=np.float64)
        gripper = float(stage.get("gripper_m_per_finger"))

        # 目标有效性校验
        if arm.shape != (6,) or not np.all(np.isfinite(arm)) or not np.isfinite(gripper):
            raise ValueError(f"{stage['name']} 包含无效目标")

        # 关节限位校验
        for name, value in zip((f"joint{i}" for i in range(1, 7)), arm):
            limits = arm_limits[name]
            if not limits["lower"] <= value <= limits["upper"]:
                raise ValueError(f"{stage['name']} 的 {name}={value:.4f} 超出限位")

        # 夹爪限位校验
        if not gripper_config["command_min_m"] <= gripper <= gripper_config["command_max_m"]:
            raise ValueError(f"{stage['name']} 的夹爪目标 {gripper:.4f}m 超出限位")

        # 拼接 6 个关节角 + 1 个夹爪位置 → 7 维目标向量
        goal = np.concatenate([arm, [gripper]])
        print(
            f"[Execute] {stage['name']} arm={np.round(arm, 4).tolist()} "
            f"gripper={gripper:.4f}m/finger"
        )

        # 用 smoothstep 曲线平滑运动到目标
        _move_target(
            simulation,
            goal,
            durations[stage["name"]],
            stage_name=stage["name"],
            sample_callback=sample_callback,
        )

        # 记录运动结束后的实际关节位置
        actual = simulation.robot.get_joint_positions(joint_indices=simulation.joint_indices)
        print(f"[Execute] {stage['name']} actual={np.round(actual, 4).tolist()}")
        return gripper

    def update_grasp_success(close_target_m: float) -> None:
        """根据夹爪闭合残余和 effort 更新 close 阶段的接触判定。

        核心逻辑：
        - close 目标为较小的滑块位置（夹紧）。若夹到物体，实际位置会被物体
          顶住无法到达目标值 → 残余量 = 实际 - 目标。
        - 如果 Isaac 版本支持直接读取 effort，则使用测量值；
          否则用 position drive 的位置残差 × stiffness 估算夹紧力。
        """
        nonlocal block_ok, effort_ok, gripper_actual_after_close, gripper_block_m
        nonlocal gripper_effort, gripper_effort_source

        gripper_actual_after_close = _read_gripper_position_m(simulation)
        measured_effort = _read_gripper_effort(simulation)

        # close 目标为较小滑块位置。若夹到物体，实际位置会被物体顶住，
        # 无法闭到目标值；这个残余闭合量比单看物体高度更接近真机判断。
        gripper_block_m = max(0.0, gripper_actual_after_close - close_target_m)
        block_ok = gripper_block_m >= min_block_m

        if measured_effort is None:
            # Isaac 有些版本读不到 measured effort。仿真里 position drive 的
            # 夹紧力近似等于位置残差乘 stiffness；这相当于真机 grasp_driver.py
            # 里的"被物体挡住后不再继续闭合/失速"的替代量。
            gripper_effort = gripper_block_m * float(gripper_config["stiffness"])
            gripper_effort_source = "estimated_from_position_drive"
        else:
            gripper_effort = measured_effort
            gripper_effort_source = "measured"

        effort_ok = gripper_effort >= min_effort

        print(
            f"[Execute] close_check block_ok={block_ok} effort_ok={effort_ok} "
            f"attempt={grasp_attempts}/{max_grasp_attempts} "
            f"gripper_block={gripper_block_m:.4f}m threshold={min_block_m:.4f}m "
            f"effort={gripper_effort:.4f} source={gripper_effort_source} "
            f"effort_threshold={min_effort:.3f}"
        )

    def update_lift_success() -> None:
        """retreat 后确认物体跟着夹爪移动；三项条件同时满足才算成功。

        测量 retreat 前后物体的 Z 坐标差；必须 > min_lift_m 才判定为
        真正提起了物体。防止"夹取瞬间滑落"被误判为成功。
        """
        nonlocal grasp_success, lift_ok, object_z_after_retreat, object_lift_m

        object_z_after_retreat = _read_grasp_object_z(simulation)
        object_lift_m = object_z_after_retreat - object_z_before_attempt
        lift_ok = object_lift_m >= min_lift_m
        grasp_success = block_ok and effort_ok and lift_ok

        print(
            f"[Execute] lift_check lift_ok={lift_ok} "
            f"object_z_before={object_z_before_attempt:.4f}m "
            f"object_z_after={object_z_after_retreat:.4f}m "
            f"lift={object_lift_m:.4f}m threshold={min_lift_m:.4f}m "
            f"grasp_success={grasp_success}"
        )

    # ------------------------------------------------------------------
    # 执行抓取循环
    # ------------------------------------------------------------------

    # 第一步：张开夹爪（为抓取做准备）
    run_stage("open")

    # 最多尝试 max_grasp_attempts 次抓取
    for attempt in range(1, max_grasp_attempts + 1):
        grasp_attempts = attempt
        print(f"[Execute] grasp attempt {grasp_attempts}/{max_grasp_attempts}")

        # 接近 → 夹取 → 闭合
        run_stage("pregrasp")
        run_stage("grasp")

        # 记录抓取前物体高度，用于后续 lift 判定
        object_z_before_attempt = _read_grasp_object_z(simulation)

        close_target_m = run_stage("close")
        update_grasp_success(close_target_m)

        # gripper_block 只是 close 瞬间是否被挡住。必须 retreat 后再看物体
        # 是否真的跟着动，否则"夹取瞬间滑落、香蕉没动"会被误判成功。
        run_stage("retreat")
        update_lift_success()

        if grasp_success:
            break  # 成功，退出重试循环

        if attempt < max_grasp_attempts:
            # 失败重试：先回到预夹取点并重新张开，再从同一目标再次前进夹取。
            # 这样不会带着闭合夹爪直接冲向下一次 grasp。
            print("[Execute] grasp failed; reopen at pregrasp and retry")
            run_stage("pregrasp")

    # ------------------------------------------------------------------
    # 放置阶段（仅抓取成功时执行）
    # ------------------------------------------------------------------
    if grasp_success:
        for stage_name in (
            "return",
            "place_pregrasp",
            "place",
            "release",
            "place_retreat",
            "finish_return",
        ):
            run_stage(stage_name)
    else:
        # 三次仍失败：不再去放置点；此时已在 retreat/pregrasp 位，直接闭爪回初始姿态。
        # finish_return 本身就是闭爪 ready 姿态，执行完后感知端会等待下一次检测。
        print("[Execute] grasp failed after retries; return finish_return with closed gripper")
        run_stage("finish_return")

    print("[Execute] cycle complete; ready for the next detection")

    # 返回详细的执行结果，供感知端记录和分析
    return {
        "grasp_success": grasp_success,
        "grasp_block_ok": block_ok,
        "grasp_effort_ok": effort_ok,
        "grasp_lift_ok": lift_ok,
        "grasp_attempts": grasp_attempts,
        "grasp_retry_limit": max_grasp_attempts,
        "grasp_object_z_before_attempt_m": object_z_before_attempt,
        "grasp_object_z_after_retreat_m": object_z_after_retreat,
        "grasp_object_lift_m": object_lift_m,
        "grasp_object_lift_threshold_m": min_lift_m,
        "grasp_gripper_block_m": gripper_block_m,
        "grasp_gripper_block_threshold_m": min_block_m,
        "grasp_gripper_effort": gripper_effort,
        "grasp_gripper_effort_source": gripper_effort_source,
        "grasp_gripper_effort_threshold": min_effort,
        "grasp_gripper_actual_after_close_m": gripper_actual_after_close,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    """主函数：初始化仿真 → 导出 RGB-D → 循环执行抓取计划。

    命令行参数：
    --output         RGB-D 输出路径（默认 /tmp/rebot_sim_rgbd.npz）
    --width / --height  导出分辨率（默认 640×360）
    --export-hz      导出频率（默认 5 Hz）
    --settle-seconds 观测姿态稳定时间（默认读取 dm_sim.yaml）
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--export-hz", type=float, default=5.0)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=None,
        help="开始导出前保持 ready_pose 的时间；默认读取 dm_sim.yaml",
    )

    # Isaac Sim 会在命令行附加 ``--/rtx/...`` 参数，因此不能使用 parse_args()。
    # parse_known_args() 会忽略未知参数，只提取我们定义的参数。
    args, _isaac_args = parser.parse_known_args()

    # 参数有效性校验
    if args.width <= 0 or args.height <= 0 or args.export_hz <= 0:
        raise ValueError("width, height and export-hz must be positive")
    if args.settle_seconds is not None and args.settle_seconds < 0:
        raise ValueError("settle-seconds cannot be negative")

    # 注册信号处理，支持 Ctrl+C 优雅退出
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # 确保输出目录存在
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 初始化仿真
    # ------------------------------------------------------------------

    # 复用现有场景创建代码，但关闭关节 UDP，避免两个控制入口同时改目标。
    simulation = DMSimulation(listen_udp=False)
    try:
        simulation.setup()  # 加载 USD 场景、创建 PhysX 世界、初始化机械臂

        # Isaac Sim 的 Camera API 和旋转变换工具
        from isaacsim.sensors.camera import Camera
        from isaacsim.core.utils.rotations import quat_to_rot_matrix

        # 绑定已有 Camera Prim（在 rebotarm_dm_with_camera.usd 中定义），
        # 而不是创建一台与机械臂无关的新相机。
        camera = Camera(
            prim_path=CAMERA_PATH,
            resolution=(args.width, args.height),
            frequency=args.export_hz,
        )
        camera.initialize()

        # 添加 distance_to_image_plane 注解：这是与彩色图像像素对齐的 Z 深度，
        # 适合针孔反投影。区别于 distance_to_camera（到相机光心的欧氏距离）。
        camera.add_distance_to_image_plane_to_frame()

        # 获取相机内参矩阵（3×3）
        K = camera.get_intrinsics_matrix()

        # ------------------------------------------------------------------
        # 移动到观测姿态
        # ------------------------------------------------------------------

        # 普通 initial_state 只保证安全加载，不保证腕部相机能看到桌面。
        # RGB-D 导出使用独立观测姿态，避免为了感知修改接收端和其他仿真入口的初始状态。
        sim2real_config = simulation.config.get("sim2real", {})
        ready_arm = np.asarray(sim2real_config["ready_arm"], dtype=np.float64)
        if ready_arm.shape != (6,) or not np.all(np.isfinite(ready_arm)):
            raise ValueError("sim2real.ready_arm 必须是 6 个有限关节角")

        # 构建观测目标：[6 个关节角, 夹爪位置]
        ready_target = simulation.target.copy()
        ready_target[:6] = ready_arm
        ready_target[6] = float(sim2real_config["camera_view_gripper_m"])

        # 稳定时间：命令行参数优先，否则从配置文件读取
        settle_seconds = (
            float(args.settle_seconds)
            if args.settle_seconds is not None
            else float(sim2real_config["ready_duration_s"])
        )
        if settle_seconds < 0:
            raise ValueError("sim2real.ready_duration_s 不能为负数")

        # setup() 在物理启动前只下发过一次目标。这里在实际物理步中持续保持观测
        # 姿态，确保机械臂运动到位后才开始导出桌面图像。
        _move_target(simulation, ready_target, settle_seconds)
        actual = simulation.robot.get_joint_positions(joint_indices=simulation.joint_indices)
        print(f"[RGBD] camera-view target={np.round(simulation.target, 4).tolist()}")
        print(f"[RGBD] settled actual={np.round(actual, 4).tolist()}")
        print(f"[RGBD] camera={CAMERA_PATH}")
        print(f"[RGBD] output={args.output} resolution={args.width}x{args.height}")
        print(f"[RGBD] K=\n{K}")

        # ------------------------------------------------------------------
        # 主循环：导出帧 + 执行抓取计划
        # ------------------------------------------------------------------

        period = 1.0 / args.export_hz            # 导出间隔（秒）
        next_export = time.perf_counter()         # 下次导出时间
        session_started = time.time()             # 会话开始时间（用于过滤旧计划）
        last_plan_timestamp = -1.0                # 上一个已执行计划的去重时间戳

        while _running and simulation.app.is_running():
            # 始终步进物理世界（即使不渲染，也需要更新关节位置）
            simulation.world.step(render=True)

            now = time.perf_counter()
            if now < next_export:
                continue  # 未到导出时间，跳过

            # --- 导出 RGB-D 帧 ---
            color_rgb = camera.get_rgb()
            depth_m = camera.get_depth()
            if color_rgb is not None and depth_m is not None:
                # Isaac 返回 RGB，OpenCV/现有抓取代码使用 BGR → 通道反转
                color_bgr = np.ascontiguousarray(color_rgb[..., ::-1])
                depth_m = np.asarray(depth_m, dtype=np.float32)

                # 天空、裁剪面外区域可能返回 inf/nan；下游约定 0 表示无效深度
                depth_m[~np.isfinite(depth_m)] = 0.0

                # 获取相机在世界坐标系中的位姿
                # ros 轴在 Isaac Camera API 中表示计算机视觉相机坐标系（Z 前、Y 下、
                # X 右）；返回的四元数为 (w, x, y, z)，可直接构造相机到 /World 的刚体变换。
                camera_position, camera_quaternion = camera.get_world_pose(camera_axes="ros")
                T_camera_to_world = np.eye(4, dtype=np.float64)
                T_camera_to_world[:3, :3] = quat_to_rot_matrix(camera_quaternion)          # 旋转部分
                T_camera_to_world[:3, 3] = np.asarray(camera_position, dtype=np.float64)   # 平移部分

                # 原子写入帧文件
                _write_frame(args.output, color_bgr, depth_m, K, T_camera_to_world)

            # --- 检查并执行抓取计划 ---
            # 每个新计划只执行一次；完成后写回确认，感知端才会生成下一轮计划。
            if DEFAULT_PLAN.is_file():
                plan = json.loads(DEFAULT_PLAN.read_text(encoding="utf-8"))
                plan_timestamp = float(plan.get("timestamp", -1.0))

                # 三个条件过滤：来源是 sim、时间戳在本次会话之后、时间戳比上次执行的更新
                if (
                    plan.get("source") == "sim"
                    and plan_timestamp >= session_started
                    and plan_timestamp > last_plan_timestamp
                ):
                    last_plan_timestamp = plan_timestamp

                    # 执行抓取放置循环
                    execution_result = _execute_plan(simulation, plan)

                    # 将执行结果写回计划文件，感知端读取后生成下一轮计划
                    plan.update(execution_result)
                    plan["executed"] = True
                    plan["executed_at"] = time.time()

                    # 原子写入完成标记（先写临时文件再替换）
                    completed_path = DEFAULT_PLAN.with_suffix(".executed.tmp")
                    completed_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
                    completed_path.replace(DEFAULT_PLAN)

            # 更新下次导出时间
            next_export = now + period

    except BaseException:
        # SimulationApp.close() 的 fast shutdown 可能在 Python 打印 traceback 前
        # 结束进程，因此必须先输出并刷新真实异常。
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        # 确保无论如何都正确关闭仿真
        simulation.close()


# ---------------------------------------------------------------------------
# 脚本入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
