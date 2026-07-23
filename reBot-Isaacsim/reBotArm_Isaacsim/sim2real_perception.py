#!/usr/bin/env python3
"""面向 Isaac Sim/真机 RGB-D 的共用 YOLO + SAM 传统抓取感知。

数据流：RGB-D -> YOLO 检测 -> SAM 掩膜 -> 深度反投影 -> 抓取候选 -> IK 计划。

Sim 模式从 Isaac 导出的 NPZ 文件读取图像和相机外参，在候选连续稳定后，把
相机坐标转换到 ``/World`` 并生成完整的抓取/放置关节计划。Isaac 进程通过
JSON 文件读取并执行计划，执行完成后在同一文件写回确认，从而开始下一轮识别。

Real 模式复用相同的感知流程，但当前只输出相机坐标系抓取结果，不连接或控制
真实机械臂；真机执行前仍需可靠的手眼标定、实时 FK 和硬件安全检查。
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import yaml


# 当前脚本位于 reBot-Isaacsim/reBotArm_Isaacsim：parents[1] 才是整个 rebot 仓库。
# 将 rebot_grasp 加入模块搜索路径，是为了直接复用同一套相机、YOLO、SAM 和
# 普通几何抓取实现，确保 Sim/Real 不会各自维护一份逐渐分叉的感知算法。
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
GRASP_ROOT = REPO_ROOT / "rebot_grasp"
sys.path.insert(0, str(GRASP_ROOT))

from drivers.camera import make_camera  # noqa: E402
from utils.ordinary_grasp import draw_grasp, estimate_grasps, select_best_grasp  # noqa: E402
from utils.sam_utils import draw_sam_masks_overlay, load_sam_refiner  # noqa: E402
from utils.yolo_utils import load_yolo  # noqa: E402


# 两个 Python 环境通过 /tmp 文件通信：Isaac 必须运行在其官方 Python 中，
# YOLO/SAM 则运行在 rebotarm_gpu 中，因此不能直接在进程内互相 import。
DEFAULT_FRAME = Path("/tmp/rebot_sim_rgbd.npz")          # Isaac 输出的最新 RGB-D 帧
DEFAULT_RESULT = Path("/tmp/rebot_grasp_candidate.json") # 便于查看/调试的感知结果
DEFAULT_PLAN = Path("/tmp/rebot_sim_grasp_plan.json")     # Isaac 实际读取的关节计划

# 连续三帧用于抑制单帧检测和深度噪声；三维位置最大偏差超过 1 cm 时不下发计划。
STABLE_FRAMES = 3
MAX_POSITION_SPREAD_M = 0.010
RESULT_LOG_INTERVAL_S = 1.0


def _handle_sigterm(_signum, _frame) -> None:
    """让启动脚本的 SIGTERM 走正常 finally，确保相机和窗口被释放。"""
    raise KeyboardInterrupt


def _load_sim_frame(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """读取 Isaac 导出的完整 RGB-D 帧，并复制数组后立即关闭 NPZ。

    ``color_bgr`` 和 ``depth_mm`` 与 rebot_grasp 的相机接口保持一致；``K`` 是
    3x3 相机内参；``T_camera_to_world`` 把相机坐标点转换到 Isaac ``/World``。
    使用 ``copy()`` 后才能安全退出 ``with`` 并关闭 NPZ 底层文件句柄。
    """
    with np.load(path, allow_pickle=False) as data:
        return (
            data["color_bgr"].copy(),
            data["depth_mm"].copy(),
            data["K"].copy(),
            data["T_camera_to_world"].copy(),
            float(data["timestamp"]),
        )


def _median_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """融合多帧姿态，并把结果投影回合法的 SO(3) 旋转矩阵。

    逐元素中位数能够降低单帧姿态跳变，但结果通常不再严格正交。SVD 的
    ``U @ Vt`` 会求出离该矩阵最近的正交矩阵；若行列式为 -1，则它是镜像而
    不是旋转，需要翻转一列使最终行列式回到 +1。
    """
    matrix = np.median(np.stack(rotations), axis=0)
    U, _singular_values, Vt = np.linalg.svd(matrix)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0.0:
        U[:, -1] *= -1.0
        rotation = U @ Vt
    return rotation


def _save_candidate(
    path: Path,
    best,
    source: str,
    timestamp: float,
    T_camera_to_world: np.ndarray | None,
    position_spread_m: float,
) -> np.ndarray | None:
    """以统一 JSON 格式原子发布最佳抓取候选。

    顶层字段继续保留相机坐标。仿真帧额外写入 ``sim_world``；真机仍需完成
    手眼标定和实时 FK，不能直接把相机坐标作为机器人运动目标。
    """
    # 顶层数据始终使用相机坐标，保证 sim 和 real 的输出结构一致。
    payload = {
        "source": source,
        "timestamp": timestamp,
        "frame": "camera",
        "class_name": best.class_name,
        "confidence": float(best.conf),
        "position_m": best.position.tolist(),
        "rotation": best.rotation.tolist(),
        "tcp_rotation": best.tcp_rotation.tolist() if best.tcp_rotation is not None else None,
        "jaw_width_m": float(best.jaw_width_m),
        "stable_frames": STABLE_FRAMES,
        "position_spread_m": float(position_spread_m),
    }
    world_position = None
    if T_camera_to_world is not None:
        T_camera_to_world = np.asarray(T_camera_to_world, dtype=np.float64)
        if T_camera_to_world.shape != (4, 4) or not np.all(np.isfinite(T_camera_to_world)):
            raise ValueError("T_camera_to_world 必须是有限的 4x4 矩阵")
        # 刚体变换 p_world = R_camera_to_world @ p_camera + t_camera_to_world。
        # 姿态矩阵没有平移分量，因此只需左乘同一个旋转矩阵。
        world_position = (
            T_camera_to_world[:3, :3] @ np.asarray(best.position, dtype=np.float64)
            + T_camera_to_world[:3, 3]
        )
        world_rotation = T_camera_to_world[:3, :3] @ np.asarray(best.rotation, dtype=np.float64)
        world_tcp_rotation = (
            T_camera_to_world[:3, :3] @ np.asarray(best.tcp_rotation, dtype=np.float64)
            if best.tcp_rotation is not None
            else None
        )
        payload["sim_world"] = {
            "frame": "/World",
            "position_m": world_position.tolist(),
            "rotation": world_rotation.tolist(),
            "tcp_rotation": world_tcp_rotation.tolist() if world_tcp_rotation is not None else None,
            "T_camera_to_world": T_camera_to_world.tolist(),
        }
    # 先写临时文件再原子替换，避免另一个进程读到只写了一半的 JSON。
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)
    return world_position


def _save_sim_plan(
    path: Path,
    best,
    timestamp: float,
    T_camera_to_world: np.ndarray,
    grasp_cfg: dict,
) -> None:
    """计算抓取和竖直放置 IK，并把完整关节计划写给 Isaac 进程。

    本函数只做几何计算与文件发布，不调用 Isaac API，也不发送真实机械臂命令。
    每一个 ``arm`` 都是六轴目标关节角，``gripper_m_per_finger`` 是单侧滑块位移。
    """
    sdk_root = GRASP_ROOT / "sdk" / "reBotArm_control_py"
    if str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))
    from reBotArm_control_py.kinematics import compute_ik

    dm_cfg = yaml.safe_load((REPO_ROOT / "reBot-Isaacsim/config/dm_sim.yaml").read_text(encoding="utf-8"))
    if best.tcp_rotation is None:
        raise ValueError("抓取候选缺少 tcp_rotation，不能计算 IK")
    # best.position 和 best.tcp_rotation 均来自相机坐标系。将二者转换到 /World
    # 后，才能把目标交给以机器人基座/世界坐标建模的 reBot IK。
    T_camera_to_world = np.asarray(T_camera_to_world, dtype=np.float64)
    tcp_rotation_world = T_camera_to_world[:3, :3] @ np.asarray(best.tcp_rotation, dtype=np.float64)
    object_position_world = (
        T_camera_to_world[:3, :3] @ np.asarray(best.position, dtype=np.float64)
        + T_camera_to_world[:3, 3]
    )

    pregrasp_offset = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
    # reBot TCP 坐标约定：+X 是夹爪朝物体前进的方向。因此：
    #   grasp    = 视觉物体点沿 +X 深入 1.5 cm，避免只碰到物体表面；
    #   pregrasp = grasp 沿 -X 后退 8 cm，先安全到达物体前方再直线接近。
    # 1.5 cm 的实际数值来自 default.yaml 的 insertion_depth_m，不在源码重复配置。
    insertion_depth = float(grasp_cfg["insertion_depth_m"])
    tool_x = tcp_rotation_world[:, 0]
    grasp_position = object_position_world + tool_x * insertion_depth
    pregrasp_position = grasp_position - tool_x * pregrasp_offset

    # 以上一阶段作为下一次 IK 的初值，能让迭代更容易收敛到连续、相近的关节解，
    # 避免同一个末端位姿突然选择机械臂的另一条构型分支。
    ready_q = np.asarray(dm_cfg["sim2real"]["ready_arm"], dtype=np.float64)
    pregrasp_ik = compute_ik(ready_q, pregrasp_position, tcp_rotation_world)
    if not pregrasp_ik.success:
        raise RuntimeError(f"pregrasp IK 失败，error={pregrasp_ik.error:.6f}")
    pregrasp_q = np.asarray(pregrasp_ik.q[:6], dtype=np.float64)

    grasp_ik = compute_ik(pregrasp_q, grasp_position, tcp_rotation_world)
    if not grasp_ik.success:
        raise RuntimeError(f"grasp IK 失败，error={grasp_ik.error:.6f}")
    grasp_q = np.asarray(grasp_ik.q[:6], dtype=np.float64)

    # 放置时重新构造一个正交 TCP 坐标系：
    #   X = [0, 0, -1]：夹爪前进方向竖直指向桌面；
    #   Y = 世界 +Y：放置 yaw 固定，避免继承抓取 yaw 后让腕部落入不可达分支；
    #   Z = X × Y：补齐右手坐标系。
    # 注意：固定的是放置点还不够，姿态也要固定；否则同一个位置会因为 yaw 不同导致 IK 失败。
    vertical_tool_x = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    vertical_tool_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    vertical_tool_z = np.cross(vertical_tool_x, vertical_tool_y)
    vertical_tcp_rotation = np.column_stack(
        [vertical_tool_x, vertical_tool_y, vertical_tool_z]
    )
    # 放置点回到本轮夹取位置的正上方 3 cm。这里的 object_position_world 是视觉
    # 检测到的香蕉中心；放置时只抬高 Z，不改 X/Y，便于观察“抓起 -> 原位上方放下”。
    place_height_offset_m = 0.03
    place_object_position = object_position_world + np.array([0.0, 0.0, place_height_offset_m])
    # reBot TCP +X 为夹爪前进方向。竖直放置时 +X 指向桌面，所以实际 TCP 目标点
    # 需要从期望物体中心沿 +X 深入 insertion_depth。
    place_position = place_object_position + vertical_tool_x * insertion_depth
    place_pregrasp_position = place_position - vertical_tool_x * pregrasp_offset

    place_pregrasp_ik = compute_ik(ready_q, place_pregrasp_position, vertical_tcp_rotation)
    if not place_pregrasp_ik.success:
        raise RuntimeError(f"place_pregrasp IK 失败，error={place_pregrasp_ik.error:.6f}")
    place_pregrasp_q = np.asarray(place_pregrasp_ik.q[:6], dtype=np.float64)

    place_ik = compute_ik(place_pregrasp_q, place_position, vertical_tcp_rotation)
    if not place_ik.success:
        raise RuntimeError(f"place IK 失败，error={place_ik.error:.6f}")
    place_q = np.asarray(place_ik.q[:6], dtype=np.float64)

    gripper_cfg = dm_cfg["gripper"]
    # 感知输出 jaw_width_m 是左右指尖之间的总开口，而 Isaac 控制的是单侧手指
    # 滑动距离，因此先除以 2；再增加每侧 3 mm 余量，避免接近时提前碰撞物体。
    open_per_finger = np.clip(
        float(best.jaw_width_m) / 2.0 + 0.003,
        gripper_cfg["command_min_m"],
        gripper_cfg["command_max_m"],
    )
    # 闭合时给最小位置目标。碰到香蕉后手指不会穿透，Position Drive 的剩余
    # 位置误差会转化为法向夹紧力，物体依靠 PhysX 接触摩擦被抬起。
    close_per_finger = float(gripper_cfg["command_min_m"])

    # 阶段中的 gripper 值始终保持到下一阶段：close/retreat/return/place 使用闭合
    # 目标，确保搬运途中不松手；release 之后保持张开并先撤离物体，避免刚放下
    # 就闭爪碰撞香蕉。最后 finish_return 回到相机可观察桌面的 ready_arm 时重新
    # 闭合夹爪，让每轮循环结束状态与初始闭合状态一致。
    payload = {
        "timestamp": timestamp,
        "source": "sim",
        "frame": "/World",
        "object_position_m": object_position_world.tolist(),
        "grasp_position_m": grasp_position.tolist(),
        "pregrasp_position_m": pregrasp_position.tolist(),
        "insertion_depth_m_along_tcp_x": insertion_depth,
        "tcp_rotation": tcp_rotation_world.tolist(),
        "place_position_m": place_position.tolist(),
        "place_pregrasp_position_m": place_pregrasp_position.tolist(),
        "place_object_position_m": place_object_position.tolist(),
        "place_height_offset_m": place_height_offset_m,
        "place_tcp_rotation": vertical_tcp_rotation.tolist(),
        "jaw_width_m": float(best.jaw_width_m),
        # Isaac 完成所有 stages 后会把该字段原子改成 true；感知端收到确认前
        # 不会覆盖当前计划，从而避免机械臂运动期间重复规划。
        "executed": False,
        "stages": [
            {"name": "open", "arm": ready_q.tolist(), "gripper_m_per_finger": float(open_per_finger)},
            {"name": "pregrasp", "arm": pregrasp_q.tolist(), "gripper_m_per_finger": float(open_per_finger)},
            {"name": "grasp", "arm": grasp_q.tolist(), "gripper_m_per_finger": float(open_per_finger)},
            {"name": "close", "arm": grasp_q.tolist(), "gripper_m_per_finger": close_per_finger},
            {"name": "retreat", "arm": pregrasp_q.tolist(), "gripper_m_per_finger": close_per_finger},
            {"name": "return", "arm": ready_q.tolist(), "gripper_m_per_finger": close_per_finger},
            {
                "name": "place_pregrasp",
                "arm": place_pregrasp_q.tolist(),
                "gripper_m_per_finger": close_per_finger,
            },
            {"name": "place", "arm": place_q.tolist(), "gripper_m_per_finger": close_per_finger},
            {"name": "release", "arm": place_q.tolist(), "gripper_m_per_finger": float(open_per_finger)},
            {
                "name": "place_retreat",
                "arm": place_pregrasp_q.tolist(),
                "gripper_m_per_finger": float(open_per_finger),
            },
            {
                "name": "finish_return",
                "arm": ready_q.tolist(),
                "gripper_m_per_finger": close_per_finger,
            },
        ],
        "ik": {
            "pregrasp_error": float(pregrasp_ik.error),
            "grasp_error": float(grasp_ik.error),
            "place_pregrasp_error": float(place_pregrasp_ik.error),
            "place_error": float(place_ik.error),
        },
    }
    # 发布计划同样采用原子替换，保证 Isaac 只会看到完整的 stages 列表。
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def main() -> None:
    # source 只决定帧从哪里获取以及是否生成 Isaac 计划；从 YOLO 开始的检测、
    # 深度几何与候选选择代码由 Sim/Real 共用，这是 sim2real 一致性的基础。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("sim", "real"), required=True)
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--config", type=Path, default=GRASP_ROOT / "config/default.yaml")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _handle_sigterm)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # 本入口明确验证 YOLO + SAM 组合，因此即使 default.yaml 为普通脚本关闭了
    # SAM，这里也强制启用。模型路径、CUDA 设备和置信度仍全部由 YAML 提供。
    cfg.setdefault("sam", {})["enabled"] = True
    model, yolo_opts = load_yolo(cfg, project_root=GRASP_ROOT)
    sam = load_sam_refiner(cfg, project_root=GRASP_ROOT)
    depth_quantile = float(cfg.get("grasp_pipeline", {}).get("grasp", {}).get("depth_quantile", 0.5))

    # Real 模式直接复用 rebot_grasp 相机驱动；Sim 模式只读 Isaac 的 NPZ 文件，
    # 不枚举或占用任何真实 USB RGB-D 相机。
    camera = None
    if args.source == "real":
        camera = make_camera(cfg)
        camera.open()
        camera.warm_up(10)

    last_timestamp = -1.0       # 防止同一 NPZ 帧被反复送入 GPU 推理
    last_stale_warning = 0.0    # 限制“帧过期”日志频率，避免终端刷屏
    grasp_history = deque(maxlen=STABLE_FRAMES)  # 固定长度的多帧稳定窗口
    plan_written = False        # true 表示已有计划正在等待 Isaac 执行完成
    pending_plan_timestamp = -1.0  # 用时间戳把完成确认与本进程发布的计划对应起来
    last_result_log_time = 0.0     # 感知仍逐帧运行，但终端结果最多每秒打印一次
    last_result_status = ""
    window_name = f"Sim2Real grasp ({args.source})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("[Safety] perception only; no robot command is sent")
    try:
        while True:
            # 仿真完成整轮动作后会在同一计划文件中写回 executed=true。必须同时
            # 匹配 timestamp，防止误把上一次运行残留的 true 当成本轮完成确认。
            # 确认后清空历史，让下一轮必须重新积累三帧，而不是沿用移动前的候选。
            if args.source == "sim" and plan_written and DEFAULT_PLAN.is_file():
                try:
                    plan_state = json.loads(DEFAULT_PLAN.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    plan_state = {}
                if (
                    float(plan_state.get("timestamp", -1.0)) == pending_plan_timestamp
                    and plan_state.get("executed") is True
                ):
                    plan_written = False
                    pending_plan_timestamp = -1.0
                    grasp_history.clear()
                    print("\n[Cycle] complete; detecting the next grasp")
                else:
                    now = time.monotonic()
                    if now - last_stale_warning > 2.0:
                        print("[Wait] IsaacSim is executing the current grasp plan")
                        last_stale_warning = now
                    time.sleep(0.1)
                    continue

            if args.source == "sim":
                if not args.frame.is_file():
                    print(f"[Wait] {args.frame}", end="\r", flush=True)
                    time.sleep(0.1)
                    continue
                try:
                    color_bgr, depth_mm, K, T_camera_to_world, timestamp = _load_sim_frame(args.frame)
                except (KeyError, OSError, ValueError):
                    time.sleep(0.02)
                    continue
                # 旧 NPZ 可能来自上一次运行；机械臂执行动作期间导出器也会暂停
                # 刷新图像。这两种情况下帧都不能代表当前场景，所以只等待新帧，
                # 绝不能拿过期坐标继续生成抓取计划。
                frame_age = time.time() - timestamp
                if frame_age > 2.0:
                    now = time.monotonic()
                    if now - last_stale_warning > 2.0:
                        print(
                            f"[Wait] sim RGB-D is stale ({frame_age:.1f}s); "
                            "check that ./run_sim_rgbd.sh is still running"
                        )
                        last_stale_warning = now
                    time.sleep(0.1)
                    continue
                # 主循环远快于 Isaac 的导出频率。timestamp 没变就说明仍是同一帧，
                # 重复推理不会增加信息，只会无意义占用 GPU 并伪造“连续三帧”。
                if timestamp <= last_timestamp:
                    time.sleep(0.01)
                    continue
                last_timestamp = timestamp
            else:
                color_bgr, depth_mm = camera.get_frame()
                if color_bgr is None or depth_mm is None:
                    continue
                K = camera.K
                T_camera_to_world = None
                timestamp = time.time()

            # Sim/Real 从这里开始走完全相同的感知和几何抓取路径：YOLO 给出目标，
            # SAM 把粗检测细化为物体掩膜，estimate_grasps 再结合深度和内参 K
            # 反投影三维位置、估计夹爪朝向与开口宽度。
            results = model.predict(
                color_bgr,
                verbose=False,
                device=yolo_opts["device"],
                conf=yolo_opts["conf"],
                iou=yolo_opts["iou"],
            )
            sam_masks = sam.refine_results(results, color_bgr) if sam is not None else None
            grasps = estimate_grasps(
                results,
                depth_mm,
                K,
                depth_quantile=depth_quantile,
                mask_overrides=sam_masks,
            )

            preview = color_bgr.copy()
            draw_sam_masks_overlay(preview, sam_masks)
            for grasp in grasps:
                draw_grasp(preview, grasp, show_pose_text=True)
            # 当前策略沿用 rebot_grasp：只考虑有有效深度的候选，再选择 YOLO
            # 置信度最高者。若类别突然改变，旧历史属于另一物体，必须立即清空。
            best = select_best_grasp(grasps)
            if best is not None:
                if grasp_history and best.class_name != grasp_history[-1].class_name:
                    grasp_history.clear()
                grasp_history.append(best)
                if len(grasp_history) < STABLE_FRAMES:
                    print(
                        f"[Stabilizing] {best.class_name} "
                        f"{len(grasp_history)}/{STABLE_FRAMES}",
                        end="\r",
                        flush=True,
                    )
                else:
                    # 位置稳定性取三帧位置的中位数，再计算每帧到中位点的最大距离。
                    # 最大距离超过 1 cm 时只继续观察，不会生成可能撞击物体的计划。
                    positions = np.stack([grasp.position for grasp in grasp_history])
                    stable_position = np.median(positions, axis=0)
                    position_spread = float(np.max(np.linalg.norm(positions - stable_position, axis=1)))
                    if position_spread > MAX_POSITION_SPREAD_M:
                        print(
                            f"[Stabilizing] position spread={position_spread:.4f}m "
                            f"> {MAX_POSITION_SPREAD_M:.4f}m",
                            end="\r",
                            flush=True,
                        )
                    else:
                        # 候选稳定后，把位置、置信度、姿态和夹爪宽度分别做多帧融合。
                        # replace() 保留 GraspPose 其余字段，避免手工重建时遗漏数据。
                        tcp_rotations = [grasp.tcp_rotation for grasp in grasp_history]
                        stable_best = replace(
                            best,
                            conf=float(np.median([grasp.conf for grasp in grasp_history])),
                            position=stable_position,
                            rotation=_median_rotation([grasp.rotation for grasp in grasp_history]),
                            tcp_rotation=(
                                _median_rotation(tcp_rotations)
                                if all(rotation is not None for rotation in tcp_rotations)
                                else None
                            ),
                            jaw_width_m=float(np.median([grasp.jaw_width_m for grasp in grasp_history])),
                        )
                        world_position = _save_candidate(
                            args.result,
                            stable_best,
                            args.source,
                            timestamp,
                            T_camera_to_world,
                            position_spread,
                        )
                        # 一轮只允许存在一个未完成计划。Isaac 写回 executed=true 后，
                        # 顶部握手逻辑才会把 plan_written 复位并允许下一轮规划。
                        if args.source == "sim" and not plan_written:
                            try:
                                _save_sim_plan(
                                    DEFAULT_PLAN,
                                    stable_best,
                                    timestamp,
                                    T_camera_to_world,
                                    cfg.get("grasp_pipeline", {}).get("grasp", {}),
                                )
                            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                                print(f"\n[Plan] failed: {exc}")
                            else:
                                print(f"\n[Plan] saved: {DEFAULT_PLAN}")
                                plan_written = True
                                pending_plan_timestamp = timestamp
                        world_text = (
                            f" world_xyz={np.round(world_position, 4).tolist()}"
                            if world_position is not None
                            else ""
                        )
                        now_log = time.monotonic()
                        if now_log - last_result_log_time >= RESULT_LOG_INTERVAL_S:
                            print(
                                f"[Grasp] stable={STABLE_FRAMES} spread={position_spread:.4f}m "
                                f"{stable_best.class_name} conf={stable_best.conf:.2f} "
                                f"camera_xyz={np.round(stable_best.position, 4).tolist()} "
                                f"jaw={stable_best.jaw_width_m:.4f}m{world_text}"
                            )
                            last_result_log_time = now_log
                            last_result_status = "grasp"
            elif grasps:
                # YOLO/SAM 找到了区域，但该区域没有可靠深度或几何条件不满足；
                # 清空稳定窗口，防止下一次有效结果与这些无效帧拼成“三帧稳定”。
                grasp_history.clear()
                reasons = sorted({grasp.rejected_reason or "unknown" for grasp in grasps})
                status = f"invalid:{reasons}"
                if status != last_result_status:
                    print(f"[Perception] {len(grasps)} detection(s), no valid depth grasp: {reasons}")
                    last_result_status = status
            else:
                # 完全没有检测时同样清空历史。已有待执行计划不会因此取消，只有
                # Isaac 的 executed 确认能改变 plan_written，避免动作中途被覆盖。
                grasp_history.clear()
                if last_result_status != "no_detection":
                    print("[Perception] no YOLO detection")
                    last_result_status = "no_detection"

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("\n[Done] interrupted by user")
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
