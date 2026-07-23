#!/usr/bin/env python3
"""从已经跑通的 reBot-Isaacsim 抓取器收集 IsaacLab BC teacher 数据。

本脚本不启动 Isaac Sim，也不修改 ``reBot-Isaacsim`` 代码。它只读取那边现有
流程已经发布到 ``/tmp`` 的两个文件：

- ``/tmp/rebot_sim_rgbd.npz``：IsaacSim 腕部相机 RGB-D 帧；
- ``/tmp/rebot_sim_grasp_plan.json``：YOLO+SAM+传统几何抓取器生成的抓取计划。

转换后的数据保存为 IsaacLab 的 BC 格式：

``observations``:
    21 维本体状态 + 64×64×4 RGB-D，维度与 ``RgbdActorCritic`` 一致。
``actions``:
    7 维策略动作，前 3 维是末端相对平移，中间 3 维是相对旋转，第 7 维是夹爪开/关。

这里的 teacher 来源是 reBot-Isaacsim 中已经在仿真/真机验证过的传统抓取链路；
IsaacLab 只负责把它转成可训练数据，再做 BC/PPO。
"""

from __future__ import annotations  # 启用 postponed evaluation of annotations（PEP 563）

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # 用于 interpolate 缩放图像/深度图

# ---------------------------------------------------------------------------
# 路径与输出配置
# ---------------------------------------------------------------------------
# 项目根目录（脚本所在目录的上一层）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# IsaacSim 导出的 RGB-D 帧文件路径（可通过环境变量覆盖）
FRAME_PATH = Path(os.environ.get("REBOT_ISAACSIM_FRAME", "/tmp/rebot_sim_rgbd.npz"))
# IsaacSim 导出的抓取计划 JSON 文件路径（可通过环境变量覆盖）
PLAN_PATH = Path(os.environ.get("REBOT_ISAACSIM_PLAN", "/tmp/rebot_sim_grasp_plan.json"))
# 转换后的 BC 训练数据保存路径
OUTPUT_PATH = PROJECT_ROOT / "data" / "rgbd_isaacsim_teacher_latest.pt"

# ---------------------------------------------------------------------------
# 观测与动作维度常量
# ---------------------------------------------------------------------------
IMAGE_HEIGHT = 64       # RGB-D 观测图像高度（与 RgbdActorCritic 一致）
IMAGE_WIDTH = 64        # RGB-D 观测图像宽度
DEPTH_LIMIT_M = 1.5     # 深度图裁剪上限（米），超过此值的深度视为无效
PROPRIO_SIZE = 21       # 本体感知维度：7 关节角 + 7 关节速度 + 7 上一步动作
ACTION_DIM = 7          # 动作维度：3 平移 + 3 旋转 + 1 夹爪
DEFAULT_GRIPPER_M = 0.040  # 与 REBOT_DM_CFG 初始 left_finger 位置一致

# ---------------------------------------------------------------------------
# 收集策略超参（均可通过环境变量覆盖）
# ---------------------------------------------------------------------------
# 默认只转换当前最新计划，避免脚本无意中长时间阻塞。需要连续收集时设置更大值。
TARGET_PLANS = int(os.environ.get("REBOT_ISAACSIM_TEACHER_PLANS", "1"))
# 这里是"等待下一条新计划"的空闲超时，不是收集全过程总时长。
# IsaacSim 原流程执行一轮抓取/放置可能需要十几秒；collector 保持只读，不改变该流程。
WAIT_TIMEOUT_S = float(os.environ.get("REBOT_ISAACSIM_TEACHER_TIMEOUT_S", "120"))

# 每条 IsaacSim 计划展开为多步轨迹样本，而不是只取 3 个关键帧。
# 数值按 IsaacLab 动作尺度设计：单步最大平移约 2 cm，旋转约 0.10 rad。
APPROACH_STEPS = int(os.environ.get("REBOT_TEACHER_APPROACH_STEPS", "12"))   # 接近阶段步数
INSERT_STEPS = int(os.environ.get("REBOT_TEACHER_INSERT_STEPS", "8"))        # 插入阶段步数
CLOSE_STEPS = int(os.environ.get("REBOT_TEACHER_CLOSE_STEPS", "3"))          # 闭合夹爪阶段步数
RETREAT_STEPS = int(os.environ.get("REBOT_TEACHER_RETREAT_STEPS", "8"))      # 撤退阶段步数
RETURN_STEPS = int(os.environ.get("REBOT_TEACHER_RETURN_STEPS", "12"))       # 携物返回 ready 姿态步数

# reBot 底层 SDK 路径，用于加载 FK（正运动学）模型
SDK_ROOT = PROJECT_ROOT.parent / "reBot-Isaacsim" / "third_party" / "reBotArm_control_py"
# FK 工具缓存：(model, compute_fk, pad_q_for_model)，避免重复加载
_FK_CACHE: tuple[object, object, object] | None = None


def load_frame(path: Path) -> torch.Tensor:
    """读取 IsaacSim RGB-D 帧，并转成 IsaacLab 训练观测的图像部分。"""

    # 从 .npz 文件读取 BGR 彩色图和毫米深度图
    with np.load(path, allow_pickle=False) as data:
        color_bgr = data["color_bgr"].copy()  # (H, W, 3) BGR uint8
        depth_mm = data["depth_mm"].copy()    # (H, W) 深度毫米值

    # --- RGB 通道：BGR -> RGB -> resize -> 归一化 -> 零均值化 ---
    # IsaacSim 导出的是 BGR；训练观测使用 RGBD，RGB 归一化方式与 mdp.flattened_rgbd 一致。
    color_rgb = np.ascontiguousarray(color_bgr[..., ::-1])  # BGR -> RGB 通道翻转
    rgb = torch.from_numpy(color_rgb).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
    rgb = F.interpolate(rgb, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)  # 缩放到 64x64
    rgb = rgb.squeeze(0).permute(1, 2, 0) / 255.0  # 回到 (H, W, 3)，归一化到 [0, 1]
    rgb = rgb - rgb.mean(dim=(0, 1), keepdim=True)  # 减均值，零中心化

    # --- 深度通道：mm -> m -> resize -> 裁剪 -> 归一化到 [0, 1] ---
    depth_m = torch.from_numpy(depth_mm.astype(np.float32) / 1000.0).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) 米
    depth = F.interpolate(depth_m, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="nearest")  # 最近邻缩放，避免插值伪影
    depth = depth.squeeze(0).permute(1, 2, 0)  # (H, W, 1)
    depth = torch.nan_to_num(depth, nan=DEPTH_LIMIT_M, posinf=DEPTH_LIMIT_M, neginf=0.0)  # 异常值替换
    depth = depth.clamp(0.0, DEPTH_LIMIT_M) / DEPTH_LIMIT_M  # 裁剪并归一化到 [0, 1]

    # 拼接 RGB + Depth -> (64*64*4,) 一维向量
    return torch.cat((rgb, depth), dim=-1).reshape(-1).to(torch.float32)


def proprio_from_stage(
    stage: dict,
    ready_arm: np.ndarray,
    last_action: torch.Tensor,
) -> torch.Tensor:
    """把计划阶段转为与 PPO ``joint_pos_rel`` 一致的 21 维本体状态。"""

    # PPO 观测使用当前关节位置减默认关节位置；teacher 必须使用相同语义。
    arm = torch.from_numpy(np.asarray(stage["arm"], dtype=np.float32) - ready_arm.astype(np.float32))
    gripper = torch.tensor(
        [float(stage["gripper_m_per_finger"]) - DEFAULT_GRIPPER_M], dtype=torch.float32
    )
    joint_pos = torch.cat((arm, gripper))  # (7,) 关节位置
    joint_vel = torch.zeros(7, dtype=torch.float32)  # 关节速度：计划阶段无速度信息，填 0
    # 拼接为 21 维：7 关节角 + 7 关节速度 + 7 上一动作
    return torch.cat((joint_pos, joint_vel, last_action.float()))


def rotation_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    """把 3x3 旋转矩阵转成旋转向量，避免依赖 scipy。

    返回向量方向为旋转轴，长度为旋转角弧度。小角度时使用一阶近似，避免
    ``sin(theta)`` 接近 0 导致数值放大。
    """

    rotation = np.asarray(rotation, dtype=np.float64)
    # 从旋转矩阵的迹反算旋转角: trace(R) = 1 + 2*cos(theta)
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))  # 裁剪防止数值溢出
    theta = float(np.arccos(cos_theta))  # 旋转角 (rad)
    # 从反对称部分提取旋转轴方向 (未归一化)
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    # 小角度时 sin(theta) ≈ theta 导致除法不稳定，用一阶近似替代
    if theta < 1e-6:
        return 0.5 * skew  # 一阶近似: rotvec ≈ 0.5 * skew(R)
    # 标准公式: rotvec = theta / (2*sin(theta)) * skew(R)
    return skew * (theta / (2.0 * np.sin(theta)))


def make_action(
    current_xyz: np.ndarray,
    target_xyz: np.ndarray,
    current_rotation: np.ndarray,
    target_rotation: np.ndarray,
    gripper: float,
) -> torch.Tensor:
    """把相邻轨迹点的位姿差转为 IsaacLab 7 维相对 IK 动作。

    IsaacLab 环境中动作尺度为：平移 0.02 m，旋转 0.10 rad。因此 teacher
    动作也使用同一尺度归一化到 [-1, 1]，避免 BC 学到与 PPO 环境不一致的量纲。
    """

    action = torch.zeros(ACTION_DIM, dtype=torch.float32)
    # 前 3 维：末端相对平移，除以 0.02m 缩放后裁剪到 [-1, 1]
    delta = (np.asarray(target_xyz, dtype=np.float64) - np.asarray(current_xyz, dtype=np.float64)) / 0.02
    action[:3] = torch.from_numpy(np.clip(delta, -1.0, 1.0).astype(np.float32))
    # 中间 3 维：相对旋转 = R_target @ R_current^T，转旋转向量后除以 0.10rad 缩放
    delta_rotation = np.asarray(target_rotation, dtype=np.float64) @ np.asarray(current_rotation, dtype=np.float64).T
    rotvec = rotation_to_rotvec(delta_rotation) / 0.10
    action[3:6] = torch.from_numpy(np.clip(rotvec, -1.0, 1.0).astype(np.float32))
    # 第 7 维：夹爪开/关 (-1 闭合, +1 打开)
    action[6] = float(np.clip(gripper, -1.0, 1.0))
    return action


def load_fk_helpers() -> tuple[object, object, object] | None:
    """加载 reBot FK 工具；失败时返回 None，collector 会退回几何标签。

    优先 FK 的原因：计划里的 stage 已经包含 6 轴关节角，用 FK 可以得到
    ready/pregrasp/grasp/retreat 每个样本的真实 TCP 位姿，从而生成更完整的
    过程示教和姿态动作。失败通常是当前 conda 环境缺少 Pinocchio，此时仍可
    用原始 plan 里的几何点生成数据。
    """

    global _FK_CACHE
    if _FK_CACHE is not None:
        return _FK_CACHE  # 已缓存，直接返回
    if not SDK_ROOT.is_dir():
        return None  # SDK 目录不存在，无法加载
    if str(SDK_ROOT) not in sys.path:
        sys.path.insert(0, str(SDK_ROOT))  # 将 SDK 路径加入 Python 搜索路径
    try:
        from reBotArm_control_py.kinematics import compute_fk, load_robot_model, pad_q_for_model

        _FK_CACHE = (load_robot_model(), compute_fk, pad_q_for_model)  # 缓存模型和函数
        return _FK_CACHE
    except Exception as exc:
        print(f"[IsaacSimTeacher] FK unavailable, fallback to plan geometry: {exc}")
        return None


def stage_pose(stage: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """用 stage 的关节角做 FK，得到 TCP 位置和旋转矩阵。"""

    helpers = load_fk_helpers()
    if helpers is None:
        return None  # FK 不可用，返回 None 让调用方走 fallback
    model, compute_fk, pad_q_for_model = helpers
    arm = np.asarray(stage["arm"], dtype=np.float64)  # 6 轴关节角
    q = pad_q_for_model(model, arm, controlled_joints=6)  # 补齐到模型所需的全关节维度
    position, rotation, _ = compute_fk(model, q)  # 正运动学求解 TCP 位姿
    return np.asarray(position, dtype=np.float64), np.asarray(rotation, dtype=np.float64)


def fallback_pose(plan: dict, stage_name: str) -> tuple[np.ndarray, np.ndarray]:
    """FK 不可用时，用 plan 几何字段近似几个抓取阶段的 TCP 位姿。"""

    # 所有阶段共用同一个 TCP 姿态（抓取计划中的固定值）
    rotation = np.asarray(plan["tcp_rotation"], dtype=np.float64)
    # 根据阶段名选择对应的位置
    if stage_name == "pregrasp":
        position = np.asarray(plan["pregrasp_position_m"], dtype=np.float64)  # 预抓取点
    elif stage_name in {"grasp", "close"}:
        position = np.asarray(plan["grasp_position_m"], dtype=np.float64)  # 抓取点（含闭合）
    elif stage_name == "retreat":
        position = np.asarray(plan["pregrasp_position_m"], dtype=np.float64)  # 撤回点 = 预抓取点
    else:
        raise KeyError(stage_name)
    return position, rotation


def blend_stage(start_stage: dict, end_stage: dict, alpha: float) -> dict:
    """线性插值关节和夹爪，构造中间本体状态。"""

    start_arm = np.asarray(start_stage["arm"], dtype=np.float64)
    end_arm = np.asarray(end_stage["arm"], dtype=np.float64)
    start_gripper = float(start_stage["gripper_m_per_finger"])
    end_gripper = float(end_stage["gripper_m_per_finger"])
    # 球面线性插值 (LERP): result = start + alpha * (end - start)
    return {
        "arm": (start_arm + alpha * (end_arm - start_arm)).tolist(),
        "gripper_m_per_finger": start_gripper + alpha * (end_gripper - start_gripper),
    }


def add_transition_samples(
    *,
    observations: list[torch.Tensor],
    actions: list[torch.Tensor],
    labels: list[dict],
    image_obs: torch.Tensor,
    plan: dict,
    stages: dict,
    ready_arm: np.ndarray,
    last_action: torch.Tensor,
    start_name: str,
    end_name: str,
    steps: int,
    gripper_action: float,
    use_fk: bool,
) -> torch.Tensor:
    """把一个 stage 过渡展开成多条 BC 样本，并返回最后一个动作。"""

    steps = max(1, int(steps))  # 确保至少 1 步
    # 获取起点和终点的 TCP 位姿：优先 FK，失败则回退几何标签
    start_pose = stage_pose(stages[start_name]) if use_fk else None
    end_pose = stage_pose(stages[end_name]) if use_fk else None
    if start_pose is None:
        start_pose = fallback_pose(plan, start_name)
    if end_pose is None:
        end_pose = fallback_pose(plan, end_name)
    start_xyz, start_rotation = start_pose
    end_xyz, end_rotation = end_pose
    # 整个过渡段的总旋转向量（用于球面线性插值 SLERP）
    total_rotvec = rotation_to_rotvec(end_rotation @ start_rotation.T)

    for step in range(steps):
        # 起点比例 alpha0，终点比例 alpha1
        alpha0 = step / steps
        alpha1 = (step + 1) / steps
        # 线性插值关节角构造当前中间本体状态
        current_stage = blend_stage(stages[start_name], stages[end_name], alpha0)
        # 线性插值末端位置
        current_xyz = start_xyz + alpha0 * (end_xyz - start_xyz)
        target_xyz = start_xyz + alpha1 * (end_xyz - start_xyz)
        # 球面线性插值 (SLERP) 末端姿态
        current_rotation = rotvec_to_rotation(total_rotvec * alpha0) @ start_rotation
        target_rotation = rotvec_to_rotation(total_rotvec * alpha1) @ start_rotation
        # 将位姿差转为 7 维动作向量
        action = make_action(current_xyz, target_xyz, current_rotation, target_rotation, gripper_action)
        # 构造本体感知状态
        proprio = proprio_from_stage(current_stage, ready_arm, last_action)
        # 一条样本 = 21 维本体状态 + 64x64x4 RGB-D 展开向量
        observations.append(torch.cat((proprio, image_obs)).unsqueeze(0))
        actions.append(action.unsqueeze(0))
        labels.append(
            {
                "plan_timestamp": float(plan["timestamp"]),
                "segment": f"{start_name}->{end_name}",
                "step": step + 1,
                "steps": steps,
                "action": action.tolist(),
                "source": "reBot-Isaacsim",
                "pose_source": "fk" if use_fk else "plan_geometry",
            }
        )
        last_action = action  # 当前动作成为下一步的"上一步动作"
    return last_action


def rotvec_to_rotation(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues 公式：旋转向量转 3x3 旋转矩阵。"""

    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(rotvec))  # 旋转角 = 旋转向量的模
    if theta < 1e-9:
        return np.eye(3, dtype=np.float64)  # 零旋转 -> 单位矩阵
    axis = rotvec / theta  # 归一化旋转轴
    kx, ky, kz = axis
    # 旋转轴的反对称矩阵
    skew = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    # Rodrigues 公式: R = I + sin(theta)*K + (1-cos(theta))*K^2
    return np.eye(3, dtype=np.float64) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def samples_from_plan(plan: dict, image_obs: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict]]:
    """从 IsaacSim 计划中只提取“抓取并返回 ready 姿态”的 teacher 样本。"""

    # 将 stages 列表转为按名称索引的字典
    stages = {stage["name"]: stage for stage in plan["stages"]}
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    labels: list[dict] = []
    last_action = torch.zeros(ACTION_DIM, dtype=torch.float32)  # 初始动作 = 零向量

    required_stages = {"open", "pregrasp", "grasp", "close", "retreat", "return"}
    missing_stages = sorted(required_stages.difference(stages))
    if missing_stages:
        raise ValueError(f"抓取返回计划缺少阶段: {missing_stages}")

    # 返回 ready 姿态需要把 return 关节目标转换成 TCP 相对动作，因此必须有 FK。
    # 不再静默退化成只含局部抓取的旧数据，避免 BC 训练目标与 PPO 任务不一致。
    use_fk = stage_pose(stages["open"]) is not None
    if not use_fk:
        raise RuntimeError("抓取返回 teacher 需要 FK；请检查 reBot SDK 和 Pinocchio 环境")
    ready_arm = np.asarray(stages["open"]["arm"], dtype=np.float64)

    # IsaacSim plan 后续仍可包含 place/release 等阶段，但 IsaacLab 明确忽略它们。
    recipe = [
        ("open", "pregrasp", APPROACH_STEPS, 1.0),       # ready -> 预抓取（夹爪打开）
        ("pregrasp", "grasp", INSERT_STEPS, 1.0),       # 预抓取 -> 抓取（夹爪打开）
        ("grasp", "close", CLOSE_STEPS, -1.0),          # 原位闭合夹爪
        ("close", "retreat", RETREAT_STEPS, -1.0),      # 先安全撤离物体区域
        ("retreat", "return", RETURN_STEPS, -1.0),       # 保持夹紧并返回 ready 姿态
    ]

    # 按配方依次展开每个过渡段为多步样本
    for start_name, end_name, steps, gripper_action in recipe:
        last_action = add_transition_samples(
            observations=observations,
            actions=actions,
            labels=labels,
            image_obs=image_obs,
            plan=plan,
            stages=stages,
            ready_arm=ready_arm,
            last_action=last_action,
            start_name=start_name,
            end_name=end_name,
            steps=steps,
            gripper_action=gripper_action,
            use_fk=use_fk,
        )

    return observations, actions, labels


def load_plan(path: Path) -> dict:
    """读取并校验 IsaacSim 抓取计划的最小字段。"""

    plan = json.loads(path.read_text(encoding="utf-8"))  # 从 JSON 文件加载计划
    # 检查必要字段是否齐全
    required = ("timestamp", "source", "pregrasp_position_m", "grasp_position_m", "stages")
    missing = [key for key in required if key not in plan]
    if missing:
        raise ValueError(f"抓取计划缺少字段: {missing}")
    # 安全检查：只接受仿真来源的计划，拒绝真机计划
    if plan["source"] != "sim":
        raise ValueError(f"只接受 IsaacSim sim 计划，当前 source={plan['source']!r}")
    return plan


def main() -> None:
    """等待并转换 IsaacSim 抓取计划，保存为 BC teacher 数据。"""

    # --- 收集状态初始化 ---
    observations: list[torch.Tensor] = []  # 所有观测样本
    actions: list[torch.Tensor] = []       # 所有动作样本
    labels: list[dict] = []                # 所有样本标签（含元数据）
    seen_timestamps: set[float] = set()    # 已成功收集的计划时间戳
    skipped_timestamps: set[float] = set() # 已跳过（失败/重复）的计划时间戳
    # 待确认样本缓存：时间戳 -> (观测列表, 动作列表, 标签列表)
    pending_samples: dict[float, tuple[list[torch.Tensor], list[torch.Tensor], list[dict]]] = {}
    collector_started = time.time()  # 收集器启动时间
    deadline = time.time() + WAIT_TIMEOUT_S  # 空闲超时截止时间

    print(f"[IsaacSimTeacher] frame={FRAME_PATH}")
    print(f"[IsaacSimTeacher] plan={PLAN_PATH}")
    print(f"[IsaacSimTeacher] target_plans={TARGET_PLANS}")
    print(f"[IsaacSimTeacher] idle_timeout_s={WAIT_TIMEOUT_S}")

    # --- 主循环：轮询等待并收集计划 ---
    while len(seen_timestamps) < TARGET_PLANS:
        # 空闲超时检查：超过 WAIT_TIMEOUT_S 没有新计划则抛异常
        if time.time() > deadline:
            raise TimeoutError(
                f"等待新的 IsaacSim teacher 计划超时: "
                f"已收集 {len(seen_timestamps)}/{TARGET_PLANS} 个计划"
            )
        # 等待 IsaacSim 写入文件
        if not FRAME_PATH.exists() or not PLAN_PATH.exists():
            time.sleep(0.1)
            continue

        # 加载并校验计划
        plan = load_plan(PLAN_PATH)
        timestamp = float(plan["timestamp"])
        # 跳过已处理的时间戳
        if timestamp in seen_timestamps or timestamp in skipped_timestamps:
            time.sleep(0.1)
            continue

        # plan 刚生成时先缓存"规划当时"的 RGB-D 和动作标签，但不立刻计入数据集。
        # 等 IsaacSim 执行端把同一个文件写回 executed=true 后，才说明这一轮确实跑过。
        if timestamp not in pending_samples:
            image_obs = load_frame(FRAME_PATH)  # 加载当前 RGB-D 帧
            pending_samples[timestamp] = samples_from_plan(plan, image_obs)  # 展开为样本并缓存

        # 等待 IsaacSim 执行端标记 executed=true
        if plan.get("executed") is not True:
            time.sleep(0.1)
            continue
        # 安全检查：只接受本收集器启动之后执行完成的计划
        if float(plan.get("executed_at", -1.0)) < collector_started:
            time.sleep(0.1)
            continue

        # 抓取失败的计划：丢弃缓存样本，记录跳过，重置超时
        if plan.get("grasp_success") is False:
            pending_samples.pop(timestamp, None)
            skipped_timestamps.add(timestamp)
            deadline = time.time() + WAIT_TIMEOUT_S  # 失败后刷新超时，给下一轮机会
            print(
                f"[IsaacSimTeacher] skipped failed plan: "
                f"gripper_block={float(plan.get('grasp_gripper_block_m', 0.0)):.4f}m"
            )
            continue

        # 抓取成功：将缓存样本正式计入数据集
        plan_obs, plan_actions, plan_labels = pending_samples.pop(timestamp)
        observations.extend(plan_obs)
        actions.extend(plan_actions)
        labels.extend(plan_labels)
        seen_timestamps.add(timestamp)
        deadline = time.time() + WAIT_TIMEOUT_S  # 成功后刷新超时
        print(f"[IsaacSimTeacher] collected plan {len(seen_timestamps)}/{TARGET_PLANS}: samples={len(plan_actions)}")

    # --- 组装并保存数据集 ---
    dataset = {
        "observations": torch.cat(observations, dim=0).to(torch.float16),  # 观测用 fp16 节省存储
        "actions": torch.cat(actions, dim=0).to(torch.float32),            # 动作用 fp32 保持精度
        "teacher_type": "isaacsim_traditional_grasp",
        "task": "grasp_return_ready",
        "proprioception": "joint_pos_rel7,joint_vel7,last_action7",
        "source_frame": str(FRAME_PATH),
        "source_plan": str(PLAN_PATH),
        "target_plans": TARGET_PLANS,
        "collected_plans": len(seen_timestamps),
        "samples_per_plan_mean": len(actions) / max(1, len(seen_timestamps)),  # 每条计划平均样本数
        "trajectory_steps": {
            "approach": APPROACH_STEPS,
            "insert": INSERT_STEPS,
            "close": CLOSE_STEPS,
            "retreat": RETREAT_STEPS,
            "return": RETURN_STEPS,
        },
        "labels": labels,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
    torch.save(dataset, OUTPUT_PATH)
    print(f"[IsaacSimTeacher] 保存: {OUTPUT_PATH}")
    print(f"[IsaacSimTeacher] samples={dataset['actions'].shape[0]}")


if __name__ == "__main__":
    main()
