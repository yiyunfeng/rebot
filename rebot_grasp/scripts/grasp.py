"""GraspNet 真机抓取入口。

配置固定读取 config/default.yaml。
G/Space 抓取，R 恢复预览，Q/Esc 退出。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Matplotlib 默认会在用户目录写缓存；指定到 /tmp，避免只读环境下启动失败。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# OpenCV 的窗口由 Qt 创建，显式指定字体目录可减少中文或状态文字的字体警告。
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

# 当前脚本位于 rebot_grasp/scripts，因此 parents[1] 就是 rebot_grasp 根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# GraspNet 源码直接放在项目 sdk 目录中，需要把它的多个子目录加入 Python 搜索路径。
GRASPNET_ROOT = PROJECT_ROOT / "sdk" / "graspnet-baseline"
# 本脚本统一从 default.yaml 读取相机、模型、机械臂和抓取参数。
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def _prepare_imports() -> None:
    """安排项目与 GraspNet SDK 的导入顺序，避免同名 ``utils`` 包冲突。"""
    # 项目自己的 utils 必须排在第一位，否则可能误导入 GraspNet SDK 的同名包。
    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    graspnet_paths = [
        GRASPNET_ROOT,
        *(GRASPNET_ROOT / subdir for subdir in ("models", "dataset", "utils", "pointnet2", "graspnetAPI")),
    ]
    # reversed + insert(1) 可保持列表原有优先级，同时把 SDK 路径放在项目根目录之后。
    for path in reversed(graspnet_paths):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(1, path_str)


# 必须先调整 sys.path，再导入下面的项目模块和 GraspNet 模块。
_prepare_imports()

import utils.graspnet_utils as graspnet_utils  # noqa: E402
from drivers.camera import make_camera  # noqa: E402
from drivers.robot.grasp_driver import (  # noqa: E402
    GRIPPER_MAX_DISTANCE_M,
    GraspDriver,
    selected_arm_config,
)
from graspnetAPI import Grasp, GraspGroup  # noqa: E402
from reBotArm_control_py.actuator import RebotArm  # noqa: E402
from reBotArm_control_py.controllers import RebotArmEndPose  # noqa: E402
from reBotArm_control_py.kinematics import (  # noqa: E402
    get_end_effector_frame_id,
    load_robot_model,
    pad_q_for_model,
    pos_rot_to_se3,
    solve_ik,
)
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams  # noqa: E402
from utils.camera_utils import (  # noqa: E402
    compose_cam_to_base_transform,
    load_config,
    load_hand_eye,
)
from utils.transforms import (  # noqa: E402
    apply_execution_compensation_to_pose,
    canonicalize_parallel_gripper_tcp_rotation,
    graspnet_rotation_to_rebot_tcp_rotation,
    rotation_matrix_to_euler_zyx,
)
from utils.yolo_utils import (  # noqa: E402
    YoloDetection,
    detect_objects,
)
from utils.yolo_utils import (
    load_yolo as load_yolo_from_config,
)


def _wait_motion(controller: RebotArmEndPose, duration: float, extra: float = 0.6) -> None:
    """等待异步轨迹线程结束；取不到线程时按预计时长等待。"""
    # RebotArmEndPose 轨迹发送在线程里执行；没有线程时退回固定等待。
    # move_to_traj() 返回后，轨迹发送线程可能仍在控制机械臂运动。
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        # duration 是规划的运动时间，extra 和 2 秒用于容纳通信及线程收尾时间。
        thread.join(timeout=duration + extra + 2.0)
    else:
        # 某些控制器版本不提供 _send_thread，此时只能按预计运动时间等待。
        time.sleep(duration + extra)


def _move_ready(controller: RebotArmEndPose, ready_cfg: dict[str, Any]) -> None:
    """将末端移动到抓取前后共用的待机位。"""
    # ready_pose 是抓取前后统一停靠位，来自 config/default.yaml。
    # 位姿格式固定为 (x, y, z, roll, pitch, yaw)，位置单位 m，角度单位 rad。
    # get() 后面的数值仅是配置缺失时的默认值，正常运行使用 default.yaml 中的值。
    duration = float(ready_cfg.get("duration", 3.0))
    controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.25)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.35)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 1.2)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    _wait_motion(controller, duration)


class IkChecker:
    """用当前关节角检查候选抓取是否 IK 可达。"""

    def __init__(self, arm: RebotArm) -> None:
        """加载运动学模型，并保存当前机械臂的受控关节数和 IK 参数。"""
        self._arm = arm
        # groups.arm 描述机械臂关节组；夹爪关节不参与末端位姿的 IK 计算。
        self._arm_group = arm.groups.get("arm")
        if self._arm_group is None:
            raise ValueError("Hardware config missing groups.arm")
        self._n = self._arm_group.num_joints

        # load_robot_model() 返回 Pinocchio 模型，data 保存每次运动学计算的中间结果。
        self._model = load_robot_model()
        self._data = self._model.createData()
        # IK 的目标 frame 是机械臂模型中定义的末端执行器/TCP frame。
        self._end_frame_id = get_end_effector_frame_id(self._model)
        # 最大迭代次数、收敛误差、更新步长和阻尼共同控制数值 IK 的求解过程。
        self._params = IKParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)

    def check(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> tuple[bool, float]:
        """从当前关节角开始求目标 TCP 的 IK，返回是否成功和末端误差。"""
        # 使用当前真实关节角作为 IK 初值，比固定零位更容易收敛到连续可执行的解。
        # get_state() 返回的不只有机械臂关节，因此这里只取前 self._n 个受控关节角。
        q_now = self._arm.get_state(request_feedback=False)[0][: self._n]
        # Pinocchio 模型可能还包含额外关节，pad_q_for_model() 将当前关节角补齐到模型维度。
        q_init = pad_q_for_model(self._model, q_now, self._n)
        # 把 base 坐标系下的 xyz + RPY 转成 IK 求解器使用的 SE(3) 目标位姿。
        target = pos_rot_to_se3(np.array([x, y, z], dtype=np.float64), roll=roll, pitch=pitch, yaw=yaw)
        result = solve_ik(
            self._model,
            self._data,
            self._end_frame_id,
            target,
            q_init,
            self._params,
            controlled_joints=self._n,
        )
        # success 表示是否达到 tolerance；error 是最终末端位姿残差，供日志判断失败程度。
        return bool(result.success), float(result.error)


def _execute_grasp(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    retreat6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
) -> bool:
    """执行开爪、接近、闭爪、后撤、回待机位和释放的完整动作。"""
    # 三个位姿都采用 (x, y, z, roll, pitch, yaw)，这里拆开后直接传给控制器。
    # 动作顺序保持直线：开爪 -> 预抓取 -> 抓取 -> 闭爪 -> 后撤 -> 回 ready -> 释放。
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d
    xr, yr, zr, rxr, ryr, rzr = retreat6d

    print(f"[Grasp] pregrasp xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f}) rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Grasp] grasp    xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f}) rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")
    print(f"[Grasp] retreat  xyz=({xr:+.3f},{yr:+.3f},{zr:+.3f}) rpy=({rxr:+.3f},{ryr:+.3f},{rzr:+.3f})")

    print("[Grasp] Open gripper")
    # 先完全张开夹爪，避免接近物体时夹爪与物体提前碰撞。
    grasp_driver.open_gripper()

    print("[Grasp] Move to pregrasp")
    # 预抓取点位于物体前方，先到这里，再沿抓取方向靠近物体。
    if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[Grasp] Pregrasp IK failed")
        return False
    _wait_motion(controller, 2.0)

    print("[Grasp] Move to grasp")
    # 从预抓取点移动到最终夹取点；控制器返回 False 表示该段轨迹无法生成。
    if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[Grasp] Grasp IK failed")
        return False
    _wait_motion(controller, 1.5)

    print("[Grasp] Closing")
    # grasp() 会闭合夹爪，并根据夹爪反馈判断是否夹到物体。
    ok = grasp_driver.grasp()
    print("[Grasp] Holding object" if ok else "[Grasp] Empty grasp")

    print("[Grasp] Retreat")
    # 保持闭爪姿态退出物体区域。后撤失败时仍继续尝试回 ready，避免流程卡死。
    if controller.move_to_traj(xr, yr, zr, rxr, ryr, rzr, duration=1.5):
        _wait_motion(controller, 1.5)

    print("[Grasp] Return ready")
    # 把抓到的物体带回统一待机位，再执行释放。
    _move_ready(controller, ready_cfg)

    print("[Grasp] Release at ready")
    grasp_driver.release_gripper()
    return ok


def _print_grasp(grasp: Grasp) -> None:
    """打印 GraspNet 原始候选及转换后的 reBot TCP 姿态。"""
    # GraspNet 输出的是自己的夹爪坐标轴定义，不能直接作为 reBot TCP 旋转矩阵。
    # 第一步将轴方向转换成 reBot TCP 定义；第二步在平行夹爪的等价姿态中选择较稳定的一支。
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(graspnet_rotation_to_rebot_tcp_rotation(grasp.rotation_matrix))
    print("\n[G] Best GraspNet grasp:")
    print(f"  score={grasp.score:.4f} width={grasp.width:.4f} height={grasp.height:.4f} depth={grasp.depth:.4f}")
    print(f"  position_xyz={grasp.translation.tolist()}")
    print(f"  graspnet_rpy={rotation_matrix_to_euler_zyx(grasp.rotation_matrix).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def _rank_grasps(grasps: GraspGroup) -> GraspGroup:
    """复制候选，尽量去除高度重叠项，再按 GraspNet score 从高到低排序。"""
    # 复制底层数组，避免 NMS 和排序改变 infer_frame() 返回的原始候选顺序。
    ranked = GraspGroup(grasps.grasp_group_array.copy())
    try:
        # NMS 去掉位置和姿态高度重叠的重复抓取，只保留更高分的候选。
        ranked = ranked.nms()
    except Exception as exc:
        print(f"[WARN] GraspNet NMS skipped: {exc}")
    # GraspGroup.sort_by_score() 将高分候选放在前面，后续按此顺序逐个检查可达性。
    ranked.sort_by_score()
    return ranked


def _pose_z_ok(pose6d: tuple[float, ...], min_z: float) -> bool:
    """检查基座坐标系 Z 高度，避免目标低于配置的安全下限。"""
    return float(pose6d[2]) >= float(min_z)


def _select_executable_grasp(
    ik_checker: IkChecker,
    grasps: GraspGroup,
    T_cam2base: np.ndarray,
    cfg: dict[str, Any],
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float,
    min_base_z_m: float,
) -> Optional[tuple[Grasp, tuple[float, ...], tuple[float, ...], tuple[float, ...]]]:
    """按分数依次寻找同时满足高度限制和 IK 可达性的抓取。

    分数最高不代表机械臂一定能够到达，因此候选先排序，再逐个转换到 base
    坐标系、应用执行补偿并检查预抓取点和抓取点。返回第一个可执行候选。
    """
    # GraspNet 的最高分候选不一定满足机械臂工作空间和 Z 高度限制。
    # 因此这里不是直接使用 result.best，而是从高分到低分寻找第一个可执行候选。
    ranked = _rank_grasps(grasps)
    # 分别统计因高度过低和 IK 失败而跳过的数量，便于从日志判断失败原因。
    skipped_low = 0
    skipped_ik = 0
    # 保存所有失败候选中最大的 IK 残差，只用于最终日志，不参与候选排序。
    worst_err = 0.0

    for idx in range(len(ranked)):
        grasp = ranked[idx]

        # 三个偏移全部设为 0，得到物体抓取中心的原始 base 位姿。
        # 这个位姿只用于打印和排查手眼变换，不会发送给机械臂。
        raw_object6d, _, _ = graspnet_utils.grasp_to_base_poses(
            grasp,
            T_cam2base,
            0.0,
            0.0,
            0.0,
        )
        # 将相机坐标系中的候选抓取通过 T_cam2base 转到机械臂 base 坐标系：
        # raw_grasp6d 是夹取点；raw_pre6d 是沿接近方向退开的预抓取点；
        # raw_retreat6d 是夹取后退出物体区域的后撤点。
        raw_grasp6d, raw_pre6d, raw_retreat6d = graspnet_utils.grasp_to_base_poses(
            grasp,
            T_cam2base,
            pregrasp_offset_m,
            retreat_offset_m,
            insertion_depth_m,
        )
        # 标定仍可能存在少量系统误差，执行补偿只加到最终控制目标上。
        # 保留上面的 raw_* 位姿，日志中可以明确区分“计算结果”和“实际执行目标”。
        grasp6d = apply_execution_compensation_to_pose(raw_grasp6d, cfg)
        pre6d = apply_execution_compensation_to_pose(raw_pre6d, cfg)
        retreat6d = apply_execution_compensation_to_pose(raw_retreat6d, cfg)

        # 预抓取点和抓取点低于 min_base_z_m 时直接丢弃，避免末端靠近桌面以下。
        # 后撤点属于离开物体的动作，这里不把它作为候选是否可夹取的前置条件。
        if not (_pose_z_ok(pre6d, min_base_z_m) and _pose_z_ok(grasp6d, min_base_z_m)):
            skipped_low += 1
            continue

        # 必须先能到达预抓取点，再检查最终抓取点。
        # 如果预抓取点已经不可达，就不再重复计算抓取点 IK。
        pre_ok, pre_err = ik_checker.check(*pre6d)
        grasp_ok, grasp_err = ik_checker.check(*grasp6d) if pre_ok else (False, pre_err)
        worst_err = max(worst_err, pre_err, grasp_err)
        if pre_ok and grasp_ok:
            # 找到第一个满足高度和 IK 的候选后立即返回；它是“最高分的可执行候选”。
            print(f"[G] Executable rank={idx + 1}/{len(ranked)} score={grasp.score:.4f}")
            print(
                "[Base object] "
                f"xyz=({raw_object6d[0]:+.3f},{raw_object6d[1]:+.3f},{raw_object6d[2]:+.3f}) "
                f"rpy=({raw_object6d[3]:+.3f},{raw_object6d[4]:+.3f},{raw_object6d[5]:+.3f})"
            )
            print(
                "[Base raw] pregrasp "
                f"xyz=({raw_pre6d[0]:+.3f},{raw_pre6d[1]:+.3f},{raw_pre6d[2]:+.3f}) "
                f"rpy=({raw_pre6d[3]:+.3f},{raw_pre6d[4]:+.3f},{raw_pre6d[5]:+.3f})"
            )
            print(
                "[Base raw] grasp    "
                f"xyz=({raw_grasp6d[0]:+.3f},{raw_grasp6d[1]:+.3f},{raw_grasp6d[2]:+.3f}) "
                f"rpy=({raw_grasp6d[3]:+.3f},{raw_grasp6d[4]:+.3f},{raw_grasp6d[5]:+.3f})"
            )
            print(
                "[Base raw] retreat  "
                f"xyz=({raw_retreat6d[0]:+.3f},{raw_retreat6d[1]:+.3f},{raw_retreat6d[2]:+.3f}) "
                f"rpy=({raw_retreat6d[3]:+.3f},{raw_retreat6d[4]:+.3f},{raw_retreat6d[5]:+.3f})"
            )
            if skipped_low or skipped_ik:
                print(f"[G] Skipped low_z={skipped_low} ik_fail={skipped_ik}")
            # 同时返回 GraspNet 候选和三个补偿后的 base 位姿，供显示及机械臂执行。
            return grasp, grasp6d, pre6d, retreat6d
        skipped_ik += 1

    # 所有候选均不满足条件时返回 None，主循环只显示结果，不驱动机械臂。
    print(f"[G] No IK-reachable grasp: low_z={skipped_low} ik_fail={skipped_ik} max_err={worst_err:.4f}")
    return None


def main() -> int:
    """运行 GraspNet 真机预览，并在按键后选择一个可执行抓取。"""
    # main.py 已根据 perception.backend 选择入口；这里打印配置值，方便确认当前确实走 GraspNet。
    cfg = load_config(CONFIG_PATH)
    backend = str(cfg.get("perception", {}).get("backend", "graspnet")).strip().lower()
    print(f"[Mode] perception.backend={backend} -> GraspNet")

    # robot_cfg 保存机械臂、控制器和夹爪配置；ready_cfg 是抓取前后的安全待机位。
    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )
    # grasp_cfg 控制机械臂动作几何，graspnet_cfg 控制点云、网络推理和候选过滤。
    grasp_cfg = cfg.get("grasp_pipeline", {}).get("grasp", {})
    graspnet_cfg = cfg.get("graspnet", {})

    # pregrasp_offset_m：抓取前沿接近方向留出的距离。
    # retreat_offset_m：闭爪后沿退出方向移动的距离。
    # insertion_depth_m：在 GraspNet 抓取中心基础上继续插入的深度。
    # min_base_z_m：base 坐标系下允许执行的最低高度。
    pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
    retreat_offset_m = float(grasp_cfg.get("retreat_offset_m", pregrasp_offset_m))
    insertion_depth_m = float(grasp_cfg.get("insertion_depth_m", 0.0))
    min_base_z_m = float(grasp_cfg.get("min_base_z_m", 0.03))

    # checkpoint 是 GraspNet 权重；num_point 是每帧送入网络的点云采样数；
    # num_view 是网络评估的接近方向数量。
    checkpoint = graspnet_utils.resolve_checkpoint_path(graspnet_cfg.get("checkpoint", "checkpoint-rs.tar"))
    num_point = int(graspnet_cfg.get("num_point", 20000))
    num_view = int(graspnet_cfg.get("num_view", graspnet_utils.DEFAULT_NUM_VIEW))
    # collision_thresh 和 voxel_size 用于模型自由碰撞检测；深度范围用于去除无效点云。
    collision_thresh = float(graspnet_cfg.get("collision_thresh", 0.01))
    voxel_size = float(graspnet_cfg.get("voxel_size", 0.01))
    min_depth = float(graspnet_cfg.get("min_depth", 0.05))
    max_depth = float(graspnet_cfg.get("max_depth", 1.0))
    # 启用 YOLO 过滤时，只保留目标检测框附近的点云，避免 GraspNet 抓到其它物体。
    # target_margin_px/target_expand_ratio 用于适当扩大检测区域，保留物体边缘点。
    target_class = graspnet_cfg.get("target_class")
    target_margin_px = int(graspnet_cfg.get("target_margin_px", 12))
    target_expand_ratio = float(graspnet_cfg.get("target_expand_ratio", 1.0))
    use_yolo_filter = bool(graspnet_cfg.get("use_yolo_filter", True))
    # Open3D 窗口只用于查看点云和候选抓取，不参与机械臂控制。
    open3d_enabled = bool(graspnet_cfg.get("open3d_enabled", True))
    open3d_grasps = str(graspnet_cfg.get("open3d_grasps", "pre-bbox"))
    # 配置值不合法时退回 pre-bbox，防止后面找不到对应的候选集合。
    if open3d_grasps not in {"final", "bbox", "pre-bbox"}:
        open3d_grasps = "pre-bbox"
    warmup_frames = int(graspnet_cfg.get("warmup_frames", 20))

    # make_camera() 根据 camera.type 创建对应 RGB-D 相机驱动。
    cam_cfg = cfg["camera"]
    print(f"=== Init camera: {cam_cfg['type']} {cam_cfg.get('color_width')}x{cam_cfg.get('color_height')}@{cam_cfg.get('fps')} ===")
    print(f"[GraspNet] Open3D candidates window: {'enabled' if open3d_enabled else 'disabled'}, set={open3d_grasps}, top_k={graspnet_cfg.get('top_k', 50)}")
    cam = make_camera(cfg)

    # 以下变量保存上一轮预览结果。YOLO 不必每帧运行，因此中间帧复用最近一次检测。
    last_detections: list[YoloDetection] = []
    selected_target: Optional[Any] = None
    last_target_status = "target detector warming up..."
    status = "warming up camera..."
    # frozen=True 表示已经按下 G 并保留抓取瞬间，按 R 后恢复实时画面。
    frozen = False
    last_display: Optional[np.ndarray] = None
    # frame_index 控制 YOLO 推理间隔；fps_* 只负责计算预览帧率。
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0
    window_name = "Main - GraspNet Grasp"
    top_k = int(graspnet_cfg.get("top_k", 50))
    # Open3D 窗口按第一次需要时再创建，避免禁用可视化时仍加载窗口资源。
    vis: Optional[graspnet_utils.Open3DGraspWindow] = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(cam_cfg.get("color_width", 1280)), int(cam_cfg.get("color_height", 720)))
    print("\n[Keys] G/SPACE=grasp  R=resume  Q/ESC=quit\n")

    # 先设为 None，确保初始化中途失败时 finally 也能安全判断并释放已创建的资源。
    rebotarm: Optional[RebotArm] = None
    controller: Optional[RebotArmEndPose] = None
    grasp_driver: Optional[GraspDriver] = None
    ik_checker: Optional[IkChecker] = None
    T_hand_eye: Optional[np.ndarray] = None
    robot_ready = False

    try:
        # 1. 相机和标定。
        # warm_up() 丢弃相机刚启动时曝光和深度尚未稳定的若干帧。
        cam.open()
        cam.warm_up(warmup_frames)
        # K 是 3x3 相机内参矩阵，用于深度反投影及二维抓取姿态显示。
        K = cam.K.astype(np.float64)
        print("Camera intrinsics:")
        print(K)

        cam_type = str(cam_cfg.get("type", "")).lower()
        # 眼在手上模式需要“相机 -> TCP/夹爪”的手眼矩阵，才能把抓取点转换到 base。
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            # 没有正确标定时仍允许查看识别和抓取候选，但禁止机械臂执行。
            print("[WARN] Hand-eye calibration unavailable; grasp execution disabled")
            T_hand_eye = None

        # 2. 感知模型。
        print("=== Load models ===")
        # use_yolo_filter=False 时 no_yolo=True，不加载 YOLO，GraspNet 对整幅点云找抓取。
        yolo_model, yolo_opts = load_yolo_from_config(
            cfg,
            project_root=PROJECT_ROOT,
            no_yolo=not use_yolo_filter,
        )
        last_target_status = "YOLO disabled: full-scene GraspNet" if yolo_model is None else "target detector warming up..."
        # 加载 GraspNet 网络及权重；真正的逐帧推理只在按 G/Space 后发生。
        net = graspnet_utils.build_net(checkpoint, num_view)

        # 3. 机械臂和夹爪。
        print("=== Init robot ===")
        # 根据项目中的机械臂配置确定控制模式，再创建底层通信和末端位姿控制器。
        selected = selected_arm_config(robot_cfg.get("repo_root"))
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
        grasp_driver = GraspDriver(
            rebotarm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        # start() 初始化夹爪并进入可控制状态；成功后才把 robot_ready 设为 True。
        grasp_driver.start()
        robot_ready = True
        # IK 检查器只筛选候选，不直接驱动机械臂。
        ik_checker = IkChecker(rebotarm)
        print(f"[Robot] mode: {selected.controller_mode}")

        print("[Robot] Move ready")
        _move_ready(controller, ready_cfg)

        # 4. 实时预览；按 G/Space 时跑 GraspNet 并执行一次抓取。
        while True:
            # 相机每次同时返回 BGR 彩色图和以 mm 为单位的深度图；任一缺失就等待下一帧。
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            # 每秒更新一次平均 FPS，避免每帧计算出的数值跳动过大。
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # YOLO 只在实时预览阶段按 infer_every 间隔运行；冻结画面后不再更新检测框。
            if not frozen and yolo_model is not None and (frame_index == 1 or frame_index % int(yolo_opts["infer_every"]) == 0):
                try:
                    _, last_detections = detect_objects(yolo_model, color_bgr, yolo_opts)
                    # select_target() 按配置的 target_class 从多个检测物体中选出 GraspNet 目标。
                    selected_target = graspnet_utils.select_target(last_detections, target_class)
                    last_target_status = graspnet_utils.target_status_text(selected_target, last_detections, target_class)
                except Exception as exc:
                    # 检测失败时清空旧结果，防止后续把上一帧的框错误地套到当前图像。
                    last_detections = []
                    selected_target = None
                    last_target_status = f"YOLO failed: {exc}"

            # 冻结时重复显示抓取瞬间；实时状态下则在当前画面上绘制最新检测框和状态。
            if frozen and last_display is not None:
                display = last_display.copy()
            else:
                display_base = color_bgr
                if yolo_model is not None:
                    display_base = graspnet_utils.draw_detections_overlay(color_bgr, last_detections, selected_target, target_class)
                display = graspnet_utils.draw_status(
                    display_base,
                    f"LIVE {fps_value:.1f}fps | {status}",
                    last_target_status,
                    title="Main - GraspNet Grasp",
                )
            cv2.imshow(window_name, display)

            # waitKey() 同时刷新 OpenCV 窗口并读取键盘；& 0xFF 统一不同平台的键值。
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                # R 只恢复实时预览，不重新初始化相机、模型或机械臂。
                frozen = False
                last_display = None
                status = "live preview"
                continue

            if key in (ord("g"), ord("G"), ord(" ")):
                print("\n[G] Capture and run GraspNet")
                # 按键后重新取一组同步 RGB-D 帧，避免使用预览循环里已经过时的深度图。
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] Frame capture failed")
                    continue

                try:
                    # infer_frame() 内部完成：
                    # 1. RGB-D 反投影为点云并采样 num_point 个点；
                    # 2. 可选地按 YOLO 目标框限制点云区域；
                    # 3. GraspNet 生成候选，并执行碰撞、检测框和夹爪宽度过滤；
                    # 4. 返回最终候选、最高分结果以及用于 Open3D 显示的中间候选。
                    result = graspnet_utils.infer_frame(
                        net,
                        snap_color,
                        snap_depth,
                        K,
                        num_point=num_point,
                        min_depth=min_depth,
                        max_depth=max_depth,
                        collision_thresh=collision_thresh,
                        voxel_size=voxel_size,
                        yolo_model=yolo_model,
                        yolo_opts=yolo_opts,
                        target_class=target_class,
                        target_margin_px=target_margin_px,
                        target_expand_ratio=target_expand_ratio,
                        max_grasp_width_m=GRIPPER_MAX_DISTANCE_M,
                    )
                except Exception as exc:
                    status = f"inference failed: {exc}"
                    print(f"[G] {status}")
                    continue

                # 保存本次推理状态和检测结果，后面的冻结画面与日志统一使用这组数据。
                status = result.status
                last_target_status = result.target_status
                last_detections = result.detections
                selected_target = result.selected_target
                # 根据 open3d_grasps 选择显示最终、bbox 后或 bbox 前的候选集合。
                vis_grasps = graspnet_utils.visualization_grasps(result, open3d_grasps)

                print(f"[G] {status}")
                if open3d_enabled:
                    try:
                        # 第一次推理后才创建 Open3D 窗口，后续只更新点云和抓取几何体。
                        if vis is None:
                            vis = graspnet_utils.Open3DGraspWindow("GraspNet Grasps", top_k)
                        vis.update(result.o3d_cloud, vis_grasps)
                        print(f"[G] Open3D {open3d_grasps} candidates={len(vis_grasps)}")
                    except Exception as exc:
                        # Open3D 可视化失败不应影响二维预览和机械臂抓取，因此只关闭该窗口。
                        print(f"[G] Open3D failed: {exc}")
                        if vis is not None:
                            vis.close()
                            vis = None

                if result.best is None:
                    # 所有候选被碰撞、目标框或夹爪宽度过滤后，不能执行机械臂动作。
                    print("[G] No valid GraspNet grasp")
                    continue

                # 保存抓取瞬间，并在图像上保留当时的 YOLO 框和推理状态。
                frozen = True
                display_base = snap_color
                if yolo_model is not None:
                    display_base = graspnet_utils.draw_detections_overlay(snap_color, last_detections, selected_target, target_class)
                snap_display = graspnet_utils.draw_status(
                    display_base,
                    f"SNAPSHOT | {status}",
                    last_target_status,
                    frozen=True,
                    title="Main - GraspNet Grasp",
                )
                last_display = snap_display

                if T_hand_eye is None:
                    # 没有手眼矩阵时只能把相机坐标系中的抓取投影到图像，不能换算成机械臂目标。
                    graspnet_utils.draw_grasp_projections(snap_display, vis_grasps, K, top_k=top_k)
                    graspnet_utils.draw_best_grasp_projection(snap_display, result.best, K)
                    last_display = snap_display
                    print("[G] Hand-eye calibration unavailable")
                    continue

                # 眼在手上时：T_cam2base = 当前 TCP 在 base 下的位姿 × 相机相对 TCP 的手眼矩阵。
                # get_tcp_pose() 必须在抓取瞬间读取，因为相机会随夹爪一起移动。
                T_cam2base = compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)

                # 不直接执行 result.best。这里会从高分到低分逐个检查 base 高度和 IK，
                # 因此最高分不可达时，仍可能选择分数稍低但机械臂能够执行的候选。
                selected = _select_executable_grasp(
                    ik_checker,
                    result.grasps,
                    T_cam2base,
                    cfg,
                    pregrasp_offset_m,
                    retreat_offset_m,
                    insertion_depth_m,
                    min_base_z_m,
                )
                if selected is None:
                    # 没有候选通过安全高度和 IK 检查时保持冻结画面，不发送运动指令。
                    print(f"[G] No IK-reachable grasp above min_base_z={min_base_z_m:.3f}m ")
                    continue
                # best 用于显示；grasp/pre/retreat 是已经转换到 base 并应用补偿的执行位姿。
                best, grasp6d, pre6d, retreat6d = selected

                # 先更新二维调试图，再执行机械臂动作，便于从画面确认最终选择的是哪个候选。
                _print_grasp(best)
                graspnet_utils.draw_grasp_projections(snap_display, vis_grasps, K, top_k=top_k)
                graspnet_utils.draw_best_grasp_projection(snap_display, best, K)
                last_display = snap_display

                # 执行完整的开爪、接近、夹取、后撤、回待机位和释放流程。
                _execute_grasp(
                    controller,
                    grasp_driver,
                    grasp6d,
                    pre6d,
                    retreat6d,
                    ready_cfg,
                )

            # Open3D 需要在 OpenCV 主循环中持续处理窗口事件；用户关闭窗口后释放它。
            if vis is not None and not vis.poll():
                vis.close()
                vis = None

    finally:
        # 无论正常退出还是 Ctrl+C，都尽量释放夹爪、回 home 并断开。
        print("\n[Exit] Release gripper and home")
        try:
            # 只有夹爪和控制器都已成功启动时才发送释放命令。
            if robot_ready and grasp_driver is not None and controller is not None and getattr(controller, "_running", False):
                grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            # controller.end() 负责结束控制线程和断开连接；控制器未创建时直接断开底层机械臂。
            if controller is not None and getattr(controller, "_running", False):
                controller.end()
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            # 相机可能在 open() 前就初始化失败，因此关闭异常不能遮盖原始错误。
            cam.close()
        except Exception:
            pass
        # 最后关闭两个可视化窗口，释放 GUI 资源。
        if vis is not None:
            vis.close()
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == "__main__":
    try:
        # SystemExit 使用 main() 返回值作为进程退出码。
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Ctrl+C 仍会先触发 main() 中的 finally，再以 130 表示被中断。
        print("\nInterrupted.")
        raise SystemExit(130)
