"""坐标变换辅助函数。"""
import numpy as np

_ROT_X_PI = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


def _nearest_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """把近似旋转矩阵投影到合法的三维旋转矩阵集合 SO(3)。"""
    # 坐标计算和浮点误差可能让 R 不再严格正交，先统一为 float64 并检查形状。
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"rotation matrix must be (3, 3), got {R.shape}")

    if not np.all(np.isfinite(R)):
        raise ValueError("rotation matrix contains non-finite values")

    # SVD 投影：U @ Vt 是距离输入矩阵最近的正交矩阵。
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    # det=-1 表示发生了镜像反射；翻转最后一列使其回到合法旋转群 SO(3)。
    if np.linalg.det(R_ortho) < 0.0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho.astype(np.float64)


def pose6d_to_mat4(x, y, z, rx, ry, rz, degrees=False) -> np.ndarray:
    """把 6D 位姿转换为 4×4 齐次变换矩阵。

    参数：
        x, y, z：平移量，单位为米。
        rx, ry, rz：内禀 ZYX 欧拉角，依次对应 roll、pitch、yaw。
        degrees：为 True 时角度单位是度；为 False 时单位是弧度。

    返回：
        T：形状为 (4, 4) 的 NumPy 数组。
    """
    if degrees:
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

    # 绕 X 轴旋转。
    Rx = np.array([
        [1,          0,           0],
        [0,  np.cos(rx), -np.sin(rx)],
        [0,  np.sin(rx),  np.cos(rx)],
    ])
    # 绕 Y 轴旋转。
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    # 绕 Z 轴旋转。
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [         0,           0, 1],
    ])

    # 内禀 ZYX 旋转的矩阵乘法顺序：R = Rz @ Ry @ Rx。
    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def quat_to_mat4(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    """把平移和四元数转换为 4×4 齐次变换矩阵。

    参数：
        x, y, z：平移量，单位为米。
        qx, qy, qz, qw：Hamilton 形式的四元数。

    返回：
        T：形状为 (4, 4) 的 NumPy 数组。
    """
    # 四元数只有方向有意义，先归一化，避免缩放误差破坏旋转矩阵。
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [  2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [  2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)

    # 齐次矩阵左上 3×3 存旋转，最后一列前三项存平移。
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def mat4_to_pose6d(T: np.ndarray) -> tuple:
    """把 4×4 变换矩阵转换为 (x, y, z, rx, ry, rz)，角度单位为弧度。"""
    # 第四列是平移；左上角旋转矩阵再转换为 roll/pitch/yaw。
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    rpy = rotation_matrix_to_euler_zyx(T[:3, :3])
    return float(x), float(y), float(z), float(rpy[0]), float(rpy[1]), float(rpy[2])


def execution_compensation_xyz(cfg: dict) -> np.ndarray:
    """读取机械臂自身执行误差补偿，单位为米，作用在机器人 base 坐标系。

    这个补偿只用于修正机械臂执行侧的固定偏差，不参与相机内参、手眼标定、
    物体检测或抓取姿态估计。配置值应填写“发给机械臂目标位姿的修正量”：
    如果实测 TCP 总是比目标高 2cm，就把 z 写成 -0.02。
    """
    # 使用 or {} 同时兼容配置节缺失和显式写成 null 的情况。
    robot_cfg = cfg.get("robot") or {}
    compensation = robot_cfg.get("execution_compensation_base_m") or {}
    return np.array(
        [
            float(compensation.get("x", 0.0)),
            float(compensation.get("y", 0.0)),
            float(compensation.get("z", 0.0)),
        ],
        dtype=np.float64,
    )


def execution_compensation_z_by_x(x: float, cfg: dict) -> float:
    """读取只随目标 x 变化的 z 线性补偿，单位米。

    配置形式：
        robot.execution_compensation_z_by_x.enabled: true/false
        robot.execution_compensation_z_by_x.reference_x: 参考 x
        robot.execution_compensation_z_by_x.slope: z 补偿关于 x 的斜率

    最终额外 z 补偿：
        dz = slope * (target_x - reference_x)

    该补偿用于机械臂自身随伸展距离变化的高度误差；只改最终发给真机的 z，
    不参与相机内参、手眼标定、视觉检测或抓取姿态估计。
    """
    robot_cfg = cfg.get("robot") or {}
    linear_cfg = robot_cfg.get("execution_compensation_z_by_x") or {}
    if not bool(linear_cfg.get("enabled", False)):
        return 0.0

    # 以 reference_x 为零点：目标伸得比参考位置更远/更近时，按 slope 线性修正高度。
    reference_x = float(linear_cfg.get("reference_x", 0.0))
    slope = float(linear_cfg.get("slope", 0.0))
    dz = slope * (float(x) - reference_x)
    if not np.isfinite(dz):
        raise ValueError("robot.execution_compensation_z_by_x produced non-finite z offset")
    return float(dz)


def execution_compensation_y_radial(y: float, cfg: dict) -> float:
    """读取 y 方向远离中心线的对称补偿，单位米。

    适用现象：
        target_y > 0 时，真机实际 y 比目标更靠近 0；
        target_y < 0 时，真机实际 y 也比目标更靠近 0。

    补偿逻辑：
        y > deadband  -> +offset
        y < -deadband -> -offset
        其他           -> 0

    这不是固定 y 偏移，而是按目标 y 的符号把目标点往外推。
    """
    robot_cfg = cfg.get("robot") or {}
    radial_cfg = robot_cfg.get("execution_compensation_y_radial") or {}
    if not bool(radial_cfg.get("enabled", False)):
        return 0.0

    # deadband 内不补偿，避免目标在中心线附近因符号变化而左右跳动。
    deadband = abs(float(radial_cfg.get("deadband", 0.0)))
    offset = abs(float(radial_cfg.get("offset", 0.0)))
    y = float(y)
    if y > deadband:
        return offset
    if y < -deadband:
        return -offset
    return 0.0


def apply_execution_compensation_to_pose(pose6d: tuple[float, ...], cfg: dict) -> tuple[float, ...]:
    """对 6D TCP 位姿应用 base 坐标系下的机械臂执行补偿。

    只平移 x/y/z，不改 roll/pitch/yaw。这样可以保持视觉算法给出的夹爪姿态，
    同时对机械臂自身固定落点误差、随 x 变化的高度误差和 y 对称缩回误差做修正。
    """
    if len(pose6d) != 6:
        raise ValueError(f"pose6d must contain 6 values, got {len(pose6d)}")
    # 固定 xyz 偏移、随 x 变化的 z 偏移、随 y 符号变化的径向偏移分别计算。
    offset = execution_compensation_xyz(cfg)
    if not np.all(np.isfinite(offset)):
        raise ValueError("robot.execution_compensation_base_m contains non-finite value")
    x, y, z, rx, ry, rz = pose6d
    linear_z = execution_compensation_z_by_x(float(x), cfg)
    radial_y = execution_compensation_y_radial(float(y), cfg)
    # 姿态 rx/ry/rz 原样保留，只修正最终发送给机械臂的位置。
    return (
        float(x + offset[0]),
        float(y + offset[1] + radial_y),
        float(z + offset[2] + linear_z),
        float(rx),
        float(ry),
        float(rz),
    )


def rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """把旋转矩阵转换为内禀 ZYX 欧拉角。"""
    # 先投影到合法旋转矩阵，避免 arcsin/atan2 被微小数值误差影响。
    R = _nearest_rotation_matrix(R)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        # 常规情况：pitch 不在万向节锁附近，可分别求出三个角。
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        # sy≈0 表示 pitch≈±90°，roll 与 yaw 耦合；固定 yaw=0 选择一个稳定解。
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def canonicalize_parallel_gripper_tcp_rotation(R: np.ndarray) -> np.ndarray:
    """为平行夹爪选择更稳定的等价 TCP 姿态。

    对称平行夹爪绕工具 X 轴旋转 180° 后，通常仍具有相同的抓取效果。
    在 ``R`` 和 ``R @ Rx(pi)`` 两个等价姿态中，选择绝对 roll 更小的一个，
    使输出的 RPY 更稳定，也更便于查看和调试。
    """
    R = _nearest_rotation_matrix(R)
    # 平行夹爪绕工具 X 轴转 180° 后，两根夹指交换位置，但抓取效果等价。
    alt = R @ _ROT_X_PI

    # 在两个等价解中选择绝对 roll 更小的一个，使日志和 IK 初值更稳定易读。
    roll = float(rotation_matrix_to_euler_zyx(R)[0])
    alt_roll = float(rotation_matrix_to_euler_zyx(alt)[0])
    return alt if abs(alt_roll) < abs(roll) else R


def grasp_axes_to_rebot_tcp_rotation(
    grip_axis: np.ndarray,
    open_axis: np.ndarray,
    approach_axis: np.ndarray,
) -> np.ndarray:
    """把视觉抓取坐标轴映射到 reBotArm 的 TCP 坐标系。

    视觉抓取坐标系约定：
      - X = grip_axis，抓取方向
      - Y = open_axis，夹爪开合方向
      - Z = approach_axis，接近方向

    reBotArm TCP 坐标系约定：
      - X = 工具向前接近物体的方向
      - Y = 夹爪开合方向
      - Z = 按右手定则补出的方向
    """
    grip = np.asarray(grip_axis, dtype=np.float64)
    open_vec = np.asarray(open_axis, dtype=np.float64)
    approach = np.asarray(approach_axis, dtype=np.float64)

    # 三个输入轴先单位化；max(..., 1e-8) 防止异常零向量直接除零。
    grip /= max(np.linalg.norm(grip), 1e-8)
    open_vec /= max(np.linalg.norm(open_vec), 1e-8)
    approach /= max(np.linalg.norm(approach), 1e-8)

    # TCP X 轴是工具向前、进入物体的方向，在基座坐标系中通常表现为向下。
    # approach 指向相机，方向与工具前进方向相反，所以这里取负号。
    tcp_x = -approach
    # 从 open_vec 中减掉沿 tcp_x 的分量，保证 TCP X/Y 两轴互相垂直。
    tcp_y = open_vec - float(np.dot(open_vec, tcp_x)) * tcp_x
    tcp_y /= max(np.linalg.norm(tcp_y), 1e-8)
    # 右手定则由 X×Y 生成 Z，得到完整的正交 TCP 坐标系。
    tcp_z = np.cross(tcp_x, tcp_y)
    tcp_z /= max(np.linalg.norm(tcp_z), 1e-8)

    # 翻转接近轴后，保证 TCP Z 轴仍与原始抓取方向 grip 保持同向。
    if float(np.dot(tcp_z, grip)) < 0.0:
        tcp_y = -tcp_y
        tcp_z = -tcp_z

    # 旋转矩阵的每一列表示对应 TCP 轴在外部坐标系中的方向。
    R = np.column_stack([tcp_x, tcp_y, tcp_z]).astype(np.float64)
    if np.linalg.det(R) < 0.0:
        R[:, 2] *= -1.0
    return R


def grasp_rotation_to_rebot_tcp_rotation(grasp_rotation: np.ndarray) -> np.ndarray:
    """把列顺序为 [grip, open, approach] 的旋转矩阵转换为 reBotArm TCP 姿态。"""
    R = np.asarray(grasp_rotation, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"grasp_rotation must be (3, 3), got {R.shape}")
    return grasp_axes_to_rebot_tcp_rotation(R[:, 0], R[:, 1], R[:, 2])


def _make_grasp_base_transform(
    position_cam: np.ndarray,
    tcp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
) -> np.ndarray:
    """把相机系抓取位姿左乘 ``T_cam2base``，得到基座系齐次矩阵。"""
    # 先把相机系中的 position + rotation 组装成完整抓取位姿 T_grasp_cam。
    T_grasp_cam = np.eye(4, dtype=np.float64)
    T_grasp_cam[:3, :3] = np.asarray(tcp_rotation_cam, dtype=np.float64)
    T_grasp_cam[:3, 3] = np.asarray(position_cam, dtype=np.float64)

    # 变换链：grasp -> camera -> base，因此矩阵顺序为 T_cam2base @ T_grasp_cam。
    T_grasp_base = np.asarray(T_cam2base, dtype=np.float64) @ T_grasp_cam
    T_grasp_base[:3, :3] = canonicalize_parallel_gripper_tcp_rotation(T_grasp_base[:3, :3])
    return T_grasp_base


def _offset_along_tool_x(T: np.ndarray, offset_m: float) -> np.ndarray:
    """沿工具 X 轴反方向平移；正值用于从抓取点退到预抓取点。"""
    # copy() 保留原抓取矩阵；T[:3, 0] 是工具 X 轴在 base 中的单位方向。
    T_offset = T.copy()
    T_offset[:3, 3] = T[:3, 3] - T[:3, 0] * float(offset_m)
    return T_offset


def transform_grasp_pose_to_base(
    position_cam: np.ndarray,
    tcp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    insertion_depth_m: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """把相机坐标系抓取位姿转换为基座坐标系的抓取和预抓取位姿。"""
    T_grasp_base = _make_grasp_base_transform(position_cam, tcp_rotation_cam, T_cam2base)
    # insertion_depth 为正时沿工具前进方向深入，因此传入负 offset。
    T_grasp_base = _offset_along_tool_x(T_grasp_base, -insertion_depth_m)
    # pregrasp 从最终抓取点沿工具 X 轴反向退出，供机械臂直线接近。
    T_pregrasp_base = _offset_along_tool_x(T_grasp_base, pregrasp_offset_m)
    return mat4_to_pose6d(T_grasp_base), mat4_to_pose6d(T_pregrasp_base)


def transform_grasp_pose_to_base_with_retreat(
    position_cam: np.ndarray,
    tcp_rotation_cam: np.ndarray,
    T_cam2base: np.ndarray,
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    insertion_depth_m: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """把相机系抓取位姿转换为基座系的抓取、预抓取和后撤位姿。"""
    T_grasp_base = _make_grasp_base_transform(position_cam, tcp_rotation_cam, T_cam2base)
    T_grasp_base = _offset_along_tool_x(T_grasp_base, -insertion_depth_m)
    # 预抓取和抓取后撤都以最终抓取位姿为基准，但可配置不同距离。
    T_pregrasp_base = _offset_along_tool_x(T_grasp_base, pregrasp_offset_m)
    T_retreat_base = _offset_along_tool_x(T_grasp_base, retreat_offset_m)
    return mat4_to_pose6d(T_grasp_base), mat4_to_pose6d(T_pregrasp_base), mat4_to_pose6d(T_retreat_base)


def graspnet_rotation_to_rebot_tcp_rotation(grasp_rotation: np.ndarray) -> np.ndarray:
    """把 GraspNet 的 rotation_matrix 转换为 reBotArm TCP 姿态。"""
    R = np.asarray(grasp_rotation, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"grasp_rotation must be (3, 3), got {R.shape}")

    return _nearest_rotation_matrix(np.column_stack([R[:, 0], R[:, 1], np.cross(R[:, 0], R[:, 1])]))
