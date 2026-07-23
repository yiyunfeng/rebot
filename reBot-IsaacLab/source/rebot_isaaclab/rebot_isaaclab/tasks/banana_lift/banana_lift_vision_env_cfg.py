"""在基础抓取环境上增加腕部 RGB-D 观测和视觉域随机化。

这里刻意继承状态环境，而不复制机器人和物理配置：视觉策略与状态策略看到
同一个机器人、香蕉、动作空间和奖励，只是观测形式不同，便于公平比较和迁移。
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from . import mdp
from .banana_lift_env_cfg import BananaEventCfg, RebotBananaLiftEnvCfg


@configclass
class VisionEventCfg(BananaEventCfg):
    """在动力学随机化之外，加入相机标定误差和照明随机化。"""

    # 不在 reset 里直接移动挂在 gripper_base 下的 Camera prim。
    # 多环境 TiledCamera + articulation 子节点 set_world_poses 容易卡在 reset，
    # 现阶段先把相机外参误差放到真机标定/数据增强侧处理，训练主线保证稳定并行。
    # DomeLight 是整个 stage 共享的灯，因此一次 reset 批次只采样一组光照，
    # 不能对同一个全局 prim 为每个并行环境设置互相冲突的属性。
    lighting = EventTerm(
        func=mdp.randomize_dome_light,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("light"),
            "intensity_range": (750.0, 1050.0),
            "color_range": (0.85, 1.0),
        },
    )


@configclass
class VisionObservationsCfg:
    """策略观测组：保留本体状态，同时提供独立的 RGB 与深度张量。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """提供给视觉策略的本体感知和 RGB-D 观测字段。"""

        # 关节位置、速度和上一时刻动作提供本体感知；视觉无法可靠推断这些量。
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        # normalize=True 使用 Isaac Lab 图像项的标准归一化，避免 uint8 数值尺度
        # 与关节状态相差数百倍。通道顺序保持 NHWC，由后续 CNN 适配器处理。
        rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "data_type": "rgb", "normalize": True},
        )
        # distance_to_image_plane 是沿相机光轴的深度，适合由像素反投影到 3D；
        # 它不同于到相机中心的欧氏距离 distance_to_camera。
        depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
            },
        )

        def __post_init__(self):
            """保留字典形式的多模态观测，避免自动展平图像。"""

            # 视觉域随机化由 EventManager 显式完成，这里不叠加默认观测噪声。
            self.enable_corruption = False
            # RGB-D 不展平或拼接成超长向量，后续 CNN 分支可按键读取图像，
            # MLP 分支则读取 joint_pos/joint_vel/actions。
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class VisionTrainingObservationsCfg:
    """RSL-RL 2.3.3 可直接存储的二维 RGB-D 训练观测。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """按“本体状态在前、展平 RGB-D 在后”的固定顺序拼接观测。"""

        # 明确只读取 6 个机械臂关节和主动左夹指，排除 mimic 驱动的右夹指。
        # 本体观测为 7 位置 + 7 速度 + 7 上一动作，共 21 维。
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]", "left_finger"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]", "left_finger"])},
        )
        actions = ObsTerm(func=mdp.last_action)
        # 必须放在最后：RgbdActorCritic 从观测尾部恢复图像。
        rgbd = ObsTerm(
            func=mdp.flattened_rgbd,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "depth_limit_m": 1.5},
        )

        def __post_init__(self):
            """拼接为 RSL-RL 要求的 ``(num_envs, observation_dim)`` Tensor。"""
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RebotBananaLiftVisionEnvCfg(RebotBananaLiftEnvCfg):
    """RGB-D 训练环境；动作、奖励及终止条件继承自状态环境。"""

    observations: VisionObservationsCfg = VisionObservationsCfg()
    events: VisionEventCfg = VisionEventCfg()

    def __post_init__(self):
        """在状态任务上挂接已有腕部相机，并设置视觉更新频率。"""

        super().__post_init__()

        # 相机 prim 已随 rebotarm_dm_with_camera.usd 挂在末端连杆上。spawn=None
        # 表示 TiledCamera 只绑定并读取该相机，不在相同路径重复创建 Camera prim。
        self.scene.wrist_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/gripper_base/dabai_dcw_camera",
            update_period=0.05,  # 20 Hz，与真实 RGB-D 推理频率更接近。
            height=128,
            width=128,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=None,
            depth_clipping_behavior="zero",
        )
        # 不在 reset 内强制重渲染。TiledCamera 多环境时 reset 里同步等待 RTX
        # 首帧容易卡在 “raw env reset ...”；训练入口会在 reset 后先 step 几次
        # 做 warmup，让相机在正常仿真步中更新首帧。
        self.rerender_on_reset = False
        # Isaac Lab 默认会在 reset 结束等待纹理/资产加载完成；多环境 RTX 相机
        # 场景下该状态可能长时间不结束，表现为卡在 raw env reset。训练无需
        # 在 reset 阶段等待贴图完全就绪，后续 warmup/step 会推进渲染管线。
        self.wait_for_textures = False
        self.sim.render_interval = 20  # 400 Hz physics / 20 = 20 Hz camera。
        # 关闭抗锯齿以降低大量并行相机的 RTX 成本；训练随机化可覆盖边缘差异。
        self.sim.render.antialiasing_mode = "OFF"
        # RGB-D 张量显存开销较大，默认并行数保守降至 32。
        self.scene.num_envs = 32


@configclass
class RebotBananaLiftVisionTrainEnvCfg(RebotBananaLiftVisionEnvCfg):
    """供端到端 RGB-D PPO 使用的低分辨率训练环境。"""

    observations: VisionTrainingObservationsCfg = VisionTrainingObservationsCfg()

    def __post_init__(self):
        """将相机降为 64×64，并使用适合单张 16 GB GPU 的并行数。"""
        super().__post_init__()
        # 64×64 RGB-D 每环境为 16384 个 float；相比 128×128 减少 75% rollout 显存。
        self.scene.wrist_camera.height = 64
        self.scene.wrist_camera.width = 64
        self.scene.num_envs = 16
