"""基于 Isaac Lab Manager API 的 reBot 香蕉抓取/抬升基础环境。

RGB-D 主线在此配置基础上扩展视觉观测，复用同一套机器人、物理场景、
动作、奖励和成功判据，避免训练/评估/部署之间出现两套不同的动力学定义。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import (
    EventCfg as LiftEventCfg,
    LiftEnvCfg,
    RewardsCfg as LiftRewardsCfg,
)

from ...assets import BANANA_USD_PATH, REBOT_DM_CFG
from . import mdp


# 所有高度均为每个并行环境局部坐标系中的米制高度。
TABLE_TOP_Z = 0.0
BANANA_REST_Z = 0.019  # 香蕉质心静置于桌面时的初始高度。
SUCCESS_HEIGHT = 0.070  # 质心超过 7 cm 即记为一次成功抬升。
RETURN_REWARD_HEIGHT = 0.045
HOME_REWARD_STD = 0.35
HOME_JOINT_TOLERANCE = 0.12
MAX_OBJECT_EE_DISTANCE = 0.12


@configclass
class BananaRewardsCfg(LiftRewardsCfg):
    """抓取任务奖励：稠密引导负责学习过程，稀疏奖励定义最终成功。

    距离和抬升项负责学会抓取；returning_home 在夹住物体后引导六轴回到
    ready pose；success 同时检查物体高度、物体与夹爪距离和 ready 关节误差。
    """

    # 鼓励末端先接近香蕉。tanh 型距离核的 std 越小，奖励有效范围越窄。
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.08}, weight=2.0)
    # 香蕉离开桌面后给较大权重，防止策略只学会接近而不闭合夹爪。
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.045}, weight=12.0)
    # 明确关闭官方 Lift 的随机目标跟踪；本任务不训练放置或移动到放置点。
    object_goal_tracking = None
    object_goal_tracking_fine_grained = None
    returning_home = RewTerm(
        func=mdp.return_home_with_object,
        params={
            "minimum_height": RETURN_REWARD_HEIGHT,
            "home_std": HOME_REWARD_STD,
            "maximum_object_ee_distance": MAX_OBJECT_EE_DISTANCE,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
        },
        weight=10.0,
    )
    # 达标后持续给奖励，使策略学会在 ready 姿态稳定保持香蕉直到 episode 结束。
    success = RewTerm(
        func=mdp.grasp_return_success,
        params={
            "minimum_height": SUCCESS_HEIGHT,
            "home_joint_tolerance": HOME_JOINT_TOLERANCE,
            "maximum_object_ee_distance": MAX_OBJECT_EE_DISTANCE,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
        },
        weight=20.0,
    )


@configclass
class BananaEventCfg(LiftEventCfg):
    """RGB-D 抓取训练使用的动力学域随机化。

    当前阶段先固定机械臂、桌面、香蕉和夹爪的物理参数，只随机香蕉位置和
    光照。这样网络主要学习“RGB-D + joint state -> 动作”的映射，不被接触
    参数随机化干扰。等基本抓取稳定后，再逐步加回小范围物理随机化。
    """

    object_material = None
    object_mass = None
    finger_material = None
    actuator_gains = None
    # 机械臂默认 ready pose 已写在 REBOT_DM_CFG.init_state 中。
    # 不再 reset 时调用官方 reset_joints_by_offset：该函数会在多环境
    # RGB-D 训练启动阶段写入 articulation joint state，当前环境会卡在这里。
    # 起始姿态扰动后续如确有需要，应写项目内小函数并逐关节安全赋值。
    reset_arm_near_ready = None


@configclass
class RebotBananaLiftEnvCfg(LiftEnvCfg):
    """使用相对差分 IK 动作空间的 PPO 训练环境。"""

    rewards: BananaRewardsCfg = BananaRewardsCfg()
    events: BananaEventCfg = BananaEventCfg()

    def __post_init__(self):
        """在官方 Lift 默认配置上覆盖 reBot 资产、场景和控制参数。"""

        # 先让官方 LiftEnvCfg 建立默认场景、观测、终止项和命令，再针对 reBot
        # 覆盖资产及参数；若省略这一行，继承配置中的多个字段不会被初始化。
        super().__post_init__()

        # 物理仿真 400 Hz；同一个策略动作连续执行 decimation=10 个物理步，
        # 因此 PPO 决策频率为 40 Hz。渲染只需跟随策略步，无需每个物理步渲染。
        self.sim.dt = 1.0 / 400.0
        self.decimation = 10
        self.sim.render_interval = self.decimation
        # 与 reBot-Isaacsim/config/dm_sim.yaml 的 PhysX 场景开关一致。
        self.sim.physx.enable_ccd = True
        self.sim.physx.enable_stabilization = True
        self.episode_length_s = 8.0
        self.scene.num_envs = 256
        self.scene.env_spacing = 1.2

        # {ENV_REGEX_NS} 会展开成 /World/envs/env_.*/，使同一配置可复制到数百环境。
        self.scene.robot = REBOT_DM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.32, 0.0, BANANA_REST_Z), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(BANANA_USD_PATH),
                # 外观、质量、摩擦和 convexDecomposition 碰撞均与现有
                # reBot-Isaacsim 香蕉一致；这里只覆盖求解器稳定性参数。
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=2,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
            ),
        )

        # 使用本地简单长方体桌面替代在线 Nucleus 资产，使训练可离线复现。
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.0, -0.02)),
            spawn=sim_utils.CuboidCfg(
                size=(0.90, 0.70, 0.04),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
                # 与 reBot-Isaacsim/isaacsim_joint_receiver.py 的桌面颜色一致。
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.46, 0.34)),
            ),
        )
        # 地面置于桌面底部，仅作物体跌落后的兜底碰撞，不取代桌面碰撞体。
        self.scene.plane.init_state.pos = (0.0, 0.0, -0.04)
        self.scene.light.spawn = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=900.0)

        # 官方 reset_scene_to_default 默认只把关节状态瞬间写到 ready pose，
        # 不会同步隐式 PD actuator 的位置目标。此时旧目标（首次启动时为 0）
        # 会在下一个物理步重新生效，表现为机械臂刚到 ready 就下垂或摆动。
        # 同步 reset 位置目标后，机械臂会保持 REBOT_DM_CFG 中的初始关节姿态；
        # 后续 Differential IK 动作到来时仍会正常覆盖该目标。
        self.events.reset_all.params = {"reset_joint_targets": True}

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            body_name="gripper_base",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            # 网络输出通常限制在 [-1, 1]。因此单个 40 Hz 策略步最大平移
            # 2 cm、旋转 0.10 rad；IK 再将末端增量转换为 6 个关节目标。
            scale=(0.02, 0.02, 0.02, 0.10, 0.10, 0.10),
            # 与传统抓取代码一致：真正的抓取控制点不是 gripper_base 原点，
            # 而是沿夹爪局部 +X 前移 1.5 cm，约位于两夹指的有效夹持区域。
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.015, 0.0, 0.0)),
        )
        # 策略第 7 维只决定“开/关”，不直接预测任意夹爪距离。右夹指由 USD
        # mimic joint 跟随，所以这里只控制 left_finger，单位为米。
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_finger"],
            open_command_expr={"left_finger": 0.045},
            close_command_expr={"left_finger": 0.001},
        )

        # 视觉观测和奖励都不再使用随机目标，直接禁用官方 Lift 的 command term。
        self.commands.object_pose = None

        # FrameTransformer 提供奖励/观测所需的末端世界位姿。prim 路径必须与
        # 原 USD 层级严格一致，不能按常见 URDF 层级自行猜测。
        self.scene.ee_frame = FrameTransformerCfg(
            # 外层 base_link 是 articulation 容器；内层同名 base_link 才实际带
            # RigidBodyAPI，FrameTransformer 的源 frame 必须选后者。
            prim_path="{ENV_REGEX_NS}/Robot/base_link/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/base_link/gripper_base",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.015, 0.0, 0.0)),
                )
            ],
        )

        # 每个 episode 重置时随机香蕉平面位置和朝向。这里的 x/y 是叠加在
        # init_state.pos=(0.32, 0.0, BANANA_REST_Z) 上的偏移量，因此
        # x=(0.03, 0.11) 对应最终世界坐标约 0.35~0.43 m。
        self.events.reset_object_position.params["pose_range"] = {
            "x": (0.03, 0.11),
            "y": (-0.04, 0.04),
            "z": (0.0, 0.0),
            "yaw": (-0.35, 0.35),
        }
        self.events.reset_object_position.params["asset_cfg"] = SceneEntityCfg("object")

        # 香蕉质心低于桌面 2 cm 视为掉落并提前结束，节省无意义仿真步。
        self.terminations.object_dropping.params["minimum_height"] = -0.02
        # 动作变化率和关节速度惩罚在前 2 万步逐渐加入，避免训练早期过强的
        # 平滑约束阻碍探索抓取动作。
        self.curriculum.action_rate.params["num_steps"] = 20_000
        self.curriculum.joint_vel.params["num_steps"] = 20_000
