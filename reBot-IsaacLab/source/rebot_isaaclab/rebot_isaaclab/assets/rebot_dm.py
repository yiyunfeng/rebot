"""DM reBot 机械臂及香蕉资产的公共路径和 Isaac Lab 配置。

本文件只描述“机械臂如何被加载、初始关节状态是什么、由哪些执行器驱动”。
任务目标、奖励和随机化不放在这里，避免同一机器人配置在不同任务中产生差异。
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


# 从当前包位置反推 reBot-IsaacLab 根目录，避免依赖启动时的工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[4]
REBOT_USD_PATH = PROJECT_ROOT / "assets" / "rebotarm_lab.usda"
BANANA_USD_PATH = PROJECT_ROOT / "assets" / "banana_physics.usda"


REBOT_DM_CFG = ArticulationCfg(
    # 原始 USD 中不只一个 prim 带 ArticulationRootAPI。若不明确指定，Isaac Lab
    # 可能选中相机场景里的 /worldBody，而不是机械臂，随后会找不到关节和刚体。
    articulation_root_prim_path="/base_link",
    spawn=sim_utils.UsdFileCfg(
        # 这里加载的是项目内的 wrapper USD；wrapper 保留原模型层级，同时修正
        # Isaac Lab 所需的 articulation root，训练不依赖在线 Nucleus 资产。
        usd_path=str(REBOT_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # 保留真实重力；机械臂必须依靠下面的 position drive 保持姿态，
            # 不能通过关闭重力来掩盖 actuator 配置或控制目标错误。
            disable_gravity=False,
            # 限制接触穿透后的最大修正速度，减少仿真初始重叠时的剧烈弹飞。
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 当前抓取任务不依赖机械臂自碰撞；关闭可降低并行环境的计算量。
            enabled_self_collisions=False,
            # 与 dm_sim.yaml 的 TGS solver 迭代次数保持一致。
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # 与现有 Isaac Sim 传统抓取代码使用同一 ready joint pose，单位为 rad。
            # 训练和真机使用相同起始构型，可减少 sim2real 的姿态分布差异。
            "joint1": -0.00000847,
            "joint2": -0.44618440,
            "joint3": -0.71451218,
            "joint4": 0.96833512,
            "joint5": -0.00000885,
            "joint6": 0.00009368,
            # left_finger 是平移关节，单位为 m；0.040 表示夹爪基本张开。
            "left_finger": 0.040,
            # right_finger 不单独发送动作，它在 USD 中通过 PhysX mimic joint
            # 反向跟随左指。若同时控制左右指，反而可能与 mimic 约束冲突。
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            # rebotarm_dm_with_camera.usd 与原 Isaac Sim 抓取流程使用同一个
            # rebotarm_dm.usd；其中的 position drive 已由 dm_sim.yaml 调好。
            # None 表示不在 Isaac Lab 中覆盖它，直接沿用 USD 的 stiffness、
            # damping、max force 和 velocity limit，确保两套仿真动力学一致。
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger"],
            # 同样沿用 USD 中经过抓取验证的夹爪 drive（含 armature/friction）。
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
    },
    # 策略动作只能使用硬限位内侧 95% 的范围，给数值误差和真机限位留余量。
    soft_joint_pos_limit_factor=0.95,
)
