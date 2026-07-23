"""汇总官方 Lift MDP 项和 reBot 香蕉任务自定义项。

环境配置通过 ``from . import mdp`` 使用统一命名空间：通用的关节观测、动作、
距离奖励来自 Isaac Lab，项目只新增成功判据以及相机/光照随机化。
"""

# 复用官方 Lift 任务的 observation、reward、event 和 termination 函数，避免复制实现。
from isaaclab_tasks.manager_based.manipulation.lift.mdp import *  # noqa: F401,F403

# 以下三个符号是项目自己实现并显式维护的扩展点。
from .rewards import grasp_return_success, return_home_with_object
from .events import randomize_dome_light, randomize_wrist_camera
from .observations import flattened_rgbd

# __all__ 仅列出项目新增项；官方 wildcard 项仍可由环境配置按属性访问。
__all__ = [
    "flattened_rgbd",
    "grasp_return_success",
    "randomize_dome_light",
    "randomize_wrist_camera",
    "return_home_with_object",
]
