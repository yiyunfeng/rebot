"""
MoveIt 启动参数构建函数。

将 MoveItConfigs 对象转换为 move_group 节点可用的参数字典。
供 gazebo.launch.py 使用。
"""

import os


def moveit_parameters(moveit_config) -> dict:
    """将 MoveItConfigs 转换为 move_group 节点参数字典。

    参数包括：robot_description, robot_description_semantic,
    kinematics, joint_limits, ompl 规划配置等。

    Humble 兼容处理：
        ROS 2 Humble 的 MoveIt 使用旧的 request_adapters API，
        需要手动指定适配器列表并移除 response_adapters。
    """
    params = moveit_config.to_dict()

    # 配置 OMPL 运动规划器
    ompl = params.setdefault("ompl", {})
    ompl["planning_plugin"] = "ompl_interface/OMPLPlanner"

    # Humble 版本兼容
    if os.environ.get("ROS_DISTRO") == "humble":
        ompl["request_adapters"] = " ".join([
            "default_planner_request_adapters/AddTimeOptimalParameterization",
            "default_planner_request_adapters/ResolveConstraintFrames",
            "default_planner_request_adapters/FixWorkspaceBounds",
            "default_planner_request_adapters/FixStartStateBounds",
            "default_planner_request_adapters/FixStartStateCollision",
        ])
        # Humble 不支持 response_adapters，删除避免报错
        ompl.pop("response_adapters", None)

    return params
