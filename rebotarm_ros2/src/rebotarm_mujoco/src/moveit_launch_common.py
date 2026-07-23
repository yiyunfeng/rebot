"""MoveIt 参数兼容处理。

launch 文件会从已安装的 Python 包中导入这个模块。
保留为独立小文件，是为了让 MuJoCo 启动和 Gazebo 启动各自维护配置，
不互相依赖。
"""

import os


def moveit_parameters(moveit_config) -> dict:
    """把 MoveItConfigs 转成 move_group 可直接使用的参数字典。"""
    params = moveit_config.to_dict()

    # 明确使用 OMPL。不同 MoveIt 版本对 YAML 字段兼容性不完全一样，
    # 这里集中处理，launch 文件里就不用散落版本判断。
    ompl = params.setdefault("ompl", {})
    ompl["planning_plugin"] = "ompl_interface/OMPLPlanner"

    if os.environ.get("ROS_DISTRO") == "humble":
        # Humble 的 MoveIt 还使用旧 request_adapters 名称；
        # response_adapters 在 Humble 中会导致参数解析异常，所以移除。
        ompl["request_adapters"] = " ".join([
            "default_planner_request_adapters/AddTimeOptimalParameterization",
            "default_planner_request_adapters/ResolveConstraintFrames",
            "default_planner_request_adapters/FixWorkspaceBounds",
            "default_planner_request_adapters/FixStartStateBounds",
            "default_planner_request_adapters/FixStartStateCollision",
        ])
        ompl.pop("response_adapters", None)

    return params
