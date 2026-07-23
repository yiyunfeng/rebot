"""Compatibility entry for the old camera_grasp_pipeline command.

新代码请直接使用：
    ros2 run rebotarm_gazebo camera_grasp_sim
    ros2 run rebotarm_gazebo camera_grasp_hardware

这个文件只保留旧命令入口，抓取逻辑已经拆到两个独立文件，避免仿真和真机参数混在一起。
"""

from __future__ import annotations

import sys


def _is_hardware_mode(argv: list[str]) -> bool:
    """兼容旧命令里 -p mode:=hardware 的写法。"""
    return any("mode:=hardware" in arg or "mode:=real" in arg for arg in argv)


def main(args: list[str] | None = None) -> None:
    argv = sys.argv if args is None else args
    if _is_hardware_mode(list(argv)):
        from rebotarm_gazebo.camera_grasp_hardware import main as hardware_main

        hardware_main(args)
        return

    from rebotarm_gazebo.camera_grasp_sim import main as sim_main

    sim_main(args)


if __name__ == "__main__":
    main()
