"""让真机机械臂回到 home 后失能。

这个脚本不打开相机，不运行 YOLO/GraspNet，只使用 reBotArm_control_py。
用于 main.py 异常退出、ROS2 服务不可用、或机械臂已经使能但需要安全收尾的场景。

用法:
    cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp
    ./scripts/run_home_disable.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.robot.grasp_driver import selected_arm_config
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose


def parse_args() -> argparse.Namespace:
    """解析回 home 的速度、超时和是否重新初始化电机模式。"""
    parser = argparse.ArgumentParser(description="Safe home then disable the real reBotArm")
    parser.add_argument("--max-vel", type=float, default=0.35, help="safe_home 最大关节速度，越小越慢越稳")
    parser.add_argument("--timeout", type=float, default=25.0, help="safe_home 超时时间，单位秒")
    parser.add_argument(
        "--force-mode-init",
        action="store_true",
        help="强制重新写入电机模式和 PID 参数；默认复用当前模式，避免 main.py 后再次触发 ACK 超时",
    )
    return parser.parse_args()


def start_without_mode_init(controller: RebotArmEndPose, arm: RebotArm) -> None:
    """复用电机当前模式启动控制循环。

    main.py 已经把 arm 组切到 POS_VEL、gripper 切到 MIT。此时如果再调用
    controller.start()，会重新批量写 register 25-28 和 register 10；当串口桥
    刚被上一轮程序释放、或总线正忙时，这一步最容易出现 write ack timeout。

    因此收尾脚本默认只做三件事：
      1. 重新连接并读取当前关节角；
      2. 将当前角作为控制目标，避免控制循环启动瞬间跳变；
      3. 不控制夹爪，只用 arm 组 POS_VEL 命令回 home。
    """
    # 只建立通信并读取当前角度，不重复写电机工作模式和 PID 寄存器。
    arm.connect()
    q_now, _, _ = arm.get_state()
    controller._has_gripper = False
    # 控制循环的初始目标设为当前角，避免启动瞬间产生位置跳变。
    controller._q_target[:] = q_now[: controller._n]
    controller._qd_target[:] = 0.0
    arm.start_control_loop(controller._loop_cb, rate=100.0)
    controller._running = True


def main() -> int:
    """连接机械臂，安全回 home，随后失能并断开通信。"""
    args = parse_args()

    selected = selected_arm_config(None)
    arm = RebotArm()
    controller = RebotArmEndPose(arm, arm_control_mode=selected.controller_mode)

    try:
        # 默认复用当前电机模式以减少寄存器写入；只有明确指定时才完整初始化模式。
        if args.force_mode_init:
            print(f"[HomeDisable] start controller with mode init, mode={selected.controller_mode}")
            controller.start()
        else:
            print(f"[HomeDisable] start controller without mode init, mode={selected.controller_mode}")
            start_without_mode_init(controller, arm)

        print("[HomeDisable] safe_home...")
        # safe_home 内部按关节轨迹限速回零，不能用瞬时目标替代。
        controller.safe_home(max_vel=float(args.max_vel), timeout=float(args.timeout))

        # disconnect() 内部会先 stop_control_loop，再 disable_all，再关闭通信。
        print("[HomeDisable] disable and disconnect...")
        arm.disconnect()
        controller._running = False
        print("[HomeDisable] done")
        return 0
    except KeyboardInterrupt:
        print("\n[HomeDisable] interrupted; disabling before disconnect")
        try:
            arm.disable_all()
        finally:
            arm.disconnect()
        return 130
    except Exception as exc:
        print(f"[HomeDisable] failed: {exc}")
        try:
            arm.disable_all()
        finally:
            arm.disconnect()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
