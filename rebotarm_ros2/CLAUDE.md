# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 构建与运行

```bash
colcon build --symlink-install
source install/setup.bash
```

只构建特定包：
```bash
colcon build --packages-select rebotarm_gazebo --symlink-install
```

验证可执行入口：
```bash
ros2 pkg executables rebotarmcontroller
ros2 pkg executables rebotarm_gazebo
```

项目依赖 `reBotArm_control_py` SDK（`third_party/` 下）和 `motorbridge` Python 包（PyPI）。

## 包结构与职责

| 包 | 作用 |
|---|---|
| `rebotarm_msgs` | 自定义 ROS2 接口：`MoveToPoseIK` srv、`MoveToPose` action、`ArmStatus` msg 等 |
| `rebotarmcontroller` | 核心控制节点 `reBotArmController`，通过 SDK 连接真实硬件，暴露 topic/service/action |
| `rebotarm_bringup` | 启动文件、URDF 模型、RViz 配置、硬件 YAML 参数 |
| `rebotarm_moveit_config` | MoveIt 2 运动规划配置（SRDF、kinematics、ros2_control、joint limits） |
| `rebotarm_moveit_demos` | MoveIt 2 应用 demo（`draw_square`、`pick_place`） |
| `rebotarm_gazebo` | Gazebo Ignition 仿真、ros2_control、MoveIt 集成，含多模式运行 |
| `rebotarm_gazebo11` | Gazebo Classic 11 变体，含 C++ grasp plugin（ament_cmake 构建） |

## 硬件与命名空间

- **DM 型号**：USB 串口 (`/dev/ttyACM0`)，控制模式 `posvel`
- **RS 型号**：SocketCAN (`can0`)，控制模式 `mit`
- 默认命名空间：`/rebotarm`，通过 `arm_namespace` 参数覆盖
- DM 夹爪 `open=-5.0, close=0.0`；RS 夹爪 `open=5.0, close=0.0`

## reBotArmController 架构

`reBotArmController` (`rebotarmcontroller/rebotarm_controller.py`) 是硬件驱动的中心节点。内部通过组合模式委托给四个子模块：

- `HardwareManager` — 硬件连接/断开、关节轨迹执行、夹爪、FK、重力补偿
- `JointStatePublisher` — 定时发布 `/rebotarm/joint_states`
- `ArmServices` — enable/disable/safe_home/set_mode/set_zero/move_to_pose_ik/gripper 等 service
- `ArmActions` — `FollowJointTrajectory`、`MoveToPose`、`GripperCommand` action server

`MotorPassthrough` 提供低层单关节 MIT/pos_vel/vel 指令直通。

## 运行模式

### 真机直连
```bash
ros2 launch rebotarm_bringup bringup.launch.py channel:=/dev/ttyACM0
ros2 launch rebotarm_bringup driver.launch.py  # 仅控制节点，无 robot_state_publisher
```

### MoveIt 仿真（虚拟 ros2_control 硬件）
```bash
ros2 launch rebotarm_moveit_config demo.launch.py
ros2 launch rebotarm_moveit_demos pick_place.launch.py
```

### MoveIt 真机
```bash
ros2 launch rebotarm_bringup driver.launch.py
ros2 launch rebotarm_moveit_config hardware.launch.py  # 另开终端
```

### Gazebo 仿真（`rebotarm_gazebo`）
```bash
ros2 launch rebotarm_gazebo rebotarm.launch.py mode:=sim
ros2 launch rebotarm_gazebo moveit_pick_place.launch.py  # 抓取放置 demo
```

四种 mode：`sim`（纯仿真）、`hardware`（真机 MoveIt）、`twin`（数字孪生）、`gazebo_to_hardware`（Gazebo 规划后转发真机执行）。

## 关键配置

- 硬件参数：`rebotarm_bringup/config/rebotarm_hardware.yaml`（DM/RS 选择、串口、限位）
- 驱动参数：`rebotarm_bringup/config/driver_params.yaml`
- MoveIt 语义：`rebotarm_moveit_config/config/rebotarm.srdf`（DM）/ `rebotarm_rs.srdf`（RS）
- IK solver：`rebotarm_moveit_config/config/kinematics.yaml`
- 轨迹执行：`rebotarm_moveit_config/config/moveit_controllers.yaml`

DM 和 RS 分别有独立的 URDF/SRDF/YAML 配置，通过 `model` launch 参数切换。

## 项目技能

项目已安装 `ros2-manipulator-dev` skill（`.claude/skills/ros2-manipulator-dev/SKILL.md`），处理 MoveIt 2、ros2_control、URDF/SRDF、Gazebo 仿真等机械臂相关任务时参考其指引。
