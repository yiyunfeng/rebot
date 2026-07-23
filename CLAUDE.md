# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库概览

这是一个 **reBot Arm B601 六轴机械臂** 的多项目仓库，包含四个子项目：

| 目录 | 用途 | 环境/工具链 |
|------|------|------------|
| `rebotarm_ros2/` | ROS2 SDK 工作空间：硬件驱动、MoveIt 2 运动规划、Gazebo 仿真、RViz 可视化 | ROS2 Humble/Jazzy, colcon, ament_python |
| `rebot_grasp/` | 视觉抓取 demo：RGB-D 相机 + YOLO 检测 + OBB 姿态估计 + 手眼标定 + 自主抓取 | conda (environment.yml), PyTorch, YOLO |
| `reBot-Isaacsim/` | 真机→Isaac Sim 实时数字孪生：UDP JSON 关节角同步 | uv workspace, Isaac Sim Python |
| `third_party/` | 共享底层依赖库 `reBotArm_control_py`（Pinocchio 运动学 + motorbridge 电机控制 SDK） | uv |

**硬件型号**: 本仓库当前只维护 DM（大秒电机，USB 串口 `/dev/ttyACM0`）。

详细子项目文档：
- [rebotarm_ros2/CLAUDE.md](rebotarm_ros2/CLAUDE.md) — ROS2 包架构、构建命令、启动方式、MoveIt/Gazebo 配置
- [rebotarm_ros2/AGENTS.md](rebotarm_ros2/AGENTS.md) — 项目规范、编码风格、测试指南
- [rebotarm_ros2/docs/agent_lessons.md](rebotarm_ros2/docs/agent_lessons.md) — Codex 与 Claude Code 共享知识库，记录历史问题与解决方案
- [rebotarm_ros2/docs/pick_place_design.md](rebotarm_ros2/docs/pick_place_design.md) — Gazebo 抓取放置两套方案的完整设计文档
- [rebotarm_ros2/docs/moveit_interfaces_note.md](rebotarm_ros2/docs/moveit_interfaces_note.md) — MoveIt 四种运动接口对比（`/compute_ik` vs `/plan_kinematic_path` vs `/execute_trajectory` vs `/move_action`）

## 子项目间的关系

```
third_party/reBotArm_control_py  ← 底层 Python SDK（Pinocchio FK/IK、控制器、电机驱动）
    ├── rebotarm_ros2/           ← ROS2 封装（Topic/Service/Action），MoveIt 规划，Gazebo 仿真
    ├── rebot_grasp/             ← 视觉抓取（YOLO + 手眼标定 + SDK 控制）
    └── reBot-Isaacsim/          ← 数字孪生（SDK 读关节角 → UDP → Isaac Sim）
```

三个上层子项目都依赖 `reBotArm_control_py`，但通过不同方式引用：
- `rebotarm_ros2/` 通过 `third_party/` 目录下的 git clone
- `rebot_grasp/` 通过 conda 环境中 `pip install -e .` 安装到 `sdk/reBotArm_control_py/`
- `reBot-Isaacsim/` 通过 uv workspace `members` 引用 `third_party/reBotArm_control_py/`

此外都需要 `motorbridge`（PyPI 包）作为电机通信底层。

## 真机安全红线

- **真机操作前必须确认**: `model`（dm）、`channel`（`/dev/ttyACM*`）、工作空间无人员/障碍物、关节限制、急停路径
- **不得绕过**: 关节限制、碰撞检测、watchdog、厂商安全状态来让 demo 跑通
- **修改硬件配置前**: 同时检查 URDF/SRDF、控制器 YAML、joint_limits.yaml、launch 参数的一致性
- 具体规范见 [rebotarm_ros2/.claude/skills/ros2-manipulator-dev/references/hardware-safety.md](rebotarm_ros2/.claude/skills/ros2-manipulator-dev/references/hardware-safety.md)

## 关键架构概念

### DM 型号参数

| 项目 | DM |
|---|---|
| 通信 | USB 串口 `/dev/ttyACM0` |
| 控制模式 | `posvel` |
| 夹爪 open | `-5.0` |
| 夹爪 close | `0.0` |
| URDF/SRDF | `rebotarm.urdf.xacro` / `rebotarm.srdf` |

### ROS2 控制器架构（`rebotarmcontroller` 包）

`reBotArmController` 通过组合模式委托给四个子模块：
- `HardwareManager` — 硬件连接/断开、关节轨迹执行、夹爪、FK、重力补偿
- `JointStatePublisher` — 定时发布 `/rebotarm/joint_states`
- `ArmServices` — enable/disable/safe_home/set_zero/move_to_pose_ik/gripper 等服务
- `ArmActions` — FollowJointTrajectory、MoveToPose、GripperCommand action server
- `MotorPassthrough` — 低层单关节 MIT/pos_vel/vel 指令直通

默认命名空间 `/rebotarm`，通过 `arm_namespace` 参数覆盖。

## Python 环境

各子项目使用**不同的 Python 环境**：

| 子项目 | 环境管理 | 激活方式 |
|--------|---------|---------|
| `rebotarm_ros2/` | 系统 Python + ROS2 | `source /opt/ros/humble/setup.bash && source install/setup.bash` |
| `rebot_grasp/` | conda | `conda activate rebotarm` |
| `reBot-Isaacsim/` | uv workspace | `uv sync`（发送端）；Isaac Sim 自带 Python（接收端） |
| `third_party/` | uv | `uv sync` |

不要跨子项目混用 Python 环境。

## 常见构建命令

```bash
# ROS2 工作空间（在 rebotarm_ros2/ 下）
colcon build --symlink-install
colcon build --packages-select rebotarmcontroller --symlink-install
source install/setup.bash
ros2 pkg executables rebotarmcontroller

# rebot_grasp conda 环境
conda env create -f environment.yml
conda activate rebotarm
# 然后按 README 安装 SDK 和相机驱动

# reBot-Isaacsim uv 环境
cd third_party/reBotArm_control_py && uv sync
```

## Gazebo 仿真运行模式

`rebotarm_gazebo` 包支持四种 mode（通过 `mode:=` launch 参数）：
- `sim` — 纯仿真（假硬件）
- `hardware` — 真机 MoveIt（真机控制器 + move_group + RViz）
- `twin` — 数字孪生
- `gazebo_to_hardware` — Gazebo 规划 + 真机执行

注意：`gazebo.launch.py` 和 `rebotarm.launch.py` 是已有的仿真入口，不应修改。Pick & Place demo 有独立的 launch 文件。

## 项目技能

`rebotarm_ros2/.claude/skills/ros2-manipulator-dev/` — ROS2 机械臂开发技能，覆盖 MoveIt 2、ros2_control、URDF/SRDF、Gazebo 仿真。在处理此仓库的机械臂相关任务时应参考此技能。

## 来自 agent_lessons.md 的关键经验

- `.vscode/` 已在 `.gitignore` 中（曾因 `browse.vc.db` 导致仓库膨胀到 65GB）
- `setup.py` 的 `_collect_files()` 必须跳过 `__pycache__` 和 `.pyc` 文件
- `hardware.launch.py` 必须使用 `use_sim_time: False`（真机模式没有 Gazebo `/clock`）
- 修改 URDF/SRDF/控制器 YAML/joint_limits 时要作为整体一起更新
- 给 `move_to_pose` 添加了最低高度保护（`min_user_z: 0.005`），防止危险的低高度指令
- 详见 [rebotarm_ros2/docs/agent_lessons.md](rebotarm_ros2/docs/agent_lessons.md) 获取完整排查记录
