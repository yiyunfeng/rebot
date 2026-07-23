# reBotArm Gazebo / MoveIt Run Modes

本包提供 Gazebo、MoveIt、真机控制和数字孪生相关启动入口。所有命令默认在工作空间根目录运行：

```bash
cd ~/Desktop/pythonProject/rebotarm_ros2
source install/setup.bash
```

如果修改过源码，先构建：

```bash
colcon build --packages-select rebotarm_gazebo11 --symlink-install
source install/setup.bash
```

## 1. 纯仿真：MoveIt 控 Gazebo

用途：不连接真实机械臂，只在 Gazebo 中运行机械臂，MoveIt 规划后控制 Gazebo 里的 ros2_control。

```bash
ros2 launch rebotarm_gazebo11 rebotarm.launch.py mode:=sim
```

不启动 RViz：

```bash
ros2 launch rebotarm_gazebo11 rebotarm.launch.py mode:=sim use_rviz:=false
```

默认使用模型中的原始碰撞几何，避免简化碰撞盒导致关节运动被异常卡住。

指定世界文件：

```bash
ros2 launch rebotarm_gazebo11 rebotarm.launch.py \
  mode:=sim \
  world:=$(ros2 pkg prefix rebotarm_gazebo11)/share/rebotarm_gazebo11/worlds/arm_on_the_table.sdf
```

## 2. 真机：MoveIt 控真实机械臂

用途：不启动 Gazebo。启动真实硬件驱动和 MoveIt，MoveIt 通过 `rebotarmcontroller` 控制真实机械臂。

DM 默认串口：

```bash
ros2 launch rebotarm_gazebo11 rebotarm.launch.py mode:=hardware
```

DM 指定串口：

```bash
ros2 launch rebotarm_gazebo11 rebotarm.launch.py \
  mode:=hardware \
  model:=dm \
  channel:=/dev/ttyACM0
```

RS 使用 SocketCAN：

```bash
ros2 launch rebotarm_gazebo11 rebotarm.launch.py \
  mode:=hardware \
  model:=rs \
  channel:=can0
```

RS 启动前通常需要先拉起 CAN：

```bash
sudo ip link set can0 up type can bitrate 1000000
```

## 3. 数字孪生：真机驱动 Gazebo / RViz 镜像

用途：真实机械臂由 `rebotarmcontroller` 控制，Gazebo 只跟随真机 `/rebotarm/joint_states` 显示姿态，不主动控制真机。

```bash
ros2 launch rebotarm_gazebo11 rebotarm.launch.py mode:=twin
```

指定真机型号和通道：

```bash
ros2 launch rebotarm_gazebo11 rebotarm.launch.py \
  mode:=twin \
  model:=dm \
  channel:=/dev/ttyACM0
```

## 4. Gazebo / MoveIt 指令转发到真机

用途：MoveIt 使用 Gazebo 风格的控制器 action 名称，`trajectory_relay` 将轨迹转发到真实机械臂控制器。Gazebo 再通过真机 joint states 镜像真实执行结果。

```bash
ros2 launch rebotarm_gazebo11 rebotarm.launch.py mode:=gazebo_to_hardware
```

DM 指定串口：

```bash
ros2 launch rebotarm_gazebo11 rebotarm.launch.py \
  mode:=gazebo_to_hardware \
  model:=dm \
  channel:=/dev/ttyACM0
```

RS 指定 CAN：

```bash
ros2 launch rebotarm_gazebo11 rebotarm.launch.py \
  mode:=gazebo_to_hardware \
  model:=rs \
  channel:=can0
```

该模式的控制链路：

```text
MoveIt
  -> /rebotarm_controller/follow_joint_trajectory
  -> trajectory_relay
  -> /rebotarm/follow_joint_trajectory
  -> rebotarmcontroller
  -> 真机

真机 /rebotarm/joint_states
  -> joint_state_mirror
  -> Gazebo 控制器
  -> Gazebo/RViz 镜像
```

## 5. 单独启动 Gazebo 仿真底座

如果只想启动 Gazebo、MoveIt 和仿真控制器，不走统一入口：

```bash
ros2 launch rebotarm_gazebo11 gazebo.launch.py
```

使用 Gazebo 专用 controller 名称：

```bash
ros2 launch rebotarm_gazebo11 gazebo.launch.py \
  arm_controller:=gazebo_rebotarm_controller \
  gripper_controller:=gazebo_gripper_controller
```

只启动 Gazebo 和机器人，不启动 MoveIt：

```bash
ros2 launch rebotarm_gazebo11 gazebo.launch.py start_moveit:=false use_rviz:=false
```

## 6. 可用节点

查看本包可执行节点：

```bash
ros2 pkg executables rebotarm_gazebo11
```

当前包含：

```text
rebotarm_gazebo11 gazebo_robot_description
rebotarm_gazebo11 joint_state_mirror
rebotarm_gazebo11 planning_scene_objects
rebotarm_gazebo11 trajectory_relay
```

生成 Gazebo 用 robot description：

```bash
ros2 run rebotarm_gazebo11 gazebo_robot_description \
  $(ros2 pkg prefix rebotarm_gazebo11)/share/rebotarm_gazebo11/config/rebotarm_gazebo11.urdf.xacro
```

单独运行 joint state 镜像节点：

```bash
ros2 run rebotarm_gazebo11 joint_state_mirror --ros-args \
  -p source_joint_states:=/rebotarm/joint_states \
  -p arm_command_topic:=/gazebo_rebotarm_controller/joint_trajectory \
  -p gripper_command_topic:=/gazebo_gripper_controller/joint_trajectory
```

单独运行轨迹转发节点：

```bash
ros2 run rebotarm_gazebo11 trajectory_relay --ros-args \
  -p arm_input_action:=/rebotarm_controller/follow_joint_trajectory \
  -p arm_output_action:=/rebotarm/follow_joint_trajectory \
  -p gripper_input_action:=/gripper_controller/follow_joint_trajectory \
  -p gripper_output_action:=/rebotarm/gripper/command
```

## 注意

- `mode:=sim` 不连接真机，适合开发和验证 MoveIt 规划。
- `mode:=hardware` 会连接真实机械臂，启动前确认机械臂工作空间安全。
- `mode:=twin` 只镜像真机状态，不应向 Gazebo 控制器发送其他轨迹。
- `mode:=gazebo_to_hardware` 会把 MoveIt 轨迹转发到真机，建议先在 `mode:=sim` 中验证轨迹。
- 夹爪转发会把双指仿真位置映射到 `/rebotarm/gripper/command`，映射参数可在 `trajectory_relay` 节点参数中调整。
