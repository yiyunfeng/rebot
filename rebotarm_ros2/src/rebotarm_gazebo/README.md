# reBotArm Gazebo / MoveIt Run Modes

本包提供 Gazebo、MoveIt、真机控制和数字孪生相关启动入口。所有命令默认在工作空间根目录运行：

```bash
cd ~/Desktop/pythonProject/rebotarm_ros2
source install/setup.bash
```

如果修改过源码，先构建：

```bash
colcon build --packages-select rebotarm_gazebo --symlink-install
source install/setup.bash
```

## 1. 纯仿真：MoveIt 控 Gazebo

用途：不连接真实机械臂，只在 Gazebo 中运行机械臂，MoveIt 规划后控制 Gazebo 里的 ros2_control。

```bash
ros2 launch rebotarm_gazebo rebotarm.launch.py mode:=sim
```

不启动 RViz：

```bash
ros2 launch rebotarm_gazebo rebotarm.launch.py mode:=sim use_rviz:=false
```

默认使用模型中的原始碰撞几何，避免简化碰撞盒导致关节运动被异常卡住。

指定世界文件：

```bash
ros2 launch rebotarm_gazebo rebotarm.launch.py \
  mode:=sim \
  world:=$(ros2 pkg prefix rebotarm_gazebo)/share/rebotarm_gazebo/worlds/arm_on_the_table.sdf
```

## 2. 真机：MoveIt 控真实机械臂

用途：不启动 Gazebo。启动真实硬件驱动和 MoveIt，MoveIt 通过 `rebotarmcontroller` 控制真实机械臂。

DM 默认串口：

```bash
ros2 launch rebotarm_gazebo rebotarm.launch.py mode:=hardware
```

DM 指定串口：

```bash
ros2 launch rebotarm_gazebo rebotarm.launch.py \
  mode:=hardware \
  model:=dm \
  channel:=/dev/ttyACM0
```

如果同时需要一个独立的大图像窗口显示真实 DaBai DCW RGB 图像，先启动相机驱动，
再启动硬件 MoveIt，并打开 `rqt_image_view` 图像窗口：

```bash
ros2 launch orbbec_camera dabai_dcw.launch.py
ros2 launch rebotarm_gazebo rebotarm.launch.py \
  mode:=hardware \
  use_camera_rviz:=true
```

主 RViz 仍用于机械臂模型和 MoveIt 操作；图像窗口默认只显示
`/camera/color/image_raw`，不占用主视图。

## 3. 数字孪生：真机驱动 Gazebo / RViz 镜像

用途：真实机械臂由 `rebotarmcontroller` 控制，Gazebo 只跟随真机 `/rebotarm/joint_states` 显示姿态，不主动控制真机。

```bash
ros2 launch rebotarm_gazebo rebotarm.launch.py mode:=twin
```

指定 DM 真机通道：

```bash
ros2 launch rebotarm_gazebo rebotarm.launch.py \
  mode:=twin \
  model:=dm \
  channel:=/dev/ttyACM0
```

## 4. 单独启动 Gazebo 仿真底座

如果只想启动 Gazebo、MoveIt 和仿真控制器，不走统一入口：

```bash
ros2 launch rebotarm_gazebo gazebo.launch.py
```

使用 Gazebo 专用 controller 名称：

```bash
ros2 launch rebotarm_gazebo gazebo.launch.py \
  arm_controller:=gazebo_rebotarm_controller \
  gripper_controller:=gazebo_gripper_controller
```

只启动 Gazebo 和机器人，不启动 MoveIt：

```bash
ros2 launch rebotarm_gazebo gazebo.launch.py start_moveit:=false use_rviz:=false
```

### 带腕部相机的 Gazebo 仿真

`gazebo_camera.launch.py` 是带 DaBai DCW 腕部相机的独立仿真入口，不依赖 `gazebo.launch.py`。
仿真图像来自 Gazebo 虚拟相机，不需要启动真实 Orbbec 相机，也不需要运行 `rebot_grasp` 的 adapter。

```bash
# 只启动 Gazebo、机器人控制器、TF 和相机 ROS 2 话题桥接
ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=base
```

```bash
# 启动 Gazebo + RViz + 内置 OpenCV/AI 检测，不执行抓取
ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=vision
```

该模式会启动主 RViz 显示 MoveIt/机械臂 3D 模型，并额外启动
`rqt_image_view` 放大显示 `/dabai_camera/image`。图像不再塞到主 RViz 里，
避免主视图被占用或 RViz 插件冲突。

`mode:=vision/full/grasp` 会自动把仿真机械臂移动到 `table_view` 桌面观察姿态，
避免 home 姿态下腕部相机水平、看不到桌面。该姿态配置在：

```bash
rebotarm_ros2/src/rebotarm_gazebo/config/joint_pose_presets.yaml
```

也可以单独执行：

```bash
ros2 run rebotarm_gazebo joint_pose_commander --ros-args \
  --params-file src/rebotarm_gazebo/config/joint_pose_presets.yaml
```

真机手动移动到同一观察姿态时，先确认工作空间安全，再运行：

```bash
ros2 run rebotarm_gazebo joint_pose_commander --ros-args \
  --params-file src/rebotarm_gazebo/config/joint_pose_presets.yaml \
  -p command_action:=/rebotarm/follow_joint_trajectory \
  -p enable_before_move:=true
```

```bash
# 启动 Gazebo + MoveIt + RViz，不启动视觉检测
ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=moveit
```

```bash
# 完整仿真抓取链路：Gazebo + MoveIt + RViz + 检测 + 抓取 pipeline
ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=grasp
```

仿真数据流为：

```text
Gazebo 虚拟 DaBai 相机
  -> /dabai_camera/image + /dabai_camera/depth_image + /dabai_camera/camera_info
  -> camera_object_detector
  -> /dabai_camera/target_pose
  -> camera_grasp_pipeline mode=sim
  -> Gazebo 机械臂
```

### 内置相机检测：HSV

`camera_object_detector` 只用于 Gazebo 仿真绿色方块检测，固定采用 HSV，
统一发布 `/dabai_camera/target_pose`。这样仿真不依赖 AI 模型环境，ROS 2
系统 Python 和 conda 深度学习环境不会互相污染。

配置文件：

```bash
rebotarm_ros2/src/rebotarm_gazebo/config/camera_object_detector.yaml
```

```yaml
detector_backend: hsv
hsv_lower: [35, 40, 40]
hsv_upper: [90, 255, 255]
```

真机 DaBai DCW 和 `rebot_grasp` 视觉后端命令写在
`/home/yyf/Desktop/pythonProject/rebot/rebot_grasp/README_zh.md`。

## 5. 可用节点

查看本包可执行节点：

```bash
ros2 pkg executables rebotarm_gazebo
```

当前包含：

```text
rebotarm_gazebo joint_state_mirror
rebotarm_gazebo planning_scene_objects
```

单独运行 joint state 镜像节点：

```bash
ros2 run rebotarm_gazebo joint_state_mirror --ros-args \
  -p source_joint_states:=/rebotarm/joint_states \
  -p arm_command_topic:=/gazebo_rebotarm_controller/joint_trajectory \
  -p gripper_command_topic:=/gazebo_gripper_controller/joint_trajectory
```

## 注意

- `mode:=sim` 不连接真机，适合开发和验证 MoveIt 规划。
- `mode:=hardware` 会连接真实机械臂，启动前确认机械臂工作空间安全。
- `mode:=twin` 只镜像真机状态，不应向 Gazebo 控制器发送其他轨迹。
