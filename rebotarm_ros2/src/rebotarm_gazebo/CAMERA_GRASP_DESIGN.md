# 相机视觉抓取方案整理

## 1. 总体目标

本方案把“相机识别”和“机械臂执行”分开。视觉模块只负责从 RGB-D 图像中得到目标位姿，抓取模块只负责把目标位姿转换成 MoveIt 或真实机械臂动作。

仿真统一输出 topic：

```text
/dabai_camera/target_pose
```

ROS2 真机 HSV 检测输出 topic：

```text
/rebot_grasp/grasp_pose
```

只要目标 topic 存在，后面的 `camera_grasp_pipeline.py` 就可以继续执行“先到物体上方，再下降抓取”的流程。

## 2. 仿真与真机的区别

仿真不使用真实 USB 相机。仿真中的图像来自 Gazebo 虚拟 DaBai 相机模型：

```text
Gazebo 虚拟相机
  -> /dabai_camera/image
  -> /dabai_camera/depth_image
  -> /dabai_camera/camera_info
  -> camera_object_detector
  -> /dabai_camera/target_pose
  -> camera_grasp_pipeline mode=sim
  -> Gazebo 机械臂
```

ROS2 的 `hardware_hsv` 模式使用真实 DaBai DCW 相机和内置 HSV 检测：

```text
真实 DaBai DCW 相机
  -> Orbbec ROS2 驱动
  -> /camera/color/image_raw
  -> /camera/depth/image_raw
  -> /camera/color/camera_info
  -> camera_object_detector
  -> /rebot_grasp/grasp_pose
  -> camera_grasp_hardware
  -> 真实 DM 机械臂
```

注意：仿真相机和真实相机不要同时走同一条抓取链路。仿真默认使用 `/dabai_camera/*`，
真机 Orbbec 驱动默认使用 `/camera/*`，两者的启动命令分开维护。

## 2.1 运行命令

### 仿真

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebotarm_ros2
source /opt/ros/humble/setup.bash
source install/setup.bash

# 只启动 Gazebo、机器人控制器、TF 和虚拟 DaBai 相机话题
ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=base

# Gazebo + RViz + 内置检测，不执行抓取
ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=vision

# 完整仿真抓取链路
ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=grasp
```

### 真机

`rebot_grasp` 不再通过 ROS2 图像桥接，直接使用相机 SDK 和机械臂 SDK：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp
./scripts/run_main.sh
```

ROS2 工程仍保留独立的 HSV 真机测试链路，不依赖 `rebot_grasp`：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebotarm_ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rebotarm_gazebo gazebo_camera.launch.py mode:=hardware_hsv
```

## 3. 视觉模块结构

ROS 节点入口：

```text
camera_object_detector.py
```

它只负责：

- 订阅彩色图、深度图、相机内参；
- 调用检测后端；
- 把像素点和深度反投影成相机坐标系 3D 点；
- 发布 `/dabai_camera/target_pose` 和 `/dabai_camera/debug_image`。

检测算法拆分如下：

```text
camera_detection_common.py    # ObjectDetection、mask 深度取样、角度计算
camera_detection_hsv.py       # Gazebo 绿色方块 HSV 检测
```

## 4. 检测方式

### HSV

Gazebo 仿真固定方式，适合绿色方块和快速调试。真实相机和真实机械臂的
AI 视觉流程放在 `rebot_grasp/` 侧维护，不放进 Gazebo 仿真检测节点。

流程：

```text
RGB 图像
  -> HSV 阈值
  -> 形态学去噪
  -> 最大轮廓
  -> mask 内深度中位数
  -> 3D 目标点
```

优点是快、依赖少、仿真稳定；缺点是依赖颜色，不适合复杂真实场景。

### OBB

适合规则物体，例如正方体、长方体、盒子、圆柱外接框。

流程：

```text
RGB 图像
  -> YOLO-OBB / YOLO box
  -> 旋转框或普通框生成 mask
  -> mask 内深度中位数
  -> 3D 目标点 + 图像主轴角度
```

优点是速度较快，可以获得目标大概朝向；缺点是只知道框，不知道精确轮廓。

### SAM

适合真实桌面上边界复杂、物体靠得近或需要更干净点云的场景。

流程：

```text
RGB 图像
  -> YOLO 先找目标 box
  -> box 作为 SAM prompt
  -> SAM 输出目标 mask
  -> mask 内深度中位数
  -> 3D 目标点 + mask 主轴角度
```

优点是轮廓更准；缺点是依赖更重、速度更慢，需要 SAM 权重和环境兼容。

### GraspNet

适合需要 6D 抓取姿态的真实抓取验证，也可在 Gazebo RGB-D 图像上调通接口。

流程：

```text
RGB-D 图像
  -> 点云采样
  -> GraspNet 生成全场景 6D 抓取候选
  -> 可选 YOLO 目标框过滤
  -> 碰撞/宽度过滤
  -> 最优抓取位姿
```

优点是输出完整抓取姿态；缺点是依赖最重，需要 `rebot_grasp` 中的
GraspNet checkpoint、Open3D、pointnet2/knn CUDA 扩展和可用 GPU 环境。

## 5. 全局配置

配置文件：

```text
rebotarm_ros2/src/rebotarm_gazebo/config/camera_object_detector.yaml
```

核心参数：

```yaml
detector_backend: hsv
fallback_to_hsv: false
inference_stride: 1
hsv_lower: [35, 40, 40]
hsv_upper: [90, 255, 255]
min_area: 500.0
min_depth: 0.05
max_depth: 2.0
```

建议：

- Gazebo 仿真只用 `hsv`，保证基础链路稳定；
- 真实相机、真实机械臂和 AI 视觉后端放在 `rebot_grasp/` 侧维护；
- 不在 Gazebo 仿真节点里加载深度学习模型，避免依赖冲突影响 ROS 2。

## 6. 抓取执行逻辑

`camera_grasp_pipeline.py` 不关心视觉内部细节。仿真中它只订阅 HSV 检测得到的目标位姿：

```text
/dabai_camera/target_pose
```

执行流程：

```text
目标位姿
  -> TF 转到 base_link
  -> 检查目标高度是否安全
  -> 张开夹爪
  -> 移动到目标上方
  -> 竖直下降
  -> 闭合夹爪
  -> 抬升
```

模式区别：

```text
mode=sim       -> Gazebo 夹爪 topic + MoveIt 仿真
mode=hardware  -> RealController + 真实 DM 机械臂
```

## 7. 使用建议

先按这个顺序验证：

```text
1. Gazebo + HSV + mode=sim
2. Gazebo + OBB/SAM/GraspNet + mode=sim
3. 真实 DaBai DCW 只看图像和 debug_image
4. ROS2 hardware_hsv，小范围低速验证抓取
```

真实机械臂测试前必须确认工作空间清空、急停可用、关节限制和夹爪限制正常。
