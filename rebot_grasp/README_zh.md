# 🦾 reBot Arm B601 视觉夹取 Demo

<p align="center">
  <img src="https://raw.githubusercontent.com/Seeed-Projects/reBot-DevArm/main/media/v1.0.png" alt="reBot Arm B601">
</p>

<p align="center">
    <a href="./LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
    </a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Platform-Ubuntu%2022.04+-orange.svg" alt="Platform">
    <img src="https://img.shields.io/badge/Camera-RGB--D-green.svg" alt="Camera">
    <img src="https://img.shields.io/badge/Detection-YOLO-yellow.svg" alt="YOLO">
</p>

<p align="center">
  <strong>深度感知 · 目标检测 · 手眼标定 · 自主抓取 · 全开源</strong>
</p>

<p align="center">
  <strong>
    <a href="./README_zh.md">简体中文</a> &nbsp;|&nbsp;
    <a href="./README.md">English</a>
  </strong>
</p>

---

## 📖 项目介绍

**reBot Arm B601 视觉夹取 Demo** 是基于 [reBot Arm B601](https://github.com/vectorBH6/reBotArm_control_py) 机械臂控制库与 RGB-D 深度相机的视觉抓取算法演示项目。系统面向 B601-DM 机械臂，通过 YOLO 模型实时识别桌面物体，利用 OBB 最小外接矩形估计夹取姿态，经手眼标定将相机坐标系下的抓取点变换到机械臂基坐标系，最终驱动机械臂完成自主抓取。

### ✨ 核心功能

- 📷 **深度感知** — 默认使用 Orbbec DaBai DCW，也保留 Orbbec Gemini 2 与 RealSense D435i / D405 等 RGB-D 深度相机适配
- 🔍 **目标检测** — 基于 YOLO 模型识别，支持开放词汇自定义类别
- 📐 **姿态估计** — OBB 最小外接矩形短轴方向估计夹爪朝向，深度分位数估计抓取高度
- 🔄 **坐标变换** — TSAI 手眼标定（Eye-in-Hand），将相机系抓取点变换到机械臂基坐标系
- 🦾 **运动执行** — reBotArm_control_py IK + 轨迹控制器，内置夹爪力控状态机

---

## ⚙️ 硬件配置

| 组件 | 型号 / 要求 |
|------|------------|
| 机械臂 | reBot Arm B601-DM |
| 深度相机 | Orbbec DaBai DCW（默认）、Orbbec Gemini 2、Intel RealSense D435i / D405 |
| 通信接口 | USB2CAN 串口桥接器（机械臂）；USB 3.0（相机） |
| 主机 | Ubuntu 22.04+，Python 3.10，x86_64 |

**接线说明**

1. 将深度相机通过 USB 3.0 连接到主机
2. 将 USB2CAN 适配器连接到机械臂 CAN 总线并插入主机 USB 口
3. 配置设备权限：

```bash
sudo chmod a+rw /dev/bus/usb/*/*   # 深度相机 USB 权限
sudo chmod 666 /dev/ttyUSB0        # USB2CAN（端口号按实际调整）
```

---

## 🚀 快速上手

### Step 1. 克隆仓库

优先使用 Seeed-Projects 官方仓库：

```bash
git clone https://github.com/Seeed-Projects/reBot-DevArm-Grasp.git rebot_grasp
cd rebot_grasp
```

也可以使用当前开发仓库：

```bash
git clone https://github.com/EclipseaHime017/reBot-DevArm-Grasp.git rebot_grasp
cd rebot_grasp
```

### Step 2. 创建并配置 conda 环境

```bash
conda env create -f environment.yml
conda activate rebotarm
```

### Step 3. 安装机械臂控制库

```bash
git clone https://github.com/vectorBH6/reBotArm_control_py.git sdk/reBotArm_control_py
cd sdk/reBotArm_control_py
pip install -e .
cd ../..
```

如果 `pip install -e .` 报 `Multiple top-level packages discovered in a flat-layout`，请在 `reBotArm_control_py` 的 `pyproject.toml` 中加入显式包发现配置，然后重新执行 `pip install -e .`：

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["reBotArm_control_py*"]
```

B601-DM 通过 SDK 仓库中的 DM 配置运行。请在 `sdk/reBotArm_control_py/config/rebotarm.yaml` 中确认 `hardware_yaml` 指向：

```yaml
hardware_yaml: rebotarm_dm.yaml
```

视觉抓取程序会读取该 SDK 配置，并使用 DM 的 `posvel` 控制模式与夹爪参数。

### Step 4. 安装深度相机 SDK

本项目默认使用 Orbbec DaBai DCW，并保留 Orbbec Gemini 2 与 RealSense D435i / D405 等 RGB-D 深度相机适配。请根据实际使用的相机安装对应 SDK；如果当前环境已经能正常导入相机驱动，可跳过本步骤。

**Orbbec DaBai DCW（默认）**

DaBai DCW 使用本仓库 `sdk/pyorbbecsdk` 中的 Orbbec Python SDK 驱动。先进入 `rebotarm` 虚拟环境，再安装本地 SDK：

```bash
conda activate rebotarm
cd sdk/pyorbbecsdk
pip install -r requirements.txt
pip install -e .
cd ../..
python -c "from drivers.camera import OrbbecDaBaiDCW; print('DaBai DCW driver OK')"
```

项目驱动会在启动时自动预加载 `sdk/pyorbbecsdk/install/lib` 下的 Orbbec 动态库；若要在命令行中直接 `import pyorbbecsdk`，需要先设置 `LD_LIBRARY_PATH=sdk/pyorbbecsdk/install/lib`。

默认配置在 `config/default.yaml`：

```yaml
camera:
  type: orbbec_dabai_dcw
  model: "Orbbec DaBai DCW"
  color_width: 640
  color_height: 360
  depth_width: 640
  depth_height: 360
  fps: 30
```

当前 DaBai DCW 接在 USB2 时，`640x360@30fps` 实测有效 RGB-D 帧比例最高；如果换分辨率或接口，再重新做帧率测试。

驱动启动后会从 SDK 读取相机出厂标定的 `rgb_intrinsic`、`rgb_distortion` 和 `depth_scale`，这些参数不要手动硬编码到抓取代码里。若后续重新做 OpenCV 标定，可把结果保存到 `config/calibration/orbbec_dabai_dcw/intrinsics.npz`。

**Orbbec Gemini 2**

Orbbec Gemini 2 依赖 **pyorbbecsdk**（Orbbec SDK v2 的 Python 版本）。优先推荐直接安装预编译 Python 包：

**方式一：通过 pip 安装（推荐）**

```bash
pip install pyorbbecsdk2
```

**方式二：从 GitHub 获取**

```bash
# 安装编译依赖
sudo apt-get install -y cmake build-essential libusb-1.0-0-dev

cd sdk
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
pip install -e .
```

对于中国大陆用户可以使用
```bash
git clone https://gitee.com/orbbecdeveloper/pyorbbecsdk.git
```

源码安装时，请先通过 CMake 编译生成原生扩展，确保 `install/lib` 中已有 `pyorbbecsdk*.so` 和 Orbbec 动态库，再执行 `pip install -e .`。

注意，如果上述安装过程中均发生错误导致安装失败，请参考下方Orbbec官方文档进行安装操作。

**验证安装**

```bash
python -c "import pyorbbecsdk; print('pyorbbecsdk OK')"
```

**RealSense D435i / D405**

RealSense 相机依赖 `pyrealsense2`。通常可以直接通过 pip 安装：

```bash
pip install pyrealsense2
python -c "import pyrealsense2; print('pyrealsense2 OK')"
```

如果系统需要完整的 RealSense 工具链或 udev 规则，请参考 RealSense SDK 官方文档安装 `librealsense2`。

**Orbbec udev 规则（首次使用必须）**

```bash
cd sdk/pyorbbecsdk
sudo bash scripts/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**OrbbecViewer（可选，用于验证相机）**

下载预编译包后运行 `OrbbecViewer`，可在运行 Demo 前确认相机连接和深度流正常。

- GitHub：https://github.com/orbbec/OrbbecSDK_v2/releases
- Gitee：https://gitee.com/orbbecdeveloper/OrbbecSDK_v2/releases

**SDK 资料汇总**

| 资料 | 链接 |
|------|------|
| Gemini 2 产品页 | https://www.orbbec.com.cn/index/Product/info.html?cate=38&id=51 |
| 开发资料总链接 | https://www.orbbec.com.cn/index/Download2025/info.html?cate=121&id=1 |
| Orbbec SDK v2 | https://github.com/orbbec/OrbbecSDK_v2 |
| SDK v2 API 文档 | https://orbbec.github.io/docs/OrbbecSDKv2_API_User_Guide/ |
| pyorbbecsdk | https://github.com/orbbec/pyorbbecsdk |
| pyorbbecsdk 文档 | https://orbbec.github.io/pyorbbecsdk/index.html |
| ROS2 Wrapper | https://github.com/orbbec/OrbbecSDK_ROS2/tree/v2-main |
| Intel RealSense SDK | https://github.com/realsenseai/librealsense |

### Step 5. 配置 GraspNet（可选）

为了实现对物体夹取姿态更准确的估计，本项目对[graspnet-baseline]{https://github.com/graspnet/graspnet-baseline}进行了适配，从而提升机械臂夹取的性能。

GraspNet 的 `pointnet2` / `knn` 扩展需要 CUDA 编译器。开始前先确认当前环境可以找到 `nvcc`，并检查 `nvcc` 的 CUDA 版本是否和 PyTorch 编译时使用的 CUDA 版本一致：

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

如果没有 `nvcc`，或 `nvcc` 显示的 CUDA 版本与 `torch.version.cuda` 不一致，请安装与当前 PyTorch CUDA 版本匹配的 CUDA 编译器。例如 PyTorch 显示 `13.0` 时：

```bash
conda install -c nvidia cuda-nvcc=13.0
```

也可以反过来安装与当前 `nvcc` 版本匹配的 PyTorch。两者必须一致，否则编译 `pointnet2` / `knn` 时会出现 `The detected CUDA version (...) mismatches the version that was used to compile PyTorch (...)`。

```bash
cd sdk
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline

# 按你的 CUDA 版本安装 PyTorch 后，再安装 GraspNet 运行依赖
pip install open3d tensorboard Pillow tqdm

# 编译本地算子前配置 CUDA 编译路径。
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"
export CPATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/include:$CPATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 编译 CUDA 算子
cd pointnet2
pip install . --no-build-isolation
cd ../knn
pip install . --no-build-isolation
cd ..

# 安装 GraspNet API
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
sed -i "s/'sklearn'/'scikit-learn'/" setup.py
pip install .
cd ../../..
```

***注：如果直接参考graspnet-baseline官方仓库文档使用 `python setup.py install` 可能报 CUDA / PyTorch 相关错误，建议使用 `pip install . --no-build-isolation`，让扩展在当前 conda 环境中复用已安装的 PyTorch 与 CUDA 配置进行编译。***

***如果编译时报 `fatal error: cusparse.h: No such file or directory`，先运行 `find $CONDA_PREFIX -name cusparse.h`，并把包含 `cusparse.h` 的目录加入 `CPATH` / `CPLUS_INCLUDE_PATH`。如果 CUDA 头文件来自 conda `cuda-toolkit`，路径通常是 `$CONDA_PREFIX/targets/x86_64-linux/include`，而不是上面示例里的 pip `nvidia/cu13/include` 路径。***

***此外，GraspNet API 的依赖中可能仍使用 `sklearn` 包名。上面的 `sed` 命令会将 `sklearn` 替换为 `scikit-learn`，避免安装时出现包名提示。除非同步调整 GraspNet API 的依赖栈，否则建议保留其 `numpy==1.23.4` 约束，因为 `transforms3d==0.3.1` 仍使用 `np.float` 等 NumPy 别名。***

下载 GraspNet 官方预训练权重 `checkpoint-rs.tar`，并放到项目约定目录：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp
conda activate rebotarm_gpu
mkdir -p sdk/graspnet-baseline/checkpoints
pip install gdown
gdown 1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk -O sdk/graspnet-baseline/checkpoints/checkpoint-rs.tar
```

上面的文件 ID 来自 `graspnet-baseline` 官方仓库 README 中的 `checkpoint-rs.tar` 下载项。

然后在 `config/default.yaml` 中确认：

```yaml
graspnet:
  checkpoint: "checkpoint-rs.tar"
```

`checkpoint` 支持三种写法：仅文件名会自动从 `sdk/graspnet-baseline/checkpoints/` 查找；相对路径会按项目根目录解析；绝对路径会直接使用。

---

## 📁 目录结构

```
rebot_grasp/
├── config/
│   ├── default.yaml              # 主配置文件
│   └── calibration/
│       └── <camera_type>/
│           ├── intrinsics.npz    # 相机内参
│           └── hand_eye.npz      # 手眼标定结果
├── drivers/
│   ├── camera/
│   │   ├── base.py               # 相机抽象基类
│   │   ├── orbbec_dabai_dcw.py   # DaBai DCW 驱动（默认）
│   │   ├── orbbec_gemini2.py     # Gemini 2 驱动
│   │   └── realsense.py          # RealSense 驱动（备用）
│   └── robot/
│       └── grasp_driver.py       # 基于机械臂 SDK 的轻量抓取辅助
├── calibration/
│   ├── aruco_pose.py             # ArUco 位姿估计
│   └── hand_eye.py               # 手眼标定求解
├── utils/
│   ├── ordinary_grasp.py         # OBB 抓取姿态估计与可视化
│   └── transforms.py             # 坐标变换工具
├── scripts/
│   ├── main.py                   # 主抓取程序
│   ├── ordinary_grasp_pipeline.py
│   ├── object_detection.py
│   └── collect_handeye_eih.py
├── sdk/
│   ├── pyorbbecsdk/              # Orbbec SDK Python 封装
│   └── reBotArm_control_py/      # reBot Arm SDK
└── environment.yml               # 推荐的 conda 环境文件
```

---

## 🛠️ 配置说明

### 配置文件

编辑 `config/default.yaml`，确认以下关键参数：

```yaml
camera:
  type: orbbec_dabai_dcw
  serial: null
  color_width: 640
  color_height: 360
  depth_width: 640
  depth_height: 360
  fps: 30

calibration:
  aruco:
    marker_length_m: 0.1
    dict_id: 0
    target_marker_id: 0
  hand_eye_method: TSAI
  hand_eye_compensation_m:
    x: 0.00
    y: 0.0
    z: -0.01

detection:
  conf_threshold: 0.5
  iou_threshold: 0.45

perception:
  backend: "obb"       # obb / sam / graspnet

robot:
  repo_root: null
  execution_compensation_base_m:
    x: 0.0
    y: 0.0
    z: 0.0
  gripper:
    dm:
      angle_open: 5.0
      counterclockwise: true
      tau_max: 1.5
      close_torque: 1.0
      default_force: 0.30
  ready_pose:
    x: 0.3
    y: 0.0
    z: 0.3
    roll: 0.0
    pitch: 0.7
    duration: 3.0

yolo:
  model_name: "yoloe-26l-seg.pt"
  device: "cpu"          # GPU 可改为 "cuda:0"
  use_world: true
  custom_classes:
    - "yellow banana"
    - "water bottle"
    - "light blue coffee cup"
    - "cup"
    - "green object"
    - "tool"

sam:
  enabled: false
  model_name: "sam_b.pt"
  device: "cpu"
  conf_threshold: 0.01

grasp_pipeline:
  infer_every_live: 3
  grasp:
    depth_quantile: 0.5
    pregrasp_offset_m: 0.080
    insertion_depth_m: 0.015
    min_base_z_m: 0.00

graspnet:
  checkpoint: "checkpoint-rs.tar"
  num_view: 300
  num_point: 20000
  collision_thresh: 0.01
  voxel_size: 0.01
  min_depth: 0.05
  max_depth: 1.0
  top_k: 50
  target_class: null
  target_margin_px: 12
  target_expand_ratio: 1.35
  use_yolo_filter: true
  max_grasp_width_m: 0.065
```

### YAML参数说明

- `camera.type`：相机类型，默认 `orbbec_dabai_dcw`，也保留 `realsense_d435i`、`realsense_d405`、`orbbec_gemini2`。
- `camera.serial`：指定设备序列号；`null` 表示使用第一台可用设备。
- `calibration.aruco.marker_length_m`：手眼标定用 ArUco 边长，单位米。
- `calibration.hand_eye_compensation_m`：手眼标定后的 XYZ 手动平移补偿，作用在机器人基坐标系下，单位为米。三项全为 `0.0` 时，补偿矩阵为单位矩阵。
- `detection.conf_threshold`：YOLO 检测置信度阈值。
- `detection.iou_threshold`：YOLO NMS IoU 阈值。
- `perception.backend`：直接相机抓取使用的视觉后端，可选 `obb`、`sam`、`graspnet`。
- `sam.enabled`：是否启用 SAM 精分割。启用后使用 YOLO 检测框作为 prompt，SAM mask 会替换普通抓取流程中的 YOLO mask / bbox 区域，用于更精确地估计中心、主轴和深度。
- `sam.model_name`：SAM 权重名或路径。仅文件名会从 `models/` 目录查找；绝对路径会直接使用。
- `robot.repo_root`：`reBotArm_control_py` 仓库根目录；为 `null` 时默认使用 `sdk/reBotArm_control_py`。
- `robot.execution_compensation_base_m`：机械臂自身执行误差补偿，作用在机器人 base 坐标系下，单位米。它只修正最终发给真机的 `grasp/pregrasp/retreat` 目标位姿，不参与相机内参、手眼标定和视觉检测。例：实测 TCP 总是比目标高 `2cm`，则 `z` 填 `-0.02`。
- `robot.gripper.dm`：DM 夹爪参数。`angle_open`、`close_torque`、`default_force` 均填写正数数值；`counterclockwise` 表示闭合时采用的电机转动方向，代码会据此推导张开角度和闭合力矩的符号。`tau_max` 为力矩上限。其余夹爪行为参数在 `drivers/robot/grasp_driver.py` 中定义。
- `robot.ready_pose`：启动后先到达的预备位，抓取结束后也会回到这里。
- 机械臂配置：在 SDK 的 `sdk/reBotArm_control_py/config/rebotarm.yaml` 中保持 `hardware_yaml: rebotarm_dm.yaml`。
- `grasp_pipeline.infer_every_live`：实时预览时每 N 帧跑一次检测，减轻 CPU/GPU 压力。
- `grasp_pipeline.grasp.depth_quantile`：短轴抓取管线使用的深度分位数，值越大通常抓取点越深。
- `grasp_pipeline.grasp.pregrasp_offset_m`：预抓取位相对最终抓取位，沿末端进给方向回退的距离，单位米。
- `grasp_pipeline.grasp.insertion_depth_m`：GraspNet 执行时沿进给方向额外插入的距离。
- `grasp_pipeline.grasp.min_base_z_m`：机械臂基坐标系下允许的最低抓取高度。
- `graspnet`：`scripts/graspnet_camera_demo.py` 和 `scripts/grasp.py` 使用的 GraspNet 运行参数。

### 模型选择库

YOLO 模型会从 `rebot_grasp/models/` 目录加载；如果模型文件不存在，Ultralytics 通常会尝试自动下载。

常用模型：

| 模型 | 说明 |
| --- | --- |
| `yoloe-26l-seg.pt` | 开放词汇 + 分割，当前默认 |
| `yoloe-26s-seg.pt` | 更轻量，速度更快 |
| `yolov8n-seg.pt` | 封闭类别分割，小模型 |
| `yolov8s-seg.pt` | 封闭类别分割，精度更高 |

当模型名包含 `world` / `yoloe`，并且 `yolo.use_world=true` 时，程序会调用 `model.set_classes(custom_classes)`，将 `yolo.custom_classes` 注入为开放词汇类别。普通 `yolov8*-seg.pt` 模型会忽略这组开放词汇类别。

---

## 🎬 运行与调试

### 0. 确认 DM 机械臂与 SDK 配置

运行会连接机械臂的脚本前，请先确认 B601-DM、电源和 SDK 配置一致：

- 请先完成 [B601-DM 快速入门](https://wiki.seeedstudio.com/cn/rebot_b601_dm_getting_started/)。
- 在 `sdk/reBotArm_control_py/config/rebotarm.yaml` 中确认 DM 硬件配置：

```yaml
hardware_yaml: rebotarm_dm.yaml
```

- B601-DM 使用 24V DC 电源，请确认电源适配器和接线与机械臂版本一致。
- 使用 B601-DM 时，请确认 SDK 配置中的串口桥接器设备路径与实际设备一致。

### 1. 手眼标定（抓取前必做）

```bash
python scripts/collect_handeye_eih.py
```

自动模式下，机械臂会自动遍历 50 个预设位姿，检测到 ArUco 稳定后自动采样。正常结束或中途打断时，脚本都会尝试计算并保存标定结果；至少需要 5 个样本，建议 ≥15 个样本以获得更稳的结果。

如需手动推动机械臂采集，可使用：

```bash
python scripts/collect_handeye_eih.py --manual
```

手动模式下，机械臂会进入重力补偿状态。将末端推到合适视角后按 `Enter` 采集，按 `c` 或 `q` 结束并计算。

### 2. `scripts/main.py` — 主抓取程序

完整的视觉抓取流水线：

1. 初始化 RGB-D 相机，确认图像流可用
2. 机械臂与夹爪使能，移动到预备高位
3. 实时相机预览 + YOLO 目标检测与实例分割
4. OBB 短轴估计夹爪朝向，深度分位数估计抓取高度
5. 按 `G` 冻结帧，经手眼变换计算机械臂目标位姿
6. 机械臂移动到预抓取点 → 下降 → 夹爪闭合 → 提升 → 回预备位

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp

./scripts/run_main.sh
```

### 3. `scripts/ordinary_grasp_pipeline.py` — 简化抓取测试

不依赖机械臂，仅验证 OBB 抓取姿态估计和可视化效果，适合调试感知模块。

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp
conda activate rebotarm_gpu

# OBB/普通抓取调试：config/default.yaml 中保持 sam.enabled: false
python scripts/ordinary_grasp_pipeline.py

# SAM 调试：config/default.yaml 中改为 sam.enabled: true
python scripts/ordinary_grasp_pipeline.py
```

### 4. `scripts/graspnet_camera_demo.py` — GraspNet 相机估计 Demo

不连接机械臂，仅使用 RGB-D 相机运行 GraspNet 6D 夹取姿态估计。脚本会保留实时相机预览，并使用 YOLO 检测框选择目标区域，再从 GraspNet 全场景候选中筛选目标 bbox 内的可行夹取。按 `G` 或 `Space` 对当前帧推理，按 `R` 恢复实时预览，按 `Q` 或 `Esc` 退出；推理后可通过 Open3D 查看点云与夹取候选。

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp
./scripts/run_graspnet_camera_demo.sh
./scripts/run_graspnet_camera_demo.sh --camera-type orbbec_dabai_dcw
```

### 5. `scripts/grasp.py` — GraspNet 机械臂抓取程序

`scripts/grasp.py` 由 `scripts/main.py` 在 `perception.backend: "graspnet"` 时自动调用。参数统一写在 `config/default.yaml` 的 `graspnet`、`grasp_pipeline`、`robot` 段里，正式运行只用统一入口：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp
./scripts/run_main.sh
```

### 6. `scripts/object_detection.py` — 基础检测 Demo

纯 YOLO 检测演示，实时显示检测框和置信度，无抓取逻辑。

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp
conda activate rebotarm_gpu
python scripts/object_detection.py
```

### 7. 真机误差与手眼检查脚本

#### 7.1 机械臂 TCP 到位测试

该脚本只控制机械臂移动到指定 TCP 位姿，不打开相机，不运行视觉模型。默认会应用
`config/default.yaml` 中的 `robot.execution_compensation_*`，用于验证补偿后的实际落点。

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp

./scripts/run_move_pose_error_test.sh \
  --x 0.20 \
  --y 0.00 \
  --z 0.20 \
  --roll 0.0 \
  --pitch 0.7 \
  --yaw 0.0
```

交互模式：

```bash
./scripts/run_move_pose_error_test.sh --interactive
```

交互输入示例：

```text
pose> 0.20 0.00 0.20 0.0 0.7 0.0
pose> 0.30 0.15 0.10 0.0 0.7 0.0
pose> ready
pose> q
```

如果要测原始未补偿位姿，在命令末尾加 `--raw`：

```bash
./scripts/run_move_pose_error_test.sh --interactive --raw
```

#### 7.2 相机点转 base 坐标检查

该脚本只读取真实相机和当前 TCP/FK，不发送机械臂运动命令。用于点击桌面固定点，
打印该点在相机坐标和机械臂 base 坐标下的位置，方便用尺子检查手眼/深度误差。

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebot_grasp
./scripts/run_check_camera_base_point.sh
```

操作：

```text
鼠标左键：打印点击点 camera xyz / base xyz
R：刷新当前 TCP 位姿
Q 或 Esc：退出
```

正确测试流程：

```text
1. 先把机械臂移动到固定拍照姿态并停稳；
2. 运行 run_check_camera_base_point.sh；
3. 点击桌面固定标记点；
4. 用尺子量该标记点相对 base 的真实坐标；
5. 对比终端输出的 base xyz。
```

仿真命令不在本项目运行；请看
`/home/yyf/Desktop/pythonProject/rebot/rebotarm_ros2/src/rebotarm_gazebo/README.md` 的
“带腕部相机的 Gazebo 仿真”。

---

## ❓ FAQ

### 1. `ModuleNotFoundError: No module named 'motorbridge'`

这通常表示当前 Python 环境还没有安装机械臂 SDK 依赖。请确认已激活项目环境，并重新同步环境与安装机械臂 SDK：

```bash
conda activate rebotarm_gpu
conda env update -n rebotarm_gpu -f environment.yml
cd sdk/reBotArm_control_py && pip install -e .
```

### 2. 按 `G` 后不执行抓取

常见原因包括：

- `hand_eye.npz` 不存在
- 手眼标定模式不是 `eye_in_hand`
- 当前目标位姿 IK 不可达

先确认 `config/default.yaml` 中的相机、手眼标定和 `perception.backend` 配置正确，再运行 `./scripts/run_main.sh`。

### 3. 抓取点深度不稳定

可以优先检查和调整：

- `grasp_pipeline.grasp.depth_quantile`
- 相机与目标工作区域的安装高度
- 目标表面的反光情况

### 4. GraspNet 报 `pointnet2_utils` 无法从 `pointnet2` 导入

这通常是 `sdk/graspnet-baseline/pointnet2` 本地 CUDA 扩展没有在当前 conda 环境中正确编译安装，或 Python 路径解析到了错误的 `pointnet2` 包。建议确认已激活项目环境，并在同一个环境中重新编译安装 `pointnet2` 与 `knn`：

```bash
conda activate rebotarm_gpu
cd sdk/graspnet-baseline/pointnet2
pip install . --no-build-isolation

cd ../knn
pip install . --no-build-isolation
```

验证：

```bash
python -c "from pointnet2 import pointnet2_utils; print('Submodule import works')"
```

### 5. 当前显卡运行 GraspNet 时出现 CUDA 架构不兼容

如果出现 `no kernel image is available for execution on the device` 或 PyTorch 提示当前 GPU 的 CUDA capability 不受支持，通常说明当前 PyTorch wheel 不包含该显卡架构对应的 CUDA kernel。建议安装支持当前 CUDA/显卡架构的 PyTorch 版本，然后重新编译 GraspNet 的本地 CUDA 扩展。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

cd sdk/graspnet-baseline/pointnet2
pip install . --no-build-isolation

cd ../knn
pip install . --no-build-isolation
```

如需手动指定编译架构，可在重新编译前设置 `TORCH_CUDA_ARCH_LIST`，具体取值请按当前显卡架构和 PyTorch/CUDA 版本确认。

### 6. GraspNet 推理时报 `RuntimeError: CPU not supported`

`pointnet2` 中的采样算子只支持 CUDA tensor。请确认 CUDA 可用、GraspNet 网络和输入点云都在 GPU 上，并且 `pointnet2` / `knn` 是在当前环境和当前 PyTorch 版本下编译的。

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出为 `False`，需要先修复 CUDA / PyTorch 安装；如果输出为 `True` 但仍报错，建议重新编译 `pointnet2` 和 `knn`。

---

## 📄 参考资料

- [reBotArm_control_py](https://github.com/vectorBH6/reBotArm_control_py) — 机械臂控制库
- [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) — reBot 机械臂开源项目
- [Orbbec Gemini 2 产品页](https://www.orbbec.com.cn/index/Product/info.html?cate=38&id=51)
- [Orbbec SDK v2](https://github.com/orbbec/OrbbecSDK_v2)
- [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)
- [RealSense SDK](https://github.com/realsenseai/librealsense)
- [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)

---

## ☎ 联系我们

- **技术支持**：[提交 Issue](https://github.com/Seeed-Projects/reBot-DevArm-Grasp/issues)

---

<p align="center">
  <strong>🌟 如果本项目对你有帮助，欢迎点个 Star！</strong>
</p>
