# Sim2Real YOLO + SAM 传统抓取

本文介绍 `reBotArm_Isaacsim` 目录下各条命令的功能，以及仿真抓取和真机抓取的启动方法。

## 命令功能速查

| 命令 | 功能 | 是否连接真机 |
|---|---|---|
| `./build_dm_asset.sh` | 把 DM 机械臂 MJCF 资产转换为 Isaac Sim 使用的 USD | 否 |
| `./run_isaacsim_receiver.sh` | 启动 Isaac Sim 场景，并通过 UDP 接收关节和夹爪目标 | 否 |
| `./run_test_sender.sh` | 发送内置测试轨迹，验证仿真机械臂、UDP 和夹爪联动 | 否 |
| `./run_sender.sh` | 连接真实机械臂，启用重力补偿，并把真机关节角发送到 Isaac Sim | **是** |
| `./run_sim_rgbd.sh` | 启动 Isaac Sim、导出腕部 RGB-D，并执行感知端生成的仿真抓取计划 | 否 |
| `./run_sim2real_perception.sh --source sim` | 读取仿真 RGB-D，运行 YOLO + SAM，并生成仿真抓取计划 | 否 |
| `./run_sim2real_perception.sh --source real` | 连接真实 RGB-D 相机，只运行感知和结果预览，不控制机械臂 | 只连接相机 |
| `./run_sim_grasp.sh` | 一条命令同时启动仿真 RGB-D 和仿真感知，循环完成抓取与放置 | 否 |
| `./run_real_grasp.sh` | 同时启动真机感知和执行器，自动循环完成抓取与放置 | **是** |

所有命令默认在以下目录执行：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/reBotArm_Isaacsim
```

## 一键运行 Isaac Sim 抓取

```bash
./run_sim_grasp.sh
```

该命令会自动启动两个进程：

1. `run_sim_rgbd.sh` 使用 Isaac Sim 官方 Python 启动仿真、导出腕部 RGB-D，并等待抓取计划。
2. `run_sim2real_perception.sh --source sim` 使用 `rebotarm_gpu` Conda 环境运行 YOLO + SAM，读取 RGB-D 并生成抓取计划。

仿真会循环执行：回到准备姿态、检测、张开夹爪、移动到预抓取位、接近、闭合夹爪、抬起、返回、竖直放置、松开、撤离、再次返回准备姿态。夹爪依靠 PhysX 接触和摩擦力夹持物体，不使用吸附或焊接，也不会连接真实机械臂。

按 `Ctrl+C` 可停止两个进程并关闭仿真。

## 分终端运行仿真抓取

需要分别观察仿真端和感知端日志时，可以使用两个终端启动。

终端 1 使用 Isaac Sim 官方 Python 环境：

```bash
./run_sim_rgbd.sh
```

该命令加载现有 DM 场景，把最新腕部相机帧导出到：

```text
/tmp/rebot_sim_rgbd.npz
```

`run_sim_rgbd.sh` 支持把以下参数转交给 `isaacsim_rgbd_exporter.py`：

| 参数 | 作用 | 默认值 |
|---|---|---|
| `--output PATH` | RGB-D 帧输出路径 | `/tmp/rebot_sim_rgbd.npz` |
| `--width N` | 输出图像宽度，单位为像素 | `640` |
| `--height N` | 输出图像高度，单位为像素 | `360` |
| `--export-hz HZ` | RGB-D 帧写盘频率 | `5.0` |
| `--settle-seconds SEC` | 导出前在观测姿态保持的时间；不指定时读取 `dm_sim.yaml` | 配置文件值 |

例如，以 10 Hz 导出 1280×720 RGB-D：

```bash
./run_sim_rgbd.sh --width 1280 --height 720 --export-hz 10
```

终端 2 使用 `rebotarm_gpu` Conda 环境：

```bash
./run_sim2real_perception.sh --source sim
```

感知命令支持以下参数：

| 参数 | 作用 | 默认值 |
|---|---|---|
| `--source {sim,real}` | 选择仿真 RGB-D 文件或真实 RGB-D 相机；必填 | 无 |
| `--frame PATH` | Sim 模式读取的 RGB-D NPZ 文件 | `/tmp/rebot_sim_rgbd.npz` |
| `--result PATH` | 感知候选结果 JSON 输出路径 | `/tmp/rebot_grasp_candidate.json` |
| `--config PATH` | YOLO、SAM 和抓取算法配置文件 | `rebot_grasp/config/default.yaml` |

## Isaac Sim 关节镜像与无硬件测试

只启动 Isaac Sim UDP 接收端：

```bash
./run_isaacsim_receiver.sh
```

如果只想加载和运行场景、不监听 UDP 端口：

```bash
./run_isaacsim_receiver.sh --no-udp
```

无硬件验证时，在第二个终端发送内置测试轨迹：

```bash
./run_test_sender.sh
```

需要把真实机械臂的关节运动镜像到 Isaac Sim 时，才运行：

```bash
./run_sender.sh
```

`run_sender.sh` 会连接真实机械臂并启用重力补偿。运行前必须确认 DM 型号、串口、关节与夹爪限位、工作空间已经清空，并确保急停可用。

## 真机循环抓取

确认 B601-DM、RGB-D 相机和急停状态正常，清空工作空间，然后直接运行：

```bash
./run_real_grasp.sh
```

启动和每轮执行均不需要输入确认。程序会自动执行：

```text
ready_pose -> 检测稳定候选 -> 张开夹爪 -> 预抓取 -> 接近 -> 力控闭合
           -> 抬起 -> 返回 -> 竖直放置 -> 松开 -> 撤离 -> ready_pose
           -> 等待相机稳定 -> 检测下一目标
```

启动脚本会运行两个进程：

- `sim2real_perception.py --source real` 独占 RGB-D 相机，持续发布稳定的 YOLO + SAM 抓取候选。
- `real_grasp_executor.py` 独占 DM 串口，把候选转换为真机动作并循环执行。

每轮回到 `ready_pose` 后，执行器只接受相机稳定时间之后生成的新候选，不会重复使用机械臂运动期间的旧结果。按 `Ctrl+C` 停止循环，启动脚本会同时结束感知进程并断开机械臂。

放置点跟随本轮视觉检测到的物体位置：不使用固定坐标，也不进行随机采样。仿真在当前物体位置正上方 `0.03 m` 竖直放置，真机把当前相机坐标转换到机器人 base 后原位置竖直放置。夹爪使用固定总开度 `0.03 m`，不再与视觉估计宽度相加；程序会在每轮执行前打印实际开度。若某个抓取或放置位姿 IK 失败，执行器会回到 `ready_pose`、释放夹爪并等待新的稳定候选，不会直接退出整个循环。

Sim 和 Real 感知都会把相机坐标系中的候选结果写到：

```text
/tmp/rebot_grasp_candidate.json
```

只检查真实相机感知、不连接或控制机械臂时，运行：

```bash
./run_sim2real_perception.sh --source real
```

在预览窗口按 `Q` 或 `Esc` 可退出纯感知模式。

## 环境变量

| 环境变量 | 作用 | 默认值 |
|---|---|---|
| `ISAACSIM_ROOT` | Isaac Sim 安装或运行目录 | 由对应启动脚本定义 |
| `REBOT_CONDA_ENV` | 感知和真机抓取使用的 Conda 环境 | `rebotarm_gpu` |
