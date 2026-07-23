# reBotArm DM Isaac Sim

该项目使用 DM 版 reBotArm B601。机械臂质量、惯量、关节轴、限位和夹爪结构来自已经验证过的 MuJoCo 模型：

```text
assets/DM-rebot-dev-arm/source/reBot-DevArm_gripper.xml
```

Isaac Sim 运行时只读取生成后的 DM USD，所有可调参数集中在 `config/dm_sim.yaml`。

## 已实现

- DM 六轴机械臂，按 MuJoCo 的质量和惯量导入。
- 400 Hz PhysX，TGS 求解器，32/4 次位置/速度迭代。
- 六轴 position drive，保留 MuJoCo 的刚度、阻尼、力矩和速度限制。
- MuJoCo 双滑轨夹爪：左指驱动，右指通过 1:1 mimic 约束跟随。
- 夹爪开度 `0.001~0.050 m`，物理行程 `0~0.051 m`。
- 指尖、方块和桌面的独立摩擦材质。
- 与 MuJoCo 相同尺寸的桌面和 80 g、50 mm 绿色方块。
- UDP 接收关节目标；运行中使用 PD drive，不直接改写关节状态。

## 目录

```text
reBot-Isaacsim/
├── assets/DM-rebot-dev-arm/source/       # DM MJCF、网格和纹理
├── config/dm_sim.yaml                    # 唯一仿真配置
├── usd/DM-rebot-dev-arm/                 # 生成的 DM USD
└── reBotArm_Isaacsim/
    ├── build_dm_asset.sh                 # MJCF -> USD
    ├── run_isaacsim_receiver.sh          # 启动 Isaac Sim 场景
    ├── run_test_sender.sh                # 无硬件测试轨迹
    └── run_sender.sh                     # 真机重力补偿镜像
```

## 首次生成资产

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim
./reBotArm_Isaacsim/build_dm_asset.sh
```

成功时会打印 8 个关节：

```text
joint1 joint2 joint3 joint4 joint5 joint6 left_finger right_finger
```

修改 MJCF、驱动增益、夹爪参数或指尖摩擦后，可手动重新执行生成脚本；直接启动时也会检测更新并自动重建。

## 启动仿真

终端 1：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/reBotArm_Isaacsim
./run_isaacsim_receiver.sh
```

脚本会自动打开 Isaac Sim GUI，创建桌面和方块，加载 DM 机械臂并开始 400 Hz 物理仿真。
如果 MJCF、YAML 或资产生成器比现有 USD 新，启动脚本会先自动重建一次；后续启动不会重复生成。

终端 2，无硬件验证：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/reBotArm_Isaacsim
./run_test_sender.sh
```

测试轨迹只使用 DM 合法范围，夹爪命令单位直接是米。

真机镜像使用：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/reBotArm_Isaacsim
./run_sender.sh
```

真机运行前确认 DM 电机型号、通信通道、关节限位、工作空间和急停状态。该发送端会使能机械臂并进入重力补偿，不要在未确认安全时运行。

## Isaac Sim GUI 操作

### 查看机械臂和场景

启动后在 `Stage` 面板中可以看到：

```text
/World/reBotArmDM
/World/Table
/World/GraspObject
/World/defaultGroundPlane
```

- `Alt + 左键`：绕观察点旋转。
- `Alt + 中键`：平移视角。
- 鼠标滚轮：缩放。
- 按 `F`：聚焦 Stage 中选中的 Prim。
- 顶部 `Play/Pause`：开始或暂停物理仿真。

### 查看关节参数

1. 在 `Stage` 展开 `/World/reBotArmDM`。
2. 夹爪固定底座显示为 `gripper_base`，左右滑轨显示为 `left_finger_link` 和 `right_finger_link`。
3. 搜索并选择 `joint1` 到 `joint6`、`left_finger` 或 `right_finger`。
4. 在 `Property` 面板查看 `Physics Joint`、`Drive`、限位、最大力和最大速度。
5. `right_finger` 没有独立 drive；它通过 `PhysxMimicJointAPI` 跟随 `left_finger`。

不要在仿真播放时直接修改关节 Transform。需要手工检查时先暂停，再修改 drive 的 `Target Position`，然后继续播放。

### 调试增益

打开：

```text
Tools > Robotics > Asset Editors > Gain Tuner
```

选择 `/World/reBotArmDM` 后可检查关节响应。正常调参顺序：

1. 先保持 MuJoCo 原始 stiffness。
2. 有振荡时逐步增加 damping。
3. 响应太慢且没有达到力矩上限时，再小幅增加 stiffness。
4. 夹爪接触不稳时，先检查时间步、求解迭代和摩擦，不要只增大夹爪刚度。

### 移动待抓物体

1. 暂停仿真。
2. 选择 `/World/GraspObject`。
3. 使用 Move 工具调整位置，保持方块底面不低于桌面顶面 `z=0`。
4. 点击 Play，让方块先稳定落在桌面上，再执行机械臂动作。

## 关键物理参数

| 项目 | Isaac Sim 值 | MuJoCo 来源 |
|---|---:|---|
| 物理频率 | 400 Hz | `timestep=0.0025` |
| 求解器 | TGS | 对应高稳定接触配置 |
| 六轴 stiffness | 800, 800, 800, 600, 600, 600 | position actuator `kp` |
| 六轴 damping | 6, 6, 6, 2, 2, 2 | joint damping |
| 六轴最大力矩 | 27, 27, 27, 7, 7, 7 Nm | `actuatorfrcrange` |
| 六轴最大速度 | 1 rad/s | 写入 USD 时转换为 57.30 degree/s |
| 夹爪 stiffness | 2000 | finger position actuator |
| 夹爪 damping | 174 | `kv=124` + joint damping `50` |
| 夹爪 armature | 0.243 | finger joint armature |
| 夹爪最大力 | 50 N | finger actuator force range |
| 指尖静摩擦 | 1.5 | finger geom friction |
| 方块质量 | 0.08 kg | MuJoCo grasp scene |
| 方块对角惯量 | 0.00004 kg·m² | MuJoCo grasp scene |

MuJoCo Newton 的 `iterations=50` 不能逐项等同于 PhysX。当前使用 TGS 32/4 作为单机械臂接触仿真的初始值；若出现穿透或夹取抖动，可在 `dm_sim.yaml` 中提高到 64/8，但会增加计算量。

## 配置原则

只修改 `config/dm_sim.yaml`：

- `simulation`：物理频率、渲染频率、求解器迭代。
- `network`：UDP 地址、端口和发送频率。
- `hardware_mirror`：真机关节滤波和夹爪电机角到米制开度的换算。
- `arm.joints`：六轴限位、刚度、阻尼、最大力矩和速度。
- `gripper`：夹爪行程、驱动、armature、摩擦和最大力。
- `materials`：指尖、物体、桌面摩擦。
- `scene`：桌面和物体尺寸、位置与质量。

UDP 数据格式：

```json
{
  "sequence": 1,
  "timestamp": 1718000000.0,
  "joint_positions": [0.0, -0.2, -0.3, 0.0, 0.0, 0.0],
  "gripper_position": 0.035
}
```

`joint_positions` 单位为弧度，`gripper_position` 单位为米。接收端会按 DM 限位裁剪非法目标。

## 常见问题

### USD 不存在

先运行：

```bash
./reBotArm_Isaacsim/build_dm_asset.sh
```

### UDP 5005 被占用

```bash
fuser -v -n udp 5005
```

关闭旧的接收端后再启动，不要同时运行两个 Isaac Sim 接收端。

### 机械臂不动但 GUI 正常

先运行 `run_test_sender.sh`。若测试轨迹正常，问题在真机发送端或通信；若测试轨迹也不动，查看接收端是否打印新的 `sequence` 和 target。

### 夹爪不能稳定夹住方块

依次检查：指尖是否接触物体、夹爪目标是否在 `0.001~0.050 m`、方块质量、摩擦材质、400 Hz 时间步和 TGS 迭代。当前实现不使用吸附或焊接，夹取完全依赖接触力和摩擦。
