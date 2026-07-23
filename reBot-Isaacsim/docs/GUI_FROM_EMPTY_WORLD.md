# Isaac Sim 4.5：从空白 Stage 搭建 reBotArm DM（GUI 逐项操作）

本文只适用于 **NVIDIA Isaac Sim 4.5.0（Kit 106.5）**。菜单、扩展名和 ROS 2
OmniGraph 节点均按 4.5 编写，不套用 5.x/6.x 界面。

本文的“从空白 Stage 搭建”采用项目可复现的正确路线：新建空 Stage，在 GUI 中导入本仓库
的 MJCF，再用 GUI 配置场景、检查物理参数和创建 ROS 2 Action Graph。**不建议在 GUI
里凭 STL 手工重建 9 个刚体和 8 个关节**：MJCF 已包含精确的 link 相对位姿、惯量主轴、
关节轴和碰撞排除关系，手工抄录既慢又很容易产生坐标系错误。

> 正式资产由 `reBotArm_Isaacsim/build_dm_asset.py` 生成。GUI 实验文件请另存为新 USD，
> 不要覆盖 `usd/DM-rebot-dev-arm/rebotarm_dm.usd`。

## 1. 本文依据及版本边界

项目参数来源（优先级从高到低）：

1. `config/dm_sim.yaml`：Isaac Sim 最终仿真参数；
2. `assets/DM-rebot-dev-arm/source/reBot-DevArm_gripper.xml`：结构、局部位姿、质量和惯量；
3. `reBotArm_Isaacsim/build_dm_asset.py`：导入后修正的 drive、mimic、材料和场景设置；
4. Isaac Sim 4.5 官方文档：只用于确认 4.5 的 GUI 菜单和节点行为。

容易混淆但必须区分的版本差异：

- 4.5 的 MJCF 入口是 `File > Import`，扩展名是 `isaacsim.asset.importer.mjcf`；
- 4.5 的 Physics Inspector 是 `Tools > Physics > Physics Inspector`；
- 4.5 的 `ROS2 Publish Joint State` 直接设置 `targetPrim`。不要加入 Isaac Sim 6.0
  migration 文档中的 `Isaac Read Joint State`；
- 4.5 的 Joint State 控制图使用 `On Playback Tick`。只有确实要求“每个物理步执行”时，
  才改用 `On Physics Step` 并将 Action Graph 的 `pipelineStage` 设为
  `PipelineStageOnDemand`；
- 有的 4.5 页面把物理 Prim 写作 `Physics Scene`，有的写作 `Simulation Scene`。本机
  `Create > Physics` 菜单出现哪个就选哪个；创建后 Stage 中都应得到一个
  `PhysicsScene` 类型 Prim。这是 4.5 官方页面本身的命名差异，不是让用户寻找两个对象。

## 2. 开始前确认

### 2.1 软件和项目

1. 启动 Isaac Sim。
2. 点击顶部 `Help > About`，确认版本为 `4.5.0`。
3. 确认以下文件存在：

   ```text
   /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/
   ├── assets/DM-rebot-dev-arm/source/reBot-DevArm_gripper.xml
   ├── config/dm_sim.yaml
   └── usd/DM-rebot-dev-arm/rebotarm_dm.usd
   ```

4. 若要使用 ROS 2，必须从已 source ROS 2 Humble 的终端启动 Isaac Sim。先关闭当前
   Isaac Sim，再在终端执行（安装路径按本机修改）：

   ```bash
   source /opt/ros/humble/setup.bash
   cd <isaac-sim-4.5.0-install-root>
   ./isaac-sim.sh
   ```

   不使用 ROS 2 时可跳过此项。

### 2.2 启用 4.5 扩展

1. 点击 `Window > Extensions`。
2. 在搜索框输入 `isaacsim.asset.importer.mjcf`。
3. 选择 **MJCF Importer**，确认右侧 `ENABLED` 开关打开；建议打开 `AUTOLOAD`。
4. 若需要 ROS 2，再搜索 `isaacsim.ros2.bridge`，打开 `ENABLED` 和 `AUTOLOAD`。
5. 若需要调关节，再搜索并启用：
   - `isaacsim.robot_setup.gain_tuner`
   - Physics Inspector（通常随 Physics UI 默认加载）
6. 关闭 Extensions 窗口。若 bridge 提示 ROS 2 library 加载失败，说明 Isaac Sim 启动前
   未正确 source ROS 2；不要继续搭 Action Graph，退出后按 2.1 重新启动。

## 3. 建立空白 Stage

### 3.1 新建并保存

1. 点击 `File > New`。
2. 如果弹出“保存当前 Stage”提示，按实际需要先保存；不要覆盖项目正式 USD。
3. 新 Stage 通常已有 `/World`、`/Environment/defaultLight`。在 Stage 面板单击 `/World`，
   右键菜单若显示 `Set as Default Prim`，点击它；若显示为不可用或已经是默认 Prim，则无需操作。
4. 点击 `File > Save As`，建议保存到项目外的实验目录，文件名例如：

   ```text
   rebot_gui_45_01_empty.usd
   ```

### 3.2 确认 Stage 单位

Isaac Sim 新 Stage 默认采用 Z-up、米制。不要仅凭视觉判断：

1. 点击 `Window > Stage` 和 `Window > Property`，保证两个面板可见。
2. 在 Stage 面板空白处单击，或打开 `Window > Layer` 后选择 Root Layer。
3. 检查 Stage/Layer 属性：
   - `Up Axis = Z`
   - `Meters Per Unit = 1.0`
   - `Time Codes Per Second = 60`（本项目渲染基准 60 Hz）

`Meters Per Unit` 若不是 1.0，先改为 1.0 再导入。项目 MJCF 中机械臂主体 STL 按米，
手指 mesh 已在 XML 中显式使用 `scale="0.001 0.001 0.001"`，不要再给整机乘 0.001。

## 4. 创建 Physics Scene、地面和灯光

### 4.1 Physics Scene

1. 点击 `Create > Physics > Physics Scene`。若本机菜单文字是
   `Create > Physics > Simulation Scene`，选择它。
2. 在 Stage 中选择新建的 `/World/physicsScene`（实际自动名称可能为 `/physicsScene`）。
3. 在右侧 Property 面板展开物理场景属性，按下表设置：

| 4.5 Property 字段 | 值 | 说明 |
|---|---:|---|
| Gravity Direction | `(0, 0, -1)` | Z 轴向上场景的重力方向 |
| Gravity Magnitude | `9.81` | `m/s²`，不是 `981`；后者只适用于厘米单位 Stage |
| Simulation Steps Per Second | `400` | 对应 `dt = 0.0025 s` |
| Solver Type | `TGS` | 项目 `dm_sim.yaml` 的目标值 |
| Enable GPU Dynamics | `Off` | 单台机械臂优先 CPU；符合 4.5 官方环境教程建议 |
| Broadphase Type | `MBP` | CPU 场景使用 |
| Enable Stabilization | `On` | 项目配置值 |

`Enable CCD` 不是应该在 Physics Scene 上“一键全局打开”的同名字段。CCD 是刚体级设置，
只对高速、薄小动态物体按需添加；机械臂常规低速关节不要为了对齐 YAML 而到处勾选。

### 4.2 地面

1. 点击 `Create > Physics > Ground Plane`。
2. 在 Stage 中选中创建的 Ground Plane 根 Prim。
3. 在 Property 的 Transform 中设置 `Translate Z = -0.04`，对应
   `dm_sim.yaml: scene.ground_z`。
4. 不要给 Ground Plane 添加 Rigid Body；它应是静态 Collider。

### 4.3 灯光

新 Stage 已有 `defaultLight`，先保留它即可。若场景过暗：

1. 选择 `/Environment/defaultLight`。
2. 在 Property 的 `Main > Intensity` 设置 `300` 作为起点。
3. 需要轮廓光时点击 `Create > Light > Distant Light`（部分 4.5 菜单显示
   `Create > Lights > Distant Light`）。
4. 选择新灯，在 `Main > Intensity` 从 `500` 开始调整。

灯光不是物理配置，不要靠修改材质颜色补偿曝光。

## 5. 用 4.5 MJCF Importer 加入 reBotArm

### 5.1 打开导入器

1. 点击 `File > Import`。
2. 在文件选择器中进入：

   ```text
   /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/
   assets/DM-rebot-dev-arm/source/
   ```

3. 选择 `reBot-DevArm_gripper.xml`，点击文件选择器的 `Open`。
4. 若 `.xml` 不在可导入格式中，返回 2.2 检查
   `isaacsim.asset.importer.mjcf`，不要改用 URDF Importer。

### 5.2 Import Options

4.5 Import Options 的具体排列可能随窗口宽度变化，但字段含义不变。使用以下值：

| Import Options | 选择/数值 | 原因 |
|---|---|---|
| USD Output | 选择一个实验输出目录 | 不写入 source 目录，不覆盖正式资产 |
| Fix Base Link | `On` | 本项目是固定底座机械臂 |
| Import Sites | `On` | 保留标定 site 和 `eef_trace_site` |
| Make Instanceable | `Off` | 后续需要逐 Prim 检查和修改 |
| Create Physics Scene | `Off` | 第 4 节已经创建，避免出现两个 PhysicsScene |
| Import Inertia Tensor | `On` | 使用 XML 的质量、质心、主惯量和主轴 |
| Self Collision | `Off` | 项目生成器明确关闭 articulation self-collision |
| Collision From Visuals | `On` | XML 的 geom 是 mesh，需生成碰撞 |
| Collision Type | `Convex Hull` | 机械臂 link 先用稳定的凸包；不要用 Triangle Mesh 动态碰撞 |
| Distance Scale | `1.0` | Stage 和 MJCF 均按米；手指缩放已写在 XML |
| Density | 保持默认 | 有显式 inertial 的 link 不用 density 反推质量 |

说明：若 4.5 导入窗口没有显示某一行，就不要在其它相似字段上猜测填写。导入后按第 6 节
直接检查 USD 属性。项目的正式生成脚本仍是最终权威。

5. 点击导入窗口右下角 `Import`。
6. 导入结束后关闭窗口。
7. 在 Stage 中确认出现 `/World/base_link`。**本项目正确 robot prim path 是
   `/World/base_link`，不是 `/World/reBotArmDM`。**
8. `File > Save As` 保存为 `rebot_gui_45_02_robot.usd`。

## 6. 导入后必须逐项核对

### 6.1 结构和关节父子关系

展开 `/World/base_link`。Importer 可能将 joint Prim 放在各 child link 下，不要求它们集中在
`Joints` 文件夹；判断正确性的依据是名称和 Body0/Body1，不是 Stage 面板是否“整齐”。

| Joint | Parent body | Child body | XML child 相对位姿 `(pos; quat wxyz)` | Axis | 类型 |
|---|---|---|---|---|---|
| joint1 | base_link | link1 | `(-0.00008416,0,0.08465); identity` | `(0,0,1)` | Revolute |
| joint2 | link1 | link2 | `(0.020084,0.031625,0.05555); (0.707105,-0.707108,0,0)` | `(0,0,-1)` | Revolute |
| joint3 | link2 | link3 | `(-0.264,0,0); identity` | `(0,0,1)` | Revolute |
| joint4 | link3 | link4 | `(0.2426,-0.054,-0.001625); identity` | `(0,0,1)` | Revolute |
| joint5 | link4 | link5 | `(0.078308,-0.0375,-0.03); (0.707105,-0.707108,0,0)` | `(0,0,1)` | Revolute |
| joint6 | link5 | link6 | `(0.028008,0,0.04); (0.707105,0,0.707108,0)` | `(0,0,1)` | Revolute |
| left_finger | link6 | left_finger_link | `(-0.005,0.078,0.076); identity` | `(0,-1,0)` | Prismatic |
| right_finger | link6 | right_finger_link | `(-0.005,-0.078,0.076); identity` | `(0,1,0)` | Prismatic |

在 Stage 搜索框逐个输入 joint 名称，选中 joint，在 Property 检查：

1. `Physics > Body 0` 指向 Parent；
2. `Physics > Body 1` 指向 Child；
3. 不要交换 Body 0/1。4.5 官方物理文档明确要求 articulation 中 Body 0 为父、Body 1 为子；
4. joint2 的 XML axis 为 `(0,0,-1)`，但 Importer 可能通过局部关节 frame 旋转后显示
   USD 的 X/Y/Z 轴枚举。应以实际运动方向和限位验证，不能仅比较一个轴字符。

### 6.2 质量、质心和惯量

选择每个 link 的 Rigid Body Prim，在 Property 展开 `Mass`。至少核对：

| Link | Mass (kg) | Center of Mass (m) | Diagonal Inertia (kg·m²) |
|---|---:|---|---|
| link1 | 0.1613 | `(0.000113615,-0.00061632,0.0236476)` | `(0.000272817,0.000213413,0.000154640)` |
| link2 | 1.3266 | `(-0.132256,-0.0030617,-0.0308307)` | `(0.0128139,0.0125599,0.000733734)` |
| link3 | 0.8353 | `(0.12104,-0.0536211,-0.0310138)` | `(0.00648251,0.00632715,0.000467564)` |
| link4 | 0.5200 | `(0.0608201,-0.0511712,-0.0302995)` | `(0.000889485,0.000667165,0.000323211)` |
| link5 | 0.3830 | `(-0.00502802,0.00000173866,0.0386233)` | `(0.000217429,0.000212237,0.000157335)` |
| link6 | 0.8663 | `(0.00000550089,-0.0000371802,0.0578217)` | `(0.00127909,0.000983777,0.000513199)` |
| left_finger_link | 0.034796 | `(0.017344,-0.0060692,0)` | `(0.0000248003,0.0000141700,0.0000120797)` |
| right_finger_link | 0.034796 | `(0.017344,0.0060692,0)` | `(0.0000248002,0.0000141700,0.0000120798)` |

还要检查 `Principal Axes`/惯量旋转。不能只录入 diagonal inertia 而丢弃 XML 的 inertial
quaternion，否则惯性张量方向会错。base_link 和 gripper_base 在 XML 中没有显式 inertial，
由几何/导入器处理；不要凭空填质量。

### 6.3 关节限位和 drive

Revolute Joint 的 USD Property 通常以 **degree** 显示角度。项目 YAML 使用 rad，换算值如下：

| Joint | Lower (rad / deg) | Upper (rad / deg) | Stiffness | Damping | Max Force | Max Joint Velocity |
|---|---:|---:|---:|---:|---:|---:|
| joint1 | -2.80 / -160.428 | 2.80 / 160.428 | 800 | 6 | 27 Nm | 1 rad/s |
| joint2 | -3.14 / -179.909 | 0 / 0 | 800 | 6 | 27 Nm | 1 rad/s |
| joint3 | -3.14 / -179.909 | 0 / 0 | 800 | 6 | 27 Nm | 1 rad/s |
| joint4 | -1.87 / -107.143 | 1.57 / 89.954 | 600 | 2 | 7 Nm | 1 rad/s |
| joint5 | -1.57 / -89.954 | 1.57 / 89.954 | 600 | 2 | 7 Nm | 1 rad/s |
| joint6 | -3.14 / -179.909 | 3.14 / 179.909 | 600 | 2 | 7 Nm | 1 rad/s |

对每个 `joint1`～`joint6`：

1. 在 Stage 中选中 joint；
2. Property 中展开 `Revolute Joint`，检查 Lower/Upper Limit；
3. 展开 `Angular Drive`。若没有，点击 Property 面板顶部 `Add`，选择
   `Physics > Angular Drive`；
4. 设置 `Drive Type = Force`、`Target Position = 0 deg`、`Target Velocity = 0`；
5. 按表填写 `Stiffness`、`Damping`、`Max Force`；
6. `Max Joint Velocity` 若未显示，点击 `Add > Physics > PhysX Joint` 后查找该字段；
   Property 若以 degree/s 显示，`1 rad/s = 57.296 deg/s`。

注意：XML actuator 的 `forcerange` 是 `joint1～3 ±150`、`joint4～6 ±100`，但项目
`dm_sim.yaml` 和生成脚本最终覆盖为表中的 `27/7`；GUI 应使用项目最终值。

夹爪参数：

| Joint | Limit | Axis | Drive |
|---|---|---|---|
| left_finger | 0 ～ 0.051 m | `(0,-1,0)` | Linear Position Drive |
| right_finger | 0 ～ 0.051 m | `(0,1,0)` | 不添加独立 Drive，使用 mimic |

选择 `left_finger`，若没有 Linear Drive，点击 `Add > Physics > Linear Drive`，设置：

```text
Drive Type         Force
Target Position    0.0 m
Target Velocity    0.0 m/s
Stiffness          2000
Damping            174
Max Force          50 N
Max Joint Velocity 0.2 m/s
Armature            0.243
Joint Friction      1.0
```

`Damping = 174` 来自 actuator `kv=124` 加 passive joint damping `50`，不是随意经验值。

### 6.4 夹爪 mimic 的 4.5 限制

XML 的 equality 表示两指等位移，两个滑轨 axis 已相反。项目生成器在
`right_finger` 上写入 PhysX Mimic Joint API：

```text
reference joint = left_finger
gearing          = -1.0
offset           = 0.0
right_finger     无独立 linear drive
```

Isaac Sim 4.5 支持 articulation mimic joint，但不同 4.5 UI 布局不一定完整暴露 reference
axis 等字段。操作原则：

1. 选择 `right_finger`，确认不存在 Linear Drive；有则在对应 API 行的菜单中 Remove；
2. 点击 `Add > Physics`，仅当列表明确出现 `Mimic Joint` 时才添加；
3. 将 reference 拖入/选择为 `left_finger`，gearing 填 `-1`，offset 填 `0`；
4. 若 `Add` 列表没有 Mimic Joint，**不要改成两个独立 drive**。使用项目正式资产
   `rebotarm_dm.usd`，或运行 `build_dm_asset.sh` 让脚本准确写入 schema。

### 6.5 Articulation

1. 在 Stage 选择 articulation root（导入结果通常是固定根 joint 或 `/World/base_link` 上
   已存在 Articulation Root API）。
2. 在 Property 搜索 `Articulation`。若完全没有，选择正确根 Prim，点击
   `Add > Physics > Articulation Root`。
3. 展开 `PhysX Articulation`，设置：
   - `Solver Position Iteration Count = 32`
   - `Solver Velocity Iteration Count = 4`
   - `Enabled Self Collisions = Off`
4. 整机应是一个 articulation，DOF 顺序应为：

   ```text
   joint1 joint2 joint3 joint4 joint5 joint6 left_finger right_finger
   ```

不要在每个 link 上重复添加 Articulation Root，也不要创建额外 `/World/reBotArmDM` 包裹
现有 articulation；这会让项目脚本的 `/World/base_link` 路径失效。

## 7. 用 4.5 GUI 检查和移动关节

### 7.1 Physics Inspector（关节滑块）

1. 单击顶部 Play（三角形）至少运行一个物理 step；Gain Tuner/Inspector 在仿真从未运行时
   可能无法发现 articulation。
2. 点击 Pause，避免在编辑属性时机器人持续运动。
3. 点击 `Tools > Physics > Physics Inspector`。
4. 在 Inspector 的 articulation/selection 区选择 `/World/base_link` 对应的 articulation。
5. 展开关节列表，先只移动 `joint1`，目标不超过 `5 deg`。
6. 依次测试 joint2～joint6，每次恢复到初始值后再测试下一个。
7. 测试 `left_finger`：使用 `0.005 m` 小位移，观察 right_finger 是否等量反向运动。

若 Inspector 没有机器人：先 Play 一次；再检查 Articulation Root，而不是反复添加 joint。

### 7.2 Gain Tuner

1. 点击 `Tools > Robotics > Asset Editors > Gain Tuner`。
2. 在 Articulation/Robot 下拉框选择导入机器人。
3. 若列表为空，Play 后 Pause，再重新选择。
4. 先查看每个 DOF 的 stiffness/damping 是否与第 6.3 节一致。
5. 调参只在实验 USD 中进行：振荡先增加 damping；跟随慢先确认没有达到 Max Force，
   再小幅增加 stiffness。

## 8. 创建桌面、物体和物理材料

### 8.1 桌面

项目 YAML 的桌面是中心坐标加完整尺寸：

```text
position = (0.25, 0.0, -0.02) m
size     = (0.90, 0.70, 0.04) m
```

1. 点击 `Create > Shapes > Cube`。
2. Stage 中重命名为 `Table`。
3. 在 Transform 设置：
   - Translate `(0.25, 0, -0.02)`
   - Scale `(0.45, 0.35, 0.02)`，因为 4.5 默认 Cube 边长为 2 个 stage unit；
4. 点击 Property 顶部 `Add > Physics > Collider`。
5. 不添加 Rigid Body，使其保持静态。

### 8.2 抓取物体

正式场景使用香蕉，不是旧文档中的 5 cm 方块：

```text
asset_path   assets/banana/bananas_1k.usdc
source prim  /bananas/bananas_a
position     (0.438, 0, 0.019) m
rotation     (0, 0, 0) deg
scale        (1, 1, 1)
mass         0.12 kg
collision    Convex Decomposition
```

GUI 操作：

1. 在 Content Browser 导航到
   `reBot-Isaacsim/assets/banana/bananas_1k.usdc`；若 Content Browser 不可见，点击
   `Window > Browsers > Content`。
2. 双击是打开资产，不是加入当前 Stage；应将文件从 Content Browser 拖到 Viewport/Stage，
   以 **Reference** 方式加入。
3. 展开 reference，选择 `/bananas/bananas_a` 对应 Prim，设置上述 Transform。
4. 点击 `Add > Physics > Rigid Body`。
5. 点击 `Add > Physics > Mass`，设置 `Mass = 0.12`。
6. 点击 `Add > Physics > Collider`，在 Collider 的 Approximation 选择
   `Convex Decomposition`。
7. 香蕉是可能需要 CCD 的高速薄小物体：选择其 Rigid Body，点击
   `Add > Physics > PhysX Rigid Body`，只在确有穿透问题时打开 `Enable CCD`。

### 8.3 Physics Material

在 4.5 中创建 Physics Material：

1. 点击 `Create > Physics > Physics Material`；
2. 在 Stage 中将三个 material 分别重命名为 `FingertipMaterial`、`ObjectMaterial`、
   `TableMaterial`；
3. 在 Property 填写：

| Material | Static Friction | Dynamic Friction | Restitution |
|---|---:|---:|---:|
| FingertipMaterial | 1.5 | 1.5 | 0 |
| ObjectMaterial | 1.8 | 1.8 | 0 |
| TableMaterial | 1.0 | 1.0 | 0 |

4. 将 Physics Material 从 Stage 拖到目标 collision Prim；若拖放未绑定，选中 collision，
   在 Property 的 Physics Material relationship 中选择对应 material。
5. Fingertip 只绑定左右手指 collision，Object 绑定香蕉 collider，Table 绑定桌面 collider。

## 9. ROS 2 Joint State Action Graph（严格按 4.5）

当前仓库主控制通道是 UDP；本节是可选的 ROS 2 Bridge 验证。ROS 2、UDP sender、
Physics Inspector 和 GUI Drive Target 一次只能有一个命令源。

### 9.1 创建图和节点

1. 确认 2.2 已启用 `isaacsim.ros2.bridge`。
2. 点击 `Create > Visual Scripting > Action Graph`。
3. 在弹窗中保留默认 Graph Path（例如 `/World/ActionGraph`），点击 `OK`。
4. 若 Graph Editor 未自动打开，点击 `Window > Visual Scripting > Action Graph`；某些
   4.5 布局入口显示在 `Window > Graph Editors > Action Graph`。
5. 在图编辑器空白处按 `Tab` 或右键 `Add Node`，逐个搜索并加入：
   - `On Playback Tick`
   - `Isaac Read Simulation Time`
   - `ROS2 Publish Joint State`
   - `ROS2 Subscribe Joint State`
   - `Articulation Controller`

这里**不要加入 `Isaac Read Joint State`**；那是 6.0 的 publisher migration 流程。

### 9.2 设置节点 Property

逐个点击节点，在右侧 Property/节点输入区设置：

| Node | Input | 值 |
|---|---|---|
| ROS2 Publish Joint State | `targetPrim` | `/World/base_link` 的 articulation root Prim |
| ROS2 Publish Joint State | `topicName` | `joint_states`（UI 有 namespace 时不要重复写 `/`） |
| ROS2 Subscribe Joint State | `topicName` | `joint_command` |
| Articulation Controller | `targetPrim` | 与 publisher 相同的 articulation root |
| Articulation Controller | `robotPath` | 仅在不用 targetPrim 时填 `/World/base_link`，二选一 |

设置 `targetPrim` 的推荐按钮操作：点击字段右侧 `+`/`Add Target`，再到 Stage 选择
articulation root，然后点击 `Add`/`Select`。不要把 visual mesh、link1 或单个 joint 填进去。

### 9.3 连线

从输出端口拖到输入端口，按下列关系连接：

```text
On Playback Tick.tick
  -> ROS2 Publish Joint State.execIn
  -> ROS2 Subscribe Joint State.execIn
  -> Articulation Controller.execIn

Isaac Read Simulation Time.simulationTime
  -> ROS2 Publish Joint State.timeStamp

ROS2 Subscribe Joint State.jointNames
  -> Articulation Controller.jointNames
ROS2 Subscribe Joint State.positionCommand
  -> Articulation Controller.positionCommand
ROS2 Subscribe Joint State.velocityCommand
  -> Articulation Controller.velocityCommand
ROS2 Subscribe Joint State.effortCommand
  -> Articulation Controller.effortCommand
```

一个输出端口可以连接多个 execIn；不要把三个控制节点首尾串成“publish 完才 subscribe”。

### 9.4 验证

1. 保存 USD，点击 Play。
2. 在已 source 同一 ROS_DOMAIN_ID 的终端执行：

   ```bash
   ros2 topic list
   ros2 topic echo /joint_states
   ```

3. 先只给 joint1 发送小目标：

   ```bash
   ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
     "{name: ['joint1'], position: [0.05]}"
   ```

4. joint1 正确后再测试六轴，不要首先发送接近限位的姿态：

   ```bash
   ros2 topic pub --once /joint_command sensor_msgs/msg/JointState \
     "{name: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'], \
      position: [0.0, -0.1, -0.1, 0.0, 0.0, 0.0]}"
   ```

若 `/joint_states` 不存在：先看 Isaac Sim Console 中 bridge 是否加载成功，再检查 Play 状态、
`targetPrim` 和 ROS_DOMAIN_ID；不要通过乱换 topic 名掩盖 bridge 启动失败。

## 10. 分阶段验收与故障定位

每个阶段另存一个 USD：

```text
rebot_gui_45_01_empty.usd
rebot_gui_45_02_robot.usd
rebot_gui_45_03_physics_checked.usd
rebot_gui_45_04_scene.usd
rebot_gui_45_05_ros2.usd
```

### 10.1 Play 前

- [ ] `/World/base_link` 存在且没有额外缩放；
- [ ] Physics Scene 只有一个，400 steps/s、TGS、重力 9.81；
- [ ] articulation 只有一个，8 个 DOF 名称正确；
- [ ] joint2 的父子关系和负轴没有被“看起来不顺”而擅自修改；
- [ ] Revolute 限位没有把 rad 数字直接填进 degree 字段；
- [ ] right_finger 没有独立 drive；
- [ ] 初始状态没有明显 collider 穿透。

### 10.2 Play 后

- [ ] base 固定且不下落；
- [ ] 单关节小目标方向、限位正确；
- [ ] Physics Inspector 显示 8 个 DOF；
- [ ] 两指等量反向开合；
- [ ] 香蕉落在桌上，不穿透、不持续弹跳；
- [ ] Console 无 articulation topology、invalid body、NaN 或 mimic 错误。

常见现象与检查顺序：

| 现象 | 首先检查 | 不应采用的“修复” |
|---|---|---|
| 整机掉落 | Import 的 Fix Base Link、fixed root joint | 把 base 质量设成极大 |
| 一按 Play 爆开 | 初始碰撞穿透、joint frame、重复 Rigid Body | 关闭所有碰撞 |
| joint2 反向 | XML 负轴、局部 frame、Body0/Body1 | 交换上下限 |
| Inspector 无 DOF | 是否 Play 过、Articulation Root 是否唯一 | 每个 link 都加 root |
| 右指不动 | mimic reference/gearing、右指是否误加 drive | 两个 ROS controller 同发目标 |
| ROS 2 无 topic | bridge 加载日志、Play、domain、targetPrim | 引入 6.0 Read Joint State 节点 |

## 11. 正式资产与 GUI 实验的关系

GUI 验证完成后，不要把实验 USD 当作新的唯一真源。结构、质量、关节 frame 等修改应回到
MJCF；Isaac/PhysX 参数应回到 `dm_sim.yaml` 和生成脚本，然后运行：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim
./reBotArm_Isaacsim/build_dm_asset.sh
```

生成脚本会补齐 GUI/MJCF Importer 不便可靠完成的 mimic、最终 drive、材料和场景参数，并
打印 joint 列表与 mimic 检查结果。生成后用本文第 7 节再次检查正式
`usd/DM-rebot-dev-arm/rebotarm_dm.usd`。

## 12. Isaac Sim 4.5 官方参考

- [Isaac Sim 4.5 Environment Setup](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/gui/tutorial_intro_environment_setup.html)
- [Isaac Sim 4.5 Import MJCF](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/import_mjcf.html)
- [Isaac Sim 4.5 Physics Simulation Fundamentals](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/physics/simulation_fundamentals.html)
- [Isaac Sim 4.5 Physics Inspector](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/physics/joint_inspector.html)
- [Isaac Sim 4.5 Gain Tuner](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/ext_isaacsim_robot_setup_gain_tuner.html)
- [Isaac Sim 4.5 ROS 2 Bridge API](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.ros2.bridge/docs/index.html)
- [Isaac Sim 4.5 Release Notes](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/overview/release_notes.html)

以上链接均固定到 `/4.5.0/`。不要用 `latest` 页面判断 4.5 的按钮、节点或端口。
