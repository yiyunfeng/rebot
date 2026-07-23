# reBot Isaac Lab 香蕉抓取

该项目复用 `../reBot-Isaacsim` 中已经验证过的 reBot DM USD、末端相机和香蕉模型，使用本机
Isaac Lab `v2.1.1` 建立并行训练环境。它不会修改原有 Isaac Sim 抓取代码，也不会连接真机。

## 当前任务：抓取香蕉并返回 ready 姿态

已包含：

- reBot articulation 与香蕉刚体资产接入；
- 香蕉位置/朝向和小范围光照随机化；机器人、香蕉与桌面物理参数保持基准不变；
- 40 Hz relative differential IK 动作与二值夹爪动作；
- 接近、抬升、携物返回 ready、成功奖励及动作平滑惩罚；
- 64×64 RGB-D 端到端 CNN PPO 训练入口；
- 复用机械臂 USD 中的腕部相机，并保留小范围光照随机化；
- 自动成功率评估和 95% Wilson 置信区间；
- TorchScript 导出和 GUI 策略播放入口。
- 复用 reBot-Isaacsim 已验证传统抓取器的 teacher 数据转换、RGB-D 行为克隆预训练和 sim2real dry-run 入口。

当前主线只训练“抓取香蕉 → 安全撤离 → 保持夹紧返回 ready 姿态”。不训练放置点
跟踪、下降、松爪或放置。`reBot-Isaacsim` 原有抓取放置流程保持不变，IsaacLab 只从
它发布的 plan 中读取到 `return` 阶段，忽略后面的放置阶段。

训练 episode 从与 IsaacSim teacher 一致的 `ready_arm` 附近开始。`home -> ready_pose`
属于确定性的安全准备动作，应由传统控制完成，不放进 RGB-D 策略学习数据。

## 使用顺序

以下命令都在本目录执行，并固定使用已有 `isaaclab` conda 环境。

```bash
# 0. 直接跑完整短测闭环：资产检查 -> IsaacSim teacher -> BC -> PPO -> 评估 -> 导出 -> 报告
#    注意：teacher 采集前，需要先按下面第 2 步启动 reBot-Isaacsim 抓取流程。
./run.sh quick

# 1. 一次性生成带刚体、质量、摩擦和 convex decomposition 的香蕉资产
./run.sh build-asset

# 2. 采集 IsaacSim teacher 示教数据；先让 reBot-Isaacsim 生成 /tmp 帧和抓取计划
./run.sh collect-teacher

# 3. 用 teacher 数据做 RGB-D 行为克隆预训练；默认 3 epoch 短测
./run.sh train-bc

# 4. RGB-D CNN PPO 训练；默认 8 环境、200 iteration，存在 BC 权重时自动初始化
./run.sh train

# 4.0 打开 GUI 观察 4 个环境一边仿真一边训练；仍会正常保存 checkpoint
REBOT_RGBD_GUI=1 REBOT_RGBD_NUM_ENVS=4 REBOT_RGBD_ITERATIONS=50 ./run.sh train

# 4.1 接着最新 RGB-D checkpoint 继续训练，而不是从头开始
REBOT_RGBD_RESUME=1 REBOT_RGBD_NUM_ENVS=16 REBOT_RGBD_ITERATIONS=500 ./run.sh train

# 4.2 正式长跑示例
REBOT_RGBD_NUM_ENVS=16 REBOT_RGBD_ITERATIONS=1500 ./run.sh train

# 5. 自动评估最新 RGB-D checkpoint；默认 4 环境、32 episode 短测
./run.sh evaluate

# 6. 导出最新 RGB-D checkpoint 为真机部署用 TorchScript
./run.sh export

# 7. 生成一页可视化 HTML 报告，帮助直观看懂当前进度
./run.sh report

# 8. 打开 Isaac Sim GUI，直接看最新 RGB-D 策略跑几个 episode
./run.sh watch

# 9. 真机 RGB-D 策略 dry-run：只读相机和打印网络动作，不连接机械臂
./run.sh real-dry-run

# 10. 轻量单元测试：不启动 Isaac Sim
./run.sh test

# 11. 真机 RGB-D 策略安全执行入口：默认不会连接，必须显式打开环境变量并人工确认
REBOT_REAL_POLICY_ENABLE=1 ./run.sh real-execute
```

日常使用只需要记住 `./run.sh <command>`。

IsaacSim teacher 数据保存在 `data/rgbd_isaacsim_teacher_latest.pt`，BC 初始化权重保存在
`exported/rgbd_bc_policy.pt`。RGB-D PPO 训练模型保存在 `logs/rsl_rl/rebot_banana_grasp_return_rgbd/`，
评估 JSON 保存在 `results/`。
这些都是生成物，可以删除后重新跑 `./run.sh quick` 生成。

`./run.sh collect-teacher` 不会启动 IsaacSim。采集 teacher 前，先在另外两个终端运行已经验证过的
IsaacSim 抓取流程：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/reBotArm_Isaacsim
./run_sim_rgbd.sh

cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/reBotArm_Isaacsim
./run_sim2real_perception.sh --source sim
```

看到 `/tmp/rebot_sim_rgbd.npz` 和 `/tmp/rebot_sim_grasp_plan.json` 持续更新后，再回到
`reBot-IsaacLab` 执行：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-IsaacLab
REBOT_ISAACSIM_TEACHER_PLANS=20 ./run.sh collect-teacher
```

每个 IsaacSim 抓取计划只展开到 `return`：ready/pregrasp 接近、直线插入、闭爪、
撤离和携物返回 ready。默认每条成功计划约 43 个样本，可通过
`REBOT_TEACHER_APPROACH_STEPS`、`REBOT_TEACHER_INSERT_STEPS`、
`REBOT_TEACHER_CLOSE_STEPS`、`REBOT_TEACHER_RETREAT_STEPS`、
`REBOT_TEACHER_RETURN_STEPS` 微调。旧的 31 样本数据不含返回阶段，BC 会明确拒绝，
需要重新执行 `./run.sh collect-teacher`。

当前宿主机 NVIDIA 驱动为 `535.309.01`，高于 Isaac Sim 4.5 / Isaac Lab 本地文档验证过的
`535.129`。由于 Vulkan 版本字段会把 minor=309 截断显示成 53，Isaac Sim 全局用户配置已关闭
错误的版本比较；这不会禁用 RTX，也不会改变 CUDA/Vulkan 实际加载的驱动。

## 成功定义

一次成功必须同时满足：香蕉刚体原点高于 `0.070 m`、香蕉与 TCP 距离小于 `0.12 m`，
并且六个机械臂关节与默认 ready 关节角的最大误差小于 `0.12 rad`。因此仅仅抬起、
把香蕉抛起或空手返回都不会记为成功。

## 后续里程碑

1. **RGB-D PPO**：训练并评估“抓取后返回 ready”的成功率。
2. **GUI 观察**：用 `./run.sh watch` 检查策略是否完成接近、闭爪、抬升和返回。
3. **模仿学习**：主线复用 reBot-Isaacsim 已验证传统抓取器生成 teacher，再做 BC 预训练。
4. **PPO 微调**：从 BC 权重继续 PPO，使用逐级扩大随机化范围的 curriculum。
5. **Sim2Real**：导出 TorchScript；`./run.sh real-dry-run` 先验证真机相机和网络动作链路，
   `./run.sh real-execute` 在显式安全确认后才允许发送小步 TCP 动作。

真机部署前必须重新确认 DM 型号、通信通道、关节/夹爪限位、手眼标定、工作空间和急停状态；
本项目不会绕过任何已有安全检查。
