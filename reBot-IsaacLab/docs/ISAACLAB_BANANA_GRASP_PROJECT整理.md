# reBot IsaacLab 香蕉抓取项目整理

本文档整理 `/home/yyf/Desktop/pythonProject/rebot/reBot-IsaacLab` 当前已经收敛后的主线，
用于快速熟悉项目、复现短测流程，以及区分“必须保留的代码”和“可重新生成的文件”。

## 1. 当前目标

项目目标是基于 Isaac Lab 做 reBot 机械臂香蕉抓取训练：

1. 在仿真中随机香蕉位置、摩擦、质量、光照和相机误差；
2. 使用腕部 RGB-D 图像和机器人本体状态作为策略输入；
3. 先复用 `reBot-Isaacsim` 已验证传统抓取器生成 teacher，再做 BC 行为克隆预训练；
4. 用 PPO 在接触、摩擦和随机化环境中继续微调；
5. 评估成功率，导出 TorchScript，为后续 sim2real 做准备。

当前主线 teacher 不再由 IsaacLab 自己重新跑 YOLO+SAM。它只读取 `reBot-Isaacsim`
现有流程发布到 `/tmp` 的 RGB-D 帧和抓取计划，然后转换成 IsaacLab BC 数据。

## 2. 最推荐的运行方式

进入项目目录：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-IsaacLab
```

一条命令跑完整短测闭环：

```bash
./run_quick_pipeline.sh
```

注意：`run_quick_pipeline.sh` 中的 teacher 采集不会启动 IsaacSim。采集前需要先在
`reBot-Isaacsim/reBotArm_Isaacsim` 中运行已经验证过的仿真 RGB-D 导出和感知抓取流程，
感知进程需要显式传入 `--source sim`，让 `/tmp/rebot_sim_rgbd.npz` 与
`/tmp/rebot_sim_grasp_plan.json` 持续更新。

它会依次执行：

```text
run_build_asset.sh
run_collect_teacher.sh
run_train_bc.sh
run_train_rgbd.sh
run_evaluate_rgbd.sh
run_export_rgbd.sh
run_visual_report.sh
```

短测默认值：

- teacher 采集：默认读取 1 个 IsaacSim 抓取计划，生成 3 个 BC 样本；
- BC：3 epoch；
- PPO：8 个环境，200 iteration；
- eval：4 个环境，32 episode。

正式长跑示例：

```bash
REBOT_RGBD_NUM_ENVS=16 REBOT_RGBD_ITERATIONS=1500 ./run_train_rgbd.sh
REBOT_RGBD_EVAL_NUM_ENVS=8 REBOT_RGBD_EVAL_EPISODES=256 ./run_evaluate_rgbd.sh
```

如果已经训练过，要接着最新 checkpoint 继续：

```bash
REBOT_RGBD_RESUME=1 REBOT_RGBD_NUM_ENVS=16 REBOT_RGBD_ITERATIONS=500 ./run_train_rgbd.sh
```

收集多条 IsaacSim teacher 计划：

```bash
REBOT_ISAACSIM_TEACHER_PLANS=20 ./run_collect_teacher.sh
```

## 3. 主线文件说明

### 3.1 顶层脚本

| 文件 | 作用 |
| --- | --- |
| `run_quick_pipeline.sh` | 一条命令跑完整短测闭环。 |
| `run_build_asset.sh` | 检查香蕉物理资产是否存在、结构是否正确。 |
| `run_collect_teacher.sh` | 主线：读取 reBot-Isaacsim 的 `/tmp` RGB-D 帧和抓取计划，转换成 BC teacher 数据。 |
| `run_real_policy_dry_run.sh` | sim2real dry-run：真机相机输入，网络输出动作，不连接机械臂。 |
| `run_train_bc.sh` | 用 teacher 数据做行为克隆预训练。 |
| `run_train_rgbd.sh` | 从 BC 权重或随机策略开始做 RGB-D PPO 训练。 |
| `run_evaluate_rgbd.sh` | 评估最新 PPO checkpoint 的抓取返回成功率。 |
| `run_export_rgbd.sh` | 导出最新 PPO checkpoint 为 TorchScript。 |
| `run_visual_report.sh` | 生成 HTML 可视化报告，不启动仿真。 |
| `run_watch_rgbd.sh` | 打开 Isaac Sim GUI，播放最新策略 rollout。 |

### 3.2 训练与评估脚本

| 文件 | 作用 |
| --- | --- |
| `scripts/collect_isaacsim_teacher.py` | 读取 reBot-Isaacsim 已验证抓取器输出，生成 IsaacLab BC teacher 数据。 |
| `scripts/train_rgbd_bc.py` | 训练 `RgbdActorCritic` 的 actor，输出 `rgbd_bc_policy.pt`。 |
| `scripts/train_rgbd_ppo.py` | RGB-D PPO 训练；存在 BC 权重时自动初始化。 |
| `scripts/evaluate_rgbd_policy.py` | 统计 episode 成功率和 Wilson 置信区间。 |
| `scripts/export_rgbd_policy.py` | 导出确定性 actor TorchScript。 |
| `scripts/make_visual_report.py` | 汇总 teacher/BC/PPO/eval/export 状态到 HTML。 |
| `scripts/watch_rgbd_policy.py` | GUI 中观察策略是否完成接近、闭爪、抬升和返回 ready。 |
| `scripts/real_rgbd_policy_dry_run.py` | 真机相机 + TorchScript 策略 dry-run，不发机械臂命令。 |

### 3.3 任务配置

| 文件 | 作用 |
| --- | --- |
| `source/rebot_isaaclab/rebot_isaaclab/tasks/banana_lift/banana_lift_env_cfg.py` | 机器人、香蕉、桌面、动作、奖励、随机化的基础配置。 |
| `source/rebot_isaaclab/rebot_isaaclab/tasks/banana_lift/banana_lift_vision_env_cfg.py` | 在基础环境上增加腕部 RGB-D 观测和视觉随机化。 |
| `source/rebot_isaaclab/rebot_isaaclab/tasks/banana_lift/mdp/events.py` | 相机标定误差和 DomeLight 光照随机化。 |
| `source/rebot_isaaclab/rebot_isaaclab/tasks/banana_lift/mdp/observations.py` | 把 RGB-D 归一化并展平为 RSL-RL 可接收的二维张量。 |
| `source/rebot_isaaclab/rebot_isaaclab/tasks/banana_lift/mdp/rewards.py` | 携物返回 ready 奖励和成功判据。 |
| `source/rebot_isaaclab/rebot_isaaclab/rgbd_actor_critic.py` | CNN + MLP Actor-Critic，读取 64x64x4 RGB-D 和 21 维本体状态。 |

## 4. 动作和观测定义

策略输入：

```text
21 维本体状态 + 64*64*4 RGB-D = 16405 维
```

本体状态为：

```text
joint_pos7 + joint_vel7 + last_action7
```

动作输出为 7 维：

| 维度 | 含义 |
| --- | --- |
| 0-2 | 末端 xyz 相对位移，环境中每步最大约 0.02 m。 |
| 3-5 | 末端 roll/pitch/yaw 相对旋转，环境中每步最大约 0.10 rad。 |
| 6 | 夹爪开合：正数张开，负数闭合。 |

抓取控制点不是 `gripper_base` 原点，而是沿夹爪局部 `+X` 偏移 `1.5 cm`，
这与传统夹取代码里“沿夹爪 x 延长 1.5 cm”的要求保持一致。

## 5. 香蕉资产说明

核心资产：

```text
assets/banana_physics.usda
```

它包含：

- 可见香蕉：引用 `../reBot-Isaacsim/assets/banana/bananas_1k.usdc`；
- 与 reBot-Isaacsim 一致的原香蕉 mesh `convexDecomposition` 碰撞；
- `0.12 kg` 质量、`1.8` 静/动摩擦和 `0.002 m` contact offset。

之前三段 capsule 的轮廓与可见香蕉不完全重合，会出现“夹爪看起来
还没碰到，香蕉已经移动”。当前已禁用这三段旧碰撞体，以原 mesh 碰撞为准。

## 6. 生成物和清理规则

这些是生成物，可以删除后重新运行流程生成：

| 路径 | 说明 |
| --- | --- |
| `data/rgbd_isaacsim_teacher_latest.pt` | 从 reBot-Isaacsim 抓取计划转换来的 teacher 数据。 |
| `exported/rgbd_bc_policy.pt` | BC 初始化权重。 |
| `exported/rgbd_policy_latest.pt` | PPO 导出的 TorchScript。 |
| `exported/rgbd_policy_latest.json` | TorchScript metadata。 |
| `logs/rsl_rl/rebot_banana_grasp_return_rgbd/` | PPO checkpoint 和 TensorBoard 事件。 |
| `results/eval_rgbd_*.json` | 评估结果。 |
| `reports/banana_grasp_report.html` | 可视化报告。 |
| `__pycache__/`、`.pytest_cache/`、`.ruff_cache/` | Python/测试/格式检查缓存。 |

这些不是垃圾，应该保留：

| 路径 | 说明 |
| --- | --- |
| `assets/banana_physics.usda` | 当前任务使用的香蕉物理资产。 |
| `assets/rebotarm_lab.usda` | reBot Isaac Lab 资产 wrapper。 |
| `source/rebot_isaaclab/` | 任务、网络、配置源码。 |
| `scripts/*.py` | 主流程脚本。 |
| `tests/` | 纯 Python 测试。 |

## 7. 开发迭代过程整理

本阶段一路迭代后，最终保留的主线是：

1. **资产阶段**
   - 接入 reBot DM USD；
   - 构建 `banana_physics.usda`；
   - 复用 reBot-Isaacsim 原香蕉 mesh 的 `convexDecomposition` 碰撞。

2. **基础环境阶段**
   - 配置桌面、香蕉、DomeLight；
   - episode 从与 IsaacSim teacher 一致的 ready_arm 附近开始；
   - 只对六个机械臂关节加入 ±0.05 rad 小扰动，home -> ready_pose 不交给网络学习；
   - 配置 relative differential IK；
   - 配置二值夹爪动作；
   - 配置接近、抬升、携物返回 ready 和成功奖励；不使用放置目标。

3. **视觉阶段**
   - 加入腕部 TiledCamera；
   - 输出 64x64 RGB-D；
   - 随机相机外参、焦距和光照。

4. **PPO 阶段**
   - 自定义 `RgbdActorCritic`；
   - 从 RGB-D 和本体状态直接输出 7 维动作；
   - 评估抓取返回成功率。

5. **模仿学习阶段**
   - 改为复用 reBot-Isaacsim 已验证传统抓取器作为 teacher；
   - 增加 BC 预训练；
   - PPO 训练时自动使用 BC 权重初始化。

6. **收尾阶段**
   - 增加单命令短流程；
   - 更新可视化报告；
   - 修复 export checkpoint 环境变量默认值提前求值的问题；
   - 清理旧迭代入口、缓存和旧生成产物。

## 8. 后续 sim2real 注意点

当前项目还不应该直接驱动真机。真机前必须确认：

- DM 机械臂型号和通信通道；
- 关节限位和夹爪限位；
- 手眼标定；
- 相机真实 RGB-D 输入是否与训练观测布局一致；
- 工作空间清空、急停可用；
- 真机端必须保留碰撞检查、限位、watchdog，不要绕过安全逻辑。

真机部署推荐流程：

1. 在仿真中用 `run_watch_rgbd.sh` 观察策略动作是否合理；
2. 用 `run_export_rgbd.sh` 导出 TorchScript；
3. 先运行 `run_real_policy_dry_run.sh`，只读真机相机并打印网络动作；
4. 在真机端只让网络输出受限末端增量和夹爪开合；
5. 网络输出必须经过安全限幅、工作空间裁剪和速度限制；
6. 先空夹爪低速测试，再放入香蕉。
