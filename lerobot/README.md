# reBot B601 × LeRobot

这是一个只包含 reBot Arm B601 所需功能的轻量集成层。LeRobot 框架作为 Python 依赖安装，
本目录不复制 Seeed/Hugging Face 的完整仓库。

核心范围：

- B601：MIT 重力补偿拖动示教、POS_VEL 策略部署、7 维弧度制状态与动作。
- 单台 Orbbec DaBai DCW：同步 RGB、模型深度、原始 `uint16` 毫米深度。
- 数据：键盘夹爪、episode 录制、无损深度 sidecar、质量检查。
- 策略：ACT、Diffusion Policy 从头训练；π0、π0.5 官方基座微调和本地部署。

未经安全确认，机器人配置默认 `readonly`，不会使能电机。

## 独立环境

本项目使用 `rebot_lerobot`，不要在 `rebotarm_gpu` 中安装 LeRobot。后者保留给现有抓取和
机械臂工程。新环境还会屏蔽用户目录与 ROS Humble 的 Python/Pinocchio 动态库，避免串包。

```bash
cd /home/yyf/Desktop/pythonProject/rebot/lerobot
conda env create -f environment.yml
conda activate rebot_lerobot

# LeRobot 0.4.4 要求 torchvision >= 0.21；使用匹配的 CUDA 12.4 wheel。
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 安装官方框架；π0/π0.5 使用官方 0.4.x 专用 Transformers 分支。
pip install lerobot==0.4.4 torchcodec==0.2.1
pip install --no-deps \
  "git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
pip install "tokenizers>=0.21,<0.22" sentencepiece

# 安装轻量插件、本地机械臂依赖与 Orbbec SDK。
pip install -e ".[test]"
pip install meshcat pin matplotlib motorbridge
pip install ../rebot_grasp/sdk/pyorbbecsdk
```

当前机器从 ROS 终端启动 Conda，默认会继承 `/opt/ros/humble`。首次创建环境后执行：

```bash
conda env config vars set -n rebot_lerobot \
  PYTHONNOUSERSITE=1 PYTHONPATH= \
  LD_LIBRARY_PATH=/home/yyf/miniconda3/envs/rebot_lerobot/lib/python3.10/site-packages/cmeel.prefix/lib
conda deactivate
conda activate rebot_lerobot
```

## 拖动示教与夹爪

本方案没有 leader arm。机械臂在 `teach` 模式下使用 MIT 重力补偿，直接拖动 6 个关节；
夹爪不能靠拖动可靠示教，使用键盘记录：

- `Space`：开始或保存 episode。
- `O`：打开夹爪。
- `C`：以受限前馈力矩闭合夹爪。
- `S`：保持当前夹爪位置。
- `R`：归档当前尝试后重录，不直接删除采集文件。
- `Esc`：立即停止控制循环并失能电机。

示例（真机运行前必须核对 B601 型号、通信通道、限位、工作区净空与物理急停）：

```bash
rebot-record \
  --root ./datasets/pick_cube \
  --task "抓起方块并放入盒中" \
  --num-episodes 50 \
  --confirm-hardware-safe
```

中断后若数据集已保存至少一个 episode，可从已有数量继续录到指定总数：

```bash
rebot-record \
  --root ./datasets/pick_cube \
  --task "抓起方块并放入盒中" \
  --num-episodes 50 \
  --resume \
  --confirm-hardware-safe
```

`--num-episodes` 表示数据集最终总数，不是本次新增数量。零 episode 的中断尝试仍保存在
`discarded/`，这类目录不能续录，应使用新目录。

每次相机取帧同时得到一台 DaBai DCW 的同步 RGB 与 D2C 深度。训练使用固定量程映射后的
三通道深度图，`raw_depth/` 另外保存 `uint16` 毫米深度，不能用普通 8 位灰度图替代。
相机流固定配置为 `640×360@30 FPS`。DaBai DCW 不支持硬件帧同步，适配层沿用
`rebot_grasp` 的策略，对偶发不完整 RGB-D frameset 做有限重试，不会复用旧深度帧。

## 训练

先执行数据质量检查，再选择策略：

```bash
rebot-check-dataset --root ./datasets/pick_cube

rebot-train --policy act       --dataset-root ./datasets/pick_cube
rebot-train --policy diffusion --dataset-root ./datasets/pick_cube

rebot-download-models --model all
rebot-train --policy pi0  --dataset-root ./datasets/pick_cube
rebot-train --policy pi05 --dataset-root ./datasets/pick_cube
```

ACT 和 Diffusion Policy 从头训练。π0/π0.5 微调官方基础模型：RGB 映射到
`base_0_rgb`，深度映射到 `left_wrist_0_rgb`，缺少的第三相机由模型掩码补空。

## 本地部署

部署固定使用 POS_VEL，不能在同一连接中从 MIT 动态切换。初次验证默认限制为 10 Hz、
单步不超过 0.04 rad、关节速度不超过 0.3 rad/s：

```bash
rebot-deploy \
  --checkpoint ./outputs/act/checkpoints/last/pretrained_model \
  --dataset-root ./datasets/pick_cube \
  --task "抓起方块并放入盒中" \
  --confirm-hardware-safe
```

已在 `rebot_lerobot` 中验证 `torch 2.6.0+cu124` 可识别 RTX 4070 Ti SUPER 并执行 CUDA
张量计算。相机也已验证可连续返回 `640×360` RGB、模型深度和 `uint16` 毫米深度。
机械臂运动与拖动示教仍必须按前述安全条件分阶段验证，软件测试通过不代表真机运动测试已通过。
