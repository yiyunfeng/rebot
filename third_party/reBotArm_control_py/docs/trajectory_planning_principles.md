# reBot-DevArm 轨迹规划原理

## 概述

该项目轨迹规划为**全部手写实现**，位于 `reBotArm_control_py/trajectory/` 包中，仅依赖 `numpy` + `pinocchio`（提供 SE(3) 运算、正运动学 FK、Jacobian）。不依赖 MoveIt、ROS、pyroboplan 等外部轨迹库。

轨迹规划分两个层级：

| 层级 | 文件 | 职责 |
|------|------|------|
| 笛卡尔空间采样 | [sampler.py](../reBotArm_control_py/trajectory/sampler.py) | SE(3) 测地线插值 + 时间剖面 |
| 关节空间跟踪 | [clik_tracker.py](../reBotArm_control_py/trajectory/clik_tracker.py) | CLIK 将笛卡尔轨迹转为关节轨迹 |
| 统一入口 | [trajectory_planner.py](../reBotArm_control_py/trajectory/trajectory_planner.py) | 组合采样 + 跟踪 + 统计 |
| 简单关节插值 | [rebotarm_endpose_controller.py:199-206](../reBotArm_control_py/controllers/rebotarm_endpose_controller.py#L199-L206) | safe_home 用关节空间最小 jerk |

数据流：

```
start_pose ──┐                 ┌── q_start
end_pose ────┤ sampler.py      ├── clik_tracker.py ──→ 关节轨迹点列表
duration ────┘ → 笛卡尔轨迹点  ─┘   (CLIK + 零空间)
               (SE(3) + 剖面)
```

---

## 1. 笛卡尔空间采样（sampler.py）

### 1.1 SE(3) 测地线插值

核心函数 `_se3_interpolate(a, b, s)` ([sampler.py:78-84](../reBotArm_control_py/trajectory/sampler.py#L78-L84))：

```python
def _se3_interpolate(a, b, s):
    return a * exp6(log6(a⁻¹ * b) * s)
```

**原理**：

1. 计算相对变换： `Δ = a⁻¹ * b` — 从 `a` 到 `b` 的 SE(3) 变换
2. 取李代数对数： `v = log6(Δ)` — 将 SE(3) 映射到 se(3)（6 维 twist 向量）得到最短路径的"速度"
3. 标量插值： `v * s` — 在 se(3) 线性空间中按比例 s ∈ [0,1] 缩放
4. 指数映射回 SE(3)： `exp6(v * s)` — 从李代数回到李群
5. 作用到起点： `a * exp6(...)` — 将插值后的相对变换应用到起点

这就是 SE(3) 流形上的**测地线（最短路径）**，等价于恒定螺旋运动（screw motion）。位置走直线、姿态走最短弧。

### 1.2 时间剖面

核心函数 `_apply_profile(t, profile, accel_ratio)` ([sampler.py:55-75](../reBotArm_control_py/trajectory/sampler.py#L55-L75))：

输入归一化时间 t ∈ [0, 1]，输出归一化弧长 s ∈ [0, 1]。

#### Linear

```
s = t
```

匀速运动。简单但起止有加速度跳变（不连续 jerk）。

#### Minimum Jerk

```
s(t) = 10t³ - 15t⁴ + 6t⁵
```

**推导**：最小化 jerk 泛函 ∫(d³s/dt³)² dt，满足边界条件：

- s(0)=0, s(1)=1（位置）
- s'(0)=0, s'(1)=0（零起止速度）
- s''(0)=0, s''(1)=0（零起止加速度）

解得该 5 次多项式。速度曲线为 `s'(t) = 30t² - 60t³ + 30t⁴`，呈钟形，在 t=0.5 时最大速度为 1.875（是线性速度的 1.875 倍）。

**优点**：起止无冲击，jerk 全局最优，适合大部分场景。

#### Trapezoid

```
加速段（t ≤ ta）:    s = ½ · vm/ta · t²
匀速段（ta < t ≤ 1-ta）: s = ½·vm·ta + vm·(t-ta)
减速段（t > 1-ta）:   s = 1 - ½·vm/ta · (1-t)²
其中 vm = 2/(1-ta)（梯形速度的峰值速度）
```

梯形速度剖面，加速比 `accel_ratio` 控制加速段时间占比（默认 0.25）。速度曲线为梯形，加速度分段常数。

**优点**：最大速度更低（相比 min jerk），适合有明确速度限制的场景。

### 1.3 离散采样

`plan_cartesian_geodesic_trajectory()` ([sampler.py:87-118](../reBotArm_control_py/trajectory/sampler.py#L87-L118))：

```python
n = max(2, ceil(duration / dt) + 1)
dt_sample = duration / (n - 1)
for i in range(n):
    t = i * dt_sample
    s = apply_profile(t / duration, profile)
    traj.add_point(t, se3_interpolate(start, end, s))
```

按 `dt`（默认 0.02s，即 50Hz）均匀离散时间轴，每个时间点先经时间剖面得到弧长 s，再做 SE(3) 测地线插值。

---

## 2. 关节空间跟踪（clik_tracker.py）

`track_trajectory()` ([clik_tracker.py:62-127](../reBotArm_control_py/trajectory/clik_tracker.py#L62-L127)) 将笛卡尔轨迹点逐点转为关节配置。

### 2.1 闭环逆运动学（CLIK）

对每个笛卡尔目标点，迭代求解 IK：

```
q_{k+1} = q_k + step_size * dq
```

其中 dq 通过 DLS 伪逆计算：

```
J = getFrameJacobian(q_k)                    # 6×n 末端雅可比
err = log6(oMf⁻¹ * T_target).vector          # se(3) 位姿误差（6 维 twist）
λ = damping * max(1.0, ||err|| * 10)         # Levenberg-Marquardt 自适应阻尼
dq = step_size * Jᵀ (JJᵀ + λI)⁻¹ err         # DLS 伪逆求解
```

**收敛条件**：`||err|| < tolerance`（默认 1e-4）或达到 `max_iter`（默认 200）。

**为什么用 DLS 而非直接伪逆**：当机器人接近奇异位形时，J 条件数很差，直接求 `J⁺` 会产生巨大的关节速度。DLS 在 JJᵀ 对角线上加阻尼 λ，等价于 L2 正则化：

```
min ||J·dq - err||² + λ||dq||²
```

既最小化末端误差，又惩罚大幅关节运动。

**自适应阻尼（Levenberg-Marquardt 风格）**：`λ = damping * max(1.0, ||err|| * 10)`。
误差大时阻尼大 → 梯度下降（稳定），误差小时阻尼小 → Gauss-Newton（快速收敛到精确解）。

### 2.2 零空间投影（关节限位避让）

当 `null_gain > 0` 时启用：

```
g = joint_limit_gradient(q)                   # 指向关节范围中心的梯度
dq += null_gain * (I - Jᵀ(JJᵀ + λI)⁻¹J) * g  # 零空间投影
```

**原理**：`(I - J⁺J)` 是零空间投影算子 — 它将任意向量投影到 Jacobian 的零空间中，即**不影响末端位姿的关节运动方向**。梯度 g 指向各关节范围中心：

```
g[i] = (hi[i] - q[i]) - (q[i] - lo[i]) ∝ hi - lo
      ─────────────────────────────────
             (hi[i]-q[i])*(q[i]-lo[i])
```

- 关节接近上限时 g < 0 → 推动向下
- 关节接近下限时 g > 0 → 推动向上
- 关节在中心时 g = 0

### 2.3 关节限位钳制

`_clamp_config()` ([clik_tracker.py:47-59](../reBotArm_control_py/trajectory/clik_tracker.py#L47-L59)) 在每次 IK 迭代后将 q 裁剪到 `[lowerPositionLimit, upperPositionLimit]`，防止 NaN 扩散。

### 2.4 IK 参数

```python
@dataclass
class IKParams:
    max_iter: int = 200      # 每点最大 IK 迭代次数
    tolerance: float = 1e-4  # se(3) 误差范数收敛阈值
    damping: float = 1e-6    # DLS 基础阻尼
    step_size: float = 0.8   # 迭代步长（< 1 防止过冲）
```

---

## 3. 轨迹规划器（trajectory_planner.py）

### 3.1 关节空间轨迹规划

`plan_joint_space_trajectory()` ([trajectory_planner.py:38-81](../reBotArm_control_py/trajectory/trajectory_planner.py#L38-L81))：

```
输入: q_start, q_end, duration
步骤:
  1. FK(q_start) → T_start
  2. FK(q_end)   → T_end
  3. plan_cartesian_geodesic_trajectory(T_start, T_end, duration) → 笛卡尔轨迹
  4. track_trajectory(笛卡尔轨迹, q_start) → 关节轨迹点列表
```

### 3.2 轨迹统计

`compute_traj_stats()` ([trajectory_planner.py:85-132](../reBotArm_control_py/trajectory/trajectory_planner.py#L85-L132))：

计算关节轨迹的跟踪质量：

| 指标 | 含义 |
|------|------|
| `success_rate` | IK 收敛点占比 |
| `max_ik_error` | 最大 se(3) 跟踪误差 |
| `avg_ik_error` | 平均 se(3) 跟踪误差 |

具体做法：对每个关节轨迹点做 FK，将实际位姿与参考笛卡尔轨迹点的位姿比较，取 log6 误差的范数。

---

## 4. 辅助实现

### 4.1 safe_home 关节空间最小 jerk

位于 [rebotarm_endpose_controller.py:199-206](../reBotArm_control_py/controllers/rebotarm_endpose_controller.py#L199-L206)：

```python
s = t / t_total
q[:, i] = q_start[i] + Δq[i] * (10s³ - 15s⁴ + 6s⁵)
```

这是在**关节空间**中对每个关节独立做最小 jerk 插值，用于 `safe_home()` 方法平滑回到零位。与笛卡尔测地线不同，这里不保证末端走直线，只保证各关节平滑运动。

---

## 5. 总结

```
┌─────────────────────────────────────────────────────┐
│                    三个时间剖面                       │
│  Linear  │  Min Jerk (默认)  │  Trapezoid           │
│  s=t     │  10t³-15t⁴+6t⁵   │  梯形速度             │
└──────────────────────┬──────────────────────────────┘
                       │ s ∈ [0,1] 归一化弧长
                       ▼
┌─────────────────────────────────────────────────────┐
│              SE(3) 测地线插值                         │
│  T(s) = T_start * exp6(log6(T_start⁻¹ * T_end) * s) │
│  = 恒定螺旋运动（位置直线 + 姿态最短弧）               │
└──────────────────────┬──────────────────────────────┘
                       │ 笛卡尔轨迹点（时间+SE(3)位姿）
                       ▼
┌─────────────────────────────────────────────────────┐
│              CLIK 跟踪（关节空间）                     │
│  DLS 伪逆 (JJᵀ+λI)⁻¹  + 自适应阻尼 λ                 │
│  零空间投影 (I-J⁺J)g — 关节限位避让                   │
│  关节限位钳制                                         │
└──────────────────────┬──────────────────────────────┘
                       │ 关节轨迹点（时间+关节角+收敛标记）
                       ▼
                  发送到电机执行
```

### 关键设计决策

1. **笛卡尔空间规划 + 关节空间跟踪的两阶段架构**：先规划末端位姿的几何路径，再用 CLIK 转为关节轨迹。比直接在关节空间插值更可控 — 末端路径是精确的测地线。
2. **最小 jerk 作为默认剖面**：起止零速度零加速度，无冲击，适合大多数操作任务。
3. **DLS + 自适应阻尼**：在奇异位形附近保持数值稳定，牺牲一点末端精度换取关节速度的平滑。
4. **零空间关节限位避让**：在不影响末端位姿的前提下，利用冗余自由度远离关节限位。
