# MoveIt 运动接口方式记录

本文记录项目里几种常见 MoveIt / 控制器调用方式，以及它们在
`simple_pick_place.py`、`moveit_pick_place.py`、`pick_place.py` 中的区别。

## 1. `/compute_ik`

代码示例：

```python
self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
```

用途：

- 只做逆运动学求解。
- 输入末端位姿，输出一组关节角。
- 不负责路径规划。
- 不负责执行。

典型流程：

```text
目标 TCP 位姿
  -> /compute_ik
  -> 得到 joint1~joint6 目标角
  -> FollowJointTrajectory 发给控制器
```

项目对应：

- `src/rebotarm_gazebo/src/rebotarm_gazebo/simple_pick_place.py`
- `src/rebotarm_gazebo11/src/rebotarm_gazebo11/simple_pick_place.py`

特点：

- 代码简单，适合学习。
- 运动快。
- 可以在发送关节目标前处理 joint wrap，比如 joint6 避免整圈旋转。
- 不做完整路径规划，避障能力弱。

## 2. `/plan_kinematic_path`

代码示例：

```python
self._planner = self.node.create_client(GetMotionPlan, "/plan_kinematic_path")
```

用途：

- 只让 MoveIt 规划轨迹。
- 输入 `MotionPlanRequest`。
- 输出 `RobotTrajectory`。
- 不直接执行。

典型流程：

```text
起始关节状态 + 目标约束
  -> /plan_kinematic_path
  -> 得到 RobotTrajectory
  -> 可检查或修正轨迹
  -> /execute_trajectory 执行
```

项目对应：

- `src/rebotarm_moveit_demos/rebotarm_moveit_demos/pick_place.py`
- `src/rebotarm_gazebo/src/rebotarm_gazebo/moveit_pick_place.py`

特点：

- 可以拿到规划后的轨迹。
- 可以在执行前检查、打印、修正轨迹。
- 适合处理 joint6 wrap / 轨迹连续性问题。
- 适合做碰撞检测和规划场景。

## 3. `/execute_trajectory`

代码示例：

```python
self._execute = ActionClient(
    self.node, ExecuteTrajectory, "/execute_trajectory"
)
```

用途：

- 只执行已有轨迹。
- 输入已经规划好的 `RobotTrajectory`。
- 不负责生成轨迹。

典型流程：

```text
RobotTrajectory
  -> /execute_trajectory
  -> MoveIt trajectory execution manager
  -> ros2_control / 真机控制器
```

项目对应：

- `src/rebotarm_moveit_demos/rebotarm_moveit_demos/demo_common.py`
- `src/rebotarm_gazebo/src/rebotarm_gazebo/moveit_pick_place.py`

特点：

- 和 `/plan_kinematic_path` 配合使用。
- 比 `/move_action` 更可控，因为中间能拿到轨迹。

## 4. `/move_action`

代码示例：

```python
self.move_group = ActionClient(self, MoveGroup, "/move_action")
```

用途：

- MoveIt 的一体化 action。
- 规划和执行都由 MoveGroup 完成。

典型流程：

```text
目标约束
  -> /move_action
  -> MoveIt 规划
  -> MoveIt 执行
```

特点：

- 代码最短。
- 不方便在规划后、执行前修改轨迹。
- 对 joint6 这种等价角跳变不友好。

当前建议：

- 教学和简单动作：使用 `/compute_ik + FollowJointTrajectory`。
- 需要碰撞检测、规划场景、attach/detach：使用
  `/plan_kinematic_path + /execute_trajectory`。
- 不建议在复杂 Pick & Place 中继续使用 `/move_action`，因为它不方便处理轨迹连续性。

## 5. 三种方式对比

| 方式 | 是否规划路径 | 是否执行 | 是否能修改轨迹 | 项目用途 |
|---|---:|---:|---:|---|
| `/compute_ik` | 否 | 否 | 是，执行前可改关节目标 | 简单教学、快速动作 |
| `/plan_kinematic_path` | 是 | 否 | 是，执行前可改 RobotTrajectory | MoveIt Pick & Place |
| `/execute_trajectory` | 否 | 是 | 执行前已可改 | 执行规划轨迹 |
| `/move_action` | 是 | 是 | 否 | 简单 MoveIt 调用 |

## 6. joint6 大转原因

机械臂旋转关节存在 `2*pi` 等价角：

```text
q, q + 2*pi, q - 2*pi
```

它们在物理姿态上相同，但控制器按数值差运动。若规划或 IK 返回了另一个等价角，就可能出现腕部整圈旋转。

解决思路：

- `simple_pick_place.py`：在 IK 后、发送轨迹前，把目标角折算到离当前角最近的等价值。
- `moveit_pick_place.py`：先通过 `/plan_kinematic_path` 拿到完整轨迹，再对轨迹点做连续化处理，然后用 `/execute_trajectory` 执行。

