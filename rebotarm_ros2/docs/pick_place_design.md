# reBotArm Gazebo 夹取放置 — 两套方案设计文档

> 写给 Codex 重新优化用。包含完整的环境拓扑、接口定义、当前代码问题、优化方向。

## 1. 目标

在 Gazebo 仿真中完成完整的 Pick & Place：

- 桌面上生成 **6cm 绿色正方体**
- 机械臂从 Home 位置出发，**竖直向下**夹取正方体
- 抬升后移动到桌面另一处，**水平方向**放置
- 机械臂回到初始位置
- 仿真环境不能改动原有 `gazebo.launch.py` 和 `rebotarm.launch.py`

---

## 2. 文件清单

所有文件均在 `src/rebotarm_gazebo/` 下：

```
src/rebotarm_gazebo/
├── cube_spawner.py              # 正方体工具类（两方案共享）
├── simple_pick_place.py         # 方案1 任务节点
├── moveit_pick_place.py         # 方案2 任务节点
launch/
├── simple_pick_place.launch.py  # 方案1 启动文件
├── moveit_pick_place.launch.py  # 方案2 启动文件
setup.py                         # 已添加 simple_pick_place / moveit_pick_place 入口
```

运行方式：

```bash
colcon build --packages-select rebotarm_gazebo

# 方案1：简化版（无 MoveIt）
ros2 launch rebotarm_gazebo simple_pick_place.launch.py

# 方案2：MoveIt 版（碰撞检测 + 运动规划）
ros2 launch rebotarm_gazebo moveit_pick_place.launch.py
```

---

## 3. 仿真环境拓扑

### 3.1 TF 树

```
world
  │  static_transform_publisher (0.05, 0, 0.265)
  ▼
base_link
  │  joint1 (revolute, Z轴)
  ▼
link1 → link2 → link3 → link4 → link5 → link6
                                            │  gripper_joint (fixed)
                                            ▼
                                       gripper_link
                                       ├── gripper_joint1 (prismatic, X轴) → gripper_left
                                       ├── gripper_joint2 (prismatic, X轴) → gripper_right
                                       └── gripper_tcp (末端)
```

### 3.2 控制器（`config/gazebo_controllers.yaml`）

| 控制器 | 关节 | Action 接口 | Topic 接口 |
|--------|------|------------|-----------|
| `joint_state_broadcaster` | 全部 | - | `/joint_states` |
| `rebotarm_controller` | joint1~joint6 | `/rebotarm_controller/follow_joint_trajectory` | `/rebotarm_controller/joint_trajectory` |
| `gripper_controller` | gripper_joint1 | `/gripper_controller/follow_joint_trajectory` | `/gripper_controller/joint_trajectory` |
| `gripper_mirror_controller` | gripper_joint2 | - | `/gripper_mirror_controller/joint_trajectory` |

> `gripper_mirror.py` 节点订阅 `gripper_controller/state`，将 gripper_joint1 的 `desired.positions` 同步到 gripper_joint2。

### 3.3 MoveIt 控制器映射（`config/moveit_controllers.yaml`）

```yaml
rebotarm_controller:
  action_ns: follow_joint_trajectory
  type: FollowJointTrajectory
  joints: [joint1, joint2, joint3, joint4, joint5, joint6]

gripper_controller:
  action_ns: follow_joint_trajectory
  type: FollowJointTrajectory
  joints: [gripper_joint1]
```

### 3.4 初始位置（`config/initial_positions.yaml`）

```yaml
joint1: 0.0, joint2: -0.05, joint3: -0.05
joint4: 0.0, joint5: 0.0,  joint6: 0.0
gripper_joint1: 0.0, gripper_joint2: 0.0
```

### 3.5 桌面 & 正方体位置

```
桌面 (arm_on_the_table.sdf):
  Gazebo 模型: pose = (0.28, 0, 0)  yaw = 1.5708

桌面碰撞物体 (planning_scene_objects.py):
  TABLE_X = 0.28, TABLE_Y = 0.0, TABLE_Z = 0.0, TABLE_YAW = 1.5708
  TABLE_TOP_LENGTH = 0.4, TABLE_TOP_WIDTH = 0.6, TABLE_TOP_THICKNESS = 0.03
  TABLE_TOP_Z = 0.245

桌面物理位置计算:
  桌腿底部 z = TABLE_Z = 0
  桌面板中心 z = TABLE_Z + TABLE_TOP_Z = 0 + 0.245 = 0.245
  桌面板顶面 z = 0.245 + 0.03/2 = 0.26

正方体 (6cm):
  桌面上的正方体底面 z = 0.26
  正方体中心 z = 0.26 + 0.06/2 = 0.29

  夹取位置 CUBE_PICK  = (0.35,  0.15, 0.29)  ← 桌面右前侧
  放置位置 CUBE_PLACE = (0.20, -0.15, 0.29)  ← 桌面左后侧
```

### 3.6 夹爪参数

```
max_gripper_width = 0.09m          (来自 trajectory_relay.py 默认值)
每个 joint 行程 = 0 ~ 0.045m      (half of max_width)
TCP frame name = "gripper_tcp"

GRIPPER_OPEN  = 0.04               (夹爪张开，单位 m)
GRIPPER_CLOSE = 0.012              (夹 6cm 方块 ≈ (0.09-0.06)/2 = 0.015，留余量)
```

---

## 4. 方案1 — 简化版

### 4.1 设计思路

**完全不使用 MoveIt。** 关节路径点硬编码在文件顶部，直接通过 ros2_control 的 Action/Topic 接口控制机械臂。

- 无碰撞检测
- 无运动规划（IK 也没有）
- 代码约 250 行，结构为线性 10 步顺序执行

### 4.2 架构

```
SimplePickPlace(Node)
  ├── ActionClient → /rebotarm_controller/follow_joint_trajectory    # arm 运动
  ├── Publisher    → /gripper_controller/joint_trajectory            # 夹爪开合
  ├── tf2_ros.Buffer → lookup_transform("world", "gripper_tcp")     # 吸附用
  └── CubeSpawner    → subprocess + SetEntityPose                   # 正方体管理
```

### 4.3 10 步流程

```
Step  1: cube.spawn(CUBE_PICK)                    → 正方体出现在桌面
Step  2: gripper_pub(OPEN=0.04) + sleep(0.6)      → 张开夹爪
Step  3: arm.go(PRE_PICK)                          → 关节运动到正方体上方
Step  4: arm.go(PICK)                              → 关节运动到夹取位置
Step  5: gripper_pub(CLOSE=0.012) + sleep(1.2)     → 闭合夹爪
         snap_cube()                               → 正方体瞬移到夹爪 TCP
Step  6: arm.go(PRE_PICK) + snap_cube()            → 抬升（正方体跟随）
Step  7: arm.go(PRE_PLACE) + snap_cube()           → 移到放置位置上方
Step  8: arm.go(PLACE) + snap_cube()               → 下降到放置位置
Step  9: gripper_pub(OPEN) + sleep(0.6)            → 释放
         cube.move_to(CUBE_PLACE)                  → 正方体留在桌面
Step 10: arm.go(PRE_PLACE) + arm.go(HOME)          → 回初始位置
```

### 4.4 关节路径点（硬编码，需调参）

```python
# [joint1, joint2, joint3, joint4, joint5, joint6]  单位：弧度
HOME      = [ 0.0,  -0.05, -0.05,  0.0,   0.0,   0.0]   # 初始位置
PRE_PICK  = [ 0.46, -1.0,  -1.6,   0.6,  -1.57,  0.0]   # 正方体上方, 夹爪竖直向下
PICK      = [ 0.46, -1.3,  -1.35,  0.3,  -1.57,  0.0]   # 夹取位置, 夹爪竖直向下
PRE_PLACE = [ 0.0,  -1.0,  -1.6,   0.6,   0.0,   0.0]   # 放置上方, 夹爪水平
PLACE     = [ 0.0,  -1.3,  -1.35,  0.3,   0.0,   0.0]   # 放置位置, 夹爪水平

# joint5 = -1.57 → 夹爪竖直向下
# joint5 =  0.0  → 夹爪水平朝前
```

### 4.5 运动执行：单点 JointTrajectory

```python
def _go(positions):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = JointTrajectory(
        joint_names = [joint1..joint6],
        points = [JointTrajectoryPoint(positions=positions, time_from_start=MOVE_TIME=2.0s)]
    )
    arm_client.send_goal_async(goal)
    # 阻塞等待完成
```

每次只发一个目标点，控制器内部做插值。

### 4.6 吸附：tf2 + SetEntityPose 瞬移

```python
def _snap_cube():
    tf = tf_buffer.lookup_transform("world", "gripper_tcp", now)
    cube.move_to(tf.x, tf.y, tf.z - 0.04)   # 偏移到夹爪内侧
```

`move_to` 先尝试 `SetEntityPose` 服务，失败则降级为 delete + create 重建。

### 4.7 CubeSpawner 实现

```python
class CubeSpawner:
    def spawn(x, y, z):    # subprocess: ros2 run ros_gz_sim create -string <sdf>
    def move_to(x, y, z):  # SetEntityPose 服务 or 降级 spawn
    def remove():          # subprocess: ros2 run ros_gz_sim delete
```

正方体 SDF 模板硬编码为边长 0.06m 的绿色方盒，含 inertial/visual/collision。

### 4.8 方案1 已知问题

| # | 问题 | 严重度 | 说明 |
|---|------|--------|------|
| 1 | **关节路径点是估算值** | **CRITICAL** | 5 个数组均为人工估算，未经实际验证。需要运行 MoveIt 版后从 `/joint_states` 抄录正确值 |
| 2 | **每段只发一个目标点** | HIGH | 无多 waypoint 插值，控制器内部路径不可控，可能扫过桌面 |
| 3 | **time.sleep 等待夹爪** | MEDIUM | 夹爪未到位就继续执行的风险。应订阅 `/gripper_controller/state` 确认 |
| 4 | **吸附仅在关键点瞬移** | MEDIUM | 正方体在运动过程中不跟随，只在到达后"跳"到新位置 |
| 5 | **CubeSpawner 用 subprocess** | MEDIUM | 应改用 `ros_gz_sim` 的 service API |
| 6 | **MOVE_TIME 统一 2.0s** | LOW | 不同距离段应使用不同时长 |
| 7 | **gripper_joint2 未显式控制** | LOW | 依赖 `gripper_mirror.py` 同步，耦合度高 |

---

## 5. 方案2 — MoveIt 版

### 5.1 设计思路

使用 `moveit.planning.MoveItPy`（MoveIt 2 的官方 Python API）进行运动规划和碰撞检测。

- 正方体先在 Gazebo 中生成，同时作为 MoveIt `CollisionObject` 加入规划场景
- 夹取后变为 `AttachedCollisionObject`（MoveIt 规划时自动考虑附着物体）
- 运动规划使用 OMPL

### 5.2 架构

```
MoveItPickPlace(Node)
  ├── MoveItPy
  │     ├── get_planning_component("arm") → 规划 arm 运动
  │     └── get_planning_scene_monitor()  → 管理碰撞/附着物体
  ├── Publisher    → /gripper_controller/joint_trajectory   # 夹爪
  ├── tf2_ros.Buffer → lookup_transform("world", "gripper_tcp")  # 吸附
  └── CubeSpawner    → Gazebo 正方体管理（同方案1）
```

### 5.3 10 步流程

```
Step 1: cube.spawn(pick) + 场景.add(正方体CollisionObject)
Step 2: gripper(OPEN)
Step 3: arm.set_pose_target(above_cube, gripper_down)  + plan + execute
Step 4: arm.set_pose_target(at_cube, gripper_down)     + plan(cartesian) + execute
Step 5: gripper(CLOSE) + sleep(1.2) + 场景.attach(正方体→gripper_tcp) + snap()
Step 6: arm.set_pose_target(lift, gripper_down) + plan + execute + snap()
Step 7: arm.set_pose_target(above_place, horizontal) + plan + execute + snap()
Step 8: arm.set_pose_target(at_place, horizontal) + plan(cartesian) + execute + snap()
Step 9: 场景.detach(正方体) + gripper(OPEN) + sleep(0.6) + cube.move_to(place)
Step 10: arm.set_named_target("home") + plan + execute
```

### 5.4 目标位姿

```python
# 夹取阶段：夹爪竖直向下（绕 Y 轴转 90° → ry=0.707, rw=0.707）
pick_above = (0.35, 0.15, 0.41)   # 正方体上方 12cm
pick_at    = (0.35, 0.15, 0.32)   # 夹取位置（上方 3cm）

# 放置阶段：夹爪水平朝前（默认四元数）
place_above = (0.2, -0.15, 0.41)
place_at    = (0.2, -0.15, 0.32)
```

### 5.5 运动规划：MoveItPy API

```python
def _go(target_pose):
    arm = moveit.get_planning_component("arm")
    arm.set_pose_target(target_pose, "gripper_tcp")
    plan = arm.plan()                            # OMPL 自动规划
    arm.execute(plan.trajectory, blocking=True)

def _go_linear(target_pose):
    arm = moveit.get_planning_component("arm")
    arm.set_pose_target(target_pose, "gripper_tcp")
    plan = arm.plan(planner_id="RRTConnectkConfigDefault")  # 伪笛卡尔
    arm.execute(plan.trajectory, blocking=True)
```

### 5.6 碰撞场景管理

```python
def _update_scene(objects, attached):
    scene = PlanningScene(is_diff=True)           # 增量更新
    scene.world.collision_objects = objects       # CollisionObject 列表
    scene.robot_state.attached_collision_objects = attached
    moveit.get_planning_scene_monitor().update_planning_scene(scene)

# 添加正方体到场景
_box_collision("green_cube", 0.06, x, y, z)       # CollisionObject.ADD

# 夹取后：从场景移除 → 附着到夹爪
_attach("green_cube"):
    scene.remove("green_cube")                     # CollisionObject.REMOVE
    scene.attach(AttachedCollisionObject(
        object = box(0.06, offset_z=-0.03),        # 相对 TCP 偏移
        link_name = "gripper_tcp",
        touch_links = ["gripper_tcp"],             # 不检测自碰撞
    ))

# 释放：从夹爪分离
_detach("green_cube"):
    scene.detach(AttachedCollisionObject.REMOVE)
```

### 5.7 方案2 已知问题

| # | 问题 | 严重度 | 说明 |
|---|------|--------|------|
| 1 | **MoveItPy 在 run() 中初始化** | HIGH | 不是 `__init__` 中，spin 时序可能有问题 |
| 2 | **`_go_linear` 不是真正的笛卡尔路径** | HIGH | `arm.plan()` 不支持笛卡尔约束。应该用 `compute_cartesian_path()` 并构造 waypoints 列表 |
| 3 | **PlanningComponent 重复获取** | MEDIUM | 每次 `_go()` 都调 `get_planning_component("arm")` |
| 4 | **gripper publisher 每次 create** | MEDIUM | `_move_gripper()` 中每调一次就新建 publisher |
| 5 | **拾取姿态四元数需验证** | MEDIUM | `ry=0.707, rw=0.707` 是否对应"夹爪竖直向下"取决于 URDF 中 TCP frame 的默认朝向 |
| 6 | **launch 缺少 planning_scene_objects** | MEDIUM | 桌面碰撞物体未添加，MoveIt 不知道桌面的存在，规划可能穿透桌子 |
| 7 | **没有用 MoveIt Grasp/PickPlace pipeline** | LOW | 手写了完整流程，未用内置 `pick()`/`place()` |
| 8 | **吸附同步问题** | MEDIUM | 同方案1 |

---

## 6. Launch 文件

| 组件 | simple_pick_place.launch.py | moveit_pick_place.launch.py |
|------|---------------------------|---------------------------|
| Gazebo (gz_sim.launch.py) | ✅ | ✅ |
| 机械臂 spawn (ros_gz_sim create) | ✅ | ✅ |
| clock_bridge | ✅ | ✅ |
| robot_state_publisher | ✅ | ✅ |
| static_tf (world→base_link) | ✅ | ✅ |
| joint_state_broadcaster | ✅ | ✅ |
| rebotarm_controller | ✅ | ✅ |
| gripper_controller | ✅ | ✅ |
| gripper_mirror_controller | ✅ | ✅ |
| move_group | ❌ | ✅ |
| planning_scene_objects (桌面碰撞) | ❌ | ❌ **缺失** |
| 任务节点 | simple_pick_place | moveit_pick_place |

**启动顺序**（通过 `RegisterEventHandler(OnProcessExit)` 串行化）：

```
方案1: jsp → [arm, grip, grip_m] → task
方案2: jsp → [arm, grip, grip_m] → move_group → task
```

---

## 7. 关键约束

1. **只改 `src/rebotarm_gazebo/` 目录下的文件**。不能修改 `rebotarm_bringup`、`rebotarm_moveit_config`、`rebotarm_moveit_demos` 等已有包。

2. **仿真环境不变**。Gazebo 世界文件、控制器配置、URDF/SRDF 保持原样。

3. **两个 launch 文件独立运行**，不与已有的 `gazebo.launch.py` 冲突。

4. **方案1 的关节值需要验证**。优先实现从 `/joint_states` 自动记录关节值的方法，或者用 IK（PyKDL 或 `/compute_ik` service）动态计算。

5. **夹爪只有 gripper_joint1 被主控**。gripper_joint2 通过 `gripper_mirror.py` 节点跟随。

---

## 8. 优化优先级

### P0 — 必须修
- [ ] **方案1: 获取正确的关节路径点**（用 IK 或录制/回放）
- [ ] **方案2: 实现真正的笛卡尔路径**（`compute_cartesian_path`）
- [ ] **方案2 launch: 补充 `planning_scene_objects` 节点**（添加桌面碰撞）

### P1 — 应该修
- [ ] **方案1: 多 waypoint 轨迹**（每段运动 3-5 个插值点）
- [ ] **方案1: 夹爪状态反馈**（订阅 state 替代 time.sleep）
- [ ] **方案2: MoveItPy 在 `__init__` 初始化**（正确处理生命周期）
- [ ] **共享: CubeSpawner 改用 service API**（替代 subprocess）
- [ ] **共享: 吸附实时跟随**（定时器在运动过程中持续更新正方体位置）

### P2 — 可以修
- [ ] **方案2: 复用 PlanningComponent 和 Publisher 实例**
- [ ] **方案2: 验证拾取/放置姿态四元数**
- [ ] 统一用 ROS parameters 配置所有位置参数
- [ ] 添加正方体位置合法性检查（是否在桌面上、是否在工作空间内）
- [ ] 添加错误恢复（运动失败后安全回 Home）
