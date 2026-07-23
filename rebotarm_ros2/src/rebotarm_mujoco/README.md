# rebotarm_mujoco

MuJoCo 版 reBot Arm B601 DM 仿真包。该包位于 `rebotarm_ros2/src/rebotarm_mujoco/`，所有 MuJoCo 模型、MoveIt 配置和启动文件都放在本目录内。

## 兼容环境

- ROS 2: Humble
- Python: 3.10
- MoveIt 2: 使用当前 `rebotarm_ros2` 环境
- MuJoCo: 推荐 `mujoco==3.2.7`

当前 ROS 2 使用本地 `/usr/bin/python3`。MuJoCo 需要安装到运行 `ros2 launch`
时使用的 Python 环境中，通常是本地 Python：

```bash
python3 -m pip install --user "mujoco==3.2.7"
python3 -c "import mujoco; print(mujoco.__version__)"
```

## 构建

现在该包已经放入 `rebotarm_ros2/src/`，按普通 ROS 2 workspace 构建即可：

```bash
cd /home/yyf/Desktop/pythonProject/rebot/rebotarm_ros2
source /opt/ros/humble/setup.bash
colcon build --packages-select rebotarm_mujoco --symlink-install
source install/setup.bash
```

## 启动

```bash
ros2 launch rebotarm_mujoco mujoco.launch.py
```

启动后：

- MuJoCo 节点发布 `/joint_states`
- MuJoCo 节点提供 `/rebotarm_controller/follow_joint_trajectory`
- MuJoCo 节点提供 `/gripper_controller/follow_joint_trajectory`
- MoveIt `move_group` 使用本目录下的 `config/` 配置
- 默认打开 MuJoCo GUI，MoveIt 执行轨迹时 GUI 会同步显示同一个仿真实例

如果当前终端没有图形界面，先在 `config/mujoco_params.yaml` 中临时改为：

```yaml
use_viewer: false
```

## 夹取方式

MuJoCo 模型使用摩擦和接触力夹取物体，不使用吸附或固定关节。相关参数在 `models/rebotarm_dm.xml` 中：

- `finger_pad` 的 `friction`
- `green_cube` 的 `friction`
- 接触稳定性参数 `solref` / `solimp`

如果夹不稳，优先调这些参数和夹爪闭合轨迹，不要加吸附插件。
