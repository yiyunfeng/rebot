# Agent Lessons

Codex 为主、Claude Code 为辅的**共享知识库**。每次会话开始时双方都应读取此文件。

**角色定位**：
- **Codex**：主要解决问题、设计方案、编写代码
- **Claude Code**：辅助排查、解释说明、执行简单操作、记录同步

## 记录格式

每条问题记录必须包含：
- **现象**（报了什么错/什么异常行为）
- **排查思路**（从哪入手、怎么定位的、排除了哪些可能性）
- **根因**（真正的原因是什么）
- **解决**（具体改了什么）

不只是记结论，**要记推理过程** — 让对方能学会你的排查方法论。

---

## 行为规则（双方遵守）

### 使用中文回复
- 默认用中文沟通；代码、命令、库名、API 名称等技术术语可保留英文。

### 代码需要适量注释
- 为关键流程、非直观逻辑、边界条件和安全相关代码添加简洁注释。

### 不要擅自安装系统软件
- 先说明当前缺少什么，再询问用户是否愿意手动安装或授权安装。

### 不要擅自删除或覆盖文件
- 不运行 `rm`、`del`、`git rm`；确需删除时先征得用户确认。

### 真实机械臂相关改动优先安全
- 修改硬件配置前先确认 `model`、`channel`、关节限制、控制器名称和急停路径。

### 问题解决后同步记录
- 每解决一个明确问题，都要把现象、排查思路、根因、解决方法和验证结果同步到本文件。
- 记录要面向另一个 agent 接手：Claude/Codex 读完后应能知道问题为什么发生、改了哪些文件、如何确认修复有效。
- 不要只写“已解决”，要写清楚造成问题的配置、代码路径或运行方式。

---

## 问题与解决记录

> 格式：`### <问题简述>` + 现象 / 根因 / 解决。打上标签便于快速定位。

### CubeSpawner 导入失败（ModuleNotFoundError: No module named 'cube_spawner'）
- **现象**: `simple_pick_place` 和 `moveit_pick_place` 运行时报 `ModuleNotFoundError`
- **根因**: `from cube_spawner import CubeSpawner` 不带包前缀，colcon install 后包名为 `rebotarm_gazebo`，找不到裸模块名
- **解决**: 改为 `from rebotarm_gazebo.cube_spawner import CubeSpawner`
- **相关文件**: `simple_pick_place.py`, `moveit_pick_place.py`

### rebotarm_gazebo11 CMake 找不到 src/rebotarm_gazebo11/ 目录
- **现象**: `colcon build` 报 `ament_cmake_symlink_install_directory() can't find`
- **根因**: `CMakeLists.txt` 写 `PACKAGE_DIR src/${PROJECT_NAME}`，但 `src/` 下没有 `rebotarm_gazebo11/` 子目录
- **解决**: 改为 `PACKAGE_DIR src`，与 `setup.py` 的 `package_dir={package_name: "src"}` 一致
- **相关文件**: `rebotarm_gazebo11/CMakeLists.txt`

### driver.launch.py 启动时 URDF 找不到
- **现象**: `reBot_B601_DM_with_gripper.urdf does not exist`，`ValueError: does not contain a valid URDF model`
- **根因**: `rebotarm_hardware.yaml` 的 DM overrides 将 `urdf_path` 改为 `reBot_B601_DM_with_gripper.urdf`，但 SDK（`third_party/`）目录下只有 `reBot-DevArm_fixend.urdf`。SDK 的 `robot_model.py` 以 SDK 根目录为基准解析相对路径
- **解决**: 切回注释掉的老配置（`reBot-DevArm_fixend.urdf` + `end_link`），或用正确路径替换。如需用带夹爪的 URDF，需将 `reBot_B601_DM_with_gripper.urdf` 和 gripper mesh 文件复制到 SDK 目录并将 mesh 路径从 `package://` 改为相对路径
- **相关文件**: `rebotarm_bringup/config/rebotarm_hardware.yaml`, `third_party/reBotArm_control_py/urdf/...`

### hardware.launch.py 报 "is not a valid package name"
- **现象**: `'/install/rebotarm_gazebo/share/rebotarm_gazebo' is not a valid package name`
- **根因**: `FindPackageShare(gazebo_share)` — `gazebo_share` 是 `get_package_share_directory()` 返回的路径字符串，不是包名；同时 `LaunchConfiguration("gazebo_moveit.rviz")` 引用了不存在的参数名
- **解决**: `FindPackageShare("rebotarm_gazebo")` + `LaunchConfiguration("rviz_config")`
- **相关文件**: `hardware.launch.py`

### moveit_pick_place hardware 模式没启动真机驱动
- **现象**: `mode:=hardware` 只启动了 move_group + RViz，`reBotArmController` 没运行
- **根因**: `moveit_pick_place.launch.py` 只 include 了 `hardware.launch.py`，漏了 `driver.launch.py`
- **解决**: 增加 `hw_env` include `driver.launch.py`，再加 `hw_moveit` include `hardware.launch.py`
- **相关文件**: `moveit_pick_place.launch.py`

### simple_pick_place IK 失败 code=-31 (NO_IK_SOLUTION)
- **现象**: 仿真下 `pick_above` IK 返回 -31
- **根因**: 排查确认 `gripper_tcp` frame 存在（不是 frame 缺失问题）。`-0.265` 偏移是 world→base_link 的 Z 坐标转换。最终目标位姿超出工作空间或 KDL IK 求解器无法收敛
- **相关文件**: `simple_pick_place.py`

### Git 仓库膨胀到 65GB（.vscode/browse.vc.db）
- **现象**: `.git` 目录 65GB，推送 GitHub 超时
- **根因**: `.vscode/browse.vc.db`（VS Code C++ IntelliSense 符号数据库，1.08GB）被提交了两次
- **解决**: `git filter-branch` 删除历史中的 db 文件，`.gitignore` 加入 `.vscode/`。GitHub 推送超时是因为 pack 太大（715MB），逐个 commit 推送解决
- **相关文件**: `.gitignore`

### GitHub 推送超时
- **现象**: `git push` 报 `RPC failed; curl 56 GnuTLS recv error`
- **根因**: pack 文件太大（715MB），HTTPS 单次传输被断开
- **解决**: 分批逐个 commit 推送（`git push project <commit>:refs/heads/main`）

### RealController 架构设计
- **背景**: 需要给真实机械臂操作提供一个干净的 API，供后续开发复用
- **设计**: 非 Node 类，组合模式注入 Node 引用。方法分层：便捷 API（`enable/open_gripper/close_gripper/home/move_joints/move_to_pose`）、IK 求解（`solve_ik/solve_ik_hardware`）、夹爪控制（`move_gripper` → 模拟宽度自动映射硬件指令值）
- **用法**: `arm = RealController(node)` → `arm.enable()` → `arm.move_joints([...])` → `arm.move_to_pose(pose)`
- **相关文件**: `src/rebotarm_gazebo/src/real_controller.py`

### 全局配置的含义
- **规则**: 用户说「全局配置」→ 操作 `~/.claude/CLAUDE.md`，不是项目 memory
- **规则**: 用户说「git 推送」→ 只包含 `src/` `docs/` `media/` `*.md`，排除 `.vscode/` `.codex/` `.claude/` `build/` `install/` 等

### move_to_pose Z 补偿与最低高度保护
- **现象**: `/rebotarm/move_to_pose` 输入较小 `z` 时，真实夹爪高度与期望不一致；用户要求 `move_to_pose` 最低不低于 5mm。
- **排查思路**: 先确认 action 类型只有 `geometry_msgs/Pose`，没有 `header.frame_id`；再检查 `ros_actions.py`，发现目标直接转成 `x,y,z,roll,pitch,yaw` 后传给 SDK；最后确认原先只检查补偿后的 `command_z`，没有检查用户输入的 `user_z`。
- **根因**: `min_command_z` 只限制补偿后发送给 SDK 的高度，不能表达“用户输入 z 最低不低于 5mm”。
- **解决**: 在 `rebotarm_hardware.yaml` 增加 `pose_compensation.min_user_z: 0.005`；在 `HardwareManager.compensated_pose_z()` 中先检查 `user_z`，再计算 `command_z = user_z + z_offset`，最后检查 `min_command_z`。
- **验证**: `z=0.000` 会抛出 `move_to_pose user z=0.000 below safe limit 0.005`；`z=0.005` 会补偿为 `command_z=0.050`。
- **相关文件**: `src/rebotarm_bringup/config/rebotarm_hardware.yaml`, `src/rebotarmcontroller/rebotarmcontroller/hardware_manager.py`, `src/rebotarmcontroller/rebotarmcontroller/ros_actions.py`, `src/rebotarmcontroller/rebotarmcontroller/ros_services.py`

### hardware 模式 RobotState warning 与末端拖不动
- **现象**: `ros2 launch rebotarm_gazebo rebotarm.launch.py mode:=hardware` 后，RViz/MoveIt 出现 RobotState warning，末端目标姿态拖不动或交互不正常。
- **排查思路**: 跟踪 `rebotarm.launch.py mode:=hardware` 的 include 链路，确认它启动 `driver.launch.py` 和 `hardware.launch.py`；随后检查 `hardware.launch.py` 的时间源和 RViz 配置，发现 hardware 模式仍使用 `use_sim_time: True`，但该模式没有 Gazebo `/clock`。
- **根因**: 真机模式没有仿真时钟，`robot_state_publisher`、RViz、MoveIt 使用 `use_sim_time: True` 会造成 TF/RobotState 时间异常；同时默认 RViz 配置里 `MotionPlanning` 插件未启用，不利于直接拖动末端目标。
- **解决**: 在 `hardware.launch.py` 中统一使用 `hardware_time = {"use_sim_time": False}`；修正 `rviz_config` 路径拼接；在 `gazebo_moveit.rviz` 中默认启用 `MotionPlanning`。
- **验证**: `python3 -m py_compile src/rebotarm_gazebo/launch/hardware.launch.py` 通过。包级构建后续被本地 `install/` 残留文件阻塞，不是 launch 语法问题。
- **相关文件**: `src/rebotarm_gazebo/launch/hardware.launch.py`, `src/rebotarm_gazebo/rviz/gazebo_moveit.rviz`

### rebotarm_gazebo symlink install 被 __pycache__ 影响
- **现象**: `colcon build --packages-select rebotarm_gazebo --symlink-install` 报找不到或冲突的 `hardware.launch.cpython-310.pyc`。
- **排查思路**: 检查 `setup.py` 的 `data_files` 收集逻辑，发现 `_collect_files("launch")` 递归收集所有文件，包括 `launch/__pycache__/*.pyc`。
- **根因**: Python 编译缓存被当作 ROS package data 安装，`--symlink-install` 下容易产生缺失或已存在冲突。
- **解决**: 修改 `_collect_files()`，跳过 `__pycache__` 目录以及 `.pyc`/`.pyo` 文件。
- **验证**: `python3 -m py_compile src/rebotarm_gazebo/setup.py` 通过；后续构建仍可能需要清理旧 `install/rebotarm_gazebo` 残留后再跑。
- **相关文件**: `src/rebotarm_gazebo/setup.py`
