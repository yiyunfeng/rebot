# Repository Guidelines

## Project Structure & Module Organization

This is a ROS 2 workspace for reBot Arm B601 DM/RS. Source packages live in `src/`:
`rebotarm_msgs` defines custom `msg/`, `srv/`, and `action/` interfaces;
`rebotarmcontroller` contains the Python hardware driver and examples;
`rebotarm_bringup` contains launch, hardware config, URDF, meshes, and RViz files;
`rebotarm_moveit_config` contains MoveIt 2, SRDF, ros2_control, and planner config;
`rebotarm_moveit_demos` contains demo launch files and Python demos. Gazebo simulation
assets and launch modes are in `rebotarm_gazebo` and `rebotarm_gazebo11`. Keep generated
workspace outputs (`build/`, `install/`, `log/`) out of commits.

## Build, Test, and Development Commands

Run commands from the workspace root after sourcing your ROS 2 distribution.

```bash
colcon build --symlink-install
source install/setup.bash
colcon test
colcon test-result --verbose
```

Use package-scoped builds while iterating, for example
`colcon build --packages-select rebotarmcontroller --symlink-install`. Start real
hardware with `ros2 launch rebotarm_bringup bringup.launch.py`; start Gazebo/MoveIt
simulation with `ros2 launch rebotarm_gazebo rebotarm.launch.py mode:=sim`.

## Coding Style & Naming Conventions

Python packages use `ament_python`, four-space indentation, `snake_case` for modules,
functions, topics, and parameters, and `PascalCase` only for classes. Preserve existing
ROS names such as `/rebotarm/joint_states`, planning groups, joint names, controller
names, and action/service names unless all dependent config is updated together. Keep
launch files parameterized for model, channel, `use_rviz`, and simulation mode when
adding new entry points.

## Testing Guidelines

There are few committed tests today; add focused `pytest` tests under a package-level
`test/` directory for pure Python logic. For interface, launch, or config changes, run
`colcon test` and smoke-check the affected path in RViz, MoveIt, or Gazebo. Validate
URDF/Xacro and controller changes together because `*.urdf.xacro`, `*.srdf`,
`ros2_controllers.yaml`, and `joint_limits.yaml` form one contract.

## Commit & Pull Request Guidelines

Recent history uses short Conventional Commit-style prefixes such as `feat:`, `fix:`,
and `doc:`. Keep commits focused and explain user-visible behavior or safety impact.
Pull requests should include a summary, changed packages, test or launch commands run,
linked issues when applicable, and screenshots/log snippets for RViz, MoveIt, or Gazebo
behavior changes.

## Safety & Configuration Tips

Real hardware commands can move the arm. Before testing hardware launch files, confirm
the model (`dm` or `rs`), channel (`/dev/ttyACM0` or `can0`), clear workspace, joint
limits, emergency stop path, and low-speed settings. Do not bypass safety limits,
collision checking, watchdogs, or vendor safety states to make a demo work.
