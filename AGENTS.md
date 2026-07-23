# Repository Guidelines

## Project Structure & Module Organization

This repository groups several reBot Arm B601 projects. `rebotarm_ros2/` is the ROS 2 workspace for hardware drivers, MoveIt 2, Gazebo, RViz, custom interfaces, and demos. Its packages live under `rebotarm_ros2/src/`. `rebot_grasp/` contains the RGB-D grasping demo, with calibration code in `calibration/`, runtime scripts in `scripts/`, camera/robot drivers in `drivers/`, and YAML config in `config/`. `reBot-Isaacsim/` contains the Isaac Sim digital-twin workflow, including sender/receiver scripts in `reBotArm_Isaacsim/` and USD assets in `usd/`. Shared low-level control code is kept in `third_party/reBotArm_control_py/`.

## Build, Test, and Development Commands

Use the environment for the subproject you are changing.

```bash
cd rebotarm_ros2 && colcon build --symlink-install
cd rebotarm_ros2 && colcon test && colcon test-result --verbose
cd rebot_grasp && conda env create -f environment.yml
cd reBot-Isaacsim/third_party/reBotArm_control_py && uv sync
```

After ROS 2 builds, run `source install/setup.bash` before `ros2 launch` or `ros2 run`. For Isaac Sim, start the receiver before the sender as described in `reBot-Isaacsim/README.md`.

## Coding Style & Naming Conventions

Python code uses four-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Keep ROS package names, topics, services, actions, planning groups, joint names, and controller names stable unless all dependent launch/config files are updated together. Prefer existing helper modules and YAML configuration over hardcoded paths or hardware parameters.

## Testing Guidelines

Add focused `pytest` tests for pure Python logic when practical. For ROS 2 interface, launch, URDF/SRDF, controller, or MoveIt changes, run `colcon test` and smoke-test the affected launch path in RViz, Gazebo, or hardware simulation. For grasping changes, validate camera import, calibration config, and the target script with the relevant RGB-D device or a documented mock path.

## Bug Analysis & Log Attribution

When analyzing bugs, always point to the exact log lines that support the diagnosis. Quote the key messages, explain what each line means, and state why the conclusion follows. If rejecting another possible cause, cite the log evidence or missing evidence. For ROS 2 / MoveIt / Gazebo issues, identify the failing node, topic/action/service, link/joint name, controller, or planning group whenever the log provides it.

## Commit & Pull Request Guidelines

Use concise Conventional Commit messages such as `feat: add grasp calibration check` or `fix: clamp unsafe z target`. Pull requests should include a summary, affected subproject, commands run, hardware/simulation mode tested, linked issues, and screenshots or logs for RViz, Gazebo, Isaac Sim, or vision behavior changes.

## Safety & Configuration Tips

Real hardware can move unexpectedly. Before hardware runs, confirm the DM model, channel (`/dev/ttyACM*`), joint limits, gripper limits, clear workspace, and emergency stop access. Never bypass collision checks, watchdogs, vendor safety states, or configured limits to make a demo pass.
