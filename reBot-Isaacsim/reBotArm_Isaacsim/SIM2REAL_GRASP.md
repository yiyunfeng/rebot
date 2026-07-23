# Sim2Real YOLO + SAM traditional grasp

## Complete Isaac Sim grasp

Run one command:

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/reBotArm_Isaacsim
./run_sim_grasp.sh
```

The simulation repeatedly performs: ready, detect, open, pregrasp, grasp,
force-by-position close, retreat, return ready, vertical place, release, retreat,
and return ready. The fingers hold the banana through PhysX contact friction.
No physical robot is connected.

## Isaac Sim RGB-D

Terminal 1 uses Isaac Sim's Python environment:

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/reBotArm_Isaacsim
./run_sim_rgbd.sh
```

It loads the existing DM scene and exports the latest wrist-camera frame to:

```text
/tmp/rebot_sim_rgbd.npz
```

## Sim perception

Terminal 2 uses the `rebotarm_gpu` Conda environment:

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/reBotArm_Isaacsim
./run_sim2real_perception.sh --source sim
```

## Real robot grasp

Real execution uses the B601-DM configuration selected by
`rebot_grasp/sdk/reBotArm_control_py/config/rebotarm.yaml`. Stop other programs
using the RGB-D camera or DM serial channel, clear the workspace, verify joint
and gripper limits, and keep the emergency stop ready. Then run:

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/reBotArm_Isaacsim
./run_real_grasp.sh
```

Type `RUN REAL` only after completing the printed safety checks. The launcher
starts two independent processes in `rebotarm_gpu`:

- `sim2real_perception.py --source real` owns the RGB-D camera and publishes a
  stable YOLO + SAM grasp candidate.
- `real_grasp_executor.py` owns the DM serial channel, combines the candidate
  with eye-in-hand calibration and live TCP FK, then controls the robot.

The executor waits at `ready_pose`. For every candidate, press Enter to run one
supervised cycle: open, pregrasp, grasp, force-controlled close, retreat, return
ready, vertical place, release, retreat, and return ready. Type `q` instead of
Enter to stop. Real motion is deliberately not unattended because this path has
IK and height checks but no collision planner.

Both Sim and Real perception write the same camera-frame candidate format to:

```text
/tmp/rebot_grasp_candidate.json
```

To inspect Real perception without connecting the robot, continue to use:

```bash
./run_sim2real_perception.sh --source real
```

Press `Q` or `Esc` in the preview window to stop perception-only mode.
