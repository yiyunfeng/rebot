# reBotArm DM Isaac Sim

This project contains the DM reBotArm B601 model only. Mass, inertia, joint axes, limits, drive gains, and the two-slide gripper are imported from the validated MuJoCo model under `assets/DM-rebot-dev-arm/source/`.

All runtime parameters live in `config/dm_sim.yaml`.

## Build

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim
./reBotArm_Isaacsim/build_dm_asset.sh
```

## Run

Start Isaac Sim:

```bash
cd /home/yyf/Desktop/pythonProject/rebot/reBot-Isaacsim/reBotArm_Isaacsim
./run_isaacsim_receiver.sh
```

Run the hardware-free DM trajectory in a second terminal:

```bash
./run_test_sender.sh
```

For the physical-arm gravity-compensation mirror, use `./run_sender.sh` only after checking the motor model, communication channel, joint and gripper limits, clear workspace, and emergency stop.

The receiver uses a 400 Hz PhysX loop and sends position targets to the drives. It does not teleport joints during normal operation. The left finger is driven and the right finger follows through a PhysX mimic constraint, matching the MuJoCo equality constraint.

See [README.md](README.md) for the physical parameters, GUI workflow, UDP protocol, and troubleshooting.
