#!/usr/bin/env python3
"""加载 DM reBotArm 场景，并可选接收 UDP 关节目标。

默认模式以固定频率运行物理仿真、接收真机关节角并驱动 USD articulation；
``--no-udp`` 仅加载和模拟场景，供 RGB-D 导出和 GUI 调试复用。
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import time
from pathlib import Path

import numpy as np
import yaml
from isaacsim import SimulationApp

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "dm_sim.yaml"

# 右指由 PhysX mimic 跟随左指，因此控制目标只包含六轴和 left_finger。
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
DRIVEN_JOINT_NAMES = ARM_JOINT_NAMES + ("left_finger",)

_running = True


def stop(_signum, _frame) -> None:
    """收到退出信号后让仿真循环正常清理 socket 和 SimulationApp。"""
    global _running
    print(f"[DM] received signal {signal.Signals(_signum).name}; shutting down")
    _running = False


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)


class DMSimulation:
    """DM 场景、关节控制和可选 UDP 接收器。

    ``isaacsim_rgbd_exporter.py`` 复用这个类来加载同一套场景，同时关闭 UDP，
    避免 RGB-D 抓取流程和硬件镜像流程同时修改关节目标。
    """

    def __init__(self, listen_udp: bool = True) -> None:
        with CONFIG_PATH.open("r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)

        self.listen_udp = listen_udp
        self.socket = None
        if self.listen_udp:
            network = self.config["network"]
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((network["host"], int(network["port"])))
            self.socket.setblocking(False)

        self.app = SimulationApp({"headless": False})
        self.world = None
        self.robot = None
        self.joint_indices = None
        self.initial_target = np.asarray(
            self.config["initial_state"]["arm"] + [self.config["initial_state"]["gripper_m"]],
            dtype=np.float64,
        )
        self.target = self.initial_target.copy()
        self.last_sequence = -1
        self.reset_requested = False
        self.control_window = None

    def setup(self) -> None:
        """打开带相机的 World USD，并创建物理场景、桌面和香蕉。"""
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        asset = self.config["asset"]
        scene_path = REPO_ROOT / asset["scene_usd_path"]
        if not scene_path.is_file():
            raise FileNotFoundError(f"带相机的 DM 场景 USD 不存在: {scene_path}")
        if not open_stage(str(scene_path)):
            raise RuntimeError(f"无法打开带相机的 DM 场景 USD: {scene_path}")

        simulation = self.config["simulation"]
        physics_dt = 1.0 / float(simulation["physics_hz"])
        rendering_dt = 1.0 / float(simulation["rendering_hz"])
        self.world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)

        physics = self.world.get_physics_context()
        physics.set_solver_type(simulation["solver_type"])
        physics.enable_ccd(bool(simulation["enable_ccd"]))
        physics.enable_stablization(bool(simulation["enable_stabilization"]))

        materials = self.config["materials"]
        table_material = PhysicsMaterial(
            "/World/PhysicsMaterials/Table",
            static_friction=materials["table"]["static_friction"],
            dynamic_friction=materials["table"]["dynamic_friction"],
            restitution=materials["table"]["restitution"],
        )
        object_material = PhysicsMaterial(
            "/World/PhysicsMaterials/Object",
            static_friction=materials["object"]["static_friction"],
            dynamic_friction=materials["object"]["dynamic_friction"],
            restitution=materials["object"]["restitution"],
        )
        fingertip_material = PhysicsMaterial(
            "/World/PhysicsMaterials/Fingertip",
            static_friction=materials["fingertip"]["static_friction"],
            dynamic_friction=materials["fingertip"]["dynamic_friction"],
            restitution=materials["fingertip"]["restitution"],
        )

        stage = self.world.stage
        for prim_path in self.config["gripper"]["collision_prim_paths"]:
            collision_prim = stage.GetPrimAtPath(prim_path)
            if not collision_prim.IsValid() or not collision_prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(f"夹爪碰撞 Prim 无效: {prim_path}")
            UsdShade.MaterialBindingAPI.Apply(collision_prim).Bind(
                fingertip_material.material,
                materialPurpose="physics",
            )

        scene = self.config["scene"]
        self.world.scene.add(
            FixedCuboid(
                "/World/Ground",
                name="ground",
                position=np.array([0.0, 0.0, scene["ground_z"] - 0.01]),
                scale=np.array([4.0, 4.0, 0.02]),
                size=1.0,
                visible=True,
                physics_material=table_material,
            )
        )
        table = self.world.scene.add(
            FixedCuboid(
                "/World/Table",
                name="table",
                position=np.asarray(scene["table"]["position"]),
                scale=np.asarray(scene["table"]["size"]),
                size=1.0,
                color=np.array([0.58, 0.46, 0.34]),
                physics_material=table_material,
            )
        )

        object_config = scene["object"]
        object_asset = REPO_ROOT / object_config["asset_path"]
        if not object_asset.is_file():
            raise FileNotFoundError(f"香蕉 USD 不存在: {object_asset}")
        banana = stage.DefinePrim("/World/GraspObject", "Xform")
        banana.GetReferences().AddReference(str(object_asset), object_config["source_prim_path"])
        banana_xform = UsdGeom.XformCommonAPI(banana)
        banana_xform.SetTranslate(Gf.Vec3d(*object_config["position"]))
        banana_xform.SetRotate(
            Gf.Vec3f(*object_config["rotation_deg"]),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )
        banana_xform.SetScale(Gf.Vec3f(*object_config["scale"]))
        UsdPhysics.RigidBodyAPI.Apply(banana)
        UsdPhysics.MassAPI.Apply(banana).CreateMassAttr().Set(float(object_config["mass"]))

        banana_material = UsdShade.Material.Define(stage, "/World/Looks/BananaVisual")
        banana_shader = UsdShade.Shader.Define(stage, "/World/Looks/BananaVisual/Shader")
        banana_shader.CreateIdAttr("UsdPreviewSurface")
        banana_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.72, 0.08))
        banana_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        banana_material.CreateSurfaceOutput().ConnectToSource(banana_shader.ConnectableAPI(), "surface")

        banana_meshes = [prim for prim in Usd.PrimRange(banana) if prim.IsA(UsdGeom.Mesh)]
        if not banana_meshes:
            raise RuntimeError(f"香蕉资产没有 Mesh: {object_asset}")
        for mesh in banana_meshes:
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(banana_material)
            UsdPhysics.CollisionAPI.Apply(mesh)
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh)
            mesh_collision.CreateApproximationAttr().Set(object_config["collision_approximation"])
            PhysxSchema.PhysxCollisionAPI.Apply(mesh).CreateContactOffsetAttr().Set(0.002)
            UsdShade.MaterialBindingAPI.Apply(mesh).Bind(
                object_material.material,
                materialPurpose="physics",
            )

        table.set_contact_offset(0.002)

        self.robot = self.world.scene.add(SingleArticulation(asset["robot_prim_path"], name="rebotarm_dm"))
        self.world.reset()
        self.robot.initialize()

        dof_names = list(self.robot.dof_names)
        missing = [name for name in DRIVEN_JOINT_NAMES + ("right_finger",) if name not in dof_names]
        if missing:
            raise RuntimeError(f"DM USD 缺少关节: {missing}; 当前 DOF: {dof_names}")
        self.joint_indices = np.asarray([dof_names.index(name) for name in DRIVEN_JOINT_NAMES])

        self._apply_target()
        self._create_control_window()
        set_camera_view(
            eye=np.array([1.05, 0.95, 0.72]),
            target=np.array([0.25, 0.0, 0.18]),
            camera_prim_path="/OmniverseKit_Persp",
        )
        print(f"[DM] scene USD: {scene_path}")
        print(f"[DM] DOF: {dof_names}")
        print(
            f"[DM] physics={simulation['physics_hz']}Hz render={simulation['rendering_hz']}Hz "
            f"solver={simulation['solver_type']}"
        )

    def _create_control_window(self) -> None:
        """创建独立复位按钮，避免 standalone 模式依赖顶部 Stop。"""
        import omni.ui as ui

        self.control_window = ui.Window("reBotArm Controls", width=260, height=90)
        with self.control_window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Restore dm_sim.yaml initial_state")
                ui.Button("Reset Robot Pose", clicked_fn=self._request_reset)

    def _request_reset(self) -> None:
        self.reset_requested = True

    def _reset_scene(self) -> None:
        """恢复 YAML 初始关节目标和所有动态物体初始状态。"""
        self.target[:] = self.initial_target
        self.last_sequence = -1
        self.world.reset()
        self.robot.initialize()
        self._apply_target()
        self.reset_requested = False
        print(f"[DM] reset target={np.round(self.target, 4).tolist()}")

    def _receive_latest(self) -> None:
        """清空本轮 UDP 队列，并只采用 sequence 最大的有效数据包。"""
        if self.socket is None:
            return

        latest = None
        while True:
            try:
                packet, _address = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                payload = json.loads(packet.decode("utf-8"))
                arm = np.asarray(payload["joint_positions"], dtype=np.float64)
                if arm.shape != (6,) or not np.all(np.isfinite(arm)):
                    raise ValueError(f"joint_positions 必须是 6 个有限数值，当前 {arm}")
                candidate = (int(payload["sequence"]), arm, payload.get("gripper_position"))
                if latest is None or candidate[0] > latest[0]:
                    latest = candidate
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                print(f"[UDP] 忽略无效数据: {exc}")

        if latest is None:
            return

        sequence, arm, gripper = latest
        if sequence <= self.last_sequence:
            return

        limits = self.config["arm"]["joints"]
        self.target[:6] = np.asarray(
            [np.clip(value, limits[name]["lower"], limits[name]["upper"]) for name, value in zip(ARM_JOINT_NAMES, arm)]
        )
        if gripper is not None:
            gripper_config = self.config["gripper"]
            self.target[6] = np.clip(float(gripper), gripper_config["command_min_m"], gripper_config["command_max_m"])
        self.last_sequence = sequence

    def _apply_target(self) -> None:
        """通过 position drive 下发目标，不直接瞬移关节状态。"""
        from isaacsim.core.utils.types import ArticulationAction

        self.robot.apply_action(
            ArticulationAction(
                joint_positions=self.target.copy(),
                joint_indices=self.joint_indices,
            )
        )

    def run(self) -> None:
        """运行固定频率物理循环，并按较低频率渲染 GUI。"""
        simulation = self.config["simulation"]
        physics_hz = int(simulation["physics_hz"])
        render_every = max(1, round(physics_hz / int(simulation["rendering_hz"])))
        next_step = time.perf_counter()
        step = 0

        if self.listen_udp:
            network = self.config["network"]
            print(f"[UDP] listening on {network['host']}:{network['port']}")
        else:
            print("[UDP] disabled; scene-only mode")

        while _running and self.app.is_running():
            if self.reset_requested:
                self._reset_scene()
            if self.listen_udp:
                self._receive_latest()
                self._apply_target()
            self.world.step(render=(step % render_every == 0))

            if step % physics_hz == 0:
                actual = self.robot.get_joint_positions(joint_indices=self.joint_indices)
                print(
                    f"[DM] target={np.round(self.target, 4).tolist()} "
                    f"actual={np.round(actual, 4).tolist()}"
                )

            step += 1
            next_step += 1.0 / physics_hz
            delay = next_step - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:
                next_step = time.perf_counter()

    def close(self) -> None:
        """释放 UI、UDP socket 和 Isaac Sim 应用。"""
        self.control_window = None
        if self.socket is not None:
            self.socket.close()
        self.app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-udp",
        action="store_true",
        help="load and simulate the World scene without opening a UDP socket",
    )
    args, _isaac_args = parser.parse_known_args()

    simulation = DMSimulation(listen_udp=not args.no_udp)
    try:
        simulation.setup()
        simulation.run()
    finally:
        simulation.close()


if __name__ == "__main__":
    main()
