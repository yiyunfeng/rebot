#!/usr/bin/env python3
"""Import the tuned DM MuJoCo model and generate a reusable Isaac Sim USD."""

from __future__ import annotations

import math
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from isaacsim import SimulationApp


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "dm_sim.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def find_joint(stage, joint_name: str):
    from pxr import UsdPhysics

    for prim in stage.Traverse():
        if prim.GetName() == joint_name and (
            prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
        ):
            return prim
    raise RuntimeError(f"Imported USD is missing joint: {joint_name}")


def find_prim(stage, prim_name: str):
    for prim in stage.Traverse():
        if prim.GetName() == prim_name:
            return prim
    raise RuntimeError(f"Imported USD is missing prim: {prim_name}")


def configure_drive(joint_prim, drive_name: str, values: dict) -> None:
    from pxr import Sdf, UsdPhysics

    drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_name)
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_name)

    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(float(values["stiffness"]))
    drive.CreateDampingAttr().Set(float(values["damping"]))
    drive.CreateMaxForceAttr().Set(float(values["max_force"]))
    drive.CreateTargetVelocityAttr().Set(0.0)
    max_velocity = float(values["max_velocity"])
    if drive_name == "angular":
        max_velocity = math.degrees(max_velocity)
    joint_prim.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(max_velocity)


def configure_joint_physics(joint_prim, armature: float, friction: float) -> None:
    from pxr import PhysxSchema

    joint_api = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
    joint_api.CreateArmatureAttr().Set(float(armature))
    joint_api.CreateJointFrictionAttr().Set(float(friction))


def configure_gripper_mimic(stage) -> None:
    """Match the MuJoCo equality: right_finger position equals left_finger."""
    from pxr import PhysxSchema, UsdPhysics

    left_joint = find_joint(stage, "left_finger")
    right_joint = find_joint(stage, "right_finger")

    # MuJoCo only drives left_finger; right_finger follows through equality.
    right_joint.RemoveAPI(UsdPhysics.DriveAPI, "linear")
    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(right_joint, UsdPhysics.Tokens.rotX)
    mimic.GetReferenceJointRel().SetTargets([left_joint.GetPath()])
    mimic.GetReferenceJointAxisAttr().Set(UsdPhysics.Tokens.rotX)
    # PhysX defines reference = -gearing * mimic - offset.
    mimic.GetGearingAttr().Set(-1.0)
    mimic.GetOffsetAttr().Set(0.0)


def configure_fingertip_material(stage, config: dict) -> None:
    from pxr import UsdPhysics, UsdShade

    root_path = stage.GetDefaultPrim().GetPath()
    material = UsdShade.Material.Define(stage, root_path.AppendPath("PhysicsMaterials/Fingertip"))
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    values = config["materials"]["fingertip"]
    material_api.CreateStaticFrictionAttr().Set(float(values["static_friction"]))
    material_api.CreateDynamicFrictionAttr().Set(float(values["dynamic_friction"]))
    material_api.CreateRestitutionAttr().Set(float(values["restitution"]))

    for link_name in ("left_finger_link", "right_finger_link"):
        link_prim = find_prim(stage, link_name)
        collision_prim = stage.GetPrimAtPath(link_prim.GetPath().AppendChild("collisions"))
        if not collision_prim.IsValid():
            raise RuntimeError(f"Imported USD is missing finger collisions: {link_name}")
        binding = UsdShade.MaterialBindingAPI.Apply(collision_prim)
        binding.Bind(material, UsdShade.Tokens.strongerThanDescendants, "physics")


def configure_articulation(stage, config: dict) -> None:
    from pxr import PhysxSchema, Sdf, UsdPhysics

    articulation_prim = None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_prim = prim
            break
    if articulation_prim is None:
        raise RuntimeError("Imported USD has no ArticulationRootAPI")

    simulation = config["simulation"]
    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(articulation_prim)
    articulation_api.CreateSolverPositionIterationCountAttr().Set(
        int(simulation["solver_position_iterations"])
    )
    articulation_api.CreateSolverVelocityIterationCountAttr().Set(
        int(simulation["solver_velocity_iterations"])
    )
    articulation_prim.CreateAttribute(
        "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool
    ).Set(False)


def configure_robot(stage, config: dict) -> None:
    from pxr import UsdPhysics

    for joint_name, values in config["arm"]["joints"].items():
        configure_drive(find_joint(stage, joint_name), "angular", values)

    gripper = config["gripper"]
    gripper_drive = {
        "stiffness": gripper["stiffness"],
        "damping": gripper["damping"],
        "max_force": gripper["max_force"],
        "max_velocity": gripper["max_velocity"],
    }
    for joint_name in gripper["joints"]:
        joint_prim = find_joint(stage, joint_name)
        joint = UsdPhysics.PrismaticJoint(joint_prim)
        joint.CreateLowerLimitAttr().Set(float(gripper["lower_limit_m"]))
        joint.CreateUpperLimitAttr().Set(float(gripper["upper_limit_m"]))
        configure_joint_physics(
            joint_prim,
            armature=gripper["armature"],
            friction=gripper["joint_friction"],
        )

    configure_drive(find_joint(stage, "left_finger"), "linear", gripper_drive)
    configure_gripper_mimic(stage)
    configure_fingertip_material(stage, config)
    configure_articulation(stage, config)

    find_prim(stage, "gripper_base")
    if any(prim.GetName() == "camera_focus" for prim in stage.Traverse()):
        raise RuntimeError("Obsolete camera_focus helper was imported")


def backup_existing(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.{stamp}.bak{path.suffix}")
    shutil.copy2(path, backup)
    print(f"[BuildDM] Existing USD backed up to: {backup}")


def main() -> None:
    config = load_config()
    source_path = REPO_ROOT / config["asset"]["source_mjcf"]
    usd_path = REPO_ROOT / config["asset"]["usd_path"]
    if not source_path.is_file():
        raise FileNotFoundError(f"DM MJCF not found: {source_path}")

    usd_path.parent.mkdir(parents=True, exist_ok=True)
    backup_existing(usd_path)

    simulation_app = SimulationApp({"headless": True})
    try:
        import omni.kit.app
        import omni.kit.commands
        from pxr import PhysxSchema, Usd, UsdPhysics

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)
        for _ in range(5):
            simulation_app.update()

        status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
        if not status:
            raise RuntimeError("MJCFCreateImportConfig failed")
        import_config.set_fix_base(True)
        import_config.set_import_inertia_tensor(True)
        import_config.set_distance_scale(1.0)
        import_config.set_density(0.0)
        import_config.set_self_collision(False)
        import_config.set_make_default_prim(True)
        import_config.set_create_physics_scene(False)
        import_config.set_import_sites(True)
        import_config.set_visualize_collision_geoms(False)

        status, result = omni.kit.commands.execute(
            "MJCFCreateAsset",
            mjcf_path=str(source_path),
            import_config=import_config,
            prim_path="/reBotArmDM",
            dest_path=str(usd_path),
        )
        if not status:
            raise RuntimeError(f"MJCFCreateAsset failed: {result}")

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"Unable to open generated USD: {usd_path}")
        configure_robot(stage, config)
        stage.GetRootLayer().Save()

        joint_names = [
            prim.GetName()
            for prim in stage.Traverse()
            if prim.GetTypeName() in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
        ]
        print(f"[BuildDM] USD: {usd_path}")
        print(f"[BuildDM] defaultPrim: {stage.GetDefaultPrim().GetPath()}")
        print(f"[BuildDM] joints: {joint_names}")
        print(f"[BuildDM] gripper base: {find_prim(stage, 'gripper_base').GetPath()}")
        right_joint = find_joint(stage, "right_finger")
        mimic = PhysxSchema.PhysxMimicJointAPI(right_joint, UsdPhysics.Tokens.rotX)
        print(
            "[BuildDM] gripper mimic: right_finger -> left_finger, "
            f"gearing={mimic.GetGearingAttr().Get()}"
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
