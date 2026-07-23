"""
Gazebo 机器人描述生成工具。

用途：将 URDF/xacro 文件转换为 Gazebo 可用的 URDF 或 SDF 格式，
并对关节参数进行调整（力矩限制、阻尼、初始位置等）。

两种输出格式：
    - URDF: 给 robot_state_publisher 用，发布 TF 树
    - SDF: 给 Gazebo 用，生成仿真世界中的机器人模型

工作流程（SDF 模式）：
    1. 解析 xacro → URDF XML
    2. 调整关节力矩限制（Gazebo 需要比真实值更大的力矩）
    3. URDF → SDF 转换（调用 gz sdf -p 命令）
    4. 设置关节初始位置（从 ros2_control 配置读取）
    5. 添加关节阻尼（让运动更平滑）
    6. 可选：添加 world → base_link 的固定关节
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import xacro


# ---------------------------------------------------------------------------
# Gazebo 关节参数
# ---------------------------------------------------------------------------
# 这些值针对 Gazebo 仿真调优，与 URDF 中真实的 effort limit 不同。
# Gazebo 需要更大的力矩来保证仿真稳定性。

# 每个关节在 Gazebo 中的最大力矩（Nm）
_GAZEBO_EFFORT_LIMITS = {
    "joint1": "120",
    "joint2": "220",
    "joint3": "180",
    "joint4": "80",
    "joint5": "80",
    "joint6": "50",
    "gripper_joint1": "1000",
    "gripper_joint2": "1000",
}

# 每个关节的阻尼系数（Nms/rad），让仿真运动更平滑
_GAZEBO_JOINT_DAMPING = {
    "joint1": "1.0",
    "joint2": "8.0",
    "joint3": "6.0",
    "joint4": "2.0",
    "joint5": "1.5",
    "joint6": "1.0",
    "gripper_joint1": "0.01",
    "gripper_joint2": "0.01",
}

# ---------------------------------------------------------------------------
# XML 辅助函数
# ---------------------------------------------------------------------------

def _find_model(sdf_root: ET.Element) -> ET.Element:
    """在 SDF 根元素中查找 <model> 子元素。"""
    model = sdf_root.find("model")
    if model is None:
        raise RuntimeError("转换后的 SDF 不含顶层 <model> 元素")
    return model


def _get_or_create(parent: ET.Element, tag: str) -> ET.Element:
    """获取子元素，如果不存在则创建。"""
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _set_text(parent: ET.Element, tag: str, text: str) -> None:
    """设置子元素的文本内容（子元素不存在则自动创建）。"""
    _get_or_create(parent, tag).text = text


# ---------------------------------------------------------------------------
# URDF 处理
# ---------------------------------------------------------------------------

def _tune_joint_effort(robot: ET.Element) -> None:
    """将 URDF 中所有关节的力矩限制改为 Gazebo 适用的值。

    遍历 <joint> 元素，将其 <limit effort="..."/> 替换为
    _GAZEBO_EFFORT_LIMITS 中的预设值。
    """
    for joint in robot.findall("joint"):
        joint_name = joint.attrib.get("name", "")
        new_effort = _GAZEBO_EFFORT_LIMITS.get(joint_name)
        if new_effort is None:
            continue
        limit = joint.find("limit")
        if limit is not None:
            limit.set("effort", new_effort)


# 如果以后夹爪 collision 又导致 Gazebo 物理卡死，可以临时启用这一段。
# 默认不要启用：否则方块无法和夹爪发生真实物理接触。
#
# _COLLISIONLESS_LINKS = {"gripper_link", "gripper_left", "gripper_right"}
#
#
# def _remove_gripper_collisions(robot: ET.Element) -> None:
#     """移除夹爪 link 的 <collision> 元素。"""
#     for link in robot.findall("link"):
#         if link.attrib.get("name", "") not in _COLLISIONLESS_LINKS:
#             continue
#         for collision in link.findall("collision"):
#             link.remove(collision)
#
#
def process_urdf(xacro_path: str) -> str:
    """处理 xacro 文件，生成调整后的 URDF 字符串。

    步骤：
        1. 用 xacro 库展开所有宏和 include
        2. 调整为 Gazebo 适用的关节力矩
        3. 返回 URDF XML 字符串

    Args:
        xacro_path: .urdf.xacro 文件路径。

    Returns:
        处理后的 URDF XML 字符串。
    """
    # xacro.process_file 会处理 <xacro:include> 和 <xacro:macro> 展开
    doc = xacro.process_file(xacro_path)
    xml_text = doc.toprettyxml(indent="  ")
    robot = ET.fromstring(xml_text)

    _tune_joint_effort(robot)

    return ET.tostring(robot, encoding="unicode")


# ---------------------------------------------------------------------------
# SDF 处理
# ---------------------------------------------------------------------------

def _add_world_fixed_joint(sdf_root: ET.Element, child_link: str) -> None:
    """在 SDF 模型中添加 world → base_link 的固定关节。

    让机械臂固定在 Gazebo 世界中，不会因重力掉落。
    如果关节已存在则跳过。

    Args:
        sdf_root: SDF XML 根元素。
        child_link: 固定关节的子 link（通常是 base_link）。
    """
    model = _find_model(sdf_root)

    # 检查是否已有 world_fixed
    for joint in model.findall("joint"):
        if joint.attrib.get("name") == "world_fixed":
            return

    # 创建固定关节
    joint = ET.Element("joint", {"name": "world_fixed", "type": "fixed"})
    _set_text(joint, "parent", "world")
    _set_text(joint, "child", child_link)
    model.insert(0, joint)


def _extract_initial_positions(robot: ET.Element) -> dict[str, str]:
    """从 ros2_control 配置中读取关节初始位置。

    读取路径：
        <ros2_control>/<joint>/<state_interface name="position">/<param name="initial_value">

    Returns:
        {关节名: 初始位置值} 字典。
    """
    positions = {}
    for joint in robot.findall("./ros2_control/joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        for si in joint.findall("state_interface"):
            if si.attrib.get("name") != "position":
                continue
            for param in si.findall("param"):
                if param.attrib.get("name") == "initial_value" and param.text:
                    positions[name] = param.text.strip()
                    break
    return positions


def _apply_initial_positions(
    sdf_root: ET.Element,
    initial_positions: dict[str, str],
) -> None:
    """将关节初始位置写入 SDF 模型。

    对每个关节，在 <axis> 下添加 <initial_position> 元素。
    SDF 版本设为 1.7 以确保兼容 initial_position 字段。
    """
    model = _find_model(sdf_root)
    sdf_root.set("version", "1.7")

    for joint in model.findall("joint"):
        name = joint.attrib.get("name", "")
        pos = initial_positions.get(name)
        if pos is None:
            continue
        axis = joint.find("axis")
        if axis is None:
            continue

        # 移除旧值后插入新值（放在 axis 的第一个子元素位置）
        old = axis.find("initial_position")
        if old is not None:
            axis.remove(old)
        elem = ET.Element("initial_position")
        elem.text = pos
        axis.insert(0, elem)


def _apply_joint_damping(sdf_root: ET.Element) -> None:
    """为 SDF 中每个关节添加阻尼系数。

    阻尼让关节运动更平滑，防止仿真中机械臂震荡。
    阻尼值来自 _GAZEBO_JOINT_DAMPING。
    """
    model = _find_model(sdf_root)
    for joint in model.findall("joint"):
        name = joint.attrib.get("name", "")
        damping = _GAZEBO_JOINT_DAMPING.get(name)
        if damping is None:
            continue
        axis = joint.find("axis")
        if axis is None:
            continue
        dynamics = _get_or_create(axis, "dynamics")
        _set_text(dynamics, "damping", damping)


def process_sdf(
    xacro_path: str,
    *,
    world_fixed: bool,
    fixed_child_link: str,
) -> str:
    """处理 xacro 文件，生成 Gazebo SDF 字符串。

    完整流程（7 步）：
        1. 生成调整后的 URDF
        2. 从 ros2_control 提取初始关节位置（备用）
        3. 调用 gz sdf -p 将 URDF 转为 SDF
        4. 写入关节初始位置
        5. 写入关节阻尼
        6. 可选：添加 world fixed joint

    Args:
        xacro_path: .urdf.xacro 文件路径。
        world_fixed: 是否添加 world→base_link 固定关节。
        fixed_child_link: 固定关节的子 link 名。

    Returns:
        Gazebo SDF XML 字符串。
    """
    # 1. 生成调整后的 URDF
    urdf_text = process_urdf(xacro_path)
    robot = ET.fromstring(urdf_text)

    # 2. 提取初始位置（在移除碰撞体之前）
    initial_positions = _extract_initial_positions(robot)

    # 3. 默认保留夹爪 collision，让 Gazebo 方块能和夹爪发生真实接触。
    #    如果需要回避夹爪物理卡死，可恢复上面的 _remove_gripper_collisions()
    #    并在这里调用：
    #
    #    _remove_gripper_collisions(robot)
    urdf_text = ET.tostring(robot, encoding="unicode")

    # 4. URDF → SDF：写入临时文件，调用 Gazebo Classic 转换命令
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(urdf_text)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["gz", "sdf", "-p", tmp_path],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(tmp_path)  # 无论如何删除临时文件

    # 5-7. SDF 后处理
    sdf_root = ET.fromstring(result.stdout)
    _apply_initial_positions(sdf_root, initial_positions)
    _apply_joint_damping(sdf_root)
    if world_fixed:
        _add_world_fixed_joint(sdf_root, fixed_child_link)

    return ET.tostring(sdf_root, encoding="unicode")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """命令行工具入口。

    用法示例：
        # 输出 URDF（给 robot_state_publisher）
        ros2 run rebotarm_gazebo11 gazebo_robot_description robot.urdf.xacro

        # 输出 SDF（给 Gazebo 生成模型）
        ros2 run rebotarm_gazebo11 gazebo_robot_description robot.urdf.xacro \\
            --format sdf --world-fixed
    """
    parser = argparse.ArgumentParser(
        description="将 xacro/URDF 转换为 Gazebo 兼容的 URDF 或 SDF 格式"
    )
    parser.add_argument("xacro_file", help="输入的 .urdf.xacro 文件路径")
    parser.add_argument(
        "--format",
        choices=("urdf", "sdf"),
        default="urdf",
        help="输出格式: urdf（给 robot_state_publisher）或 sdf（给 Gazebo）",
    )
    parser.add_argument(
        "--world-fixed",
        action="store_true",
        help="在 SDF 中添加 world → base_link 的固定关节",
    )
    parser.add_argument(
        "--fixed-child-link",
        default="base_link",
        help="world fixed joint 的子 link 名称（默认 base_link）",
    )
    args = parser.parse_args(argv)

    if args.format == "sdf":
        output = process_sdf(
            args.xacro_file,
            world_fixed=args.world_fixed,
            fixed_child_link=args.fixed_child_link,
        )
    else:
        output = process_urdf(args.xacro_file)

    sys.stdout.write(output)


if __name__ == "__main__":
    main()
