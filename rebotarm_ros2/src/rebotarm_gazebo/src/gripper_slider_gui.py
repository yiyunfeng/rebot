from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from builtin_interfaces.msg import Duration
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class GripperSliderGui(Node):
    """Simple popup slider that sends gripper JointTrajectory commands."""

    def __init__(self) -> None:
        super().__init__("gripper_slider_gui")

        self.declare_parameter("command_topic", "/gripper_controller/joint_trajectory")
        self.declare_parameter("joint_names", ["gripper_joint1", "gripper_joint2"])
        self.declare_parameter("min_position", 0.0)
        self.declare_parameter("max_position", 0.0715)
        self.declare_parameter("initial_position", 0.0)
        self.declare_parameter("motion_duration", 0.5)

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._joint_names = [str(name) for name in self.get_parameter("joint_names").value]
        self._min = float(self.get_parameter("min_position").value)
        self._max = float(self.get_parameter("max_position").value)
        self._duration = float(self.get_parameter("motion_duration").value)

        self._publisher = self.create_publisher(JointTrajectory, self._command_topic, 10)

    def publish_positions(self, positions: list[float]) -> None:
        positions = [max(self._min, min(self._max, float(position))) for position in positions]

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = _duration(self._duration)

        trajectory = JointTrajectory()
        trajectory.joint_names = list(self._joint_names)
        trajectory.points = [point]

        self._publisher.publish(trajectory)
        targets = ", ".join(
            f"{name}={position:.4f}" for name, position in zip(self._joint_names, positions)
        )
        self.get_logger().info(f"gripper -> {targets}")


def _duration(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int((seconds - sec) * 1_000_000_000))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GripperSliderGui()

    root = tk.Tk()
    root.title("Gazebo Gripper")
    root.resizable(False, False)

    initial = float(node.get_parameter("initial_position").value)
    values = [tk.DoubleVar(value=initial) for _ in node._joint_names]
    labels = [tk.StringVar(value=f"{initial:.4f} m") for _ in node._joint_names]

    def current_positions() -> list[float]:
        return [value.get() for value in values]

    def set_all(position: float) -> None:
        for value, label in zip(values, labels):
            value.set(position)
            label.set(f"{position:.4f} m")
        node.publish_positions(current_positions())

    def on_slide(index: int, raw_value: str) -> None:
        position = float(raw_value)
        labels[index].set(f"{position:.4f} m")

    def on_release(_event) -> None:
        node.publish_positions(current_positions())

    frame = ttk.Frame(root, padding=12)
    frame.grid(row=0, column=0, sticky="nsew")

    for index, joint_name in enumerate(node._joint_names):
        row = index * 2
        ttk.Label(frame, text=joint_name).grid(row=row, column=0, columnspan=3, sticky="w")
        ttk.Label(frame, textvariable=labels[index], width=10).grid(row=row, column=3, sticky="e")

        slider = ttk.Scale(
            frame,
            from_=node._min,
            to=node._max,
            orient="horizontal",
            variable=values[index],
            command=lambda raw, i=index: on_slide(i, raw),
            length=300,
        )
        slider.grid(row=row + 1, column=0, columnspan=4, pady=(6, 10), sticky="ew")
        slider.bind("<ButtonRelease-1>", on_release)

    button_row = len(node._joint_names) * 2
    ttk.Button(frame, text="Close", command=lambda: set_all(node._min)).grid(
        row=button_row, column=0, sticky="ew", padx=(0, 6)
    )
    ttk.Button(frame, text="Half", command=lambda: set_all((node._min + node._max) / 2.0)).grid(
        row=button_row, column=1, sticky="ew", padx=6
    )
    ttk.Button(frame, text="Open", command=lambda: set_all(node._max)).grid(
        row=button_row, column=2, sticky="ew", padx=6
    )
    ttk.Button(frame, text="Send", command=lambda: node.publish_positions(current_positions())).grid(
        row=button_row, column=3, sticky="ew", padx=(6, 0)
    )

    def tick() -> None:
        rclpy.spin_once(node, timeout_sec=0.0)
        root.after(50, tick)

    def close() -> None:
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    tick()

    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
