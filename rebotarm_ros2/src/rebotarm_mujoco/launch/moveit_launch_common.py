"""兼容入口。

ROS 2 launch 文件不能稳定 import 同目录模块，因此真正实现放在
rebotarm_mujoco.moveit_launch_common。这个文件保留给人工阅读时对照
Gazebo 包的同名文件。
"""

from rebotarm_mujoco.moveit_launch_common import moveit_parameters
