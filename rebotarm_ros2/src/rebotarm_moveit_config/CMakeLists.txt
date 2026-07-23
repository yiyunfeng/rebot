cmake_minimum_required(VERSION 3.22)
project(rebotarm_moveit_config)

find_package(ament_cmake REQUIRED)

install(FILES
  config/initial_positions.yaml
  config/joint_limits.yaml
  config/kinematics.yaml
  config/moveit_controllers.yaml
  config/moveit_hardware_controllers.yaml
  config/ompl_planning.yaml
  config/rebotarm.ros2_control.xacro
  config/rebotarm.srdf
  config/rebotarm.urdf.xacro
  config/ros2_controllers.yaml
  DESTINATION share/${PROJECT_NAME}/config
)

install(FILES
  launch/demo.launch.py
  launch/demo_view.rviz
  launch/hardware.launch.py
  launch/moveit.rviz
  launch/moveit_launch_common.py
  DESTINATION share/${PROJECT_NAME}/launch
)

install(FILES .setup_assistant
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
