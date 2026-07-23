cmake_minimum_required(VERSION 3.8)
project(rebotarm_gazebo11)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Protobuf REQUIRED)

set(GRASP_PLUGIN_DIR plugins/gazebo_grasp_plugin)
set(GAZEBO_HELPERS_DIR plugins/gazebo_version_helpers)

protobuf_generate_cpp(
  PROTO_SRCS
  PROTO_HDRS
  ${GRASP_PLUGIN_DIR}/msgs/grasp_event.proto
)

add_library(gazebo_grasp_msgs SHARED ${PROTO_SRCS})
target_include_directories(gazebo_grasp_msgs PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(gazebo_grasp_msgs ${Protobuf_LIBRARIES})

add_library(
  gazebo_version_helpers SHARED
  ${GAZEBO_HELPERS_DIR}/src/GazeboVersionHelpers.cpp
)
target_include_directories(
  gazebo_version_helpers PUBLIC
  ${GAZEBO_HELPERS_DIR}/include
  ${GAZEBO_INCLUDE_DIRS}
)
ament_target_dependencies(gazebo_version_helpers gazebo_dev)

add_library(
  gazebo_grasp_fix SHARED
  ${GRASP_PLUGIN_DIR}/src/GazeboGraspFix.cpp
  ${GRASP_PLUGIN_DIR}/src/GazeboGraspGripper.cpp
)
target_include_directories(
  gazebo_grasp_fix PUBLIC
  ${GRASP_PLUGIN_DIR}/include
  ${GAZEBO_HELPERS_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${GAZEBO_INCLUDE_DIRS}
)
ament_target_dependencies(
  gazebo_grasp_fix
  gazebo_dev
  gazebo_ros
  geometry_msgs
  rclcpp
  std_msgs
)
target_link_libraries(
  gazebo_grasp_fix
  gazebo_grasp_msgs
  gazebo_version_helpers
  ${GAZEBO_LIBRARIES}
)
add_dependencies(gazebo_grasp_fix gazebo_grasp_msgs gazebo_version_helpers)

ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src)

install(
  PROGRAMS
    scripts/gazebo_robot_description
    scripts/gripper_slider_gui
    scripts/joint_state_mirror
    scripts/moveit_pick_place
    scripts/planning_scene_objects
    scripts/simple_pick_place
    scripts/trajectory_relay
  DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS gazebo_grasp_fix gazebo_grasp_msgs gazebo_version_helpers
  ARCHIVE DESTINATION share/${PROJECT_NAME}/resource
  LIBRARY DESTINATION share/${PROJECT_NAME}/resource
  RUNTIME DESTINATION share/${PROJECT_NAME}/resource
)

install(DIRECTORY launch config rviz worlds DESTINATION share/${PROJECT_NAME})
install(FILES package.xml README.md .setup_assistant DESTINATION share/${PROJECT_NAME})

ament_package()
