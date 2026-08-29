cmake_minimum_required(VERSION 3.8)
project(lego_executor)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(oracle_moveit_executor src/oracle_moveit_executor.cpp)
target_compile_features(oracle_moveit_executor PUBLIC cxx_std_17)
ament_target_dependencies(oracle_moveit_executor
  control_msgs
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  rclcpp
  rclcpp_action
  std_msgs
  std_srvs
  tf2
  tf2_geometry_msgs
  trajectory_msgs
)
target_link_libraries(oracle_moveit_executor yaml-cpp)

install(TARGETS oracle_moveit_executor DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
