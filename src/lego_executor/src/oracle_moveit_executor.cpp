// Copyright 2026 Wenbo Ma
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <tf2/LinearMath/Quaternion.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
using namespace std::chrono_literals;

namespace
{

struct BlockState
{
  std::string id;
  std::string type;
  std::string color;
  geometry_msgs::msg::Pose pose;
};

struct Task
{
  std::string id;
  std::string type;
  std::string color;
  double grasp_spin_deg{90.0};
  geometry_msgs::msg::Pose target;
};

geometry_msgs::msg::Quaternion down_orientation(double yaw)
{
  tf2::Quaternion quaternion;
  quaternion.setRPY(M_PI, 0.0, yaw);
  return tf2::toMsg(quaternion);
}

double yaw_from_wxyz(const YAML::Node & value)
{
  if (!value || value.size() != 4) {
    return 0.0;
  }
  const double w = value[0].as<double>();
  const double x = value[1].as<double>();
  const double y = value[2].as<double>();
  const double z = value[3].as<double>();
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

std::vector<double> dimensions_for(const std::string & type)
{
  if (type == "brick_4x2") {
    return {0.064, 0.032, 0.019};
  }
  return {0.032, 0.032, 0.019};
}

class ExecutorNode : public rclcpp::Node
{
public:
  ExecutorNode()
  : Node("oracle_moveit_executor")
  {
    declare_parameter("auto_reset", true);
    declare_parameter("plan_file", "");
    declare_parameter("wait_timeout_s", 30.0);
    declare_parameter("planning_time_s", 10.0);
    declare_parameter("planning_attempts", 3);
    declare_parameter("velocity_scale", 0.75);
    declare_parameter("acceleration_scale", 0.75);
    declare_parameter("approach_height_m", 0.15);
    declare_parameter("gripper_offset_m", 0.1234);
    declare_parameter("grasp_descent_m", 0.170);
    declare_parameter("place_descent_m", 0.155);
    declare_parameter("place_press_depth_m", 0.001);
    declare_parameter("cartesian_speed_scale", 0.30);
    declare_parameter("place_speed_scale", 0.30);
    declare_parameter("place_acceleration_scale", 0.20);
    declare_parameter("lift_speed_scale", 0.55);
    declare_parameter("transport_rotation_speed_scale", 0.30);
    declare_parameter("transport_velocity_scale", 0.65);
    declare_parameter("transport_acceleration_scale", 0.30);
    declare_parameter("large_part_motion_scale", 1.0);
    declare_parameter("gripper_open_m", 0.04);
    declare_parameter("gripper_closed_m", 0.014);
    declare_parameter("gripper_duration_s", 0.35);
    declare_parameter("motion_settle_s", 0.10);
    declare_parameter("release_settle_s", 0.20);
    declare_parameter("grasp_confirmation_timeout_s", 5.0);
    declare_parameter("tool_yaw_offset_deg", 45.0);
    declare_parameter("verify_lift_m", 0.025);
    declare_parameter("source_mode", "oracle");
    declare_parameter("perception_service", "/lego_vision/detect");
    declare_parameter("placement_perception_service", "/lego_vision/detect_placement");
    declare_parameter("visual_place_correction", true);
    declare_parameter("visual_place_correction_iterations", 2);
    declare_parameter(
      "visual_place_correction_excluded_colors", std::vector<std::string>{"white"});
    declare_parameter("observe_pose_xyz", std::vector<double>{0.45, -0.35, 0.50});
    declare_parameter(
      "observe_joint_positions",
      std::vector<double>{-2.435602, -0.169040, 1.699489, -1.853112,
        0.175230, 1.871465, -1.571720});
    declare_parameter("grasp_offset_xy", std::vector<double>{0.0, 0.0});
    declare_parameter("place_offset_xy", std::vector<double>{0.0, 0.0});

    goal_subscription_ = create_subscription<std_msgs::msg::String>(
      "/lego_bench/goal", 10,
      [this](const std_msgs::msg::String::SharedPtr message) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        goal_ = YAML::Load(message->data);
        data_cv_.notify_all();
      });
    if (source_mode() != "oracle" && source_mode() != "vision") {
      throw std::runtime_error("source_mode must be 'oracle' or 'vision'");
    }
    const std::string state_topic =
      source_mode() == "oracle" ? "/lego_bench/ground_truth" : "/mj_bridge/benchmark_state";
    state_subscription_ = create_subscription<std_msgs::msg::String>(
      state_topic, 10,
      [this](const std_msgs::msg::String::SharedPtr message) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        state_ = YAML::Load(message->data);
        data_cv_.notify_all();
      });
    reset_client_ = create_client<std_srvs::srv::Trigger>("/mj_bridge/reset");
    result_client_ = create_client<std_srvs::srv::Trigger>("/mj_bridge/result");
    perception_client_ = create_client<std_srvs::srv::Trigger>(
      get_parameter("perception_service").as_string());
    placement_perception_client_ = create_client<std_srvs::srv::Trigger>(
      get_parameter("placement_perception_service").as_string());
    gripper_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
      this, "/mj_panda_hand_controller/follow_joint_trajectory");
  }

  bool wait_for_benchmark()
  {
    const auto timeout = std::chrono::duration<double>(get_parameter("wait_timeout_s").as_double());
    std::unique_lock<std::mutex> lock(data_mutex_);
    return data_cv_.wait_for(
      lock, timeout, [this]() {return goal_ && (source_mode() == "vision" || state_);});
  }

  std::string source_mode() const
  {
    return get_parameter("source_mode").as_string();
  }

  bool call_reset()
  {
    if (!get_parameter("auto_reset").as_bool()) {
      return true;
    }
    if (!reset_client_->wait_for_service(30s)) {
      RCLCPP_ERROR(get_logger(), "Benchmark reset service is unavailable");
      return false;
    }
    auto future = reset_client_->async_send_request(
      std::make_shared<std_srvs::srv::Trigger::Request>());
    if (future.wait_for(30s) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Benchmark reset timed out");
      return false;
    }
    RCLCPP_INFO(get_logger(), "Benchmark episode reset");
    std::this_thread::sleep_for(
      std::chrono::duration<double>(get_parameter("motion_settle_s").as_double()));
    return future.get()->success;
  }

  std::vector<Task> tasks() const
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::map<std::string, YAML::Node> targets;
    const YAML::Node target_nodes = goal_["target_blocks"];
    for (std::size_t index = 0; index < target_nodes.size(); ++index) {
      const YAML::Node target = target_nodes[index];
      targets[target["id"].as<std::string>()] = target;
    }

    std::vector<std::pair<std::string, double>> order;
    const std::string plan_file = get_parameter("plan_file").as_string();
    if (!plan_file.empty()) {
      YAML::Node plan = YAML::LoadFile(plan_file);
      for (const auto & step : plan["steps"]) {
        order.emplace_back(
          step["id"].as<std::string>(), step["grasp_spin_deg"].as<double>(90.0));
      }
    } else {
      for (std::size_t index = 0; index < target_nodes.size(); ++index) {
        const YAML::Node target = target_nodes[index];
        order.emplace_back(target["id"].as<std::string>(), 90.0);
      }
      std::stable_sort(
        order.begin(), order.end(), [&targets](
          const std::pair<std::string, double> & left,
          const std::pair<std::string, double> & right) {
          return targets[left.first]["position"][2].as<double>() <
          targets[right.first]["position"][2].as<double>();
        });
    }

    std::vector<Task> result;
    for (const auto & item : order) {
      if (!targets.count(item.first)) {
        throw std::runtime_error("plan references unknown block ID: " + item.first);
      }
      const YAML::Node target = targets[item.first];
      Task task;
      task.id = item.first;
      task.type = target["type"].as<std::string>();
      task.color = target["color"].as<std::string>("unknown");
      task.grasp_spin_deg = item.second;
      task.target.position.x = target["position"][0].as<double>();
      task.target.position.y = target["position"][1].as<double>();
      task.target.position.z = target["position"][2].as<double>();
      const double yaw = target["yaw_rad"].as<double>(0.0);
      const double offset = get_parameter("tool_yaw_offset_deg").as_double() * M_PI / 180.0;
      task.target.orientation = down_orientation(yaw + offset);
      result.push_back(task);
    }
    return result;
  }

  BlockState block(const std::string & id) const
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (source_mode() == "vision") {
      const auto found = visual_sources_.find(id);
      if (found == visual_sources_.end()) {
        throw std::runtime_error("vision source is unavailable for block ID: " + id);
      }
      return found->second;
    }
    const YAML::Node value = state_["blocks"][id];
    if (!value) {
      throw std::runtime_error("ground truth does not contain block ID: " + id);
    }
    BlockState block;
    block.id = id;
    block.type = value["type"].as<std::string>();
    block.color = value["color"].as<std::string>("unknown");
    block.pose.position.x = value["position"][0].as<double>();
    block.pose.position.y = value["position"][1].as<double>();
    block.pose.position.z = value["position"][2].as<double>();
    const double offset = get_parameter("tool_yaw_offset_deg").as_double() * M_PI / 180.0;
    block.pose.orientation = down_orientation(yaw_from_wxyz(value["quaternion_wxyz"]) + offset);
    return block;
  }

  bool wait_for_grasp(const Task & task)
  {
    const auto timeout = std::chrono::duration<double>(
      get_parameter("grasp_confirmation_timeout_s").as_double());
    std::unique_lock<std::mutex> lock(data_mutex_);
    return data_cv_.wait_for(
      lock, timeout, [this, &task]() {
        const YAML::Node grasped = state_["grasped_block"];
        if (!grasped) {
          return false;
        }
        const std::string grasped_id = grasped.as<std::string>("");
        if (source_mode() == "oracle") {
          return grasped_id == task.id;
        }
        const YAML::Node blocks = goal_["target_blocks"];
        for (std::size_t index = 0; index < blocks.size(); ++index) {
          const YAML::Node candidate = blocks[index];
          if (candidate["id"].as<std::string>("") != grasped_id) {
            continue;
          }
          return candidate["type"].as<std::string>("") == task.type &&
          candidate["color"].as<std::string>("unknown") == task.color;
        }
        return false;
      });
  }

  bool detect_source(const Task & task)
  {
    if (source_mode() != "vision") {
      return true;
    }
    if (!perception_client_->wait_for_service(20s)) {
      RCLCPP_ERROR(get_logger(), "Visual perception service is unavailable");
      return false;
    }
    auto future = perception_client_->async_send_request(
      std::make_shared<std_srvs::srv::Trigger::Request>());
    if (future.wait_for(180s) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Visual perception timed out");
      return false;
    }
    const auto response = future.get();
    if (!response->success) {
      RCLCPP_ERROR(get_logger(), "Visual perception failed: %s", response->message.c_str());
      return false;
    }

    const YAML::Node payload = YAML::Load(response->message);
    const YAML::Node detections = payload["detections"];
    YAML::Node selected;
    bool found = false;
    double selected_score = -1.0;
    for (std::size_t index = 0; index < detections.size(); ++index) {
      const YAML::Node candidate = detections[index];
      if (candidate["type"].as<std::string>("") != task.type ||
        candidate["color"].as<std::string>("") != task.color)
      {
        continue;
      }
      const double score = candidate["score"].as<double>(0.0);
      if (score > selected_score) {
        selected = candidate;
        found = true;
        selected_score = score;
      }
    }
    if (!found) {
      RCLCPP_ERROR(
        get_logger(), "Vision did not find a %s %s for task %s",
        task.color.c_str(), task.type.c_str(), task.id.c_str());
      return false;
    }

    BlockState source;
    source.id = task.id;
    source.type = task.type;
    source.color = task.color;
    source.pose.position.x = selected["position"][0].as<double>();
    source.pose.position.y = selected["position"][1].as<double>();
    source.pose.position.z = selected["position"][2].as<double>();
    const double yaw = selected["yaw_rad"].as<double>(0.0);
    const double offset = get_parameter("tool_yaw_offset_deg").as_double() * M_PI / 180.0;
    source.pose.orientation = down_orientation(yaw + offset);
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      visual_sources_[task.id] = source;
    }
    RCLCPP_INFO(
      get_logger(),
      "[%s] VISUAL_SOURCE backend=%s position=(%.4f, %.4f, %.4f) yaw=%.1fdeg score=%.3f",
      task.id.c_str(), payload["backend"].as<std::string>("unknown").c_str(),
      source.pose.position.x, source.pose.position.y, source.pose.position.z,
      yaw * 180.0 / M_PI, selected_score);
    return true;
  }

  bool detect_carried(const Task & task, BlockState & carried)
  {
    if (!placement_perception_client_->wait_for_service(20s)) {
      RCLCPP_ERROR(get_logger(), "Placement perception service is unavailable");
      return false;
    }
    auto future = placement_perception_client_->async_send_request(
      std::make_shared<std_srvs::srv::Trigger::Request>());
    if (future.wait_for(180s) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Placement perception timed out");
      return false;
    }
    const auto response = future.get();
    if (!response->success) {
      RCLCPP_ERROR(get_logger(), "Placement perception failed: %s", response->message.c_str());
      return false;
    }

    const YAML::Node payload = YAML::Load(response->message);
    const YAML::Node detections = payload["detections"];
    YAML::Node selected;
    bool found = false;
    double selected_height = std::numeric_limits<double>::max();
    for (std::size_t index = 0; index < detections.size(); ++index) {
      const YAML::Node candidate = detections[index];
      if (candidate["type"].as<std::string>("") != task.type ||
        candidate["color"].as<std::string>("") != task.color)
      {
        continue;
      }
      const double x = candidate["position"][0].as<double>();
      const double y = candidate["position"][1].as<double>();
      const double z = candidate["position"][2].as<double>();
      if (std::hypot(x - task.target.position.x, y - task.target.position.y) > 0.08) {
        continue;
      }
      // The held part is the lowest matching coloured object in the elevated
      // assembly ROI; the hand and wrist remain above it in the camera view.
      if (z < selected_height) {
        selected = candidate;
        found = true;
        selected_height = z;
      }
    }
    if (!found) {
      RCLCPP_ERROR(
        get_logger(), "Vision did not find the carried %s %s for task %s",
        task.color.c_str(), task.type.c_str(), task.id.c_str());
      return false;
    }

    carried.id = task.id;
    carried.type = task.type;
    carried.color = task.color;
    carried.pose.position.x = selected["position"][0].as<double>();
    carried.pose.position.y = selected["position"][1].as<double>();
    carried.pose.position.z = selected["position"][2].as<double>();
    const double yaw = selected["yaw_rad"].as<double>(0.0);
    carried.pose.orientation = down_orientation(yaw);
    RCLCPP_INFO(
      get_logger(),
      "[%s] VISUAL_PLACE_SOURCE backend=%s position=(%.4f, %.4f, %.4f) yaw=%.1fdeg",
      task.id.c_str(), payload["backend"].as<std::string>("unknown").c_str(),
      carried.pose.position.x, carried.pose.position.y, carried.pose.position.z,
      yaw * 180.0 / M_PI);
    return true;
  }

  bool command_gripper(double position)
  {
    if (!gripper_client_->wait_for_action_server(10s)) {
      RCLCPP_ERROR(get_logger(), "Panda hand action server is unavailable");
      return false;
    }
    FollowJointTrajectory::Goal goal;
    goal.trajectory.joint_names = {"panda_finger_joint1", "panda_finger_joint2"};
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = {position, position};
    point.time_from_start = rclcpp::Duration::from_seconds(
      get_parameter("gripper_duration_s").as_double());
    goal.trajectory.points.push_back(point);
    auto handle_future = gripper_client_->async_send_goal(goal);
    if (handle_future.wait_for(5s) != std::future_status::ready || !handle_future.get()) {
      RCLCPP_ERROR(get_logger(), "Panda hand goal was rejected or timed out");
      return false;
    }
    auto result_future = gripper_client_->async_get_result(handle_future.get());
    if (result_future.wait_for(8s) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Panda hand result timed out");
      return false;
    }
    return result_future.get().code == rclcpp_action::ResultCode::SUCCEEDED;
  }

  std::string export_result()
  {
    if (!result_client_->wait_for_service(10s)) {
      return "result service unavailable";
    }
    auto future = result_client_->async_send_request(
      std::make_shared<std_srvs::srv::Trigger::Request>());
    if (future.wait_for(10s) != std::future_status::ready) {
      return "result service timed out";
    }
    return future.get()->message;
  }

private:
  mutable std::mutex data_mutex_;
  std::condition_variable data_cv_;
  YAML::Node goal_;
  YAML::Node state_;
  std::map<std::string, BlockState> visual_sources_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr goal_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_subscription_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr result_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr perception_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr placement_perception_client_;
  rclcpp_action::Client<FollowJointTrajectory>::SharedPtr gripper_client_;
};

bool move_linear_to_pose(
  ExecutorNode & node,
  moveit::planning_interface::MoveGroupInterface & arm,
  const geometry_msgs::msg::Pose & target,
  double speed_scale,
  double acceleration_scale = -1.0);

bool move_to_pose(
  ExecutorNode & node,
  moveit::planning_interface::MoveGroupInterface & arm,
  const geometry_msgs::msg::Pose & pose,
  const std::string & label,
  bool keep_tool_down = false)
{
  if (keep_tool_down) {
    moveit_msgs::msg::OrientationConstraint orientation;
    orientation.header.frame_id = arm.getPlanningFrame();
    orientation.link_name = arm.getEndEffectorLink();
    orientation.orientation = pose.orientation;
    orientation.absolute_x_axis_tolerance = 0.45;
    orientation.absolute_y_axis_tolerance = 0.45;
    orientation.absolute_z_axis_tolerance = M_PI;
    orientation.weight = 1.0;
    moveit_msgs::msg::Constraints constraints;
    constraints.orientation_constraints.push_back(orientation);
    arm.setPathConstraints(constraints);
  }
  const int attempts = node.get_parameter("planning_attempts").as_int();
  for (int attempt = 1; attempt <= attempts; ++attempt) {
    arm.setStartStateToCurrentState();
    arm.setPoseTarget(pose);
    if (arm.move() == moveit::core::MoveItErrorCode::SUCCESS) {
      arm.clearPoseTargets();
      arm.clearPathConstraints();
      // Wait for one or two bridge publications before deriving the next
      // Cartesian trajectory from the measured state.
      std::this_thread::sleep_for(
        std::chrono::duration<double>(node.get_parameter("motion_settle_s").as_double()));
      return true;
    }
    RCLCPP_WARN(
      node.get_logger(), "%s planning attempt %d/%d failed", label.c_str(), attempt, attempts);
  }
  arm.clearPoseTargets();
  arm.clearPathConstraints();
  RCLCPP_WARN(
    node.get_logger(),
    "%s MoveIt planning failed; trying a direct Cartesian fallback without collision checking",
    label.c_str());
  return move_linear_to_pose(
    node, arm, pose, node.get_parameter("cartesian_speed_scale").as_double());
}

bool move_linear_to_pose(
  ExecutorNode & node,
  moveit::planning_interface::MoveGroupInterface & arm,
  const geometry_msgs::msg::Pose & target,
  double speed_scale,
  double acceleration_scale)
{
  arm.setStartStateToCurrentState();
  moveit_msgs::msg::RobotTrajectory trajectory_message;
  const double fraction = arm.computeCartesianPath({target}, 0.005, 0.0, trajectory_message, false);
  if (fraction < 0.98) {
    RCLCPP_ERROR(node.get_logger(), "Cartesian path incomplete: %.3f", fraction);
    return false;
  }
  robot_trajectory::RobotTrajectory trajectory(arm.getRobotModel(), arm.getName());
  trajectory.setRobotTrajectoryMsg(*arm.getCurrentState(), trajectory_message);
  trajectory_processing::TimeOptimalTrajectoryGeneration timing;
  const double acceleration = acceleration_scale > 0.0 ? acceleration_scale : speed_scale;
  if (!timing.computeTimeStamps(trajectory, speed_scale, acceleration)) {
    RCLCPP_ERROR(node.get_logger(), "Cartesian trajectory timing failed");
    return false;
  }
  trajectory.getRobotTrajectoryMsg(trajectory_message);
  return arm.execute(trajectory_message) == moveit::core::MoveItErrorCode::SUCCESS;
}

bool move_linear(
  ExecutorNode & node,
  moveit::planning_interface::MoveGroupInterface & arm,
  double z_delta,
  double speed_scale,
  double acceleration_scale = -1.0)
{
  geometry_msgs::msg::Pose target = arm.getCurrentPose().pose;
  target.position.z += z_delta;
  return move_linear_to_pose(node, arm, target, speed_scale, acceleration_scale);
}

bool move_to_observe(
  ExecutorNode & node,
  moveit::planning_interface::MoveGroupInterface & arm)
{
  const auto xyz = node.get_parameter("observe_pose_xyz").as_double_array();
  const auto joints = node.get_parameter("observe_joint_positions").as_double_array();
  if (xyz.size() != 3 || joints.size() != 7) {
    RCLCPP_ERROR(
      node.get_logger(),
      "observe_pose_xyz and observe_joint_positions must contain 3 and 7 values");
    return false;
  }
  RCLCPP_INFO(
    node.get_logger(),
    "MOVE_TO_OBSERVE fixed view above source=(%.3f, %.3f, %.3f)",
    xyz[0], xyz[1], xyz[2]);
  arm.clearPoseTargets();
  arm.clearPathConstraints();
  const int attempts = node.get_parameter("planning_attempts").as_int();
  for (int attempt = 1; attempt <= attempts; ++attempt) {
    arm.setStartStateToCurrentState();
    if (!arm.setJointValueTarget(joints)) {
      RCLCPP_ERROR(node.get_logger(), "Fixed observation joint target is invalid");
      return false;
    }
    if (arm.move() == moveit::core::MoveItErrorCode::SUCCESS) {
      std::this_thread::sleep_for(
        std::chrono::duration<double>(node.get_parameter("motion_settle_s").as_double()));
      return true;
    }
    RCLCPP_WARN(
      node.get_logger(), "observe planning attempt %d/%d failed", attempt, attempts);
  }
  return false;
}

void add_table(
  moveit::planning_interface::PlanningSceneInterface & scene,
  const std::string & frame)
{
  moveit_msgs::msg::CollisionObject table;
  table.id = "benchmark_table";
  table.header.frame_id = frame;
  shape_msgs::msg::SolidPrimitive box;
  box.type = shape_msgs::msg::SolidPrimitive::BOX;
  box.dimensions = {0.60, 1.00, 0.04};
  geometry_msgs::msg::Pose pose;
  pose.position.x = 0.40;
  pose.position.z = 0.02;
  pose.orientation.w = 1.0;
  table.primitives.push_back(box);
  table.primitive_poses.push_back(pose);
  table.operation = moveit_msgs::msg::CollisionObject::ADD;
  scene.applyCollisionObject(table);
}

void add_block(
  moveit::planning_interface::PlanningSceneInterface & scene,
  const std::string & frame,
  const std::string & id,
  const std::string & type,
  const geometry_msgs::msg::Pose & pose)
{
  moveit_msgs::msg::CollisionObject object;
  object.id = id;
  object.header.frame_id = frame;
  shape_msgs::msg::SolidPrimitive box;
  box.type = shape_msgs::msg::SolidPrimitive::BOX;
  const auto dimensions = dimensions_for(type);
  box.dimensions.assign(dimensions.begin(), dimensions.end());
  object.primitives.push_back(box);
  object.primitive_poses.push_back(pose);
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  scene.applyCollisionObject(object);
}

bool execute_task(
  ExecutorNode & node,
  moveit::planning_interface::MoveGroupInterface & arm,
  moveit::planning_interface::PlanningSceneInterface & scene,
  const Task & task)
{
  const BlockState source = node.block(task.id);
  const double approach = node.get_parameter("approach_height_m").as_double();
  const double gripper_offset = node.get_parameter("gripper_offset_m").as_double();
  const double grasp_descent = node.get_parameter("grasp_descent_m").as_double();
  const double place_descent = node.get_parameter("place_descent_m").as_double();
  const double place_press_depth = node.get_parameter("place_press_depth_m").as_double();
  const double cartesian_speed = node.get_parameter("cartesian_speed_scale").as_double();
  const double place_speed = node.get_parameter("place_speed_scale").as_double();
  const double place_acceleration =
    node.get_parameter("place_acceleration_scale").as_double();
  const double lift_speed = node.get_parameter("lift_speed_scale").as_double();
  const double rotation_speed =
    node.get_parameter("transport_rotation_speed_scale").as_double();
  const double transport_velocity =
    node.get_parameter("transport_velocity_scale").as_double();
  const double transport_acceleration =
    node.get_parameter("transport_acceleration_scale").as_double();
  const double large_part_scale = task.type == "brick_4x2" ?
    node.get_parameter("large_part_motion_scale").as_double() : 1.0;

  const auto grasp_offset = node.get_parameter("grasp_offset_xy").as_double_array();
  if (grasp_offset.size() != 2) {
    RCLCPP_ERROR(node.get_logger(), "grasp_offset_xy must contain two values");
    return false;
  }
  geometry_msgs::msg::Pose pregrasp = source.pose;
  pregrasp.position.x += grasp_offset[0];
  pregrasp.position.y += grasp_offset[1];
  pregrasp.position.z += gripper_offset + approach;
  RCLCPP_INFO(
    node.get_logger(), "[%s] GRASP_CENTER target=(%.4f, %.4f)m offset=(%.4f, %.4f)m",
    task.id.c_str(), source.pose.position.x, source.pose.position.y,
    grasp_offset[0], grasp_offset[1]);
  RCLCPP_INFO(node.get_logger(), "[%s] MOVE_TO_PREGRASP", task.id.c_str());
  if (!move_to_pose(node, arm, pregrasp, "pregrasp", true)) {
    return false;
  }
  if (!node.command_gripper(node.get_parameter("gripper_open_m").as_double())) {
    return false;
  }
  RCLCPP_INFO(node.get_logger(), "[%s] DESCEND_TO_GRASP", task.id.c_str());
  if (!move_linear(node, arm, -grasp_descent, cartesian_speed)) {
    return false;
  }
  RCLCPP_INFO(node.get_logger(), "[%s] CLOSE_GRIPPER", task.id.c_str());
  if (!node.command_gripper(node.get_parameter("gripper_closed_m").as_double())) {
    return false;
  }
  if (!node.wait_for_grasp(task)) {
    RCLCPP_ERROR(
      node.get_logger(), "[%s] dual-fingertip grasp was not confirmed", task.id.c_str());
    return false;
  }
  RCLCPP_INFO(node.get_logger(), "[%s] GRASP_CONFIRMED", task.id.c_str());

  add_block(scene, arm.getPlanningFrame(), task.id, task.type, source.pose);
  arm.attachObject(
    task.id, "panda_hand", {"panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_link8"});
  RCLCPP_INFO(node.get_logger(), "[%s] LIFT", task.id.c_str());
  if (!move_linear(node, arm, grasp_descent, lift_speed * large_part_scale)) {
    return false;
  }
  std::this_thread::sleep_for(
    std::chrono::duration<double>(node.get_parameter("motion_settle_s").as_double()));
  if (node.source_mode() == "oracle") {
    const BlockState lifted = node.block(task.id);
    if (lifted.pose.position.z - source.pose.position.z <
      node.get_parameter("verify_lift_m").as_double())
    {
      RCLCPP_ERROR(node.get_logger(), "[%s] lift verification failed", task.id.c_str());
      return false;
    }
  }

  geometry_msgs::msg::Pose aligned_for_transport = arm.getCurrentPose().pose;
  aligned_for_transport.orientation = task.target.orientation;
  RCLCPP_INFO(node.get_logger(), "[%s] ALIGN_TOOL_FOR_PLACE", task.id.c_str());
  if (!move_linear_to_pose(
      node, arm, aligned_for_transport, rotation_speed * large_part_scale))
  {
    return false;
  }

  geometry_msgs::msg::Pose preplace = task.target;
  const auto place_offset = node.get_parameter("place_offset_xy").as_double_array();
  if (place_offset.size() != 2) {
    RCLCPP_ERROR(node.get_logger(), "place_offset_xy must contain two values");
    return false;
  }
  preplace.position.x += place_offset[0];
  preplace.position.y += place_offset[1];
  preplace.position.z += gripper_offset + approach;
  RCLCPP_INFO(node.get_logger(), "[%s] MOVE_TO_PREPLACE", task.id.c_str());
  arm.setMaxVelocityScalingFactor(transport_velocity * large_part_scale);
  arm.setMaxAccelerationScalingFactor(transport_acceleration * large_part_scale);
  const bool reached_preplace = move_to_pose(node, arm, preplace, "preplace", true);
  arm.setMaxVelocityScalingFactor(node.get_parameter("velocity_scale").as_double());
  arm.setMaxAccelerationScalingFactor(node.get_parameter("acceleration_scale").as_double());
  if (!reached_preplace) {
    return false;
  }
  if (node.source_mode() == "oracle") {
    const BlockState carried = node.block(task.id);
    geometry_msgs::msg::Pose corrected = arm.getCurrentPose().pose;
    const double correction_x = task.target.position.x - carried.pose.position.x;
    const double correction_y = task.target.position.y - carried.pose.position.y;
    const double correction_z =
      task.target.position.z + place_descent - place_press_depth - carried.pose.position.z;
    corrected.position.x += correction_x;
    corrected.position.y += correction_y;
    corrected.position.z += correction_z;
    RCLCPP_INFO(
      node.get_logger(), "[%s] ALIGN_CARRIED_BLOCK delta=(%.4f, %.4f, %.4f)m",
      task.id.c_str(), correction_x, correction_y, correction_z);
    if (!move_linear_to_pose(node, arm, corrected, cartesian_speed)) {
      return false;
    }
  } else if (node.get_parameter("visual_place_correction").as_bool()) {
    const auto excluded_colors =
      node.get_parameter("visual_place_correction_excluded_colors").as_string_array();
    if (std::find(excluded_colors.begin(), excluded_colors.end(), task.color) !=
      excluded_colors.end())
    {
      RCLCPP_WARN(
        node.get_logger(),
        "[%s] VISUAL_PLACE_CORRECTION_SKIPPED color=%s is excluded",
        task.id.c_str(), task.color.c_str());
    } else {
      const int iterations = std::max(
        1, static_cast<int>(node.get_parameter("visual_place_correction_iterations").as_int()));
      for (int iteration = 0; iteration < iterations; ++iteration) {
        BlockState carried;
        if (!node.detect_carried(task, carried)) {
          RCLCPP_WARN(
            node.get_logger(),
            "[%s] VISUAL_PLACE_CORRECTION_SKIPPED iteration=%d/%d; "
            "continuing from the last observable pose",
            task.id.c_str(), iteration + 1, iterations);
          break;
        }
        geometry_msgs::msg::Pose corrected = arm.getCurrentPose().pose;
        const double correction_x = task.target.position.x - carried.pose.position.x;
        const double correction_y = task.target.position.y - carried.pose.position.y;
        corrected.position.x += correction_x;
        corrected.position.y += correction_y;
        RCLCPP_INFO(
          node.get_logger(),
          "[%s] VISUAL_ALIGN_CARRIED_BLOCK iteration=%d/%d delta=(%.4f, %.4f, 0.0000)m",
          task.id.c_str(), iteration + 1, iterations, correction_x, correction_y);
        if (!move_linear_to_pose(node, arm, corrected, cartesian_speed)) {
          return false;
        }
      }
    }
  }
  RCLCPP_INFO(
    node.get_logger(), "[%s] DESCEND_TO_TARGET distance=%.4fm",
    task.id.c_str(), place_descent);
  if (!move_linear(
      node, arm, -place_descent, place_speed * large_part_scale,
      place_acceleration * large_part_scale))
  {
    return false;
  }
  arm.detachObject(task.id);
  scene.removeCollisionObjects({task.id});
  RCLCPP_INFO(node.get_logger(), "[%s] OPEN_GRIPPER", task.id.c_str());
  if (!node.command_gripper(node.get_parameter("gripper_open_m").as_double())) {
    return false;
  }
  std::this_thread::sleep_for(
    std::chrono::duration<double>(node.get_parameter("release_settle_s").as_double()));
  add_block(scene, arm.getPlanningFrame(), task.id, task.type, task.target);
  RCLCPP_INFO(node.get_logger(), "[%s] RETREAT", task.id.c_str());
  return move_linear(node, arm, approach, lift_speed * large_part_scale);
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ExecutorNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() {executor.spin();});
  int return_code = 1;

  try {
    if (!node->wait_for_benchmark()) {
      throw std::runtime_error("timed out waiting for benchmark input");
    }
    if (!node->call_reset()) {
      throw std::runtime_error("benchmark reset failed");
    }

    moveit::planning_interface::MoveGroupInterface arm(node, "panda_arm");
    moveit::planning_interface::PlanningSceneInterface scene;
    arm.setPlanningTime(node->get_parameter("planning_time_s").as_double());
    arm.setNumPlanningAttempts(node->get_parameter("planning_attempts").as_int());
    arm.setMaxVelocityScalingFactor(node->get_parameter("velocity_scale").as_double());
    arm.setMaxAccelerationScalingFactor(node->get_parameter("acceleration_scale").as_double());
    add_table(scene, arm.getPlanningFrame());
    std::this_thread::sleep_for(
      std::chrono::duration<double>(node->get_parameter("motion_settle_s").as_double()));

    const auto tasks = node->tasks();
    RCLCPP_INFO(
      node->get_logger(), "Loaded %zu stable-ID assembly tasks (source_mode=%s)",
      tasks.size(), node->source_mode().c_str());
    bool success = node->command_gripper(node->get_parameter("gripper_open_m").as_double());
    for (std::size_t index = 0; success && index < tasks.size(); ++index) {
      RCLCPP_INFO(
        node->get_logger(), "Task %zu/%zu: %s", index + 1, tasks.size(), tasks[index].id.c_str());
      if (node->source_mode() == "vision") {
        success = move_to_observe(*node, arm) && node->detect_source(tasks[index]);
        if (!success) {
          break;
        }
      }
      success = execute_task(*node, arm, scene, tasks[index]);
    }
    const std::string result = node->export_result();
    RCLCPP_INFO(node->get_logger(), "Benchmark result: %s", result.c_str());
    return_code = success ? 0 : 2;
  } catch (const std::exception & error) {
    RCLCPP_ERROR(node->get_logger(), "Executor failed: %s", error.what());
    RCLCPP_INFO(node->get_logger(), "Benchmark result: %s", node->export_result().c_str());
    return_code = 2;
  }

  executor.cancel();
  if (spin_thread.joinable()) {
    spin_thread.join();
  }
  rclcpp::shutdown();
  return return_code;
}
