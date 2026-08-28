#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
using namespace std::chrono_literals;

namespace
{

struct BlockState
{
  std::string id;
  std::string type;
  geometry_msgs::msg::Pose pose;
};

struct Task
{
  std::string id;
  std::string type;
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
    declare_parameter("velocity_scale", 0.35);
    declare_parameter("acceleration_scale", 0.35);
    declare_parameter("approach_height_m", 0.15);
    declare_parameter("gripper_offset_m", 0.1234);
    declare_parameter("grasp_descent_m", 0.165);
    declare_parameter("place_descent_m", 0.15);
    declare_parameter("cartesian_speed_scale", 0.12);
    declare_parameter("lift_speed_scale", 0.35);
    declare_parameter("gripper_open_m", 0.04);
    declare_parameter("gripper_closed_m", 0.014);
    declare_parameter("grasp_confirmation_timeout_s", 2.0);
    declare_parameter("tool_yaw_offset_deg", 45.0);
    declare_parameter("verify_lift_m", 0.025);

    goal_subscription_ = create_subscription<std_msgs::msg::String>(
      "/lego_bench/goal", 10,
      [this](const std_msgs::msg::String::SharedPtr message) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        goal_ = YAML::Load(message->data);
        data_cv_.notify_all();
      });
    state_subscription_ = create_subscription<std_msgs::msg::String>(
      "/lego_bench/ground_truth", 10,
      [this](const std_msgs::msg::String::SharedPtr message) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        state_ = YAML::Load(message->data);
        data_cv_.notify_all();
      });
    reset_client_ = create_client<std_srvs::srv::Trigger>("/mj_bridge/reset");
    result_client_ = create_client<std_srvs::srv::Trigger>("/mj_bridge/result");
    gripper_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
      this, "/mj_panda_hand_controller/follow_joint_trajectory");
  }

  bool wait_for_benchmark()
  {
    const auto timeout = std::chrono::duration<double>(get_parameter("wait_timeout_s").as_double());
    std::unique_lock<std::mutex> lock(data_mutex_);
    return data_cv_.wait_for(lock, timeout, [this]() {return goal_ && state_;});
  }

  bool call_reset()
  {
    if (!get_parameter("auto_reset").as_bool()) {
      return true;
    }
    if (!reset_client_->wait_for_service(10s)) {
      RCLCPP_ERROR(get_logger(), "Benchmark reset service is unavailable");
      return false;
    }
    auto future = reset_client_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
    if (future.wait_for(10s) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Benchmark reset timed out");
      return false;
    }
    RCLCPP_INFO(get_logger(), "Benchmark episode reset");
    std::this_thread::sleep_for(500ms);
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
    const YAML::Node value = state_["blocks"][id];
    if (!value) {
      throw std::runtime_error("ground truth does not contain block ID: " + id);
    }
    BlockState block;
    block.id = id;
    block.type = value["type"].as<std::string>();
    block.pose.position.x = value["position"][0].as<double>();
    block.pose.position.y = value["position"][1].as<double>();
    block.pose.position.z = value["position"][2].as<double>();
    const double offset = get_parameter("tool_yaw_offset_deg").as_double() * M_PI / 180.0;
    block.pose.orientation = down_orientation(yaw_from_wxyz(value["quaternion_wxyz"]) + offset);
    return block;
  }

  bool wait_for_grasp(const std::string & id)
  {
    const auto timeout = std::chrono::duration<double>(
      get_parameter("grasp_confirmation_timeout_s").as_double());
    std::unique_lock<std::mutex> lock(data_mutex_);
    return data_cv_.wait_for(lock, timeout, [this, &id]() {
      const YAML::Node grasped = state_["grasped_block"];
      return grasped && grasped.as<std::string>("") == id;
    });
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
    point.time_from_start = rclcpp::Duration::from_seconds(1.0);
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
    auto future = result_client_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
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
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr goal_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_subscription_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr result_client_;
  rclcpp_action::Client<FollowJointTrajectory>::SharedPtr gripper_client_;
};

bool move_to_pose(
  ExecutorNode & node,
  moveit::planning_interface::MoveGroupInterface & arm,
  const geometry_msgs::msg::Pose & pose,
  const std::string & label)
{
  const int attempts = node.get_parameter("planning_attempts").as_int();
  for (int attempt = 1; attempt <= attempts; ++attempt) {
    arm.setStartStateToCurrentState();
    arm.setPoseTarget(pose);
    if (arm.move() == moveit::core::MoveItErrorCode::SUCCESS) {
      arm.clearPoseTargets();
      // Let the bridge publish the converged joint state before deriving the
      // next Cartesian trajectory. This avoids a stale start state after a
      // long point-to-point motion.
      std::this_thread::sleep_for(500ms);
      return true;
    }
    RCLCPP_WARN(
      node.get_logger(), "%s planning attempt %d/%d failed", label.c_str(), attempt, attempts);
  }
  arm.clearPoseTargets();
  return false;
}

bool move_linear_to_pose(
  ExecutorNode & node,
  moveit::planning_interface::MoveGroupInterface & arm,
  const geometry_msgs::msg::Pose & target,
  double speed_scale)
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
  if (!timing.computeTimeStamps(trajectory, speed_scale, speed_scale)) {
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
  double speed_scale)
{
  geometry_msgs::msg::Pose target = arm.getCurrentPose().pose;
  target.position.z += z_delta;
  return move_linear_to_pose(node, arm, target, speed_scale);
}

void add_table(moveit::planning_interface::PlanningSceneInterface & scene, const std::string & frame)
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
  const double cartesian_speed = node.get_parameter("cartesian_speed_scale").as_double();
  const double lift_speed = node.get_parameter("lift_speed_scale").as_double();

  geometry_msgs::msg::Pose pregrasp = source.pose;
  pregrasp.position.z += gripper_offset + approach;
  RCLCPP_INFO(node.get_logger(), "[%s] MOVE_TO_PREGRASP", task.id.c_str());
  if (!move_to_pose(node, arm, pregrasp, "pregrasp")) {
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
  if (!node.wait_for_grasp(task.id)) {
    RCLCPP_ERROR(
      node.get_logger(), "[%s] dual-fingertip grasp was not confirmed", task.id.c_str());
    return false;
  }
  RCLCPP_INFO(node.get_logger(), "[%s] GRASP_CONFIRMED", task.id.c_str());

  add_block(scene, arm.getPlanningFrame(), task.id, task.type, source.pose);
  arm.attachObject(
    task.id, "panda_hand", {"panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_link8"});
  RCLCPP_INFO(node.get_logger(), "[%s] LIFT", task.id.c_str());
  if (!move_linear(node, arm, grasp_descent, lift_speed)) {
    return false;
  }
  std::this_thread::sleep_for(500ms);
  const BlockState lifted = node.block(task.id);
  if (lifted.pose.position.z - source.pose.position.z < node.get_parameter("verify_lift_m").as_double()) {
    RCLCPP_ERROR(node.get_logger(), "[%s] lift verification failed", task.id.c_str());
    return false;
  }

  geometry_msgs::msg::Pose preplace = task.target;
  preplace.position.z += gripper_offset + approach;
  RCLCPP_INFO(node.get_logger(), "[%s] MOVE_TO_PREPLACE", task.id.c_str());
  if (!move_to_pose(node, arm, preplace, "preplace")) {
    return false;
  }
  const BlockState carried = node.block(task.id);
  geometry_msgs::msg::Pose corrected = arm.getCurrentPose().pose;
  const double correction_x = task.target.position.x - carried.pose.position.x;
  const double correction_y = task.target.position.y - carried.pose.position.y;
  const double correction_z =
    task.target.position.z + place_descent - carried.pose.position.z;
  corrected.position.x += correction_x;
  corrected.position.y += correction_y;
  corrected.position.z += correction_z;
  RCLCPP_INFO(
    node.get_logger(), "[%s] ALIGN_CARRIED_BLOCK delta=(%.4f, %.4f, %.4f)m",
    task.id.c_str(), correction_x, correction_y, correction_z);
  if (!move_linear_to_pose(node, arm, corrected, cartesian_speed)) {
    return false;
  }
  RCLCPP_INFO(node.get_logger(), "[%s] DESCEND_TO_TARGET", task.id.c_str());
  if (!move_linear(node, arm, -place_descent, cartesian_speed)) {
    return false;
  }
  arm.detachObject(task.id);
  scene.removeCollisionObjects({task.id});
  RCLCPP_INFO(node.get_logger(), "[%s] OPEN_GRIPPER", task.id.c_str());
  if (!node.command_gripper(node.get_parameter("gripper_open_m").as_double())) {
    return false;
  }
  std::this_thread::sleep_for(500ms);
  add_block(scene, arm.getPlanningFrame(), task.id, task.type, task.target);
  RCLCPP_INFO(node.get_logger(), "[%s] RETREAT", task.id.c_str());
  return move_linear(node, arm, approach, lift_speed);
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
      throw std::runtime_error("timed out waiting for benchmark goal and ground truth");
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
    std::this_thread::sleep_for(500ms);

    const auto tasks = node->tasks();
    RCLCPP_INFO(node->get_logger(), "Loaded %zu stable-ID assembly tasks", tasks.size());
    bool success = node->command_gripper(node->get_parameter("gripper_open_m").as_double());
    for (std::size_t index = 0; success && index < tasks.size(); ++index) {
      RCLCPP_INFO(
        node->get_logger(), "Task %zu/%zu: %s", index + 1, tasks.size(), tasks[index].id.c_str());
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
