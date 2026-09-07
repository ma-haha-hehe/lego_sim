import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

from mj_bridge.benchmark_cli import generate


def launch_setup(context):
    product = LaunchConfiguration("product").perform(context)
    seed = int(LaunchConfiguration("seed").perform(context))
    output_dir = LaunchConfiguration("output_dir").perform(context)
    headless = LaunchConfiguration("headless")
    observation = LaunchConfiguration("observation").perform(context)
    connection_mode = LaunchConfiguration("connection_mode").perform(context)
    executor_mode = LaunchConfiguration("executor").perform(context)
    vision_backend = LaunchConfiguration("vision_backend").perform(context)
    foundationpose_root = LaunchConfiguration("foundationpose_root").perform(context)
    _, scene = generate(product, seed, output_dir)

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={"ros2_control_hardware_type": "mock_components"},
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    controllers = {
        "moveit_controller_manager": (
            "moveit_simple_controller_manager/MoveItSimpleControllerManager"
        ),
        "moveit_simple_controller_manager": {
            "controller_names": ["mj_panda_arm_controller", "mj_panda_hand_controller"],
            "mj_panda_arm_controller": {
                "type": "FollowJointTrajectory", "action_ns": "follow_joint_trajectory",
                "default": True,
                "joints": [f"panda_joint{i}" for i in range(1, 8)],
            },
            "mj_panda_hand_controller": {
                "type": "FollowJointTrajectory", "action_ns": "follow_joint_trajectory",
                "default": True,
                "joints": ["panda_finger_joint1", "panda_finger_joint2"],
            },
        },
    }
    common_env = {
        "MJ_BRIDGE_MODEL": str(scene),
        "LEGO_BENCH_MANIFEST": str(Path(output_dir).resolve() / "episode_manifest.yaml"),
        "LEGO_BENCH_RUN_DIR": str(Path(output_dir).resolve()),
        "LEGO_BENCH_OBSERVATION": observation,
        "LEGO_BENCH_CONNECTION_MODE": connection_mode,
        "MJ_BRIDGE_HEADLESS": LaunchConfiguration("headless").perform(context),
    }
    if observation == "rgbd" and common_env["MJ_BRIDGE_HEADLESS"].lower() == "true":
        common_env["MUJOCO_GL"] = "egl"
    if observation == "rgbd" and executor_mode == "vision":
        common_env["LEGO_BENCH_CAMERA_ON_DEMAND"] = "true"

    move_group = Node(
        package="moveit_ros_move_group", executable="move_group", output="screen",
        parameters=[moveit_config.to_dict(), controllers, {
            "moveit_manage_controllers": False,
            "trajectory_execution.allowed_execution_duration_scaling": 12.0,
            "trajectory_execution.allowed_goal_duration_margin": 15.0,
            "trajectory_execution.allowed_start_tolerance": 0.05,
        }],
    )
    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[moveit_config.robot_description],
    )
    static_tf = Node(
        package="tf2_ros", executable="static_transform_publisher", output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "world", "panda_link0"],
    )
    camera_tf = Node(
        package="tf2_ros", executable="static_transform_publisher", output="screen",
        arguments=[
            "--x", "0.49", "--y", "-0.15", "--z", "1.2",
            "--qx", "1", "--qy", "0", "--qz", "0", "--qw", "0",
            "--frame-id", "world", "--child-frame-id", "realsense",
        ],
    )
    placement_camera_tf = Node(
        package="tf2_ros", executable="static_transform_publisher", output="screen",
        arguments=[
            "--x", "0.85", "--y", "0.35", "--z", "0.55",
            "--qx", "0.63143547", "--qy", "0.63143547",
            "--qz", "-0.31825972", "--qw", "-0.31825972",
            "--frame-id", "world", "--child-frame-id", "placement_camera",
        ],
    )
    bridge = Node(
        package="mj_bridge", executable="mj_bridge", output="screen", additional_env=common_env,
    )
    rviz_config = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"),
        "launch",
        "moveit.rviz",
    )
    rviz = Node(
        package="rviz2", executable="rviz2", output="log", arguments=["-d", rviz_config],
        parameters=[moveit_config.to_dict()], condition=UnlessCondition(headless),
    )
    nodes = [static_tf, camera_tf, placement_camera_tf, rsp, move_group, bridge, rviz]
    if executor_mode == "oracle":
        executor_config = os.path.join(
            get_package_share_directory("lego_executor"), "config", "executor.yaml"
        )
        nodes.append(Node(
            package="lego_executor", executable="oracle_moveit_executor", output="screen",
            parameters=[moveit_config.to_dict(), executor_config, {
                "plan_file": str(Path(output_dir).resolve() / "execution_plan.yaml"),
            }],
        ))
    elif executor_mode == "vision":
        if observation != "rgbd":
            raise RuntimeError("executor:=vision requires observation:=rgbd")
        executor_config = os.path.join(
            get_package_share_directory("lego_executor"), "config", "executor.yaml"
        )
        nodes.append(Node(
            package="mj_bridge", executable="lego-vision", name="lego_vision", output="screen",
            parameters=[{
                "backend": vision_backend,
                "foundationpose_root": foundationpose_root,
                "debug_dir": str(Path(output_dir).resolve() / "vision"),
            }],
        ))
        nodes.append(Node(
            package="lego_executor", executable="visual_moveit_executor",
            name="visual_moveit_executor", output="screen",
            parameters=[moveit_config.to_dict(), executor_config, {
                "plan_file": str(Path(output_dir).resolve() / "execution_plan.yaml"),
            }],
        ))
    return nodes


def generate_launch_description():
    share = get_package_share_directory("mj_bridge")
    return LaunchDescription([
        DeclareLaunchArgument(
            "product",
            default_value=os.path.join(share, "examples", "traffic_light.yaml"),
        ),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("output_dir", default_value="/tmp/lego_bench/latest"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("observation", default_value="oracle"),
        DeclareLaunchArgument("connection_mode", default_value="physics"),
        DeclareLaunchArgument("executor", default_value="none"),
        DeclareLaunchArgument("vision_backend", default_value="foundationpose"),
        DeclareLaunchArgument(
            "foundationpose_root", default_value=os.environ.get("FOUNDATIONPOSE_DIR", "")
        ),
        OpaqueFunction(function=launch_setup),
    ])
