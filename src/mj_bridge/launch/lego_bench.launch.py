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
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
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

    move_group = Node(
        package="moveit_ros_move_group", executable="move_group", output="screen",
        parameters=[moveit_config.to_dict(), controllers, {
            "moveit_manage_controllers": False,
            "trajectory_execution.allowed_execution_duration_scaling": 3.0,
            "trajectory_execution.allowed_goal_duration_margin": 5.0,
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
            "--x", "0.4", "--y", "0", "--z", "1.2",
            "--qx", "1", "--qy", "0", "--qz", "0", "--qw", "0",
            "--frame-id", "world", "--child-frame-id", "realsense",
        ],
    )
    bridge = Node(
        package="mj_bridge", executable="mj_bridge", output="screen", additional_env=common_env,
    )
    rviz_config = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"), "launch", "moveit.rviz"
    )
    rviz = Node(
        package="rviz2", executable="rviz2", output="log", arguments=["-d", rviz_config],
        parameters=[moveit_config.to_dict()], condition=UnlessCondition(headless),
    )
    return [static_tf, camera_tf, rsp, move_group, bridge, rviz]


def generate_launch_description():
    share = get_package_share_directory("mj_bridge")
    return LaunchDescription([
        DeclareLaunchArgument("product", default_value=os.path.join(share, "examples", "traffic_light.yaml")),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("output_dir", default_value="/tmp/lego_bench/latest"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("observation", default_value="oracle"),
        DeclareLaunchArgument("connection_mode", default_value="snap"),
        OpaqueFunction(function=launch_setup),
    ])
