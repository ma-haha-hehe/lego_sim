"""Start the oracle reference executor against an existing benchmark episode."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context):
    plan_file = LaunchConfiguration("plan_file").perform(context)
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
    executor_config = os.path.join(
        get_package_share_directory("lego_executor"), "config", "executor.yaml"
    )
    return [Node(
        package="lego_executor",
        executable="oracle_moveit_executor",
        name="oracle_moveit_executor",
        output="screen",
        parameters=[moveit_config.to_dict(), executor_config, {"plan_file": plan_file}],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("plan_file", default_value=""),
        OpaqueFunction(function=launch_setup),
    ])
