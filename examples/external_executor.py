#!/usr/bin/env python3
"""ROS 2 adapter skeleton for a third-party LEGO assembly policy."""
import json

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint


ARM_JOINTS = [f"panda_joint{index}" for index in range(1, 8)]
HAND_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


class ExternalExecutor(Node):
    def __init__(self):
        super().__init__("external_lego_executor")
        self.goal = None
        self.state = None
        self.started = False
        self.create_subscription(String, "/lego_bench/goal", self.on_goal, 10)
        self.create_subscription(String, "/lego_bench/ground_truth", self.on_state, 10)
        self.reset_client = self.create_client(Trigger, "/mj_bridge/reset")
        self.result_client = self.create_client(Trigger, "/mj_bridge/result")
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/mj_panda_arm_controller/follow_joint_trajectory",
        )
        self.hand_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/mj_panda_hand_controller/follow_joint_trajectory",
        )
        self.timer = self.create_timer(0.2, self.tick)

    def on_goal(self, msg):
        self.goal = json.loads(msg.data)

    def on_state(self, msg):
        self.state = json.loads(msg.data)

    async def call_trigger(self, client, name, require_success=True):
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"{name} service is unavailable")
        response = await client.call_async(Trigger.Request())
        if require_success and not response.success:
            raise RuntimeError(f"{name} failed: {response.message}")
        return response

    async def reset_episode(self):
        response = await self.call_trigger(self.reset_client, "reset")
        return json.loads(response.message)

    async def export_result(self):
        response = await self.call_trigger(
            self.result_client, "result", require_success=False
        )
        return json.loads(response.message)

    async def send_trajectory(self, client, joint_names, positions, duration_s):
        if len(joint_names) != len(positions):
            raise ValueError("joint_names and positions must have equal length")
        if not client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("trajectory action server is unavailable")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        nanoseconds = int(float(duration_s) * 1_000_000_000)
        point.time_from_start = Duration(
            sec=nanoseconds // 1_000_000_000,
            nanosec=nanoseconds % 1_000_000_000,
        )
        goal.trajectory.points = [point]
        handle = await client.send_goal_async(goal)
        if not handle.accepted:
            raise RuntimeError("trajectory goal was rejected")
        return await handle.get_result_async()

    async def send_arm(self, positions, duration_s=2.0):
        return await self.send_trajectory(
            self.arm_client, ARM_JOINTS, positions, duration_s
        )

    async def send_gripper(self, finger_position, duration_s=0.4):
        return await self.send_trajectory(
            self.hand_client,
            HAND_JOINTS,
            [finger_position, finger_position],
            duration_s,
        )

    async def policy(self, observation, goal):
        """Replace this method with perception, planning, and execution code."""
        block_count = len(goal.get("target_blocks", []))
        self.get_logger().info(
            f"Episode {goal.get('episode_id')} is ready with {block_count} parts"
        )
        self.get_logger().info(
            "Implement policy(); helpers for reset, arm, gripper, and result are ready"
        )

    async def tick(self):
        if self.started or self.goal is None or self.state is None:
            return
        self.started = True
        try:
            await self.policy(self.state, self.goal)
        except Exception as error:  # Keep the node alive for inspection and reset.
            self.get_logger().error(f"Policy failed: {error}")


def main():
    rclpy.init()
    node = ExternalExecutor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
