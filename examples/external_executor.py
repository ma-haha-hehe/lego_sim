#!/usr/bin/env python3
"""Minimal third-party policy adapter. Replace `policy()` with your method."""
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ExternalExecutor(Node):
    def __init__(self):
        super().__init__("external_lego_executor")
        self.goal = None
        self.state = None
        self.create_subscription(String, "/lego_bench/goal", self.on_goal, 10)
        self.create_subscription(String, "/lego_bench/ground_truth", self.on_state, 10)
        self.timer = self.create_timer(0.2, self.tick)

    def on_goal(self, msg):
        self.goal = json.loads(msg.data)

    def on_state(self, msg):
        self.state = json.loads(msg.data)

    def policy(self, observation, goal):
        """Return/send actions here using MoveIt or the FollowJointTrajectory actions."""
        # Action endpoints:
        # /mj_panda_arm_controller/follow_joint_trajectory
        # /mj_panda_hand_controller/follow_joint_trajectory
        return None

    def tick(self):
        if self.goal is not None and self.state is not None:
            self.policy(self.state, self.goal)


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
