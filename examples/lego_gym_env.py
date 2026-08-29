#!/usr/bin/env python3
"""Minimal direct-Python policy loop for LEGO Bench."""
from pathlib import Path

from mj_bridge.gym_env import LegoBenchEnv


def main() -> None:
    product = Path(__file__).resolve().parent / "products" / "traffic_light.yaml"
    with LegoBenchEnv(
        product,
        seed=42,
        output_dir="runs/python-example",
        observation_mode="oracle",
    ) as environment:
        observation, info = environment.reset()
        print(f"episode: {observation['episode_id']}")
        print(f"actuators: {environment.actuator_names}")
        print(f"action shape: {environment.action_shape}")

        # Replace this hold command with the output of a controller or policy.
        observation, reward, terminated, truncated, info = environment.step(
            environment.neutral_action
        )
        print(
            f"reward={reward:.3f}, terminated={terminated}, "
            f"truncated={truncated}"
        )
        result = environment.export_result()
        print(f"result: {result}")


if __name__ == "__main__":
    main()
