# Executor integration

The simulator and the execution method are separate ROS 2 processes. Start the
benchmark with `executor:=none`, then launch a policy in the same ROS domain.
The policy does not need to import benchmark internals.

## Required lifecycle

1. Subscribe to `/lego_bench/goal` for stable IDs and target poses.
2. Obtain source poses from `/lego_bench/ground_truth` in oracle mode, or infer
   them from the RGB-D topics in perception mode.
3. Optionally call `/mj_bridge/reset` before execution.
4. Command the Panda through MoveIt or the two trajectory actions.
5. Call `/mj_bridge/result` after the final placement.

The `/goal` and `/ground_truth` messages are JSON strings. The goal includes
the complete normalized product. Ground truth remains available to the
benchmark for scoring and debugging, but an RGB-D method must not consume it;
use the camera topics and TF instead.

## Using a custom execution plan

`lego-bench generate` writes `execution_plan.yaml` next to the scene and
manifest. The supplied plan contains ordered `steps`, each with `id`, `type`,
`color`, `grasp_spin_deg`, and `depends_on`. A custom method may ignore this
file, modify it, or replace the planner while retaining block IDs from the
product YAML.

Run the reference executor against an already running episode with:

```bash
ros2 launch lego_executor oracle_executor.launch.py \
  plan_file:=$PWD/runs/my-episode/execution_plan.yaml
```

For a minimal Python subscriber and service client, see
`examples/external_executor.py`.

The adapter exposes four helpers intended to be called from its asynchronous
`policy()` method:

```python
await self.reset_episode()
await self.send_arm(joint_positions, duration_s=2.0)
await self.send_gripper(0.014, duration_s=0.4)
result = await self.export_result()
```

Arm commands contain seven positions in Panda joint order. Gripper commands use
one finger position for both coupled fingers: `0.04` is open and the reference
physics grasp uses `0.014`. A method may send multi-point trajectories directly
through the action endpoint when velocity or timing profiles are part of the
experiment.

The result service always writes `actual_state.json` and `result.json`. Its ROS
success flag is the assembly success, so a completed but unsuccessful trial is
still a valid benchmark result and must be retained.

## Evaluation contract

A report should record the product YAML, seed, observation mode, connection
mode, tolerance configuration, repository commit, and simulator version.
Oracle and RGB-D results, or snap and physics results, are different
experimental conditions and should not be pooled.

See [Evaluation protocol](evaluation-protocol.md) for seed handling, required
artifacts, and the minimum fields to report.
