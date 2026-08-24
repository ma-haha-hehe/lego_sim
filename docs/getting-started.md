# Getting started

## Benchmark responsibilities

The benchmark owns product validation, episode generation, MuJoCo physics, Panda control endpoints, observations, reset semantics, and scoring. The included executor is a reference implementation; external methods may replace its detector, grasp generator, task planner, motion planner, or complete policy.

Every run follows the same sequence:

1. Normalize and validate the product file.
2. Sample non-overlapping source poses from the episode seed.
3. Generate a self-contained MuJoCo scene and episode manifest.
4. Start the simulator and ROS 2 interfaces.
5. Allow an external method to observe and control the robot.
6. Score the complete assembly and export the final state.

## Environment setup

```bash
bash install_sim_system_deps.sh
source enter_sim_env.sh
colcon build --packages-select mj_bridge lego_executor --symlink-install
source enter_sim_env.sh
```

## Running an episode

Run the supplied reference executor:

```bash
./run_reference_pipeline.sh --headless
```

The script accepts `--product`, `--seed`, `--output-dir`, `--observation`, and
`--connection-mode`. It expects the workspace to have been built first.

Start an episode for an external method:

```bash
ros2 launch mj_bridge lego_bench.launch.py \
  product:=$PWD/examples/products/traffic_light.yaml \
  seed:=0 \
  output_dir:=$PWD/runs/traffic-light-0 \
  headless:=true \
  observation:=oracle \
  connection_mode:=snap \
  executor:=none
```

The launch starts `robot_state_publisher`, the static world and camera transforms, MoveIt `move_group`, and the MuJoCo bridge.

## Episode outputs

The output directory contains:

- `product.normalized.yaml`: canonical schema-v1 product
- `episode_manifest.yaml`: seed, source poses, target poses, and body mapping
- `execution_plan.yaml`: stable-ID, dependency-aware reference assembly order
- `scene.xml`: generated MuJoCo scene
- `actual_state.json`: authoritative state at result time
- `result.json`: aggregate and per-part metrics

Call `/mj_bridge/result` before shutting down if the final JSON artifacts are required. `/mj_bridge/reset` restores joint positions, free bodies, velocities, gripper state, simplified connections, timer, and metric counters.

## Choosing an evaluation mode

Use `oracle` to isolate planning and control performance. Use `rgbd` when the method should perform its own perception. Use `snap` for repeatable assembly engagement or `physics` to evaluate without automatic connections. Record these choices with every result.
