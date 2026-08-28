# LEGO Assembly Benchmark for Panda and MuJoCo

A reproducible ROS 2 benchmark for evaluating manipulation, perception, and assembly methods with a Franka Emika Panda robot in MuJoCo.

The benchmark takes a product description in YAML, creates the required loose parts at deterministic random poses, exposes standard ROS 2 control and observation interfaces, and evaluates the final assembly. The environment is independent of the policy: a method may use MoveIt, direct joint trajectories, ground-truth state, RGB-D input, or its own perception and planning stack.

## Features

- Deterministic episode generation from a product YAML and integer seed
- Non-overlapping random placement inside a fixed table workspace
- Panda arm and gripper control through `FollowJointTrajectory`
- MoveIt 2 planning and trajectory execution
- Ground-truth and RGB-D observation modes
- RGB, metric depth, instance segmentation, camera calibration, and TF
- Explicit reset and result services for automated evaluation
- Dependency-aware assembly planning with stable part IDs
- Reference Panda/MoveIt pick-and-place state machine
- Per-part pose errors, completion rate, runtime, timeout, collision, and stability metrics
- Headless execution, batch generation, Docker configuration, and CI
- Compatibility with product files used by the original project

## Supported platform

The reference setup targets Ubuntu 22.04, ROS 2 Humble, Python 3.10, MuJoCo 3.x, and MoveIt 2.

## Installation

```bash
git clone https://github.com/ma-haha-hehe/lego_sim.git
cd lego_sim
bash install_sim_system_deps.sh
source enter_sim_env.sh
colcon build --packages-select mj_bridge lego_executor --symlink-install
source enter_sim_env.sh
```

`enter_sim_env.sh` activates ROS 2, the local Python virtual environment, and the built benchmark package. It does not start a simulation.

## Quick start

Validate a product and generate a standalone episode:

```bash
lego-bench validate examples/products/traffic_light.yaml
lego-bench generate \
  --product examples/products/traffic_light.yaml \
  --seed 42 \
  --output-dir runs/traffic-light-42
```

Run the complete reference pipeline (episode generation, MuJoCo, MoveIt,
planning, execution, and result export):

```bash
./run_reference_pipeline.sh --headless
```

Remove `--headless` to open MuJoCo and RViz. The traffic-light example takes
about one minute on a typical workstation. The launch remains open after the
executor finishes so the final scene can be inspected; press `Ctrl+C` to stop it.
Results are written under `runs/reference-seed-42/`.

To start the environment without the reference executor:

```bash
ros2 launch mj_bridge lego_bench.launch.py \
  product:=$PWD/examples/products/traffic_light.yaml \
  seed:=42 \
  headless:=false \
  observation:=oracle \
  connection_mode:=snap \
  executor:=none
```

This mode is intended for an external policy. For a headless RGB-D episode,
set `headless:=true` and `observation:=rgbd`.

## Product format

Products use schema version 1:

```yaml
schema_version: 1
product:
  name: traffic_light
blocks:
  - id: green_base
    type: brick_2x2
    color: green
    target:
      position: [0.0, 0.0, 0.0]
      yaw_deg: 0
```

Target positions are expressed in metres relative to the center of the assembly plate. Yaw is expressed in degrees. Part IDs must be unique.

Version 0.2 provides `brick_2x2` and `brick_4x2`. A product may contain any number and arrangement of registered parts that fit in the configured source and assembly workspaces. See [Product format](docs/product-format.md) for the schema, coordinate conventions, and legacy conversion rules.

## ROS 2 interface

| Interface | Type | Purpose |
|---|---|---|
| `/mj_panda_arm_controller/follow_joint_trajectory` | Action | Panda arm control |
| `/mj_panda_hand_controller/follow_joint_trajectory` | Action | Gripper control |
| `/joint_states` | Topic | Robot state |
| `/lego_bench/goal` | Topic | Product and target poses as JSON |
| `/lego_bench/ground_truth` | Topic | Authoritative part poses as JSON |
| `/camera/color/image_raw` | Topic | `rgb8` image |
| `/camera/depth/image_raw` | Topic | `32FC1` metric depth |
| `/camera/segmentation` | Topic | MuJoCo `32SC2` object ID/type image |
| `/camera/camera_info` | Topic | Pinhole camera calibration |
| `/mj_bridge/benchmark_state` | Topic | Live score as JSON |
| `/mj_bridge/reset` | Service | Restore the initial episode state |
| `/mj_bridge/result` | Service | Score and export the current state |

The full launch also provides MoveIt actions and services, including `/move_action`, `/execute_trajectory`, and `/plan_kinematic_path`. The camera frame is `realsense`; its static transform from `world` is published by the launch file.

An integration skeleton is available in [examples/external_executor.py](examples/external_executor.py).

## Reference executor

Episode generation writes `execution_plan.yaml`. The planner infers direct
support relationships from target geometry, orders lower layers before upper
layers, retains stable YAML IDs, and records the original 90-degree-first grasp
accessibility preference. The C++ executor then runs this state machine for each
part:

```text
move above source
  -> open gripper
  -> descend vertically
  -> close gripper and confirm dual-fingertip contact
  -> lift vertically back to the source approach height
  -> move above target
  -> align the carried part with a Cartesian segment
  -> descend vertically
  -> open gripper
  -> retreat vertically
```

Only the source-to-target overhead transfers use general motion planning.
Approach, lift, carried-part alignment, placement, and retreat are Cartesian
paths. A lift does not begin until the simulator reports sustained contact on
both fingertips with the requested part.

The reference executor consumes only the public benchmark topics, actions, and
services. It is a baseline and an executable integration example, not a required
part of an evaluation method. Replace `executor:=oracle` with `executor:=none`
and run your own ROS 2 node to test another task planner, perception system,
grasp generator, controller, or complete policy. See
[Executor integration](docs/executor-integration.md).

## Evaluation modes

`observation:=oracle` publishes exact MuJoCo poses and is intended for planning and control experiments. `observation:=rgbd` enables image-based evaluation.

`connection_mode:=snap` uses deterministic grasp attachment and documented
target-aware stud alignment. This isolates sequencing and motion-planning
experiments from contact-model variance. `connection_mode:=physics` disables
these constraints and leaves grasping and engagement to MuJoCo contacts.
Results should only be compared when product, seed, observation mode,
connection mode, tolerances, and simulator version are identical.

Query and save a result with:

```bash
ros2 service call /mj_bridge/result std_srvs/srv/Trigger '{}'
```

The episode directory contains the normalized product, episode manifest, generated MuJoCo scene, actual final state, and result JSON. The manifest is the complete record needed to reproduce the initial scene.

## Batch experiments

```bash
lego-bench batch-generate \
  --product examples/products/traffic_light.yaml \
  --first-seed 0 \
  --count 100 \
  --output-dir runs/traffic-light

lego-bench summarize runs/traffic-light
```

## Docker

```bash
docker compose -f docker-compose.benchmark.yaml up --build
```

The Compose configuration uses host networking for ROS 2 discovery and stores episode outputs under `runs/`.

## Repository layout

```text
examples/                     Product and executor examples
docs/                         Format, architecture, and integration notes
src/mj_bridge/launch/         Complete ROS 2 launch file
src/mj_bridge/mj_bridge/      Simulator bridge, generator, schema, and assets
src/mj_bridge/test/           Determinism, compatibility, and scoring tests
src/lego_executor/            Reference planner adapter and MoveIt executor
```

## Testing

```bash
source enter_sim_env.sh
python -m pytest -q src/mj_bridge/test/test_benchmark_core.py
colcon test --packages-select mj_bridge lego_executor
```

Do not replace `PYTHONPATH` after sourcing the environment script. ROS 2 adds its
Python packages to that variable, and replacing it can hide packages such as
`ament_flake8` and `ament_pep257` from the virtual environment.

## License and third-party material

The benchmark is released under the Apache License 2.0. The Panda assets retain their original Apache-2.0 notice. Public benchmark scenes use box and cylinder primitives for the assembly parts; legacy meshes with incomplete provenance are not included. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

LEGO is a trademark of the LEGO Group, which does not sponsor, authorize, or endorse this project.

## Citation

If this benchmark supports published work, cite the repository and the version or commit used for the experiments. A machine-readable entry is provided in [CITATION.cff](CITATION.cff).
