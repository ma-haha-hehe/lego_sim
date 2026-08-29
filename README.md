# LEGO Assembly Benchmark for Panda and MuJoCo

[![benchmark-ci](https://github.com/ma-haha-hehe/lego_sim/actions/workflows/benchmark-ci.yml/badge.svg)](https://github.com/ma-haha-hehe/lego_sim/actions/workflows/benchmark-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

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

Remove `--headless` to open MuJoCo and RViz. The launch remains open after the
executor finishes so the final scene can be inspected; press `Ctrl+C` to stop it.
Results are written under `runs/reference-seed-42/`.

To start the environment without the reference executor:

```bash
ros2 launch mj_bridge lego_bench.launch.py \
  product:=$PWD/examples/products/traffic_light.yaml \
  seed:=42 \
  headless:=false \
  observation:=oracle \
  connection_mode:=physics \
  executor:=none
```

This mode is intended for an external policy. For a headless RGB-D episode,
set `headless:=true` and `observation:=rgbd`.

## Bring your own method

The simulator does not require the supplied executor. Start the environment
with `executor:=none`, then run a policy in a second terminal after sourcing the
same workspace. The policy may use MoveIt, publish its own joint trajectories,
or replace the entire perception and planning stack.

```bash
# Terminal 1
source enter_sim_env.sh
ros2 launch mj_bridge lego_bench.launch.py \
  product:=$PWD/examples/products/bridge.yaml \
  seed:=17 \
  output_dir:=$PWD/runs/my-method/bridge-17 \
  headless:=false \
  observation:=oracle \
  connection_mode:=physics \
  executor:=none

# Terminal 2
source enter_sim_env.sh
python examples/external_executor.py
```

`examples/external_executor.py` includes clients for reset, arm trajectories,
gripper trajectories, and result export. Replace its `policy()` method with the
method under test. Interface details are in
[Executor integration](docs/executor-integration.md), and reporting rules are
in the [Evaluation protocol](docs/evaluation-protocol.md).

Python policies can also bypass ROS 2 and MoveIt while retaining the same
product generation, MuJoCo model, Panda actuators, and scoring contract:

```python
from mj_bridge.gym_env import LegoBenchEnv

with LegoBenchEnv(
    "examples/products/traffic_light.yaml",
    seed=42,
    output_dir="runs/python-policy",
) as env:
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(
        env.neutral_action
    )
    result = env.export_result()
```

Actions are nine absolute actuator controls in the model's actuator order.
Replace `neutral_action` with a controller or policy output. The environment
uses raw contact physics and supports both oracle and RGB-D observations. See
[Direct Python API](docs/python-api.md) and the runnable
[Python example](examples/lego_gym_env.py).

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

The reference executor is designed for top-down pick-and-place assembly. A
valid file is not a promise that every structure is physically executable:
targets must remain on the 12-by-12 assembly plate, inside the Panda workspace,
and accessible from above. Lateral insertion, part reorientation, unsupported
floating structures, and unregistered shapes require a custom method or an
extension to the part registry and scene builder.

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

Overhead transfers try MoveIt planning first. If all planning attempts fail,
the reference executor falls back to a direct Cartesian point-to-point motion
with collision checking disabled. Vertical approach and retreat segments always
use direct Cartesian motion and do not invoke the OMPL planner.

Reference motion speed, Cartesian speed, gripper duration, and the short
state-publication delays between actions are configured in
`src/lego_executor/config/executor.yaml`. The defaults provide a fast baseline;
reduce the speed scales when testing a controller with lower dynamic limits.

The reference executor consumes only the public benchmark topics, actions, and
services. It is a baseline and an executable integration example, not a required
part of an evaluation method. Replace `executor:=oracle` with `executor:=none`
and run your own ROS 2 node to test another task planner, perception system,
grasp generator, controller, or complete policy. See
[Executor integration](docs/executor-integration.md).

## Evaluation modes

`observation:=oracle` publishes exact MuJoCo poses and is intended for planning and control experiments. `observation:=rgbd` enables image-based evaluation.

`connection_mode:=physics` is the default and leaves grasping and engagement
to MuJoCo contacts. In physics mode, grasp confirmation requires sustained
contact with both fingertips, but no weld, pose correction, or target snap is
created. Brick mass and principal inertia come from the public part registry;
gravity, friction, contact compliance, and the Panda actuator dynamics remain
active in both modes. Run the reference executor against raw contact physics
with:

```bash
./run_reference_pipeline.sh --connection-mode physics
```

The compound collision model includes an open underside so studs enter the
brick cavity during placement. It approximates the external shell and fit, but
is not a calibrated material model of stud-and-tube interference.
`connection_mode:=snap` adds deterministic grasp attachment and target-aware
stud alignment, which is useful when planner experiments must be isolated from
contact-model variance.
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

`batch-generate` prepares episode files only. A policy evaluation must run each
seed and call `/mj_bridge/result`; see the evaluation protocol for the required
artifacts and reporting fields.

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
src/mj_bridge/mj_bridge/      Simulator bridge, direct Python API, generator, schema, and assets
src/mj_bridge/test/           Determinism, compatibility, and scoring tests
src/lego_executor/            Reference planner adapter and MoveIt executor
```

## Testing

```bash
./check_benchmark.sh
```

The repository check builds both ROS packages, runs the benchmark tests,
validates and generates every example product, loads each generated MJCF model,
compiles the policy examples, and runs the executor lint tests.

## License and third-party material

The benchmark is released under the Apache License 2.0. The Panda assets retain their original Apache-2.0 notice. Public benchmark scenes use box and cylinder primitives for the assembly parts; legacy meshes with incomplete provenance are not included. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

LEGO is a trademark of the LEGO Group, which does not sponsor, authorize, or endorse this project.

## Citation

If this benchmark supports published work, cite the repository and the version or commit used for the experiments. A machine-readable entry is provided in [CITATION.cff](CITATION.cff).
