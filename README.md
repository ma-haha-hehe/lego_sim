# LEGO Assembly Benchmark for Panda and MuJoCo

[![benchmark-ci](https://github.com/ma-haha-hehe/lego_sim/actions/workflows/benchmark-ci.yml/badge.svg)](https://github.com/ma-haha-hehe/lego_sim/actions/workflows/benchmark-ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A reproducible ROS 2 benchmark for evaluating manipulation, perception, and assembly methods with a Franka Emika Panda robot in MuJoCo.

The benchmark takes a product description in YAML, creates the required loose parts at deterministic random poses, exposes standard ROS 2 control and observation interfaces, and evaluates the final assembly. The environment is independent of the policy: a method may use MoveIt, direct joint trajectories, ground-truth state, RGB-D input, or its own perception and planning stack.

## Features

- Deterministic episode generation from a product YAML and integer seed
- Non-overlapping random positions with 0/90-degree source orientations
- Panda arm and gripper control through `FollowJointTrajectory`
- MoveIt 2 planning and trajectory execution
- Ground-truth and RGB-D observation modes with an explicit no-oracle boundary
- RGB, metric depth, camera calibration, and TF
- Overhead source observation and oblique placement verification cameras
- GroundingDINO, SAM, and FoundationPose visual reference pipeline
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

Run the complete reference pipeline (episode generation, MuJoCo, direct
Cartesian execution, and result export):

```bash
./run_reference_pipeline.sh --headless
```

Direct Cartesian point-to-point motion is the default. It disables general
obstacle planning and joins each manipulation waypoint with simple straight
segments. Select the original MoveIt/OMPL planner for the overhead motions when
an experiment requires collision-aware planning:

```bash
./run_reference_pipeline.sh --motion-mode moveit --headless
```

Remove `--headless` to open MuJoCo and RViz. The launch remains open after the
executor finishes so the final scene can be inspected; press `Ctrl+C` to stop it.
Results are written under `runs/reference-seed-42/`.

To run the visual reference pipeline, see
[Visual pick-and-place pipeline](docs/visual-pipeline.md). A CUDA-free geometry
backend is included for simulator integration tests; production visual runs use
GroundingDINO, SAM and FoundationPose.

```bash
./run_visual_pipeline.sh --headless
```

The visual launcher defaults to `--vision-backend auto`. It selects the
FoundationPose pipeline only when its CUDA preflight passes, otherwise it
reports the fallback and runs the CPU RGB-D geometry backend. Use
`--vision-backend foundationpose` or `--vision-backend geometry` to make an
experiment explicit. Oracle runs remain separate and never act as a hidden
fallback for RGB-D evaluation.

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
uses contact-verified grasp physics and supports both oracle and RGB-D
observations. See
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
| `/lego_bench/ground_truth` | Topic | Authoritative part poses as JSON (oracle mode only) |
| `/camera/color/image_raw` | Topic | `rgb8` image |
| `/camera/depth/image_raw` | Topic | `32FC1` metric depth |
| `/camera/camera_info` | Topic | Pinhole camera calibration |
| `/lego_bench/vision_detections` | Topic | Visual pose estimates as JSON |
| `/mj_bridge/capture_rgbd` | Service | Request one fresh RGB-D frame at a stationary pose |
| `/mj_bridge/capture_placement_rgbd` | Service | Request an oblique RGB-D frame of a carried part |
| `/lego_vision/detect` | Service | Estimate loose-part poses from the overhead view |
| `/lego_vision/detect_placement` | Service | Estimate the carried-part pose before placement |
| `/mj_bridge/benchmark_state` | Topic | Live score as JSON |
| `/mj_bridge/reset` | Service | Restore the initial episode state |
| `/mj_bridge/result` | Service | Score and export the current state |

The full launch also provides MoveIt actions and services, including `/move_action`, `/execute_trajectory`, and `/plan_kinematic_path`. The camera frames are `realsense` and `placement_camera`; both static transforms from `world` are published by the launch file.

An integration skeleton is available in [examples/external_executor.py](examples/external_executor.py).

## Reference executor

Episode generation writes `execution_plan.yaml`. The planner follows the same
reverse-disassembly strategy used by the original robot pipeline: it removes
accessible upper parts first, records a face-aligned grasp for each removal,
then reverses that list into a bottom-up assembly order. It preserves the YAML
IDs and direct support relationships. Grasp angles are always either 0 or 90
degrees in the part's local frame, with 90 degrees preferred when both finger
corridors are clear. This distinction matters for rectangular 4x2 bricks and is
not overridden by the executor. The executor also compensates for the Panda
hand's 45-degree mounting angle, so these values describe the physical
finger-closing axis rather than the `panda_link8` axis. The C++ executor then
runs this state machine for each part:

```text
move above source
  -> open gripper
  -> descend vertically
  -> close gripper and confirm dual-fingertip contact
  -> lift vertically back to the source approach height
  -> move directly above target while setting its target yaw
  -> descend vertically
  -> open gripper
  -> retreat vertically
```

The default `cartesian` motion mode bypasses OMPL. Every state sends one direct
Cartesian line from the current pose to its target pose. Approach, lift,
placement, and retreat are vertical lines; observation, pre-grasp, and overhead
transport are single point-to-point lines. Collision checking is disabled for
these paths. MoveIt still supplies robot kinematics, trajectory timing, and
execution, but does not choose the route. Empty moves run at a higher speed than
contact and payload motions; the latter retain separate acceleration limits so
long bricks remain stable in the gripper. Targets already reached within the
motion tolerance are skipped instead of sending a redundant trajectory.

Pass `--motion-mode moveit` to restore collision-aware planning for the source
and target approach poses. The vertical manipulation strokes remain Cartesian
in both modes. If MoveIt planning fails, the same direct point-to-point route is
used as a fallback. A lift never begins until the simulator reports sustained
contact on both fingertips with the requested part.

Reference motion speed, Cartesian speed, gripper duration, and the short
state-publication delays between actions are configured in
`src/lego_executor/config/executor.yaml`. The defaults provide a fast baseline;
reduce the speed scales when testing a controller with lower dynamic limits.
Clear-space translations and wrist rotations use `free_motion_time_scale`,
while motion with a grasped part uses the more conservative
`payload_motion_time_scale`. Grasp and placement descents use their own
`contact_motion_time_scale`; loaded lifts retain nominal timing. The gripper
reaches its final preload before the bridge records the carried-part reference,
so finger seating is not mistaken for transport slip.
The simulator advances two complete physics/controller frames per display
frame by default, so the GUI plays the unchanged physical trajectory at roughly
twice wall-clock speed. Set `MJ_BRIDGE_SIM_SPEED=1` for real-time playback.

The reference executor consumes only the public benchmark topics, actions, and
services. It is a baseline and an executable integration example, not a required
part of an evaluation method. Replace `executor:=oracle` with `executor:=none`
and run your own ROS 2 node to test another task planner, perception system,
grasp generator, controller, or complete policy. See
[Executor integration](docs/executor-integration.md).

## Evaluation modes

`observation:=oracle` publishes exact MuJoCo poses and is intended for planning
and control experiments. `observation:=rgbd` publishes only RGB-D images and
camera calibration; ground-truth poses and simulator segmentation are not
published in this mode.

`connection_mode:=physics` is the default. Grasp confirmation requires
sustained contact with both fingertips. At release, a part inside the valid
stud capture region retains its measured pose relative to the contacted
support; this models the holding force of stud interference that the primitive
collision geometry cannot resolve. The latch does not move the part to its
YAML target or repair XY/yaw error. Brick mass and principal inertia come from
the public part registry; gravity, friction, contact compliance, and the Panda
actuator dynamics remain active. Run the reference executor with
with:

```bash
./run_reference_pipeline.sh --connection-mode physics
```

The compound collision model includes an open underside so studs enter the
brick cavity during placement. A 2-by-2 part receives a bounded 0.5 mm seating
stroke after first support contact. A supported 4-by-2 beam receives 2 mm so
both sides can engage; a base-layer beam uses the shorter stroke. It
approximates the shell with a slightly enlarged collision opening in place of
the moulded lead-in chamfer and a thin internal load surface at the nominal
19.2 mm layer pitch. It is not a calibrated material model of stud-and-tube
interference. Rubber fingertip pads use a high contact
coefficient to resist in-hand slip; ABS-to-ABS and ABS-to-base contacts use a
lower coefficient so the primitive stud geometry can self-centre during the
seating stroke.
`connection_mode:=snap` adds deterministic grasp attachment and target-aware
stud alignment, which is useful when planner experiments must be isolated from
contact-model variance.
Results should only be compared when product, seed, observation mode,
connection mode, tolerances, and simulator version are identical.

Query and save a result with:

```bash
ros2 service call /mj_bridge/result std_srvs/srv/Trigger '{}'
```

The episode directory contains the normalized product, episode manifest, generated MuJoCo scene, actual final state, and result JSON. Loose-part positions are randomized from the seed while their initial yaw is sampled from 0 or 90 degrees. The manifest is the complete record needed to reproduce the initial scene.

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

Before a release or demonstration, run the full headless check:

```bash
./check_benchmark.sh --e2e
```

This runs six isolated ROS episodes using contact-verified grasp physics and
pose-preserving stud engagement: oracle and RGB-D geometry runs for
`single_block`, `traffic_light`, and `bridge`. The multi-part
cases exercise dependency ordering, repeated perception, stacked placement,
and a two-support bridge. The runner requires every part to complete the full
state sequence and enforces final-pose, grasp-slip, collision, and stability
limits. It also checks the run artifacts and visual debug outputs, assigns each
episode a separate ROS domain, stops every launch automatically, and writes an
`e2e_summary.json` report under `runs/e2e/<timestamp>/`.

The release check applies the published benchmark limits of 6 mm XY, 4 mm Z,
and 8 degrees yaw to both observation modes. Grasp translation and rotation are
limited separately to 2 mm and 3 degrees, so a part cannot pass merely because
it happens to settle near the target after slipping in the gripper.

Run a smaller subset while developing:

```bash
python3 tests/run_e2e.py --product traffic_light --case geometry
python3 tests/run_e2e.py --product bridge --case oracle --seed 1
```

## License and third-party material

The benchmark is released under the Apache License 2.0. The Panda assets retain their original Apache-2.0 notice. Public benchmark scenes use box and cylinder primitives for the assembly parts; legacy meshes with incomplete provenance are not included. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

LEGO is a trademark of the LEGO Group, which does not sponsor, authorize, or endorse this project.

## Citation

If this benchmark supports published work, cite the repository and the version or commit used for the experiments. A machine-readable entry is provided in [CITATION.cff](CITATION.cff).
