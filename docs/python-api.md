# Direct Python API

`LegoBenchEnv` runs the generated MuJoCo model in the calling Python process.
It is intended for reinforcement learning, scripted controllers, and policy
evaluation that do not require ROS 2, RViz, or MoveIt.

## Create an environment

```python
from mj_bridge.gym_env import LegoBenchEnv

env = LegoBenchEnv(
    "examples/products/traffic_light.yaml",
    seed=42,
    output_dir="runs/my-policy",
    observation_mode="oracle",
    frame_skip=20,
    max_episode_steps=5000,
)
observation, info = env.reset()
```

Any schema-v1 product YAML accepted by `lego-bench validate` can be used. The
seed controls randomized, non-overlapping source positions and chooses a 0- or
90-degree initial yaw for every part. Changing the reset seed regenerates the
episode:

```python
observation, info = env.reset(seed=43)
```

## Actions

An action is a NumPy-compatible vector of absolute actuator controls. The
default Panda model has seven arm actuators and two finger actuators:

```python
print(env.actuator_names)
print(env.action_shape)
lower, upper = env.action_bounds

action = policy(observation)
observation, reward, terminated, truncated, info = env.step(action)
```

Commands are checked for the exact shape and finite values, then clipped to
the actuator control ranges. `frame_skip` MuJoCo steps are executed for each
policy step. `neutral_action` returns the reset-time command and is useful for
smoke tests; it is not an assembly policy. The direct environment loads the
same Panda home configuration and applies the same arm gravity feed-forward as
the ROS bridge.

The step return follows the current Gymnasium convention:

- `observation`: policy-visible robot, goal, and sensor data
- `reward`: current fraction of correctly placed blocks
- `terminated`: the complete assembly satisfies all tolerances
- `truncated`: `max_episode_steps` was reached
- `info`: score details, step count, connection mode, and reference plan

## Observations

`observation_mode="oracle"` includes exact block poses under `blocks`. This is
appropriate for motion-control and task-planning experiments.

`observation_mode="rgbd"` returns `rgb` and metric `depth` arrays instead. It
does not include block ground truth. Configure MuJoCo for headless rendering
before Python imports the package:

```bash
MUJOCO_GL=egl python your_policy.py
```

Both modes include the target assembly, Panda joint positions and velocities,
current actuator controls, episode ID, seed, and simulation time.

## Results and artifacts

```python
result = env.export_result()
env.close()
```

The output directory contains the normalized product, episode manifest,
execution plan, generated scene, `actual_state.json`, and `result.json`. Use a
context manager to close the renderer automatically:

```python
with LegoBenchEnv("product.yaml", seed=0) as env:
    observation, info = env.reset()
    # Run the policy.
    result = env.export_result()
```

The direct API deliberately uses `physics` connection semantics. It does not
create hidden grasp welds or target snaps. Use the ROS 2 launch interface when
the deterministic `snap` condition is required.
