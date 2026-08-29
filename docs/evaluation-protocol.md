# Evaluation protocol

This protocol keeps results from different methods comparable. A benchmark run
is defined by the product, seed, observation mode, connection mode, tolerance
configuration, simulator version, and repository commit. Changing any of these
creates a different experimental condition.

## Before a run

1. Validate the product with `lego-bench validate`.
2. Record the repository commit with `git rev-parse HEAD`.
3. Choose `oracle` or `rgbd` before developing against the episode.
4. Choose `physics` or `snap` and keep that choice fixed across methods.
5. Use the same seed list for every method in a comparison.

The `physics` condition measures the complete contact-rich task. The `snap`
condition removes much of the grasp and stud-engagement variance and is useful
for isolating task planning or arm motion. Results from the two conditions must
not be combined.

## Running an external method

Start the environment without the supplied executor:

```bash
ros2 launch mj_bridge lego_bench.launch.py \
  product:=$PWD/examples/products/traffic_light.yaml \
  seed:=42 \
  output_dir:=$PWD/runs/my-method/seed-42 \
  headless:=true \
  observation:=oracle \
  connection_mode:=physics \
  executor:=none
```

Start the method in a second terminal in the same `ROS_DOMAIN_ID`. The adapter
in `examples/external_executor.py` contains ready-to-use reset, arm trajectory,
gripper trajectory, and result clients.

After execution, request the score before stopping the simulator:

```bash
ros2 service call /mj_bridge/result std_srvs/srv/Trigger '{}'
```

## Repeated trials

Use at least several seeds for contact-physics comparisons. `batch-generate`
prepares reproducible scenes; it does not run a policy. A policy runner should
launch one episode per seed, wait for completion or timeout, call the result
service, and retain each run directory.

Report success rate and mean completion together with per-part position and yaw
errors. Runtime is measured from episode reset. Also report timeouts, collision
counts, and stability violations; silently discarding failed episodes biases the
result.

## Artifacts to retain

Keep the following files for every reported run:

- original product YAML
- `product.normalized.yaml`
- `episode_manifest.yaml`
- `execution_plan.yaml`, if used by the method
- `actual_state.json`
- `result.json`

The generated `scene.xml` and copied model assets make the episode self-contained
and are useful when a result needs to be reproduced exactly.
