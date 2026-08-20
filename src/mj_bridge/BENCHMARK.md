# Benchmark configuration

`mj_bridge/benchmark.yaml` defines the episode timeout and final pose tolerances:

- `position_tolerance_xy_m`: horizontal center-position tolerance
- `position_tolerance_z_m`: vertical center-position tolerance
- `yaw_tolerance_deg`: symmetry-aware yaw tolerance
- `max_episode_time_s`: wall-clock episode timeout

The remaining lift fields support the original single-object grasp task when no product manifest is supplied. Product episodes use complete-assembly scoring.

## Episode services

```bash
ros2 service call /mj_bridge/reset std_srvs/srv/Trigger '{}'
ros2 service call /mj_bridge/result std_srvs/srv/Trigger '{}'
ros2 topic echo /mj_bridge/benchmark_state
```

The result service exports `actual_state.json` and `result.json` when the run directory is configured by the CLI or launch file.
