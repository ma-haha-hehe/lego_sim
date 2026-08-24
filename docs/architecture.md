# Architecture

The benchmark separates scenario definition from method execution.

```text
product YAML + seed
        |
        v
validation and normalization
        |
        v
episode manifest -----> standalone MuJoCo scene
        |                         |
        |                         v
        +-----------------> MuJoCo ROS bridge
                                  |
                +-----------------+-----------------+
                |                 |                 |
             actions          observations      scoring
                |                 |                 |
                +------ reference or external method ------+
```

`benchmark_core.py` contains pure-Python product handling, seeded placement, and scoring. `executor_planner.py` derives a stable-ID, bottom-up execution plan. `scene_builder.py` converts an episode manifest into MJCF. `mj_bridge3.py` owns simulation time, trajectory actions, observations, reset, and result export. `lego_bench.launch.py` adds the robot description, TF, MoveIt planning services, and optionally the `lego_executor` baseline.

## Reproducibility boundary

The product and seed determine the generated source poses. The episode manifest stores both source and target poses, so a run remains reproducible even if the sampling implementation changes later. A reported result should retain the manifest, simulator version, benchmark commit, observation mode, connection mode, and tolerance configuration.

## Connection models

The `physics` model leaves all grasps and assembly contacts to MuJoCo. The `snap` model adds deterministic gripper attachment after closing, then aligns a released part with its declared target and direct support layer inside a fixed capture region. The model is deliberately explicit because it is part of the experimental condition.

## Scoring

Parts are matched by stable ID rather than nearest-neighbor assignment. XY, Z, and symmetry-aware yaw errors are evaluated independently. An assembly succeeds only when every required part is within tolerance. The live and exported result also reports completion, elapsed time, timeout, robot-environment contact events, and part stability violations.
