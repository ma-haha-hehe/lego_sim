# Contributing

Contributions should keep the benchmark deterministic and avoid changing an existing experimental condition without a version change.

Before opening a pull request:

1. Build the ROS package with `colcon build --packages-select mj_bridge`.
2. Run `PYTHONPATH=src/mj_bridge python -m pytest -q src/mj_bridge/test`.
3. Validate any added product files with `lego-bench validate`.
4. Document changes to schemas, physics, tolerances, observations, or metrics.
5. Include provenance and redistribution terms for every new asset.

Keep generated scenes, episode outputs, virtual environments, model weights, and build artifacts out of commits. New part types should include tests for seed reproducibility, collision-free spawning, MJCF loading, and scoring.
