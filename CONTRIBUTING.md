# Contributing

Contributions should keep the benchmark deterministic and avoid changing an existing experimental condition without a version change.

Before opening a pull request:

1. Run `./check_benchmark.sh` from a configured workspace.
2. Validate any additional product datasets with `lego-bench validate`.
3. Document changes to schemas, physics, tolerances, observations, or metrics.
4. Include provenance and redistribution terms for every new asset.
5. Add an entry under `Unreleased` in `CHANGELOG.md` for user-visible changes.

Keep generated scenes, episode outputs, virtual environments, model weights, and build artifacts out of commits. New part types should include tests for seed reproducibility, collision-free spawning, MJCF loading, and scoring.
