# Changelog

## Unreleased

- Increased reference arm, Cartesian, and gripper speeds while retaining
  measured-state convergence and physical grasp confirmation.
- Added a complete third-party ROS 2 policy adapter with reset, trajectory, and
  result helpers.
- Added an evaluation protocol, public support boundaries, and a one-command
  repository check.
- Extended CI coverage to every bundled product and policy example.

## 0.2.0 - 2026-08-24

- Added dependency-aware assembly plans with stable part IDs.
- Added a reference Panda/MoveIt executor with an explicit pick-and-place state machine.
- Added deterministic snap-mode grasping and target-aware layer engagement.
- Hardened table and part contacts to prevent visible surface penetration under arm load.
- Aligned snap-grasp visual and collision geometry with the physical fingertip center.
- Made attachment contact-gated during closing, delayed release until opening completes, added closed-loop pre-placement alignment, and made hand-to-support transfers atomic.
- Corrected Panda actuator commands and required dual-fingertip contact before grasp attachment.
- Matched the real-robot pick/place sequence with Cartesian descent, lift, placement alignment, and retreat segments.
- Added model-based arm feed-forward, endpoint convergence checks, and a tool-down constraint for stable real-time visualization.
- Added balanced headlight, key, and fill lighting for clearer workspace visibility.
- Added a direct Cartesian fallback when overhead MoveIt planning cannot find a path.
- Made raw MuJoCo contact physics the default connection mode, added contact-only grasp confirmation, registered part masses and inertias, hollow underside collision shells, and a small configurable placement preload.
- Added a direct Python environment for non-ROS policies with seeded reset, validated Panda actuator actions, oracle or RGB-D observations, standard five-value steps, scoring, and result export.
- Added a single-command reference pipeline and documented external executor integration.
- Extended the container, dependency installer, and CI checks for the executor package.

## 0.1.0 - 2026-08-20

- Added schema-v1 product descriptions and legacy YAML conversion.
- Added deterministic, non-overlapping loose-part generation.
- Added complete-assembly scoring and result export.
- Added Panda trajectory actions, MoveIt launch integration, reset services, ground-truth state, RGB-D images, segmentation, calibration, and TF.
- Added headless operation, batch generation, Docker configuration, CI, and a clean public release workflow.
