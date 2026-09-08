#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRODUCT="${REPO_DIR}/examples/products/traffic_light.yaml"
SEED=42
OUTPUT_DIR="${REPO_DIR}/runs/visual-seed-42"
HEADLESS=false
VISION_BACKEND=auto
CONNECTION_MODE=physics
MOTION_MODE=cartesian
CHECK_VISION=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --product) PRODUCT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --vision-backend) VISION_BACKEND="$2"; shift 2 ;;
    --connection-mode) CONNECTION_MODE="$2"; shift 2 ;;
    --motion-mode) MOTION_MODE="$2"; shift 2 ;;
    --check-vision) CHECK_VISION=true; shift ;;
    --headless) HEADLESS=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--product FILE] [--seed N] [--output-dir DIR] [--headless]"
      echo "          [--vision-backend auto|foundationpose|geometry] [--connection-mode physics|snap]"
      echo "          [--motion-mode cartesian|moveit]"
      echo "          [--check-vision]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "${VISION_BACKEND}" in
  auto|foundationpose|geometry) ;;
  *)
    echo "--vision-backend must be auto, foundationpose, or geometry." >&2
    exit 2
    ;;
esac

case "${MOTION_MODE}" in
  cartesian|moveit) ;;
  *) echo "--motion-mode must be cartesian or moveit." >&2; exit 2 ;;
esac

set +u
source "${REPO_DIR}/enter_sim_env.sh" >/dev/null
set -u

if [[ "${CHECK_VISION}" == "true" ]]; then
  exec python -m mj_bridge.vision_pipeline \
    --foundationpose-root "${FOUNDATIONPOSE_DIR:-}" --device cuda
fi

if [[ "${VISION_BACKEND}" == "auto" ]]; then
  if python -m mj_bridge.vision_pipeline \
      --foundationpose-root "${FOUNDATIONPOSE_DIR:-}" --device cuda \
      >/dev/null 2>&1; then
    VISION_BACKEND=foundationpose
    echo "Vision backend: foundationpose (CUDA preflight passed)"
  else
    VISION_BACKEND=geometry
    echo "Vision backend: geometry (CUDA/FoundationPose preflight unavailable)"
    echo "Run '$0 --check-vision' for production-backend diagnostics."
  fi
elif [[ "${VISION_BACKEND}" == "foundationpose" ]]; then
  if [[ -z "${FOUNDATIONPOSE_DIR:-}" ]]; then
    echo "FOUNDATIONPOSE_DIR must point to an installed FoundationPose checkout." >&2
    echo "Use --vision-backend auto or --vision-backend geometry on this machine." >&2
    exit 2
  fi
  if ! python -m mj_bridge.vision_pipeline \
      --foundationpose-root "${FOUNDATIONPOSE_DIR}" --device cuda \
      >/dev/null 2>&1; then
    echo "FoundationPose preflight failed:" >&2
    python -m mj_bridge.vision_pipeline \
      --foundationpose-root "${FOUNDATIONPOSE_DIR}" --device cuda >&2 || true
    exit 2
  fi
  echo "Vision backend: foundationpose (CUDA preflight passed)"
else
  echo "Vision backend: geometry (explicit selection)"
fi

LAUNCH_ARGS=(
  "product:=${PRODUCT}"
  "seed:=${SEED}"
  "output_dir:=${OUTPUT_DIR}"
  "headless:=${HEADLESS}"
  "observation:=rgbd"
  "connection_mode:=${CONNECTION_MODE}"
  "motion_mode:=${MOTION_MODE}"
  "executor:=vision"
  "vision_backend:=${VISION_BACKEND}"
)
if [[ -n "${FOUNDATIONPOSE_DIR:-}" ]]; then
  LAUNCH_ARGS+=("foundationpose_root:=${FOUNDATIONPOSE_DIR}")
fi

exec ros2 launch mj_bridge lego_bench.launch.py "${LAUNCH_ARGS[@]}"
