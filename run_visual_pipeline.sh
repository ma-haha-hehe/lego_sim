#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRODUCT="${REPO_DIR}/examples/products/traffic_light.yaml"
SEED=42
OUTPUT_DIR="${REPO_DIR}/runs/visual-seed-42"
HEADLESS=false
VISION_BACKEND=foundationpose
CONNECTION_MODE=physics
CHECK_VISION=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --product) PRODUCT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --vision-backend) VISION_BACKEND="$2"; shift 2 ;;
    --connection-mode) CONNECTION_MODE="$2"; shift 2 ;;
    --check-vision) CHECK_VISION=true; shift ;;
    --headless) HEADLESS=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--product FILE] [--seed N] [--output-dir DIR] [--headless]"
      echo "          [--vision-backend foundationpose|geometry] [--connection-mode physics|snap]"
      echo "          [--check-vision]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "${VISION_BACKEND}" == "foundationpose" && -z "${FOUNDATIONPOSE_DIR:-}" ]]; then
  echo "FOUNDATIONPOSE_DIR must point to an installed FoundationPose checkout." >&2
  echo "For a simulator-only smoke test, pass --vision-backend geometry." >&2
  exit 2
fi

set +u
source "${REPO_DIR}/enter_sim_env.sh" >/dev/null
set -u

if [[ "${CHECK_VISION}" == "true" ]]; then
  exec python -m mj_bridge.vision_pipeline \
    --foundationpose-root "${FOUNDATIONPOSE_DIR:-}" --device cuda
fi

LAUNCH_ARGS=(
  "product:=${PRODUCT}"
  "seed:=${SEED}"
  "output_dir:=${OUTPUT_DIR}"
  "headless:=${HEADLESS}"
  "observation:=rgbd"
  "connection_mode:=${CONNECTION_MODE}"
  "executor:=vision"
  "vision_backend:=${VISION_BACKEND}"
)
if [[ -n "${FOUNDATIONPOSE_DIR:-}" ]]; then
  LAUNCH_ARGS+=("foundationpose_root:=${FOUNDATIONPOSE_DIR}")
fi

exec ros2 launch mj_bridge lego_bench.launch.py "${LAUNCH_ARGS[@]}"
