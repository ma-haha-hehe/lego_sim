#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRODUCT="${REPO_DIR}/examples/products/traffic_light.yaml"
SEED=42
OUTPUT_DIR="${REPO_DIR}/runs/reference-seed-42"
HEADLESS=false
OBSERVATION=oracle
CONNECTION_MODE=physics

while [[ $# -gt 0 ]]; do
  case "$1" in
    --product) PRODUCT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --observation) OBSERVATION="$2"; shift 2 ;;
    --connection-mode) CONNECTION_MODE="$2"; shift 2 ;;
    --headless) HEADLESS=true; shift ;;
    -h|--help)
      echo "Usage: $0 [--product FILE] [--seed N] [--output-dir DIR] [--headless]"
      echo "          [--observation oracle|rgbd] [--connection-mode snap|physics]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

set +u
source "${REPO_DIR}/enter_sim_env.sh" >/dev/null
set -u

exec ros2 launch mj_bridge lego_bench.launch.py \
  product:="${PRODUCT}" \
  seed:="${SEED}" \
  output_dir:="${OUTPUT_DIR}" \
  headless:="${HEADLESS}" \
  observation:="${OBSERVATION}" \
  connection_mode:="${CONNECTION_MODE}" \
  executor:=oracle
