#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u
source "${REPO_DIR}/enter_sim_env.sh"
set -u

if ! python -c 'import mujoco, rclpy, control_msgs' >/dev/null 2>&1; then
  echo "Dependencies are incomplete. Run: bash install_sim_system_deps.sh" >&2
  exit 1
fi

colcon build --packages-select mj_bridge --symlink-install
set +u
source "${REPO_DIR}/install/mj_bridge/share/mj_bridge/package.bash"
set -u
PRODUCT="${LEGO_BENCH_PRODUCT:-${REPO_DIR}/examples/products/traffic_light.yaml}"
SEED="${LEGO_BENCH_SEED:-0}"
# Optional positional arguments: product.yaml [seed]. Flags pass through.
if [[ $# -gt 0 && "${1}" != -* ]]; then
  PRODUCT="$1"
  shift
  if [[ $# -gt 0 && "$1" =~ ^-?[0-9]+$ ]]; then
    SEED="$1"
    shift
  fi
fi
exec lego-bench run --product "${PRODUCT}" --seed "${SEED}" "$@"
