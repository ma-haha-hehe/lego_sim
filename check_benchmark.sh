#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_DIR="$(mktemp -d -t lego-bench-check-XXXXXX)"
trap 'rm -rf "${CHECK_DIR}"' EXIT

set +u
source /opt/ros/humble/setup.bash
set -u
if [[ -f "${REPO_DIR}/.venv/bin/activate" ]]; then
  set +u
  source "${REPO_DIR}/.venv/bin/activate"
  set -u
fi

cd "${REPO_DIR}"
colcon build --packages-select mj_bridge lego_executor --symlink-install
set +u
source "${REPO_DIR}/install/setup.bash"
set -u

export PYTHONPATH="${REPO_DIR}/src/mj_bridge:${PYTHONPATH:-}"
python3 -m pytest -q src/mj_bridge/test/test_benchmark_core.py
python3 -m py_compile examples/external_executor.py examples/lego_gym_env.py

for product in examples/products/*.yaml; do
  name="$(basename "${product}" .yaml)"
  output="${CHECK_DIR}/${name}"
  python3 -m mj_bridge.benchmark_cli validate "${product}"
  python3 -m mj_bridge.benchmark_cli generate \
    --product "${product}" --seed 42 --output-dir "${output}"
  python3 - "${output}/scene.xml" <<'PY'
import sys

import mujoco

mujoco.MjModel.from_xml_path(sys.argv[1])
PY
done

colcon test --packages-select lego_executor --event-handlers console_direct+
colcon test-result --test-result-base build/lego_executor --verbose

echo "Repository check passed: packages, tests, examples, and generated scenes are valid."
