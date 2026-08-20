#!/usr/bin/env bash
# Source this file; do not execute it in a child shell.
_lego_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS 2 Humble was not found at /opt/ros/humble." >&2
  return 1 2>/dev/null || exit 1
fi

source /opt/ros/humble/setup.bash
if [[ -f "${_lego_repo}/.venv/bin/activate" ]]; then
  source "${_lego_repo}/.venv/bin/activate"
else
  echo "Python environment not found: ${_lego_repo}/.venv" >&2
  return 1 2>/dev/null || exit 1
fi
# ROS entry points may use the system interpreter, so expose venv packages.
export PYTHONPATH="${_lego_repo}/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
export ROBOT_KOREA_ROOT="${_lego_repo}"
export MJ_BRIDGE_BENCHMARK_CONFIG="${_lego_repo}/src/mj_bridge/mj_bridge/benchmark.yaml"
if [[ -f "${_lego_repo}/install/mj_bridge/share/mj_bridge/package.bash" ]]; then
  # Load only this package to avoid unrelated overlays in a larger workspace.
  source "${_lego_repo}/install/mj_bridge/share/mj_bridge/package.bash"
  export PATH="${_lego_repo}/install/mj_bridge/lib/mj_bridge:${PATH}"
fi
cd "${_lego_repo}"
echo "Panda + MuJoCo benchmark environment is ready"
echo "Python: $(command -v python)"
echo "Build: colcon build --packages-select mj_bridge --symlink-install"
echo "Test: python -m pytest -q src/mj_bridge/test/test_benchmark_core.py"
echo "Validate: lego-bench validate examples/products/traffic_light.yaml"
echo "Simulator only: ./run_panda_lego_sim.sh --headless"
echo "MoveIt pipeline: ros2 launch mj_bridge lego_bench.launch.py headless:=true"
unset _lego_repo
