#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  python3-venv \
  python3-pip \
  python3-rosdep \
  python3-colcon-common-extensions \
  ros-humble-control-msgs \
  ros-humble-moveit \
  ros-humble-moveit-resources-panda-moveit-config \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui \
  ros-humble-robot-state-publisher \
  ros-humble-tf2-ros \
  ros-humble-rviz2 \
  ros-humble-xacro \
  libgl1 \
  libglfw3

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv --system-site-packages "${REPO_DIR}/.venv"
"${REPO_DIR}/.venv/bin/python" -m pip install --upgrade pip
"${REPO_DIR}/.venv/bin/python" -m pip install -r "${REPO_DIR}/requirements-sim.txt"

set +u
source /opt/ros/humble/setup.bash
set -u
if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  "${SUDO[@]}" rosdep init
fi
rosdep update
rosdep install --from-paths "${REPO_DIR}/src/mj_bridge" --ignore-src -r -y

echo "ROS and Python dependencies are installed. Next: source enter_sim_env.sh"
