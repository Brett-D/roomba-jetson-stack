#!/usr/bin/env bash
set -eo pipefail
while pgrep -x apt-get >/dev/null || pgrep -x dpkg >/dev/null; do sleep 5; done
source /opt/ros/jazzy/setup.bash
cd ~/mynt-eye-ros2
echo "### $(date) D-SDK"
bash scripts/build_d_sdk_ubuntu24.sh
echo "### $(date) colcon"
cd mynteye_ws
rosdep install --from-paths src -i -y --rosdistro jazzy || true
colcon build --packages-select mynteye_camera --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
echo "### $(date) DONE"