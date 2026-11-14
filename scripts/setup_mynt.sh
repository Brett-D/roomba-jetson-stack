#!/usr/bin/env bash
set -eo pipefail
while pgrep -f nv-install-docker.sh >/dev/null || pgrep -x apt-get >/dev/null || pgrep -x dpkg >/dev/null; do sleep 10; done; echo "### $(date) apt free"; sudo -n apt-get clean
echo "### $(date) disk: $(df -h / | tail -1)"
source /opt/ros/jazzy/setup.bash
cd ~
[ -d mynt-eye-ros2 ] || git clone https://github.com/antbaena/mynt-eye-ros2.git
cd mynt-eye-ros2
echo "### $(date) udev"
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1e4e", ATTR{idProduct}=="0120", MODE="0666", GROUP="plugdev"' | sudo -n tee /etc/udev/rules.d/99-mynteye-d1000.rules
sudo -n udevadm control --reload-rules && sudo -n udevadm trigger
echo "### $(date) S-SDK"
bash scripts/build_legacy_sdk_ubuntu24.sh
echo "### $(date) D-SDK"
bash scripts/build_d_sdk_ubuntu24.sh
echo "### $(date) colcon"
cd mynteye_ws
rosdep install --from-paths src -i -y --rosdistro jazzy || true
colcon build --packages-select mynteye_camera --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
echo "### $(date) DONE"