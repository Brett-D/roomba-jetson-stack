#!/usr/bin/env bash
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
APT="sudo -n -E apt-get -y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold"
echo "### $(date) base"
sudo -n usermod -aG dialout,plugdev,video brett
$APT update
$APT install software-properties-common curl git build-essential cmake pkg-config v4l-utils usbutils \
  libusb-dev libusb-1.0-0-dev libopencv-dev python3-pip htop tmux nano
sudo -n add-apt-repository -y universe
echo "### $(date) nvidia-jetpack"
$APT install nvidia-jetpack
echo "### $(date) ros apt source"
VER=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}')
curl -fsSL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${VER}/ros2-apt-source_${VER}.noble_all.deb"
sudo -n dpkg -i /tmp/ros2-apt-source.deb
$APT update
echo "### $(date) ros jazzy"
$APT install ros-jazzy-desktop ros-dev-tools python3-colcon-common-extensions python3-rosdep \
  ros-jazzy-rtabmap-ros ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-robot-localization \
  ros-jazzy-imu-tools ros-jazzy-image-pipeline ros-jazzy-image-transport-plugins ros-jazzy-cv-bridge \
  ros-jazzy-foxglove-bridge ros-jazzy-teleop-twist-keyboard ros-jazzy-tf2-tools ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher ros-jazzy-diagnostic-updater
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo -n rosdep init
rosdep update
grep -q '/opt/ros/jazzy/setup.bash' ~/.bashrc || echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
echo "### $(date) DONE"