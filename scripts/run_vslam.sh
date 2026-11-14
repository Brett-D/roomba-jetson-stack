#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source ~/roomba_ws/install/setup.bash
exec ros2 launch roomba_bringup vslam.launch.py "$@"
