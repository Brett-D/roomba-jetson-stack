#!/usr/bin/env bash
# Starts Nav2 "human robot" navigation; run while ~/run_vslam.sh is running.
source /opt/ros/jazzy/setup.bash
source ~/roomba_ws/install/setup.bash
exec ros2 launch roomba_bringup nav.launch.py "$@"
