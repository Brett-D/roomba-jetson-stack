#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source ~/roomba_ws/install/setup.bash
# graceful.py: SIGTERM / SIGINT stop the launch like Ctrl-C, also when started in the background.
exec "$(dirname "$(readlink -f "$0")")/graceful.py" ros2 launch roomba_bringup vslam.launch.py "$@"
