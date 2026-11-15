#!/usr/bin/env bash
# Starts Nav2 "human robot" navigation; run while ~/run_vslam.sh is running.
source /opt/ros/jazzy/setup.bash
source ~/roomba_ws/install/setup.bash
# graceful.py: SIGTERM / SIGINT stop the launch like Ctrl-C, also when started in the background.
exec "$(dirname "$(readlink -f "$0")")/graceful.py" ros2 launch roomba_bringup nav.launch.py "$@"
