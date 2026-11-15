#!/usr/bin/env bash
# Runs the 2D digital twin (roomba_bringup/launch/twin2d.launch.py) in the foreground: the real
# navigation stack on a simulated Roomba in the house. Ends by itself when the run is over
# (results in out_dir), or on Ctrl-C / SIGTERM.
#   scripts/run_twin2d.sh scenario:=src/roomba_bringup/config/twin2d/explore_from_dock.yaml \
#                         out_dir:=/data/twin/twin2d/runs/try1
#   scripts/run_twin2d.sh exit_on_finish:=false foxglove:=true    # drive it by hand / watch it
# Web app: http://<jetson>:18097 (password in <out_dir>/home/.config/roomba_web/initial_password).
# ROS_DOMAIN_ID defaults to 68 and discovery stays on this machine, apart from a running robot.
HERE=$(dirname "$(readlink -f "$0")")
WS=$(dirname "$HERE")
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-68}
export ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}
set +u
source /opt/ros/jazzy/setup.bash
# An overlay workspace (a worktree) builds only some packages: its underlay is ~/roomba_ws.
if [ "$WS" != "$(readlink -f ~/roomba_ws)" ] && [ -f ~/roomba_ws/install/setup.bash ]; then
  source ~/roomba_ws/install/setup.bash
fi
source "$WS/install/setup.bash"
exec "$HERE/graceful.py" ros2 launch roomba_bringup twin2d.launch.py "$@"
