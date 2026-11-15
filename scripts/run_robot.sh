#!/usr/bin/env bash
# Starts the robot in the background: camera + VSLAM + Roomba driver + Nav2 + web app.
#   ~/run_robot.sh                 reuse the saved map (localization), or start one if none exists
#   ~/run_robot.sh mapping         start a NEW map (the current one is copied to ~/maps/backup first)
#   ~/run_robot.sh localization    require the saved map
# Extra launch arguments pass through, e.g.:  ~/run_robot.sh mapping cam_z:=0.16 nav:=false
# Log: ~/robot.log    Stop: ~/stop_slam.sh, or SIGTERM / SIGINT to the launch, whose PID is in
# ~/robot.pid (kill $(cat ~/robot.pid)): every node stops like with Ctrl-C and RTAB-Map saves the map.
HERE=$(dirname "$(readlink -f "$0")")   # scripts/ of the workspace this script belongs to
MODE=auto
case "$1" in mapping|localization|auto) MODE=$1; shift ;; esac
if [ "$MODE" = mapping ] && [ -s ~/maps/vslam.db ]; then
  mkdir -p ~/maps/backup
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp ~/maps/vslam.db ~/maps/backup/vslam_$STAMP.db
  [ -f ~/maps/vslam.zones.json ] && cp ~/maps/vslam.zones.json ~/maps/backup/vslam_$STAMP.zones.json
  echo "Previous map backed up to ~/maps/backup/vslam_$STAMP.db"
fi
"$HERE/stop_slam.sh" > /dev/null
# graceful.py runs `ros2 launch` so that SIGTERM and SIGINT stop it like Ctrl-C: started in the
# background, plain ros2 launch ignores SIGINT and dies on SIGTERM leaving every node running.
setsid nohup bash -c 'source /opt/ros/jazzy/setup.bash; source "$0/install/setup.bash"
  exec "$0/scripts/graceful.py" ros2 launch roomba_bringup robot.launch.py "$@"' "$(dirname "$HERE")" \
  mode:=$MODE "$@" > ~/robot.log 2>&1 < /dev/null &
echo $! > ~/robot.pid   # setsid, nohup and bash exec in place: this is the launch's PID
echo "Robot starting (mode $MODE). Open http://$(hostname -I | awk '{print $1}'):8080 and wait for 'Tracking OK'."
echo "Wake the Roomba first (press Clean twice) if its lights are off."
