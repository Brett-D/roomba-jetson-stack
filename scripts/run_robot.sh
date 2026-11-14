#!/usr/bin/env bash
# Starts the robot in the background: camera + VSLAM + Roomba driver + Nav2 + web app.
#   ~/run_robot.sh                 reuse the saved map (localization), or start one if none exists
#   ~/run_robot.sh mapping         start a NEW map (the current one is copied to ~/maps/backup first)
#   ~/run_robot.sh localization    require the saved map
# Extra launch arguments pass through, e.g.:  ~/run_robot.sh mapping cam_z:=0.16 nav:=false
# Log: ~/robot.log    Stop: ~/stop_slam.sh
MODE=auto
case "$1" in mapping|localization|auto) MODE=$1; shift ;; esac
if [ "$MODE" = mapping ] && [ -s ~/maps/vslam.db ]; then
  mkdir -p ~/maps/backup
  STAMP=$(date +%Y%m%d_%H%M%S)
  cp ~/maps/vslam.db ~/maps/backup/vslam_$STAMP.db
  [ -f ~/maps/vslam.zones.json ] && cp ~/maps/vslam.zones.json ~/maps/backup/vslam_$STAMP.zones.json
  echo "Previous map backed up to ~/maps/backup/vslam_$STAMP.db"
fi
~/stop_slam.sh > /dev/null
setsid nohup bash -c 'source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
  exec ros2 launch roomba_bringup robot.launch.py "$@"' _ mode:=$MODE "$@" > ~/robot.log 2>&1 < /dev/null &
echo "Robot starting (mode $MODE). Open http://$(hostname -I | awk '{print $1}'):8080 and wait for 'Tracking OK'."
echo "Wake the Roomba first (press Clean twice) if its lights are off."
