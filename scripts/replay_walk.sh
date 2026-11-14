#!/usr/bin/env bash
# Replays a recorded walk through the whole stack (Basalt, motion filter, RTAB-Map, detector) so
# you can watch it in the web app (Map/Camera tabs) or Lichtblick. The camera driver is off and
# the map is built into /tmp/replay_view.db: your saved map ~/maps/vslam.db is not touched.
#   ~/replay_walk.sh                     newest recording in ~/bags, real time
#   ~/replay_walk.sh --rate 0.5          half speed (faster than 1 may make tracking drop frames)
#   ~/replay_walk.sh --bag ~/bags/walk_YYYYMMDD_HHMMSS
# The replayed map stays up after the recording ends. Back to normal: ~/go_live.sh
BAG=""; RATE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --rate) RATE=$2; shift 2 ;;
    --bag) BAG=$2; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done
[ -z "$BAG" ] && BAG=$(ls -dt ~/bags/walk_* 2>/dev/null | head -n1)
[ -d "$BAG" ] || { echo "No recording found (looked in ~/bags)"; exit 1; }
if [ -z "$REPLAY_CHILD" ]; then
  REPLAY_CHILD=1 setsid nohup "$0" --bag "$BAG" --rate "$RATE" > ~/replay_walk.log 2>&1 < /dev/null &
  IP=$(hostname -I | awk '{print $1}')
  echo "Replaying $(basename "$BAG") at x$RATE speed (starts in ~20 s)."
  echo "Watch: http://$IP:8080 (Map tab) or Lichtblick -> ws://$IP:8765"
  echo "Your saved map is untouched. Back to live mode: ~/go_live.sh"
  exit 0
fi
pkill -INT -f "ros2 bag play" 2>/dev/null
~/stop_slam.sh
rm -f /tmp/replay_view.db /tmp/replay_view.zones.json*
setsid ~/run_vslam.sh mode:=mapping camera:=false database_path:=/tmp/replay_view.db > ~/vslam.log 2>&1 < /dev/null &
sleep 15
source /opt/ros/jazzy/setup.bash
ros2 bag play "$BAG" --rate "$RATE" --read-ahead-queue-size 2000 \
  --topics /camera/left/image_rect_mono /camera/right/image_rect_mono /camera/left/camera_info \
           /camera/right/camera_info /camera/left/image_rect_color /camera/depth/image_rect_raw \
           /camera/depth/camera_info /camera/imu/data_raw /tf_static
echo "Replay finished. The replayed map stays up until ~/go_live.sh"
