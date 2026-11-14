#!/usr/bin/env bash
# Starts a NEW map (replaces ~/maps/vslam.db).
#   ~/map_house.sh            IR projector ON (pulsed on depth frames only, so tracking images
#                             stay dot-free), 15 Hz, good depth in dim rooms.
#   ~/map_house.sh --ir-off   30 Hz, IR off, less depth (bright, well-textured rooms).
#   ~/map_house.sh --record [--minutes N]   also record the raw camera + IMU to ~/bags
#                             (~1.4 GB/min, auto-stops after N min, default 5) so the
#                             walk can be replayed to tune mapping without walking it again.
# When you're done mapping, run ~/run_vslam.sh (reuses this map) and ~/run_nav.sh.
IR_ARGS="ir_depth_only:=true ir_intensity:=4"; RATE="15 Hz, IR on"; RECORD=0; MINUTES=5
while [ $# -gt 0 ]; do
  case "$1" in
    --ir-off) IR_ARGS="ir_depth_only:=false ir_intensity:=0"; RATE="30 Hz, IR off"; shift ;;
    --record) RECORD=1; shift ;;
    --minutes) MINUTES=$2; shift 2 ;;
    *) break ;;
  esac
done
~/stop_slam.sh > /dev/null
setsid nohup ~/run_vslam.sh mode:=mapping $IR_ARGS "$@" > ~/vslam.log 2>&1 < /dev/null &
echo "Mapping started (new map, $RATE). Open http://$(hostname -I | awk '{print $1}'):8080 and wait for 'Tracking OK'."
if [ $RECORD = 1 ]; then
  mkdir -p ~/bags; BAG=~/bags/walk_$(date +%Y%m%d_%H%M%S)
  (source /opt/ros/jazzy/setup.bash; sleep 8
   exec setsid nohup timeout -s INT $((MINUTES * 60)) ros2 bag record -s mcap -o "$BAG" \
     /camera/left/image_rect_mono /camera/right/image_rect_mono /camera/left/camera_info \
     /camera/right/camera_info /camera/left/image_rect_color /camera/depth/image_rect_raw \
     /camera/depth/camera_info /camera/imu/data_raw /tf_static /visual_slam/status \
     /visual_slam/tracking/odometry > ~/bag_record.log 2>&1 < /dev/null) &
  echo "Recording to $BAG (max $MINUTES min, or until ~/stop_slam.sh). Free disk: $(df -h ~ | awk 'NR==2{print $4}')"
fi
