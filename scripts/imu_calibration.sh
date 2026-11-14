#!/usr/bin/env bash
# Overnight IMU noise calibration for cuVSLAM.
# Put the camera where it cannot be bumped (no desk you'll lean on), then run:
#   ~/imu_calibration.sh        # 10 min warm-up + 3 h recording
#   ~/imu_calibration.sh 2      # 10 min warm-up + 2 h recording
# It stops SLAM, records, analyses, writes ~/.config/roomba/imu_noise.yaml, then restarts SLAM.
HOURS=${1:-3}
if [ -z "$IMU_CAL_CHILD" ]; then
  IMU_CAL_CHILD=1 setsid nohup "$0" "$@" > ~/imu_calibration.log 2>&1 < /dev/null &
  echo "IMU calibration started: 10 min warm-up + ${HOURS} h. Do not touch the camera."
  echo "Progress: tail -f ~/imu_calibration.log"
  exit 0
fi
source /opt/ros/jazzy/setup.bash
source ~/roomba_ws/install/setup.bash
~/stop_slam.sh
ros2 launch mynteye_d1000_driver d1000.launch.py &
DRIVER=$!
sleep 10
ros2 run roomba_perception imu_calibrate --ros-args -p duration_s:=$((HOURS * 3600)).0
RC=$?
kill -INT $DRIVER; sleep 5; pkill -f 'mynteye_d1000_driver/d1000_node'
echo "Calibration finished (exit $RC). Restarting SLAM."
setsid nohup ~/run_vslam.sh > ~/vslam.log 2>&1 < /dev/null &
