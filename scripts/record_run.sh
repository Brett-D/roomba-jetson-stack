#!/usr/bin/env bash
# Records the RAW sensor inputs that both SLAM pipelines need, so the same drive can be replayed
# offline through pipeline A and B (scripts/compare_pipelines.sh). Start the robot first
# (~/run_robot.sh mapping ...), then run this and drive. Only what the camera and the Roomba
# produce is recorded - no odometry, no odom/map TF, no maps - plus /cmd_vel for reference.
#
#   record_run.sh                  record until Ctrl-C
#   record_run.sh --duration 600   stop after 600 s
#   record_run.sh --zstd           zstd file compression (saves ~30-50 %, costs CPU while driving)
#   record_run.sh --name kitchen   ~/bags/robot_<timestamp>_kitchen
#
# Output: ~/bags/robot_<YYYYmmdd_HHMMSS>[_name]/ (mcap) with record_info.txt (rate, size).
# Prints the measured data rate (MB/min) and how many minutes the disk still holds.
DURATION=0; COMPRESS=(); NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --duration) DURATION=$2; shift 2 ;;
    --zstd) COMPRESS=(--compression-mode file --compression-format zstd); shift ;;
    --name) NAME="_$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done
source /opt/ros/jazzy/setup.bash
# Workspace messages (roomba_msgs/SensorState for /roomba/sensors) must be known to the recorder.
[ -f ~/roomba_ws/install/setup.bash ] && source ~/roomba_ws/install/setup.bash
# Raw inputs. Basalt and cuVSLAM: rectified mono stereo + camera_infos + raw IMU (200 Hz).
# RTAB-Map: left color + depth + left camera_info (+ IMU orientation, recomputed by imu_filter on
# replay). nvblox and floor_obstacles: depth + depth camera_info. EKF / watchdog: /wheel/odom +
# raw IMU. Camera and mount transforms: /tf_static. The rest is context for the comparison.
TOPICS=(
  /camera/left/image_rect_mono /camera/right/image_rect_mono
  /camera/left/camera_info /camera/right/camera_info
  /camera/left/image_rect_color
  /camera/depth/image_rect_raw /camera/depth/camera_info
  /camera/imu/data_raw
  /wheel/odom /roomba/sensors /roomba/mode /battery_state
  /tf_static
  /cmd_vel /cmd_vel_nav
)
if ! timeout 8 ros2 topic echo --once /camera/imu/data_raw > /dev/null 2>&1; then
  echo "No camera data (/camera/imu/data_raw). Start the robot first: ~/run_robot.sh mapping ..."
  exit 1
fi
timeout 10 ros2 topic echo --once /wheel/odom > /dev/null 2>&1 || \
  echo "WARNING: no /wheel/odom (Roomba asleep or off?) - recording without wheel odometry."

STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p ~/bags
OUT=$(readlink -f ~/bags)/robot_${STAMP}${NAME}
echo "Recording to $OUT (Ctrl-C to stop)"
# Job control: the recorder gets its own process group and a normal SIGINT disposition (a
# non-interactive shell would start it with SIGINT ignored); Ctrl-C reaches this script, whose
# trap stops the recorder cleanly so the mcap file is finalized.
# Started in the background (setsid/nohup), bash hands the recorder SIGINT *ignored* and Python
# then never installs its handler: the --duration stop and Ctrl-C did nothing and one run filled
# the disk. Reset SIGINT to default right before exec so the recorder handles it again.
set -m
python3 -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp(sys.argv[1], sys.argv[1:])' \
  ros2 bag record -s mcap "${COMPRESS[@]}" -o "$OUT" --topics "${TOPICS[@]}" > "${OUT}.log" 2>&1 &
REC=$!
set +m
STOP_AT=""
stop_rec() {   # SIGINT finalizes the mcap; escalate if the recorder doesn't exit
  [ -z "$STOP_AT" ] && STOP_AT=$(date +%s) && kill -INT $REC 2>/dev/null
}
trap stop_rec INT TERM
MIN_FREE=$((3 * 1000 * 1000 * 1000))   # stop with 3 GB left on the disk

START=$(date +%s.%N)
LAST_T=$START; LAST_B=0
while kill -0 $REC 2>/dev/null; do
  sleep 1
  NOW=$(date +%s.%N)
  ELAPSED=$(echo "$NOW - $START" | bc)
  if [ "$DURATION" != 0 ] && [ "$(echo "$ELAPSED >= $DURATION" | bc)" = 1 ]; then
    stop_rec
  fi
  if [ "$(df -B1 --output=avail "$(dirname "$OUT")" | tail -1)" -lt $MIN_FREE ] && [ -z "$STOP_AT" ]; then
    echo "  disk almost full: stopping"; stop_rec
  fi
  if [ -n "$STOP_AT" ]; then
    WAITED=$(( $(date +%s) - STOP_AT ))
    [ $WAITED -ge 15 ] && kill -TERM $REC 2>/dev/null
    [ $WAITED -ge 30 ] && { echo "  recorder ignored SIGINT/SIGTERM: killed (run ros2 bag reindex)"; kill -KILL $REC 2>/dev/null; }
  fi
  if [ "$(echo "$NOW - $LAST_T >= 15" | bc)" = 1 ]; then
    B=$(du -sb "$OUT" 2>/dev/null | cut -f1); B=${B:-0}
    RATE=$(echo "($B - $LAST_B) / ($NOW - $LAST_T) * 60 / 1000000" | bc -l)
    FREE=$(df -B1 --output=avail "$OUT" | tail -1)
    printf "  %4.0f s  %7.1f MB  %6.0f MB/min  (%.0f min of disk left)\n" "$ELAPSED" \
      "$(echo "$B / 1000000" | bc -l)" "$RATE" "$(echo "$FREE / ($RATE * 1000000 + 1)" | bc -l)"
    LAST_T=$NOW; LAST_B=$B
  fi
done
wait $REC 2>/dev/null
END=$(date +%s.%N)
B=$(du -sb "$OUT" 2>/dev/null | cut -f1); B=${B:-0}
DUR=$(echo "$END - $START" | bc -l)
RATE=$(echo "$B / $DUR * 60 / 1000000" | bc -l)
FREE=$(df -B1 --output=avail "$OUT" | tail -1)
mv "${OUT}.log" "$OUT/record.log" 2>/dev/null
{
  printf "bag: %s\nwall_duration_s: %.1f\nsize_mb: %.1f\nrate_mb_per_min: %.0f\n" "$OUT" "$DUR" \
    "$(echo "$B / 1000000" | bc -l)" "$RATE"
  printf "disk_free_gb: %.0f\nminutes_of_disk_left_at_this_rate: %.0f\n" "$(echo "$FREE / 1000000000" | bc -l)" \
    "$(echo "$FREE / ($RATE * 1000000 + 1)" | bc -l)"
  echo "compression: ${COMPRESS[*]:-none}"
  echo "topics:"
  ros2 bag info "$OUT" 2>/dev/null | sed -n '/Topic information/,$p' | sed 's/^/  /'
} > "$OUT/record_info.txt"
cat "$OUT/record_info.txt"
echo "Compare pipelines on it: ~/roomba_ws/scripts/compare_pipelines.sh $OUT"
