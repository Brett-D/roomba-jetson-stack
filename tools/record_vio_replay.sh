#!/usr/bin/env bash
# Replays a raw recording (scripts/record_run.sh) through the VIO pipeline and records what
# vslam_watchdog consumes (VIO odometry and status, raw IMU, wheel odometry, TF) plus the live
# watchdog's outputs, as input for tools/ab_watchdog_offline.py. Sim time, camera and Roomba
# drivers off, no Nav2; RTAB-Map maps into $DB (never ~/maps/vslam.db).
#   tools/record_vio_replay.sh BAG OUT_BAG [robot.launch.py args]   (default: odometry:=cuvslam fusion:=ekf)
#   env: OFF=start offset s, DUR=seconds (default: whole bag), RATE=1.0, DB=/data/maps/compare/vio_replay.db
source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
BAG=${1:?usage: record_vio_replay.sh BAG OUT_BAG [launch args]}; OUT=${2:?OUT_BAG}; shift 2
[ $# -gt 0 ] || set -- odometry:=cuvslam fusion:=ekf obstacles:=nvblox
DB=${DB:-/data/maps/compare/vio_replay.db}
case "$(readlink -f "$(dirname "$DB")")/$(basename "$DB")" in "$(readlink -f ~/maps)"/*) echo "refusing to map into ~/maps"; exit 1 ;; esac
if pgrep -f "ros2 launch roomba_bringup" > /dev/null || pgrep -f "ros2 bag play" > /dev/null; then
  echo "The robot stack or a replay is running: stop it first (~/stop_slam.sh)."; exit 1
fi
mkdir -p "$(dirname "$DB")" "$(dirname "$OUT")"
# The bag's /tf_static (camera frames) is published once at its start, so a replay from an offset
# would have none: keep publishing it (latched) for the whole run.
python3 - "$BAG" > "$OUT.tf_static.log" 2>&1 <<'EOF' &
import sys, rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from tf2_msgs.msg import TFMessage
r = SequentialReader()
r.open(StorageOptions(uri=sys.argv[1], storage_id='mcap'), ConverterOptions('', ''))
transforms, end = [], r.get_metadata().starting_time.nanoseconds + int(10e9)
while r.has_next():  # unfiltered: a topic filter scans the whole (index-less) recording per message
    topic, data, t_ns = r.read_next()
    if t_ns > end:  # /tf_static is published at the start
        break
    if topic == '/tf_static':
        transforms += deserialize_message(data, TFMessage).transforms
rclpy.init()
n = Node('bag_tf_static')
n.create_publisher(TFMessage, '/tf_static', QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)).publish(
    TFMessage(transforms=transforms))
print(f'ready: {len(transforms)} static transforms', flush=True)
rclpy.spin(n)
EOF
TFPID=$!
setsid ros2 launch roomba_bringup robot.launch.py mode:=mapping database_path:="$DB" camera:=false driver:=false \
  use_sim_time:=true people:=false foxglove:=false web:=false nav:=false "$@" > "$OUT.launch.log" 2>&1 < /dev/null &
for i in $(seq 1 75); do ros2 node list 2>/dev/null | grep -qx /vslam_watchdog && break; sleep 2; done
sleep 10  # VIO node and RTAB-Map bring-up
ros2 bag record -s mcap -o "$OUT" --topics /visual_slam/tracking/odometry /visual_slam/status /camera/imu/data_raw \
  /wheel/odom /tf_static /tf /odom /vslam_filter/status /vslam_filter/vio /rosout > "$OUT.record.log" 2>&1 &
sleep 4
for i in $(seq 1 90); do grep -q ready "$OUT.tf_static.log" && break; sleep 2; done
PLAY=(--clock 50 --rate "${RATE:-1.0}" --read-ahead-queue-size 2000)
[ -n "$OFF" ] && PLAY+=(--start-offset "$OFF"); [ -n "$DUR" ] && PLAY+=(--playback-duration "$DUR")
ros2 bag play "$BAG" "${PLAY[@]}" --topics /camera/left/image_rect_mono /camera/right/image_rect_mono \
  /camera/left/camera_info /camera/right/camera_info /camera/left/image_rect_color /camera/depth/image_rect_raw \
  /camera/depth/camera_info /camera/imu/data_raw /wheel/odom /roomba/sensors /roomba/mode /battery_state /tf_static \
  > "$OUT.play.log" 2>&1
sleep 5
# SIGTERM: a recorder started in the background of a script ignores SIGINT.
pkill -TERM -f "ros2 bag record -s mcap -o $OUT"; sleep 5
~/stop_slam.sh
kill -TERM $TFPID 2>/dev/null
ros2 bag info "$OUT" | grep -E "Duration|Topic:"