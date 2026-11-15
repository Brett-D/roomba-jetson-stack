#!/usr/bin/env bash
# Replays a raw recording (scripts/record_run.sh) through the mapping pipeline and records RTAB-Map's
# grid and path plus TF, as input for tools/ab_map_gap_filler_offline.py (and the CPU A/B,
# tools/ab_map_gap_filler.sh). Sim time, camera and Roomba drivers off, no Nav2; RTAB-Map maps into
# $DB (never ~/maps). Runs in its own ROS domain and stops only the processes it started.
#   tools/record_map_replay.sh BAG OUT_BAG [robot.launch.py args]   (default: odometry:=cuvslam fusion:=ekf)
#   env: OFF=start offset s, DUR=seconds (default: whole bag), RATE=1.0, DB=/data/maps/compare/map_replay.db,
#        ROS_DOMAIN_ID (default 62)
source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-62}
BAG=${1:?usage: record_map_replay.sh BAG OUT_BAG [launch args]}; OUT=${2:?OUT_BAG}; shift 2
[ $# -gt 0 ] || set -- odometry:=cuvslam fusion:=ekf obstacles:=nvblox
DB=${DB:-/data/maps/compare/map_replay.db}
case "$(readlink -f "$(dirname "$DB")")/$(basename "$DB")" in "$(readlink -f ~/maps)"/*) echo "refusing to map into ~/maps"; exit 1 ;; esac
mkdir -p "$(dirname "$DB")" "$(dirname "$OUT")"
PIDS=()
stop() {  # each helper runs in its own session: stop exactly those process groups
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done; sleep 8
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done
}
trap stop EXIT
# The bag's /tf_static is published once at its start: keep publishing it (latched) for the whole run.
setsid python3 - "$BAG" > "$OUT.tf_static.log" 2>&1 <<'EOF' &
import sys, rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from tf2_msgs.msg import TFMessage
r = SequentialReader()
r.open(StorageOptions(uri=sys.argv[1], storage_id='mcap'), ConverterOptions('', ''))
transforms, end = [], r.get_metadata().starting_time.nanoseconds + int(10e9)
while r.has_next():
    topic, data, t_ns = r.read_next()
    if t_ns > end:
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
PIDS+=($!)
setsid ros2 launch roomba_bringup robot.launch.py mode:=mapping database_path:="$DB" camera:=false driver:=false \
  use_sim_time:=true people:=false foxglove:=false web:=false nav:=false "$@" > "$OUT.launch.log" 2>&1 < /dev/null &
PIDS+=($!)
for i in $(seq 1 75); do ros2 node list 2>/dev/null | grep -q rtabmap && break; sleep 2; done
sleep 15  # VIO node and RTAB-Map bring-up
setsid ros2 bag record -s mcap -o "$OUT" --topics /rtabmap/map /rtabmap/mapPath /tf /tf_static \
  > "$OUT.record.log" 2>&1 < /dev/null &
PIDS+=($!)
sleep 4
for i in $(seq 1 90); do grep -q ready "$OUT.tf_static.log" && break; sleep 2; done
PLAY=(--clock 50 --rate "${RATE:-1.0}" --read-ahead-queue-size 2000)
[ -n "$OFF" ] && PLAY+=(--start-offset "$OFF"); [ -n "$DUR" ] && PLAY+=(--playback-duration "$DUR")
ros2 bag play "$BAG" "${PLAY[@]}" --topics /camera/left/image_rect_mono /camera/right/image_rect_mono \
  /camera/left/camera_info /camera/right/camera_info /camera/left/image_rect_color /camera/depth/image_rect_raw \
  /camera/depth/camera_info /camera/imu/data_raw /wheel/odom /roomba/sensors /roomba/mode /battery_state /tf_static \
  > "$OUT.play.log" 2>&1
sleep 8
kill -TERM -- "-${PIDS[2]}" 2>/dev/null; sleep 5   # recorder first, so the bag is closed cleanly
stop; trap - EXIT
ros2 bag info "$OUT" | grep -E "Duration|Topic:"
