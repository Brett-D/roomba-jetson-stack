#!/usr/bin/env bash
# Replays a raw recording (scripts/record_run.sh) through the whole robot stack - VIO, RTAB-Map,
# Nav2 - with sim time and no camera or Roomba driver, starts the explorer from the bag's start
# point and steers it with scripted commands (start, stop, start, home, start), recording what the
# explorer and nav_coach consume and publish. Checks that they work against the real Nav2
# (goals accepted, feedback, results, cancels) and provides real costmaps for
# tools/ab_explorer_offline.py and tools/ab_nav_coach_offline.py. Nothing drives (the bag's motion
# is replayed), so goals succeed only if the recorded drive passes them.
# RTAB-Map maps into $DB (never ~/maps). Stops only the processes it started.
#   tools/ab_explorer_live.sh BAG OUT_BAG [robot.launch.py args]   (default: odometry:=cuvslam fusion:=ekf
#                                                                    obstacles:=nvblox; rust_nodes:=true)
#   env: OFF=start offset s, DUR=seconds (default 900), RATE=1.0, DB=/data/maps/compare/explorer_replay.db,
#        OVERLAY=install dir to source on top, ROS_DOMAIN_ID (default 64), CMDS="40:start 400:stop 430:start 640:home 720:start" (s after play start)
source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
[ -n "$OVERLAY" ] && source "$OVERLAY/setup.bash"   # e.g. a worktree install with the nodes under test
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-64}
BAG=${1:?usage: ab_explorer_live.sh BAG OUT_BAG [launch args]}; OUT=${2:?OUT_BAG}; shift 2
[ $# -gt 0 ] || set -- odometry:=cuvslam fusion:=ekf obstacles:=nvblox
DB=${DB:-/data/maps/compare/explorer_replay.db}; DUR=${DUR:-900}
CMDS=${CMDS:-"40:start 400:stop 430:start 640:home 720:start"}
case "$(readlink -f "$(dirname "$DB")")/$(basename "$DB")" in "$(readlink -f ~/maps)"/*) echo "refusing to map into ~/maps"; exit 1 ;; esac
mkdir -p "$(dirname "$DB")" "$(dirname "$OUT")"
PIDS=()
stop() {  # each helper runs in its own session: stop exactly those process groups
  for p in "${PIDS[@]}"; do kill -INT -- "-$p" 2>/dev/null; done; sleep 10
  for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done; sleep 5
  for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; done
}
trap stop EXIT
# The bag's /tf_static is published once at its start: keep publishing it (latched) for the whole
# run, and /roomba/mode "full" (the driver's latched mode) so the explorer may drive.
setsid python3 - "$BAG" > "$OUT.latched.log" 2>&1 <<'EOF' &
import sys, rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from std_msgs.msg import String
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
n = Node('bag_latched')
latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
n.create_publisher(TFMessage, '/tf_static', latched).publish(TFMessage(transforms=transforms))
n.create_publisher(String, '/roomba/mode', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)).publish(
    String(data='full'))
print(f'ready: {len(transforms)} static transforms', flush=True)
rclpy.spin(n)
EOF
PIDS+=($!)
for i in $(seq 1 90); do grep -q ready "$OUT.latched.log" && break; sleep 2; done   # reading the bag takes a while
setsid ros2 launch roomba_bringup robot.launch.py mode:=mapping database_path:="$DB" camera:=false driver:=false \
  use_sim_time:=true people:=false foxglove:=false web:=false unstuck_skill:=false goal_ranking:=rule "$@" \
  > "$OUT.launch.log" 2>&1 < /dev/null &   # the classical explorer: the Python one has no learned parts
PIDS+=($!)
# Play as soon as the odometry nodes are up: Nav2 activation fails if odom -> base_link does not
# appear within ~25 s of the launch.
for i in $(seq 1 75); do ros2 node list 2>/dev/null | grep -q ekf_filter_node && break; sleep 2; done
sleep 3
setsid ros2 bag record -s mcap --include-hidden-topics -o "$OUT" --topics /global_costmap/costmap /tf /tf_static \
  /roomba/sensors /roomba/mode /battery_state /vslam_filter/status /explorer/status /explorer/command \
  /perception/ignore_zones/command /nav_coach/instruction /navigate_to_pose/_action/status \
  /navigate_to_pose/_action/feedback /cmd_vel_nav /cmd_vel /collision_monitor_state /perception/obstacles /rosout \
  > "$OUT.record.log" 2>&1 < /dev/null &
PIDS+=($!)
REC=$!
sleep 2
# Commands from one long-lived publisher (like the web app): a fresh `ros2 topic pub --once` loses
# its message when the explorer has not matched it yet.
setsid python3 - "$CMDS" > "$OUT.commands.log" 2>&1 <<'EOF' &
import sys, time, rclpy
from rclpy.node import Node
from std_msgs.msg import String
rclpy.init()
n = Node('explorer_commands')
pub = n.create_publisher(String, '/explorer/command', 10)
t0 = time.time()
for item in sys.argv[1].split():
    at, cmd = item.split(':', 1)
    while time.time() - t0 < float(at):
        rclpy.spin_once(n, timeout_sec=0.2)
    pub.publish(String(data=cmd))
    print(time.strftime('%T'), 'command', cmd, 'subscribers', pub.get_subscription_count(), flush=True)
rclpy.spin(n)
EOF
PIDS+=($!)
PLAY=(--clock 50 --rate "${RATE:-1.0}" --read-ahead-queue-size 2000 --playback-duration "$DUR")
[ -n "$OFF" ] && PLAY+=(--start-offset "$OFF")
ros2 bag play "$BAG" "${PLAY[@]}" --topics /camera/left/image_rect_mono /camera/right/image_rect_mono \
  /camera/left/camera_info /camera/right/camera_info /camera/left/image_rect_color /camera/depth/image_rect_raw \
  /camera/depth/camera_info /camera/imu/data_raw /wheel/odom /roomba/sensors /battery_state /tf_static \
  > "$OUT.play.log" 2>&1
sleep 5
kill -TERM -- "-$REC" 2>/dev/null; sleep 6   # recorder first (it ignores SIGINT here), so the bag is closed cleanly
stop; trap - EXIT
ros2 bag info "$OUT" | grep -E "Duration|Topic:"
