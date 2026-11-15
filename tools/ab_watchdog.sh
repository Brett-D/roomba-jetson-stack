#!/usr/bin/env bash
# CPU A/B of the Rust vslam_watchdog (roomba_rs) against the Python one (roomba_perception): replays
# DUR s of a VIO recording (tools/record_vio_replay.sh) from OFF s in, runs both nodes side by side
# on remapped outputs with the robot's parameters (MODE=ekf: publish_tf false as with fusion:=ekf,
# MODE=tf: publish_tf true) and reports CPU use, output counts and the last status of each. No VIO
# runs, so recoveries cannot reset anything. Bag playback drops IMU samples (best effort, depth 5),
# so the outputs only roughly agree here; tools/ab_watchdog_offline.py compares them exactly.
#   tools/ab_watchdog.sh /data/bags/vio_cuvslam_fresh_map8        (OFF=0 DUR=300 MODE=ekf)
source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
BAG=${1:?usage: ab_watchdog.sh VIO_BAG  (env OFF, DUR, MODE=ekf|tf, RS=rust binary)}
OFF=${OFF:-0}; DUR=${DUR:-300}; MODE=${MODE:-ekf}; LOG=${LOG:-$HOME/.ros/ab_watchdog}; mkdir -p "$LOG"
RS=${RS:-$(ros2 pkg prefix roomba_rs)/lib/roomba_rs/vslam_watchdog}
P="-p base_frame:=base_link -p max_speed_mps:=0.6 -p max_accel_mps2:=1.5 -p max_vertical_speed_mps:=0.3
   -p max_height_m:=1.0 -p wheel_odom_topic:=/wheel/odom -p planar:=true -p publish_tf:=$([ "$MODE" = ekf ] && echo false || echo true)"
R="-r /tf:=/X/tf -r /odom:=/X/odom -r /vslam_filter/vio:=/X/vio -r /vslam_filter/status:=/X/status"
ros2 run roomba_perception vslam_watchdog --ros-args -r __node:=py_watchdog $P ${R//X/py} > "$LOG/py.log" 2>&1 &
$RS --ros-args -r __node:=rs_watchdog $P ${R//X/rs} > "$LOG/rs.log" 2>&1 &
sleep 5
PYPID=$(pgrep -f 'lib/roomba_perception/vslam_watchdog --ros-args'); RSPID=$(pgrep -f "$(basename "$(dirname "$RS")")/vslam_watchdog --ros-args" | head -1)
echo "pids py $PYPID rs $RSPID"
python3 - "$MODE" "$((DUR + 4))" > "$LOG/collect.txt" 2>&1 <<'EOF' &
import sys, time, rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
rclpy.init()
n = Node('ab_watchdog_collect')
count, status = {'py': 0, 'rs': 0}, {}
topic = 'vio' if sys.argv[1] == 'ekf' else 'odom'
for k in count:
    n.create_subscription(Odometry, f'/{k}/{topic}', lambda m, k=k: count.__setitem__(k, count[k] + 1), 100)
    n.create_subscription(String, f'/{k}/status', lambda m, k=k: status.__setitem__(k, m.data), 10)
end = time.time() + float(sys.argv[2])
while time.time() < end:
    rclpy.spin_once(n, timeout_sec=0.2)
for k in count:
    print(f'{k}: {count[k]} poses on /{k}/{topic}; last status {status.get(k)}')
EOF
COL=$!
cpu() { awk '{print $14 + $15}' /proc/$1/stat; }
P0=$(cpu $PYPID); R0=$(cpu $RSPID); T0=$(date +%s.%N)
ros2 bag play "$BAG" --start-offset "$OFF" --playback-duration "$DUR" --topics /visual_slam/tracking/odometry \
  /visual_slam/status /camera/imu/data_raw /wheel/odom /tf_static > "$LOG/play.log" 2>&1
P1=$(cpu $PYPID); R1=$(cpu $RSPID); T1=$(date +%s.%N)
wait $COL
HZ=$(getconf CLK_TCK)
echo "CPU over replay (% of one core): python $(echo "($P1-$P0)/$HZ/($T1-$T0)*100" | bc -l | cut -c1-5)  rust $(echo "($R1-$R0)/$HZ/($T1-$T0)*100" | bc -l | cut -c1-5)"
echo "max RSS (MB): python $(awk '/VmHWM/ {print $2/1024}' /proc/$PYPID/status)  rust $(awk '/VmHWM/ {print $2/1024}' /proc/$RSPID/status)"
cat "$LOG/collect.txt"
kill -TERM $RSPID 2>/dev/null; kill -INT $PYPID 2>/dev/null; sleep 2