#!/usr/bin/env bash
# CPU A/B of the Rust explorer and nav_coach (roomba_rs) against the Python ones (roomba_perception):
# replays DUR s of a recording made by tools/ab_explorer_live.sh (global costmap, TF, sensors, battery,
# velocity commands, collision monitor, obstacle clouds, navigate_to_pose status and feedback) at
# its own rate and runs both pairs side by side with the robot's parameters on renamed nodes. Both
# explorers explore: a fake NavigateToPose server per explorer accepts every goal and reports
# success after GOAL_S s, so each looks for a new frontier every few seconds. Reports CPU use
# (% of one core) and peak RSS per node, and the explorers' last status. Both explorers get "start"
# every 20 s; OFF should be past the recording's first map -> base_link TF (~300 s into one made by
# ab_explorer_live.sh), or they cannot start.
#   tools/ab_explorer_cpu.sh RECORDING        (env OFF=300 DUR=400 GOAL_S=3 ROS_DOMAIN_ID=64, RS=dir of the Rust nodes)
source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
[ -n "$OVERLAY" ] && source "$OVERLAY/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-64}
BAG=${1:?usage: ab_explorer_cpu.sh RECORDING}
OFF=${OFF:-300}; DUR=${DUR:-400}; GOAL_S=${GOAL_S:-3}; LOG=${LOG:-$HOME/.ros/ab_explorer_cpu}; mkdir -p "$LOG"
RS=${RS:-$(ros2 pkg prefix roomba_rs)/lib/roomba_rs}
PY=$(ros2 pkg prefix roomba_perception)/lib/roomba_perception
EXP="-p base_frame:=base_link -p low_battery_fraction:=0.2 -p bump_mark_width_m:=0.5 -p bump_light_signal:=200"
COACH="-p base_frame:=base_link -p robot_mode:=true"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done; sleep 2; for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null; done; }
trap cleanup EXIT
# Fake NavigateToPose servers for /py/navigate_to_pose and /rs/navigate_to_pose.
python3 - "$GOAL_S" > "$LOG/server.log" 2>&1 <<'EOF' &
import sys, time, rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
rclpy.init()
n = Node('fake_navigate_to_pose')
count = {}
def make(ns):
    def execute(handle):
        count[ns] = count.get(ns, 0) + 1
        end = time.time() + float(sys.argv[1])
        while time.time() < end:
            if handle.is_cancel_requested:
                handle.canceled()
                return NavigateToPose.Result()
            time.sleep(0.05)
        handle.succeed()
        return NavigateToPose.Result()
    return ActionServer(n, NavigateToPose, f'/{ns}/navigate_to_pose', execute,
                        cancel_callback=lambda h: rclpy.action.CancelResponse.ACCEPT)
servers = [make('py'), make('rs')]
ex = MultiThreadedExecutor(4)
ex.add_node(n)
try:
    ex.spin()
finally:
    print('goals', count, flush=True)
EOF
PIDS+=($!)
"$PY/explorer" --ros-args -r __node:=py_explorer -r navigate_to_pose:=/py/navigate_to_pose \
  -r /perception/ignore_zones/command:=/py/zones $EXP > "$LOG/py_explorer.log" 2>&1 &
PIDS+=($!); PYE=$!
"$RS/explorer" --ros-args -r __node:=rs_explorer -r navigate_to_pose:=/rs/navigate_to_pose \
  -r /perception/ignore_zones/command:=/rs/zones $EXP > "$LOG/rs_explorer.log" 2>&1 &
PIDS+=($!); RSE=$!
"$PY/nav_coach" --ros-args -r __node:=py_nav_coach -r /nav_coach/instruction:=/py/instruction $COACH \
  > "$LOG/py_nav_coach.log" 2>&1 &
PIDS+=($!); PYC=$!
"$RS/nav_coach" --ros-args -r __node:=rs_nav_coach -r /nav_coach/instruction:=/rs/instruction $COACH \
  > "$LOG/rs_nav_coach.log" 2>&1 &
PIDS+=($!); RSC=$!
sleep 5
# Status / instruction collector, and "start" for both explorers once the recording's map is in.
python3 - "$((DUR + 2))" > "$LOG/collect.txt" 2>&1 <<'EOF' &
import sys, time, rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
rclpy.init()
n = Node('ab_explorer_collect')
status, instr = {}, {'py': 0, 'rs': 0}
latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
for k in ('py', 'rs'):
    n.create_subscription(String, f'/{k}_explorer/status', lambda m, k=k: status.__setitem__(k, m.data), latched)
    n.create_subscription(String, f'/{k}/instruction', lambda m, k=k: instr.__setitem__(k, instr[k] + 1), 10)
cmd = {k: n.create_publisher(String, f'/{k}_explorer/command', 10) for k in ('py', 'rs')}
# The driver's latched mode (the recording's single message goes out before the player has matched).
n.create_publisher(String, '/roomba/mode', latched).publish(String(data='full'))
t0, last_start = time.time(), 0.0
end = t0 + float(sys.argv[1])
while time.time() < end:
    rclpy.spin_once(n, timeout_sec=0.2)
    # "start" every 20 s: the recording has no map -> base TF for its first ~40 s, and an explorer
    # that finished starts over (already exploring: ignored).
    if time.time() - last_start > 20:
        for p in cmd.values():
            p.publish(String(data='start'))
        last_start = time.time()
for k in ('py', 'rs'):
    print(f'{k}: {instr[k]} instructions; explorer status {status.get(k)}')
EOF
COL=$!
cpu() { awk '{print $14 + $15}' /proc/$1/stat; }
P0=$(cpu $PYE); R0=$(cpu $RSE); PC0=$(cpu $PYC); RC0=$(cpu $RSC); T0=$(date +%s.%N)
ros2 bag play "$BAG" --start-offset "$OFF" --playback-duration "$DUR" --read-ahead-queue-size 2000 --topics \
  /global_costmap/costmap /tf /tf_static /roomba/sensors /battery_state /vslam_filter/status \
  /cmd_vel_nav /cmd_vel /collision_monitor_state /perception/obstacles /navigate_to_pose/_action/status \
  /navigate_to_pose/_action/feedback > "$LOG/play.log" 2>&1
P1=$(cpu $PYE); R1=$(cpu $RSE); PC1=$(cpu $PYC); RC1=$(cpu $RSC); T1=$(date +%s.%N)
wait $COL
HZ=$(getconf CLK_TCK)
pct() { echo "($2-$1)/$HZ/($T1-$T0)*100" | bc -l | cut -c1-5; }
rss() { awk '/VmHWM/ {printf "%.1f", $2/1024}' /proc/$1/status; }
echo "replayed $(echo "$T1-$T0" | bc | cut -c1-5) s of $BAG from ${OFF} s"
echo "CPU (% of one core): explorer python $(pct $P0 $P1)  rust $(pct $R0 $R1);  nav_coach python $(pct $PC0 $PC1)  rust $(pct $RC0 $RC1)"
echo "peak RSS (MB):       explorer python $(rss $PYE)  rust $(rss $RSE);  nav_coach python $(rss $PYC)  rust $(rss $RSC)"
cat "$LOG/collect.txt"
