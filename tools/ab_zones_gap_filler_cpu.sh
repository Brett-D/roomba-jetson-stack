#!/usr/bin/env bash
# CPU / memory A/B of the Rust ignore_zones and map_gap_filler (roomba_rs) against the Python ones
# (roomba_perception): runs all four side by side on remapped outputs, replays DUR s of a map
# recording (tools/record_map_replay.sh: /rtabmap/map, /rtabmap/mapPath, /tf, /tf_static) from OFF s
# in at real rate, sends both zone nodes the same explorer-style bump mark (keep-out, 60 s ttl) every
# 30 s, and reports CPU use, peak RSS, output counts and whether the last filled maps are identical.
# Both zone nodes start from the same zones file (ZONES, default: 12 generated zones). Uses ROS
# domain 62 unless ROS_DOMAIN_ID is set. tools/ab_*_offline.py compare the outputs exactly.
#   tools/ab_zones_gap_filler_cpu.sh /data/bags/map_replay        (OFF=300 DUR=600)
source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-62}
BAG=${1:?usage: ab_zones_gap_filler_cpu.sh MAP_BAG  (env OFF, DUR, ZONES, RS=dir with the rust binaries)}
OFF=${OFF:-300}; DUR=${DUR:-600}; LOG=${LOG:-$HOME/.ros/ab_zones_gap_filler}; mkdir -p "$LOG"
RS=${RS:-$(ros2 pkg prefix roomba_rs)/lib/roomba_rs}
if [ -n "$ZONES" ]; then cp "$ZONES" "$LOG/zones.json"; else python3 - "$LOG/zones.json" <<'EOF'
import json, math, sys
zones = []
for k in range(12):
    cx, cy, r = (k % 4) * 3.0 - 4.5, (k // 4) * 3.0 - 3.0, 0.4 + 0.1 * k
    pts = [[round(cx + r * math.cos(a * math.pi / 3), 3), round(cy + r * math.sin(a * math.pi / 3), 3)] for a in range(6)]
    zones.append({'name': f'zone_{k}', 'points': pts, 'kind': 'keepout' if k % 2 else 'exclusion'})
json.dump({'frame': 'map', 'zones': zones}, open(sys.argv[1], 'w'), indent=1)
EOF
fi
cp "$LOG/zones.json" "$LOG/py.zones.json"; cp "$LOG/zones.json" "$LOG/rs.zones.json"
Z="-r /perception/ignore_zones:=/X/zones -r /perception/ignore_zones/markers:=/X/markers -r /keepout_filter_mask:=/X/mask
   -r /costmap_filter_info:=/X/filter_info"
ros2 run roomba_perception ignore_zones --ros-args -r __node:=py_zones -p zones_file:=$LOG/py.zones.json ${Z//X/py} > "$LOG/py_zones.log" 2>&1 &
"$RS/ignore_zones" --ros-args -r __node:=rs_zones -p zones_file:=$LOG/rs.zones.json ${Z//X/rs} > "$LOG/rs_zones.log" 2>&1 &
ros2 run roomba_perception map_gap_filler --ros-args -r __node:=py_gap -p output_topic:=/py/map_filled > "$LOG/py_gap.log" 2>&1 &
"$RS/map_gap_filler" --ros-args -r __node:=rs_gap -p output_topic:=/rs/map_filled > "$LOG/rs_gap.log" 2>&1 &
sleep 6
pid() { pgrep -f "$1" | head -1; }
PYZ=$(pid 'roomba_perception/ignore_zones --ros-args'); RSZ=$(pid "$RS/ignore_zones --ros-args")
PYG=$(pid 'roomba_perception/map_gap_filler --ros-args'); RSG=$(pid "$RS/map_gap_filler --ros-args")
echo "pids: ignore_zones py $PYZ rs $RSZ, map_gap_filler py $PYG rs $RSG"
python3 - "$((DUR + 4))" > "$LOG/collect.txt" 2>&1 <<'EOF' &
import json, sys, time, rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
rclpy.init()
n = Node('ab_zones_gap_filler_collect')
latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
count, last = {}, {}
def keep(key):
    def cb(m):
        count[key] = count.get(key, 0) + 1
        last[key] = m
    return cb
for side in ('py', 'rs'):
    n.create_subscription(OccupancyGrid, f'/{side}/map_filled', keep(f'{side} map_filled'), latched)
    n.create_subscription(OccupancyGrid, f'/{side}/mask', keep(f'{side} mask'), latched)
    n.create_subscription(String, f'/{side}/zones', keep(f'{side} zones'), latched)
    n.create_subscription(MarkerArray, f'/{side}/markers', keep(f'{side} markers'), 10)
cmd = {side: n.create_publisher(String, f'/{side}_zones/command', 10) for side in ('py', 'rs')}
end, next_bump, k = time.time() + float(sys.argv[1]), time.time() + 5, 0
while time.time() < end:
    rclpy.spin_once(n, timeout_sec=0.2)
    if time.time() > next_bump:
        k += 1
        pts = [[k * 0.1, 0.0], [k * 0.1 + 0.12, 0.0], [k * 0.1 + 0.12, 0.5], [k * 0.1, 0.5]]
        text = json.dumps({'op': 'add', 'kind': 'keepout', 'name': f'bump_{k}', 'points': pts, 'margin': 0.15, 'ttl_s': 60.0})
        for p in cmd.values():
            p.publish(String(data=text))
        next_bump += 30
for key in sorted(count):
    print(f'{key}: {count[key]} messages')
same = lambda key: 'py' + key in last and 'rs' + key in last  # noqa: E731
if same(' map_filled'):
    a, b = last['py map_filled'], last['rs map_filled']
    print('last filled maps identical:', a.info == b.info and list(a.data) == list(b.data), f'({a.info.width} x {a.info.height})')
if same(' mask'):
    a, b = last['py mask'], last['rs mask']
    print('last keep-out masks identical:', a.info == b.info and list(a.data) == list(b.data), f'({a.info.width} x {a.info.height})')
print(f'{k} bump marks sent to each')
EOF
COL=$!
cpu() { awk '{print $14 + $15}' /proc/$1/stat; }
declare -A C0
for p in $PYZ $RSZ $PYG $RSG; do C0[$p]=$(cpu $p); done; T0=$(date +%s.%N)
ros2 bag play "$BAG" --start-offset "$OFF" --playback-duration "$DUR" --topics /rtabmap/map /rtabmap/mapPath /tf /tf_static \
  > "$LOG/play.log" 2>&1
T1=$(date +%s.%N)
HZ=$(getconf CLK_TCK)
pct() { echo "($(cpu $1)-${C0[$1]})/$HZ/($T1-$T0)*100" | bc -l | cut -c1-5; }
rss() { awk '/VmHWM/ {printf "%.1f", $2/1024}' /proc/$1/status; }
echo "CPU over replay (% of one core) / peak RSS (MB):"
echo "  ignore_zones    python $(pct $PYZ) / $(rss $PYZ)   rust $(pct $RSZ) / $(rss $RSZ)"
echo "  map_gap_filler  python $(pct $PYG) / $(rss $PYG)   rust $(pct $RSG) / $(rss $RSG)"
wait $COL
cat "$LOG/collect.txt"
kill -TERM $RSZ $RSG 2>/dev/null; kill -INT $PYZ $PYG 2>/dev/null; sleep 3
kill -KILL $PYZ $RSZ $PYG $RSG 2>/dev/null; true
