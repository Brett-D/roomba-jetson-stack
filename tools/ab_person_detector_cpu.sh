#!/usr/bin/env bash
# CPU / latency / memory A/B of person_detector at the real camera rate: plays DUR s of a raw
# recording from OFF s in (left color, registered depth, right mono, camera infos) through the
# Python node (roomba_perception) and then the Rust one (roomba_rs), each alone, with the robot's
# parameters (classes person/cat/dog, floor grid on). A helper latches TF map -> odom -> base_link ->
# camera (static), an RTAB-Map grid (the last one of MAP_BAG, tools/record_map_replay.sh) and ignore
# zones; a probe subscribes to the annotated JPEG (as the web app / Foxglove do) and times each frame
# from its camera_info's arrival to its detections. Reports CPU (% of one core), peak RSS, frames,
# latency and the nodes' own inference / total timing lines (logged every 30 s).
#   tools/ab_person_detector_cpu.sh RAW_BAG [MAP_BAG]     env: OFF=0 DUR=300 ROS_DOMAIN_ID=65
#     RS=rust binary (default: installed roomba_rs), ORDER="py rs" (rs0: Rust without its CUDA graph),
#     LOG=/data/rs_lanes/people/cpu
source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-65}
BAG=${1:?usage: ab_person_detector_cpu.sh RAW_BAG [MAP_BAG]}; MAPBAG=${2:-}
OFF=${OFF:-0}; DUR=${DUR:-300}; LOG=${LOG:-/data/rs_lanes/people/cpu}; mkdir -p "$LOG"
RS=${RS:-$(ros2 pkg prefix roomba_rs)/lib/roomba_rs/person_detector}
PY=$(ros2 pkg prefix roomba_perception)/lib/roomba_perception/person_detector
cat > "$LOG/params.yaml" <<'EOF'
/**:
  ros__parameters:
    classes: [0, 15, 16]
    require_localization: false
    floor_grid: true
EOF

python3 - "$BAG" "$MAPBAG" > "$LOG/world.log" 2>&1 <<'EOF' &
import json, sys, rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
def read(uri, topics, until_s=None):
    r = SequentialReader()
    r.open(StorageOptions(uri=uri, storage_id='mcap'), ConverterOptions('', ''))
    end = r.get_metadata().starting_time.nanoseconds + int(until_s * 1e9) if until_s else None
    while r.has_next():
        topic, data, t = r.read_next()
        if end and t > end:
            break
        if topic in topics:
            yield topic, data
transforms = [t for _, d in read(sys.argv[1], {'/tf_static'}, 10) for t in deserialize_message(d, TFMessage).transforms]
def tf(parent, child, x=0.0, z=0.0):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.x, t.transform.translation.z, t.transform.rotation.w = x, z, 1.0
    return t
transforms += [tf('map', 'odom'), tf('odom', 'base_link'), tf('base_link', 'camera_link', 0.153, 0.13)]
grid = None
if sys.argv[2]:
    for _, d in read(sys.argv[2], {'/rtabmap/map'}):
        grid = d
rclpy.init()
n = Node('person_ab_world')
latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
n.create_publisher(TFMessage, '/tf_static', QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)).publish(
    TFMessage(transforms=transforms))
if grid:
    m = deserialize_message(grid, OccupancyGrid)
    n.create_publisher(OccupancyGrid, '/rtabmap/map', latched).publish(m)
zones = {'frame': 'map', 'zones': [{'name': 'test', 'points': [[2, -1], [4, -1], [4, 1], [2, 1]]},
                                   {'name': 'nav', 'kind': 'keepout', 'points': [[0, 0], [1, 0], [1, 1]]}]}
n.create_publisher(String, '/perception/ignore_zones', latched).publish(String(data=json.dumps(zones)))
print(f'ready: {len(transforms)} static transforms, map {grid is not None}', flush=True)
rclpy.spin(n)
EOF
WORLD=$!
trap 'kill -INT $WORLD 2>/dev/null' EXIT
for i in $(seq 1 300); do grep -q ready "$LOG/world.log" && break; sleep 1; done   # reading the bags takes a while

PROBE=$(cat <<'EOF'
import json, signal, sys, time, rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage
from vision_msgs.msg import Detection2DArray
rclpy.init()
n = Node('person_ab_probe')
arrive, lat, counts = {}, [], {'info': 0, 'det': 0, 'jpeg': 0}
key = lambda h: h.stamp.sec * 1_000_000_000 + h.stamp.nanosec
def on_info(m):
    counts['info'] += 1
    arrive[key(m.header)] = time.monotonic()
def on_det(m):
    counts['det'] += 1
    t = arrive.pop(key(m.header), None)
    if t is not None:
        lat.append(1000 * (time.monotonic() - t))
n.create_subscription(CameraInfo, '/camera/left/camera_info', on_info, 50)
n.create_subscription(Detection2DArray, '/perception/person_detector/detections_2d', on_det, 50)
n.create_subscription(CompressedImage, '/perception/person_detector/annotated/compressed',
                      lambda m: counts.__setitem__('jpeg', counts['jpeg'] + 1), 5)
stop = []
signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
while not stop:
    rclpy.spin_once(n, timeout_sec=0.2)
lat.sort()
q = lambda f: round(lat[min(len(lat) - 1, int(f * len(lat)))], 1) if lat else None
print(json.dumps({**counts, 'latency_ms': {'median': q(0.5), 'p90': q(0.9), 'p99': q(0.99), 'max': q(1.0)}}))
EOF
)

cpu() { awk '{print $14 + $15}' "/proc/$1/stat"; }
HZ=$(getconf CLK_TCK)
for impl in ${ORDER:-py rs}; do
  BIN=$PY; [ "$impl" != py ] && BIN=$RS
  GRAPH=1; [ "$impl" = rs0 ] && GRAPH=0   # rs0: Rust without the CUDA graph
  PERSON_DETECTOR_CUDA_GRAPH=$GRAPH "$BIN" --ros-args -r __ns:=/perception -r __node:=person_detector --params-file "$LOG/params.yaml" > "$LOG/$impl.log" 2>&1 &
  PID=$!
  for i in $(seq 1 60); do grep -q "Loaded" "$LOG/$impl.log" && break; sleep 1; done
  python3 -c "$PROBE" > "$LOG/$impl.probe" 2>&1 &
  PROBE_PID=$!
  sleep 4
  C0=$(cpu $PID); T0=$(date +%s.%N)
  ros2 bag play "$BAG" --start-offset "$OFF" --playback-duration "$DUR" --read-ahead-queue-size 100 \
    --topics /camera/left/image_rect_color /camera/depth/image_rect_raw /camera/right/image_rect_mono \
    /camera/left/camera_info /camera/right/camera_info > "$LOG/$impl.play.log" 2>&1
  sleep 2
  C1=$(cpu $PID); T1=$(date +%s.%N)
  RSS=$(awk '/VmHWM/ {printf "%.0f", $2/1024}' "/proc/$PID/status")
  kill -TERM $PROBE_PID; sleep 2
  kill -TERM $PID; sleep 3; kill -KILL $PID 2>/dev/null
  echo "$impl: CPU $(echo "($C1-$C0)/$HZ/($T1-$T0)*100" | bc -l | cut -c1-5) % of one core over $(echo "$T1-$T0" | bc | cut -c1-6) s, peak RSS $RSS MB"
  echo "   probe: $(cat "$LOG/$impl.probe" | tail -1)"
  echo "   node:  $(grep -o '[0-9.]* Hz, inference [0-9.]* ms, total [0-9.]* ms/frame' "$LOG/$impl.log" | awk '{hz+=$1; inf+=$4; tot+=$7; n++} END {if (n) printf "%d stats lines, mean %.1f Hz, inference %.1f ms, total %.1f ms/frame", n, hz/n, inf/n, tot/n}')"
done
