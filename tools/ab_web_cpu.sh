#!/usr/bin/env bash
# CPU and peak memory of the web app: roomba_web's server.py and the Rust web_server (roomba_rs)
# side by side on the same live inputs - a recorded drive played at real rate (camera preview,
# battery, Roomba sensors and mode; VIO TF, status and filter status from a VIO replay recording)
# plus tools/ab_web_load.py (growing map, path, plan, costmap, map -> odom TF, and a browser-like
# client per server: websocket, MJPEG stream, fetches on version changes). Outputs remapped like
# tools/ab_web.sh; test HOME for the password.
#   tools/ab_web_cpu.sh OUT_DIR    env: DUR=330 (measured s), WARM=40, OFFSET=300 (s into the bags),
#                                  ROBOT_BAG, VIO_BAG, PY_PORT, RS_PORT, RS_BIN, ROS_DOMAIN_ID (63)
OUT=${1:?usage: ab_web_cpu.sh OUT_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
[ -f "$HERE/../install/setup.bash" ] && source "$HERE/../install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-63}
DUR=${DUR:-330}; WARM=${WARM:-40}; OFFSET=${OFFSET:-300}
ROBOT_BAG=${ROBOT_BAG:-/data/bags/robot_20261114_145937_fresh_map8}; VIO_BAG=${VIO_BAG:-/data/bags/vio_cuvslam_fresh_map8}
PY_PORT=${PY_PORT:-18081}; RS_PORT=${RS_PORT:-18082}
RS_BIN=${RS_BIN:-$(ros2 pkg prefix roomba_rs)/lib/roomba_rs/web_server}
PY_BIN=$(ros2 pkg prefix roomba_web)/lib/roomba_web/web_server
mkdir -p "$OUT/home"
export AB_PASSWORD=${AB_PASSWORD:-ab-test-password-1}
HOME="$OUT/home" python3 -c "import os; from roomba_web.auth import set_password; set_password(os.environ['AB_PASSWORD'])"
remaps() {
  echo -r __node:=web_server_$1 -r /cmd_vel:=/$1/cmd_vel -r /cmd_vel_nav:=/$1/cmd_vel_nav \
    -r /perception/ignore_zones/command:=/$1/zone_command -r /explorer/command:=/$1/explorer_command \
    -r /perception/person_detector/floor_grid:=/$1/floor_grid -r navigate_to_pose:=/$1/navigate_to_pose \
    -r /roomba/set_mode:=/$1/set_mode
}
PIDS=()
stop() { for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; done; sleep 3
         for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; done; }
trap stop EXIT
HOME="$OUT/home" setsid python3 "$PY_BIN" --ros-args -p port:=$PY_PORT $(remaps py) > "$OUT/py_server.log" 2>&1 < /dev/null &
PY_PID=$!; PIDS+=($PY_PID)
HOME="$OUT/home" setsid "$RS_BIN" --ros-args -p port:=$RS_PORT $(remaps rs) > "$OUT/rs_server.log" 2>&1 < /dev/null &
RS_PID=$!; PIDS+=($RS_PID)
setsid ros2 bag play "$ROBOT_BAG" --start-offset "$OFFSET" --read-ahead-queue-size 1000 --topics /camera/left/image_rect_mono \
  /battery_state /roomba/sensors /roomba/mode /tf_static > "$OUT/play_robot.log" 2>&1 < /dev/null &
PIDS+=($!)
setsid ros2 bag play "$VIO_BAG" --start-offset "$OFFSET" --read-ahead-queue-size 1000 --topics /tf /tf_static \
  /visual_slam/status /vslam_filter/status > "$OUT/play_vio.log" 2>&1 < /dev/null &
PIDS+=($!)
# wait until both recordings are playing (opening an unindexed mcap takes a while)
for i in $(seq 1 120); do
  timeout 3 ros2 topic echo --once /battery_state > /dev/null 2>&1 && timeout 3 ros2 topic echo --once /visual_slam/status > /dev/null 2>&1 && break
  sleep 2
done
echo "bags playing after ~$((i * 5)) s"
setsid python3 -u "$HERE/ab_web_load.py" --ports $PY_PORT $RS_PORT --duration $((WARM + DUR + 5)) > "$OUT/load.log" 2>&1 < /dev/null &
LOAD=$!; PIDS+=($LOAD)
sleep "$WARM"
ticks() { awk '{print $14 + $15}' "/proc/$1/stat"; }
T0=$(date +%s.%N); PY0=$(ticks $PY_PID); RS0=$(ticks $RS_PID)
sleep "$DUR"
T1=$(date +%s.%N); PY1=$(ticks $PY_PID); RS1=$(ticks $RS_PID)
hwm() { awk '/VmHWM/ {print $2}' "/proc/$1/status"; }
rss() { awk '/VmRSS/ {print $2}' "/proc/$1/status"; }
PYH=$(hwm $PY_PID); RSH=$(hwm $RS_PID); PYR=$(rss $PY_PID); RSR=$(rss $RS_PID)
wait $LOAD 2>/dev/null
python3 - "$T0" "$T1" "$PY0" "$PY1" "$RS0" "$RS1" "$PYH" "$RSH" "$PYR" "$RSR" "$OUT" <<'EOF'
import json, os, sys
t0, t1, py0, py1, rs0, rs1, pyh, rsh, pyr, rsr = map(float, sys.argv[1:11])
hz = os.sysconf('SC_CLK_TCK')
res = {'seconds': round(t1 - t0, 1),
       'python': {'cpu_percent': round((py1 - py0) / hz / (t1 - t0) * 100, 1), 'peak_rss_mb': round(pyh / 1024, 1), 'rss_mb': round(pyr / 1024, 1)},
       'rust': {'cpu_percent': round((rs1 - rs0) / hz / (t1 - t0) * 100, 1), 'peak_rss_mb': round(rsh / 1024, 1), 'rss_mb': round(rsr / 1024, 1)}}
try:
    res['traffic'] = json.loads(open(os.path.join(sys.argv[11], 'load.log')).read().strip().splitlines()[-1])
except Exception as e:  # noqa: BLE001
    res['traffic'] = str(e)
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(sys.argv[11], 'cpu.json'), 'w'), indent=1)
EOF
stop; trap - EXIT
