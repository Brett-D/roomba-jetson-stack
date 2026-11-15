#!/usr/bin/env bash
# Side-by-side A/B of the web app: roomba_web's server.py and the Rust web_server (roomba_rs) run
# at the same time on the same inputs; tools/ab_web.py drives both (scripted topics, TF, fake
# Nav2 action server and /roomba/set_mode service, HTTP and websocket clients) and compares every
# response, websocket message, MJPEG frame and published command. Each server's outputs are
# remapped under /py and /rs. A test HOME holds the password (never ~/.config/roomba_web).
#   tools/ab_web.sh OUT_DIR [BAG]     env: PY_PORT=18081 RS_PORT=18082 ROS_DOMAIN_ID=63, RS_BIN
OUT=${1:?usage: ab_web.sh OUT_DIR [BAG]}; BAG=${2:-/data/bags/robot_20261114_145937_fresh_map8}
HERE=$(cd "$(dirname "$0")" && pwd)
source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
[ -f "$HERE/../install/setup.bash" ] && source "$HERE/../install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-63}
PY_PORT=${PY_PORT:-18081}; RS_PORT=${RS_PORT:-18082}
RS_BIN=${RS_BIN:-$(ros2 pkg prefix roomba_rs)/lib/roomba_rs/web_server}
PY_BIN=$(ros2 pkg prefix roomba_web)/lib/roomba_web/web_server
mkdir -p "$OUT/home"
export AB_PASSWORD=${AB_PASSWORD:-ab-test-password-1}
HOME="$OUT/home" python3 -c "import os; from roomba_web.auth import set_password; set_password(os.environ['AB_PASSWORD'])"

remaps() {  # every output of a server under /$1
  echo -r __node:=web_server_$1 -r /cmd_vel:=/$1/cmd_vel -r /cmd_vel_nav:=/$1/cmd_vel_nav \
    -r /perception/ignore_zones/command:=/$1/zone_command -r /explorer/command:=/$1/explorer_command \
    -r /perception/person_detector/floor_grid:=/$1/floor_grid -r navigate_to_pose:=/$1/navigate_to_pose \
    -r /roomba/set_mode:=/$1/set_mode
}
PIDS=()
stop() { for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done; sleep 2; for p in "${PIDS[@]}"; do kill -KILL "$p" 2>/dev/null; done; }
trap stop EXIT
HOME="$OUT/home" python3 "$PY_BIN" --ros-args -p port:=$PY_PORT $(remaps py) > "$OUT/py_server.log" 2>&1 < /dev/null &
PIDS+=($!)
HOME="$OUT/home" "$RS_BIN" --ros-args -p port:=$RS_PORT $(remaps rs) > "$OUT/rs_server.log" 2>&1 < /dev/null &
PIDS+=($!)
sleep 4
python3 -u "$HERE/ab_web.py" --py-port $PY_PORT --rs-port $RS_PORT --bag "$BAG" --out "$OUT" 2>&1 | tee "$OUT/ab_web.txt"
status=${PIPESTATUS[0]}
stop; trap - EXIT
exit $status
