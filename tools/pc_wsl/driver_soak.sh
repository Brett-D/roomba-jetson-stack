#!/usr/bin/env bash
# Runs a roomba_driver binary against fake_roomba.py (steady scenario) for SECS seconds and
# reports whether it survived and its /wheel/odom rate.  driver_soak.sh <binary> <name> <domain> <secs>
BIN=$1; NAME=$2; DOMAIN=$3; SECS=${4:-120}
S=$(cd "$(dirname "$0")" && pwd); WS=$(cd "$S/../.." && pwd)
W=$HOME/pc_wsl_smoke/$NAME
rm -rf "$W"; mkdir -p "$W"
source "$S/roomba_env.sh"
export ROS_DOMAIN_ID=$DOMAIN ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_LOG_DIR=$W/roslog
python3 - "$WS/src/roomba_bringup/config/roomba_driver_robot.yaml" "$W/tty" "$W/driver.yaml" <<'EOF'
import sys, yaml
src, port, out = sys.argv[1:4]
d = yaml.safe_load(open(src))
d['roomba_driver']['ros__parameters']['port'] = port
yaml.safe_dump(d, open(out, 'w'))
EOF
python3 $WS/src/roomba-serial-ros2/tools/fake_roomba.py --link "$W/tty" --log "$W/serial.jsonl" --t0-file "$W/t0.json" --scenario steady &
FAKE=$!
for _ in $(seq 50); do [ -e "$W/tty" ] && break; sleep 0.1; done
# the robot YAML's port is the real adapter; -p would lose against the params file (rclrs)
"$BIN" --ros-args --params-file "$W/driver.yaml" > "$W/driver.log" 2>&1 &
DRV=$!
sleep 5
timeout 8 ros2 topic hz /wheel/odom --window 100 2>/dev/null | grep -m1 'average rate' > "$W/hz.txt"
SECONDS=0
while kill -0 $DRV 2>/dev/null; do
  sleep 0.5
  [ $SECONDS -ge $SECS ] && break
done
if kill -0 $DRV 2>/dev/null; then
  echo "$NAME: driver alive after ${SECS}s; $(cat $W/hz.txt); port $(grep -o 'starting on [^ ]*' $W/driver.log)"
  kill -TERM $DRV; wait $DRV; echo "$NAME: exit code after SIGTERM $?"
else
  wait $DRV; rc=$?
  echo "$NAME: driver DIED after $((SECONDS + 13))s (rc $rc); $(cat $W/hz.txt)"
  grep -A2 'Error' $W/driver.log | head -3
fi
tail -c 300 $W/serial.jsonl | tail -1
kill -TERM $FAKE; wait $FAKE 2>/dev/null
