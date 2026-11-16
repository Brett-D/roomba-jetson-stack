#!/usr/bin/env bash
# PC smoke-test stack runner. One stack = Nav2 (robot config) + Rust nodes + kinsim + optionally the
# Rust roomba_driver on tools/fake_roomba.py, isolated by ROS_DOMAIN_ID (localhost discovery only).
#   run_stack.sh start <name> <domain> [driver]   -> ~/pc_wsl_smoke/<name>/{launch.log,pgid}
#   run_stack.sh stop <name>
set -e
CMD=$1; NAME=$2; DOMAIN=$3; DRIVER=$4
SMOKE=$(cd "$(dirname "$0")" && pwd)
WS=$(cd "$SMOKE/../.." && pwd)
W=$HOME/pc_wsl_smoke/$NAME
source "$SMOKE/roomba_env.sh"
if [ "$CMD" = stop ]; then
  [ -f "$W/pgid" ] || exit 0
  G=$(cat "$W/pgid")
  kill -INT -- -$G 2>/dev/null || true
  for _ in $(seq 100); do pgrep -g $G >/dev/null || break; sleep 0.1; done
  kill -TERM -- -$G 2>/dev/null || true
  for _ in $(seq 50); do pgrep -g $G >/dev/null || break; sleep 0.1; done
  kill -KILL -- -$G 2>/dev/null || true
  rm -f "$W/pgid"
  exit 0
fi
rm -rf "$W"; mkdir -p "$W"
python3 - "$W" <<'EOF'
import sys
w = sys.argv[1]
W, H, res, ox, oy = 240, 200, 0.05, -2.0, -5.0
g = [[254] * W for _ in range(H)]
def cell(x, y):
    return int((x - ox) / res), H - 1 - int((y - oy) / res)
def box(x0, y0, x1, y1):
    c0, r1 = cell(x0, y0); c1, r0 = cell(x1, y1)
    for r in range(max(r0, 0), min(r1, H - 1) + 1):
        for c in range(max(c0, 0), min(c1, W - 1) + 1):
            g[r][c] = 0
box(-2.0, -5.0, 10.0, -4.9); box(-2.0, 4.9, 10.0, 5.0); box(-2.0, -5.0, -1.9, 5.0); box(9.9, -5.0, 10.0, 5.0)
box(3.95, -5.0, 4.05, -0.6); box(3.95, 0.6, 4.05, 5.0)        # interior wall, 1.2 m door at y=0
box(1.5, 0.8, 2.5, 1.6)                                          # furniture
with open(f'{w}/map.pgm', 'wb') as f:
    f.write(f'P5\n{W} {H}\n255\n'.encode())
    f.write(bytes(v for row in g for v in row))
open(f'{w}/map.yaml', 'w').write(f'image: map.pgm\nresolution: {res}\norigin: [{ox}, {oy}, 0.0]\n'
                                 'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\nmode: trinary\n')
EOF
export ROS_DOMAIN_ID=$DOMAIN ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_LOG_DIR=$W/roslog
EXTRA=
NAVP=
if [ -n "$SERVER_TIMEOUT_MS" ]; then
  # sim runs under load: the robot's 20 ms BT action-server timeout is too tight
  sed "s/default_server_timeout: 20\b/default_server_timeout: $SERVER_TIMEOUT_MS/" \
    "$(ros2 pkg prefix roomba_bringup)/share/roomba_bringup/config/nav2_robot.yaml" > "$W/nav2.yaml"
  grep -q "default_server_timeout: $SERVER_TIMEOUT_MS" "$W/nav2.yaml"
  NAVP="nav_params:=$W/nav2.yaml"
fi
if [ "$DRIVER" = driver ]; then
  python3 - "$WS/src/roomba_bringup/config/roomba_driver_robot.yaml" "$W/tty" "$W/driver.yaml" <<'EOF'
import sys, yaml
src, port, out = sys.argv[1:4]
d = yaml.safe_load(open(src))
d['roomba_driver']['ros__parameters']['port'] = port
yaml.safe_dump(d, open(out, 'w'))
EOF
  grep -q "port: $W/tty" "$W/driver.yaml"
  EXTRA="driver_params:=$W/driver.yaml"
  FAKE="python3 $WS/tools/fake_roomba.py --link $W/tty --log $W/serial.jsonl --t0-file $W/t0.json --scenario steady &"
fi
setsid bash -c "echo \$\$ > $W/pgid; $FAKE sleep 1; exec ros2 launch $SMOKE/smoke.launch.py work:=$W $EXTRA $NAVP" > "$W/launch.log" 2>&1 < /dev/null &
for _ in $(seq 50); do [ -s "$W/pgid" ] && break; sleep 0.1; done
echo "started $NAME domain $DOMAIN pgid $(cat $W/pgid)"
