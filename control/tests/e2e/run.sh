#!/usr/bin/env bash
# End-to-end test of the control page with stand-in robot scripts: no ROS launch, no serial port,
# no power-off. Starts test instances on 127.0.0.1:$E2E_PORT (18081) and drives them with curl.
#   control/tests/e2e/run.sh [path/to/roomba-control]     (default: control/target/release/...)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
CONTROL=$(cd "$HERE/../.." && pwd)
BIN=${1:-$CONTROL/target/release/roomba-control}
PORT=${E2E_PORT:-18081}
BASE="http://127.0.0.1:$PORT"
ORIGIN="Origin: $BASE"
AUTH="$CONTROL/tests/fixtures/auth_roomba_rs.json"   # password: fixture-password
export FAKE_DIR="$CONTROL/target/e2e"
export FAKE_MATCH="fake-robot-launch-$$"
FAKE_DRIVER="fake-roomba-driver-$$"
export FAKE_DRIVER_MATCH="$FAKE_DRIVER"   # fake_stop_slam.sh kills these, like stop_slam.sh the driver
export FAKE_START_DELAY=2
rm -rf "$FAKE_DIR" && mkdir -p "$FAKE_DIR"
JAR="$FAKE_DIR/jar"
MUTE="$FAKE_DIR/cfg/roomba/sound_muted"

fails=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: expected '$3', got '$2'"; fails=$((fails + 1)); fi; }
check_has() { case "$2" in *"$3"*) echo "ok   $1";; *) echo "FAIL $1: '$3' not in '$2'"; fails=$((fails + 1));; esac; }
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
J() { python3 -c 'import json,sys; v=eval(sys.argv[1], {"d": json.load(sys.stdin)}); print(v if isinstance(v, str) else json.dumps(v))' "$1"; }
st() { curl -s -b "$JAR" "$BASE/api/status" | J "$1"; }
# A required state that doesn't come in time fails the run there (the EXIT trap cleans up).
wait_for() {   # expression, value, seconds
    local i
    for i in $(seq 1 $(($3 * 4))); do [ "$(st "$1")" = "$2" ] && return 0; sleep 0.25; done
    echo "FAIL waited $3 s for $1 == $2; it is $(st "$1")"
    echo "FAILED (stopped at a missed deadline)"
    exit 1
}
login() { code -c "$JAR" -H "$ORIGIN" --data-urlencode "password=$1" "$BASE/login"; }
post() { code -b "$JAR" -X POST -H "$ORIGIN" "$BASE/api/$1"; }
# PIDs of the fake drivers (bash loops with $FAKE_DRIVER; not the control page, which has it too).
fake_drivers() { for p in $(pgrep -f -- "$FAKE_DRIVER"); do [ "$(cat /proc/$p/comm 2>/dev/null)" = bash ] && echo "$p"; done; }
kill_fakes() {
    for p in $(pgrep -f -- "$FAKE_MATCH"); do [ "$(cat /proc/$p/comm 2>/dev/null)" = bash ] && kill -9 "$p"; done
}
start_instance() {
    "$BIN" --listen "${LISTEN:-127.0.0.1:$PORT}" --auth-file "$AUTH" --pid-file "$FAKE_DIR/robot.pid" \
        --log-file "$FAKE_DIR/robot.log" --start-cmd "$HERE/fake_run_robot.sh" \
        --force-stop-cmd "$HERE/fake_stop_slam.sh" --launch-match "$FAKE_MATCH" --driver-match "$FAKE_DRIVER" --learner-match "fake-battery-learner-$$" \
        --serial-dev "$FAKE_DIR/no-such-tty" --ros-setup "sleep 1; false" --robot-web-port 18099 --sound-mute-file "$MUTE" --state-dir "$FAKE_DIR/state" "$@" \
        >> "$FAKE_DIR/server.log" 2>&1 &
    SERVER=$!
    for _ in $(seq 1 40); do [ "$(code "$BASE/login")" = 200 ] && return; sleep 0.25; done
    echo "the server didn't start:"; cat "$FAKE_DIR/server.log"; exit 1
}
stop_instance() { kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null; rm -f "$JAR"; }
# A failed wait_for exits mid-test: the standalone fake drivers (detached loops with $FAKE_DRIVER,
# not matched by kill_fakes) must go too, so a failed run leaves nothing behind.
trap 'stop_instance; kill_fakes; for p in $(fake_drivers); do kill -9 "$p"; done' EXIT

echo "== instance A: dry-run power-off, 6 s stop timeout"
start_instance --dry-run-poweroff --stop-timeout-s 6

check "/ without login redirects" "$(curl -s -o /dev/null -w '%{http_code} %{redirect_url}' "$BASE/")" "303 $BASE/login"
check_has "login page" "$(curl -s "$BASE/login")" 'name="password"'
check "status needs login" "$(code "$BASE/api/status")" 401
check "actions need login" "$(code -X POST -H "$ORIGIN" "$BASE/api/start")" 401
check "wrong password" "$(login wrong-password)" 401
check "login without Origin" "$(code -c "$JAR" --data-urlencode password=fixture-password "$BASE/login")" 403
check "login from another site" "$(code -c "$JAR" -H 'Origin: http://evil.example' --data-urlencode password=fixture-password "$BASE/login")" 403
check "right password" "$(login fixture-password)" 303
cookie=$(curl -s -D - -o /dev/null -H "$ORIGIN" --data-urlencode password=fixture-password "$BASE/login" | grep -i '^set-cookie:')
check_has "cookie is HttpOnly" "$cookie" "HttpOnly"
check_has "cookie is SameSite=Strict" "$cookie" "SameSite=Strict"
check "token is 256 bits" "$(echo "$cookie" | sed -n 's/.*roomba_control=\([0-9a-f]*\);.*/\1/p' | tr -d '\r\n' | wc -c)" 64
headers=$(curl -s -D - -o /dev/null -b "$JAR" "$BASE/")
check_has "page can't be framed" "$headers" "X-Frame-Options: DENY"
check_has "CSP allows only the robot page in frames" "$headers" "frame-src http://127.0.0.1:18099"

check "status" "$(st 'd["robot"]["state"], d["power"], d["roomba"]["link"]')" '["stopped", "on", "missing"]'
check "status has the health fields" "$(st 'sorted(k for k in ("camera","cpu_temp_c","wifi","uptime_s","disk","log") if k in d)')" \
    '["camera", "cpu_temp_c", "disk", "log", "uptime_s", "wifi"]'
check "no battery data, NEW PACK unavailable" "$(st 'd["battery"], d["pack_swap"]["available"], d["pack_swap"]["reason"]')" '[null, false, "Robot stack not running."]'
check "page URL from the Host header" "$(st 'd["page_url"], d["robot_url"]')" "[\"$BASE/\", \"http://127.0.0.1:18099/\"]"
check "GET on an action" "$(code -b "$JAR" "$BASE/api/start")" 405
check "action without Origin" "$(code -b "$JAR" -X POST "$BASE/api/start")" 403
check "action from another site" "$(code -b "$JAR" -X POST -H 'Origin: http://evil.example' "$BASE/api/start")" 403
check "action from the robot web app's port" "$(code -b "$JAR" -X POST -H 'Origin: http://127.0.0.1:18099' "$BASE/api/start")" 403

echo "== start / stop"
check "start" "$(post start)" 202
check "starting" "$(st 'd["robot"]["state"]')" starting
wait_for 'd["robot"]["state"]' running 15
check "running with the PID from the PID file" "$(st 'd["robot"]["pid"]')" "$(cat "$FAKE_DIR/robot.pid")"
check_has "robot log" "$(st '"|".join(d["log"]["lines"])')" "Tracking OK"
check "start while running" "$(post start)" 409
check "comms check while running" "$(post comms)" 202
check "a second one is refused" "$(post comms)" 409
wait_for 'd["roomba"]["checking"]' null 30
check "comms check went over ROS" "$(st 'd["roomba"]["result"]["via"], d["roomba"]["result"]["link"]')" '["ros", "not_responding"]'
check "stop" "$(post stop)" 202
check "stopping" "$(st 'd["robot"]["state"]')" stopping
check_has "stopping message" "$(st 'd["busy"]["message"]')" "saving map"
wait_for 'd["robot"]["state"]' stopped 15
check_has "stopped by SIGTERM" "$(st 'd["last"]["message"]')" "Stopped after"
check "no stop_slam.sh" "$([ -e "$FAKE_DIR/force_stop_called" ] && echo called || echo no)" no
check "PID file removed" "$([ -e "$FAKE_DIR/robot.pid" ] && echo present || echo gone)" gone
check_has "launch got SIGTERM" "$(cat "$FAKE_DIR/robot.log")" "SIGTERM: shutting down"
check_has "stop when stopped" "$(curl -s -b "$JAR" -X POST -H "$ORIGIN" "$BASE/api/stop")" "not running"

echo "== stop falls back to stop_slam.sh"
echo 30 > "$FAKE_DIR/stop_delay"
post start > /dev/null
wait_for 'd["robot"]["state"]' running 15
post stop > /dev/null
sleep 3
check "still stopping before the timeout" "$(st 'd["robot"]["state"]')" stopping
wait_for 'd["robot"]["state"]' stopped 20
check "stop_slam.sh ran" "$([ -e "$FAKE_DIR/force_stop_called" ] && echo called || echo no)" called
check_has "fallback message" "$(st 'd["last"]["message"]')" "stop_slam.sh"
rm -f "$FAKE_DIR/force_stop_called" "$FAKE_DIR/stop_delay"

echo "== a launch not started from here"
setsid bash -c 'while :; do sleep 0.2; done' "$FAKE_MATCH" > /dev/null 2>&1 < /dev/null &
disown
sleep 0.5
check "seen as running" "$(st 'd["robot"]["state"], d["robot"]["external"], d["robot"]["pid"]')" '["running", true, null]'
check "start refused" "$(post start)" 409
check "stop" "$(post stop)" 202
wait_for 'd["robot"]["state"]' stopped 10
check "stopped with stop_slam.sh" "$([ -e "$FAKE_DIR/force_stop_called" ] && echo called || echo no)" called
rm -f "$FAKE_DIR/force_stop_called"

echo "== roomba_driver left without its launch"
setsid bash -c 'while :; do sleep 0.2; done' "$FAKE_DRIVER" > /dev/null 2>&1 < /dev/null &
disown
sleep 0.5
check "seen as running, left over" "$(st 'd["robot"]["state"], d["robot"]["leftover"], d["robot"]["pid"]')" '["running", true, null]'
check "start refused" "$(post start)" 409
check "stop" "$(post stop)" 202
wait_for 'd["robot"]["state"]' stopped 15
check "stopped with stop_slam.sh" "$([ -e "$FAKE_DIR/force_stop_called" ] && echo called || echo no)" called
check "the driver is gone" "$(fake_drivers)" ""
rm -f "$FAKE_DIR/force_stop_called"

echo "== the launch ends but its driver doesn't"
post start > /dev/null
wait_for 'd["robot"]["state"]' running 15
setsid bash -c 'while :; do sleep 0.2; done' "$FAKE_DRIVER" > /dev/null 2>&1 < /dev/null &
disown
check "stop" "$(post stop)" 202
wait_for 'd["robot"]["state"]' stopped 30
check "stop_slam.sh for the leftover driver" "$([ -e "$FAKE_DIR/force_stop_called" ] && echo called || echo no)" called
check_has "message" "$(st 'd["last"]["message"]')" "stop_slam.sh"
check "the driver is gone" "$(fake_drivers)" ""
rm -f "$FAKE_DIR/force_stop_called"

echo "== comms check, stack stopped"
check "comms check" "$(post comms)" 202
wait_for 'd["roomba"]["checking"]' null 10
check "serial: port missing" "$(st 'd["roomba"]["result"]["via"], d["roomba"]["link"]')" '["serial", "missing"]'
check_has "port missing message" "$(st 'd["roomba"]["result"]["message"]')" "Port missing: check the USB-serial cable"
setsid bash -c 'while :; do sleep 0.2; done' "$FAKE_DRIVER" > /dev/null 2>&1 < /dev/null &
driver=$!
disown
sleep 0.5
check "driver running: comms check" "$(post comms)" 202
wait_for 'd["roomba"]["checking"]' null 30
check "the port is left to the driver: ROS" "$(st 'd["roomba"]["result"]["via"]')" ros
kill -9 "$driver"

echo "== sound switch"
check "sound on at first" "$(st 'd["sound"]["muted"]')" false
check "mute" "$(post sound/mute)" 202
check "mute file created, with its directory" "$([ -e "$MUTE" ] && echo yes || echo no)" yes
check "status shows muted" "$(st 'd["sound"]["muted"]')" true
check "unmute without Origin" "$(code -b "$JAR" -X POST "$BASE/api/sound/unmute")" 403
check "unmute" "$(post sound/unmute)" 202
check "mute file removed" "$([ -e "$MUTE" ] && echo yes || echo no)" no
check "unmute when not muted" "$(post sound/unmute)" 202
check "GET on the sound switch" "$(code -b "$JAR" "$BASE/api/sound/mute")" 405

echo "== stalled clients can't hold up a Stop"
# 160 connections from 8 other addresses: silent, slowloris headers, login bodies a byte a second
# (80: more than the old 64 request threads), and app.js requests that are never read.
post start > /dev/null
wait_for 'd["robot"]["state"]' running 15
python3 "$HERE/stall_clients.py" stall "$PORT" "$FAKE_DIR/stall_ready" 13 > "$FAKE_DIR/stall_report" 2>&1 &
STALL=$!
for _ in $(seq 1 40); do [ -e "$FAKE_DIR/stall_ready" ] && break; sleep 0.25; done
check "stalled clients connected" "$([ -e "$FAKE_DIR/stall_ready" ] && echo yes || echo no)" yes
# The status code if answered within 2 s.
fast() { curl -s -o /dev/null -m 10 -w '%{http_code} %{time_total}' "$@" | python3 -c 'import sys; c, t = sys.stdin.read().split(); print(c if float(t) < 2 else f"{c} after {t} s")'; }
check "status at once" "$(fast -b "$JAR" "$BASE/api/status")" 200
check "disarm at once" "$(fast -b "$JAR" -X POST -H "$ORIGIN" "$BASE/api/vacuum/disarm")" 202
check "stop at once" "$(fast -b "$JAR" -X POST -H "$ORIGIN" "$BASE/api/stop")" 202
wait_for 'd["robot"]["state"]' stopped 20
wait "$STALL"
check "the stalled clients dropped at their deadlines" "$(cat "$FAKE_DIR/stall_report")" "idle 32/32, slowloris 32/32, slowbody 80/80"
check "one address gets at most 32 connections" "$(python3 "$HERE/stall_clients.py" cap "$PORT")" "refused 8/40"
check "headers over 16 KB" "$(python3 "$HERE/stall_clients.py" bighead "$PORT")" 431
check "a body over 16 KB" "$(python3 "$HERE/stall_clients.py" bigbody "$PORT")" 413

echo "== shut down (dry run)"
post start > /dev/null
wait_for 'd["robot"]["state"]' running 15
check "shutdown" "$(post shutdown)" 202
wait_for 'd["robot"]["state"], d["busy"]["op"]' '["stopping", "shutdown"]' 3 && echo "ok   stops the robot first"
wait_for 'd["power"]' off 20
check "robot stopped, power off (dry run)" "$(st 'd["robot"]["state"], d["dry_run"], d["last"]["ok"]')" '["stopped", true, true]'
check_has "dry-run message" "$(st 'd["last"]["message"]')" "Dry run: would now run"
check "no stop_slam.sh" "$([ -e "$FAKE_DIR/force_stop_called" ] && echo called || echo no)" no
check "start after shutdown" "$(post start)" 409

echo "== logout and login backoff"
check "logout from another site refused" "$(code -b "$JAR" -X POST -H "Origin: http://elsewhere.example" "$BASE/logout")" 403
check "and its cookie left alone" "$(curl -s -o /dev/null -D - -b "$JAR" -X POST -H "Origin: http://elsewhere.example" "$BASE/logout" | grep -c -i '^set-cookie')" 0
check "still logged in" "$(code -b "$JAR" "$BASE/api/status")" 200
token=$(awk '$6 == "roomba_control" { print $7 }' "$JAR")
check "logout" "$(code -b "$JAR" -c "$JAR" -X POST -H "$ORIGIN" "$BASE/logout")" 303
check "the old session is gone" "$(code -H "Cookie: roomba_control=$token" "$BASE/api/status")" 401
for i in 1 2 3 4; do login "wrong-$i" > /dev/null; done
# The 5th and 6th at once: one waits for the other's password check, then finds the lockout.
login wrong-5 > "$FAKE_DIR/login5" & l5=$!
login wrong-6 > "$FAKE_DIR/login6" & l6=$!
wait "$l5" "$l6"
check "a login waiting for the password check is locked out too" "$(printf '%s\n' "$(cat "$FAKE_DIR/login5")" "$(cat "$FAKE_DIR/login6")" | sort | paste -sd' ')" "401 429"
check "locked out after 5 wrong passwords" "$(login fixture-password)" 429
stop_instance

echo "== instance B: 2 s sessions, power-off command runs"
start_instance --session-ttl-s 2 --poweroff-cmd "touch '$FAKE_DIR/poweroff_called'"
login fixture-password > /dev/null
check "logged in" "$(code -b "$JAR" "$BASE/api/status")" 200
sleep 2.5
check "session expired" "$(code -b "$JAR" "$BASE/api/status")" 401
login fixture-password > /dev/null
check "shutdown" "$(post shutdown)" 202
sleep 1.5
check "the power-off waits for a browser to see it coming" "$([ -e "$FAKE_DIR/poweroff_called" ] && echo ran || echo no)" no
login fixture-password > /dev/null   # 2 s sessions
check "powering off" "$(st 'd["power"]')" powering_off
wait_for 'd["power"]' off 10
check "power-off command ran" "$([ -e "$FAKE_DIR/poweroff_called" ] && echo ran || echo no)" ran
check_has "message" "$(st 'd["last"]["message"]')" "Powering off."
stop_instance

echo "== instance C: power-off command fails"
start_instance --poweroff-cmd "echo 'sudo: a password is required' >&2; exit 1"
login fixture-password > /dev/null
post shutdown > /dev/null
wait_for 'd["busy"]' null 10
check "power stays on" "$(st 'd["power"], d["last"]["ok"]')" '["on", false]'
check_has "failure message" "$(st 'd["last"]["message"]')" "Power-off failed: sudo: a password is required"
stop_instance

echo "== a second copy leaves the running one's motor commands alone"
start_instance
# A motor command of the running page on its way: its own process group, listed in the registry.
setsid bash -c 'sleep 60; true' /roomba/cleaning_motors-e2e > /dev/null 2>&1 < /dev/null &
MOTOR=$!
mkdir -p "$FAKE_DIR/state" && echo "$MOTOR" > "$FAKE_DIR/state/motor-commands"
sleep 0.3
second() {   # a second copy with the same state directory, listening on $1: its exit code
    timeout 5 "$BIN" --listen "$1" --auth-file "$AUTH" --pid-file "$FAKE_DIR/robot.pid" --log-file "$FAKE_DIR/robot.log" \
        --start-cmd /bin/false --force-stop-cmd /bin/false --launch-match none-2 --driver-match none-2 --learner-match none-2 \
        --serial-dev "$FAKE_DIR/no-such-tty" --ros-setup false --state-dir "$FAKE_DIR/state" --dry-run-poweroff >> "$FAKE_DIR/second.log" 2>&1
    echo $?
}
motor() { [ -e "/proc/$MOTOR" ] && [ "$(awk '{print $3}' "/proc/$MOTOR/stat")" != Z ] && echo running || echo killed; }
check "a copy on the same port exits" "$(second "127.0.0.1:$PORT")" 1
check "without touching the motor command" "$(motor)" running
check "a copy on another port exits too (the state directory is locked)" "$(second "127.0.0.1:$((PORT + 3))")" 1
check "the motor command still runs" "$(motor)" running
check "and is still listed" "$(cat "$FAKE_DIR/state/motor-commands")" "$MOTOR"
kill "$MOTOR"
wait "$MOTOR" 2>/dev/null
stop_instance

echo "== instance D: [::] takes IPv6 and IPv4"
LISTEN="[::]:$PORT" start_instance
check "logged" "$(grep -c "listening on \[::\]:$PORT (IPv6 and IPv4)" "$FAKE_DIR/server.log")" 1
check "IPv4" "$(code -4 "http://127.0.0.1:$PORT/login")" 200
check "IPv6" "$(code -6 -g "http://[::1]:$PORT/login")" 200
B6="http://[::1]:$PORT"
check "login over IPv6 (Host and Origin [::1]:port)" "$(code -g -c "$JAR" -H "Origin: $B6" --data-urlencode password=fixture-password "$B6/login")" 303
check "page and robot page URLs" "$(curl -s -g -b "$JAR" "$B6/api/status" | J '[d["page_url"], d["robot_url"]]')" "[\"http://[::1]:$PORT/\", \"http://[::1]:18099/\"]"
check "an action from the IPv6 page" "$(code -g -b "$JAR" -X POST -H "Origin: $B6" "$B6/api/sound/mute")" 202
check "Origin: the robot web app's port refused" "$(code -g -b "$JAR" -X POST -H "Origin: http://[::1]:18099" "$B6/api/sound/unmute")" 403
check "frame-src for an IPv6 host (CSP has no IPv6 literals)" "$(curl -s -g -D - -o /dev/null -b "$JAR" "$B6/" | grep -i -o 'frame-src [^;]*')" "frame-src http://*:18099"
stop_instance

echo "== instance E: [::] without IPv6 (--no-ipv6)"
LISTEN="[::]:$PORT" start_instance --no-ipv6
check "logged" "$(grep -c "listening on 0.0.0.0:$PORT (IPv4 only: IPv6 turned off)" "$FAKE_DIR/server.log")" 1
check "IPv4" "$(code -4 "http://127.0.0.1:$PORT/login")" 200
check "no IPv6" "$(code -6 -g -m 3 "http://[::1]:$PORT/login")" 000
stop_instance

if [ "$fails" -eq 0 ]; then echo "ALL PASSED"; else echo "$fails FAILED"; exit 1; fi
