#!/usr/bin/env bash
# The control page's ROS side (SOUND buttons, VACUUM ARMED, vacuum off before a stop, BATTERY and
# NEW PACK) against fake_driver.py and fake_battery_learner.py in a private ROS domain, never the
# robot's (domain 0). Needs ROS Jazzy and the workspace's roomba_msgs:
#   control/tests/e2e/ros_controls.sh [path/to/roomba-control]
set -u
export ROS_DOMAIN_ID=${E2E_ROS_DOMAIN_ID:-78}
[ "$ROS_DOMAIN_ID" != 0 ] || { echo "refusing to run in ROS domain 0 (the robot's)"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
CONTROL=$(cd "$HERE/../.." && pwd)
BIN=${1:-$CONTROL/target/release/roomba-control}
PORT=${E2E_PORT:-18083}
BASE="http://127.0.0.1:$PORT"
ORIGIN="Origin: $BASE"
SETUP="source /opt/ros/jazzy/setup.bash && source ${ROBOT_WS:-$HOME/roomba_ws}/install/setup.bash"
export FAKE_DIR="$CONTROL/target/e2e-ros"
export FAKE_MATCH="fake-robot-launch-ros-$$"
export FAKE_START_DELAY=1
TOKEN="fake-roomba-driver-$$"
LTOKEN="fake-battery-learner-$$"
rm -rf "$FAKE_DIR" && mkdir -p "$FAKE_DIR"
JAR="$FAKE_DIR/jar"
LOG="$FAKE_DIR/driver.jsonl"

fails=0
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: expected '$3', got '$2'"; fails=$((fails + 1)); fi; }
check_has() { case "$2" in *"$3"*) echo "ok   $1";; *) echo "FAIL $1: '$3' not in '$2'"; fails=$((fails + 1));; esac; }
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
J() { python3 -c 'import json,sys; v=eval(sys.argv[1], {"d": json.load(sys.stdin)}); print(v if isinstance(v, str) else json.dumps(v))' "$1"; }
st() { curl -s -b "$JAR" "$BASE/api/status" | J "$1"; }
wait_for() {   # expression, value, seconds
    local i
    for i in $(seq 1 $(($3 * 4))); do [ "$(st "$1")" = "$2" ] && return 0; sleep 0.25; done
    echo "FAIL waited $3 s for $1 == $2; it is $(st "$1")"
    fails=$((fails + 1))
    return 1
}
post() { code -b "$JAR" -X POST -H "$ORIGIN" "$BASE/api/$1"; }
# The last driver.jsonl entry of a kind, as JSON with the fields given (python expression on e).
last_entry() { python3 -c 'import json,sys; es=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; es=[e for e in es if e["kind"]==sys.argv[2]]; e=es[-1] if es else None; print(json.dumps(eval(sys.argv[3])) if e else "none")' "$LOG" "$1" "$2"; }
wait_entry() {   # kind, expression, value, seconds
    local i
    for i in $(seq 1 $(($4 * 2))); do [ "$(last_entry "$1" "$2")" = "$3" ] && return 0; sleep 0.5; done
    return 1
}
driver_up() {
    bash -c "$SETUP && exec python3 '$HERE/fake_driver.py' '$TOKEN'" > "$FAKE_DIR/driver.out" 2>&1 < /dev/null &
    DRIVER=$!
}
learner_up() {
    bash -c "$SETUP && exec python3 '$HERE/fake_battery_learner.py' '$LTOKEN'" > "$FAKE_DIR/learner.out" 2>&1 < /dev/null &
    LEARNER=$!
}
server_up() {
    "$BIN" --listen "127.0.0.1:$PORT" --auth-file "$CONTROL/tests/fixtures/auth_roomba_rs.json" \
        --pid-file "$FAKE_DIR/robot.pid" --log-file "$FAKE_DIR/robot.log" --start-cmd "$HERE/fake_run_robot.sh" \
        --force-stop-cmd "$HERE/fake_stop_slam.sh" --launch-match "$FAKE_MATCH" --driver-match "$TOKEN" \
        --learner-match "$LTOKEN" --serial-dev "$FAKE_DIR/no-such-tty" --ros-setup "$SETUP" \
        --sound-mute-file "$FAKE_DIR/sound_muted" --robot-web-port 18099 --dry-run-poweroff >> "$FAKE_DIR/server.log" 2>&1 &
    SERVER=$!
    for _ in $(seq 1 40); do [ "$(code "$BASE/login")" = 200 ] && break; sleep 0.25; done
    code -c "$JAR" -H "$ORIGIN" --data-urlencode password=fixture-password "$BASE/login" > /dev/null
}
cleanup() {
    kill "$SERVER" "$DRIVER" "${LEARNER:-}" 2>/dev/null
    for p in $(pgrep -f -- "$FAKE_MATCH"); do [ "$(cat /proc/$p/comm 2>/dev/null)" = bash ] && kill -9 "$p"; done
    bash -c "$SETUP && ros2 daemon stop" > /dev/null 2>&1
}
trap cleanup EXIT

echo 'passive' > "$FAKE_DIR/mode"
driver_up
server_up
post start > /dev/null
wait_for 'd["robot"]["state"]' running 15

echo "== OI mode passive: no sounds, no arming"
wait_for 'd["ros"]["mode"]' passive 40
check "unavailable" "$(st 'd["ros"]["available"]')" false
check_has "reason" "$(st 'd["ros"]["reason"]')" "OI mode passive: needs Safe or Full"
check "sound refused" "$(post tone/ow)" 409
check "arming refused" "$(post vacuum/arm)" 409

echo "== Safe mode: the sound buttons"
echo 'safe' > "$FAKE_DIR/mode"
wait_for 'd["ros"]["available"]' true 15
check "OW" "$(post tone/ow)" 202
check "debounced" "$(post tone/hello)" 409
wait_for 'd["tones"]["pending"]' null 40
check "OW played" "$(st 'd["tones"]["last"]["key"], d["tones"]["last"]["ok"]')" '["ow", true]'
check "OW song" "$(last_entry song '[e["song"], e["notes"], e["durations"], e["play"]]')" '[2, [96, 91, 86, 81, 76], [3, 3, 3, 4, 12], true]'
touch "$FAKE_DIR/sound_muted"   # MUTE silences only automatic sounds
for t in "armed 3 [88, 30, 88, 30, 95] [6, 3, 6, 3, 18]" \
         "chatter 4 [96, 84, 100, 79, 103, 88, 108, 91, 98] [4, 3, 5, 3, 4, 3, 6, 3, 10]" \
         "hello 0 [67, 72, 76] [10, 10, 24]" "uhoh 1 [76, 76, 69] [12, 6, 28]"; do
    read -r key song _ <<< "$t"
    rest=${t#* * }; notes=${rest%%] *}]; durs=${rest#*] }
    sleep 1.1
    check "$key" "$(post tone/$key)" 202
    wait_for 'd["tones"]["pending"]' null 40
    check "$key song" "$(last_entry song '[e["song"], e["notes"], e["durations"], e["play"]]')" "[$song, $notes, $durs, true]"
done
check "played while muted" "$(st 'd["tones"]["last"]["ok"]')" true
rm -f "$FAKE_DIR/sound_muted"

echo "== BATTERY: battery_state without battery_learner"
wait_for 'd["battery"] is not None' true 40
check "voltage and charge, no pack or capacity" "$(st '[d["battery"][k] for k in ("voltage", "percentage", "pack", "capacity_ah")]')" '[18.2, 56.0, null, null]'
check "NEW PACK unavailable" "$(st 'd["pack_swap"]["available"], d["pack_swap"]["reason"]')" '[false, "battery_learner isn'"'"'t running (not installed yet?)."]'
check "NEW PACK refused" "$(post battery/new_pack)" 409

echo "== NEW PACK with battery_learner"
learner_up
echo 4.2 > "$FAKE_DIR/capacity"
wait_for 'd["pack_swap"]["available"]' true 40
wait_for 'd["battery"]["capacity_ah"]' 4.2 10
check "new pack" "$(post battery/new_pack)" 202
check "one at a time" "$(post battery/new_pack)" 409
wait_for 'd["pack_swap"]["pending"]' false 40
check "battery_learner's answer" "$(st 'd["pack_swap"]["last"]["ok"], d["pack_swap"]["last"]["message"]')" '[true, "new pack 1 at 100 %"]'
wait_for 'd["battery"]["pack"]' 1 10
kill "$LEARNER"
wait_for 'd["pack_swap"]["available"]' false 10

echo "== VACUUM ARMED"
check "arm" "$(post vacuum/arm)" 202
wait_for 'd["vacuum"]["armed"]' true 40
check "motors on" "$(last_entry motors '[e["main_brush"], e["side_brush"], e["vacuum"], e["use_pwm"]]')" '[true, true, true, false]'
wait_for 'd["tones"]["pending"]' null 40
check "ARMED tone played" "$(last_entry song 'e["song"]')" 3

echo "== disarmed by a cliff (and all-off sent)"
echo 'cliff_front_left' > "$FAKE_DIR/hazard"
wait_for 'd["vacuum"]["armed"]' false 40
check_has "why" "$(st 'd["vacuum"]["note"]["message"]')" "cliff front left"
wait_entry motors '[e["main_brush"], e["side_brush"], e["vacuum"]]' '[false, false, false]' 30
check "all off sent" "$(last_entry motors '[e["main_brush"], e["side_brush"], e["vacuum"]]')" '[false, false, false]'
rm -f "$FAKE_DIR/hazard"

echo "== disarmed when the OI mode leaves Safe / Full"
sleep 1
post vacuum/arm > /dev/null
wait_for 'd["vacuum"]["armed"]' true 40
echo 'passive' > "$FAKE_DIR/mode"
wait_for 'd["vacuum"]["armed"]' false 15
check_has "why" "$(st 'd["vacuum"]["note"]["message"]')" "OI mode passive"
echo 'safe' > "$FAKE_DIR/mode"
wait_for 'd["ros"]["available"]' true 15

echo "== switched off by hand"
post vacuum/arm > /dev/null
wait_for 'd["vacuum"]["armed"]' true 40
check "disarm" "$(post vacuum/disarm)" 202
wait_for 'd["vacuum"]["armed"]' false 40
check "all off" "$(last_entry motors '[e["main_brush"], e["side_brush"], e["vacuum"]]')" '[false, false, false]'

echo "== vacuum off before the robot stops"
post vacuum/arm > /dev/null
wait_for 'd["vacuum"]["armed"]' true 40
check "stop" "$(post stop)" 202
wait_for 'd["robot"]["state"]' stopped 60
off_at=$(last_entry motors 'e["t"] if not e["vacuum"] else None')
check "all off before SIGTERM" "$(python3 -c "print($off_at < $(cat "$FAKE_DIR/sigterm_at"))")" True
check "disarmed" "$(st 'd["vacuum"]["armed"]')" false

echo "== disarmed when the driver goes away"
post start > /dev/null
wait_for 'd["robot"]["state"]' running 15
post vacuum/arm > /dev/null
wait_for 'd["vacuum"]["armed"]' true 40
kill "$DRIVER"
wait_for 'd["vacuum"]["armed"]' false 15
check_has "why" "$(st 'd["vacuum"]["note"]["message"]')" "robot stack stopped"
check "sounds unavailable" "$(st 'd["ros"]["available"], d["ros"]["reason"]')" '[false, "Waiting for roomba_driver\u2026"]'

echo "== shut down (dry run) switches the vacuum off first"
driver_up
wait_for 'd["ros"]["available"]' true 40
post vacuum/arm > /dev/null
wait_for 'd["vacuum"]["armed"]' true 40
rm -f "$FAKE_DIR/sigterm_at"
check "shutdown" "$(post shutdown)" 202
wait_for 'd["power"]' off 60
off_at=$(last_entry motors 'e["t"] if not e["vacuum"] else None')
check "all off before SIGTERM" "$(python3 -c "print($off_at < $(cat "$FAKE_DIR/sigterm_at"))")" True

if [ "$fails" -eq 0 ]; then echo "ALL PASSED"; else echo "$fails FAILED"; exit 1; fi
