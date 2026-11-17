#!/usr/bin/env bash
# Stand-in for scripts/run_robot.sh in the control page's end-to-end test (run.sh): like it, it
# starts a long-running "launch" in the background, writes that PID to the PID file and returns.
# The fake launch logs "Tracking OK" and, on SIGTERM, writes the time to $FAKE_DIR/sigterm_at and
# exits after $FAKE_DIR/stop_delay seconds (3 by default; make it longer than --stop-timeout-s to
# test the stop_slam.sh fallback).
set -e
: "${FAKE_DIR:?}" "${FAKE_MATCH:?}"
echo "fake run_robot.sh: clearing leftovers"
sleep "${FAKE_START_DELAY:-2}"          # the real script runs stop_slam.sh first
delay=$(cat "$FAKE_DIR/stop_delay" 2>/dev/null || echo 3)
# $0 of the fake launch is $FAKE_MATCH, so its command line matches --launch-match.
setsid nohup bash -c '
  trap "date +%s.%N > \"\$2/sigterm_at\"; echo \"[launch] SIGTERM: shutting down, saving map\"; sleep $1; echo \"[launch] done\"; exit 0" TERM
  echo "[vslam_watchdog] Tracking OK"
  while :; do sleep 0.2; done' "$FAKE_MATCH" "$delay" "$FAKE_DIR" > "$FAKE_DIR/robot.log" 2>&1 < /dev/null &
echo $! > "$FAKE_DIR/robot.pid"
echo "Robot starting (fake, mode auto)."
