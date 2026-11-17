#!/usr/bin/env bash
# Stand-in for scripts/stop_slam.sh in the end-to-end test: records that it ran, kills the fake
# launches (bash processes whose command line has $FAKE_MATCH; not the control page, whose command
# line has it too) and removes the PID file.
: "${FAKE_DIR:?}" "${FAKE_MATCH:?}"
touch "$FAKE_DIR/force_stop_called"
for p in $(pgrep -f -- "$FAKE_MATCH"); do
    [ "$(cat /proc/$p/comm 2>/dev/null)" = bash ] && kill -9 "$p"
done
rm -f "$FAKE_DIR/robot.pid"
echo "all stopped"
