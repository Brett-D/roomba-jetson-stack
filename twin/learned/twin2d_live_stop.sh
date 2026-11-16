#!/usr/bin/env bash
# Stops the live unstuck demo (twin/learned/twin2d_live.py) and everything it started: SIGTERM to
# the loop (it stops the current run cleanly), then any process still marked LEARNED_LIVE=1.
OUT=${1:-/root/twin2d_data/runs/learned_live}
P=$(cat "$OUT/live.pid" 2>/dev/null)
if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then
  kill -TERM "$P"
  for _ in $(seq 40); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
fi
left() { grep -l -s -z '^LEARNED_LIVE=1$' /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3; }
for sig in INT TERM KILL; do
  pids=$(left)
  [ -z "$pids" ] && break
  kill -$sig $pids 2>/dev/null
  sleep 4
done
echo "live demo stopped; remaining: $(left | wc -l) processes"
