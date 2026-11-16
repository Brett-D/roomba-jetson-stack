#!/usr/bin/env bash
# N isolated stacks in parallel (domains BASE+1..BASE+N), each driving NavigateToPose loops through
# the test house; CPU/RAM sampled while they all navigate.   parallel.sh <N> [loops] [base_domain]
N=$1; LOOPS=${2:-2}; BASE=${3:-80}
S=$(cd "$(dirname "$0")" && pwd)
source "$S/roomba_env.sh"
OUT=$HOME/pc_wsl_logs/parallel_$N; rm -rf "$OUT"; mkdir -p "$OUT"
for i in $(seq $N); do "$S/run_stack.sh" start p$i $((BASE + i)) driver; done
sleep 2
G=""; for i in $(seq $N); do G="$G $(cat ~/pc_wsl_smoke/p$i/pgid)"; done
python3 "$S/measure.py" 15 $G > "$OUT/idle_startup.txt"
for i in $(seq $N); do
  ( export ROS_DOMAIN_ID=$((BASE + i)) ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_LOG_DIR=$HOME/pc_wsl_smoke/p$i/roslog_check
    timeout 600 python3 "$S/smoke_check.py" --driver --loops $LOOPS --wait-active 120 > "$OUT/check_p$i.txt" 2>&1 ) &
done
sleep 25
python3 "$S/measure.py" 30 $G > "$OUT/navigating.txt"
wait
for i in $(seq $N); do grep -H 'process has died' ~/pc_wsl_smoke/p$i/launch.log | cut -c1-140; done > "$OUT/died_before_stop.txt"
for i in $(seq $N); do "$S/run_stack.sh" stop p$i; done
echo "== $N stacks: startup/idle (15 s)"; tail -1 "$OUT/idle_startup.txt"
echo "== $N stacks: navigating (30 s)"; cat "$OUT/navigating.txt"
for i in $(seq $N); do echo "-- p$i: $(grep -c SUCCEEDED $OUT/check_p$i.txt) goals succeeded; $(grep -E 'SMOKE|wheel' $OUT/check_p$i.txt | tr '\n' ' ')"; grep -hE 'ABORTED|timeout|rejected' "$OUT/check_p$i.txt"; done
echo "deaths before stop: $(wc -l < $OUT/died_before_stop.txt)"; cat "$OUT/died_before_stop.txt"
pgrep -f 'roomba_rs|nav2_|fake_roomba|kinsim|ros2 launch' | wc -l
