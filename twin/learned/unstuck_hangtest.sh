#!/bin/bash
# Does an idle unstuck_skill node keep answering goals across system-clock steps (WSL steps its clock
# every ~30 s)? Starts one node per ROS domain and sends a short goal every ~13 s; logs
# "answered" / "NOANSWER" per goal (and whether the node exited, e.g. through its watchdog).
#   source /root/learned_ws/learned_env.sh
#   bash twin/learned/unstuck_hangtest.sh OUT LABEL BINARY DOMAIN...
#   python3 tools/pc_wsl/clockwatch.py 300 --stream     # in parallel: when the clock was stepped
# The goal fails at once without a costmap / TF (that is an answer); only the action round trip counts.
out=$1; label=$2; bin=$3; shift 3
goals=${GOALS:-10}
model=${MODEL:-$(ros2 pkg prefix roomba_rs)/share/roomba_rs/models/unstuck_policy.onnx}
mkdir -p "$out"
for d in "$@"; do
  ( export ROS_DOMAIN_ID=$d
    "$bin" --ros-args -p model:="$model" > "$out/${label}_$d.node.log" 2>&1 &
    np=$!
    sleep 5
    for i in $(seq 1 "$goals"); do
      if timeout 15 ros2 action send_goal /unstuck_skill roomba_skills_msgs/action/Unstuck "{time_limit_s: 1.0}" \
          > "$out/${label}_${d}_goal$i.log" 2>&1; then r=answered; else r=NOANSWER; fi
      kill -0 $np 2>/dev/null || r="$r (node exited)"
      echo "$(date +%T.%N | cut -c1-12) goal $i: $r" >> "$out/${label}_$d.log"
      sleep 12
    done
    kill -INT $np 2>/dev/null; wait $np
  ) &
done
wait
grep -c answered "$out/${label}"_*[0-9].log