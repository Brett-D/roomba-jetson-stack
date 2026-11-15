#!/usr/bin/env bash
# Replays one recording (scripts/record_run.sh) offline through both pipelines and compares them:
#   A  odometry:=basalt  fusion:=watchdog obstacles:=floor   (the default robot stack)
#   B  odometry:=cuvslam fusion:=ekf      obstacles:=nvblox  (GPU)
# Each run builds its own RTAB-Map database; ~/maps/vslam.db is never touched. The camera driver
# and the Roomba driver are off (camera:=false driver:=false): the bag provides their topics,
# the bag's /clock drives every node (use_sim_time:=true). Nav2 runs (costmaps, collision
# monitor) so CPU/GPU include the obstacle pipelines, but nothing can drive: no driver, no goals.
#
#   compare_pipelines.sh ~/bags/robot_YYYYmmdd_HHMMSS            both pipelines, real time
#   compare_pipelines.sh <bag> --rate 0.5                         slower (if replay drops frames)
#   compare_pipelines.sh <bag> --only B                           one pipeline (re-render the other)
#   compare_pipelines.sh <bag> --background                       detach; log in <outdir>/<bag>.log
#
# Output (/data/maps/compare/, or ~/maps/compare without /data):
#   <bag>_A.db, <bag>_B.db              RTAB-Map databases (open with rtabmap-databaseViewer)
#   <bag>_{A,B}.log / _play.log         launch / bag player logs
#   <bag>_{A,B}.json, _perf.json        metrics (compare_monitor.py) and CPU/GPU (perf_sample.py)
#   <bag>_side_by_side.png              both 2D grids at the same scale, with the driven path
#   <bag>_overlay.png                   obstacles only in A (red), only in B (blue), both (black)
#   <bag>_metrics.md                    the comparison table
BAG=""; RATE=1.0; ONLY=""; BACKGROUND=0
while [ $# -gt 0 ]; do
  case "$1" in
    --rate) RATE=$2; shift 2 ;;
    --only) ONLY=$2; shift 2 ;;
    --background) BACKGROUND=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    -*) echo "Unknown option: $1"; exit 1 ;;
    *) BAG=$1; shift ;;
  esac
done
[ -n "$BAG" ] && [ -d "$BAG" ] || { echo "Usage: $0 <bag directory> [--rate R] [--only A|B] [--background]"; exit 1; }
BAG=$(readlink -f "$BAG"); NAME=$(basename "$BAG")
if [ -d /data ]; then OUTDIR=/data/maps/compare; else OUTDIR=~/maps/compare; fi
mkdir -p "$OUTDIR"
if [ "$BACKGROUND" = 1 ]; then
  ARGS=(--rate "$RATE"); [ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
  setsid nohup "$0" "$BAG" "${ARGS[@]}" > "$OUTDIR/$NAME.log" 2>&1 < /dev/null &
  echo "Comparing in the background; log: $OUTDIR/$NAME.log, results: $OUTDIR/${NAME}_metrics.md"
  exit 0
fi

WS=$HOME/roomba_ws
TOOLS=$WS/tools
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
if pgrep -f "ros2 launch roomba_bringup" > /dev/null || pgrep -f "ros2 bag play" > /dev/null; then
  echo "The robot stack or a replay is running: stop it first (~/stop_slam.sh)."
  exit 1
fi

LABEL_A="A: basalt + watchdog + floor_obstacles"
LABEL_B="B: cuvslam + EKF + nvblox"
ARGS_A=(odometry:=basalt fusion:=watchdog obstacles:=floor)
ARGS_B=(odometry:=cuvslam fusion:=ekf obstacles:=nvblox)
COMMON=(mode:=mapping camera:=false driver:=false use_sim_time:=true web:=false people:=false
        foxglove:=false nav:=true)
# Everything recorded except the drive commands (reference only; nothing must act on them).
PLAY_TOPICS=(/camera/left/image_rect_mono /camera/right/image_rect_mono /camera/left/camera_info
             /camera/right/camera_info /camera/left/image_rect_color /camera/depth/image_rect_raw
             /camera/depth/camera_info /camera/imu/data_raw /wheel/odom /roomba/sensors /roomba/mode
             /battery_state /tf_static)
BAG_S=$(ros2 bag info "$BAG" | sed -n 's/^Duration: *\([0-9.]*\)s.*/\1/p')
[ -n "$BAG_S" ] || { echo "Could not read the bag ($BAG)"; exit 1; }
PLAY_S=$(echo "$BAG_S / $RATE" | bc -l)
printf "Bag %s: %.0f s, replay at x%s (~%.0f s per pipeline + ~1 min startup)\n" "$NAME" "$BAG_S" "$RATE" "$PLAY_S"
ros2 bag info "$BAG" | grep -q /wheel/odom || echo "Note: no /wheel/odom in this bag (the EKF runs on gyro + VIO only)."

wait_nodes() {  # wait_nodes <timeout_s> <node>...
  local t_end=$(( $(date +%s) + $1 )); shift
  while [ "$(date +%s)" -lt "$t_end" ]; do
    local list; list=$(ros2 node list 2>/dev/null)
    local missing=0
    for n in "$@"; do echo "$list" | grep -qx "$n" || missing=1; done
    [ $missing = 0 ] && return 0
    sleep 2
  done
  echo "Timed out waiting for nodes: $*"; return 1
}

run_pipeline() {  # run_pipeline A|B
  local P=$1 PREFIX="$OUTDIR/${NAME}_$1" ARGS NODES
  if [ "$P" = A ]; then ARGS=("${ARGS_A[@]}"); NODES=(/basalt_vio /vslam_watchdog);
  else ARGS=("${ARGS_B[@]}"); NODES=(/visual_slam_node /vslam_watchdog /ekf_filter_node /nvblox_node); fi
  echo "=== Pipeline $P: ${ARGS[*]}"
  rm -f "$PREFIX.db" "$PREFIX".json "$PREFIX"_*.json "$PREFIX"_*.npz "$PREFIX"_*.npy "$PREFIX"_*.txt
  # graceful.py: the SIGINT from stop_slam.sh reaches the launch (started in the background, plain
  # ros2 launch ignores it), which stops every node and lets RTAB-Map save its database.
  setsid "$WS/scripts/graceful.py" ros2 launch roomba_bringup robot.launch.py "${COMMON[@]}" "${ARGS[@]}" \
    database_path:="$PREFIX.db" \
    > "$PREFIX.log" 2>&1 < /dev/null &
  if ! wait_nodes 150 /rtabmap/rtabmap /collision_monitor "${NODES[@]}"; then
    ~/stop_slam.sh > /dev/null; return 1
  fi
  sleep 8  # parameters, nvblox GPU allocation, lifecycle bring-up
  python3 "$TOOLS/compare_monitor.py" "$PREFIX" --ros-args -p use_sim_time:=true > "${PREFIX}_monitor.txt" 2>&1 &
  local MON=$!
  sleep 2
  python3 "$TOOLS/perf_sample.py" "$(printf %.0f "$PLAY_S")" --json "${PREFIX}_perf.json" > "${PREFIX}_perf.txt" 2>&1 &
  local PERF=$!
  echo "  replaying..."
  ros2 bag play "$BAG" --clock 50 --rate "$RATE" --read-ahead-queue-size 2000 \
    --topics "${PLAY_TOPICS[@]}" > "${PREFIX}_play.log" 2>&1
  sleep 6  # last frames through RTAB-Map, final grid published
  kill -INT $MON 2>/dev/null; wait $MON
  wait $PERF
  ~/stop_slam.sh > /dev/null   # RTAB-Map saves its database on shutdown
  cat "${PREFIX}_perf.txt" | head -12 | sed 's/^/  /'
  echo "  metrics: $(tail -1 "${PREFIX}_monitor.txt")"
}

for P in A B; do
  [ -n "$ONLY" ] && [ "$ONLY" != "$P" ] && continue
  run_pipeline $P || echo "Pipeline $P FAILED (see $OUTDIR/${NAME}_$P.log)"
done

python3 "$TOOLS/compare_render.py" "$OUTDIR" "$NAME" "$LABEL_A" "$OUTDIR/${NAME}_A" "$LABEL_B" "$OUTDIR/${NAME}_B"
echo "Results: $OUTDIR/${NAME}_metrics.md, ${NAME}_side_by_side.png, ${NAME}_overlay.png"
