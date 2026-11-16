#!/usr/bin/env bash
# cuVSLAM + nvblox on the RTX 5090 inside WSL, fed by smoke/synth_stereo.py.
S=$(cd "$(dirname "$0")" && pwd); WS=$(cd "$S/../.." && pwd)
OUT=$HOME/pc_wsl_logs/gpu_test; rm -rf "$OUT"; mkdir -p "$OUT"
source /opt/ros/jazzy/setup.bash
export CUDA_CACHE_MAXSIZE=4294967296 ROS_DOMAIN_ID=61 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_LOG_DIR=$OUT/roslog
setsid bash -c "echo \$\$ > $OUT/pgid
python3 $S/synth_stereo.py > $OUT/synth.log 2>&1 &
ros2 launch $S/cuvslam.launch.py > $OUT/cuvslam.log 2>&1 &
ros2 run nvblox_ros nvblox_node --ros-args --params-file $WS/src/roomba_bringup/config/nvblox_robot.yaml \
  -r camera_0/depth/image:=/synth/depth/image -r camera_0/depth/camera_info:=/synth/depth/camera_info \
  > $OUT/nvblox.log 2>&1 &
wait" > /dev/null 2>&1 < /dev/null &
sleep ${WARMUP:-15}
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > "$OUT/gpu_before.txt"
python3 "$S/gpu_check.py" ${SECS:-30} | tee "$OUT/check.txt"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
G=$(cat "$OUT/pgid")
ps -o pcpu=,rss=,args= -g $G | grep -E 'component_container|nvblox_node|synth' | cut -c1-90
kill -INT -- -$G; sleep 5; kill -KILL -- -$G 2>/dev/null
echo "--- cuvslam log"; grep -vE '^\s*$' "$OUT/cuvslam.log" | grep -iE 'error|warn|cuda|gpu|sm_|version|tracking|lost' | head -12
echo "--- nvblox log"; grep -iE 'error|warn|cuda|gpu|fatal' "$OUT/nvblox.log" | head -12
tail -3 "$OUT/synth.log"
