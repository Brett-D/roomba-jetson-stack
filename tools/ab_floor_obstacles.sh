#!/usr/bin/env bash
# A/B check of the Rust floor_obstacles (roomba_rs) against the Python one (roomba_perception):
# replays DUR s of a recorded drive (record_run.sh) from OFF s in, runs both nodes with the robot's
# parameters on remapped topics, and prints CPU use plus a frame-by-frame comparison.
#   tools/ab_floor_obstacles.sh /data/bags/robot_..._fresh_map8        (OFF=150 DUR=90 by default)
# Small differences on a few frames are expected: each node uses the IMU sample it has at the time.
source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
BAG=${1:?usage: ab_floor_obstacles.sh BAG_DIR  (env OFF=start s, DUR=seconds)}; OFF=${OFF:-150}; DUR=${DUR:-90}
cat > /tmp/fo_params.yaml <<'EOF'
/**:
  ros__parameters:
    fixed_camera_height_m: 0.11
    fallback_camera_height_m: 0.4
    pause_turn_rate_rps: 0.5
    min_obstacle_height_m: 0.045
    min_cell_points: 4
    max_range_m: 2.5
EOF
R="-r /perception/obstacles:=/X/obstacles -r /perception/obstacles_mapping:=/X/obstacles_mapping -r /perception/floor_points:=/X/floor_points -r /perception/floor_status:=/X/floor_status"
ros2 run imu_filter_madgwick imu_filter_madgwick_node --ros-args -p use_mag:=false -p publish_tf:=false -p world_frame:=enu -r imu/data_raw:=/camera/imu/data_raw -r imu/data:=/camera/imu/data > /tmp/ab_imu.log 2>&1 &
ros2 run roomba_perception floor_obstacles --ros-args -r __node:=py_floor --params-file /tmp/fo_params.yaml ${R//X/py} > /tmp/ab_py.log 2>&1 &
$(ros2 pkg prefix roomba_rs)/lib/roomba_rs/floor_obstacles --ros-args -r __node:=rs_floor --params-file /tmp/fo_params.yaml ${R//X/rs} > /tmp/ab_rs.log 2>&1 &
sleep 5
PYPID=$(pgrep -f 'lib/roomba_perception/floor_obstacles --ros-args'); RSPID=$(pgrep -f 'lib/roomba_rs/floor_obstacles --ros-args')
echo "pids py $PYPID rs $RSPID"
ros2 bag play $BAG --topics /tf_static > /dev/null 2>&1
python3 "$(dirname "$0")/ab_collect.py" $((DUR + 8)) > /tmp/ab_result.txt 2>&1 &
COL=$!
cpu() { awk '{print $14 + $15}' /proc/$1/stat; }
P0=$(cpu $PYPID); R0=$(cpu $RSPID); T0=$(date +%s.%N)
ros2 bag play $BAG --start-offset $OFF --playback-duration $DUR --topics /camera/depth/image_rect_raw /camera/depth/camera_info /camera/imu/data_raw > /tmp/ab_play.log 2>&1
P1=$(cpu $PYPID); R1=$(cpu $RSPID); T1=$(date +%s.%N)
wait $COL
HZ=$(getconf CLK_TCK)
echo "CPU over replay (% of one core): python $(echo "($P1-$P0)/$HZ/($T1-$T0)*100" | bc -l | cut -c1-5)  rust $(echo "($R1-$R0)/$HZ/($T1-$T0)*100" | bc -l | cut -c1-5)"
cat /tmp/ab_result.txt | grep -v Warn
kill -TERM $(pgrep -f 'lib/roomba_rs/floor_obstacles --ros-args') 2>/dev/null; kill -INT $(pgrep -f 'lib/roomba_perception/floor_obstacles --ros-args') $(pgrep -f imu_filter_madgwick_node) 2>/dev/null; sleep 2
tail -2 /tmp/ab_rs.log | cut -c1-200