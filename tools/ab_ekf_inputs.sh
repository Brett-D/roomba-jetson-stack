#!/usr/bin/env bash
# A/B check of the Rust ekf_inputs (roomba_rs) against the Python one (roomba_perception): replays
# DUR s of a recorded drive (wheel odometry + raw gyro) from OFF s in, runs both nodes on remapped
# topics and compares every /ekf/imu and /ekf/wheel_odom message by stamp, plus CPU use.
#   tools/ab_ekf_inputs.sh /data/bags/robot_..._fresh_map8        (OFF=150 DUR=90 by default)
# The gyro subscription is best effort with a queue of 5: bursty bag playback drops samples, so
# compare the logic with RATE=0.2 (slow replay, nothing dropped); RATE=1 measures CPU.
source /opt/ros/jazzy/setup.bash; source ~/roomba_ws/install/setup.bash
BAG=${1:?usage: ab_ekf_inputs.sh BAG_DIR  (env OFF=start s, DUR=seconds, RS=rust binary)}; OFF=${OFF:-150}; DUR=${DUR:-90}
RS=${RS:-$(ros2 pkg prefix roomba_rs)/lib/roomba_rs/ekf_inputs}
R="-r /ekf/imu:=/X/imu -r /ekf/wheel_odom:=/X/wheel_odom -r /ekf/inputs_status:=/X/inputs_status"
ros2 run roomba_perception ekf_inputs --ros-args -r __node:=py_ekf ${R//X/py} > /tmp/ab_ekf_py.log 2>&1 &
$RS --ros-args -r __node:=rs_ekf ${R//X/rs} > /tmp/ab_ekf_rs.log 2>&1 &
sleep 6
PYPID=$(pgrep -f 'lib/roomba_perception/ekf_inputs --ros-args'); RSPID=$(pgrep -f "$(basename $(dirname $RS))/ekf_inputs --ros-args" | head -1)
echo "pids py $PYPID rs $RSPID"
python3 "$(dirname "$0")/ab_ekf_collect.py" $(echo "$DUR / ${RATE:-1.0} + 6" | bc) > /tmp/ab_ekf_result.txt 2>&1 &
COL=$!
cpu() { awk '{print $14 + $15}' /proc/$1/stat; }
P0=$(cpu $PYPID); R0=$(cpu $RSPID); T0=$(date +%s.%N)
ros2 bag play $BAG --rate ${RATE:-1.0} --start-offset $OFF --playback-duration $DUR --topics /wheel/odom /camera/imu/data_raw > /tmp/ab_ekf_play.log 2>&1
P1=$(cpu $PYPID); R1=$(cpu $RSPID); T1=$(date +%s.%N)
wait $COL
HZ=$(getconf CLK_TCK)
echo "CPU over replay (% of one core): python $(echo "($P1-$P0)/$HZ/($T1-$T0)*100" | bc -l | cut -c1-5)  rust $(echo "($R1-$R0)/$HZ/($T1-$T0)*100" | bc -l | cut -c1-5)"
grep -v Warn /tmp/ab_ekf_result.txt
kill -TERM $RSPID 2>/dev/null; kill -INT $PYPID 2>/dev/null; sleep 2
grep -aE "Gyro noise|Initial gyro bias" /tmp/ab_ekf_py.log /tmp/ab_ekf_rs.log | sed -E 's/\[[0-9]+\.[0-9]+\] //' | cut -c1-230