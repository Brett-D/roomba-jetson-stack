#!/usr/bin/env python3
"""Exact A/B of ekf_inputs: feeds the same recorded message sequence (no ROS transport, so nothing
is dropped) to the Python node's callbacks and to the Rust core (examples/ekf_replay.rs) and
compares every output. The monotonic clock is driven by the bag's receive times in both.
  tools/ab_ekf_offline.py BAG_DIR [offset_s] [duration_s]
"""
import math, os, subprocess, sys
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import roomba_perception.ekf_inputs as E

bag, off, dur = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 150.0, float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
rclpy.init()
node = E.EkfInputs()
sim = [0.0]
E.time.monotonic = lambda: sim[0]
py_out = []
node.pub_wheel.publish = lambda m: py_out.append(('W', m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.twist.twist.linear.x,
                                                   m.twist.twist.angular.z, m.twist.covariance[0], m.twist.covariance[35]))
node.pub_imu.publish = lambda m: py_out.append(('I', m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.angular_velocity.x,
                                                 m.angular_velocity.y, m.angular_velocity.z, math.sqrt(m.angular_velocity_covariance[0])))
reader = SequentialReader()
reader.open(StorageOptions(uri=bag, storage_id='mcap'), ConverterOptions('', ''))
reader.set_filter(StorageFilter(topics=['/wheel/odom', '/camera/imu/data_raw']))
start = reader.get_metadata().starting_time.nanoseconds + int(off * 1e9)
reader.seek(start)
n = node.noise_n or 0.0
events = [f'C,{n!r},{(node.noise_k or 0.0)!r},{node.imu_rate!r},{node.bias[0]!r},{node.bias[1]!r},{node.bias[2]!r},{node.quiet_rate!r}']
while reader.has_next():
    topic, data, t_ns = reader.read_next()
    if t_ns > start + dur * 1e9:
        break
    sim[0] = t_ns * 1e-9
    if topic == '/wheel/odom':
        m = deserialize_message(data, Odometry)
        node._on_wheel(m)
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        events.append(f'W,{sim[0]!r},{st!r},{m.twist.twist.linear.x!r},{m.twist.twist.angular.z!r}')
    else:
        m = deserialize_message(data, Imu)
        node._on_imu(m)
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        w = m.angular_velocity
        events.append(f'I,{sim[0]!r},{st!r},{w.x!r},{w.y!r},{w.z!r}')
rs = subprocess.run([sys.argv[4] if len(sys.argv) > 4 else os.path.expanduser('~/rust_probe/target/release/examples/ekf_replay')],
                    input='\n'.join(events) + '\n', capture_output=True, text=True, check=True).stdout.split()
rs_out = [(p[0], *map(float, p[1:])) for p in (l.split(',') for l in rs)]
print(f'events: wheel {sum(e[0] == "W" for e in events)}  gyro {sum(e[0] == "I" for e in events)}')
for kind, cols in (('W', ['vx', 'wz', 'cov0', 'cov35']), ('I', ['mx', 'my', 'mz', 'sd'])):
    a = [o for o in py_out if o[0] == kind]; b = [o for o in rs_out if o[0] == kind]
    worst = [max((abs(x[i] - y[i]) for x, y in zip(a, b)), default=0.0) for i in range(2, 6)]
    rel = max((abs(x[5] - y[5]) / max(abs(x[5]), 1e-30) for x, y in zip(a, b)), default=0.0)
    stamps = sum(abs(x[1] - y[1]) < 1e-9 for x, y in zip(a, b))
    print(f'{kind}: python {len(a)}  rust {len(b)}  same stamps {stamps}  worst abs diff ' +
          '  '.join(f'{c} {w:.1e}' for c, w in zip(cols, worst)) + f'  (last col relative {rel:.1e})')
print(f'bias: python {[round(math.degrees(x), 5) for x in node.bias]} deg/s, samples {node.bias_samples}')