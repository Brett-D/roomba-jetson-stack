"""Image throughput probe: img_probe.py pub|sub <secs> [width height hz]"""
import os
import sys
import time

if os.environ.get('TWIN_ROS_PY'):
    sys.path.append(os.environ['TWIN_ROS_PY'])

import rclpy  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

role, secs = sys.argv[1], float(sys.argv[2])
w, h, hz = (int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (848, 480, 30.0)
rclpy.init()
node = rclpy.create_node('img_probe_' + role)
if role == 'pub':
    pub = node.create_publisher(Image, '/pc_wsl_probe/image', qos_profile_sensor_data)
    msg = Image(width=w, height=h, encoding='bgr8', step=w * 3)
    msg.data = bytes(w * h * 3)
    t0 = time.time()
    n = 0
    while time.time() - t0 < secs:
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        n += 1
        time.sleep(max(0.0, t0 + n / hz - time.time()))
    print(f'[pub] {n} images {w}x{h} bgr8 ({len(msg.data) / 1e6:.2f} MB) in {time.time() - t0:.1f}s', flush=True)
else:
    stats = {'n': 0, 'bytes': 0, 'lat': [], 't_first': None}

    def cb(m):
        now = node.get_clock().now().nanoseconds * 1e-9
        stats['n'] += 1
        stats['bytes'] += len(m.data)
        stats['lat'].append(now - (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9))
        if stats['t_first'] is None:
            stats['t_first'] = time.time()
        stats['t_last'] = time.time()
    node.create_subscription(Image, '/pc_wsl_probe/image', cb, qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < secs:
        rclpy.spin_once(node, timeout_sec=0.1)
    dur = stats.get('t_last', t0) - (stats['t_first'] or t0)
    lat = sorted(stats['lat'])
    med = lat[len(lat) // 2] * 1000 if lat else float('nan')
    print(f'[sub] {stats["n"]} images, {stats["n"] / max(dur, 1e-6):.1f} Hz, {stats["bytes"] / max(dur, 1e-6) / 1e6:.1f} MB/s, '
          f'latency median {med:.1f} ms (clock offsets between Windows and WSL included)', flush=True)
node.destroy_node()
rclpy.shutdown()
