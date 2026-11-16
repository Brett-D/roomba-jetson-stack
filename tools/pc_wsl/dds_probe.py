"""Tiny cross-host ROS 2 probe: publishes /pc_wsl_probe/<me> and counts /pc_wsl_probe/<other>."""
import os
import sys
import time

if os.environ.get('TWIN_ROS_PY'):
    sys.path.append(os.environ['TWIN_ROS_PY'])

import rclpy  # noqa: E402
from std_msgs.msg import String  # noqa: E402

me, other, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
rclpy.init()
node = rclpy.create_node('dds_probe_' + me)
got = []
node.create_subscription(String, '/pc_wsl_probe/' + other, lambda m: got.append(m.data), 10)
pub = node.create_publisher(String, '/pc_wsl_probe/' + me, 10)
t0 = time.time()
i = 0
first = None
while time.time() - t0 < secs:
    pub.publish(String(data=f'{me} {i}'))
    i += 1
    rclpy.spin_once(node, timeout_sec=0.1)
    if got and first is None:
        first = time.time() - t0
names = sorted(set(node.get_node_names()))
print(f'[{me}] rmw={os.environ.get("RMW_IMPLEMENTATION")} sent={i} received={len(got)} from {other} '
      f'first_after_s={None if first is None else round(first, 1)} last={got[-1:]} nodes={names}', flush=True)
node.destroy_node()
rclpy.shutdown()
