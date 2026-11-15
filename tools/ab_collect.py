"""Collects /py/* and /rs/* floor_obstacles outputs for tools/ab_floor_obstacles.sh."""
import sys, time, json, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
rclpy.init(); n = Node('ab_collect'); clouds = {'py': {}, 'rs': {}}; status = {'py': [], 'rs': []}
for k in ('py', 'rs'):
    n.create_subscription(PointCloud2, f'/{k}/obstacles', lambda m, k=k: clouds[k].__setitem__(
        (m.header.stamp.sec, m.header.stamp.nanosec), np.frombuffer(bytes(m.data), np.float32).reshape(-1, 3).copy()), QoSProfile(depth=50))
    n.create_subscription(String, f'/{k}/floor_status', lambda m, k=k: status[k].append(json.loads(m.data)), QoSProfile(depth=50))
end = time.time() + float(sys.argv[1])
while time.time() < end: rclpy.spin_once(n, timeout_sec=0.1)
common = sorted(set(clouds['py']) & set(clouds['rs']))
same_n = sum(len(clouds['py'][s]) == len(clouds['rs'][s]) for s in common)
maxdiff = 0.0; exact = 0
for s in common:
    a, b = clouds['py'][s], clouds['rs'][s]
    if len(a) == len(b):
        if len(a) == 0: exact += 1; continue
        d = np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).max(); maxdiff = max(maxdiff, d); exact += d < 1e-5
print(f"frames: py {len(clouds['py'])}  rs {len(clouds['rs'])}  same stamp {len(common)}  same point count {same_n}  identical points {exact}  max coord diff {maxdiff:.2e} m")
tot = lambda k, f: sum(s.get(f, 0) for s in status[k])
for f in ('obstacle_points', 'speckles_removed', 'floor_points'):
    print(f"  {f:17s} py {tot('py', f):8d}  rs {tot('rs', f):8d}")
print('  py last status', json.dumps(status['py'][-1]) if status['py'] else None)
print('  rs last status', json.dumps(status['rs'][-1]) if status['rs'] else None)
blind = [len([s for s in status[k] if s.get('blind')]) for k in ('py', 'rs')]; print('  blind frames py/rs', blind)