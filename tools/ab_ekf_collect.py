"""Collects /py/* and /rs/* ekf_inputs outputs for tools/ab_ekf_inputs.sh."""
import sys, time, json, math, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String
rclpy.init(); n = Node('ab_ekf_collect')
imu = {'py': {}, 'rs': {}}; odo = {'py': {}, 'rs': {}}; st = {'py': [], 'rs': []}
key = lambda h: (h.stamp.sec, h.stamp.nanosec)
for k in ('py', 'rs'):
    n.create_subscription(Imu, f'/{k}/imu', lambda m, k=k: imu[k].__setitem__(key(m.header), (
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z, m.angular_velocity_covariance[0])), QoSProfile(depth=500))
    n.create_subscription(Odometry, f'/{k}/wheel_odom', lambda m, k=k: odo[k].__setitem__(key(m.header), (
        m.twist.twist.linear.x, m.twist.twist.angular.z, m.twist.covariance[0], m.twist.covariance[35], m.pose.covariance[0], m.child_frame_id)), QoSProfile(depth=200))
    n.create_subscription(String, f'/{k}/inputs_status', lambda m, k=k: st[k].append(json.loads(m.data)), QoSProfile(depth=20))
end = time.time() + float(sys.argv[1])
while time.time() < end: rclpy.spin_once(n, timeout_sec=0.05)
def cmp(name, d, tol):
    common = sorted(set(d['py']) & set(d['rs']))
    worst = [0.0] * 4; equal = 0
    for s in common:
        a, b = d['py'][s], d['rs'][s]
        diffs = [abs(x - y) for x, y in zip(a, b) if isinstance(x, float)]
        for i, v in enumerate(diffs[:4]): worst[i] = max(worst[i], v)
        equal += all(v <= tol for v in diffs) and all(x == y for x, y in zip(a, b) if isinstance(x, str))
    print(f"{name}: py {len(d['py'])}  rs {len(d['rs'])}  same stamp {len(common)}  equal (<= {tol:g}) {equal}  worst diffs {['%.2e' % w for w in worst]}")
cmp('imu ', imu, 1e-9); cmp('wheel', odo, 1e-12)
for k in ('py', 'rs'):
    print(f'  {k} last status', json.dumps(st[k][-1]) if st[k] else None)