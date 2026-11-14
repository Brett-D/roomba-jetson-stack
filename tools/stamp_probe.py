import rclpy, time
from rclpy.node import Node
from nav_msgs.msg import Odometry
from roomba_msgs.msg import SensorState
from std_msgs.msg import String
rclpy.init(); n = Node('stamp_probe')
st = {}
def cb(k):
    def f(m):
        st[k] = (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, n.get_clock().now().nanoseconds * 1e-9)
    return f
n.create_subscription(Odometry, '/odom', cb('odom'), 10)
n.create_subscription(Odometry, '/wheel/odom', cb('wheel'), 10)
n.create_subscription(Odometry, '/visual_slam/tracking/odometry', cb('vio'), 10)
s = {}
n.create_subscription(SensorState, '/roomba/sensors', lambda m: s.setdefault('m', m), 2)
f = {}
n.create_subscription(String, '/vslam_filter/status', lambda m: f.setdefault('m', m.data), 2)
end = time.time() + 3
while time.time() < end: rclpy.spin_once(n, timeout_sec=0.05)
for k, (stamp, recv) in st.items():
    print(f'{k:6s} stamp age at receipt: {(recv - stamp) * 1000:7.1f} ms')
m = s.get('m')
if m: print('bump_left', m.bump_left, 'bump_right', m.bump_right, 'wheel_drop', m.wheel_drop_left, m.wheel_drop_right, 'light_bump', [m.light_bump_left_signal, m.light_bump_front_left_signal, m.light_bump_center_left_signal, m.light_bump_center_right_signal, m.light_bump_front_right_signal, m.light_bump_right_signal])
print('filter', f.get('m'))