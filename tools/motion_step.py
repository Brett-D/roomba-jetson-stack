import math, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from rclpy.qos import qos_profile_sensor_data

def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

class T(Node):
    def __init__(self):
        super().__init__('motion_test')
        self.m = {}
        for k, topic in (('odom', '/odom'), ('wheel', '/wheel/odom'), ('vio', '/visual_slam/tracking/odometry')):
            self.create_subscription(Odometry, topic, lambda msg, k=k: self.m.__setitem__(k, msg), 10)
        self.gyro_yaw, self.acc, self.last_imu = 0.0, None, None
        self.create_subscription(Imu, '/camera/imu/data_raw', self.on_imu, qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, '/cmd_vel_nav', 1)
    def on_imu(self, msg):
        a = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        self.acc = a if self.acc is None else 0.98 * self.acc + 0.02 * a
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu is not None and self.acc is not None:
            w = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            self.gyro_yaw += float(w @ (self.acc / np.linalg.norm(self.acc))) * (t - self.last_imu)
        self.last_imu = t
    def spin_for(self, s):
        end = time.time() + s
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
    def snap(self):
        out = {k: (m.pose.pose.position.x, m.pose.pose.position.y, yaw(m.pose.pose.orientation)) for k, m in self.m.items()}
        out['gyro'] = (0.0, 0.0, self.gyro_yaw)
        return out
    def move(self, v, w, s):
        end = time.time() + s
        while time.time() < end:
            t = Twist(); t.linear.x = v; t.angular.z = w
            self.pub.publish(t); self.spin_for(0.1)
        for _ in range(3):
            self.pub.publish(Twist()); self.spin_for(0.05)
        self.spin_for(1.2)

step = sys.argv[1]
rclpy.init(); n = T(); n.spin_for(1.0)
while len(n.m) < 3 or n.last_imu is None: n.spin_for(0.2)
a = n.snap()
if step == 'forward': n.move(0.15, 0.0, 40 / 15)
elif step == 'reverse': n.move(-0.15, 0.0, 40 / 15)
elif step == 'left': n.move(0.0, 0.5, math.pi / 2 / 0.5)
elif step == 'right': n.move(0.0, -0.5, math.pi / 2 / 0.5)
elif step == 'fastleft': n.move(0.0, 1.0, math.pi / 2)
elif step == 'fastright': n.move(0.0, -1.0, math.pi / 2)
b = n.snap()
for k in ('odom', 'vio', 'wheel', 'gyro'):
    if k in a and k in b:
        d = math.hypot(b[k][0] - a[k][0], b[k][1] - a[k][1]) * 100
        y = math.degrees(math.atan2(math.sin(b[k][2] - a[k][2]), math.cos(b[k][2] - a[k][2])))
        name = {'odom': 'VSLAM (filtered)', 'vio': 'VIO raw', 'wheel': 'wheels', 'gyro': 'gyro'}[k]
        print(f'{step:8s} {name:17s}: {"" if k == "gyro" else f"{d:5.1f} cm, "}{y:+6.1f} deg', flush=True)