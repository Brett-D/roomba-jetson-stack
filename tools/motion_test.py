import math, time, json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

class T(Node):
    def __init__(self):
        super().__init__('motion_test')
        self.vio = self.wh = None
        self.mode = None
        self.create_subscription(Odometry, '/odom', lambda m: setattr(self, 'vio', m), 10)
        self.create_subscription(Odometry, '/wheel/odom', lambda m: setattr(self, 'wh', m), 10)
        self.create_subscription(String, '/collision_monitor_state', lambda m: None, 1)
        self.pub = self.create_publisher(Twist, '/cmd_vel_nav', 1)
    def pose(self, m):
        p = m.pose.pose.position
        return p.x, p.y, yaw(m.pose.pose.orientation)
    def spin_for(self, s):
        end = time.time() + s
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
    def move(self, v, w, s):
        end = time.time() + s
        while time.time() < end:
            t = Twist(); t.linear.x = v; t.angular.z = w
            self.pub.publish(t)
            self.spin_for(0.1)
        for _ in range(3):
            self.pub.publish(Twist()); self.spin_for(0.05)
        self.spin_for(1.0)

rclpy.init(); n = T(); n.spin_for(2.0)
assert n.vio and n.wh, 'no odometry'
def snap():
    return n.pose(n.vio), n.pose(n.wh)
def report(name, a, b):
    (v0, w0), (v1, w1) = a, b
    dv = math.hypot(v1[0]-v0[0], v1[1]-v0[1]); dw = math.hypot(w1[0]-w0[0], w1[1]-w0[1])
    yv = math.degrees(math.atan2(math.sin(v1[2]-v0[2]), math.cos(v1[2]-v0[2])))
    yw = math.degrees(math.atan2(math.sin(w1[2]-w0[2]), math.cos(w1[2]-w0[2])))
    print(f'{name:14s} VSLAM: {dv*100:5.1f} cm, {yv:+6.1f} deg | wheels: {dw*100:5.1f} cm, {yw:+6.1f} deg', flush=True)
start = snap()
a = snap(); n.move(0.15, 0.0, 2.0); report('forward', a, snap())
a = snap(); n.move(0.0, 0.5, math.pi / 2 / 0.5); report('turn left', a, snap())
a = snap(); n.move(0.0, -0.5, math.pi / 2 / 0.5); report('turn right', a, snap())
a = snap(); n.move(-0.15, 0.0, 2.0); report('reverse', a, snap())
report('net', start, snap())