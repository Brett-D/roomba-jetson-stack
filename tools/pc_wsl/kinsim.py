"""Minimal stand-in for the robot base in the PC smoke test: integrates /cmd_vel (unicycle) into
odom -> base_link TF and /odom, and publishes empty obstacle / floor clouds so the costmaps and the
collision monitor have current (obstacle-free) sensor data."""
import math

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster


class KinSim(Node):
    def __init__(self):
        super().__init__('kinsim')
        self.x = self.y = self.th = 0.0
        self.v = self.w = 0.0
        self.last_cmd = 0.0
        self.dt = 1.0 / 30.0
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.odom = self.create_publisher(Odometry, '/odom', 10)
        self.tf = TransformBroadcaster(self)
        self.clouds = [self.create_publisher(PointCloud2, t, 2) for t in
                       ('/perception/obstacles', '/perception/obstacles_mapping', '/perception/floor_points')]
        self.create_timer(self.dt, self.step)
        self.create_timer(0.1, self.publish_clouds)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, msg):
        self.v, self.w = msg.linear.x, msg.angular.z
        self.last_cmd = self.now_s()

    def step(self):
        if self.now_s() - self.last_cmd > 0.5:
            self.v = self.w = 0.0
        self.th += self.w * self.dt
        self.x += self.v * math.cos(self.th) * self.dt
        self.y += self.v * math.sin(self.th) * self.dt
        stamp = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = math.sin(self.th / 2)
        t.transform.rotation.w = math.cos(self.th / 2)
        self.tf.sendTransform(t)
        o = Odometry()
        o.header = t.header
        o.child_frame_id = 'base_link'
        o.pose.pose.position.x = self.x
        o.pose.pose.position.y = self.y
        o.pose.pose.orientation = t.transform.rotation
        o.twist.twist.linear.x = self.v
        o.twist.twist.angular.z = self.w
        self.odom.publish(o)

    def publish_clouds(self):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.height = 1
        msg.width = 0
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate('xyz')]
        msg.point_step = 12
        msg.row_step = 0
        msg.is_dense = True
        for p in self.clouds:
            p.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(KinSim())


if __name__ == '__main__':
    main()
