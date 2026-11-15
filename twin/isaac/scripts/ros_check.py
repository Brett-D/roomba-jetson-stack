"""Check the twin's ROS 2 topics like the real D1000 + Roomba stack would see them.

Run in a second terminal with the bundled Jazzy rclpy (see ros_env.ps1):
  . twin\\isaac\\ros_env.ps1 ; python ros_check.py --seconds 20 --drive

Reports per topic: message count, header-stamp rate (sim time) and wall rate, encoding, size,
frame_id, CameraInfo K/P, and with --drive sends /cmd_vel and checks the ground-truth pose moved.
Writes the summary to --out as JSON.
"""
import argparse
import json
import math
import os
import sys
import time

if os.environ.get('TWIN_ROS_PY'):
    sys.path.append(os.environ['TWIN_ROS_PY'])

import rclpy  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu
from tf2_msgs.msg import TFMessage

TOPICS = {
    '/camera/left/image_rect_color': (Image, 'bgr8'),
    '/camera/left/image_rect_mono': (Image, 'mono8'),
    '/camera/right/image_rect_mono': (Image, 'mono8'),
    '/camera/depth/image_rect_raw': (Image, '16UC1'),
    '/camera/left/camera_info': (CameraInfo, None),
    '/camera/right/camera_info': (CameraInfo, None),
    '/camera/depth/camera_info': (CameraInfo, None),
    '/camera/imu/data_raw': (Imu, None),
    '/wheel/odom': (Odometry, None),
    '/ground_truth/odom': (Odometry, None),
    '/clock': (Clock, None),
}


class Checker(Node):
    def __init__(self):
        super().__init__('twin_ros_check')
        self.stats = {t: {'n': 0, 'stamps': [], 'wall': [], 'sample': None} for t in TOPICS}
        self.tf_static = []
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        for topic, (typ, _) in TOPICS.items():
            # /clock is best effort (rclcpp ClockQoS); a best-effort reader also matches reliable writers
            qos = qos_profile_sensor_data if topic == '/clock' else reliable
            self.create_subscription(typ, topic, lambda m, t=topic: self.cb(t, m), qos)
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(TFMessage, '/tf_static', self.tf_cb, latched)
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.gt = None

    def tf_cb(self, msg):
        for tr in msg.transforms:
            t, q = tr.transform.translation, tr.transform.rotation
            self.tf_static.append({'parent': tr.header.frame_id, 'child': tr.child_frame_id,
                                   'xyz': [round(t.x, 5), round(t.y, 5), round(t.z, 5)],
                                   'xyzw': [round(q.x, 5), round(q.y, 5), round(q.z, 5), round(q.w, 5)]})

    def cb(self, topic, msg):
        s = self.stats[topic]
        s['n'] += 1
        s['wall'].append(time.monotonic())
        if hasattr(msg, 'header'):
            s['stamps'].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        elif hasattr(msg, 'clock'):
            s['stamps'].append(msg.clock.sec + msg.clock.nanosec * 1e-9)
        if topic == '/ground_truth/odom':
            self.gt = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if s['sample'] is None:
            d = {}
            if isinstance(msg, Image):
                d = {'encoding': msg.encoding, 'width': msg.width, 'height': msg.height, 'step': msg.step,
                     'frame_id': msg.header.frame_id, 'bytes': len(msg.data), 'is_bigendian': msg.is_bigendian}
            elif isinstance(msg, CameraInfo):
                d = {'frame_id': msg.header.frame_id, 'width': msg.width, 'height': msg.height,
                     'k': [round(v, 4) for v in msg.k], 'p': [round(v, 4) for v in msg.p],
                     'd': list(msg.d), 'distortion_model': msg.distortion_model}
            elif isinstance(msg, Imu):
                d = {'frame_id': msg.header.frame_id,
                     'accel': [round(msg.linear_acceleration.x, 3), round(msg.linear_acceleration.y, 3),
                               round(msg.linear_acceleration.z, 3)]}
            elif isinstance(msg, Odometry):
                d = {'frame_id': msg.header.frame_id, 'child_frame_id': msg.child_frame_id}
            s['sample'] = d


def rate(v):
    if len(v) < 3:
        return None
    span = v[-1] - v[0]
    return round((len(v) - 1) / span, 2) if span > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--drive', action='store_true', help='send /cmd_vel 0.2 m/s for half the time')
    ap.add_argument('--out', default=r'E:\twin\eval\ros_check.json')
    args = ap.parse_args()
    rclpy.init()
    node = Checker()
    t0 = time.monotonic()
    gt0 = None
    while time.monotonic() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.02)
        el = time.monotonic() - t0
        if args.drive and node.gt is not None:
            if gt0 is None and el > 2.0:
                gt0 = node.gt
            if gt0 is not None and el < args.seconds * 0.75:
                tw = Twist()
                tw.linear.x = 0.2
                tw.angular.z = 0.3
                node.cmd.publish(tw)
    out = {'seconds': args.seconds, 'topics': {}, 'tf_static': node.tf_static}
    for t, s in node.stats.items():
        out['topics'][t] = {'count': s['n'], 'stamp_rate_hz': rate(s['stamps']), 'wall_rate_hz': rate(s['wall']),
                            'expected_encoding': TOPICS[t][1], **(s['sample'] or {})}
    if s := node.stats['/clock']['stamps']:
        w = node.stats['/clock']['wall']
        out['sim_time_s'] = round(s[-1] - s[0], 2)
        out['real_time_factor'] = round((s[-1] - s[0]) / (w[-1] - w[0]), 3) if w[-1] > w[0] else None
    if args.drive:
        out['drive_test'] = {'gt_start': gt0, 'gt_end': node.gt,
                             'moved_m': round(math.dist(gt0, node.gt), 3) if gt0 and node.gt else None}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    for t, v in out['topics'].items():
        print(f"{t:36s} n={v['count']:5d} stamp_hz={v['stamp_rate_hz']} wall_hz={v['wall_rate_hz']} "
              f"{v.get('encoding', '')} {v.get('frame_id', '')}")
    print('tf_static:', len(node.tf_static), 'transforms; rtf', out.get('real_time_factor'),
          'drive', out.get('drive_test'))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
