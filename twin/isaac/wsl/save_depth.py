#!/usr/bin/env python3
"""Save the depth images the stack receives (/camera/depth/image_rect_raw, 16UC1 mm) as 16-bit PNGs named by their
sim-time stamp (ns), for videos composed later with the Isaac recording (make_depth_video.py).

  save_depth.py --out /mnt/e/twin/videos/raw/<name>_depth [--seconds 120]
"""
import argparse
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seconds', type=float, default=600.0, help='wall-clock limit')
    a, _ = ap.parse_known_args()
    os.makedirs(a.out, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node('depth_saver')
    n = [0]

    def cb(m):
        if m.encoding != '16UC1':
            return
        d = np.frombuffer(m.data, np.uint16).reshape(m.height, m.step // 2)[:, :m.width]
        cv2.imwrite(os.path.join(a.out, f'{m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec}.png'), d)
        n[0] += 1

    node.create_subscription(Image, '/camera/depth/image_rect_raw', cb, QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
    t0 = time.time()
    try:
        while rclpy.ok() and time.time() - t0 < a.seconds and not os.path.exists(os.path.join(a.out, 'STOP')):
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        print('saved', n[0], flush=True)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
