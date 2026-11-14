#!/usr/bin/env python3
"""Pose drift of a stationary robot: samples TF parent -> child for N seconds.

  python3 drift_check.py [seconds] [parent] [child]     (default 60 odom base_link)

Prints start/end pose, net drift (cm, deg) and the largest excursion from the start.
"""
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


def yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    parent = sys.argv[2] if len(sys.argv) > 2 else 'odom'
    child = sys.argv[3] if len(sys.argv) > 3 else 'base_link'
    rclpy.init()
    n = Node('drift_check')
    buf = Buffer()
    TransformListener(buf, n)
    samples = []
    t_end = time.monotonic() + dur + 2.0
    while time.monotonic() < t_end:
        rclpy.spin_once(n, timeout_sec=0.05)
        try:
            tf = buf.lookup_transform(parent, child, Time())
        except Exception:  # noqa: BLE001
            continue
        tr, q = tf.transform.translation, tf.transform.rotation
        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        if not samples or stamp > samples[-1][0]:
            samples.append((stamp, tr.x, tr.y, yaw(q)))
    if len(samples) < 2:
        print(f'No TF {parent} -> {child}')
        return
    s0, s1 = samples[0], samples[-1]
    d = math.hypot(s1[1] - s0[1], s1[2] - s0[2])
    dyaw = math.degrees(math.atan2(math.sin(s1[3] - s0[3]), math.cos(s1[3] - s0[3])))
    maxd = max(math.hypot(s[1] - s0[1], s[2] - s0[2]) for s in samples)
    maxy = max(abs(math.degrees(math.atan2(math.sin(s[3] - s0[3]), math.cos(s[3] - s0[3])))) for s in samples)
    print(f'{parent} -> {child} over {s1[0] - s0[0]:.1f} s ({len(samples)} samples): '
          f'drift {d * 100:.2f} cm, yaw {dyaw:+.3f} deg; max excursion {maxd * 100:.2f} cm, {maxy:.3f} deg')
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
