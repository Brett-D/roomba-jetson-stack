#!/usr/bin/env python3
"""Checks the TF tree: every frame has one parent, the robot chain resolves, edge rates.

  python3 tf_check.py [seconds]

Prints each edge (parent -> child) with its rate (static edges marked), flags frames that
got more than one parent, and looks up map -> base_link -> camera optical/IMU frames.
"""
import sys
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

CHAIN = [('map', 'odom'), ('odom', 'base_link'), ('base_link', 'camera_link'),
         ('base_link', 'camera_left_optical_frame'), ('base_link', 'camera_imu_frame')]


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    rclpy.init()
    n = Node('tf_check')
    buf = Buffer()
    TransformListener(buf, n)
    counts = defaultdict(int)
    static = set()
    parents = defaultdict(set)
    stamps = defaultdict(list)

    def on_tf(msg, is_static):
        for t in msg.transforms:
            edge = (t.header.frame_id, t.child_frame_id)
            parents[t.child_frame_id].add(t.header.frame_id)
            if is_static:
                static.add(edge)
            else:
                counts[edge] += 1
                stamps[edge].append(t.header.stamp.sec + t.header.stamp.nanosec * 1e-9)

    n.create_subscription(TFMessage, '/tf', lambda m: on_tf(m, False), 100)
    n.create_subscription(TFMessage, '/tf_static', lambda m: on_tf(m, True),
                          QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    t_end = time.monotonic() + dur
    while time.monotonic() < t_end:
        rclpy.spin_once(n, timeout_sec=0.1)
    ok = True
    print('TF edges:')
    for edge in sorted(set(counts) | static):
        if edge in static:
            print(f'  {edge[0]} -> {edge[1]}  (static)')
        else:
            s = stamps[edge]
            dup = len(s) - len(set(s))
            back = sum(1 for a, b in zip(s, s[1:]) if b < a)
            print(f'  {edge[0]} -> {edge[1]}  {counts[edge] / dur:.1f} Hz'
                  + (f'  [{dup} duplicate stamps, {back} out of order: two publishers?]' if dup or back else ''))
            if back:
                ok = False
    for child, ps in sorted(parents.items()):
        if len(ps) > 1:
            ok = False
            print(f'  MULTIPLE PARENTS: {child} <- {sorted(ps)}')
    for a, b in CHAIN:
        try:
            buf.lookup_transform(a, b, Time())
            print(f'  lookup {a} -> {b}: ok')
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f'  lookup {a} -> {b}: FAILED ({type(e).__name__})')
    print('TF tree OK (single-parented, chain complete)' if ok else 'TF tree has PROBLEMS')
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
