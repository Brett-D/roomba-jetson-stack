#!/usr/bin/env python3
"""Snapshot of an OccupancyGrid / costmap topic: cell counts and a PNG.

  python3 costmap_snapshot.py /local_costmap/costmap out.png [timeout_s]

Prints free / unknown / inflated / lethal cell counts. PNG: white free, grey unknown, blue
inflation, black lethal (up = +y).
"""
import sys
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def main():
    topic, out = sys.argv[1], sys.argv[2]
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    rclpy.init()
    n = Node('costmap_snapshot')
    got = {}
    for dur in (DurabilityPolicy.TRANSIENT_LOCAL, DurabilityPolicy.VOLATILE):
        n.create_subscription(OccupancyGrid, topic, lambda m: got.setdefault('m', m),
                              QoSProfile(depth=1, durability=dur, reliability=ReliabilityPolicy.RELIABLE))
    t_end = time.monotonic() + timeout
    while 'm' not in got and time.monotonic() < t_end:
        rclpy.spin_once(n, timeout_sec=0.1)
    if 'm' not in got:
        print(f'{topic}: no message within {timeout:.0f} s')
        return
    m = got['m']
    a = np.array(m.data, np.int16).reshape(m.info.height, m.info.width)
    img = np.full(a.shape + (3,), 128, np.uint8)
    img[a == 0] = (255, 255, 255)
    img[(a > 0) & (a < 99)] = (255, 160, 120)
    img[a >= 99] = (0, 0, 0)
    img = cv2.flip(img, 0)
    k = max(1, int(600 / max(a.shape)))
    cv2.imwrite(out, cv2.resize(img, (a.shape[1] * k, a.shape[0] * k), interpolation=cv2.INTER_NEAREST))
    print(f'{topic} [{m.header.frame_id}] {m.info.width}x{m.info.height} @ {m.info.resolution} m: '
          f'free {int((a == 0).sum())}, unknown {int((a < 0).sum())}, '
          f'inflated {int(((a > 0) & (a < 99)).sum())}, lethal {int((a >= 99).sum())} -> {out}')
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
