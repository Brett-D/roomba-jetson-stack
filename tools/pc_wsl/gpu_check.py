"""Compares cuVSLAM odometry with the synthetic ground truth and reads nvblox's ESDF slice."""
import math
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from nvblox_msgs.msg import DistanceMapSlice
from rclpy.node import Node


class Check(Node):
    def __init__(self):
        super().__init__('gpu_check')
        self.gt = []
        self.vo = []
        self.slices = []
        self.create_subscription(Odometry, '/synth/gt_odom', lambda m: self.gt.append(self.xyz(m)), 50)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', lambda m: self.vo.append(self.xyz(m)), 50)
        self.create_subscription(DistanceMapSlice, '/nvblox_node/static_map_slice', self.slices.append, 5)

    @staticmethod
    def xyz(m):
        p = m.pose.pose.position
        return (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.x, p.y, p.z)


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30
    rclpy.init()
    n = Check()
    t0 = time.time()
    while time.time() - t0 < secs:
        rclpy.spin_once(n, timeout_sec=0.1)
    print(f'gt msgs {len(n.gt)}, cuVSLAM odometry msgs {len(n.vo)} ({len(n.vo) / secs:.1f} Hz)')
    if len(n.vo) > 10 and n.gt:
        gt = {round(t, 3): (x, y) for t, x, y, _ in n.gt}
        pairs = [(gt[round(t, 3)], (x, y)) for t, x, y, _ in n.vo if round(t, 3) in gt]
        if pairs:
            (gx0, gy0), (vx0, vy0) = pairs[0]
            errs = [math.hypot((vx - vx0) - (gx - gx0), (vy - vy0) - (gy - gy0)) for (gx, gy), (vx, vy) in pairs]
            span = max(g[1] for g, _ in pairs) - min(g[1] for g, _ in pairs)
            vspan = max(v[1] for _, v in pairs) - min(v[1] for _, v in pairs)
            print(f'{len(pairs)} stamp-matched poses: ground-truth lateral span {span:.3f} m, cuVSLAM span {vspan:.3f} m, '
                  f'position error vs ground truth (start-aligned) median {sorted(errs)[len(errs) // 2] * 100:.1f} cm, '
                  f'max {max(errs) * 100:.1f} cm')
    if n.slices:
        s = n.slices[-1]
        known = [d for d in s.data if d != s.unknown_value]
        near = [d for d in known if d < 0.2]
        print(f'nvblox slices {len(n.slices)}; last {s.width}x{s.height} @ {s.resolution} m: {len(known)} known cells, '
              f'{len(near)} within 0.2 m of an obstacle')
    else:
        print('no nvblox slice received')


if __name__ == '__main__':
    main()
