#!/usr/bin/env python3
"""Fills small gaps in RTAB-Map's occupancy grid so dotted obstacles become solid walls.

A railing's thin posts (and other sparse, see-through obstacles) reach the map as a dotted line
of occupied cells with free cells between them, which the planner happily routes through.
A morphological closing joins occupied cells that are at most gap_m apart; doorways and
corridors are far wider, so they stay open. Only cells are added, never removed.

Subscribes: /rtabmap/map (nav_msgs/OccupancyGrid, transient local)
Publishes:  /rtabmap/map_filled (nav_msgs/OccupancyGrid, transient local) - for the costmaps
"""
import cv2
import numpy as np
import rclpy
import rclpy.executors
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile


def fill_gaps(grid, resolution, gap_m, occupied=65):
    """Returns a copy of the int grid (-1 unknown, 0..100) with gaps up to gap_m between
    occupied cells closed (set to 100), plus how many cells were added."""
    occ = (grid >= occupied).astype(np.uint8)
    # A closing bridges gaps up to the kernel diameter minus one cell: radius = half the gap.
    r = max(1, int(np.ceil(gap_m / resolution / 2.0)))
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel)
    added = (closed > 0) & (occ == 0)
    out = grid.copy()
    out[added] = 100
    return out, int(added.sum())


class MapGapFiller(Node):

    def __init__(self):
        super().__init__('map_gap_filler')
        self.gap = float(self.declare_parameter('gap_m', 0.15).value)
        self.occupied = int(self.declare_parameter('occupied_threshold', 65).value)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(OccupancyGrid, self.declare_parameter(
            'output_topic', '/rtabmap/map_filled').value, latched)
        self.create_subscription(OccupancyGrid, self.declare_parameter(
            'input_topic', '/rtabmap/map').value, self._on_map, latched)
        self.logged = False

    def _on_map(self, msg):
        info = msg.info
        if not info.width or not info.height:
            self.pub.publish(msg)
            return
        grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        out, added = fill_gaps(grid, info.resolution, self.gap, self.occupied)
        filled = OccupancyGrid(header=msg.header, info=info)
        filled.data = out.astype(np.int8).ravel().tolist()
        self.pub.publish(filled)
        if not self.logged:
            self.logged = True
            self.get_logger().info(f'Filling gaps up to {self.gap:.2f} m between obstacles '
                                   f'({added} cells added to the first map)')


def main():
    rclpy.init()
    node = MapGapFiller()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()