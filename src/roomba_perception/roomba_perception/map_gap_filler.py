#!/usr/bin/env python3
"""Fills small gaps in RTAB-Map's occupancy grid so dotted obstacles become solid walls.

A railing's thin posts (and other sparse, see-through obstacles) reach the map as a dotted line
of occupied cells with free cells between them, which the planner happily routes through.
A morphological closing joins occupied cells that are at most gap_m apart; doorways and
corridors are far wider, so they stay open. Only cells are added, never removed.

The depth camera often sees only every few posts of a railing, so the remaining dots can be
further apart than any safe closing. Those are joined by geometry instead: small dots that line
up (fence_min_posts or more, each within fence_spacing_m of the next, within fence_tolerance_m of
a straight line) are connected into a wall - never across a place the robot has driven through.

Subscribes: /rtabmap/map (nav_msgs/OccupancyGrid, transient local), /rtabmap/mapPath
Publishes:  /rtabmap/map_filled (nav_msgs/OccupancyGrid, transient local) - for the costmaps
"""
import cv2
import numpy as np
import rclpy
import rclpy.executors
from nav_msgs.msg import OccupancyGrid, Path
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


def fence_links(grid, resolution, spacing_m=0.45, tolerance_m=0.06, min_posts=3, max_dot_cells=6,
                occupied=65, keep_clear=None):
    """Pairs of dot centroids (cells, (x, y)) to join: runs of >= min_posts small occupied blobs on
    a straight line, consecutive ones at most spacing_m apart. keep_clear: bool mask of cells a
    link must not cross (where the robot has driven)."""
    occ = (grid >= occupied).astype(np.uint8)
    n, _, stats, cent = cv2.connectedComponentsWithStats(occ, connectivity=8)
    dots = np.array([cent[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] <= max_dot_cells])
    if len(dots) < min_posts:
        return []
    max_link, tol = spacing_m / resolution, tolerance_m / resolution
    diff = dots[None, :, :] - dots[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    links = set()
    for i, j in zip(*np.nonzero(np.triu((dist >= 1.5) & (dist <= max_link)))):
        a = dots[i]
        u = diff[i, j] / dist[i, j]
        off = dots - a
        along, across = off @ u, off @ np.array([-u[1], u[0]])
        on = np.nonzero((np.abs(across) <= tol) & (along >= -dist[i, j] - max_link * min_posts)
                        & (along <= 2 * dist[i, j] + max_link * min_posts))[0]
        if len(on) < min_posts:
            continue
        on = on[np.argsort(along[on])]
        pos = along[on]
        lo, hi = int(np.nonzero(on == i)[0][0]), int(np.nonzero(on == j)[0][0])
        if np.any(np.diff(pos[lo:hi + 1]) > max_link):
            continue
        while lo > 0 and pos[lo] - pos[lo - 1] <= max_link:
            lo -= 1
        while hi < len(on) - 1 and pos[hi + 1] - pos[hi] <= max_link:
            hi += 1
        if hi - lo + 1 < min_posts:
            continue
        for k in range(lo, hi):
            links.add((int(on[k]), int(on[k + 1])) if on[k] < on[k + 1] else (int(on[k + 1]), int(on[k])))
    out = []
    for i, j in links:
        p, q = np.round(dots[i]).astype(int), np.round(dots[j]).astype(int)
        if keep_clear is not None:
            line = np.zeros(grid.shape, np.uint8)
            cv2.line(line, tuple(p), tuple(q), 1, 1)
            if np.any(keep_clear & (line > 0)):
                continue
        out.append((tuple(p), tuple(q)))
    return out


def join_fences(grid, resolution, keep_clear=None, **kw):
    """Returns a copy of the grid with fence_links() drawn as occupied cells, and the link count."""
    links = fence_links(grid, resolution, keep_clear=keep_clear, **kw)
    if not links:
        return grid, 0
    mask = np.zeros(grid.shape, np.uint8)
    for p, q in links:
        cv2.line(mask, p, q, 1, 1)
    out = grid.copy()
    out[(mask > 0) & (grid < 65)] = 100
    return out, len(links)


class MapGapFiller(Node):

    def __init__(self):
        super().__init__('map_gap_filler')
        self.gap = float(self.declare_parameter('gap_m', 0.10).value)
        self.occupied = int(self.declare_parameter('occupied_threshold', 65).value)
        p = self.declare_parameter
        self.fence = bool(p('join_fences', True).value)
        self.fence_kw = dict(spacing_m=float(p('fence_spacing_m', 0.45).value),
                             tolerance_m=float(p('fence_tolerance_m', 0.06).value),
                             min_posts=int(p('fence_min_posts', 3).value),
                             max_dot_cells=int(p('fence_max_dot_cells', 6).value),
                             occupied=self.occupied)
        self.clear_radius = float(p('path_clear_radius_m', 0.17).value)   # robot radius
        self.path = []          # (x, y) of the robot's path in the map frame
        self.last_map = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(OccupancyGrid, self.declare_parameter(
            'output_topic', '/rtabmap/map_filled').value, latched)
        self.create_subscription(OccupancyGrid, self.declare_parameter(
            'input_topic', '/rtabmap/map').value, self._on_map, latched)
        self.create_subscription(Path, '/rtabmap/mapPath', self._on_path, QoSProfile(depth=1))
        self.logged = False
        self.fences = -1

    def _on_path(self, msg):
        self.path = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]

    def _driven_mask(self, info):
        if not self.path:
            return None
        mask = np.zeros((info.height, info.width), np.uint8)
        r = max(1, int(round(self.clear_radius / info.resolution)))
        pts = np.array([[(x - info.origin.position.x) / info.resolution,
                         (y - info.origin.position.y) / info.resolution] for x, y in self.path])
        pts = np.round(pts).astype(np.int32)
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 1, 2 * r + 1)
        return mask > 0

    def _on_map(self, msg):
        info = msg.info
        if not info.width or not info.height:
            self.pub.publish(msg)
            return
        grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        out, added = fill_gaps(grid, info.resolution, self.gap, self.occupied)
        if self.fence:
            # Detect on the raw grid (the closing would merge dots into blobs), then add both.
            fenced, links = join_fences(grid, info.resolution, self._driven_mask(info), **self.fence_kw)
            out = np.maximum(out, fenced)
            if links != self.fences:
                self.fences = links
                self.get_logger().info(f'{links} fence link(s) joining lined-up dots', throttle_duration_sec=30.0)
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