"""Geometry helpers for deciding whether a detection should be ignored.

All geometry is 2D in the map frame (x, y). Zones are polygons drawn by the user;
line of sight is traced on RTAB-Map's 2D occupancy grid.
"""
import json
import math

import numpy as np


def quat_rotate(q, v):
    """Rotates vector v by quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def apply_transform(tf, p):
    """Applies a geometry_msgs/TransformStamped to point p (x, y, z)."""
    r = tf.transform.rotation
    t = tf.transform.translation
    x, y, z = quat_rotate((r.x, r.y, r.z, r.w), p)
    return (x + t.x, y + t.y, z + t.z)


def point_in_polygon(x, y, pts):
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class ZoneSet:
    """User-drawn ignore zones, received as JSON: {"frame": "map", "zones": [{"name", "points"}]}.
    Navigation keep-out zones (kind "keepout") share the message but are not ignore zones."""

    def __init__(self):
        self.frame = 'map'
        self.zones = []

    def load_json(self, text):
        data = json.loads(text)
        self.frame = data.get('frame', 'map')
        self.zones = [z for z in data.get('zones', [])
                      if len(z.get('points', [])) >= 3 and z.get('kind', 'exclusion') != 'keepout']

    def containing(self, x, y):
        for z in self.zones:
            if point_in_polygon(x, y, z['points']):
                return z.get('name', 'zone')
        return None


class GridLineOfSight:
    """Checks whether occupied cells of a nav_msgs/OccupancyGrid block a 2D ray."""

    def __init__(self, occupied_threshold=65, min_blocking_cells=2):
        self.occupied_threshold = occupied_threshold
        self.min_blocking_cells = min_blocking_cells
        self.grid = None
        self.res = None
        self.ox = self.oy = 0.0
        self.yaw = 0.0
        self.frame = None

    def update(self, msg):
        info = msg.info
        self.grid = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
        self.res = info.resolution
        o = info.origin
        self.ox, self.oy = o.position.x, o.position.y
        q = o.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.frame = msg.header.frame_id

    @property
    def ready(self):
        return self.grid is not None

    def _cell(self, x, y):
        dx, dy = x - self.ox, y - self.oy
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        gx, gy = c * dx - s * dy, s * dx + c * dy
        return int(math.floor(gx / self.res)), int(math.floor(gy / self.res))

    def blocked(self, start, end, skip_start_m=0.15, skip_end_m=0.35):
        """Returns the number of distinct occupied cells crossed between start and end, excluding
        skip_start_m near the camera and skip_end_m around the target (the target's own body)."""
        if not self.ready:
            return 0
        sx, sy = start
        ex, ey = end
        length = math.hypot(ex - sx, ey - sy)
        if length <= skip_start_m + skip_end_m:
            return 0
        h, w = self.grid.shape
        step = self.res * 0.5
        n = int((length - skip_start_m - skip_end_m) / step) + 1
        ux, uy = (ex - sx) / length, (ey - sy) / length
        hits = set()
        for i in range(n):
            d = skip_start_m + i * step
            cx, cy = self._cell(sx + ux * d, sy + uy * d)
            if 0 <= cx < w and 0 <= cy < h and self.grid[cy, cx] >= self.occupied_threshold:
                hits.add((cx, cy))
        return len(hits)
