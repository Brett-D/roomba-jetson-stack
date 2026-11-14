#!/usr/bin/env python3
"""Zone editor: exclusion zones (people inside are ignored) and navigation keep-out zones.

Drawing: in the 3D panel use "Publish point" and click the zone's corners on the map.
Clicking within close_distance_m of the first corner closes the polygon. Commands can
also be sent as std_msgs/String on ~/command: "close", "undo", "cancel", "clear", or JSON
ops from the web app ({"op": "add" | "delete" | "rename" | "clear", ...}). Zones have a
"kind": "exclusion" (default) or "keepout".

Keep-out zones become a Nav2 keep-out filter mask, grown by keepout_margin_m because Nav2
plans the robot's centre (the Roomba's body reaches 0.17 m beyond it). A zone can carry its
own "margin", and a "ttl_s": such temporary zones (e.g. bumper hits marked by the explorer)
expire on their own and are never saved with the map.

Zones are stored in the map frame, so they stay valid while the same map database is
used (mapping session that created it, or later runs with localization:=true). With
clear_on_start (set by the launch file in mapping mode) old zones are moved to .bak.

Subscribes:
  /clicked_point            geometry_msgs/PointStamped (any frame; transformed to map)
  ~/command                 std_msgs/String
Publishes:
  /perception/ignore_zones          std_msgs/String JSON (transient local, all kinds)
  /perception/ignore_zones/markers  visualization_msgs/MarkerArray
  /keepout_filter_mask              nav_msgs/OccupancyGrid (transient local)
  /costmap_filter_info              nav2_msgs/CostmapFilterInfo (transient local)
"""
import json
import math
import os
import time

import numpy as np
import rclpy
import rclpy.executors
from geometry_msgs.msg import Point, PointStamped
from nav2_msgs.msg import CostmapFilterInfo
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from roomba_perception.scene_filter import apply_transform

MAX_ZONES = 100
MAX_ZONE_POINTS = 64
KINDS = ('exclusion', 'keepout')


def keepout_mask(polygons, resolution, margin, margins=None):
    """OccupancyGrid data (row-major, 100 = keep out, 0 = free) covering the polygons plus margin.

    margins: optional per-polygon margins (None entries use `margin`).

    Returns (origin_x, origin_y, width, height, data) or None when there are no polygons.
    """
    if not polygons:
        return None
    margins = [margin if m is None else m for m in (margins or [None] * len(polygons))]
    pts = np.array([p for poly in polygons for p in poly], dtype=float)
    pad = max(margins) + 2 * resolution
    ox, oy = pts.min(axis=0) - pad
    w = int(math.ceil((pts[:, 0].max() + pad - ox) / resolution))
    h = int(math.ceil((pts[:, 1].max() + pad - oy) / resolution))
    xs = ox + (np.arange(w) + 0.5) * resolution
    ys = oy + (np.arange(h) + 0.5) * resolution
    cx, cy = np.meshgrid(xs, ys)
    blocked = np.zeros((h, w), dtype=bool)
    for poly, poly_margin in zip(polygons, margins):
        p = np.array(poly, dtype=float)
        q = np.roll(p, -1, axis=0)
        inside = np.zeros((h, w), dtype=bool)
        near = np.zeros((h, w), dtype=bool)
        for (x1, y1), (x2, y2) in zip(p, q):
            # Ray casting for the inside test.
            crosses = ((y1 > cy) != (y2 > cy)) & (cx < (x2 - x1) * (cy - y1) / ((y2 - y1) or 1e-12) + x1)
            inside ^= crosses
            # Distance to this edge for the margin.
            dx, dy = x2 - x1, y2 - y1
            t = np.clip(((cx - x1) * dx + (cy - y1) * dy) / ((dx * dx + dy * dy) or 1e-12), 0.0, 1.0)
            near |= np.hypot(cx - (x1 + t * dx), cy - (y1 + t * dy)) <= poly_margin
        blocked |= inside | near
    return float(ox), float(oy), w, h, np.where(blocked, 100, 0).astype(np.int8).ravel().tolist()


class IgnoreZones(Node):

    def __init__(self):
        super().__init__('ignore_zones')
        self.zones_file = os.path.expanduser(
            self.declare_parameter('zones_file', '~/maps/vslam.zones.json').value)
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.close_dist = float(self.declare_parameter('close_distance_m', 0.25).value)
        self.keepout_margin = float(self.declare_parameter('keepout_margin_m', 0.2).value)
        self.mask_resolution = float(self.declare_parameter('keepout_resolution_m', 0.05).value)

        self.tf_buffer = Buffer()
        # spin_thread=True spins this node from a second executor; under load it races the main
        # executor's wait set and the main loop stops delivering callbacks (seen live: the
        # obstacle stream froze and the collision monitor stopped the robot). The main
        # executor also processes /tf here, so lookups must not block.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.zones = []
        self.draft = []
        if bool(self.declare_parameter('clear_on_start', False).value) and os.path.exists(self.zones_file):
            # A new mapping session starts a new map frame; old zones would land in wrong places.
            os.replace(self.zones_file, self.zones_file + '.bak')
            self.get_logger().info(f'New map session: previous zones moved to {self.zones_file}.bak')
        self._load()

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_zones = self.create_publisher(String, '/perception/ignore_zones', latched)
        self.pub_markers = self.create_publisher(MarkerArray, '/perception/ignore_zones/markers', 5)
        self.pub_mask = self.create_publisher(OccupancyGrid, '/keepout_filter_mask', latched)
        self.pub_filter_info = self.create_publisher(CostmapFilterInfo, '/costmap_filter_info', latched)
        info = CostmapFilterInfo(type=0, filter_mask_topic='/keepout_filter_mask', base=0.0, multiplier=1.0)
        info.header.frame_id = self.map_frame
        info.header.stamp = self.get_clock().now().to_msg()
        self.pub_filter_info.publish(info)
        self.create_subscription(PointStamped, '/clicked_point', self._on_click, 10)
        self.create_subscription(String, '~/command', self._on_command, 10)
        self.create_timer(1.0, self._publish_markers)
        self.create_timer(2.0, self._expire)
        self._publish_zones()
        self.get_logger().info(f'{len(self.zones)} ignore zone(s) loaded from {self.zones_file}')

    def _load(self):
        try:
            with open(self.zones_file) as f:
                data = json.load(f)
            self.zones = data.get('zones', [])
        except FileNotFoundError:
            self.zones = []
        except (OSError, ValueError) as e:
            self.get_logger().error(f'Could not read {self.zones_file}: {e}')
            self.zones = []

    def _save(self):
        os.makedirs(os.path.dirname(self.zones_file), exist_ok=True)
        tmp = self.zones_file + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'frame': self.map_frame, 'zones': [z for z in self.zones if 'expires' not in z]},
                      f, indent=1)
        os.replace(tmp, self.zones_file)

    def _changed(self):
        self._save()
        self._publish_zones()
        self._publish_markers()

    def _publish_zones(self):
        self.pub_zones.publish(String(data=json.dumps({'frame': self.map_frame, 'zones': self.zones})))
        self._publish_mask()

    def _expire(self):
        now = time.time()
        expired = [z['name'] for z in self.zones if z.get('expires', math.inf) <= now]
        if expired:
            self.zones = [z for z in self.zones if z.get('expires', math.inf) > now]
            self.get_logger().info(f'Expired temporary zone(s): {", ".join(expired)}')
            self._changed()

    def _publish_mask(self):
        keep = [z for z in self.zones if z.get('kind') == 'keepout']
        mask = keepout_mask([z['points'] for z in keep], self.mask_resolution, self.keepout_margin,
                            [z.get('margin') for z in keep])
        if mask is None:
            # The filter needs a mask; a single free cell means "no keep-out anywhere".
            mask = (0.0, 0.0, 1, 1, [0])
        ox, oy, w, h, data = mask
        grid = OccupancyGrid()
        grid.header.frame_id = self.map_frame
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = self.mask_resolution
        grid.info.width, grid.info.height = w, h
        grid.info.origin.position.x, grid.info.origin.position.y = ox, oy
        grid.info.origin.orientation.w = 1.0
        grid.data = data
        self.pub_mask.publish(grid)

    def _next_name(self, kind='exclusion'):
        prefix = 'keepout' if kind == 'keepout' else 'zone'
        used = {z['name'] for z in self.zones}
        i = 1
        while f'{prefix}_{i}' in used:
            i += 1
        return f'{prefix}_{i}'

    def _close(self):
        if len(self.draft) < 3:
            self.get_logger().warn('A zone needs at least 3 corners')
            return
        name = self._next_name()
        self.zones.append({'name': name, 'points': self.draft})
        self.draft = []
        self.get_logger().info(f'Added {name}')
        self._changed()

    def _on_click(self, msg):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, msg.header.frame_id, Time())
        except TransformException as e:
            self.get_logger().warn(f'Cannot place point from {msg.header.frame_id}: {e}')
            return
        x, y, _ = apply_transform(tf, (msg.point.x, msg.point.y, msg.point.z))
        if len(self.draft) >= 3 and math.hypot(x - self.draft[0][0], y - self.draft[0][1]) < self.close_dist:
            self._close()
            return
        self.draft.append([round(x, 3), round(y, 3)])
        self.get_logger().info(f'Corner {len(self.draft)} at ({x:.2f}, {y:.2f})')
        self._publish_markers()

    def _on_command(self, msg):
        text = msg.data.strip()
        if text.startswith('{'):
            self._on_json_command(text)
            return
        cmd = text.lower()
        if cmd == 'close':
            self._close()
        elif cmd == 'cancel':
            self.draft = []
            self._publish_markers()
        elif cmd == 'undo':
            if self.draft:
                self.draft.pop()
                self._publish_markers()
            elif self.zones:
                removed = self.zones.pop()
                self.get_logger().info(f'Removed {removed["name"]}')
                self._changed()
        elif cmd == 'clear':
            self.zones, self.draft = [], []
            self.get_logger().info('Cleared all zones')
            self._changed()
        else:
            self.get_logger().warn(f'Unknown command "{msg.data}" (use close/undo/cancel/clear)')

    def _on_json_command(self, text):
        """JSON ops from the web app:
        {"op": "add", "points": [[x, y], ...], "name": "optional", "kind": "exclusion" | "keepout"}
        {"op": "delete", "name": "zone_1"}
        {"op": "rename", "name": "zone_1", "new_name": "tv"}
        {"op": "clear"}
        """
        try:
            cmd = json.loads(text)
            op = cmd.get('op')
            if op == 'add':
                kind = cmd.get('kind') or 'exclusion'
                if kind not in KINDS:
                    raise ValueError(f'kind must be one of {KINDS}')
                pts = cmd.get('points', [])
                if not (3 <= len(pts) <= MAX_ZONE_POINTS):
                    raise ValueError(f'zone needs 3..{MAX_ZONE_POINTS} points')
                pts = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in pts]
                if not all(math.isfinite(v) and abs(v) < 1000.0 for p in pts for v in p):
                    raise ValueError('point coordinates out of range')
                if len(self.zones) >= MAX_ZONES:
                    raise ValueError(f'at most {MAX_ZONES} zones')
                name = self._clean_name(cmd.get('name')) or self._next_name(kind)
                if any(z['name'] == name for z in self.zones):
                    raise ValueError(f'zone "{name}" already exists')
                zone = {'name': name, 'points': pts, 'kind': kind}
                if cmd.get('margin') is not None:
                    margin = float(cmd['margin'])
                    if not 0.0 <= margin <= 1.0:
                        raise ValueError('margin must be 0..1 m')
                    zone['margin'] = round(margin, 3)
                if cmd.get('ttl_s') is not None:
                    ttl = float(cmd['ttl_s'])
                    if not 0.0 < ttl <= 86400.0:
                        raise ValueError('ttl_s must be 0..86400')
                    zone['expires'] = round(time.time() + ttl, 1)
                self.zones.append(zone)
                self.get_logger().info(f'Added {kind} zone {name} ({len(pts)} corners)')
            elif op == 'delete':
                name = cmd.get('name')
                before = len(self.zones)
                self.zones = [z for z in self.zones if z['name'] != name]
                if len(self.zones) == before:
                    raise ValueError(f'no zone "{name}"')
                self.get_logger().info(f'Deleted {name}')
            elif op == 'rename':
                new = self._clean_name(cmd.get('new_name'))
                if not new or any(z['name'] == new for z in self.zones):
                    raise ValueError('invalid or duplicate new_name')
                for z in self.zones:
                    if z['name'] == cmd.get('name'):
                        z['name'] = new
                        break
                else:
                    raise ValueError(f'no zone "{cmd.get("name")}"')
            elif op == 'clear':
                self.zones, self.draft = [], []
            else:
                raise ValueError(f'unknown op "{op}"')
        except (ValueError, TypeError, KeyError, IndexError) as e:
            self.get_logger().warn(f'Rejected zone command: {e}')
            return
        self._changed()

    @staticmethod
    def _clean_name(name):
        if not name:
            return None
        name = ''.join(c for c in str(name) if c.isalnum() or c in ' _-').strip()[:40]
        return name or None

    def _publish_markers(self):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        clear = Marker(action=Marker.DELETEALL)
        clear.header.frame_id, clear.header.stamp = self.map_frame, stamp
        arr.markers.append(clear)

        def line(mid, pts, closed, rgba, ns):
            m = Marker(ns=ns, id=mid, type=Marker.LINE_STRIP, action=Marker.ADD)
            m.header.frame_id, m.header.stamp = self.map_frame, stamp
            m.pose.orientation.w = 1.0
            m.scale.x = 0.04
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=0.02) for p in pts]
            if closed and pts:
                m.points.append(Point(x=float(pts[0][0]), y=float(pts[0][1]), z=0.02))
            return m

        for i, z in enumerate(self.zones):
            pts = z['points']
            color = (0.9, 0.15, 0.15, 1.0) if z.get('kind') == 'keepout' else (1.0, 0.45, 0.0, 1.0)
            arr.markers.append(line(i, pts, True, color, 'zones'))
            t = Marker(ns='zone_names', id=i, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
                       text=z['name'])
            t.header.frame_id, t.header.stamp = self.map_frame, stamp
            t.pose.position.x = sum(p[0] for p in pts) / len(pts)
            t.pose.position.y = sum(p[1] for p in pts) / len(pts)
            t.pose.position.z = 0.15
            t.pose.orientation.w = 1.0
            t.scale.z = 0.2
            t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 0.6, 0.2, 1.0
            arr.markers.append(t)

        if self.draft:
            arr.markers.append(line(0, self.draft, False, (1.0, 1.0, 0.0, 1.0), 'draft'))
            dots = Marker(ns='draft_corners', id=0, type=Marker.SPHERE_LIST, action=Marker.ADD)
            dots.header.frame_id, dots.header.stamp = self.map_frame, stamp
            dots.pose.orientation.w = 1.0
            dots.scale.x = dots.scale.y = dots.scale.z = 0.08
            dots.color.r, dots.color.g, dots.color.b, dots.color.a = 1.0, 1.0, 0.0, 1.0
            dots.points = [Point(x=float(p[0]), y=float(p[1]), z=0.02) for p in self.draft]
            arr.markers.append(dots)
        self.pub_markers.publish(arr)


def main():
    rclpy.init()
    node = IgnoreZones()
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
