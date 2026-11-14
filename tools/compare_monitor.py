#!/usr/bin/env python3
"""Collects SLAM metrics during a bag replay (scripts/compare_pipelines.sh).

  python3 compare_monitor.py OUT_PREFIX --ros-args -p use_sim_time:=true

Runs until SIGINT/SIGTERM, then writes OUT_PREFIX.json (metrics) and OUT_PREFIX_grid.npz (the
final RTAB-Map occupancy grid: data, resolution, origin) for compare_render.py.

Metrics: visual tracking losses (/visual_slam/status vo_state OK -> FAILED, the same topic for
Basalt and cuVSLAM), vslam_watchdog recoveries / filtered frames, RTAB-Map loop closures and
proximity detections (/rtabmap/info), nodes, odom and map path lengths, largest single odom step,
the final pose relative to the first one in map and odom (≈ loop error if the drive returned to
its start), and the final grid's known/free/occupied area.
"""
import json
import math
import signal
import sys

import numpy as np
import rclpy
import rclpy.executors
from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rtabmap_msgs.msg import Info
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

VO_OK, VO_FAILED = 1, 2


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Monitor(Node):

    def __init__(self, prefix):
        super().__init__('compare_monitor')
        self.prefix = prefix
        self.m = {'tracking_losses': 0, 'tracking_failed_s': 0.0, 'status_msgs': 0,
                  'watchdog_recoveries': 0, 'watchdog_filtered': 0, 'watchdog_wheel_followed': 0,
                  'loop_closures': 0, 'proximity_closures': 0, 'rtabmap_nodes': 0,
                  'rtabmap_updates': 0}
        self.vo_state = None
        self.fail_t = None
        self.grid = None
        self.tracks = {'odom': [], 'map': []}
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.create_subscription(VisualSlamStatus, '/visual_slam/status', self._on_vo, 20)
        self.create_subscription(String, '/vslam_filter/status', self._on_filter, 5)
        self.create_subscription(Info, '/rtabmap/info', self._on_info, 10)
        self.create_subscription(
            OccupancyGrid, '/rtabmap/map', lambda g: setattr(self, 'grid', g),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        self.create_timer(0.2, self._sample)

    def _on_vo(self, msg):
        self.m['status_msgs'] += 1
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if msg.vo_state == VO_FAILED and self.vo_state == VO_OK:
            self.m['tracking_losses'] += 1
            self.fail_t = t
        if msg.vo_state == VO_OK and self.fail_t is not None:
            self.m['tracking_failed_s'] += max(0.0, t - self.fail_t)
            self.fail_t = None
        self.vo_state = msg.vo_state

    def _on_filter(self, msg):
        try:
            s = json.loads(msg.data)
        except ValueError:
            return
        self.m['watchdog_recoveries'] = max(self.m['watchdog_recoveries'], s.get('recoveries', 0))
        self.m['watchdog_filtered'] = max(self.m['watchdog_filtered'], s.get('filtered', 0))
        self.m['watchdog_wheel_followed'] = max(self.m['watchdog_wheel_followed'], s.get('wheel_followed', 0))

    def _on_info(self, msg):
        self.m['rtabmap_updates'] += 1
        if msg.loop_closure_id > 0:
            self.m['loop_closures'] += 1
        if msg.proximity_detection_id > 0:
            self.m['proximity_closures'] += 1
        self.m['rtabmap_nodes'] = max(self.m['rtabmap_nodes'], msg.ref_id)

    def _sample(self):
        for frame, track in self.tracks.items():
            try:
                tf = self.tf.lookup_transform(frame, 'base_link', Time())
            except Exception:  # noqa: BLE001
                continue
            st = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            tr = tf.transform.translation
            if not track or st > track[-1][0]:
                track.append((st, tr.x, tr.y, _yaw(tf.transform.rotation)))

    def finish(self):
        m = self.m
        for frame, tr in self.tracks.items():
            if len(tr) < 2:
                m[f'{frame}_samples'] = len(tr)
                continue
            a = np.array(tr)
            steps = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 2]))
            dyaw = math.degrees(math.atan2(math.sin(a[-1, 3] - a[0, 3]), math.cos(a[-1, 3] - a[0, 3])))
            m[f'{frame}_samples'] = len(tr)
            m[f'{frame}_duration_s'] = round(float(a[-1, 0] - a[0, 0]), 1)
            m[f'{frame}_path_m'] = round(float(steps.sum()), 3)
            m[f'{frame}_max_step_m'] = round(float(steps.max()), 3)
            m[f'{frame}_end_vs_start_m'] = round(float(math.hypot(a[-1, 1] - a[0, 1], a[-1, 2] - a[0, 2])), 3)
            m[f'{frame}_end_vs_start_yaw_deg'] = round(dyaw, 2)
            np.save(f'{self.prefix}_{frame}_track.npy', a)
        if self.grid is not None:
            g = self.grid
            d = np.array(g.data, np.int8).reshape(g.info.height, g.info.width)
            res = g.info.resolution
            m['grid_resolution_m'] = round(res, 3)
            m['grid_known_cells'] = int((d >= 0).sum())
            m['grid_free_cells'] = int((d == 0).sum())
            m['grid_occupied_cells'] = int((d >= 65).sum())
            m['grid_known_area_m2'] = round(m['grid_known_cells'] * res * res, 2)
            m['grid_free_area_m2'] = round(m['grid_free_cells'] * res * res, 2)
            m['grid_occupied_area_m2'] = round(m['grid_occupied_cells'] * res * res, 2)
            np.savez_compressed(f'{self.prefix}_grid.npz', data=d, resolution=res,
                                origin=(g.info.origin.position.x, g.info.origin.position.y))
        else:
            m['grid_known_cells'] = None
        with open(f'{self.prefix}.json', 'w') as f:
            json.dump(m, f, indent=1)
        print(json.dumps(m))


def main():
    rclpy.init(args=sys.argv)
    prefix = rclpy.utilities.remove_ros_args(sys.argv)[1]
    node = Monitor(prefix)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    node.finish()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
