#!/usr/bin/env python3
"""Splits the depth camera's view into floor and obstacles for Nav2.

Each depth frame (decimated) becomes 3D points in the camera optical frame. The IMU
orientation (imu_filter_madgwick, gravity-aligned) gives the "up" direction in that frame,
so every point gets a height relative to the camera. The floor is the lowest dense height
band; anything min_height..max_height above it is an obstacle. Floor points are published
separately so Nav2 can clear cells where an obstacle was removed.

Works with a hand-held camera at any height or tilt, as long as some floor is visible;
otherwise it keeps the last floor estimate for floor_memory_s, then uses
fallback_camera_height_m. On the robot (rigid mount) fixed_camera_height_m pins the floor
height instead. pause_turn_rate_rps > 0 holds back the costmap copies
(/perception/obstacles_mapping, /perception/floor_points) while the gyro turns faster than
that, because the pose lags and points would smear across the costmap; /perception/obstacles
(collision monitor) is always published. Logs a status line every 5 s.

Subscribes: /camera/depth/image_rect_raw, /camera/depth/camera_info, /camera/imu/data
Publishes:  /perception/obstacles, /perception/obstacles_mapping (same points, paused during
            fast turns), /perception/floor_points (PointCloud2, optical frame)
            /perception/floor_status (std_msgs/String JSON: camera height above floor)
"""
import json
import math
import time

import numpy as np
import rclpy
import rclpy.executors
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from roomba_perception.scene_filter import quat_rotate


def _cloud(header, pts):
    msg = PointCloud2(header=header, height=1, width=len(pts), is_dense=True, is_bigendian=False,
                      point_step=12, row_step=12 * len(pts))
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate('xyz')]
    msg.data = np.ascontiguousarray(pts, dtype=np.float32).tobytes()
    return msg


class FloorObstacles(Node):

    def __init__(self):
        super().__init__('floor_obstacles')
        p = self.declare_parameter
        self.decimation = int(p('decimation', 4).value)
        self.rate_hz = float(p('rate_hz', 6.0).value)
        self.min_range = float(p('min_range_m', 0.3).value)
        self.max_range = float(p('max_range_m', 4.0).value)
        self.min_h = float(p('min_obstacle_height_m', 0.06).value)
        self.max_h = float(p('max_obstacle_height_m', 1.0).value)
        self.floor_band = float(p('floor_band_m', 0.03).value)
        self.fallback_height = float(p('fallback_camera_height_m', 0.4).value)
        # Rigid robot mount: the floor is always this far below the camera (0 = estimate it).
        # The measured height is still reported, which is handy for checking the mount.
        self.fixed_height = float(p('fixed_camera_height_m', 0.0).value)
        self.floor_memory = float(p('floor_memory_s', 30.0).value)
        self.imu_frame = p('imu_frame', 'camera_imu_frame').value
        self.pause_turn_rate = float(p('pause_turn_rate_rps', 0.0).value)
        self.last_fast_turn = 0.0
        self.paused_frames = 0
        self.last_log = 0.0

        self.info = None
        self.imu_q = None
        self.floor_h = None       # floor height relative to camera (negative = below)
        self.floor_seen = 0.0
        self.last_proc = 0.0
        self.R_opt_imu = None
        self.tf_buffer = Buffer()
        # spin_thread=True spins this node from a second executor; under load it races the main
        # executor's wait set and the main loop stops delivering callbacks (seen live: the
        # obstacle stream froze and the collision monitor stopped the robot). The main
        # executor also processes /tf here, so lookups must not block.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.create_subscription(CameraInfo, '/camera/depth/camera_info', self._on_info, 2)
        self.create_subscription(Imu, '/camera/imu/data', self._on_imu, QoSProfile(depth=20))
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self._on_depth, 2)
        self.pub_obs = self.create_publisher(PointCloud2, '/perception/obstacles', 2)
        self.pub_obs_map = self.create_publisher(PointCloud2, '/perception/obstacles_mapping', 2)
        self.pub_floor = self.create_publisher(PointCloud2, '/perception/floor_points', 2)
        self.pub_status = self.create_publisher(String, '/perception/floor_status', 2)

    def _on_info(self, msg):
        self.info = msg

    def _on_imu(self, msg):
        o = msg.orientation
        self.imu_q = (o.x, o.y, o.z, o.w)
        w = msg.angular_velocity
        if self.pause_turn_rate > 0.0 and math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z) > self.pause_turn_rate:
            self.last_fast_turn = time.monotonic()

    def _up_in_optical(self, frame):
        """World 'up' expressed in the depth image's optical frame, or None."""
        if self.imu_q is None:
            return None
        if self.R_opt_imu is None:
            try:
                tf = self.tf_buffer.lookup_transform(frame, self.imu_frame, Time())
            except TransformException:
                return None
            r = tf.transform.rotation
            self.R_opt_imu = (r.x, r.y, r.z, r.w)
        x, y, z, w = self.imu_q
        up_imu = quat_rotate((-x, -y, -z, w), (0.0, 0.0, 1.0))  # inverse rotation of world up
        up = np.array(quat_rotate(self.R_opt_imu, up_imu))
        return up / np.linalg.norm(up)

    def _estimate_floor(self, h):
        """Lowest dense height band below the camera, refined by its median."""
        below = h[h < -0.08]
        if below.size < 50:
            return None
        bins = np.arange(-3.0, -0.06, 0.02)
        counts, edges = np.histogram(below, bins=bins)
        need = max(40, int(0.03 * h.size))
        dense = np.nonzero(counts >= need)[0]
        if dense.size == 0:
            return None
        lo = edges[dense[0]]
        band = below[(below >= lo - 0.03) & (below <= lo + 0.05)]
        return float(np.median(band)) if band.size else None

    def _on_depth(self, msg):
        now = time.monotonic()
        if self.info is None or now - self.last_proc < 1.0 / self.rate_hz:
            return
        self.last_proc = now
        up = self._up_in_optical(msg.header.frame_id)
        if up is None:
            return

        d = self.decimation
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
        depth = depth[::d, ::d].astype(np.float32) * 0.001
        fx, fy, cx, cy = self.info.p[0], self.info.p[5], self.info.p[2], self.info.p[6]
        v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
        z = depth.ravel()
        ok = (z > self.min_range) & (z < self.max_range)
        z = z[ok]
        x = (u.ravel()[ok] * d - cx) * z / fx
        y = (v.ravel()[ok] * d - cy) * z / fy
        pts = np.stack([x, y, z], axis=1)
        if pts.shape[0] < 100:
            return
        h = pts @ up

        est = self._estimate_floor(h)
        if self.fixed_height > 0.0:
            self.floor_h = -self.fixed_height
        elif est is not None:
            self.floor_h = est if self.floor_h is None else 0.7 * self.floor_h + 0.3 * est
            self.floor_seen = now
        elif self.floor_h is None or now - self.floor_seen > self.floor_memory:
            self.floor_h = -self.fallback_height
        rel = h - self.floor_h

        obstacles = pts[(rel > self.min_h) & (rel < self.max_h)]
        floor = pts[np.abs(rel) < self.floor_band]
        obs_cloud = _cloud(msg.header, obstacles)
        self.pub_obs.publish(obs_cloud)
        # Depth frames arrive ~80 ms late, so the costmap copies also pause just after a turn.
        if now - self.last_fast_turn < 0.3:
            self.paused_frames += 1
        else:
            self.pub_obs_map.publish(obs_cloud)
            self.pub_floor.publish(_cloud(msg.header, floor[::2]))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(up[1])))))
        status = {
            'camera_height_m': round(-self.floor_h, 2),
            'measured_height_m': round(-est, 3) if est is not None else None,
            'fixed_height': self.fixed_height > 0.0,
            'floor_visible': est is not None,
            'tilt_deg': round(tilt, 1),
            'obstacle_points': int(obstacles.shape[0]),
            'floor_points': int(floor.shape[0]),
            'paused_frames': self.paused_frames,
        }
        self.pub_status.publish(String(data=json.dumps(status)))
        if now - self.last_log > 5.0:
            self.last_log = now
            self.get_logger().info(f'floor: {status}')


def main():
    rclpy.init()
    node = FloorObstacles()
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
