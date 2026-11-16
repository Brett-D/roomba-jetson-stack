"""Synthetic rectified stereo + depth camera for checking cuVSLAM / nvblox on the PC.

Three horizontal bands of the image see textured walls at 1.2, 1.8 and 2.4 m. The camera slides
left/right (0.3 m amplitude, 8 s period) and turns slightly; ground truth goes out as TF
odom -> base_link and /synth/gt_odom. Topics: /synth/{left,right}/image_rect (mono8),
/synth/{left,right}/camera_info, /synth/depth/image (16UC1, mm), /synth/depth/camera_info.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

W, H, F, B = 640, 400, 400.0, 0.12
CX, CY = W / 2, H / 2
BANDS = [(0, 130, 2.4), (130, 270, 1.8), (270, 400, 1.2)]
TEXEL = 0.02


class Synth(Node):
    def __init__(self):
        super().__init__('synth_stereo')
        rng = np.random.default_rng(7)
        self.tex = [rng.integers(0, 256, (512, 512), dtype=np.uint8) for _ in BANDS]
        self.pubs = {k: self.create_publisher(Image, f'/synth/{k}/image_rect' if k != 'depth' else '/synth/depth/image', 5)
                     for k in ('left', 'right', 'depth')}
        self.infos = {k: self.create_publisher(CameraInfo, f'/synth/{k}/camera_info', 5) for k in ('left', 'right', 'depth')}
        self.gt = self.create_publisher(Odometry, '/synth/gt_odom', 5)
        self.tf = TransformBroadcaster(self)
        st = StaticTransformBroadcaster(self)
        mounts = []
        for name, y in (('left_optical', 0.0), ('right_optical', -B)):
            t = TransformStamped()
            t.header.frame_id = 'base_link'
            t.child_frame_id = name
            t.transform.translation.y = y
            # optical: z forward, x right, y down
            t.transform.rotation.x, t.transform.rotation.y = -0.5, 0.5
            t.transform.rotation.z, t.transform.rotation.w = -0.5, 0.5
            mounts.append(t)
        st.sendTransform(mounts)
        self.t0 = self.get_clock().now()
        self.u = np.arange(W, dtype=np.float64)
        self.create_timer(1.0 / 30.0, self.tick)

    def info(self, frame, stamp, tx):
        m = CameraInfo()
        m.header.frame_id, m.header.stamp = frame, stamp
        m.width, m.height = W, H
        m.distortion_model = 'plumb_bob'
        m.d = [0.0] * 5
        m.k = [F, 0.0, CX, 0.0, F, CY, 0.0, 0.0, 1.0]
        m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        m.p = [F, 0.0, CX, tx, 0.0, F, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
        return m

    def render(self, xc, yaw, zoff=0.0):
        img = np.empty((H, W), np.uint8)
        depth = np.empty((H, W), np.uint16)
        ang = np.arctan2(self.u - CX, F) + yaw
        for (r0, r1, zw), tex in zip(BANDS, self.tex):
            z = zw - zoff
            rows = np.arange(r0, r1, dtype=np.float64)
            # wall at forward distance z (world frame, camera at lateral xc looking along +z, turned by yaw)
            xw = xc + z * np.tan(ang)
            zc = z * np.cos(ang - yaw) / np.cos(ang)        # depth along the optical axis
            yw = (rows - CY)[:, None] / F * zc[None, :]
            ix = np.floor(xw / TEXEL).astype(np.int64) % tex.shape[1]
            iy = np.floor(yw / TEXEL).astype(np.int64) % tex.shape[0]
            img[r0:r1] = tex[iy, np.broadcast_to(ix, iy.shape)]
            depth[r0:r1] = np.broadcast_to(np.clip(zc * 1000, 0, 65535).astype(np.uint16), (r1 - r0, W))
        return img, depth

    def tick(self):
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9
        xc = 0.3 * math.sin(2 * math.pi * t / 8.0)
        yaw = 0.05 * math.sin(2 * math.pi * t / 11.0)
        stamp = now.to_msg()
        left, depth = self.render(xc, yaw)
        right, _ = self.render(xc + B * math.cos(yaw), yaw, -B * math.sin(yaw))
        for k, data, enc, step in (('left', left, 'mono8', W), ('right', right, 'mono8', W), ('depth', depth, '16UC1', 2 * W)):
            m = Image()
            m.header.stamp = stamp
            m.header.frame_id = 'right_optical' if k == 'right' else 'left_optical'
            m.height, m.width, m.encoding, m.step = H, W, enc, step
            m.data = data.tobytes()
            self.pubs[k].publish(m)
        self.infos['left'].publish(self.info('left_optical', stamp, 0.0))
        self.infos['right'].publish(self.info('right_optical', stamp, -F * B))
        self.infos['depth'].publish(self.info('left_optical', stamp, 0.0))
        # base_link: x forward (optical z), y left (optical -x); the camera's lateral xc is base -y
        tr = TransformStamped()
        tr.header.stamp, tr.header.frame_id, tr.child_frame_id = stamp, 'odom', 'base_link'
        tr.transform.translation.y = -xc
        tr.transform.translation.z = 0.3
        tr.transform.rotation.z, tr.transform.rotation.w = math.sin(-yaw / 2), math.cos(-yaw / 2)
        self.tf.sendTransform(tr)
        o = Odometry()
        o.header, o.child_frame_id = tr.header, 'base_link'
        o.pose.pose.position.y = -xc
        o.pose.pose.orientation = tr.transform.rotation
        self.gt.publish(o)


def main():
    rclpy.init()
    rclpy.spin(Synth())


if __name__ == '__main__':
    main()
