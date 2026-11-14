#!/usr/bin/env python3
"""Person detection with distance for the Roomba robot.

YOLO (TensorRT engine) runs on the rectified left color image. Each detection's
distance comes from the aligned hardware depth image; if the box has too little
valid depth (beyond IR range, dark clothing), it falls back to block-matching the
rectified stereo pair inside the box.

Subscribes (exact-time synchronized; the D1000 driver gives all three one stamp):
  /camera/left/image_rect_color   sensor_msgs/Image (bgr8)
  /camera/depth/image_rect_raw    sensor_msgs/Image (16UC1, mm, registered to left)
  /camera/right/image_rect_mono   sensor_msgs/Image (mono8)
  /camera/left/camera_info, /camera/right/camera_info

Publishes:
  ~/annotated/compressed   sensor_msgs/CompressedImage  boxes + "person 1.82 m", plus a floor
                           grid (0.25 m squares on the map floor; toggle with ~/floor_grid Bool)
  ~/detections_2d          vision_msgs/Detection2DArray (all detections)
  ~/detections_3d          vision_msgs/Detection3DArray (relevant only; left optical frame, meters)
  ~/ignored_3d             vision_msgs/Detection3DArray (ignored: in an ignore zone or behind
                           occupied map cells; Detection3D.id holds the reason)
  ~/markers                visualization_msgs/MarkerArray (labeled 3D markers; ignored in gray)

Ignore filtering needs TF to the map frame (RTAB-Map + cuVSLAM). Without a map or TF,
nothing is ignored, so obstacle avoidance never loses a person because of missing data.
"""
import ctypes
import os
import time

import cv2
import message_filters
import numpy as np
import rclpy
import rclpy.executors
import tensorrt as trt
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import (Detection2D, Detection2DArray, Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)
from visualization_msgs.msg import Marker, MarkerArray

from roomba_perception.floor_grid import draw_floor_grid, quat_matrix
from roomba_perception.scene_filter import GridLineOfSight, ZoneSet, apply_transform

COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush']


class _Cuda:
    """Minimal CUDA runtime access via ctypes (avoids pycuda/cuda-python dependencies)."""

    def __init__(self):
        self.lib = None
        for name in ('libcudart.so', '/usr/local/cuda/lib64/libcudart.so'):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if self.lib is None:
            raise RuntimeError('libcudart.so not found')
        self.lib.cudaMallocManaged.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                               ctypes.c_uint]
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]

    @staticmethod
    def _check(rc, what):
        if rc != 0:
            raise RuntimeError(f'{what} failed (cudaError {rc})')

    def managed(self, nbytes):
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMallocManaged(ctypes.byref(ptr), nbytes, 1), 'cudaMallocManaged')
        return ptr.value

    def stream(self):
        s = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(s)), 'cudaStreamCreate')
        return s.value

    def sync(self, stream):
        self._check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)), 'cudaStreamSynchronize')


class TrtYolo:
    """Runs an Ultralytics YOLO (v8/11) TensorRT engine with output (1, 4+C, N)."""

    def __init__(self, engine_path):
        self.cuda = _Cuda()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'Failed to load TensorRT engine {engine_path}')
        self.ctx = self.engine.create_execution_context()
        self.stream = self.cuda.stream()
        # Jetson iGPU shares DRAM: managed buffers are written/read by numpy directly (sync first).
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            ptr = self.cuda.managed(nbytes)
            arr = np.ctypeslib.as_array((ctypes.c_byte * nbytes).from_address(ptr)).view(dtype).reshape(shape)
            self.ctx.set_tensor_address(name, ptr)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inp = arr
            else:
                self.out = arr
        self.in_h, self.in_w = self.inp.shape[2], self.inp.shape[3]

    def preprocess(self, bgr):
        h, w = bgr.shape[:2]
        r = min(self.in_w / w, self.in_h / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else bgr
        px, py = (self.in_w - nw) // 2, (self.in_h - nh) // 2
        canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = img
        self.inp[0] = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)
        return r, px, py

    def infer(self):
        if not self.ctx.execute_async_v3(self.stream):
            raise RuntimeError('TensorRT execution failed')
        self.cuda.sync(self.stream)
        return np.array(self.out[0], copy=True)


class PersonDetector(Node):

    def __init__(self):
        super().__init__('person_detector')
        engine = self.declare_parameter(
            'engine_path', os.path.expanduser('~/models/yolo11n_384x640_fp16.engine')).value
        self.conf = float(self.declare_parameter('conf_threshold', 0.45).value)
        self.nms = float(self.declare_parameter('nms_threshold', 0.5).value)
        self.class_ids = [int(c) for c in self.declare_parameter('classes', [0]).value]
        self.jpeg_quality = int(self.declare_parameter('jpeg_quality', 80).value)
        self.min_depth_px = int(self.declare_parameter('min_depth_pixels', 15).value)

        self.yolo = TrtYolo(engine)
        self.get_logger().info(
            f'Loaded {engine} (input {self.yolo.in_w}x{self.yolo.in_h}); classes '
            f'{[COCO_NAMES[c] for c in self.class_ids]}')

        self.left_info = None
        self.baseline = None
        self.create_subscription(CameraInfo, '/camera/left/camera_info', self._on_left_info, 5)
        self.create_subscription(CameraInfo, '/camera/right/camera_info', self._on_right_info, 5)

        qos = QoSProfile(depth=5)
        subs = [
            message_filters.Subscriber(self, Image, '/camera/left/image_rect_color', qos_profile=qos),
            message_filters.Subscriber(self, Image, '/camera/depth/image_rect_raw', qos_profile=qos),
            message_filters.Subscriber(self, Image, '/camera/right/image_rect_mono', qos_profile=qos),
        ]
        self.sync = message_filters.TimeSynchronizer(subs, 10)
        self.sync.registerCallback(self._on_frames)

        self.pub_img = self.create_publisher(CompressedImage, '~/annotated/compressed', 5)
        self.pub_2d = self.create_publisher(Detection2DArray, '~/detections_2d', 5)
        self.pub_3d = self.create_publisher(Detection3DArray, '~/detections_3d', 5)
        self.pub_ignored = self.create_publisher(Detection3DArray, '~/ignored_3d', 5)
        self.pub_markers = self.create_publisher(MarkerArray, '~/markers', 5)

        # Ignore filtering (zones + line of sight on the 2D map).
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.use_zones = bool(self.declare_parameter('ignore_zones', True).value)
        self.use_los = bool(self.declare_parameter('line_of_sight', True).value)
        self.los = GridLineOfSight(
            int(self.declare_parameter('occupied_threshold', 65).value),
            int(self.declare_parameter('min_blocking_cells', 2).value))
        self.los_skip_end = float(self.declare_parameter('target_margin_m', 0.35).value)
        self.zones = ZoneSet()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        # RTAB-Map publishes the grid latched (transient local) and only on change.
        self.create_subscription(OccupancyGrid, self.declare_parameter('map_topic', '/rtabmap/map').value,
                                 self.los.update,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(
            String, '/perception/ignore_zones', self._on_zones,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._tf_warned = False
        self._tf_retry_after = 0.0
        # When reusing a saved map (localization mode), map positions are meaningless until
        # RTAB-Map recognizes the place, so zones/line of sight stay off until then.
        self.require_localization = bool(self.declare_parameter('require_localization', False).value)
        self.localized = not self.require_localization
        self.create_subscription(PoseWithCovarianceStamped, '/rtabmap/localization_pose',
                                 self._on_localized, QoSProfile(depth=1))

        self.stereo = cv2.StereoBM_create(numDisparities=160, blockSize=15)
        # Floor grid on the annotated image: world-fixed on the map floor, or on base_link's
        # floor (moving with the robot) until the map frame exists.
        self.grid_on = bool(self.declare_parameter('floor_grid', True).value)
        self.grid_frames = [f for f in (self.map_frame,
                                        self.declare_parameter('floor_grid_fallback_frame', 'base_link').value) if f]
        self.create_subscription(Bool, '~/floor_grid', self._on_grid_toggle, 2)
        self.n_frames = 0
        self.t_infer = 0.0
        self.t_total = 0.0
        self.create_timer(30.0, self._log_stats)

    def _on_left_info(self, msg):
        self.left_info = msg

    def _on_grid_toggle(self, msg):
        self.grid_on = bool(msg.data)
        self.get_logger().info(f'Floor grid {"on" if self.grid_on else "off"}')

    def _draw_grid(self, img, header, fx, fy, cx, cy):
        """Floor grid of the first available ground frame, at the image stamp."""
        for frame in self.grid_frames:
            for stamp, wait in ((Time.from_msg(header.stamp), 0.02), (Time(), 0.0)):
                try:
                    tf = self.tf_buffer.lookup_transform(header.frame_id, frame, stamp,
                                                         timeout=Duration(seconds=wait))
                except TransformException:
                    continue
                r, t = tf.transform.rotation, tf.transform.translation
                draw_floor_grid(img, quat_matrix(r.x, r.y, r.z, r.w), np.array([t.x, t.y, t.z]),
                                fx, fy, cx, cy)
                return

    def _on_zones(self, msg):
        try:
            self.zones.load_json(msg.data)
            self.get_logger().info(f'{len(self.zones.zones)} ignore zone(s) active')
        except ValueError as e:
            self.get_logger().error(f'Bad ignore zone message: {e}')

    def _on_localized(self, _msg):
        if not self.localized:
            self.localized = True
            self.get_logger().info('Localized in saved map: exclusion zones and line of sight active')

    def _map_transform(self, header):
        """Transform map <- camera optical frame at the image stamp (latest as fallback)."""
        if not (self.use_zones or self.use_los) or not self.localized:
            return None
        if time.monotonic() < self._tf_retry_after:
            return None
        err = None
        for stamp, wait in ((Time.from_msg(header.stamp), 0.03), (Time(), 0.0)):
            try:
                return self.tf_buffer.lookup_transform(
                    self.map_frame, header.frame_id, stamp, timeout=Duration(seconds=wait))
            except TransformException as e:
                err = e
        # Back off so a missing map doesn't add lookup latency to every frame.
        self._tf_retry_after = time.monotonic() + 1.0
        if not self._tf_warned:
            self.get_logger().warn(f'No {self.map_frame} transform yet, not filtering: {err}')
            self._tf_warned = True
        return None

    def _ignore_reason(self, tf_map, point_optical):
        """Returns why a detection at point_optical (camera optical frame) should be ignored."""
        if tf_map is None:
            return None
        px, py, _ = apply_transform(tf_map, point_optical)
        if self.use_zones:
            zone = self.zones.containing(px, py)
            if zone:
                return f'in {zone}'
        if self.use_los and self.los.ready and self.los.frame == self.map_frame:
            cx, cy, _ = apply_transform(tf_map, (0.0, 0.0, 0.0))
            if self.los.blocked((cx, cy), (px, py), skip_end_m=self.los_skip_end) >= self.los.min_blocking_cells:
                return 'behind obstacle'
        return None

    def _on_right_info(self, msg):
        fx = msg.p[0]
        if fx > 0 and msg.p[3] != 0.0:
            self.baseline = -msg.p[3] / fx

    def _log_stats(self):
        if self.n_frames:
            self.get_logger().info(
                f'{self.n_frames / 30.0:.1f} Hz, inference {1000 * self.t_infer / self.n_frames:.1f} ms, '
                f'total {1000 * self.t_total / self.n_frames:.1f} ms/frame')
        self.n_frames, self.t_infer, self.t_total = 0, 0.0, 0.0

    @staticmethod
    def _to_numpy(msg, dtype, channels):
        arr = np.frombuffer(msg.data, dtype=dtype)
        row = msg.step // np.dtype(dtype).itemsize
        arr = arr.reshape(msg.height, row)[:, :msg.width * channels]
        return arr.reshape(msg.height, msg.width, channels) if channels > 1 else arr

    def _detect(self, bgr):
        r, px, py = self.yolo.preprocess(bgr)
        t0 = time.perf_counter()
        out = self.yolo.infer()  # (4 + num_classes, N)
        self.t_infer += time.perf_counter() - t0
        scores = out[4:][self.class_ids]
        best = scores.max(axis=0)
        keep = best > self.conf
        if not np.any(keep):
            return []
        cls = np.array(self.class_ids)[scores[:, keep].argmax(axis=0)]
        cx, cy, w, h = out[0, keep], out[1, keep], out[2, keep], out[3, keep]
        x1 = (cx - w / 2 - px) / r
        y1 = (cy - h / 2 - py) / r
        bw, bh = w / r, h / r
        conf = best[keep]
        boxes = np.stack([x1, y1, bw, bh], axis=1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), self.conf, self.nms)
        H, W = bgr.shape[:2]
        dets = []
        for i in np.array(idx).flatten():
            bx1 = int(np.clip(boxes[i, 0], 0, W - 1))
            by1 = int(np.clip(boxes[i, 1], 0, H - 1))
            bx2 = int(np.clip(boxes[i, 0] + boxes[i, 2], 0, W - 1))
            by2 = int(np.clip(boxes[i, 1] + boxes[i, 3], 0, H - 1))
            if bx2 - bx1 >= 4 and by2 - by1 >= 4:
                dets.append((bx1, by1, bx2, by2, float(conf[i]), int(cls[i])))
        return dets

    def _distance(self, depth, left_gray, right_gray, box, fx):
        """Returns (z_m, source, (u, v)) using the torso region of the box."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rx1, rx2 = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
        ry1, ry2 = int(y1 + 0.2 * h), int(y2 - 0.35 * h)
        ry2, rx2 = max(ry2, ry1 + 2), max(rx2, rx1 + 2)
        uv = ((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0)

        roi = depth[ry1:ry2, rx1:rx2]
        valid = roi[roi > 0]
        if valid.size >= self.min_depth_px:
            # Low percentile = the person's front surface; background showing through is farther.
            return float(np.percentile(valid, 30)) / 1000.0, 'depth', uv

        if self.baseline and fx > 0:
            # Stereo fallback: block-match the box's row band (StereoBM needs full-width rows).
            b1, b2 = max(0, y1), min(left_gray.shape[0], y2)
            if b2 - b1 >= 16:
                disp = self.stereo.compute(left_gray[b1:b2], right_gray[b1:b2]).astype(np.float32) / 16.0
                d = disp[ry1 - b1:ry2 - b1, rx1:rx2]
                d = d[d > 1.0]
                if d.size >= self.min_depth_px:
                    return float(fx * self.baseline / np.percentile(d, 70)), 'stereo', uv
        return None, 'none', uv

    def _on_frames(self, color_msg, depth_msg, right_msg):
        if self.left_info is None or not rclpy.ok():
            return
        t_start = time.perf_counter()
        bgr = self._to_numpy(color_msg, np.uint8, 3)
        depth = self._to_numpy(depth_msg, np.uint16, 1)
        right = self._to_numpy(right_msg, np.uint8, 1)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        fx, fy = self.left_info.p[0], self.left_info.p[5]
        cx, cy = self.left_info.p[2], self.left_info.p[6]

        dets = self._detect(bgr)
        header = color_msg.header
        arr2d, arr3d = Detection2DArray(header=header), Detection3DArray(header=header)
        ignored3d, markers = Detection3DArray(header=header), MarkerArray()
        annotated = bgr.copy() if self.pub_img.get_subscription_count() > 0 else None
        if annotated is not None and self.grid_on:
            self._draw_grid(annotated, color_msg.header, fx, fy, cx, cy)
        tf_map = self._map_transform(header) if dets else None

        for k, (x1, y1, x2, y2, score, cls) in enumerate(dets):
            name = COCO_NAMES[cls] if cls < len(COCO_NAMES) else str(cls)
            z, source, (u, v) = self._distance(depth, gray, right, (x1, y1, x2, y2), fx)
            reason = None

            d2 = Detection2D(header=header)
            d2.bbox.center.position.x = (x1 + x2) / 2.0
            d2.bbox.center.position.y = (y1 + y2) / 2.0
            d2.bbox.size_x, d2.bbox.size_y = float(x2 - x1), float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = name, score
            d2.results.append(hyp)
            arr2d.detections.append(d2)

            if z is not None:
                X, Y = (u - cx) * z / fx, (v - cy) * z / fy
                reason = self._ignore_reason(tf_map, (X, Y, z))
                d3 = Detection3D(header=header)
                d3.bbox.center.position.x, d3.bbox.center.position.y, d3.bbox.center.position.z = X, Y, z
                d3.bbox.center.orientation.w = 1.0
                d3.bbox.size.x = (x2 - x1) * z / fx
                d3.bbox.size.y = (y2 - y1) * z / fy
                d3.bbox.size.z = 0.3
                hyp3 = ObjectHypothesisWithPose()
                hyp3.hypothesis.class_id, hyp3.hypothesis.score = name, score
                hyp3.pose.pose.position = d3.bbox.center.position
                d3.results.append(hyp3)
                d3.id = reason or ''  # why it was ignored (empty for active detections)
                (ignored3d if reason else arr3d).detections.append(d3)
                text = f'{name} {z:.2f} m' + (f' ({reason})' if reason else '')
                markers.markers.extend(self._markers(header, k, X, Y, z, text, ignored=bool(reason)))

            if annotated is not None:
                if reason:
                    color = (110, 110, 110)
                else:
                    color = {'depth': (0, 220, 0), 'stereo': (0, 200, 255), 'none': (200, 200, 200)}[source]
                label = f'{name} {z:.2f} m' if z is not None else f'{name} ? m'
                if reason:
                    label += f' - ignored: {reason}'
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = max(y1, th + 8)
                cv2.rectangle(annotated, (x1, ty - th - 8), (x1 + tw + 8, ty), color, -1)
                cv2.putText(annotated, label, (x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255) if reason else (0, 0, 0), 2, cv2.LINE_AA)

        self.pub_2d.publish(arr2d)
        self.pub_3d.publish(arr3d)
        self.pub_ignored.publish(ignored3d)
        if markers.markers:
            self.pub_markers.publish(markers)
        if annotated is not None:
            ok, jpg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                msg = CompressedImage(header=header, format='jpeg')
                msg.data = jpg.tobytes()
                self.pub_img.publish(msg)

        self.n_frames += 1
        self.t_total += time.perf_counter() - t_start

    @staticmethod
    def _markers(header, k, x, y, z, text, ignored=False):
        lifetime = Duration(seconds=0.4).to_msg()
        sphere = Marker(header=header, ns='people', id=2 * k, type=Marker.SPHERE, action=Marker.ADD)
        sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = x, y, z
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
        if ignored:
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 0.5, 0.5, 0.5, 0.6
        else:
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 0.1, 0.9, 0.2, 0.8
        sphere.lifetime = lifetime
        label = Marker(header=header, ns='people_labels', id=2 * k + 1, type=Marker.TEXT_VIEW_FACING,
                       action=Marker.ADD, text=text)
        label.pose.position.x, label.pose.position.y, label.pose.position.z = x, y - 0.35, z
        label.pose.orientation.w = 1.0
        label.scale.z = 0.18
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.lifetime = lifetime
        return [sphere, label]


def main():
    rclpy.init()
    node = PersonDetector()
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
