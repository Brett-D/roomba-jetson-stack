#!/usr/bin/env python3
"""Pull camera calibration, static TF, real message rates and sample frames out of a robot bag.

Runs on the Jetson (ROS 2 Jazzy). The recordings have no MCAP message index, so the whole file is
scanned once; only the messages that are kept get deserialized.

  python3 extract_bag.py /data/bags/robot_20261114_145937_fresh_map8 /data/twin/isaac/bag_extract

Writes:
  calib.json            first CameraInfo per camera, every /tf_static transform, per-topic rates
  left/*.png            every --left-every'th /camera/left/image_rect_color frame (bgr8)
  depth/*.png           every --aux-every'th /camera/depth/image_rect_raw frame (16-bit mm)
  right/*.png           every --aux-every'th /camera/right/image_rect_mono frame
  left_mono/*.png       every --aux-every'th /camera/left/image_rect_mono frame
  wheel_odom.csv        /wheel/odom at <= 10 Hz (t, x, y, yaw, v, w) in the wheel_odom frame
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

INFO_TOPICS = ['/camera/left/camera_info', '/camera/right/camera_info', '/camera/depth/camera_info']
LEFT = '/camera/left/image_rect_color'
DEPTH = '/camera/depth/image_rect_raw'
RIGHT = '/camera/right/image_rect_mono'
LEFT_MONO = '/camera/left/image_rect_mono'
IMU = '/camera/imu/data_raw'
ODOM = '/wheel/odom'
TF_STATIC = '/tf_static'
SUBDIR = {LEFT: 'left', DEPTH: 'depth', RIGHT: 'right', LEFT_MONO: 'left_mono'}


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def image_to_array(msg):
    if msg.encoding in ('bgr8', 'rgb8'):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        arr = arr.reshape(msg.height, msg.width, 3)
        return arr if msg.encoding == 'bgr8' else arr[:, :, ::-1]
    if msg.encoding in ('mono8', '8UC1'):
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
    if msg.encoding in ('16UC1', 'mono16'):
        dt = np.dtype('>u2') if msg.is_bigendian else np.dtype('<u2')
        return np.frombuffer(msg.data, dt).reshape(msg.height, msg.step // 2)[:, :msg.width].astype(np.uint16)
    raise ValueError(f'unsupported encoding {msg.encoding}')


def info_dict(msg):
    return {
        'frame_id': msg.header.frame_id, 'width': msg.width, 'height': msg.height,
        'distortion_model': msg.distortion_model, 'd': list(msg.d), 'k': list(msg.k),
        'r': list(msg.r), 'p': list(msg.p), 'binning_x': msg.binning_x, 'binning_y': msg.binning_y,
    }


def rate_stats(stamps, subsample=1):
    if len(stamps) < 3:
        return {'count': len(stamps)}
    stamps = sorted(stamps)
    dts = [(b - a) / 1e9 / subsample for a, b in zip(stamps, stamps[1:]) if b > a]
    span = (stamps[-1] - stamps[0]) / 1e9
    med = statistics.median(dts) if dts else 0.0
    return {
        'count': len(stamps) * subsample, 'span_s': round(span, 3),
        'mean_hz': round((len(stamps) - 1) * subsample / span, 3) if span > 0 else None,
        'median_dt_s': round(med, 5), 'median_hz': round(1.0 / med, 3) if med > 0 else None,
        'p95_dt_s': round(sorted(dts)[int(0.95 * (len(dts) - 1))], 5) if dts else None,
        'gaps_over_0p5s': sum(1 for d in dts if d * subsample > 0.5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('out')
    ap.add_argument('--left-every', type=int, default=30, help='keep every Nth left color frame')
    ap.add_argument('--aux-every', type=int, default=150, help='keep every Nth depth/right/mono frame')
    args = ap.parse_args()

    for sub in SUBDIR.values():
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = set(INFO_TOPICS) | set(SUBDIR) | {IMU, ODOM, TF_STATIC}
    msg_types = {name: get_message(t) for name, t in types.items() if name in wanted}

    recv = {name: [] for name in types}      # bag receive times (ns)
    hdr = {name: [] for name in [LEFT, DEPTH, RIGHT, LEFT_MONO, IMU] + INFO_TOPICS}
    hdr_sub = {LEFT: 10, DEPTH: 10, RIGHT: 10, LEFT_MONO: 10, IMU: 20}
    infos, tf_static, encodings = {}, [], {}
    counters = {LEFT: 0, DEPTH: 0, RIGHT: 0, LEFT_MONO: 0, IMU: 0}
    imu_frame = None
    odom_rows, last_odom = [], -1e18
    t0 = time.time()
    n = 0
    while reader.has_next():
        topic, data, t = reader.read_next()
        n += 1
        recv[topic].append(t)
        if n % 50000 == 0:
            print(f'{n} msgs, {time.time() - t0:.0f} s', flush=True)
        if topic in INFO_TOPICS:
            msg = deserialize_message(data, msg_types[topic])
            hdr[topic].append(stamp_ns(msg.header))
            d = info_dict(msg)
            if topic not in infos:
                infos[topic] = d
                infos[topic]['changes'] = 0
            elif d['k'] != infos[topic]['k'] or d['p'] != infos[topic]['p']:
                infos[topic]['changes'] += 1
        elif topic == TF_STATIC:
            msg = deserialize_message(data, msg_types[topic])
            for tr in msg.transforms:
                q, p = tr.transform.rotation, tr.transform.translation
                tf_static.append({'parent': tr.header.frame_id, 'child': tr.child_frame_id,
                                  'xyz': [p.x, p.y, p.z], 'xyzw': [q.x, q.y, q.z, q.w]})
        elif topic in SUBDIR:
            i = counters[topic]
            counters[topic] += 1
            every = args.left_every if topic == LEFT else args.aux_every
            if i % every == 0 or i % hdr_sub[topic] == 0:
                msg = deserialize_message(data, msg_types[topic])
                if i % hdr_sub[topic] == 0:
                    hdr[topic].append(stamp_ns(msg.header))
                encodings.setdefault(topic, {'encoding': msg.encoding, 'width': msg.width,
                                             'height': msg.height, 'step': msg.step,
                                             'frame_id': msg.header.frame_id})
                if i % every == 0:
                    cv2.imwrite(os.path.join(args.out, SUBDIR[topic], f'{stamp_ns(msg.header)}.png'),
                                image_to_array(msg))
        elif topic == IMU:
            i = counters[IMU]
            counters[IMU] += 1
            if i % hdr_sub[IMU] == 0:
                msg = deserialize_message(data, msg_types[topic])
                hdr[IMU].append(stamp_ns(msg.header))
                imu_frame = msg.header.frame_id
        elif topic == ODOM:
            if t - last_odom >= 1e8:
                msg = deserialize_message(data, msg_types[topic])
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                odom_rows.append((stamp_ns(msg.header) / 1e9, p.x, p.y, yaw,
                                  msg.twist.twist.linear.x, msg.twist.twist.angular.z))
                last_odom = t

    with open(os.path.join(args.out, 'wheel_odom.csv'), 'w') as f:
        f.write('t,x,y,yaw,v,w\n')
        for r in odom_rows:
            f.write(','.join(f'{v:.6f}' for v in r) + '\n')

    out = {
        'bag': args.bag,
        'camera_info': infos,
        'tf_static': tf_static,
        'image_encodings': encodings,
        'imu_frame_id': imu_frame,
        'rates_receive_time': {k: rate_stats(v) for k, v in recv.items()},
        'rates_header': {k: rate_stats(v, hdr_sub.get(k, 1)) for k, v in hdr.items()},
        'messages_scanned': n,
        'scan_seconds': round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, 'calib.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('done', n, 'messages in', round(time.time() - t0), 's', file=sys.stderr)


if __name__ == '__main__':
    main()
