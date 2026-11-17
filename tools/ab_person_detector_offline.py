#!/usr/bin/env python3
"""Exact offline A/B of the Rust person detector (roomba_rs::person through
src/roomba_rs/examples/person_replay.rs) against roomba_perception/person_detector.py.

Both get the identical message sequence, built from a raw recording (left color, registered depth,
right mono, camera_info) plus, optionally, the TF and RTAB-Map grid of a replay of that recording
(tools/record_map_replay.sh), merged by time, plus scripted events (ignore zones placed ahead of
the robot, a bad zone message, floor-grid toggles, the annotated image's subscriber coming and
going, localization, stats; 'people': a zone dropped around the first located person). The Python node's callbacks are called directly (its publishers,
logger, cv2.imencode, time.monotonic and TF waits patched; no executor), the Rust core reads the
same events from a pipe. Per processed frame it compares the YOLO input and output tensors
(CRC32), the published Detection2D/3D arrays, ignored detections and their reasons, markers, the
annotated image before encoding (CRC32) and the JPEG (bytes, else decoded max diff / PSNR), and the
log lines.
  tools/ab_person_detector_offline.py --bag RAW [--tf-bag REPLAY] --start S --duration D
      [--params '{"classes": [0, 15, 16]}'] [--scenario robot|coverage|people] [--rust BIN] [--work DIR]
"""
import argparse
import base64
import json
import math
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from types import SimpleNamespace

import cv2
import numpy as np
import rclpy
import rclpy.node
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage

import roomba_perception.person_detector as pd

ap = argparse.ArgumentParser()
ap.add_argument('--bag', required=True)
ap.add_argument('--tf-bag')
ap.add_argument('--start', type=float, default=0.0, help='s from the start of the recording')
ap.add_argument('--duration', type=float, default=60.0)
ap.add_argument('--params', default='{}', help='node parameters (JSON)')
ap.add_argument('--scenario', default='robot', choices=['robot', 'coverage', 'people', 'plain'])
ap.add_argument('--rust', default=os.path.expanduser('~/rs_lanes/people_target/release/examples/person_replay'))
ap.add_argument('--work', default='/data/rs_lanes/people')
ap.add_argument('--examples', type=int, default=3, help='mismatches to print per kind')
args = ap.parse_args()
params = json.loads(args.params)

IMAGES = {'/camera/left/image_rect_color': 0, '/camera/depth/image_rect_raw': 1, '/camera/right/image_rect_mono': 2}


def ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


# ---- events for the Rust replay (see examples/person_replay.rs) ----
def s_(text):
    b = text.encode()
    return struct.pack('<I', len(b)) + b


def ev_info(kind, msg):
    return struct.pack('<B12d', kind, *msg.p)


def ev_image(i, msg):
    data = bytes(msg.data)
    return (struct.pack('<Bbq', 3, i, ns(msg.header.stamp)) + s_(msg.header.frame_id)
            + struct.pack('<III', msg.width, msg.height, msg.step) + struct.pack('<I', len(data)) + data)


def ev_tf(msg, static):
    out = struct.pack('<BBI', 4, 1 if static else 0, len(msg.transforms))
    for t in msg.transforms:
        tr, q = t.transform.translation, t.transform.rotation
        out += s_(t.header.frame_id) + s_(t.child_frame_id) + struct.pack(
            '<q7d', ns(t.header.stamp), tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w)
    return out


def ev_map(m):
    i, o, q = m.info, m.info.origin.position, m.info.origin.orientation
    data = np.asarray(m.data, dtype=np.int8).tobytes()
    return (struct.pack('<B', 5) + s_(m.header.frame_id) + struct.pack('<IIf2d4d', i.width, i.height, i.resolution, o.x, o.y,
                                                                        q.x, q.y, q.z, q.w)
            + struct.pack('<I', len(data)) + data)


# ---- the Python node, driven directly ----
pfile = tempfile.NamedTemporaryFile('w', suffix='.yaml', dir=args.work, delete=False)
pfile.write('/**:\n  ros__parameters:\n' + ''.join(f'    {k}: {json.dumps(v)}\n' for k, v in params.items()))
pfile.close()
rclpy.init(args=['--ros-args', '--params-file', pfile.name])
os.unlink(pfile.name)
pd.TransformListener = lambda *a, **k: None   # no listener thread: TF is fed below
fake_now = [0.0]
pd.time = SimpleNamespace(monotonic=lambda: fake_now[0], perf_counter=time.perf_counter)
annotated = []


class Cv2Proxy:
    def __getattr__(self, name):
        return getattr(cv2, name)

    @staticmethod
    def imencode(ext, img, params=None):
        annotated.append(zlib.crc32(np.ascontiguousarray(img).tobytes()))
        return cv2.imencode(ext, img, params)


pd.cv2 = Cv2Proxy()
py_logs = []


class LogProxy:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def info(self, m, **k):
        py_logs.append(['info', m])

    def warn(self, m, **k):
        py_logs.append(['warn', m])

    warning = warn

    def error(self, m, **k):
        py_logs.append(['error', m])


real_get_logger = rclpy.node.Node.get_logger
pd.PersonDetector.get_logger = lambda self: LogProxy(real_get_logger(self))
node = pd.PersonDetector()
buf = node.tf_buffer
buf.lookup_transform = lambda target, source, t, timeout=None: buf.lookup_transform_core(target, source, t)
published = []
for name in ('pub_img', 'pub_2d', 'pub_3d', 'pub_ignored', 'pub_markers', 'pub_objects'):
    if getattr(node, name, None) is not None:   # pub_objects: models with classes that are not people
        getattr(node, name).publish = (lambda n: lambda m: published.append((n, m)))(name)
annotate = [True]
node.pub_img.get_subscription_count = lambda: 1 if annotate[0] else 0

sources = []
orig_distance = node._distance


def distance(*a, **k):
    r = orig_distance(*a, **k)
    sources.append(r[1])
    return r


node._distance = distance

# ---- the Rust replay ----
rust_params = os.path.join(args.work, f'ab_params_{os.getpid()}.json')
json.dump(params, open(rust_params, 'w'))
proc = subprocess.Popen([args.rust, rust_params], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1 << 20)
lines = queue.Queue()
threading.Thread(target=lambda: [lines.put(json.loads(l)) for l in proc.stdout], daemon=True).start()
rust_logs = []


def send(b):
    proc.stdin.write(b)


def rust_next(kind, timeout=120):
    """Next Rust line of `kind` ('frame' or 'stats'), collecting log lines."""
    while True:
        line = lines.get(timeout=timeout)
        if 'log' in line:
            rust_logs.append(line['log'])
        elif kind in line or 'end' in line:
            return line


stats = dict(frames=0, py_only=0, rust_only=0, in_equal=0, out_equal=0, dets=0, d2_equal=0, d3_equal=0,
             ig_equal=0, ob_equal=0, objects=0, mk_equal=0, img_frames=0, img_equal=0, jpeg_equal=0,
             src_equal=0, stats_equal=0, stats_events=0, frames_with_dets=0, ignored=0)
src_count = {}
reasons = {}
jpeg_diffs = []
mismatch_examples = {}


def note(kind, detail):
    stats.setdefault('mismatch_' + kind, 0)
    stats['mismatch_' + kind] += 1
    ex = mismatch_examples.setdefault(kind, [])
    if len(ex) < args.examples:
        ex.append(detail)


def py_frame(stamp):
    """The Python node's published messages for one frame, as person_replay prints them."""
    out = {'d2': [], 'd3': [], 'ig': [], 'ob': [], 'mk': [], 'jpeg': None}
    for name, m in published:
        if name == 'pub_2d':
            for d in m.detections:
                r = d.results[0].hypothesis
                out['d2'].append([[d.bbox.center.position.x, d.bbox.center.position.y], [d.bbox.size_x, d.bbox.size_y],
                                  r.class_id, r.score])
                assert ns(d.header.stamp) == stamp and len(d.results) == 1 and d.bbox.center.theta == 0.0
        elif name in ('pub_3d', 'pub_ignored', 'pub_objects'):
            key = {'pub_3d': 'd3', 'pub_ignored': 'ig', 'pub_objects': 'ob'}[name]
            for d in m.detections:
                c, s, r = d.bbox.center, d.bbox.size, d.results[0]
                assert r.pose.pose.position == c.position and c.orientation.w == 1.0 and ns(d.header.stamp) == stamp
                out[key].append([[c.position.x, c.position.y, c.position.z], [s.x, s.y, s.z], r.hypothesis.class_id,
                                 r.hypothesis.score, d.id])
        elif name == 'pub_markers':
            for mk in m.markers:
                p, sc, c = mk.pose.position, mk.scale, mk.color
                assert mk.lifetime.sec == 0 and mk.lifetime.nanosec == 400_000_000 and mk.action == 0
                assert mk.pose.orientation.w == 1.0 and ns(mk.header.stamp) == stamp
                # ColorRGBA is float32 on the wire (rclpy keeps the assigned float until serialization)
                rgba = [float(np.float32(v)) for v in (c.r, c.g, c.b, c.a)]
                out['mk'].append([mk.ns, mk.id, mk.type, [p.x, p.y, p.z], [sc.x, sc.y, sc.z], rgba, mk.text])
        elif name == 'pub_img':
            assert m.format == 'jpeg'
            out['jpeg'] = bytes(m.data)
    return out


def compare(stamp, py, rs):
    stats['frames'] += 1
    stats['in_equal'] += rs['in_crc'] == zlib.crc32(node.yolo.inp.tobytes())
    if rs['in_crc'] != zlib.crc32(node.yolo.inp.tobytes()):
        note('input_tensor', stamp)
    stats['out_equal'] += rs['out_crc'] == zlib.crc32(node.yolo.out.tobytes())
    n = len(py['d2'])
    stats['dets'] += n
    stats['frames_with_dets'] += n > 0
    stats['objects'] += len(py['ob'])
    for key in ('d2', 'd3', 'ig', 'ob', 'mk'):
        if py[key] == rs.get(key, []):
            stats[key + '_equal'] += 1
        else:
            note(key, {'stamp': stamp, 'py': py[key], 'rust': rs.get(key)})
    for d in py['ig']:
        reasons[d[4]] = reasons.get(d[4], 0) + 1
    stats['ignored'] += len(py['ig'])
    if sources == rs['src']:
        stats['src_equal'] += 1
    else:
        note('source', {'stamp': stamp, 'py': list(sources), 'rust': rs['src']})
    for s in sources:
        src_count[s] = src_count.get(s, 0) + 1
    rj = rs['jpeg']
    if (py['jpeg'] is None) != (rj is None):
        note('jpeg_presence', stamp)
    elif rj is not None:
        stats['img_frames'] += 1
        if annotated and annotated[-1] == rs['img_crc']:
            stats['img_equal'] += 1
        else:
            note('annotated', stamp)
        rj = base64.b64decode(rj)
        if rj == py['jpeg']:
            stats['jpeg_equal'] += 1
        else:
            a = cv2.imdecode(np.frombuffer(py['jpeg'], np.uint8), cv2.IMREAD_COLOR).astype(np.int16)
            b = cv2.imdecode(np.frombuffer(rj, np.uint8), cv2.IMREAD_COLOR).astype(np.int16)
            mse = float(np.mean((a - b) ** 2))
            jpeg_diffs.append((int(np.abs(a - b).max()), int((a != b).any(axis=2).sum()),
                               99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)))


orig_on_frames = node._on_frames


def on_frames(c, d, r):
    published.clear()
    sources.clear()
    n_img = len(annotated)
    orig_on_frames(c, d, r)
    stamp = ns(c.header.stamp)
    if published:
        rs = rust_next('frame')
        if rs.get('frame') != stamp:
            stats['py_only'] += 1
            note('frame_alignment', {'py': stamp, 'rust': rs.get('frame')})
            return
        py = py_frame(stamp)
        compare(stamp, py, rs)
        if args.scenario == 'people' and not people_zone and py['d3']:
            place_people_zone(c.header, py['d3'][0][0])
    assert len(annotated) - n_img <= 1


people_zone = []


def place_people_zone(header, p):
    """'people' scenario: an exclusion zone around the first located detection (both nodes get it)."""
    try:
        tf = buf.lookup_transform_core('map', header.frame_id, Time.from_msg(header.stamp))
    except Exception:
        return
    x, y, _ = pd.apply_transform(tf, tuple(p))
    people_zone.append((x, y))
    feed_zones(json.dumps({'frame': 'map', 'zones': [
        {'name': 'sofa corner', 'points': [[x - 0.5, y - 0.5], [x + 0.5, y - 0.5], [x + 0.5, y + 0.5], [x - 0.5, y + 0.5]]}]}))


node.sync.callbacks = {}
node.sync.registerCallback(on_frames)


# ---- the message sequence ----
def bag_reader(path):
    r = SequentialReader()
    r.open(StorageOptions(uri=path, storage_id='mcap'), ConverterOptions('', ''))
    return r


replay = []   # (time ns, kind, message) from the TF / map replay
static_msgs = []
if args.tf_bag:
    r = bag_reader(args.tf_bag)
    while r.has_next():
        topic, data, t = r.read_next()
        if topic == '/tf_static':
            static_msgs.append(deserialize_message(data, TFMessage))
        elif topic == '/tf':
            m = deserialize_message(data, TFMessage)
            replay.append((max(ns(x.header.stamp) for x in m.transforms), 'tf', m))
        elif topic == '/rtabmap/map':
            m = deserialize_message(data, OccupancyGrid)
            replay.append((ns(m.header.stamp), 'map', m))
    replay.sort(key=lambda e: e[0])
print(f'{len(replay)} TF/map messages, {len(static_msgs)} static TF messages from the replay', flush=True)

raw = bag_reader(args.bag)
t0 = raw.get_metadata().starting_time.nanoseconds
seg0, seg1 = t0 + int(args.start * 1e9), t0 + int((args.start + args.duration) * 1e9)


def feed_static(m):
    for t in m.transforms:
        buf.set_transform_static(t, 'default_authority')
    send(ev_tf(m, True))


def feed_tf(m):
    for t in m.transforms:
        buf.set_transform(t, 'default_authority')
    send(ev_tf(m, False))


def feed_map(m):
    node.los.update(m)
    send(ev_map(m))


def feed_zones(text):
    node._on_zones(String(data=text))
    send(struct.pack('<B', 6) + s_(text))


def feed_grid(on):
    node._on_grid_toggle(Bool(data=on))
    send(struct.pack('<BB', 8, int(on)))


def feed_localized():
    node._on_localized(PoseWithCovarianceStamped())
    send(struct.pack('<B', 7))


def feed_stats():
    had = node.n_frames > 0
    node._log_stats()
    send(struct.pack('<B', 10))
    proc.stdin.flush()
    stats['stats_events'] += 1
    stats['stats_equal'] += rust_next('stats').get('stats') == had


def zones_ahead():
    """Zones around the robot's current view: an exclusion zone ahead, an unnamed one to the left,
    and a navigation keep-out zone (never an ignore zone) over everything."""
    try:
        tf = buf.lookup_transform_core('map', 'base_link', Time())
    except Exception as e:  # no map yet: zones in an unknown place
        print(f'zones: no map <- base_link ({e})', flush=True)
        return json.dumps({'frame': 'map', 'zones': [{'name': 'nowhere', 'points': [[50, 50], [51, 50], [51, 51]]}]})
    p, q = tf.transform.translation, tf.transform.rotation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    c, s = math.cos(yaw), math.sin(yaw)

    def poly(pts):
        return [[round(p.x + c * x - s * y, 3), round(p.y + s * x + c * y, 3)] for x, y in pts]
    return json.dumps({'frame': 'map', 'zones': [
        {'name': 'keep out', 'kind': 'keepout', 'points': poly([(-20, -20), (20, -20), (20, 20), (-20, 20)])},
        {'name': 'ahead', 'kind': 'exclusion', 'points': poly([(0.6, -0.7), (6.0, -0.7), (6.0, 0.4), (0.6, 0.4)])},
        {'name': '', 'points': poly([(0.4, 0.4), (6.0, 0.4), (6.0, 4.0), (0.4, 4.0)])},
        {'name': 'tiny', 'points': poly([(1, 1), (2, 2)])},
    ]})


D = args.duration
script = []   # (s into the segment, action)
if args.scenario in ('robot', 'coverage'):
    script += [(0.15 * D, lambda: feed_zones(zones_ahead())), (0.55 * D, lambda: feed_zones('not json')),
               (0.6 * D, lambda: feed_zones(zones_ahead())), (0.9 * D, lambda: feed_zones('{"zones": []}'))]
if args.scenario == 'robot':
    script += [(0.3 * D, lambda: feed_grid(False)), (0.45 * D, lambda: feed_grid(True)),
               (0.5 * D, lambda: annotate.__setitem__(0, False)), (0.6 * D, lambda: annotate.__setitem__(0, True))]
if args.scenario == 'coverage' or params.get('require_localization'):
    script += [(0.35 * D, feed_localized)]
script += [(k * 30.0, feed_stats) for k in range(1, int(D // 30) + 1)]
script.sort(key=lambda e: e[0])

for m in static_msgs:
    feed_static(m)
ri = 0
t_wall = time.time()
static_seen = bool(static_msgs)
while raw.has_next():
    topic, data, t = raw.read_next()
    if t < seg0:
        if topic == '/tf_static' and not static_seen:
            feed_static(deserialize_message(data, TFMessage))
        continue
    if t > seg1:
        break
    while ri < len(replay) and replay[ri][0] <= t:
        _, kind, m = replay[ri]
        feed_tf(m) if kind == 'tf' else feed_map(m)
        ri += 1
    while script and seg0 + script[0][0] * 1e9 <= t:
        script.pop(0)[1]()
    fake_now[0] = (t - seg0) / 1e9
    if topic == '/camera/left/camera_info':
        node._on_left_info(deserialize_message(data, CameraInfo))
        send(ev_info(1, node.left_info))
    elif topic == '/camera/right/camera_info':
        m = deserialize_message(data, CameraInfo)
        node._on_right_info(m)
        send(ev_info(2, m))
    elif topic in IMAGES:
        m = deserialize_message(data, Image)
        i = IMAGES[topic]
        send(struct.pack('<Bdb', 9, fake_now[0], int(annotate[0])))
        send(ev_image(i, m))
        proc.stdin.flush()
        node.sync.add(m, node.sync.queues[i], i)

send(struct.pack('<B', 0))
proc.stdin.close()
while True:
    line = lines.get(timeout=120)
    if 'end' in line:
        rust_frames = line['end']
        break
    if 'log' in line:
        rust_logs.append(line['log'])
    elif 'frame' in line:
        stats['rust_only'] += 1
proc.wait()
os.unlink(rust_params)

py_logs_cmp = [l for l in py_logs if ' Hz, inference ' not in l[1]]
log_equal = py_logs_cmp == rust_logs
stats.update(rust_frames=rust_frames, seconds=round(time.time() - t_wall))
print(json.dumps({'scenario': args.scenario, 'bag': os.path.basename(args.bag.rstrip('/')), 'start': args.start,
                  'duration': D, 'params': params, **stats, 'sources': src_count, 'ignore_reasons': reasons,
                  'jpeg_differing_frames': len(jpeg_diffs),
                  'jpeg_max_abs_diff': max((d[0] for d in jpeg_diffs), default=0),
                  'jpeg_max_pixels_differing': max((d[1] for d in jpeg_diffs), default=0),
                  'jpeg_min_psnr_db': min((d[2] for d in jpeg_diffs), default=None),
                  'logs_equal': log_equal, 'log_lines': len(py_logs_cmp),
                  'distinct_logs': sorted({f'{l[0]}: {l[1][:90]}' for l in py_logs_cmp})}, indent=1))
if not log_equal:
    print('logs differ:')
    for a, b in zip(py_logs_cmp + [None] * 5, rust_logs + [None] * 5):
        if a != b:
            print('  py  ', a)
            print('  rust', b)
for kind, ex in mismatch_examples.items():
    print(f'mismatch {kind}:')
    for e in ex:
        print('  ', json.dumps(e)[:2000])
node.destroy_node()
rclpy.shutdown()
