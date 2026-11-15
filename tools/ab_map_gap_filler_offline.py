#!/usr/bin/env python3
"""Exact A/B of map_gap_filler: feeds the same sequence of maps and robot paths to the Python node's
callbacks (roomba_perception/map_gap_filler.py) and to the Rust core (examples/gap_filler_replay.rs)
and compares every published grid cell for cell, plus the log lines. Nothing goes through ROS.

Inputs: /rtabmap/map and /rtabmap/mapPath from a recording (tools/record_map_replay.sh), then
synthetic cases: rows of posts (single and 2-cell, jittered, any angle) with and without the robot's
path crossing them, dense dot lattices (many ties in numpy's argsort), the recorded maps cropped,
padded and shifted (map resize / origin shift), an empty map, 1-row/1-column/all-occupied maps, no
path / 1-pose path / path off the map, and other parameters (no fence joining, other thresholds,
spacing, post counts, dot sizes, gap widths, resolutions).
  tools/ab_map_gap_filler_offline.py BAG [--synthetic N] [--seed S] [--rust BIN] [--work DIR]
"""
import argparse
import math
import os
import struct
import subprocess
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions

import roomba_perception.map_gap_filler as G

ap = argparse.ArgumentParser()
ap.add_argument('bag')
ap.add_argument('--synthetic', type=int, default=300)
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--rust', default=os.path.expanduser('~/rs_lanes/tfmap_target/release/examples/gap_filler_replay'))
ap.add_argument('--work', default='/data/rs_lanes/tfmap')
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

DEFAULT = dict(gap_m=0.10, occupied=65, join_fences=True, spacing_m=0.45, tolerance_m=0.06, min_posts=3,
               max_dot_cells=6, clear=0.17)
rclpy.init()
node = None
logs = []


class Log:
    def info(self, m, *a, **k):
        logs.append((1 if 'throttle_duration_sec' in k else 0, m))


inp = bytearray()
expected = []   # (kind, data, logs)
py_time = [0.0]


def reset(**kw):
    global node
    p = dict(DEFAULT, **kw)
    if node is not None:
        node.destroy_node()
    node = G.MapGapFiller()
    node.get_logger = lambda: Log()
    node.gap, node.occupied, node.fence, node.clear_radius = p['gap_m'], p['occupied'], p['join_fences'], p['clear']
    node.fence_kw = dict(spacing_m=p['spacing_m'], tolerance_m=p['tolerance_m'], min_posts=p['min_posts'],
                         max_dot_cells=p['max_dot_cells'], occupied=p['occupied'])
    node.pub.publish = lambda m: published.append(m)
    inp.extend(b'R' + struct.pack('<dqBddqqd', p['gap_m'], p['occupied'], p['join_fences'], p['spacing_m'],
                                  p['tolerance_m'], p['min_posts'], p['max_dot_cells'], p['clear']))


published = []


def path(points, label='path'):
    msg = Path()
    msg.header.frame_id = 'map'
    for x, y in points:
        ps = PoseStamped()
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        msg.poses.append(ps)
    feed_path(msg)


def feed_path(msg):
    node._on_path(msg)
    inp.extend(b'P' + struct.pack('<I', len(msg.poses)))
    for ps in msg.poses:
        inp.extend(struct.pack('<dd', ps.pose.position.x, ps.pose.position.y))


def grid_msg(grid, res=0.05, ox=0.0, oy=0.0):
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = float(np.float32(res))   # as received: a float32 field
    msg.info.height, msg.info.width = grid.shape
    msg.info.origin.position.x, msg.info.origin.position.y = float(ox), float(oy)
    msg.info.origin.orientation.w = 1.0
    msg.data = grid.astype(np.int8).ravel().tolist()
    return msg


def feed_map(msg, label):
    del published[:], logs[:]
    t0 = time.perf_counter()
    node._on_map(msg)
    py_time[0] += time.perf_counter() - t0
    assert len(published) == 1
    data = np.asarray(published[0].data, dtype=np.int8)
    expected.append((label, data, list(logs)))
    i = msg.info
    inp.extend(b'M' + struct.pack('<IIfdd', i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y))
    inp.extend(np.asarray(msg.data, dtype=np.int8).tobytes())


# ---- recorded maps and paths ----
reset()
r = SequentialReader()
r.open(StorageOptions(uri=args.bag, storage_id='mcap'), ConverterOptions('', ''))
r.set_filter(StorageFilter(topics=['/rtabmap/map', '/rtabmap/mapPath']))
real_maps, real_paths = [], []
while r.has_next():
    topic, data, _ = r.read_next()
    if topic == '/rtabmap/map':
        m = deserialize_message(data, OccupancyGrid)
        real_maps.append(m)
        feed_map(m, 'recorded')
    else:
        p = deserialize_message(data, Path)
        real_paths.append([(ps.pose.position.x, ps.pose.position.y) for ps in p.poses])
        feed_path(p)
n_recorded = len(expected)


# ---- synthetic ----
def post_rows(grid, rows, res):
    """Draws rows of posts; returns their end points in cells."""
    h, w = grid.shape
    ends = []
    for _ in range(rows):
        x0, y0 = rng.uniform(5, w - 5), rng.uniform(5, h - 5)
        ang = rng.uniform(0, math.pi) if rng.random() < 0.6 else rng.choice([0, math.pi / 2, math.pi / 4])
        spacing = rng.uniform(3, 11) * 0.05 / res
        n = int(rng.integers(2, 9))
        jitter = rng.choice([0.0, 0.4, 1.0])
        for k in range(n):
            x = x0 + math.cos(ang) * spacing * k + rng.uniform(-jitter, jitter)
            y = y0 + math.sin(ang) * spacing * k + rng.uniform(-jitter, jitter)
            cx, cy = int(round(x)), int(round(y))
            if 0 <= cx < w and 0 <= cy < h:
                grid[cy, cx] = rng.choice([100, 100, 70, 65])
                if rng.random() < 0.25 and cx + 1 < w:
                    grid[cy, cx + 1] = 100
        ends.append(((x0, y0), (x0 + math.cos(ang) * spacing * (n - 1), y0 + math.sin(ang) * spacing * (n - 1))))
    return ends


def scene(h, w, res):
    g = np.full((h, w), -1, np.int16)
    g[5:h - 5, 5:w - 5] = rng.choice([0, 0, 0, 10, 50], size=(h - 10, w - 10))
    for _ in range(int(rng.integers(0, 6))):   # walls
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        if rng.random() < 0.5:
            g[y:y + int(rng.integers(1, 3)), x:x + int(rng.integers(5, 40))] = 100
        else:
            g[y:y + int(rng.integers(5, 40)), x:x + int(rng.integers(1, 3))] = 100
    noise = rng.random((h, w)) < rng.choice([0.0, 0.002, 0.01])
    g[noise] = 100
    return g


variants = [dict(), dict(join_fences=False), dict(occupied=50), dict(occupied=80), dict(min_posts=2),
            dict(min_posts=4), dict(max_dot_cells=1), dict(max_dot_cells=12), dict(gap_m=0.2), dict(gap_m=0.04),
            dict(spacing_m=0.6), dict(tolerance_m=0.12), dict(clear=0.05), dict(clear=0.3)]
for s in range(args.synthetic):
    params = variants[s % len(variants)] if s % 3 == 0 else {}
    reset(**params)
    res = float(rng.choice([0.05, 0.05, 0.05, 0.03, 0.1]))
    h, w = int(rng.integers(30, 220)), int(rng.integers(30, 220))
    g = scene(h, w, res)
    ends = post_rows(g, int(rng.integers(1, 7)), res)
    ox, oy = rng.uniform(-10, 10), rng.uniform(-10, 10)
    kind = s % 5
    if kind == 1:      # the robot drove through some rows
        pts = []
        for (a, b) in ends[:2]:
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            pts += [(ox + (mx - 20) * res, oy + (my - 20) * res), (ox + (mx + 20) * res, oy + (my + 20) * res)]
        path(pts)
    elif kind == 2:    # a random walk
        p = np.cumsum(rng.normal(0, 0.15, size=(int(rng.integers(2, 200)), 2)), axis=0)
        path(p + [ox + w * res / 2, oy + h * res / 2])
    elif kind == 3:    # single pose, or off the map
        path([(ox + rng.uniform(-50, 50), oy + rng.uniform(-50, 50))] * int(rng.integers(1, 3)))
    feed_map(grid_msg(g, res, ox, oy), f'synthetic{params}')
    if s % 10 == 0:    # dense lattice of isolated dots: long runs and many equal positions
        g = np.zeros((60, 60), np.int16)
        step = int(rng.choice([2, 3, 4]))
        g[::step, ::step] = 100
        if rng.random() < 0.5:
            g[1::step * 2, ::step * 3] = 100
        path([(0.1, 0.1), (1.0, 2.5)] if rng.random() < 0.5 else [])
        feed_map(grid_msg(g, 0.05), 'lattice')

# Recorded maps cropped / padded / shifted (resize and origin shift) with the recorded path.
reset()
if real_paths:
    feed_path_msg = real_paths[-1]
    path(feed_path_msg)
for k, m in enumerate(real_maps[::max(1, len(real_maps) // 25)]):
    g = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
    res = m.info.resolution
    dx, dy = int(rng.integers(-30, 30)), int(rng.integers(-30, 30))
    g2 = np.full((g.shape[0] + 60, g.shape[1] + 60), -1, np.int16)
    g2[30 + dy:30 + dy + g.shape[0], 30 + dx:30 + dx + g.shape[1]] = g
    g2 = g2[int(rng.integers(0, 40)):, int(rng.integers(0, 40)):]
    post_rows(g2, 3, res)
    feed_map(grid_msg(g2, res, m.info.origin.position.x - (30 + dx) * res, m.info.origin.position.y - (30 + dy) * res),
             'shifted')

# Edge cases.
reset()
empty = OccupancyGrid()
empty.header.frame_id = 'map'
feed_map(empty, 'empty')
for shape in ((1, 50), (50, 1), (1, 1), (2, 2)):
    feed_map(grid_msg(np.where(rng.random(shape) < 0.5, 100, 0), 0.05), f'tiny{shape}')
feed_map(grid_msg(np.full((40, 40), 100), 0.05), 'all occupied')
feed_map(grid_msg(np.full((40, 40), -1), 0.05), 'all unknown')
path([(1e4, -1e4), (-1e4, 1e4)])
feed_map(grid_msg(np.where(rng.random((80, 80)) < 0.02, 100, 0), 0.05), 'path far away')

# ---- run Rust and compare ----
inp_path, out_path = os.path.join(args.work, 'gap_in.bin'), os.path.join(args.work, 'gap_out.bin')
with open(inp_path, 'wb') as f:
    f.write(inp)
t0 = time.perf_counter()
subprocess.run([args.rust, inp_path, out_path], check=True)
rs_time = time.perf_counter() - t0
with open(out_path, 'rb') as f:
    out = f.read()
os.remove(inp_path)
os.remove(out_path)

off = 0
maps = same = cells = diff_cells = log_diff = 0
changed = links_seen = 0
shown = 0
for label, data, exp_logs in expected:
    (n,) = struct.unpack_from('<I', out, off)
    off += 4
    got = np.frombuffer(out, dtype=np.int8, count=n, offset=off)
    off += n
    (nl,) = struct.unpack_from('<I', out, off)
    off += 4
    got_logs = []
    for _ in range(nl):
        thr, ln = struct.unpack_from('<BI', out, off)
        off += 5
        got_logs.append((thr, out[off:off + ln].decode()))
        off += ln
    maps += 1
    cells += len(data)
    d = int(np.count_nonzero(got != data)) if len(got) == len(data) else len(data)
    diff_cells += d
    same += d == 0
    log_diff += got_logs != exp_logs
    for _, text in exp_logs:
        if 'fence link' in text and not text.startswith('0 '):
            links_seen += 1
    if d or got_logs != exp_logs:
        if shown < 8:
            shown += 1
            print(f'DIFF {label}: {d} cells; logs py {exp_logs} rs {got_logs}')
assert off == len(out)
print(f'{n_recorded} recorded maps ({len(real_paths)} paths), {maps - n_recorded} synthetic/edge maps')
print(f'{maps} maps compared: {same} identical, {maps - same} different; {cells} cells, {diff_cells} differ; '
      f'{log_diff} maps with different log lines; {links_seen} fence-count log lines with links > 0')
print(f'processing time: python {py_time[0]:.2f} s, rust {rs_time:.2f} s (incl. file I/O)')
node.destroy_node()
rclpy.shutdown()
