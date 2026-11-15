#!/usr/bin/env python3
"""Exact A/B of roomba_rs::tf_buffer (examples/tf_replay.rs) against tf2 (tf2_ros.Buffer, i.e. the C++
BufferCore): feeds the /tf and /tf_static messages of a recording in order to both, plus synthetic edge
cases (parent switches, a loop, static <-> dynamic changes, '/'-prefixed ids, NaN/self transforms, a
stale timestamp), and compares many lookups along the way: random frame pairs (also unknown, empty and
'/'-prefixed names) at random times (latest, interpolated, exact stamps, +-1 ns, beyond the 10 s
cache, in the future). Compares stamps, translations, rotations, and error classes and messages.
  tools/ab_tf_buffer.py BAG [--every N] [--per K] [--seed S] [--limit MSGS] [--rust BIN] [--work DIR]
"""
import argparse
import collections
import math
import os
import random
import subprocess

import tf2_ros
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from tf2_msgs.msg import TFMessage

ap = argparse.ArgumentParser()
ap.add_argument('bag')
ap.add_argument('--every', type=int, default=5, help='lookups after every N-th message')
ap.add_argument('--per', type=int, default=20, help='lookups each time')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--limit', type=int, default=0, help='stop after this many TF messages (0 = all)')
ap.add_argument('--rust', default=os.path.expanduser('~/rs_lanes/tfmap_target/release/examples/tf_replay'))
ap.add_argument('--work', default='/data/rs_lanes/tfmap')
args = ap.parse_args()
rng = random.Random(args.seed)
S = 1_000_000_000

buf = tf2_ros.Buffer()
events, expected = [], []
frames = set()
recent = collections.deque(maxlen=300)
newest = [0]
EXTRA = ['nonexistent', '', '/map', 'map/', 'NO_PARENT']


def add(t, is_static):
    (buf.set_transform_static if is_static else buf.set_transform)(t, 'default_authority')
    tr, q = t.transform.translation, t.transform.rotation
    ns = t.header.stamp.sec * S + t.header.stamp.nanosec
    events.append('\t'.join(['T', '1' if is_static else '0', t.header.frame_id, t.child_frame_id, str(ns)]
                            + [repr(v) for v in (tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w)]))
    for f in (t.header.frame_id, t.child_frame_id):
        frames.add(f.lstrip('/') if f.startswith('/') else f)
    if not is_static and ns:
        recent.append(ns)
        newest[0] = max(newest[0], ns)


def tf(parent, child, ns, xyz=(0.0, 0.0, 0.0), yaw=0.0, q=None):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = parent, child
    t.header.stamp.sec, t.header.stamp.nanosec = ns // S, ns % S
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
    q = q or (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    r = t.transform.rotation
    r.x, r.y, r.z, r.w = q
    return t


def pick_time():
    r = rng.random()
    if r < 0.3 or not recent:
        return 0
    if r < 0.6:
        return newest[0] + rng.randint(-12 * S, S // 3)
    if r < 0.75:
        return rng.choice(recent)
    if r < 0.85:
        return rng.choice(recent) + rng.choice((-1, 1))
    if r < 0.95:
        return newest[0] + rng.choice((-1, 1)) * rng.randint(10 * S, 100 * S)
    return rng.choice((1, 5 * S, 2_000_000_000 * S))


def lookups(n):
    known = sorted(frames)
    for _ in range(n):
        pool = known if rng.random() < 0.9 else known + EXTRA
        target, source = rng.choice(pool), rng.choice(pool)
        if rng.random() < 0.03:
            source = target
        ns = max(pick_time(), 0)
        if rng.random() < 0.15:
            events.append(f'C\t{target}\t{source}\t{ns}')
            ok, msg = buf.can_transform_core(target, source, Time(nanoseconds=ns))
            expected.append(('C', ok, msg))
            continue
        events.append(f'L\t{target}\t{source}\t{ns}')
        try:
            t = buf.lookup_transform_core(target, source, Time(nanoseconds=ns))
            tr, q = t.transform.translation, t.transform.rotation
            expected.append(('ok', t.header.stamp.sec * S + t.header.stamp.nanosec,
                             (tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w)))
        except Exception as e:  # noqa: BLE001 - tf2 exception classes
            expected.append(('err', type(e).__name__, str(e)))


r = SequentialReader()
r.open(StorageOptions(uri=args.bag, storage_id='mcap'), ConverterOptions('', ''))
r.set_filter(StorageFilter(topics=['/tf', '/tf_static']))
count = 0
while r.has_next():
    topic, data, _ = r.read_next()
    for t in deserialize_message(data, TFMessage).transforms:
        add(t, topic == '/tf_static')
    count += 1
    if count % args.every == 0:
        lookups(args.per)
    if args.limit and count >= args.limit:
        break
bag_events = len(events)

# Synthetic edge cases, after the recording, in frames of their own.
t0 = newest[0] + S
for i in range(12):   # parent switch: odom until t0+5 s, then base_link (tf2 returns the older one between)
    add(tf('odom' if i < 6 else 'base_link', 'switcher', t0 + i * S, (i, 0.0, 0.0), 0.1 * i), False)
add(tf('loop_a', 'loop_b', t0), False)
add(tf('loop_b', 'loop_a', t0), False)
add(tf('/map', '/slashed', t0, (1.0, 2.0, 3.0), 0.5), True)
add(tf('map', 'flipper', 0, (1.0, 0.0, 0.0)), True)
add(tf('map', 'flipper', t0, (2.0, 0.0, 0.0)), False)   # static -> dynamic: history restarts
add(tf('map', 'flipper', t0 + S, (3.0, 0.0, 0.0)), False)
add(tf('map', 'unnormalized', t0, q=(0.0, 0.0, 2.0, 2.0)), False)   # normalized on insert
add(tf('map', 'flipped', t0, q=(0.0, 0.0, -0.6, -0.8)), False)      # sign changed on insert
add(tf('map', 'map', t0), False)                                       # self transform: ignored
add(tf('map', 'nan', t0, (float('nan'), 0.0, 0.0)), False)            # NaN: ignored
add(tf('map', 'switcher', t0 - 60 * S), False)                         # older than the cache: ignored
for i in range(400):
    lookups(1)
    frames.update(('switcher', 'loop_a', 'loop_b', 'slashed', 'flipper', 'unnormalized', 'flipped'))
    if i % 50 == 0:
        recent.extend(t0 + k * S + rng.randint(0, S) for k in range(12))
for target, source, ns in (('map', 'switcher', t0 + 5 * S + S // 2), ('odom', 'switcher', t0 + 2 * S + S // 3),
                           ('loop_a', 'map', 0), ('loop_b', 'loop_a', t0), ('slashed', 'map', 0),
                           ('/slashed', 'map', 0), ('map', 'flipper', 0), ('map', 'flipper', t0 + S // 2),
                           ('map', 'unnormalized', 0), ('flipped', 'map', 0), ('nan', 'map', 0)):
    events.append(f'L\t{target}\t{source}\t{ns}')
    try:
        t = buf.lookup_transform_core(target, source, Time(nanoseconds=ns))
        tr, q = t.transform.translation, t.transform.rotation
        expected.append(('ok', t.header.stamp.sec * S + t.header.stamp.nanosec, (tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w)))
    except Exception as e:  # noqa: BLE001
        expected.append(('err', type(e).__name__, str(e)))

path = os.path.join(args.work, 'tf_events.tsv')
with open(path, 'w') as f:
    f.write('\n'.join(events) + '\n')
out = subprocess.run([args.rust, path], capture_output=True, text=True, check=True).stdout.splitlines()
os.remove(path)
assert len(out) == len(expected), (len(out), len(expected))

stats = collections.Counter()
max_dt = max_dq = 0.0
shown = 0
for exp, line in zip(expected, out):
    got = [s.replace('\\n', '\n') for s in line.split('\t')]
    diff = False
    if exp[0] == 'C':
        stats['can_transform'] += 1
        diff = (got[0] == 'ok') != exp[1] or (not exp[1] and got[1] != exp[2])
        stats['can_transform_DIFF' if diff else 'can_transform_same'] += 1
    elif exp[0] == 'ok':
        stats['ok'] += 1
        if got[0] != 'ok' or int(got[1]) != exp[1]:
            stats['ok_DIFF'] += 1
            diff = True
        else:
            v = [float(x) for x in got[2:]]
            dt = max(abs(a - b) for a, b in zip(v[:3], exp[2][:3]))
            dq = max(abs(a - b) for a, b in zip(v[3:], exp[2][3:]))
            max_dt, max_dq = max(max_dt, dt), max(max_dq, dq)
            stats['ok_bit_exact' if v == list(exp[2]) else 'ok_within_ulps'] += 1
            if dt > 1e-9 or dq > 1e-9:
                stats['ok_TOLERANCE_EXCEEDED'] += 1
                diff = True
    else:
        stats['err'] += 1
        stats['err_' + exp[1]] += 1
        if got[0] != 'err' or got[1] != exp[1]:
            stats['err_CLASS_DIFF'] += 1
            diff = True
        elif got[2] != exp[2]:
            stats['err_MESSAGE_DIFF'] += 1
            diff = True
    if diff and shown < 10:
        shown += 1
        print('DIFF expected', exp, '\n     got     ', got)

print(f'{count} TF messages, {bag_events} events from the bag, {len(events)} with synthetic cases')
print(f'{len(expected)} queries: ' + ', '.join(f'{k} {v}' for k, v in sorted(stats.items())))
print(f'max |translation diff| {max_dt:.3g} m, max |quaternion diff| {max_dq:.3g}')
