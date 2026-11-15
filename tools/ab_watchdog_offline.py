#!/usr/bin/env python3
"""Exact A/B of vslam_watchdog: feeds the same recorded message sequence (tools/record_vio_replay.sh)
to the Python node's callbacks and to the Rust core (examples/watchdog_replay.rs) and compares every
output: TF, odometry (/odom or fused_vio_topic), status JSON, service calls and log lines. Nothing
goes through ROS transport, so nothing is dropped. time.monotonic() and the 10 Hz / 1 Hz timers
follow the bag's receive times; service calls are answered after fixed delays.
  tools/ab_watchdog_offline.py BAG [--mode ekf|tf|handheld] [--services all|no_rtabmap|no_reset]
                               [--offset S] [--duration S] [--inject] [--no-imu] [--param NAME:=VALUE]
                               [--delay SERVICE=S] [--rust BIN]
  --mode ekf       robot.launch.py fusion:=ekf: robot limits, wheels, planar, publish_tf false (B)
         tf        robot limits, wheels, planar, publish_tf true (pipeline A behaviour)
         handheld  node defaults: no wheel odometry, 3D output, publish_tf true
  --inject         adds faults every 150 s (glitch burst, 2.5 s tracking loss, rotation glitch)
"""
import argparse
import heapq
import math
import os
import re
import subprocess
import sys
from collections import Counter

import rclpy
from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from sensor_msgs.msg import Imu
from tf2_msgs.msg import TFMessage

import roomba_perception.vslam_watchdog as W

ROBOT = {'base_frame': 'base_link', 'max_speed_mps': 0.6, 'max_accel_mps2': 1.5, 'max_vertical_speed_mps': 0.3,
         'max_height_m': 1.0, 'wheel_odom_topic': '/wheel/odom', 'planar': True}  # vslam.launch.py _watchdog()
MODES = {'ekf': dict(ROBOT, publish_tf=False), 'tf': dict(ROBOT), 'handheld': {}}
SERVICES = ('pause', 'reset', 'new_map', 'resume')
READY = {'all': '1111', 'no_rtabmap': '0100', 'no_reset': '1011'}
DELAY = {'pause': 0.05, 'reset': 0.4, 'new_map': 0.1, 'resume': 0.05}
IMU, WHEEL, STATUS, VIO, TF_STATIC = ('/camera/imu/data_raw', '/wheel/odom', '/visual_slam/status',
                                      '/visual_slam/tracking/odometry', '/tf_static')

ap = argparse.ArgumentParser()
ap.add_argument('bag')
ap.add_argument('--mode', choices=MODES, default='ekf')
ap.add_argument('--services', choices=READY, default='all')
ap.add_argument('--offset', type=float, default=0.0)
ap.add_argument('--duration', type=float, default=1e9)
ap.add_argument('--inject', action='store_true')
ap.add_argument('--no-imu', action='store_true', help='drop the IMU (turn-rate checks, no gyro/leveling)')
ap.add_argument('--param', action='append', default=[], metavar='NAME:=VALUE', help='extra node parameter')
ap.add_argument('--delay', action='append', default=[], metavar='SERVICE=S', help='service response delay')
ap.add_argument('--rust', default=os.path.expanduser('~/rust_probe/target/release/examples/watchdog_replay'))
args = ap.parse_args()
params = dict(MODES[args.mode], **dict(p.split(':=', 1) for p in args.param))
DELAY.update({k: float(v) for k, v in (d.split('=', 1) for d in args.delay)})


def pstr(v):
    return ('true' if v else 'false') if isinstance(v, bool) else str(v)


rclpy.init(args=['--ros-args'] + [a for k, v in params.items() for a in ('-p', f'{k}:={pstr(v)}')])
node = W.VslamWatchdog()
sim = [0.0]
W.time.monotonic = lambda: sim[0]
py = []


class Log:
    def info(self, m, *a, **k): py.append(('LG', 'info', m))
    def warn(self, m, *a, **k): py.append(('LG', 'warn', m))
    warning = warn
    def error(self, m, *a, **k): py.append(('LG', 'error', m))


node.get_logger = lambda: Log()
xyz = lambda v: (v.x, v.y, v.z)  # noqa: E731
xyzw = lambda q: (q.x, q.y, q.z, q.w)  # noqa: E731


def send_tf(ts):
    b, a = ts
    assert (b.header.frame_id, b.child_frame_id, a.header.frame_id, a.child_frame_id) == (
        node.odom_frame, node.base_frame, node.odom_frame, node.vslam_frame)
    py.append(('TF', b.header.stamp.sec, b.header.stamp.nanosec, *xyz(b.transform.translation),
               *xyzw(b.transform.rotation), *xyz(a.transform.translation), *xyzw(a.transform.rotation)))


def pub_odom(m):
    assert (m.header.frame_id, m.child_frame_id) == (node.odom_frame, node.base_frame)
    py.append(('OD', m.header.stamp.sec, m.header.stamp.nanosec, *xyz(m.pose.pose.position),
               *xyzw(m.pose.pose.orientation), *map(float, m.pose.covariance), *xyz(m.twist.twist.linear),
               *xyz(m.twist.twist.angular), *map(float, m.twist.covariance)))


node.tf_pub.sendTransform = send_tf
node.pub_odom.publish = pub_odom
node.pub_status.publish = lambda m: py.append(('ST', m.data))

# Service clients: availability per --services, answered after DELAY.
ready = {n: c == '1' for n, c in zip(SERVICES, READY[args.services])}
pending, seq = [], [0]


class Future:
    def __init__(self, name):
        self.name = name

    def add_done_callback(self, cb):
        seq[0] += 1
        heapq.heappush(pending, (sim[0] + DELAY[self.name], seq[0], self.name, cb, self))

    def result(self):
        return None


def fake_call(name):
    def call_async(_request):
        py.append(('CA', name))
        return Future(name)
    return call_async


for name, cli in zip(SERVICES, (node.cli_pause, node.cli_reset, node.cli_new_map, node.cli_resume)):
    cli.service_is_ready = (lambda n=name: ready[n])
    cli.call_async = fake_call(name)

ev = [f'P,{k},{pstr(v)}' for k, v in params.items()] + ['R,' + ','.join(READY[args.services])]
r = repr
reader = SequentialReader()
reader.open(StorageOptions(uri=args.bag, storage_id='mcap'), ConverterOptions('', ''))
topics = ([] if args.no_imu else [IMU]) + [STATUS, VIO, TF_STATIC] + ([WHEEL] if params.get('wheel_odom_topic') else [])
reader.set_filter(StorageFilter(topics=topics))
bag_start = reader.get_metadata().starting_time.nanoseconds
start = bag_start + int(args.offset * 1e9)
t0 = start * 1e-9
k_tick = k_stat = 1


def feed_tf_static(data):
    for ts in deserialize_message(data, TFMessage).transforms:
        ev.append(f'F,{ts.header.frame_id},{ts.child_frame_id},' + ','.join(r(v) for v in xyzw(ts.transform.rotation)))
        node.tf_buffer.set_transform_static(ts, 'bag')


# /tf_static is latched: also deliver what was published before the offset.
pre = SequentialReader()
pre.open(StorageOptions(uri=args.bag, storage_id='mcap'), ConverterOptions('', ''))
pre.set_filter(StorageFilter(topics=[TF_STATIC]))
while pre.has_next():
    _, data, t_ns = pre.read_next()
    if t_ns >= start:
        break
    feed_tf_static(data)
del pre
reader.seek(start)


def advance(t):
    """Fires the timers and service responses due up to t, in time order."""
    global k_tick, k_stat
    while True:
        tt, ts = t0 + k_tick * 0.1, t0 + k_stat * 1.0
        td = pending[0][0] if pending else math.inf
        m = min(td, tt, ts)
        if m > t:
            return
        sim[0] = m
        if td == m:
            _, _, name, cb, fut = heapq.heappop(pending)
            ev.append(f'D,{r(m)},{name}')
            cb(fut)
        elif tt == m:
            ev.append(f'T,{r(m)}')
            node._tick()
            k_tick += 1
        else:
            ev.append(f'U,{r(m)}')
            node._publish_status()
            k_stat += 1


def inject_phase(t):
    """Fault injection schedule (bag time since the start of the replay)."""
    if not args.inject:
        return None
    s = (t - t0 - 60.0) % 150.0
    if t - t0 < 60.0:
        return None
    if s < 1.5:
        return 'glitch'
    if 60.0 <= s < 62.5:
        return 'lost'
    if 100.0 <= s < 100.25:
        return 'rot'
    return None


counts = Counter()
vio_n = 0
while reader.has_next():
    topic, data, t_ns = reader.read_next()
    t = t_ns * 1e-9
    if t > t0 + args.duration:
        break
    advance(t)
    sim[0] = t
    counts[topic] += 1
    phase = inject_phase(t)
    if topic == IMU:
        m = deserialize_message(data, Imu)
        w, a, s = m.angular_velocity, m.linear_acceleration, m.header.stamp
        ev.append(f'I,{r(t)},{s.sec},{s.nanosec},{m.header.frame_id},{r(w.x)},{r(w.y)},{r(w.z)},{r(a.x)},{r(a.y)},{r(a.z)}')
        node._on_imu(m)
    elif topic == WHEEL:
        m = deserialize_message(data, Odometry)
        p, q, s = m.pose.pose.position, m.pose.pose.orientation, m.header.stamp
        ev.append(f'W,{r(t)},{s.sec},{s.nanosec},{r(p.x)},{r(p.y)},{r(q.x)},{r(q.y)},{r(q.z)},{r(q.w)},'
                  f'{r(m.twist.twist.linear.x)},{r(m.twist.twist.angular.z)}')
        node._on_wheel(m)
    elif topic == STATUS:
        m = deserialize_message(data, VisualSlamStatus)
        if phase == 'lost':
            m.vo_state = W.VO_FAILED
        ev.append(f'S,{r(t)},{m.vo_state}')
        node._on_status(m)
    elif topic == VIO:
        m = deserialize_message(data, Odometry)
        vio_n += 1
        if phase == 'glitch' and vio_n % 2:
            m.pose.pose.position.x += 0.4
        elif phase == 'rot':
            q = m.pose.pose.orientation
            q.x, q.y, q.z, q.w = W.quat_mul((0.0, 0.0, math.sin(0.26), math.cos(0.26)), (q.x, q.y, q.z, q.w))
        p, q, s, tw = m.pose.pose.position, m.pose.pose.orientation, m.header.stamp, m.twist.twist
        vals = (*xyz(p), *xyzw(q), *m.pose.covariance, *xyz(tw.linear), *xyz(tw.angular), *m.twist.covariance)
        ev.append(f'O,{r(t)},{s.sec},{s.nanosec},' + ','.join(r(float(v)) for v in vals))
        node._on_odom(m)
    else:
        feed_tf_static(data)
advance(t + 1.0)

res = subprocess.run([args.rust], input='\n'.join(ev) + '\n', capture_output=True, text=True)
if res.returncode:
    sys.exit(f'rust replay failed: {res.stderr}')
rs = []
for line in res.stdout.splitlines():
    k, rest = line.split(',', 1)
    if k in ('TF', 'OD'):
        f = rest.split(',')
        rs.append((k, int(f[0]), int(f[1]), *map(float, f[2:])))
    elif k == 'LG':
        rs.append((k, *rest.split(',', 1)))
    else:
        rs.append((k, rest))

dur = t - t0
print(f'{args.bag}  mode {args.mode}  services {args.services}{"  +faults" if args.inject else ""}'
      f'{"  no IMU" if args.no_imu else ""}  {" ".join(args.param + args.delay)}  '
      f'{dur:.0f} s replayed from {args.offset:.0f} s')
print('  input: ' + '  '.join(f'{k.split("/")[-1]} {v}' for k, v in sorted(counts.items())) +
      f'  ticks {k_tick - 1}  status {k_stat - 1}')
print(f'  python node: filtered {node.filtered_total}  recoveries {node.recoveries}  wheel_followed '
      f'{node.wheel_followed}  bridged {node.bridged}  forwarded {node.forwarded}')
kinds_py, kinds_rs = [o[0] for o in py], [o[0] for o in rs]
same_order = kinds_py == kinds_rs
print(f'  outputs: python {len(py)}  rust {len(rs)}  same sequence of kinds: {same_order}')
first_diff = next((i for i, (a, b) in enumerate(zip(py, rs)) if a[0] != b[0] or a[0] not in ('TF', 'OD') and a != b),
                  None if len(py) == len(rs) else min(len(py), len(rs)))
ok = same_order and first_diff is None
for kind, groups in (('TF', (('base pos', 3, 6), ('base quat', 6, 10), ('anchor pos', 10, 13), ('anchor quat', 13, 17))),
                     ('OD', (('pos', 3, 6), ('quat', 6, 10), ('pose cov', 10, 46), ('twist', 46, 88))),
                     ('ST', ()), ('CA', ()), ('LG', ())):
    a = [o for o in py if o[0] == kind]
    b = [o for o in rs if o[0] == kind]
    line = f'  {kind}: python {len(a)}  rust {len(b)}'
    if groups:
        stamps = sum(x[1:3] == y[1:3] for x, y in zip(a, b))
        worst = [max((abs(x[i] - y[i]) for x, y in zip(a, b) for i in range(lo, hi)), default=0.0) for _, lo, hi in groups]
        # Relative to the value (at least 1): the VIO frame can run off by kilometres while
        # tracking is lost, and so does the odom -> vslam_odom anchor that cancels it.
        rel = [max((abs(x[i] - y[i]) / max(1.0, abs(x[i])) for x, y in zip(a, b) for i in range(lo, hi)), default=0.0)
               for _, lo, hi in groups]
        big = max((abs(x[i]) for x in a for i in range(3, 17 if kind == 'TF' else 6)), default=0.0)
        line += (f'  same stamps {stamps}  max abs diff ' + '  '.join(f'{g[0]} {w:.1e}' for g, w in zip(groups, worst)) +
                 f'  (largest |translation| {big:.3g} m, max relative diff {max(rel, default=0.0):.1e})')
        ok &= stamps == len(a) == len(b) and max(rel, default=0.0) < 1e-9
    else:
        same = sum(x == y for x, y in zip(a, b))
        line += f'  identical {same}'
        ok &= same == len(a) == len(b)
    print(line)
calls = [o[1] for o in py if o[0] == 'CA']
print('  service calls (python): ' + ('  '.join(f'{k} {v}' for k, v in Counter(calls).items()) or 'none') +
      (f'  sequence {" ".join(calls[:9])} ...' if calls else ''))
logs = Counter((o[1], re.sub(r'[-+]?\d+(\.\d+)?', 'N', o[2])) for o in py if o[0] == 'LG')
for (lvl, msg), n in sorted(logs.items(), key=lambda x: -x[1]):
    print(f'    {n:5d}x {lvl:5s} {msg[:110]}')
if first_diff is not None:
    print(f'  FIRST DIFFERENCE at output {first_diff}:')
    for i in range(max(0, first_diff - 2), min(first_diff + 3, max(len(py), len(rs)))):
        print(f'    py {str(py[i])[:200] if i < len(py) else "-"}\n    rs {str(rs[i])[:200] if i < len(rs) else "-"}')
print('  RESULT: ' + ('identical' if ok else 'DIFFERENT'))
node.destroy_node()
rclpy.shutdown()
sys.exit(0 if ok else 1)
