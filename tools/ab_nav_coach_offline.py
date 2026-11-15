#!/usr/bin/env python3
"""Exact A/B of nav_coach: drives the Python node (roomba_perception/nav_coach.py, its callbacks
called directly) and the Rust core (examples/nav_coach_replay.rs) with the same input sequence and
compares every instruction published by the 5 Hz timer (JSON text) and, for each obstacle cloud,
the number of points in the stop zone and the resulting "blocked" flag. Nothing goes through ROS;
time.monotonic and the timer follow the script.
  tools/ab_nav_coach_offline.py [--bag RECORDING] [--rust BIN] [--work DIR]
Scenarios: scripted Nav2 situations (all instruction codes, hold time, collision-monitor stops,
obstacle clouds in and out of the stop zone, goal results, feedback) with the robot's and the
hand-held parameters, and with --bag a recording from tools/ab_explorer_live.sh (velocity commands,
collision monitor, obstacle clouds, TF, navigate_to_pose status and feedback at their recorded
times). Also checks bit for bit that the stop-zone transform matches numpy's matmul (MM).
Build the replay first: cd src/roomba_rs && cargo build --release --examples
"""
import argparse
import json
import math
import os
import struct
import subprocess
import sys
import types
from collections import Counter

import numpy as np
import rclpy
import rclpy.node
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import TransformStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav2_msgs.msg import CollisionMonitorState
from sensor_msgs.msg import PointCloud2, PointField

import roomba_perception.nav_coach as N

ROBOT = {'base_frame': 'base_link', 'robot_mode': True}   # nav.launch.py via robot.launch.py
HANDHELD = {}                                             # node defaults (camera_link, instructions)
T0 = 1000.0
CARGO_TARGET = os.environ.get('CARGO_TARGET_DIR') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src/roomba_rs/target')
H = None


class FakePub:
    def publish(self, msg):
        H.py.append(['IN', msg.data])


def fake_publisher(self, msg_type, topic, qos, *a, **k):
    if topic != '/nav_coach/instruction':
        return rclpy.node.Node.create_publisher(self, msg_type, topic, qos, *a, **k)
    return FakePub()


def fake_timer(self, period, cb, *a, **k):
    assert period == 0.2
    H.tick_cb = cb
    return object()


def subscribe(self, msg_type, topic, cb, qos):
    H.subs[topic] = cb


N.TransformListener = lambda *a, **k: None
N.time = types.SimpleNamespace(monotonic=lambda: H.t)
for name, fn in (('create_publisher', fake_publisher), ('create_subscription', subscribe),
                 ('create_timer', fake_timer)):
    setattr(N.NavCoach, name, fn)


def tf_msg(parent, child, ns, t, q):
    ts = TransformStamped()
    ts.header.frame_id, ts.child_frame_id = parent, child
    ts.header.stamp.sec, ts.header.stamp.nanosec = divmod(int(ns), 10**9)
    ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = t
    r = ts.transform.rotation
    r.x, r.y, r.z, r.w = q
    return ts


def cloud(points, frame='camera_link'):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    m = PointCloud2()
    m.header.frame_id = frame
    m.height, m.width = 1, len(pts)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
    m.point_step, m.row_step, m.is_dense = 12, 12 * len(pts), True
    m.data = pts.tobytes()
    return m


class Harness:
    def __init__(self, name, params, work):
        global H
        H = self
        self.name, self.params, self.work = name, dict(params), work
        self.t, self.py, self.trace, self.inputs = T0, [], [], []
        self.subs, self.tick_cb, self.counts = {}, None, Counter()
        self.cloud_path = os.path.join(work, f'coach_{name}.clouds')
        self.cloud_file, self.xy_file = open(self.cloud_path, 'wb'), open(self.cloud_path + '.xy', 'wb')
        self.mm = []

    def start(self):
        rclpy.init(args=['--ros-args'] + [a for k, v in self.params.items() for a in ('-p', f'{k}:={pv(v)}')])
        self.trace.append(['P', 0, self.params, self.cloud_path])
        self.node = N.NavCoach()

    def inp(self, dt, kind, payload=None):
        self.inputs.append((T0 + dt, len(self.inputs), kind, payload))

    def ev(self, kind, *p):
        self.trace.append([kind, self.t, *p])
        self.counts[kind] += 1

    def run(self, until, mm=False):
        events = self.inputs + [(T0 + k * 0.2, -1, 'K', None) for k in range(1, int(until / 0.2) + 1)]
        events.sort(key=lambda e: (e[0], e[1] < 0, e[1]))   # ticks after inputs at the same time
        node = self.node
        for t, _, kind, p in events:
            self.t = t
            if kind == 'K':
                self.ev('K')
                self.tick_cb()
            elif kind == 'TF':
                is_static, items = p
                for parent, child, ns, tt, q in items:
                    ts = tf_msg(parent, child, ns, tt, q)
                    (node.tf_buffer.set_transform_static if is_static else node.tf_buffer.set_transform)(
                        ts, 'default_authority')
                self.ev('TF', is_static, [[a, b, int(ns), *tt, *q] for a, b, ns, tt, q in items])
            elif kind in ('V', 'O'):
                vx, wz = p
                m = Twist()
                m.linear.x, m.angular.z = float(vx), float(wz)
                self.ev(kind, m.linear.x, m.angular.z)
                self.subs['/cmd_vel_nav' if kind == 'V' else '/cmd_vel'](m)
            elif kind == 'CM':
                self.ev('CM', p)
                self.subs['/collision_monitor_state'](CollisionMonitorState(action_type=p))
            elif kind == 'OB':
                self.obstacles(p, mm)
            elif kind == 'ST':
                m = GoalStatusArray()
                for sec, nsec, st in p:
                    g = GoalStatus(status=st)
                    g.goal_info.stamp.sec, g.goal_info.stamp.nanosec = sec, nsec
                    m.status_list.append(g)
                self.ev('ST', [list(s) for s in p])
                self.subs['/navigate_to_pose/_action/status'](m)
            elif kind == 'FB':
                m = NavigateToPose.Impl.FeedbackMessage()
                m.feedback.distance_remaining = float(np.float32(p))
                self.ev('FB', m.feedback.distance_remaining)
                self.subs['/navigate_to_pose/_action/feedback'](m)

    def obstacles(self, msg, mm):
        data = bytes(msg.data)
        off = self.cloud_file.tell()
        self.cloud_file.write(data)
        self.ev('OB', msg.header.frame_id, msg.width, off, len(data))
        node = self.node
        try:
            tf = node.tf_buffer.lookup_transform(node.base_frame, msg.header.frame_id, rclpy.time.Time())
        except Exception:   # noqa: BLE001 - tf2 exceptions: the node ignores the cloud
            tf = None
        n = None
        if tf is not None:   # the node's own computation, to compare the count
            pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3) if msg.width else np.zeros((0, 3))
            r, t = tf.transform.rotation, tf.transform.translation
            base = pts @ N._quat_to_matrix(r.x, r.y, r.z, r.w).T + np.array([t.x, t.y, t.z])
            n = int(np.count_nonzero((base[:, 0] > 0.0) & (base[:, 0] < node.stop_dist)
                                     & (np.abs(base[:, 1]) < node.stop_half_width)))
            if mm and msg.width:
                xy_off = self.xy_file.tell()
                self.xy_file.write(np.ascontiguousarray(base[:, :2], dtype='<f8').tobytes())
                self.trace.append(['MM', self.t, off, len(data), [t.x, t.y, t.z, r.x, r.y, r.z, r.w], xy_off])
        self.subs['/perception/obstacles'](msg)
        self.py.append(['AH', n, node.ahead[1]])

    def compare(self, rust):
        self.cloud_file.close()
        self.xy_file.close()
        path = os.path.join(self.work, f'coach_{self.name}.events')
        with open(path, 'w') as f:
            for e in self.trace:
                f.write(json.dumps(e) + '\n')
        with open(path) as f:
            res = subprocess.run([rust], stdin=f, capture_output=True, text=True)
        if res.returncode:
            sys.exit(f'rust replay failed: {res.stderr[-2000:]}')
        rs = [json.loads(line) for line in res.stdout.splitlines()]
        self.mm = [o for o in rs if o[0] == 'MM']
        rs = [o for o in rs if o[0] != 'MM']
        self.node.destroy_node()
        rclpy.shutdown()
        return json.loads(json.dumps(self.py)), rs


def pv(v):
    return ('true' if v else 'false') if isinstance(v, bool) else repr(v) if isinstance(v, float) else str(v)


# ---- scripted scenario ----

CAM = ('base_link', 'camera_link', 0, (0.153, 0.0, 0.13), (0.0, 0.0871557, 0.0, 0.9961947))   # pitched 10 deg


def ahead_cloud(n_in, n_out=40, frame='camera_link'):
    """Points in camera_link: n_in inside the stop zone, n_out beside or beyond it."""
    rng = np.random.default_rng(n_in * 1000 + n_out)
    inside = np.column_stack([rng.uniform(0.0, 0.25, n_in), rng.uniform(-0.2, 0.2, n_in), rng.uniform(-0.1, 0.1, n_in)])
    outside = np.column_stack([rng.uniform(0.4, 2.0, n_out), rng.uniform(-1.0, 1.0, n_out), rng.uniform(-0.2, 0.3, n_out)])
    # a few right at the edges of the zone
    edges = np.array([[0.45 - 0.153, 0.0, 0.0], [0.2, 0.22, 0.0], [0.2, -0.22, 0.0], [-0.153, 0.0, 0.0]])
    return cloud(np.vstack([inside, outside, edges]), frame)


def scripted(h):
    h.inp(0.0, 'TF', (True, [CAM]))
    seq = []
    t = 1.0

    def at(dt, kind, p=None):
        seq.append((t + dt, kind, p))
    # goal 1: accepted, drives through every movement code
    at(0.0, 'ST', [(100, 5, 1)])
    at(0.1, 'ST', [(100, 5, 2)])
    moves = [(0.2, 0.0), (0.2, 0.5), (0.2, -0.5), (0.0, 0.6), (0.0, -0.6), (-0.1, 0.0), (0.02, 0.05),
             (0.2, 0.36), (0.2, 0.35), (0.039999, 0.15), (0.04, 0.149)]
    for i, (vx, wz) in enumerate(moves):
        for k in range(8):          # 8 x 0.1 s each: longer than hold_s
            at(1.0 + i * 1.0 + k * 0.1, 'O', (vx, wz))
            at(1.0 + i * 1.0 + k * 0.1, 'V', (vx, wz))
        at(1.0 + i * 1.0 + 0.5, 'FB', 3.0 - i * 0.255)
    t += 13.0
    # flicker: alternating faster than hold_s never switches
    for k in range(20):
        at(k * 0.15, 'O', (0.2, 0.5 if k % 2 else -0.5))
    t += 4.0
    # collision monitor stop (and one older than 1 s), controller zeroed by the monitor
    at(0.0, 'CM', 1)
    at(0.0, 'O', (0.2, 0.0))
    at(0.5, 'O', (0.2, 0.0))
    at(1.5, 'CM', 0)
    at(1.6, 'O', (0.0, 0.0))
    at(1.6, 'V', (0.2, 0.0))
    at(2.4, 'O', (0.0, 0.0))
    at(2.4, 'V', (0.2, 0.0))
    t += 4.0
    # obstacles ahead: forward -> stop; turning in place -> not; backing up -> not; too few points
    for k, (vx, wz, n_in) in enumerate([(0.2, 0.0, 30), (0.0, 0.6, 30), (-0.1, 0.0, 30), (0.2, 0.0, 10),
                                        (0.2, 0.0, 15), (0.2, 0.0, 14), (0.2, 0.0, 0)]):
        for j in range(6):
            at(k * 1.2 + j * 0.2, 'O', (vx, wz))
            at(k * 1.2 + j * 0.2 + 0.05, 'OB', ahead_cloud(n_in))
    at(9.0, 'OB', ahead_cloud(30, frame='unknown_frame'))           # no TF: ignored
    at(9.2, 'OB', cloud(np.zeros((0, 3))))                           # empty cloud
    t += 12.0
    # results: succeeded -> arrived (held 15 s, then idle), canceled, aborted, unknown status
    for k, final in enumerate([4, 5, 6, 0]):
        sec = 200 + k
        at(k * 20.0, 'ST', [(sec, 0, 2), (sec - 1, 0, 4)])
        at(k * 20.0 + 0.5, 'FB', 1.005 + k)
        at(k * 20.0 + 0.6, 'O', (0.2, 0.0))
        at(k * 20.0 + 2.0, 'ST', [(sec, 0, final), (sec - 1, 0, 4)])
    t += 85.0
    # ties in the status list: the first of the latest stamps wins
    at(0.0, 'ST', [(300, 7, 2), (300, 7, 4)])
    at(1.0, 'ST', [(300, 7, 6), (300, 7, 2)])
    at(2.0, 'ST', [(301, 0, 3)])
    at(3.0, 'ST', [])
    at(4.0, 'ST', [(301, 0, 5)])
    t += 6.0
    for tt, kind, p in seq:
        h.inp(tt, kind, p)
    return t + 2.0


def bag_inputs(h, bag):
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
    from tf2_msgs.msg import TFMessage
    types_ = {'/cmd_vel_nav': Twist, '/cmd_vel': Twist, '/collision_monitor_state': CollisionMonitorState,
              '/perception/obstacles': PointCloud2, '/tf': TFMessage, '/tf_static': TFMessage,
              '/navigate_to_pose/_action/status': GoalStatusArray,
              '/navigate_to_pose/_action/feedback': NavigateToPose.Impl.FeedbackMessage}
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='mcap'), ConverterOptions('', ''))
    r.set_filter(StorageFilter(topics=list(types_)))
    start, last = None, 0.0
    while r.has_next():
        topic, data, t_ns = r.read_next()
        start = start if start is not None else t_ns
        dt = (t_ns - start) * 1e-9 + 0.01
        last = dt
        m = deserialize_message(data, types_[topic])
        if topic in ('/tf', '/tf_static'):
            items = [(ts.header.frame_id, ts.child_frame_id, ts.header.stamp.sec * 10**9 + ts.header.stamp.nanosec,
                      (ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z),
                      (ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z,
                       ts.transform.rotation.w)) for ts in m.transforms]
            h.inp(dt, 'TF', (topic == '/tf_static', items))
        elif topic in ('/cmd_vel_nav', '/cmd_vel'):
            h.inp(dt, 'V' if topic == '/cmd_vel_nav' else 'O', (m.linear.x, m.angular.z))
        elif topic == '/collision_monitor_state':
            h.inp(dt, 'CM', m.action_type)
        elif topic == '/perception/obstacles':
            h.inp(dt, 'OB', m)
        elif topic.endswith('status'):
            h.inp(dt, 'ST', [(s.goal_info.stamp.sec, s.goal_info.stamp.nanosec, s.status) for s in m.status_list])
        else:
            h.inp(dt, 'FB', m.feedback.distance_remaining)
    return last + 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag')
    ap.add_argument('--rust', default=os.path.join(CARGO_TARGET, 'release/examples/nav_coach_replay'))
    ap.add_argument('--work', default='/data/rs_lanes/explorer/ab_work')
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    runs = [('scripted_robot', ROBOT, None), ('scripted_handheld', HANDHELD, None)]
    if args.bag:
        runs += [('bag_robot', ROBOT, args.bag), ('bag_handheld', HANDHELD, args.bag)]
    ok = True
    for name, params, bag in runs:
        h = Harness(name, params, args.work)
        until = bag_inputs(h, bag) if bag else scripted(h)
        h.start()
        h.run(until, mm=True)
        py, rs = h.compare(args.rust)
        first = next((i for i, (a, b) in enumerate(zip(py, rs)) if a != b),
                     None if len(py) == len(rs) else min(len(py), len(rs)))
        codes = Counter(json.loads(o[1])['code'] for o in py if o[0] == 'IN')
        ah = [o for o in py if o[0] == 'AH']
        print(f'== {name}: inputs ' + ' '.join(f'{k} {v}' for k, v in sorted(h.counts.items())))
        print(f'   instructions {sum(codes.values())} '
              f'({", ".join(f"{k} {v}" for k, v in codes.most_common())}); obstacle clouds {len(ah)} '
              f'(blocked {sum(1 for o in ah if o[2])}, no TF {sum(1 for o in ah if o[1] is None)})')
        if h.mm:
            n, a, b = (sum(o[i] for o in h.mm) for i in (1, 2, 3))
            print(f'   stop-zone transform vs numpy: {n} points, bit-exact with fused multiply-add {n - a}, '
                  f'with plain multiply/add {n - b}')
        if first is None:
            print(f'   RESULT: identical ({len(py)} outputs)')
        else:
            ok = False
            print(f'   RESULT: DIFFERENT at output {first} of {len(py)} / {len(rs)}')
            for i in range(max(0, first - 3), min(first + 3, max(len(py), len(rs)))):
                print(f'     py {py[i] if i < len(py) else "-"}\n     rs {rs[i] if i < len(rs) else "-"}')
    print('ALL IDENTICAL' if ok else 'DIFFERENCES FOUND')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
