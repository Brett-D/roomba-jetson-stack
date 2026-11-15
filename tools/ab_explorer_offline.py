#!/usr/bin/env python3
"""Exact A/B of explorer: drives the Python node (roomba_perception/explorer.py, its callbacks called
directly) and the Rust core (examples/explorer_replay.rs) with the same scripted input sequence and
compares every output in order: status JSON, NavigateToPose goals (frame, position, orientation,
stamp), cancels, controller parameter get/set requests, keep-out commands, cliff timers and log
lines, plus the frontier goals computed for every costmap (Q events). Nothing goes through ROS:
time.monotonic, the node clock and the wall clock (mark names, UTC) follow the script. A scripted
NavigateToPose server answers goals (accept / reject, succeed / abort / never, cancel -> canceled)
and fake controller parameter services answer get/set.
  tools/ab_explorer_offline.py [--scenario NAME ...] [--bag RECORDING] [--rust BIN] [--work DIR]
Synthetic scenarios run in a simulated house (ground truth grid, ray-cast "camera" revealing the
costmap, robot teleported to goals that succeed, doors, invisible railings producing bumper hits).
--bag adds a scenario from a recording made by tools/ab_explorer_live.sh: its costmaps, TF, sensors,
battery, mode and explorer commands at their recorded times, goals answered by the scripted server,
and frontier queries on every costmap at the robot's and at random positions.
Build the replay first: cd src/roomba_rs && cargo build --release --examples
"""
import argparse
import hashlib
import heapq
import json
import math
import os
import random
import subprocess
import sys
import time
import types
from collections import Counter

import cv2
import numpy as np
import rclpy
import rclpy.node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterValue
from rclpy.time import Time
from roomba_msgs.msg import SensorState
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String

import roomba_perception.explorer as E

ROBOT = {'base_frame': 'base_link', 'low_battery_fraction': 0.2, 'bump_mark_width_m': 0.5,
         'bump_light_signal': 200}   # nav.launch.py on the robot
T0, ROS0, WALL0 = 1000.0, 1794697180 * 10**9, 1794697180.0 - 1000.0
RES = float(np.float32(0.05))
CLIFFS = ('cliff_left', 'cliff_front_left', 'cliff_front_right', 'cliff_right')
LIGHTS = ('light_bump_left_signal', 'light_bump_front_left_signal', 'light_bump_center_left_signal',
          'light_bump_center_right_signal', 'light_bump_front_right_signal', 'light_bump_right_signal')
CARGO_TARGET = os.environ.get('CARGO_TARGET_DIR') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src/roomba_rs/target')
H = None   # the running harness (the patched node methods report to it)


def f32(v):
    return float(np.float32(v))


# ---- fakes for the node's ROS interfaces ----

class FakePub:
    KINDS = {'~/status': 'ST', '/perception/ignore_zones/command': 'ZN'}

    def __init__(self, topic):
        self.kind = self.KINDS[topic]

    def publish(self, msg):
        H.py.append([self.kind, msg.data])


class FakeFuture:
    def __init__(self):
        self.cbs, self.value, self.exc = [], None, None

    def add_done_callback(self, cb):
        self.cbs.append(cb)

    def result(self):
        if self.exc is not None:
            raise self.exc
        return self.value

    def set(self, value=None, exc=None):
        self.value, self.exc = value, exc
        for cb in self.cbs:
            cb(self)


class FakeClient:
    def __init__(self, name):
        self.name = name

    def service_is_ready(self):
        return H.ready['get' if self.name.endswith('get_parameters') else 'set']

    def call_async(self, req):
        f = FakeFuture()
        if self.name == '/controller_server/get_parameters':
            assert list(req.names) == [E.ROTATE_PARAM]
            H.py.append(['GP'])
            H.push(H.t + 0.05, 1, 'PR', f)
        else:
            assert self.name == '/controller_server/set_parameters'
            (p,) = req.parameters
            assert p.name == E.ROTATE_PARAM and p.value.type == 3
            H.py.append(['SP', p.value.double_value])
        return f


class FakeHandle:
    def __init__(self, seq, accepted):
        self.seq, self.accepted = seq, accepted

    def cancel_goal_async(self):
        H.py.append(['CN', self.seq])
        H.server_cancel(self.seq)
        return FakeFuture()

    def get_result_async(self):
        f = FakeFuture()
        H.result_futs[self.seq] = f
        return f


class FakeActionClient:
    def __init__(self, node, action_type, name):
        assert name == 'navigate_to_pose'

    def server_is_ready(self):
        return H.ready['nav']

    def send_goal_async(self, goal):
        seq, p = H.node.goal_seq, goal.pose
        H.py.append(['GL', seq, p.header.frame_id, p.pose.position.x, p.pose.position.y, p.pose.orientation.z,
                     p.pose.orientation.w, p.header.stamp.sec * 10**9 + p.header.stamp.nanosec])
        f = FakeFuture()
        H.server_goal(seq, p.pose, f)
        return f


class Log:
    def info(self, m, *a, **k):
        H.py.append(['LG', 'info', m])

    def warn(self, m, *a, **k):
        H.py.append(['LG', 'warn', m])

    warning = warn

    def error(self, m, *a, **k):
        H.py.append(['LG', 'error', m])


class FakeClock:
    def now(self):
        return Time(nanoseconds=H.ros)


def fake_timer(self, period, cb, *a, **k):
    if H.tick_cb is None:
        assert period == 1.0
        H.tick_cb = cb
    else:
        assert period == 0.6
        H.py.append(['TM'])
        H.push(H.t + period, 2, 'CT', cb)
    return object()


def subscribe(self, msg_type, topic, cb, qos):
    H.subs[topic] = cb


def fake_publisher(self, msg_type, topic, qos, *a, **k):
    if topic not in FakePub.KINDS:   # rclpy's own (/parameter_events)
        return rclpy.node.Node.create_publisher(self, msg_type, topic, qos, *a, **k)
    return FakePub(topic)


E.ActionClient = FakeActionClient
E.TransformListener = lambda *a, **k: None
E.time = types.SimpleNamespace(monotonic=lambda: H.t,
                               strftime=lambda fmt: time.strftime(fmt, time.gmtime(math.floor(WALL0 + H.t))))
for name, fn in (('create_publisher', fake_publisher),
                 ('create_client', lambda self, t, name: FakeClient(name)),
                 ('create_subscription', subscribe), ('create_timer', fake_timer),
                 ('destroy_timer', lambda self, timer: None), ('get_clock', lambda self: FakeClock()),
                 ('get_logger', lambda self: Log())):
    setattr(E.Explorer, name, fn)


def tf_msg(parent, child, ns, t, q):
    ts = TransformStamped()
    ts.header.frame_id, ts.child_frame_id = parent, child
    ts.header.stamp.sec, ts.header.stamp.nanosec = divmod(int(ns), 10**9)
    ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = t
    r = ts.transform.rotation
    r.x, r.y, r.z, r.w = q
    return ts


def yaw_q(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


# ---- the harness: one scenario, both implementations ----

class Harness:
    def __init__(self, name, params, work):
        global H
        H = self
        self.name, self.params, self.work = name, dict(params), work
        self.t, self.ros = T0, ROS0
        self.py, self.trace, self.heap, self.n = [], [], [], 0
        self.ready = {'nav': True, 'get': True, 'set': True}
        self.subs, self.tick_cb, self.result_futs = {}, None, {}
        self.rs_entry, self.dead = {}, set()
        self.policy, self.goal_n, self.goal_pose = None, 0, {}
        self.param_reply = (3, 1.2, None)
        self.world, self.counts = None, Counter()
        self.grid_path = os.path.join(work, f'{name}.grids')
        self.grid_file, self.grid_off, self.grid_index = open(self.grid_path, 'wb'), 0, {}
        self.node = None

    def start(self):
        """Builds the node with the scenario's parameters (after it scheduled its inputs)."""
        args = ['--ros-args'] + [a for k, v in self.params.items() for a in ('-p', f'{k}:={pv(v)}')]
        rclpy.init(args=args)
        self.trace.append(['P', 0, 0, self.params, WALL0, self.grid_path])
        self.node = E.Explorer()

    # scheduling: (time, priority, order); inputs 0, server/param responses 1, cliff timer 2, tick 3
    def push(self, t, prio, kind, payload=None):
        self.n += 1
        heapq.heappush(self.heap, (t, prio, self.n, kind, payload))
        return self.n

    def inp(self, dt, kind, payload=None):
        return self.push(T0 + dt, 0, kind, payload)

    def ev(self, kind, *payload):
        self.trace.append([kind, self.t, self.ros, *payload])
        self.counts[kind] += 1

    def run(self, until):
        k = 1
        self.push(T0 + k, 3, 'K')
        while self.heap:
            t, prio, n, kind, payload = heapq.heappop(self.heap)
            if t > T0 + until:
                break
            self.t, self.ros = t, ROS0 + round((t - T0) * 1e9)
            if kind == 'K':
                self.ev('K')
                self.tick_cb()
                k += 1
                self.push(T0 + k, 3, 'K')
            elif n not in self.dead:
                self.dispatch(kind, payload)
        self.ev('X')
        self.node._cancel_goal()

    def dispatch(self, kind, p):
        node = self.node
        if kind == 'TF':
            is_static, items = p
            for parent, child, ns, t, q in items:
                ts = tf_msg(parent, child, ns, t, q)
                if is_static:
                    node.tf_buffer.set_transform_static(ts, 'default_authority')
                else:
                    node.tf_buffer.set_transform(ts, 'default_authority')
            self.ev('TF', is_static, [[a, b, int(ns), *t, *q] for a, b, ns, t, q in items])
        elif kind == 'G':
            self.grid(p)
        elif kind == 'M':
            self.ev('M', p)
            self.subs['/roomba/mode'](String(data=p))
        elif kind == 'B':
            v = f32(p)
            self.ev('B', None if v != v else v)
            self.subs['/battery_state'](BatteryState(percentage=v))
        elif kind == 'F':
            self.ev('F', p)
            self.subs['/vslam_filter/status'](String(data=p))
        elif kind == 'C':
            self.ev('C', p)
            self.subs['~/command'](String(data=p))
        elif kind == 'S':
            m = p if isinstance(p, SensorState) else sensors(**p)
            self.ev('S', m.bump_left, m.bump_right, m.wheel_drop_left, m.wheel_drop_right,
                    [getattr(m, c) for c in CLIFFS], [getattr(m, s) for s in LIGHTS])
            self.subs['/roomba/sensors'](m)
        elif kind == 'NR':
            self.ready.update(p)
            self.ev('NR', self.ready['nav'], self.ready['get'], self.ready['set'])
        elif kind == 'GR':
            seq, accepted, fut = p
            self.ev('GR', seq, accepted)
            fut.set(FakeHandle(seq, accepted))
        elif kind == 'RS':
            seq, status = p
            self.rs_entry.pop(seq, None)
            if callable(status):
                status = status()
            self.ev('RS', seq, status)
            fut = self.result_futs.pop(seq, None)
            if fut is not None:
                fut.set(types.SimpleNamespace(status=status))
        elif kind == 'PR':
            typ, value, err = self.param_reply
            self.ev('PR', typ, value, err)
            if err is not None:
                p.set(exc=RuntimeError(err))
            else:
                p.set(types.SimpleNamespace(values=[] if typ is None else [ParameterValue(type=typ, double_value=value)]))
        elif kind == 'CT':
            self.ev('CT')
            p()
        elif kind == 'Q':
            self.ev('Q', *(p or ()))
            self.py.append(self.frontier_query(p))
        elif kind == 'CALL':
            p()
        else:
            raise ValueError(kind)

    def grid(self, msg):
        info = msg.info
        data = np.asarray(msg.data, dtype=np.int8).tobytes()
        key = hashlib.sha1(data).hexdigest()
        if key not in self.grid_index:
            self.grid_file.write(data)
            self.grid_index[key] = self.grid_off
            self.grid_off += len(data)
        self.ev('G', self.grid_index[key], len(data), info.width, info.height, info.resolution,
                info.origin.position.x, info.origin.position.y)
        self.subs['/global_costmap/costmap'](msg)

    def frontier_query(self, xy):
        node = self.node
        if node.grid is None:
            return ['FQ', None]
        if xy is None:
            pose = node._pose()
            if pose is None:
                return ['FQ', None]
            xy = pose[:2]
        info = node.grid.info
        grid = np.asarray(node.grid.data, dtype=np.int16).reshape(info.height, info.width)
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        rc = (int((xy[1] - oy) / res), int((xy[0] - ox) / res))
        goals, seed = E.frontier_goals(grid, rc, node.free_max, node.pass_max, node.clear_max, node.min_cells,
                                       node.standoff / res, node.goal_search / res)
        return ['FQ', [[int(g[0][0]), int(g[0][1]), float(g[1][0]), float(g[1][1]), int(g[2])] for g in goals],
                None if seed is None else [int(seed[0]), int(seed[1])], [rc[0], rc[1]]]

    # ---- scripted NavigateToPose server ----
    def server_goal(self, seq, pose, fut):
        n, self.goal_n = self.goal_n, self.goal_n + 1
        yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)
        self.goal_pose[seq] = (pose.position.x, pose.position.y, yaw)
        spec = self.policy(n, seq, pose.position.x, pose.position.y, yaw) if self.policy else {}
        resp = spec.get('resp', 0.05)
        accept = spec.get('accept', True)
        self.push(self.t + resp, 1, 'GR', (seq, accept, fut))
        if accept and spec.get('result') is not None:
            self.rs_entry[seq] = self.push(self.t + resp + spec['result'], 1, 'RS', (seq, spec.get('status', 4)))

    def server_cancel(self, seq):
        if seq in self.rs_entry:
            self.dead.add(self.rs_entry.pop(seq))
        self.rs_entry[seq] = self.push(self.t + 0.1, 1, 'RS', (seq, 5))

    # ---- the Rust replay ----
    def compare(self, rust):
        self.grid_file.close()
        trace = os.path.join(self.work, f'{self.name}.events')
        with open(trace, 'w') as f:
            for e in self.trace:
                f.write(json.dumps(e) + '\n')
        with open(trace) as f:
            res = subprocess.run([rust], stdin=f, capture_output=True, text=True)
        if res.returncode:
            sys.exit(f'rust replay failed: {res.stderr[-2000:]}')
        rs = [json.loads(line) for line in res.stdout.splitlines()]
        py = json.loads(json.dumps(self.py))   # same JSON round trip as the Rust side
        self.node.destroy_node()
        rclpy.shutdown()
        return py, rs


def pv(v):
    return ('true' if v else 'false') if isinstance(v, bool) else repr(v) if isinstance(v, float) else str(v)


def sensors(**kw):
    m = SensorState()
    for k, v in kw.items():
        setattr(m, k, v)
    return m


# ---- simulated house ----

class World:
    """Ground truth (0 free, 100 wall), a ray-cast camera revealing it into a Nav2-like costmap
    (unknown -1, walls 100, inscribed 99, inflation to 0.3 m), the robot's pose on TF (map -> odom
    -> base_link), doors that open and close, and invisible railings (in the truth for driving but
    never in the costmap, like posts the camera misses)."""

    def __init__(self, h, truth, start, origin=(-2.025, -1.475), radius=2.5):
        self.h, self.truth, self.origin, self.radius = h, truth, origin, radius
        self.known = np.zeros(truth.shape, bool)
        self.invisible = np.zeros(truth.shape, bool)
        self.pose, self.slam_ok, self.doors = start, True, {}

    def cell(self, x, y):
        return int((y - self.origin[1]) / RES), int((x - self.origin[0]) / RES)

    def xy(self, r, c):
        return self.origin[0] + (c + 0.5) * RES, self.origin[1] + (r + 0.5) * RES

    def door(self, name, rows, cols, closed=False):
        self.doors[name] = (rows, cols)
        self.set_door(name, closed)

    def set_door(self, name, closed):
        rows, cols = self.doors[name]
        self.truth[rows[0]:rows[1], cols[0]:cols[1]] = 100 if closed else 0

    def reveal(self):
        r0, c0 = self.cell(*self.pose[:2])
        hh, ww = self.truth.shape
        for a in np.linspace(0, 2 * math.pi, 720, endpoint=False):
            dr, dc = math.sin(a), math.cos(a)
            for s in np.arange(0, self.radius / RES, 0.5):
                r, c = int(r0 + 0.5 + dr * s), int(c0 + 0.5 + dc * s)
                if not (0 <= r < hh and 0 <= c < ww):
                    break
                self.known[r, c] = True
                if self.truth[r, c] == 100 and not self.invisible[r, c]:
                    break

    def costmap(self):
        wall = self.known & (self.truth == 100) & ~self.invisible
        d = cv2.distanceTransform((~wall).astype(np.uint8), cv2.DIST_L2, 5) * RES
        cost = np.where(d <= 0.0, 100, np.where(d <= 0.17, 99, np.where(
            d <= 0.3, (98 * np.exp(-6.0 * (d - 0.17))).astype(int), 0)))
        cost = np.where(self.known, cost, -1).astype(np.int8)
        m = OccupancyGrid()
        m.header.frame_id = 'map'
        m.info.resolution = RES
        m.info.height, m.info.width = cost.shape
        m.info.origin.position.x, m.info.origin.position.y = self.origin
        m.info.origin.orientation.w = 1.0
        m.data = cost.ravel().tolist()
        return m

    def tf(self):
        ns = self.h.ros
        x, y, yaw = self.pose
        items = [('odom', 'base_link', ns, (x - 0.3, y + 0.2, 0.0), yaw_q(yaw))]
        if self.slam_ok:
            items.insert(0, ('map', 'odom', ns, (0.3, -0.2, 0.0), (0.0, 0.0, 0.0, 1.0)))
        return items

    def publish(self, reveal=True):
        if reveal:
            self.reveal()
        self.h.dispatch('TF', (False, self.tf()))
        self.h.dispatch('G', self.costmap())
        self.h.dispatch('Q', None)

    def reachable(self, x, y):
        free = (self.truth == 0).astype(np.uint8)
        _, labels = cv2.connectedComponents(free, connectivity=8)
        a, b = self.cell(*self.pose[:2]), self.cell(x, y)
        hh, ww = free.shape
        if not (0 <= b[0] < hh and 0 <= b[1] < ww):
            return False
        return labels[a] != 0 and labels[a] == labels[b]

    def blocked_by_railing(self, x, y):
        """First invisible cell on the straight line to (x, y), as the stopping point before it."""
        (r0, c0), (r1, c1) = self.cell(*self.pose[:2]), self.cell(x, y)
        n = max(abs(r1 - r0), abs(c1 - c0), 1)
        prev = self.pose[:2]
        for i in range(n + 1):
            r, c = round(r0 + (r1 - r0) * i / n), round(c0 + (c1 - c0) * i / n)
            if self.invisible[r, c]:
                return prev
            prev = self.xy(r, c)
        return None


def house(rows=200, cols=240):
    """12 x 10 m: four rooms, doors 0.8 m wide, a sofa and a table."""
    g = np.zeros((rows, cols), np.int8)
    g[:2, :] = g[-2:, :] = g[:, :2] = g[:, -2:] = 100
    g[:, 118:120] = 100
    g[98:100, :118] = 100
    g[120:122, 120:] = 100
    g[40:56, 118:120] = 0       # door A: west <-> east
    g[98:100, 30:46] = 0        # door B: south-west <-> north-west
    g[120:122, 170:186] = 0     # door C: east <-> north-east
    g[20:36, 40:80] = 100       # sofa
    g[140:150, 160:175] = 100   # table
    return g


def moving_policy(h, speed=0.3, fail=None):
    """Goals take 2 s + distance / speed; reachable ones succeed and the robot moves there,
    unreachable ones abort after 3 s (fail(n, x, y) -> spec overrides it)."""
    def policy(n, seq, x, y, yaw):
        if fail:
            spec = fail(n, x, y)
            if spec is not None:
                return spec
        d = math.hypot(x - h.world.pose[0], y - h.world.pose[1])

        def outcome():
            if h.world.reachable(x, y):
                h.world.pose = (x, y, yaw)
                h.world.publish()
                return 4
            return 6
        return {'result': 2.0 + d / speed, 'status': outcome}
    return policy


def start_world(h, truth, start=(0.5, 0.5, 0.3), battery=0.9, mode='full', cmd_at=2.0):
    h.world = World(h, truth, start)
    h.inp(0.5, 'CALL', h.world.publish)
    h.inp(0.6, 'M', mode)
    h.inp(0.7, 'B', battery)
    h.inp(0.8, 'F', '{"state": "idle", "pose_jumps": 0}')
    if cmd_at is not None:
        h.inp(cmd_at, 'C', 'start')
    periodic_tf(h, 1.0, 3000.0)


def periodic_tf(h, t0, t1):
    """The robot's TF every second (SLAM / odometry keep publishing)."""
    for t in np.arange(t0, t1, 1.0):
        h.inp(float(t), 'CALL', lambda: h.dispatch('TF', (False, h.world.tf())))


# ---- scenarios ----

def sc_explore_house(h):
    """Explores the whole house (goals succeed, the robot moves) until nothing is left, goes home
    and checks again before finishing."""
    start_world(h, house())
    h.policy = moving_policy(h)
    return 1500


def sc_failures(h):
    """Goals rejected, aborted, succeeding without moving, never answered (60 s timeout)."""
    start_world(h, house())
    kinds = ['reject', 'abort', 'stay', 'never', 'move', 'abort', 'move', 'stay', 'move']

    def fail(n, x, y):
        k = kinds[n % len(kinds)]
        return {'reject': {'accept': False, 'resp': 0.2}, 'abort': {'result': 3.0, 'status': 6},
                'stay': {'result': 1.5, 'status': 4}, 'never': {'result': None}}.get(k)
    h.policy = moving_policy(h, fail=fail)
    return 1500


def sc_door_closed(h, reopen=None):
    """Door B shuts while the robot heads for the room behind it: that goal aborts (blacklisted)
    and the room becomes a wall in the map; the explorer retries failed areas once before
    finishing. With reopen, the door opens again later and the room gets explored."""
    truth = house()
    start_world(h, truth)
    h.world.door('B', (98, 100), (30, 46))
    state = {'closed': False}

    def fail(n, x, y):
        r, c = h.world.cell(x, y)
        if r > 100 and c < 118 and not state['closed']:
            state['closed'] = True
            h.push(h.t + 1.0, 0, 'CALL', lambda: (h.world.set_door('B', True), h.world.known.__setitem__(
                (slice(96, 102), slice(28, 48)), True), h.world.publish(reveal=False)))
            if reopen:
                h.push(h.t + reopen, 0, 'CALL', lambda: (h.world.set_door('B', False), h.world.publish(reveal=False)))
        return None
    h.policy = moving_policy(h, fail=fail)
    return 1800


def sc_door_reopens(h):
    return sc_door_closed(h, reopen=40.0)


def sc_railing(h):
    """An invisible railing across the east room: goals beyond it make the robot bump (switches,
    then the light bumpers) - two bumps skip the frontier and mark the railing; later, with a
    short mark TTL, marks expire and are re-added; a wheel drop and passive mode suppress marks."""
    truth = house()
    start_world(h, truth)
    h.world.invisible[60:120, 180:182] = True
    truth[60:120, 180:182] = 100

    def bump(t, held=0.2, **kw):
        h.push(t, 0, 'S', dict(kw))
        h.push(t + held, 0, 'S', {})

    def fail(n, x, y):
        stop = h.world.blocked_by_railing(x, y)
        if stop is None:
            return None
        sx, sy = stop
        yaw = math.atan2(y - sy, x - sx)

        def drive():
            h.world.pose = (sx, sy, yaw)
            h.world.publish()
        h.push(h.t + 2.0, 0, 'CALL', drive)
        h.push(h.t + 2.5, 0, 'S', {'bump_left': True})
        bump(h.t + 2.6, bump_left=True)                                   # still pressed: one hit
        bump(h.t + 4.0, light_bump_center_left_signal=900, light_bump_front_left_signal=300)
        bump(h.t + 5.0, bump_right=True, wheel_drop_left=True)            # wheel drop: no mark
        return {'result': 8.0, 'status': 6}
    h.policy = moving_policy(h, fail=fail)
    # sensors at 1 Hz otherwise, plus bumps while idle in passive mode
    h.inp(0.0, 'CALL', lambda: [h.inp(t, 'S', {}) for t in np.arange(1.1, 1500.0, 1.0)])
    h.inp(1.2, 'S', {'bump_left': True})        # robot_mode not set yet -> no mark
    h.inp(1.4, 'S', {})
    return 1500


def sc_cliff(h):
    """Cliff stops (mode safe/full -> passive): keep-out zone ahead of the fired sensors; none after
    a wheel drop or without a cliff; a warning without a robot position; exploring stops."""
    start_world(h, house(), mode='safe')
    h.policy = moving_policy(h)
    seq = [(20.0, {'cliff_front_left': True}, 'passive'), (40.0, None, 'safe'), (41.0, {}, None),
           (60.0, {'cliff_left': True, 'cliff_front_left': True}, 'passive'), (62.0, None, 'full'),
           (63.0, None, 'C:start'),
           (80.0, {'cliff_right': True, 'wheel_drop_right': True}, 'passive'), (85.0, None, 'full'),
           (100.0, {}, 'passive'), (101.0, None, 'safe'), (102.0, None, 'C:start'),
           (130.0, {'cliff_front_right': True, 'cliff_right': True}, 'passive'), (130.3, None, 'off'),
           (131.0, None, 'full')]
    for t, s, mode in seq:
        if s is not None:
            h.inp(t - 0.3, 'S', s)
            h.inp(t + 0.4, 'S', {})
        if mode and mode.startswith('C:'):
            h.inp(t, 'C', mode[2:])
        elif mode:
            h.inp(t, 'M', mode)
    # SLAM lost: no position for the cliff at 160 s (map -> odom stops at 140 s, stale after 10 s)
    h.inp(139.0, 'CALL', lambda: setattr(h.world, 'slam_ok', False))
    h.inp(159.7, 'S', {'cliff_left': True})
    h.inp(160.0, 'M', 'passive')
    h.inp(170.0, 'CALL', lambda: setattr(h.world, 'slam_ok', True))
    h.inp(171.0, 'M', 'safe')
    return 200


def sc_low_battery(h):
    """Battery runs low while exploring: back home (succeeds and arrives), then start is refused;
    NaN readings are ignored; a second run goes home while a goal response is still pending."""
    start_world(h, house())
    h.policy = moving_policy(h)
    for t, v in ((30.0, 0.5), (31.0, float('nan')), (60.0, 0.15), (61.0, float('nan'))):
        h.inp(t, 'B', v)
    h.inp(150.0, 'C', 'start')
    h.inp(160.0, 'B', 0.95)
    h.inp(161.0, 'C', 'start')
    h.inp(200.0, 'CALL', lambda: setattr(h, 'policy', lambda n, s, x, y, yaw: {'resp': 3.0, 'result': 5.0}))
    h.inp(206.5, 'B', 0.1)
    return 400


def sc_commands(h):
    """Start refused (no driving mode, no TF, no Nav2), unknown / odd commands, stop, resume,
    home while exploring and while returning, start while returning."""
    h.world = World(h, house(), (0.5, 0.5, 0.0))
    seq = [(1.0, 'C', 'start'), (2.0, 'M', 'full'), (3.0, 'C', 'start'), (4.0, 'CALL', h.world.publish),
           (5.0, 'NR', {'nav': False}), (6.0, 'C', 'start'), (7.0, 'NR', {'nav': True}),
           (8.0, 'C', '  START \n'), (9.0, 'C', 'Start'), (10.0, 'C', 'explore'), (11.0, 'C', ''),
           (40.0, 'C', 'stop'), (41.0, 'C', 'stop'), (50.0, 'C', 'start'), (80.0, 'C', 'home'),
           (81.0, 'C', 'start'), (82.0, 'C', 'home'), (100.0, 'C', 'stop'), (110.0, 'C', 'home'),
           (130.0, 'C', 'start'), (131.0, 'C', 'stop'), (132.0, 'C', 'start')]
    for t, k, v in seq:
        h.inp(t, k, v)
    periodic_tf(h, 4.5, 300.0)
    h.policy = moving_policy(h)
    return 300


def sc_tf_and_costmap(h):
    """Exploring before any costmap; SLAM stalls (map -> odom stops, lookups fail after the 10 s
    cache) and recovers; the robot ends up inside an obstacle (no free seed)."""
    h.world = World(h, house(), (0.5, 0.5, 0.0))
    h.inp(0.5, 'M', 'full')
    h.inp(1.0, 'CALL', lambda: h.dispatch('TF', (False, h.world.tf())))
    h.inp(1.5, 'C', 'start')
    periodic_tf(h, 2.0, 400.0)
    h.inp(6.5, 'CALL', h.world.publish)
    h.inp(30.0, 'CALL', lambda: setattr(h.world, 'slam_ok', False))
    h.inp(60.0, 'CALL', lambda: setattr(h.world, 'slam_ok', True))
    h.inp(100.0, 'CALL', lambda: (setattr(h.world, 'pose', (1.0, -0.05, 0.0)), h.world.publish()))  # in the sofa
    h.inp(130.0, 'CALL', lambda: (setattr(h.world, 'pose', (0.5, 0.5, 0.0)), h.world.publish()))
    h.policy = moving_policy(h)
    return 400


def sc_tracking(h):
    """Visual tracking recovery pauses exploring; odd status JSON (invalid, NaN, no state)."""
    start_world(h, house())
    h.policy = moving_policy(h)
    for t, s in ((10.0, '{"state": "recovering"}'), (20.0, 'not json'), (25.0, '{"state": "idle", "x": NaN}'),
                 (40.0, '{"state": "lost", "y": Infinity}'), (50.0, '{}'), (60.0, '{"state": null}'),
                 (70.0, '{"state": "idle"}')):
        h.inp(t, 'F', s)
    return 300


def sc_no_frontiers(h, return_home=True, stay_home=False):
    """A closed, fully known flat: nothing to explore - wait finish_confirm_s, go home, check again
    and finish; with return_home false it finishes at once; variants where Nav2 reports success
    without the robot moving home."""
    truth = np.full((80, 100), 100, np.int8)
    truth[2:-2, 2:-2] = 0
    if not return_home:
        h.params['return_home'] = False
    start_world(h, truth, start=(1.0, 1.0, 0.0))
    h.world.known[:] = True
    if stay_home:
        # pushed away after the start, then "home" succeeds without the robot moving
        h.inp(10.0, 'CALL', lambda: setattr(h.world, 'pose', (2.5, 1.5, 0.0)))
        h.policy = lambda n, s, x, y, yaw: {'result': 2.0, 'status': 4}
    else:
        h.policy = moving_policy(h)
    h.inp(100.0, 'C', 'start')
    return 200


def sc_no_frontiers_no_home(h):
    return sc_no_frontiers(h, return_home=False)


def sc_home_not_reached(h):
    return sc_no_frontiers(h, stay_home=True)


def sc_all_blacklisted(h):
    """Every goal aborts: all frontiers end up blacklisted, retried once, then exploring finishes."""
    start_world(h, house())
    h.policy = lambda n, s, x, y, yaw: {'result': 2.0, 'status': 6}
    return 600


def sc_param_services(h):
    """Controller turn speed: get returns a double / an integer / fails; services missing; rotate
    speed restored on stop and finish."""
    start_world(h, house())
    h.policy = moving_policy(h)
    h.param_reply = (2, 0.0, None)
    seq = [(20.0, 'C', 'stop'), (21.0, 'CALL', lambda: setattr(h, 'param_reply', (None, 0.0, 'service died'))),
           (22.0, 'C', 'start'), (30.0, 'C', 'stop'), (31.0, 'CALL', lambda: setattr(h, 'param_reply', (3, 1.8, None))),
           (32.0, 'C', 'start'), (40.0, 'C', 'stop'), (41.0, 'NR', {'get': False}), (42.0, 'C', 'start'),
           (50.0, 'C', 'stop'), (51.0, 'NR', {'set': False, 'get': True}), (52.0, 'C', 'start'), (60.0, 'C', 'stop'),
           (61.0, 'NR', {'set': True}), (62.0, 'C', 'start')]
    for t, k, v in seq:
        h.inp(t, k, v)
    return 150


def sc_rotate_off(h):
    h.params['rotate_speed_rps'] = 0.0
    return sc_explore_house(h)


def sc_superseded(h):
    """Commands while a goal request is unanswered: home / stop / start / timeout - late accepted
    responses are cancelled, late rejections ignored; shutdown cancels the active goal."""
    start_world(h, house())
    slow = {'n': 0}

    def policy(n, seq, x, y, yaw):
        slow['n'] += 1
        if n % 3 == 0:
            return {'resp': 4.0, 'result': 6.0, 'status': 4}
        if n % 3 == 1:
            return {'resp': 4.0, 'accept': False}
        return {'resp': 0.1, 'result': None}
    h.policy = policy
    for t, c in ((5.5, 'home'), (12.0, 'stop'), (13.0, 'start'), (15.0, 'stop'), (20.0, 'start')):
        h.inp(t, 'C', c)
    h.params['goal_timeout_s'] = 25.0
    return 150


SCENARIOS = {name[3:]: fn for name, fn in globals().items() if name.startswith('sc_')}


# ---- recorded scenario ----

def sc_bag(h, bag, queries=3):
    """A recording from tools/ab_explorer_live.sh at its own pace: costmaps, TF, sensors, battery,
    mode, filter status and the explorer commands sent during it; goals answered by the scripted
    server (no motion: the robot follows the recording)."""
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
    from tf2_msgs.msg import TFMessage
    types_ = {'/global_costmap/costmap': OccupancyGrid, '/tf': TFMessage, '/tf_static': TFMessage,
              '/roomba/sensors': SensorState, '/roomba/mode': String, '/battery_state': BatteryState,
              '/vslam_filter/status': String, '/explorer/command': String}
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='mcap'), ConverterOptions('', ''))
    r.set_filter(StorageFilter(topics=list(types_)))
    start = None
    rnd = random.Random(7)
    kinds = [('succeed', 6.0), ('abort', 3.0), ('succeed', 1.0), ('reject', 0), ('never', 0), ('succeed', 10.0),
             ('abort', 8.0), ('succeed', 2.0)]

    def policy(n, seq, x, y, yaw):
        k, d = kinds[n % len(kinds)]
        return {'reject': {'accept': False}, 'never': {'result': None}, 'abort': {'result': d, 'status': 6}}.get(
            k, {'result': d, 'status': 4})
    h.policy = policy
    last = 0.0
    while r.has_next():
        topic, data, t_ns = r.read_next()
        start = start if start is not None else t_ns
        dt = (t_ns - start) * 1e-9 + 0.01
        m = deserialize_message(data, types_[topic])
        last = dt
        if topic in ('/tf', '/tf_static'):
            items = [(ts.header.frame_id, ts.child_frame_id, ts.header.stamp.sec * 10**9 + ts.header.stamp.nanosec,
                      (ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z),
                      (ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z,
                       ts.transform.rotation.w)) for ts in m.transforms]
            h.inp(dt, 'TF', (topic == '/tf_static', items))
        elif topic == '/global_costmap/costmap':
            h.inp(dt, 'G', m)
            h.inp(dt, 'Q', None)
            info = m.info
            for _ in range(queries):   # frontier goals from random spots on the costmap
                x = info.origin.position.x + rnd.uniform(0, info.width) * info.resolution
                y = info.origin.position.y + rnd.uniform(0, info.height) * info.resolution
                h.inp(dt, 'Q', (x, y))
        elif topic == '/roomba/sensors':
            h.inp(dt, 'S', m)
        elif topic == '/roomba/mode':
            h.inp(dt, 'M', m.data)
        elif topic == '/battery_state':
            h.inp(dt, 'B', m.percentage)
        elif topic == '/vslam_filter/status':
            h.inp(dt, 'F', m.data)
        else:
            h.inp(dt, 'C', m.data)
    return last + 2.0


def summarize(name, h, py, rs):
    kinds = Counter(o[0] for o in py)
    first = next((i for i, (a, b) in enumerate(zip(py, rs)) if a != b), None if len(py) == len(rs) else min(len(py), len(rs)))
    states = Counter(json.loads(o[1])['state'] for o in py if o[0] == 'ST')
    msgs = {json.loads(o[1])['message'] for o in py if o[0] == 'ST'}
    logs = Counter(o[1] for o in py if o[0] == 'LG')
    fq = [o for o in py if o[0] == 'FQ']
    print(f'== {name}: {h.counts["K"]} ticks, inputs ' + ' '.join(f'{k} {v}' for k, v in sorted(h.counts.items())
                                                              if k not in ('K', 'X')))
    print(f'   outputs python {len(py)} rust {len(rs)}: ' + ' '.join(f'{k} {v}' for k, v in sorted(kinds.items())) +
          f'  (frontier queries with goals: {sum(1 for o in fq if o[1])}, goals compared: '
          f'{sum(len(o[1]) for o in fq if o[1])})')
    print(f'   states {dict(states)}; {len(msgs)} distinct messages; logs {dict(logs)}')
    if first is None:
        print('   RESULT: identical')
        return True
    print(f'   RESULT: DIFFERENT at output {first}')
    for i in range(max(0, first - 3), min(first + 3, max(len(py), len(rs)))):
        print(f'     py {str(py[i])[:300] if i < len(py) else "-"}\n     rs {str(rs[i])[:300] if i < len(rs) else "-"}')
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', nargs='*', default=None, help=f'default: all of {", ".join(SCENARIOS)}')
    ap.add_argument('--bag', help='recording from tools/ab_explorer_live.sh (adds scenario "bag")')
    ap.add_argument('--queries', type=int, default=3, help='random-position frontier queries per costmap (bag)')
    ap.add_argument('--rust', default=os.path.join(CARGO_TARGET, 'release/examples/explorer_replay'))
    ap.add_argument('--work', default='/data/rs_lanes/explorer/ab_work')
    ap.add_argument('--messages', action='store_true', help='print every distinct status message')
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    names = args.scenario if args.scenario else list(SCENARIOS)
    if args.bag and (args.scenario is None or 'bag' in names):
        names = [n for n in names if n != 'bag'] + ['bag']
    ok = True
    total = Counter()
    for name in names:
        h = Harness(name, ROBOT, args.work)
        until = sc_bag(h, args.bag, args.queries) if name == 'bag' else SCENARIOS[name](h)
        h.start()
        h.run(until)
        py, rs = h.compare(args.rust)
        ok &= summarize(name, h, py, rs)
        for o in py:
            total[o[0]] += 1
        if args.messages:
            for m in sorted({json.loads(o[1])['message'] for o in py if o[0] == 'ST'}):
                print('     ' + m)
    print('TOTAL outputs compared: ' + ' '.join(f'{k} {v}' for k, v in sorted(total.items())))
    print('ALL IDENTICAL' if ok else 'DIFFERENCES FOUND')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
