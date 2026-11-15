#!/usr/bin/env python3
"""Drives roomba_web's server.py and the Rust web_server side by side (started by tools/ab_web.sh
with their outputs remapped under /py and /rs) and compares them.

Inputs are scripted (same messages to both): TF, VIO status, maps, paths, costmap, zones,
detections, robot battery/sensors/mode, navigation instructions, explorer status, recorded camera
frames and annotated JPEGs; a fake NavigateToPose action server and /roomba/set_mode service per
server answer the clients. Compared: every HTTP response (status, body, relevant headers) for all
routes x methods x auth states plus traversal / encoding / conditional / range cases, the
websocket state stream at steady checkpoints and as the set of distinct states, MJPEG frames,
and every published command (cmd_vel, zone commands, explorer commands, floor grid, goals,
cancels, set_mode). Writes OUT/ab_web.json and prints a summary.
"""
import argparse
import asyncio
import json
import math
import os
import re
import threading
import time

import aiohttp
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped, Twist
from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import deserialize_message
from roomba_msgs.msg import SensorState
from roomba_msgs.srv import SetMode
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from sensor_msgs.msg import BatteryState, CompressedImage, Image
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

SIDES = ('py', 'rs')
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
IGNORED_HEADERS = {'date', 'server', 'connection', 'keep-alive', 'transfer-encoding'}


# ---------------------------------------------------------------- ROS side

class FakeNav:
    def __init__(self, node, side):
        self.accept, self.finish, self.current = True, False, None
        self.goals, self.cancels = [], 0
        self.server = ActionServer(node, NavigateToPose, f'/{side}/navigate_to_pose', execute_callback=self.execute,
                                   goal_callback=self.on_goal, cancel_callback=self.on_cancel,
                                   callback_group=ReentrantCallbackGroup())

    def on_goal(self, goal):
        p = goal.pose
        self.goals.append({'frame': p.header.frame_id, 'x': p.pose.position.x, 'y': p.pose.position.y,
                           'qw': p.pose.orientation.w, 'accepted': self.accept})
        return GoalResponse.ACCEPT if self.accept else GoalResponse.REJECT

    def on_cancel(self, _handle):
        self.cancels += 1
        return CancelResponse.ACCEPT

    def execute(self, handle):
        # like Nav2, a new goal preempts the running one (keeps one executor thread busy per server)
        self.current = handle
        while not handle.is_cancel_requested and not self.finish and self.current is handle:
            time.sleep(0.05)
        if handle.is_cancel_requested:
            handle.canceled()
        else:
            handle.succeed()
        return NavigateToPose.Result()


class Harness(Node):
    def __init__(self):
        super().__init__('ab_web_harness')
        p = self.create_publisher
        self.pub = {
            'jpeg': p(CompressedImage, '/perception/person_detector/annotated/compressed', 2),
            'dets': p(Detection3DArray, '/perception/person_detector/detections_3d', 5),
            'ignored': p(Detection3DArray, '/perception/person_detector/ignored_3d', 5),
            'map': p(OccupancyGrid, '/rtabmap/map', LATCHED),
            'map_filled': p(OccupancyGrid, '/rtabmap/map_filled', LATCHED),
            'path': p(Path, '/rtabmap/mapPath', 1),
            'zones': p(String, '/perception/ignore_zones', LATCHED),
            'status': p(VisualSlamStatus, '/visual_slam/status', 10),
            'loc': p(PoseWithCovarianceStamped, '/rtabmap/localization_pose', 1),
            'instruction': p(String, '/nav_coach/instruction', 10),
            'plan': p(Path, '/plan', 1),
            'costmap': p(OccupancyGrid, '/local_costmap/costmap', 1),
            'floor': p(String, '/perception/floor_status', 2),
            'filter': p(String, '/vslam_filter/status', 2),
            'mode': p(String, '/roomba/mode', LATCHED),
            'battery': p(BatteryState, '/battery_state', 2),
            'sensors': p(SensorState, '/roomba/sensors', 2),
            'explore': p(String, '/explorer/status', LATCHED),
            'image': p(Image, '/camera/left/image_rect_mono', 5),
            'tf': p(TFMessage, '/tf', 100),
            'tf_static': p(TFMessage, '/tf_static', QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)),
        }
        self.lock = threading.Lock()
        self.outputs = {s: [] for s in SIDES}       # (t, topic, data)
        for side in SIDES:
            for topic, typ, qos in (('cmd_vel', Twist, 10), ('cmd_vel_nav', Twist, 10), ('zone_command', String, 10),
                                    ('floor_grid', Bool, LATCHED)):
                self.create_subscription(typ, f'/{side}/{topic}', self._recorder(side, topic), qos)
        self.explore_subs = {}
        self.nav = {}
        self.mode_srv = {}
        self.mode_reply = (True, 'Safe mode')
        self.mode_requests = {s: [] for s in SIDES}
        self.live = {}                              # what the 10 Hz loop keeps publishing
        self.tf_pose = (1.0, 2.0, 0.5)             # map -> base_link (x, y, yaw)
        self.create_timer(0.1, self._tick)
        self.create_timer(0.05, self._tf_tick)
        static = [self.tf_msg('base_link', 'camera_link', 0.135, 0.0, 0.0, z=0.12),
                  self.tf_msg('camera_link', 'camera_left_optical_frame', 0.0, 0.03, -math.pi / 2, roll=-math.pi / 2)]
        self.pub['tf_static'].publish(TFMessage(transforms=static))

    def _recorder(self, side, topic):
        def cb(msg):
            if isinstance(msg, Twist):
                data = (msg.linear.x, msg.angular.z)
            else:
                data = msg.data
            with self.lock:
                self.outputs[side].append((time.monotonic(), topic, data))
        return cb

    def explorer_listeners(self, on):
        for side in SIDES:
            if on and side not in self.explore_subs:
                self.explore_subs[side] = self.create_subscription(String, f'/{side}/explorer_command',
                                                                   self._recorder(side, 'explorer_command'), 10)
            elif not on and side in self.explore_subs:
                self.destroy_subscription(self.explore_subs.pop(side))

    def nav_servers(self, on):
        for side in SIDES:
            if on and side not in self.nav:
                self.nav[side] = FakeNav(self, side)
            elif not on and side in self.nav:
                nav = self.nav.pop(side)
                nav.finish = True
                time.sleep(0.2)
                nav.server.destroy()

    def mode_services(self):
        for side in SIDES:
            def cb(req, res, side=side):
                self.mode_requests[side].append(req.mode)
                res.success, res.message = self.mode_reply
                res.mode = req.mode
                return res
            self.mode_srv[side] = self.create_service(SetMode, f'/{side}/set_mode', cb)

    def stamp(self):
        return self.get_clock().now().to_msg()

    def tf_msg(self, parent, child, x, y, yaw, z=0.0, roll=0.0):
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = self.stamp(), parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
        cr, sr, cy, sy = math.cos(roll / 2), math.sin(roll / 2), math.cos(yaw / 2), math.sin(yaw / 2)
        t.transform.rotation.x, t.transform.rotation.y = sr * cy, sr * sy
        t.transform.rotation.z, t.transform.rotation.w = cr * sy, cr * cy
        return t

    def _tf_tick(self):
        x, y, yaw = self.tf_pose
        self.pub['tf'].publish(TFMessage(transforms=[self.tf_msg('map', 'odom', 0.5, -0.25, 0.1),
                                                     self.tf_msg('odom', 'base_link', x, y, yaw)]))

    def _tick(self):
        live = dict(self.live)
        for key, msg in live.items():
            if key in ('dets', 'ignored'):
                msg.header.stamp = self.stamp()
            self.pub[key].publish(msg)


def detections(frame, items):
    msg = Detection3DArray()
    msg.header.frame_id = frame
    for cls, score, x, y, z, did in items:
        d = Detection3D()
        d.id = did
        d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = x, y, z
        if cls is not None:
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = cls, score
            d.results.append(h)
        msg.detections.append(d)
    return msg


def grid(frame, data, res=0.05, origin=(-3.0, -2.5), yaw=0.0):
    g = OccupancyGrid()
    g.header.frame_id = frame
    g.info.resolution = res
    g.info.height, g.info.width = data.shape
    g.info.origin.position.x, g.info.origin.position.y = origin
    g.info.origin.orientation.z, g.info.origin.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    g.data = data.astype(np.int8).ravel().tolist()
    return g


def path(frame, pts):
    p = Path()
    p.header.frame_id = frame
    for x, y in pts:
        ps = PoseStamped()
        ps.pose.position.x, ps.pose.position.y = x, y
        p.poses.append(ps)
    return p


# ---------------------------------------------------------------- HTTP side

class Resp:
    def __init__(self, status, headers, body):
        self.status, self.headers, self.body = status, headers, body

    def header(self, name):
        for k, v in self.headers:
            if k.lower() == name.lower():
                return v
        return None

    def key(self, cookie_token=None):
        hdrs = []
        for k, v in self.headers:
            k = k.lower()
            if k in IGNORED_HEADERS:
                continue
            if k == 'content-length' and (self.header('transfer-encoding') or (v == '0' and not self.body)):
                continue   # framing only: an empty chunked body and Content-Length: 0 are the same
            if k == 'set-cookie' and cookie_token:
                v = re.sub(r'roomba_session=[A-Za-z0-9_-]{43}', 'roomba_session=<token>', v)
            hdrs.append((k, v))
        return self.status, sorted(hdrs), self.body


def dechunk(data):
    out, i = b'', 0
    while True:
        j = data.index(b'\r\n', i)
        n = int(data[i:j].split(b';')[0], 16)
        if n == 0:
            return out
        out += data[j + 2:j + 2 + n]
        i = j + 2 + n + 2


async def raw(port, method, target, headers=(), body=b'', timeout=15):
    r, w = await asyncio.open_connection('127.0.0.1', port)
    lines = [f'{method} {target} HTTP/1.1', f'Host: 127.0.0.1:{port}', 'Connection: close']
    lines += [f'{k}: {v}' for k, v in headers]
    if body or method in ('POST', 'PUT', 'DELETE', 'PATCH'):
        lines.append(f'Content-Length: {len(body)}')
    w.write(('\r\n'.join(lines) + '\r\n\r\n').encode('latin-1') + body)
    await w.drain()
    data = await asyncio.wait_for(r.read(), timeout)
    w.close()
    head, _, rest = data.partition(b'\r\n\r\n')
    status_line, *hlines = head.decode('latin-1').split('\r\n')
    hdrs = [tuple(x.strip() for x in line.split(':', 1)) for line in hlines if ':' in line]
    resp = Resp(int(status_line.split()[1]), hdrs, rest)
    if (resp.header('transfer-encoding') or '').lower() == 'chunked' and method != 'HEAD' and rest:
        resp.body = dechunk(rest)
    return resp


class AB:
    def __init__(self, ports, ros, out):
        self.ports, self.ros, self.out = ports, ros, out
        self.cookies = {}
        self.results = []          # (category, label, equal, detail)
        self.ws_log = {s: [] for s in SIDES}
        self.ws_tasks = []

    def record(self, category, label, equal, detail=None):
        self.results.append((category, label, bool(equal), detail))
        if not equal:
            print(f'  DIFF [{category}] {label}: {detail}')

    async def both(self, method, target, headers=(), body=b'', auth=True, csrf=False, extra_cookie=None):
        async def one(side):
            h = list(headers)
            if auth and side in self.cookies:
                h.append(('Cookie', f'roomba_session={self.cookies[side]}'))
            if extra_cookie:
                h.append(('Cookie', extra_cookie))
            if csrf:
                h.append(('X-Requested-With', 'roomba-web'))
            return await raw(self.ports[side], method, target, h, body)
        return await asyncio.gather(one('py'), one('rs'))

    async def compare(self, category, method, target, headers=(), body=b'', auth=True, csrf=False, show=200):
        rp, rr = await self.both(method, target, headers, body, auth, csrf)
        kp, kr = rp.key(True), rr.key(True)
        detail = None
        if kp != kr:
            detail = {'py': (kp[0], kp[1], kp[2][:show]), 'rs': (kr[0], kr[1], kr[2][:show])}
        self.record(category, f'{method} {target} auth={auth} csrf={csrf}', kp == kr, detail)
        return rp, rr

    async def login(self, side, password):
        r = await raw(self.ports[side], 'POST', '/login', [('Content-Type', 'application/x-www-form-urlencoded')],
                      f'password={password}'.encode())
        m = re.search(r'roomba_session=([A-Za-z0-9_-]+)', r.header('set-cookie') or '')
        return r, (m.group(1) if m else None)

    # ---- websocket recording
    async def ws_recorder(self, side):
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(f'http://127.0.0.1:{self.ports[side]}/ws',
                                    headers={'Cookie': f'roomba_session={self.cookies[side]}'}) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self.ws_log[side].append((time.monotonic(), msg.data))

    async def checkpoint(self, name, settle=0.8, window=1.0, apis=True):
        await asyncio.sleep(settle)
        t0 = time.monotonic()
        await asyncio.sleep(window)
        t1 = time.monotonic()
        msgs = {s: [m for t, m in self.ws_log[s] if t0 <= t <= t1] for s in SIDES}
        distinct = {s: sorted(set(msgs[s])) for s in SIDES}
        steady = all(len(distinct[s]) == 1 for s in SIDES)
        equal = steady and distinct['py'] == distinct['rs']
        detail = None
        if not equal:
            detail = {s: [json.loads(m) for m in distinct[s]][:3] for s in SIDES}
            if all(distinct[s] for s in SIDES):
                a, b = json.loads(distinct['py'][-1]), json.loads(distinct['rs'][-1])
                detail = {k: (a.get(k), b.get(k)) for k in a if a.get(k) != b.get(k)}
        self.record('ws', f'{name} ({len(msgs["py"])}+{len(msgs["rs"])} msgs)', equal, detail)
        self.ws_compared = getattr(self, 'ws_compared', 0) + len(msgs['py']) + len(msgs['rs'])
        if apis:
            for target in ('/api/state', '/api/map.json', '/api/map.png', '/api/path', '/api/zones', '/api/plan',
                           '/api/obstacles'):
                await self.compare(f'api:{name}', 'GET', target, show=400)
        return json.loads(msgs['py'][-1]) if msgs['py'] else None

    def outputs_since(self, t0):
        with self.ros.lock:
            return {s: [(t, topic, data) for t, topic, data in self.ros.outputs[s] if t >= t0] for s in SIDES}

    def compare_outputs(self, label, t0, topics=None, timing=False):
        outs = self.outputs_since(t0)
        seq = {s: [(topic, data) for _, topic, data in outs[s] if topics is None or topic in topics] for s in SIDES}
        equal = seq['py'] == seq['rs']
        detail = None if equal else {s: seq[s][:12] for s in SIDES}
        self.record('commands', f'{label} ({len(seq["py"])}+{len(seq["rs"])} msgs)', equal, detail)
        return outs


# ---------------------------------------------------------------- the test script

ROUTES = ['/', '/login', '/logout', '/favicon.ico', '/stream.mjpg', '/ws', '/api/state', '/api/map.json',
          '/api/map.png', '/api/path', '/api/zones', '/api/zones/x', '/api/zones/', '/api/zones/a/b', '/api/goal',
          '/api/goal/cancel', '/api/plan', '/api/obstacles', '/api/drive', '/api/robot/mode', '/api/robot/stop',
          '/api/explore', '/api/floor_grid', '/api/nonexistent', '/nonexistent', '/static', '/static/',
          '/static/app.js', '/static/index.html', '/static/manifest.json', '/static/icon-192.png', '/static/style.css',
          '/static/../../roomba_web/package.xml', '/static/%2e%2e/%2e%2e/roomba_web/package.xml',
          '/static/..%2f..%2fpackage.xml', '/static/..%2F..%2F..%2F..%2F..%2F..%2Fetc%2Fpasswd',
          '/static//etc/passwd', '/static/%2Fetc%2Fpasswd', '/static/missing.js', '/static/app.js/',
          '/static/app.js%00', '/static/./app.js', '/static/x/../app.js', '/static/app.js/..', '/%61pi/state',
          '/%73tatic/app.js', '/api/%73tate', '//api/state', '/api//state', '/API/state', '/static/%ZZ',
          '/static/%FF', '/static/%C3%A9', '/login/', '/ws/', '/stream.mjpg?x=1', '/api/state?x=1',
          '/static/app.js?v=1', '/%6Cogin', '/static%2Fapp.js', '/api/zones/%2F', '/api/zones/a%2Fb']
METHODS = ['GET', 'HEAD', 'POST', 'DELETE', 'PUT', 'PATCH', 'OPTIONS']


async def security(ab):
    print('== auth / routing matrix')
    for target in ROUTES:
        for method in METHODS:
            if target.startswith('/login') and method == 'POST':
                continue   # login attempts are tested separately (1 s each when wrong)
            if target in ('/stream.mjpg', '/stream.mjpg?x=1') and method in ('GET', 'HEAD'):
                await ab.compare('auth', method, target, auth=False)
                continue
            await ab.compare('auth', method, target, auth=False)
            await ab.compare('auth', method, target, auth=False, headers=[('Cookie', 'roomba_session=forged-token')])
            if target == '/logout' or (target.startswith('/stream.mjpg') and method == 'GET'):
                continue
            if method in ('POST', 'DELETE', 'PUT'):
                await ab.compare('auth', method, target, csrf=False)
            if target.startswith('/ws') and method == 'GET':
                await ab.compare('auth', method, target)
                continue
            if method in ('POST', 'DELETE', 'PUT', 'PATCH', 'OPTIONS') or method in ('GET', 'HEAD'):
                await ab.compare('auth', method, target, csrf=True)
    # nothing outside www may ever be served
    for side in SIDES:
        for target in ROUTES:
            if 'passwd' in target or 'package.xml' in target:
                r = await raw(ab.ports[side], 'GET', target)
                ab.record('traversal', f'{side} {target} -> {r.status}', r.status in (403, 404) and b'root:' not in r.body)
    # websocket upgrade without a session, with a forged one, and malformed handshakes
    for side in SIDES:
        async with aiohttp.ClientSession() as s:
            for cookie in (None, 'roomba_session=forged'):
                try:
                    async with s.ws_connect(f'http://127.0.0.1:{ab.ports[side]}/ws',
                                            headers={'Cookie': cookie} if cookie else {}):
                        ok = False
                except aiohttp.WSServerHandshakeError as e:
                    ok = e.status == 401
                ab.record('auth', f'{side} websocket cookie={cookie} rejected with 401', ok)
    ws_headers = [('Upgrade', 'websocket'), ('Connection', 'Upgrade'), ('Sec-WebSocket-Version', '13')]
    for hdrs in ([], [('Upgrade', 'websocket')], ws_headers, ws_headers + [('Sec-WebSocket-Key', 'short')],
                 [('Upgrade', 'websocket'), ('Connection', 'Upgrade'), ('Sec-WebSocket-Version', '12'),
                  ('Sec-WebSocket-Key', 'dGhlIHNhbXBsZSBub25jZQ==')]):
        await ab.compare('ws-handshake', 'GET', '/ws', headers=hdrs)
    # cookie parsing variants
    for side in SIDES:
        tok = ab.cookies[side]
        for cookie, want in ((f'a=1; roomba_session={tok}; b=2', 200), (f'roomba_session="{tok}"', 200),
                             (f'roomba_session={tok}x', 401), (f'roomba_session={tok[:-1]}', 401),
                             (f'roomba_session={tok}; roomba_session=bad', 401)):
            r = await raw(ab.ports[side], 'GET', '/api/state', [('Cookie', cookie)])
            ab.record('cookies', f'{side} {cookie.replace(tok, "<tok>")} -> {r.status}', r.status == want)


async def static_files(ab):
    print('== static files: conditional and range requests')
    rp, rr = await ab.compare('static', 'GET', '/static/app.js')
    etag, lm = rp.header('etag'), rp.header('last-modified')
    for hdrs in ([('If-None-Match', etag)], [('If-None-Match', f'W/{etag}')], [('If-None-Match', '"x", ' + etag)],
                 [('If-None-Match', '*')], [('If-Modified-Since', lm)], [('If-Modified-Since', 'Thu, 01 Jan 1970 00:00:00 GMT')],
                 [('If-Match', '"nope"')], [('If-Match', etag)], [('If-Unmodified-Since', 'Thu, 01 Jan 1970 00:00:00 GMT')],
                 [('Range', 'bytes=0-99')], [('Range', 'bytes=-50')], [('Range', 'bytes=28000-')],
                 [('Range', 'bytes=100000-')], [('Range', 'bytes=5-2')], [('Range', 'items=0-1')],
                 [('Range', 'bytes=0-9'), ('If-Range', lm)], [('Range', 'bytes=0-9'), ('If-Range', 'Thu, 01 Jan 1970 00:00:00 GMT')],
                 [('Accept-Encoding', 'gzip, deflate')]):
        await ab.compare('static', 'GET', '/static/app.js', headers=hdrs)
        await ab.compare('static', 'HEAD', '/static/app.js', headers=hdrs)
    for page in ('/', '/login'):
        rp, _ = await ab.compare('static', 'GET', page)
        await ab.compare('static', 'GET', page, headers=[('If-None-Match', rp.header('etag') or '')])


async def logins(ab):
    print('== login / logout')
    form = [('Content-Type', 'application/x-www-form-urlencoded')]
    t0 = time.monotonic()
    rp, rr = await ab.compare('login', 'POST', '/login', form, b'password=wrong', auth=False)
    dt = time.monotonic() - t0
    ab.record('login', f'wrong password delayed >= 1 s (both answered after {dt:.2f} s)', dt >= 1.0)
    pw = os.environ['AB_PASSWORD']
    for body, hdrs in ((f'password={pw}', form), (f'password={pw}&password=x', form), (f'x=1&password={pw}', form),
                       (f'password={pw}', [('Content-Type', 'application/json')]), (f'password={pw}', []),
                       ('', form), (f'password={pw}', [('Content-Type', 'application/x-www-form-urlencoded; charset=utf-8')])):
        rp, rr = await ab.compare('login', 'POST', '/login', hdrs, body.encode(), auth=False)
    boundary = 'XyZ123'
    mp = (f'--{boundary}\r\nContent-Disposition: form-data; name="password"\r\n\r\n{pw}\r\n--{boundary}--\r\n').encode()
    rp, rr = await ab.compare('login', 'POST', '/login', [('Content-Type', f'multipart/form-data; boundary={boundary}')], mp, auth=False)
    cookies = {}
    for side, r in zip(SIDES, (rp, rr)):
        m = re.search(r'roomba_session=([A-Za-z0-9_-]+)', r.header('set-cookie') or '')
        cookies[side] = m.group(1) if m else None
    ab.record('login', 'multipart login gives a session', all(cookies.values()))
    # logout revokes the session (and only that one)
    for side in SIDES:
        r = await raw(ab.ports[side], 'POST', '/logout', [('Cookie', f'roomba_session={cookies[side]}')])
        after = await raw(ab.ports[side], 'GET', '/api/state', [('Cookie', f'roomba_session={cookies[side]}')])
        still = await raw(ab.ports[side], 'GET', '/api/state', [('Cookie', f'roomba_session={ab.cookies[side]}')])
        ab.record('login', f'{side} logout: {r.status} {r.header("location")} {r.header("set-cookie")}; revoked {after.status}, other {still.status}',
                  r.status == 302 and after.status == 401 and still.status == 200)
    a = await raw(ab.ports['py'], 'POST', '/logout', [('Cookie', 'roomba_session=x')])
    b = await raw(ab.ports['rs'], 'POST', '/logout', [('Cookie', 'roomba_session=x')])
    ab.record('login', 'logout without a session', a.key() == b.key(), (a.key(), b.key()))


async def streams(ab, ros, frames):
    print('== camera: preview frames and annotated passthrough')
    got = {s: [] for s in SIDES}

    async def reader(side, session, stop):
        async with session.get(f'http://127.0.0.1:{ab.ports[side]}/stream.mjpg',
                               headers={'Cookie': f'roomba_session={ab.cookies[side]}'}) as resp:
            got[side + '_headers'] = (resp.status, resp.headers.get('Content-Type'), resp.headers.get('Cache-Control'))
            while not stop.is_set():
                line = await resp.content.readline()
                if not line:
                    break
                if line.strip() != b'--frame':
                    continue
                hdrs = {}
                while (h := await resp.content.readline()) not in (b'\r\n', b''):
                    k, v = h.decode().split(':', 1)
                    hdrs[k.strip().lower()] = v.strip()
                data = await resp.content.readexactly(int(hdrs['content-length']))
                got[side].append(data)

    stop = asyncio.Event()
    async with aiohttp.ClientSession() as s:
        tasks = [asyncio.create_task(reader(side, s, stop)) for side in SIDES]
        await asyncio.sleep(3.0)   # both have subscribed to the preview topic (2 s check)
        for img in frames:
            img.header.stamp = ros.stamp()
            ros.pub['image'].publish(img)
            await asyncio.sleep(0.5)
        await asyncio.sleep(1.0)
        n_preview = {side: len(got[side]) for side in SIDES}
        jpegs = [bytes(range(i, i + 50)) * 20 for i in range(15)]
        for j in jpegs:
            m = CompressedImage()
            m.format, m.data = 'jpeg', j
            ros.pub['jpeg'].publish(m)
            await asyncio.sleep(0.2)
        await asyncio.sleep(0.5)
        # 4 viewers max: two more each (with the recorder) fill the slots, then 503
        extra = []
        for side in SIDES:
            for _ in range(3):
                extra.append(await s.get(f'http://127.0.0.1:{ab.ports[side]}/stream.mjpg',
                                         headers={'Cookie': f'roomba_session={ab.cookies[side]}'}))
        over = {}
        for side in SIDES:
            r = await raw(ab.ports[side], 'GET', '/stream.mjpg', [('Cookie', f'roomba_session={ab.cookies[side]}')], timeout=5)
            over[side] = (r.status, r.body)
        ab.record('stream', f'5th viewer refused: {over}', over['py'] == over['rs'] == (503, b'too many camera viewers'))
        for r in extra:
            r.close()
        stop.set()
        for t in tasks:
            t.cancel()
    ab.record('stream', f'stream headers {got["py_headers"]} / {got["rs_headers"]}', got['py_headers'] == got['rs_headers'])
    pv = {side: got[side][:n_preview[side]] for side in SIDES}
    an = {side: got[side][n_preview[side]:] for side in SIDES}
    same = sum(a == b for a, b in zip(pv['py'], pv['rs']))
    ab.record('stream', f'preview JPEG frames: py {len(pv["py"])}, rs {len(pv["rs"])}, byte-identical {same}',
              pv['py'] == pv['rs'] and len(pv['py']) == len(frames))
    ab.record('stream', f'annotated frames passed through: py {len(an["py"])}, rs {len(an["rs"])}',
              an['py'] == an['rs'] == jpegs, None)
    ab.preview_frames = {'sent': len(frames), 'py': len(pv['py']), 'rs': len(pv['rs']), 'identical': same}


async def heartbeat(ab):
    print('== websocket heartbeat (client that never answers pings)')
    async def one(side):
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(f'http://127.0.0.1:{ab.ports[side]}/ws', autoping=False,
                                    headers={'Cookie': f'roomba_session={ab.cookies[side]}'}) as ws:
                t0, pings, n = time.monotonic(), 0, 0
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.PING:
                        pings += 1
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        n += 1
                    if time.monotonic() - t0 > 30:
                        break
                return time.monotonic() - t0, pings, n
    (tp, pp, np_), (tr, pr, nr) = await asyncio.gather(one('py'), one('rs'))
    ab.record('ws', f'unanswered ping closes the socket: py after {tp:.1f} s ({pp} ping, {np_} msgs), rs after {tr:.1f} s ({pr} ping, {nr} msgs)',
              14 < tp < 17 and 14 < tr < 17 and pp == pr == 1 and abs(np_ - nr) <= 10)


async def scenario(ab, ros, frames):
    j = json.dumps
    csrf = [('Content-Type', 'application/json')]

    print('== state: empty')
    await ab.checkpoint('empty')

    print('== state: tracking, floor, filter, TF pose')
    st = VisualSlamStatus()
    st.vo_state = 1
    ros.live['status'] = st
    ros.live['floor'] = String(data=j({'camera_height_m': 0.2534, 'tilt_deg': 12.5, 'floor_visible': True,
                                       'fixed_height': False, 'note': 'caf\u00e9 \u2191', 'n': 10 ** 20, 'x': 1e-7}))
    ros.live['filter'] = String(data=j({'state': 'idle', 'filtered': 3, 'jumps': [0.5, 1e16], 'nan': float('nan')}))
    await ab.checkpoint('tracking ok')
    st2 = VisualSlamStatus()
    st2.vo_state = 2
    ros.live['status'] = st2
    await ab.checkpoint('tracking lost')
    ros.live.pop('status')
    await asyncio.sleep(2.2)
    await ab.checkpoint('tracking no data')

    print('== maps')
    rng = np.random.default_rng(3)
    m1 = rng.choice([-1, 0, 0, 0, 30, 64, 65, 100], size=(300, 420)).astype(np.int8)
    ros.pub['map_filled'].publish(grid('map', m1, yaw=0.3))
    await ab.checkpoint('map filled v1')
    m2 = rng.choice([-1, 0, 100], size=(310, 400)).astype(np.int8)
    ros.pub['map'].publish(grid('map', m2))
    await ab.checkpoint('raw map ignored while the filled one is fresh')
    t_filled = time.monotonic()
    m3 = rng.choice([-128, -1, 0, 50, 99, 127], size=(123, 77)).astype(np.int8)
    ros.pub['map_filled'].publish(grid('map', m3, res=0.1, origin=(1.25, -7.5)))
    await ab.checkpoint('map filled v2')
    await asyncio.sleep(max(0.0, 10.3 - (time.monotonic() - t_filled)))
    ros.pub['map'].publish(grid('map', m2))
    await ab.checkpoint('raw map after 10 s')

    print('== paths, plan, obstacles')
    pts = [(math.cos(i / 100) * i / 300, math.sin(i / 100) * i / 300) for i in range(1501)]
    ros.pub['path'].publish(path('map', pts))
    ros.pub['plan'].publish(path('map', pts[:777]))
    lethal = (rng.random((200, 160)) < 0.2) * 100
    lethal[rng.random((200, 160)) < 0.05] = 99
    ros.pub['costmap'].publish(grid('odom', lethal.astype(np.int8), origin=(-4.0, -5.0)))
    await ab.checkpoint('paths and obstacles')
    small = np.zeros((40, 40), np.int8)
    small[10:12, 5:30] = 100
    ros.pub['costmap'].publish(grid('base_link', small, origin=(-1.0, -1.0)))
    ros.pub['path'].publish(path('map', pts[:10]))
    ros.pub['plan'].publish(path('map', []))
    await ab.checkpoint('short paths, costmap in base_link')
    ros.pub['costmap'].publish(grid('nowhere', small))
    await ab.checkpoint('costmap without TF keeps the old obstacles')

    print('== zones')
    zones = {'frame': 'map', 'zones': [
        {'name': 'sofa', 'kind': 'exclusion', 'points': [[0.0, 0.0], [1.5, 0.0], [1.5, 1.25]]},
        {'name': 'k\u00fcche \u2192', 'kind': 'keepout', 'points': [[2, 2], [3, 2], [3, 3.5]], 'expires': 1e10},
        {'name': '7', 'kind': 'exclusion', 'points': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}]}
    ros.pub['zones'].publish(String(data=j(zones)))
    await ab.checkpoint('zones')

    print('== detections')
    ros.live['dets'] = detections('camera_left_optical_frame', [
        ('person', 0.8765, 0.4, -0.1, 2.3456, ''), ('cat', 0.5, -0.3, 0.2, 1.111, ''), (None, 0.0, 0.0, 0.0, 0.333, ''),
        ('dog', 0.99, 0.0, 0.1, 2.3456, 'x'), ('person', 0.4, 0.1, 0.0, float('nan'), '')])
    ros.live['ignored'] = detections('camera_left_optical_frame', [('person', 0.7, 0.0, 0.0, 1.9, 'zone sofa')])
    await ab.checkpoint('detections')
    ros.live['dets'] = detections('no_such_frame', [('person', 0.6, 0.1, 0.2, 1.5, '')])
    await ab.checkpoint('detections without TF')
    ros.live.pop('dets')
    ros.live.pop('ignored')
    await ab.checkpoint('detections expired')
    ros.tf_pose = (-2.5, 4.25, -2.0)
    await ab.checkpoint('pose moved')

    print('== robot status')
    bat = BatteryState()
    bat.percentage, bat.voltage, bat.power_supply_status = 0.4567, 14.876, 1
    ros.live['battery'] = bat
    sens = SensorState()
    sens.bump_right, sens.cliff_front_left, sens.cliff_right, sens.home_base_available, sens.oi_mode = True, True, True, True, 2
    ros.live['sensors'] = sens
    await ab.checkpoint('robot safe with hazards')
    bat2 = BatteryState()
    bat2.percentage, bat2.voltage, bat2.power_supply_status = float('nan'), 13.0, 2
    ros.live['battery'] = bat2
    ros.live.pop('sensors')
    ros.pub['mode'].publish(String(data='passive'))
    await ab.checkpoint('battery unknown, mode from /roomba/mode')
    await asyncio.sleep(3.0)
    await ab.checkpoint('mode stale -> unknown')
    ros.live['sensors'] = sens

    print('== localization pose, instructions, explorer status')
    ros.pub['loc'].publish(PoseWithCovarianceStamped())
    ros.live['instruction'] = String(data=j({'code': 'turn_left', 'arrow': '\u21b0', 'text': 'Turn left \u2013 now',
                                             'active': True, 'distance_m': 2.345}))
    ros.pub['explore'].publish(String(data=j({'state': 'idle', 'message': 'Ready', 'reached': 0, 'failed': 0})))
    await ab.checkpoint('instruction (no explorer subscriber)')
    ros.explorer_listeners(True)
    await ab.checkpoint('explorer subscribed', settle=1.5)

    print('== commands without servers')
    for body in (j({'x': 1, 'y': 2}), j({'x': 'nan', 'y': 0})):
        await ab.compare('goal', 'POST', '/api/goal', csrf, body.encode(), csrf=True)
    await ab.compare('mode', 'POST', '/api/robot/mode', csrf, j({'mode': 'safe'}).encode(), csrf=True)

    print('== zone commands')
    t0 = time.monotonic()
    bodies = [{'name': 'new zone', 'points': [[0, 0], [1, 0], [1, 1]]},
              {'name': 'sofa', 'points': [[0, 0], [1, 0], [1, 1]]},
              {'name': 7, 'points': [[0, 0], [1, 0], [1, 1]]},
              {'name': 7.0, 'points': [[0, 0], [1, 0], [1, 1]], 'kind': 'keepout'},
              {'points': [[0, '1.5'], [True, 0], [' 2_0 ', 'inf']], 'kind': ''},
              {'name': '', 'points': {'12': 1, '34': 2, '56': 3}},
              {'name': ['a'], 'points': [[0, 0, 9], [1, 0], [1, 1]], 'kind': 'keepout'},
              {'name': 'x', 'points': [[0, 0], [1, 0]]},
              {'name': 'x', 'points': [[i, i] for i in range(65)]},
              {'name': 'x', 'points': [[0, 0], [1, 0], [1]]},
              {'name': 'x', 'points': [[0, 0], [1, 0], 'ab']},
              {'name': 'x', 'points': [[0, 0], [1, 0], [1, 1]], 'kind': 'wall'},
              {'name': 'x', 'points': None}, {'name': 'x'}, [1, 2], 'str', None]
    for body in bodies:
        await ab.compare('zones', 'POST', '/api/zones', csrf, j(body).encode(), csrf=True)
    for raw_body in (b'', b'{', b'{"points": [[1e999, 0], [1, 0], [1, 1]]}', b'{"points": [[NaN, 0], [1, 0], [1, 1]]}',
                     b'{"points": [[' + b'9' * 400 + b', 0], [1, 0], [1, 1]]}', b'\xff\xfe', b'{"points": [[0,0],[1,0],[1,1]], "name": "\xc3\xa9"}'):
        await ab.compare('zones', 'POST', '/api/zones', csrf, raw_body, csrf=True)
    await ab.compare('zones', 'POST', '/api/zones', [('Content-Type', 'application/json; charset=latin-1')],
                     '{"points": [[0,0],[1,0],[1,1]], "name": "\u00e9"}'.encode('latin-1'), csrf=True)
    await ab.compare('zones', 'POST', '/api/zones', [('Content-Type', 'application/json; charset=klingon')], b'{}', csrf=True)
    for name in ('sofa', 'k%C3%BCche%20%E2%86%92', 'a%2Fb', 'a+b', '%ZZ', '%FF', 'x%00y'):
        await ab.compare('zones', 'DELETE', f'/api/zones/{name}', csrf=True)
    await asyncio.sleep(0.5)
    ab.compare_outputs('zone commands', t0, {'zone_command'})

    print('== floor grid')
    t0 = time.monotonic()
    for body in (j({'on': False}), j({'on': 'yes'}), j({'on': 0}), j({'on': []}), j({}), j([True])):
        await ab.compare('floor_grid', 'POST', '/api/floor_grid', csrf, body.encode(), csrf=True)
    await asyncio.sleep(0.5)
    ab.compare_outputs('floor grid', t0, {'floor_grid'})
    await ab.checkpoint('floor grid')

    print('== navigation')
    ros.nav_servers(True)
    ros.mode_services()
    await asyncio.sleep(1.5)
    await ab.checkpoint('nav available')
    t0 = time.monotonic()
    for body in (j({'x': 1.5, 'y': -2.25}), j({'x': '3', 'y': 4}), j({'x': 1001, 'y': 0}), j({'x': float('inf'), 'y': 0}),
                 j({'x': 1}), j({'x': None, 'y': 1}), j([1, 2])):
        await ab.compare('goal', 'POST', '/api/goal', csrf, body.encode(), csrf=True)
        await asyncio.sleep(0.3)
    await ab.checkpoint('navigating')
    ros.live['instruction'] = String(data=j({'code': 'forward', 'arrow': '\u2191', 'text': 'Go', 'active': True,
                                             'distance_m': 1.0}))
    await ab.compare('goal', 'POST', '/api/goal/cancel', csrf=True)
    await ab.checkpoint('goal canceled')
    for side in SIDES:
        ros.nav[side].accept = False
    await ab.compare('goal', 'POST', '/api/goal', csrf, j({'x': 0.5, 'y': 0.5}).encode(), csrf=True)
    await ab.checkpoint('goal rejected')
    for side in SIDES:
        ros.nav[side].accept = True
    await ab.compare('goal', 'POST', '/api/goal', csrf, j({'x': -1, 'y': 1}).encode(), csrf=True)
    await asyncio.sleep(0.5)
    ros.live['instruction'] = String(data=j({'code': 'arrived', 'arrow': '\u2713', 'text': 'Arrived', 'active': False}))
    await ab.checkpoint('arrived clears the goal')
    goals = {s: [(g['frame'], g['x'], g['y'], g['qw'], g['accepted']) for g in ros.nav[s].goals] for s in SIDES}
    ab.record('commands', f'goals sent: {goals["py"]}', goals['py'] == goals['rs'], goals)
    cancels = {s: ros.nav[s].cancels for s in SIDES}
    ab.record('commands', f'cancel requests: {cancels}', cancels['py'] == cancels['rs'])

    print('== explorer commands')
    t0 = time.monotonic()
    for body in (j({'action': 'start'}), j({'action': 'home'}), j({'action': 'stop'}), j({'action': 'jump'}),
                 j({'action': ['start']}), j({})):
        await ab.compare('explore', 'POST', '/api/explore', csrf, body.encode(), csrf=True)
    await asyncio.sleep(0.5)
    ab.compare_outputs('explorer commands', t0, {'explorer_command'})

    print('== set_mode')
    for body, reply in ((j({'mode': 'safe'}), (True, 'Safe mode')), (j({'mode': 'passive'}), (False, 'cliff detected')),
                        (j({'mode': 'full'}), None), (j({'mode': 2}), None)):
        if reply:
            ros.mode_reply = reply
        await ab.compare('mode', 'POST', '/api/robot/mode', csrf, body.encode(), csrf=True)
        await ab.checkpoint(f'mode request {body}')
    ab.record('commands', f'set_mode requests: {ros.mode_requests}', ros.mode_requests['py'] == ros.mode_requests['rs'])

    print('== drive pad')
    await drive(ab, ros)

    print('== camera')
    await streams(ab, ros, frames)


async def drive(ab, ros):
    j = json.dumps
    hdr = [('Content-Type', 'application/json')]

    async def hold(v, w, seconds, rate=0.1):
        t_end = time.monotonic() + seconds
        last = None
        while time.monotonic() < t_end:
            await ab.both('POST', '/api/drive', hdr, j({'v': v, 'w': w}).encode(), csrf=True)
            last = time.monotonic()
            await asyncio.sleep(rate)
        return last

    def analyse(label, t0, t_last, expect_topic, expect_value):
        outs = ab.outputs_since(t0)
        summary = {}
        for side in SIDES:
            tw = [(t, topic, d) for t, topic, d in outs[side] if topic in ('cmd_vel', 'cmd_vel_nav')]
            moving = [x for x in tw if x[2] != (0.0, 0.0)]
            zeros = [x for x in tw if x[2] == (0.0, 0.0)]
            stop_latency = (zeros[0][0] - t_last) if zeros else None
            summary[side] = {'moving': len(moving), 'zeros': len(zeros), 'topics': sorted({x[1] for x in tw}),
                             'values': sorted({x[2] for x in moving}), 'stop_after_s': round(stop_latency, 3) if stop_latency else None}
        ok = all(summary[s]['zeros'] == 1 and summary[s]['topics'] == [expect_topic] and summary[s]['values'] == [expect_value]
                 and 0.3 < (summary[s]['stop_after_s'] or 0) < 0.6 for s in SIDES)
        ok = ok and abs(summary['py']['moving'] - summary['rs']['moving']) <= 1
        ab.record('drive', f'{label}: {summary}', ok)
        return summary

    ab.drive_results = {}
    # without the navigation server: /cmd_vel; clamped to max speed / turn
    ros.nav_servers(False)
    await asyncio.sleep(1.5)
    await ab.compare('drive', 'POST', '/api/drive', hdr, b'{"v": "x", "w": 0}', csrf=True)
    await ab.compare('drive', 'POST', '/api/drive', hdr, b'{"v": NaN, "w": 0}', csrf=True)
    await ab.compare('drive', 'POST', '/api/drive', hdr, b'{"v": 1}', csrf=True)
    t0 = time.monotonic()
    t_last = await hold(5.0, -3.0, 2.0)
    await asyncio.sleep(1.0)
    ab.drive_results['clamped, released'] = analyse('drive clamped then released (cmd_vel)', t0, t_last, 'cmd_vel', (0.2, -1.0))
    # page closed mid-drive (requests just stop): same deadman
    t0 = time.monotonic()
    t_last = await hold(0.1, 0.5, 1.0, rate=0.05)
    await asyncio.sleep(1.0)
    ab.drive_results['requests stop'] = analyse('drive then silence', t0, t_last, 'cmd_vel', (0.1, 0.5))
    # stop button: immediate zero, the hold is dropped
    t0 = time.monotonic()
    await hold(-0.15, 0.0, 0.5)
    await ab.compare('drive', 'POST', '/api/robot/stop', csrf=True)
    t_stop = time.monotonic()
    await asyncio.sleep(1.0)
    outs = ab.outputs_since(t0)
    res = {}
    for side in SIDES:
        tw = [(t, d) for t, topic, d in outs[side] if topic in ('cmd_vel', 'cmd_vel_nav')]
        zeros = [t for t, d in tw if d == (0.0, 0.0)]
        res[side] = (len(tw) - len(zeros), len(zeros), round(zeros[0] - t_stop, 3) if zeros else None)
    ab.record('drive', f'stop button (moving, zeros, s after stop): {res}',
              all(res[s][1] >= 1 and res[s][2] is not None and res[s][2] < 0.25 for s in SIDES))
    ab.drive_results['stop button'] = res
    # with navigation running: /cmd_vel_nav; manual driving cancels the goal and stops exploring
    ros.live.pop('instruction', None)   # an 'arrived' instruction would clear the goal by itself
    ros.nav_servers(True)
    ros.pub['explore'].publish(String(data=j({'state': 'exploring', 'message': 'Exploring', 'reached': 2, 'failed': 1})))
    await asyncio.sleep(1.5)
    await ab.compare('goal', 'POST', '/api/goal', hdr, j({'x': 2, 'y': 3}).encode(), csrf=True)
    await asyncio.sleep(0.5)
    cancels0 = {s: ros.nav[s].cancels for s in SIDES}
    t0 = time.monotonic()
    t_last = await hold(0.05, 0.0, 1.0)
    await asyncio.sleep(1.0)
    ab.drive_results['navigating'] = analyse('drive while navigating (cmd_vel_nav)', t0, t_last, 'cmd_vel_nav', (0.05, 0.0))
    cancels = {s: ros.nav[s].cancels - cancels0[s] for s in SIDES}
    outs = ab.outputs_since(t0)
    stops = {s: [d for _, topic, d in outs[s] if topic == 'explorer_command'] for s in SIDES}
    ab.record('drive', f'manual drive canceled the goal {cancels} and stopped exploring {stops}',
              cancels['py'] == cancels['rs'] == 1 and stops['py'] == stops['rs'] == ['stop'])
    await ab.checkpoint('after manual drive')
    # robot driver gone: drive refused
    ros.live.pop('battery')
    await asyncio.sleep(3.2)
    await ab.compare('drive', 'POST', '/api/drive', hdr, j({'v': 0.1, 'w': 0}).encode(), csrf=True)
    await ab.checkpoint('robot gone')


def bag_frames(bag, n):
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='mcap'), ConverterOptions('', ''))
    r.set_filter(StorageFilter(topics=['/camera/left/image_rect_mono']))
    out, seen = [], 0
    while r.has_next() and len(out) < n:
        _, data, _ = r.read_next()
        seen += 1
        if seen % 15 == 0:
            out.append(deserialize_message(data, Image))
    return out


async def run(args, ros):
    ports = {'py': args.py_port, 'rs': args.rs_port}
    ab = AB(ports, ros, args.out)
    pw = os.environ['AB_PASSWORD']
    for side in SIDES:
        r, tok = await ab.login(side, pw)
        assert tok, f'{side} login failed: {r.status} {r.headers}'
        ab.cookies[side] = tok
    print(f'logged in (py set-cookie: {re.sub("=[A-Za-z0-9_-]{43}", "=<token>", r.header("set-cookie"))})')
    for side in SIDES:
        ab.ws_tasks.append(asyncio.create_task(ab.ws_recorder(side)))
    frames = bag_frames(args.bag, 20)
    await security(ab)
    await static_files(ab)
    await logins(ab)
    await scenario(ab, ros, frames)
    await heartbeat(ab)
    for t in ab.ws_tasks:
        t.cancel()
    # the whole websocket stream: every distinct state one server sent, did the other send it too?
    distinct = {s: list(dict.fromkeys(m for _, m in ab.ws_log[s])) for s in SIDES}
    only = {s: [m for m in distinct[s] if m not in set(distinct[o])] for s, o in (('py', 'rs'), ('rs', 'py'))}
    stream = {'messages': {s: len(ab.ws_log[s]) for s in SIDES}, 'distinct_states': {s: len(distinct[s]) for s in SIDES},
              'only_in': {s: len(only[s]) for s in SIDES}}
    return ab, stream, only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--py-port', type=int, default=18081)
    ap.add_argument('--rs-port', type=int, default=18082)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rclpy.init()
    ros = Harness()
    ex = MultiThreadedExecutor(num_threads=8)
    ex.add_node(ros)
    threading.Thread(target=ex.spin, daemon=True).start()
    time.sleep(1.0)
    try:
        ab, stream, only = asyncio.run(run(args, ros))
    finally:
        for nav in ros.nav.values():
            nav.finish = True
    res = ab.results
    cats = {}
    for cat, _, eq, _ in res:
        c = cat.split(':')[0]
        cats.setdefault(c, [0, 0])
        cats[c][0 if eq else 1] += 1
    print('\n== summary (equal, different)')
    for c, (e, d) in cats.items():
        print(f'  {c:14s} {e:5d} {d:4d}')
    print(f'  websocket messages compared at checkpoints: {ab.ws_compared}')
    print(f'  websocket stream: {stream}')
    for s in SIDES:
        for m in only[s][:5]:
            print(f'    only in {s}: {m[:300]}')
    with open(os.path.join(args.out, 'ab_web.json'), 'w') as f:
        json.dump({'results': [(c, lbl, eq, repr(d)[:2000]) for c, lbl, eq, d in res], 'categories': cats,
                   'stream': stream, 'only_in': {s: only[s][:50] for s in SIDES},
                   'drive': getattr(ab, 'drive_results', None), 'preview': getattr(ab, 'preview_frames', None)}, f, indent=1)
    ex.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
