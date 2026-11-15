#!/usr/bin/env python3
"""Roomba web app: live camera with detections, top-down map, exclusion-zone editor.

One process: a ROS 2 node (spun in a background thread) collects state, and an aiohttp
server (main thread) serves the single-page app, an MJPEG camera stream, a WebSocket with
live state, and a small JSON API. All routes except /login and /static require a session.

Endpoints:
  GET  /                    app (redirects to /login without a session)
  GET  /stream.mjpg         annotated camera stream
  GET  /ws                  WebSocket, pushes state JSON at ~10 Hz
  GET  /api/state           same state as one JSON response
  GET  /api/map.json|.png   occupancy grid metadata / image (top row = max y)
  GET  /api/path            RTAB-Map trajectory [[x, y], ...]
  GET  /api/zones           exclusion and keep-out zones
  POST /api/zones           {"name": opt, "points": [[x, y], ...], "kind": "exclusion" | "keepout"}
  DELETE /api/zones/{name}
  POST /api/goal            {"x": .., "y": ..} navigation goal in the map frame (needs nav.launch.py)
  POST /api/goal/cancel
  GET  /api/plan            planned path [[x, y], ...]
  GET  /api/obstacles       local costmap obstacle cells [[x, y], ...] in the map frame
  POST /api/drive           {"v": m/s, "w": rad/s} drive pad; repeat every ~100 ms, stops 0.35 s
                            after the last request (needs roomba_driver)
  POST /api/robot/mode      {"mode": "safe" | "passive"}
  POST /api/robot/stop      stop driving and cancel navigation
  POST /api/explore         {"action": "start" | "stop" | "home"} frontier exploration (robot only)
  POST /api/floor_grid      {"on": bool} floor grid overlay on the camera stream
"""
import asyncio
import json
import math
import os
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.executors
from aiohttp import WSMsgType, web
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import BatteryState, CompressedImage, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

from roomba_perception.scene_filter import apply_transform
from roomba_web.auth import Auth

try:  # robot features need the Roomba driver's interfaces
    from roomba_msgs.msg import SensorState
    from roomba_msgs.srv import SetMode
except ImportError:
    SensorState = SetMode = None

SESSION_COOKIE = 'roomba_session'
CSRF_HEADER = 'X-Requested-With'
MAX_STREAMS = 4
DRIVE_HOLD_S = 0.35           # drive pad command lifetime; the page repeats it every 0.1 s
ROBOT_MODES = {'passive': 1, 'safe': 2}
HAZARDS = (('bump_left', 'Bumper pressed'), ('bump_right', 'Bumper pressed'),
           ('wheel_drop_left', 'Wheel lifted'), ('wheel_drop_right', 'Wheel lifted'),
           ('cliff_left', 'Cliff detected'), ('cliff_front_left', 'Cliff detected'),
           ('cliff_front_right', 'Cliff detected'), ('cliff_right', 'Cliff detected'),
           ('home_base_available', 'On the dock'), ('internal_charger_available', 'Charger plugged in'))


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RosState(Node):
    """Thread-safe snapshot of everything the web app shows."""

    def __init__(self):
        super().__init__('web_server')
        self.port = int(self.declare_parameter('port', 8080).value)
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.base_frame = self.declare_parameter('base_frame', 'camera_link').value
        self.localization_mode = bool(self.declare_parameter('localization', False).value)

        self.lock = threading.Lock()
        self.jpeg, self.jpeg_seq = None, 0
        self.map_png, self.map_meta, self.map_version = None, None, 0
        self.zones, self.zones_version = [], 0
        self.path, self.path_version = [], 0
        self.dets = {'active': (0.0, []), 'excluded': (0.0, [])}
        self.vo_state, self.vo_time = 0, 0.0
        self.localized = not self.localization_mode

        self.tf_buffer = Buffer()
        # spin_thread=True would spin this whole node from a second executor,
        # which races the action client's wait set and kills the TF thread.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.create_subscription(CompressedImage, '/perception/person_detector/annotated/compressed',
                                 self._on_jpeg, QoSProfile(depth=2))
        self.create_subscription(Detection3DArray, '/perception/person_detector/detections_3d',
                                 lambda m: self._on_dets(m, 'active'), QoSProfile(depth=5))
        self.create_subscription(Detection3DArray, '/perception/person_detector/ignored_3d',
                                 lambda m: self._on_dets(m, 'excluded'), QoSProfile(depth=5))
        # RTAB-Map publishes the grid latched (transient local) and only on change.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        # Prefer the planner's map (gaps filled, lined-up railing posts joined) when it exists.
        self.create_subscription(OccupancyGrid, '/rtabmap/map', lambda m: self._on_map(m, False), latched)
        self.create_subscription(OccupancyGrid, '/rtabmap/map_filled', lambda m: self._on_map(m, True), latched)
        self.last_filled_map = -1e9
        self.create_subscription(Path, '/rtabmap/mapPath', self._on_path, QoSProfile(depth=1))
        self.create_subscription(
            String, '/perception/ignore_zones', self._on_zones,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(VisualSlamStatus, '/visual_slam/status', self._on_status, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/rtabmap/localization_pose',
                                 self._on_localized, QoSProfile(depth=1))
        self.pub_zone_cmd = self.create_publisher(String, '/perception/ignore_zones/command', 10)

        # Navigation ("human robot" test): goals from the web, path/obstacles/instructions back.
        self.nav_instruction = None
        self.nav_goal = None
        self.nav_handle = None
        self.plan, self.plan_version = [], 0
        self.obstacles, self.obstacles_version = [], 0
        self.floor = None
        self.nav_requests = queue.Queue()
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_subscription(String, '/nav_coach/instruction', self._on_instruction, 10)
        self.create_subscription(Path, '/plan', self._on_plan, QoSProfile(depth=1))
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._on_costmap, QoSProfile(depth=1))
        self.create_subscription(String, '/perception/floor_status', self._on_floor, 2)
        self.vslam_filter = None
        self.create_subscription(String, '/vslam_filter/status', self._on_filter, 2)
        self.create_timer(0.1, self._process_nav_requests)

        # Robot (roomba_driver): status, drive pad, mode changes.
        self.max_drive_speed = float(self.declare_parameter('max_drive_speed', 0.2).value)
        self.max_drive_turn = float(self.declare_parameter('max_drive_turn', 1.0).value)
        self.robot_mode, self.robot_mode_time = None, 0.0
        self.battery, self.battery_time = None, 0.0
        self.hazards = []
        self.robot_msg = None             # (monotonic, text) result of the last mode request
        self.drive = None                 # (v, w, monotonic) latest drive pad command
        self.drive_active = False
        self.robot_requests = queue.Queue()
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 1)
        self.pub_cmd_nav = self.create_publisher(Twist, '/cmd_vel_nav', 1)
        self.mode_client = None
        if SensorState is not None:
            self.create_subscription(String, '/roomba/mode', self._on_robot_mode, latched)
            self.create_subscription(BatteryState, '/battery_state', self._on_battery, 2)
            self.create_subscription(SensorState, '/roomba/sensors', self._on_robot_sensors, 2)
            self.mode_client = self.create_client(SetMode, '/roomba/set_mode')
        self.create_timer(0.1, self._drive_tick)

        # Frontier exploration (explorer node, robot only).
        self.explore = None
        self.create_subscription(String, '/explorer/status', self._on_explore, latched)
        self.pub_explore = self.create_publisher(String, '/explorer/command', 10)
        # Floor grid overlay on the camera stream (drawn by person_detector).
        self.floor_grid = bool(self.declare_parameter('floor_grid', True).value)
        self.pub_floor_grid = self.create_publisher(Bool, '/perception/person_detector/floor_grid', latched)
        self.pub_floor_grid.publish(Bool(data=self.floor_grid))

        # Plain camera preview when the person detector isn't running (e.g. people:=false):
        # subscribe to the raw mono image only while no annotated frames arrive, and encode
        # a few small frames per second so the preview stays cheap.
        self.preview_fps = float(self.declare_parameter('preview_fps', 4.0).value)
        self.preview_topic = self.declare_parameter('preview_topic', '/camera/left/image_rect_mono').value
        self.preview_width = int(self.declare_parameter('preview_width', 480).value)
        self.last_annotated = 0.0
        self.last_preview = 0.0
        self.preview_sub = None
        if self.preview_fps > 0:
            self.create_timer(2.0, self._preview_check)

    # ---- ROS callbacks (executor thread) ----
    def _on_jpeg(self, msg):
        self.last_annotated = time.monotonic()
        with self.lock:
            self.jpeg = bytes(msg.data)
            self.jpeg_seq += 1

    def _preview_check(self):
        detector_live = time.monotonic() - self.last_annotated < 3.0
        if detector_live and self.preview_sub is not None:
            self.destroy_subscription(self.preview_sub)
            self.preview_sub = None
        elif not detector_live and self.preview_sub is None:
            self.preview_sub = self.create_subscription(
                Image, self.preview_topic, self._on_preview, qos_profile_sensor_data)

    def _on_preview(self, msg):
        now = time.monotonic()
        if now - self.last_preview < 1.0 / self.preview_fps or now - self.last_annotated < 3.0:
            return
        if msg.encoding not in ('mono8', '8UC1'):
            return
        self.last_preview = now
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        if self.preview_width and msg.width > self.preview_width:
            h = int(msg.height * self.preview_width / msg.width)
            img = cv2.resize(img, (self.preview_width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()
                self.jpeg_seq += 1

    def _to_map(self, header):
        for stamp in (Time.from_msg(header.stamp), Time()):
            try:
                return self.tf_buffer.lookup_transform(self.map_frame, header.frame_id, stamp)
            except TransformException:
                continue
        return None

    def _on_dets(self, msg, kind):
        tf = self._to_map(msg.header) if msg.detections else None
        out = []
        for d in msg.detections:
            c = d.bbox.center.position
            hyp = d.results[0].hypothesis if d.results else None
            item = {'cls': hyp.class_id if hyp else '?', 'score': round(hyp.score, 2) if hyp else 0.0,
                    'dist': round(c.z, 2), 'reason': d.id or None}
            if tf is not None:
                x, y, _ = apply_transform(tf, (c.x, c.y, c.z))
                item['x'], item['y'] = round(x, 3), round(y, 3)
            out.append(item)
        with self.lock:
            self.dets[kind] = (time.monotonic(), out)

    def _on_map(self, msg, filled=False):
        if filled:
            self.last_filled_map = time.monotonic()
        elif time.monotonic() - self.last_filled_map < 10.0:
            return
        info = msg.info
        grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        img = np.zeros((info.height, info.width, 4), np.uint8)  # BGRA, unknown = transparent
        img[grid >= 0] = (70, 60, 52, 255)          # free
        img[grid >= 65] = (240, 232, 225, 255)      # occupied
        ok, png = cv2.imencode('.png', np.ascontiguousarray(np.flipud(img)))
        if not ok:
            return
        o = info.origin.position
        with self.lock:
            self.map_version += 1
            self.map_png = png.tobytes()
            self.map_meta = {'version': self.map_version, 'resolution': info.resolution,
                             'width': info.width, 'height': info.height,
                             'origin': [o.x, o.y], 'yaw': _yaw(info.origin.orientation),
                             'frame': msg.header.frame_id}

    def _on_path(self, msg):
        pts = [[round(p.pose.position.x, 3), round(p.pose.position.y, 3)] for p in msg.poses]
        if len(pts) > 1000:
            pts = pts[::math.ceil(len(pts) / 1000)]
        with self.lock:
            self.path = pts
            self.path_version += 1

    def _on_zones(self, msg):
        try:
            zones = json.loads(msg.data).get('zones', [])
        except ValueError:
            return
        with self.lock:
            self.zones = zones
            self.zones_version += 1

    def _on_status(self, msg):
        with self.lock:
            self.vo_state, self.vo_time = msg.vo_state, time.monotonic()

    def _on_localized(self, _msg):
        with self.lock:
            self.localized = True

    def _on_instruction(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.nav_instruction = (time.monotonic(), data)
            if not data.get('active') and data.get('code') in ('arrived', 'failed', 'canceled'):
                self.nav_goal = None

    def _on_plan(self, msg):
        pts = [[round(p.pose.position.x, 3), round(p.pose.position.y, 3)] for p in msg.poses]
        if len(pts) > 500:
            pts = pts[::math.ceil(len(pts) / 500)]
        with self.lock:
            self.plan = pts
            self.plan_version += 1

    def _on_costmap(self, msg):
        """Lethal cells of the local costmap, as cell centres in the map frame."""
        info = msg.info
        grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        rows, cols = np.nonzero(grid >= 100)
        if rows.size > 3000:
            keep = np.linspace(0, rows.size - 1, 3000).astype(int)
            rows, cols = rows[keep], cols[keep]
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, msg.header.frame_id, Time())
        except TransformException:
            return
        o, r = info.origin.position, info.resolution
        pts = []
        for row, col in zip(rows.tolist(), cols.tolist()):
            x, y, _ = apply_transform(tf, (o.x + (col + 0.5) * r, o.y + (row + 0.5) * r, 0.0))
            pts.append([round(x, 3), round(y, 3)])
        with self.lock:
            self.obstacles = pts
            self.obstacles_version += 1

    def _on_floor(self, msg):
        try:
            self.floor = json.loads(msg.data)
        except ValueError:
            pass

    def _on_filter(self, msg):
        try:
            self.vslam_filter = json.loads(msg.data)
        except ValueError:
            pass

    # ---- robot (executor thread) ----
    def _on_robot_mode(self, msg):
        with self.lock:
            self.robot_mode, self.robot_mode_time = msg.data, time.monotonic()

    def _on_battery(self, msg):
        pct = msg.percentage
        with self.lock:
            self.battery = {'percent': round(pct * 100) if math.isfinite(pct) else None,
                            'voltage': round(msg.voltage, 2),
                            'charging': msg.power_supply_status == BatteryState.POWER_SUPPLY_STATUS_CHARGING}
            self.battery_time = time.monotonic()

    def _on_robot_sensors(self, msg):
        found = []
        for field, text in HAZARDS:
            if getattr(msg, field) and text not in found:
                found.append(text)
        with self.lock:
            self.hazards = found
            self.robot_mode_time = time.monotonic()
            self.robot_mode = {0: 'off', 1: 'passive', 2: 'safe', 3: 'full'}.get(msg.oi_mode, self.robot_mode)

    def robot_available(self):
        with self.lock:
            return time.monotonic() - self.battery_time < 3.0

    def _publish_drive(self, v, w):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        # With navigation running, go through Nav2's collision monitor so its stop zone also
        # protects manual driving (cmd_vel_nav -> collision_monitor -> cmd_vel).
        (self.pub_cmd_nav if self.nav_available() else self.pub_cmd).publish(msg)

    def _drive_tick(self):
        """Runs in the ROS thread: forwards the drive pad while it is held, then stops once."""
        with self.lock:
            cmd = self.drive
        if cmd is not None and time.monotonic() - cmd[2] < DRIVE_HOLD_S:
            if not self.drive_active:
                self.drive_active = True
                with self.lock:
                    navigating = self.nav_goal is not None
                if navigating:   # manual driving takes over from navigation
                    self.nav_requests.put(('cancel', None))
                self.explore_command('stop')
            self._publish_drive(cmd[0], cmd[1])
        elif self.drive_active:
            self.drive_active = False
            self._publish_drive(0.0, 0.0)
        while True:
            try:
                kind, arg = self.robot_requests.get_nowait()
            except queue.Empty:
                break
            if kind == 'stop':
                with self.lock:
                    self.drive = None
                self.drive_active = False
                self.explore_command('stop')
                self._publish_drive(0.0, 0.0)
            elif kind == 'mode' and self.mode_client is not None:
                req = SetMode.Request()
                req.mode = arg
                self.mode_client.call_async(req).add_done_callback(self._on_mode_result)

    def _on_mode_result(self, future):
        try:
            res = future.result()
            text = res.message if res.success else f'Could not change mode: {res.message}'
        except Exception as e:  # noqa: BLE001 - report any service failure to the page
            text = f'Could not change mode: {e}'
        with self.lock:
            self.robot_msg = (time.monotonic(), text)

    def set_drive(self, v, w):
        v = max(-self.max_drive_speed, min(self.max_drive_speed, v))
        w = max(-self.max_drive_turn, min(self.max_drive_turn, w))
        with self.lock:
            self.drive = (v, w, time.monotonic())

    def _process_nav_requests(self):
        """Runs in the ROS thread: action calls must not happen on the web thread."""
        while True:
            try:
                kind, arg = self.nav_requests.get_nowait()
            except queue.Empty:
                return
            if kind == 'goal':
                self.explore_command('stop')   # a goal from the map takes over from exploring
                goal = NavigateToPose.Goal()
                goal.pose.header.frame_id = self.map_frame
                goal.pose.header.stamp = self.get_clock().now().to_msg()
                goal.pose.pose.position.x, goal.pose.pose.position.y = arg
                goal.pose.pose.orientation.w = 1.0
                with self.lock:
                    self.nav_goal = {'x': arg[0], 'y': arg[1]}
                self.nav_client.send_goal_async(goal).add_done_callback(self._on_goal_response)
            elif kind == 'cancel':
                with self.lock:
                    handle, self.nav_goal = self.nav_handle, None
                if handle is not None:
                    handle.cancel_goal_async()

    def _on_goal_response(self, future):
        handle = future.result()
        with self.lock:
            if handle.accepted:
                self.nav_handle = handle
            else:
                self.nav_goal = None
                self.get_logger().warn('Navigation goal rejected')

    def nav_available(self):
        return self.nav_client.server_is_ready()

    def _on_explore(self, msg):
        try:
            status = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.explore = (time.monotonic(), status)

    def explore_command(self, cmd):
        """Only sends 'stop' while the explorer is actually driving (avoids log noise)."""
        if cmd == 'stop':
            with self.lock:
                active = self.explore is not None and self.explore[1].get('state') in ('exploring', 'returning')
            if not active:
                return
        self.pub_explore.publish(String(data=cmd))

    # ---- accessors (web thread) ----
    def pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except TransformException:
            return None
        t, r = tf.transform.translation, tf.transform.rotation
        return {'x': round(t.x, 3), 'y': round(t.y, 3), 'yaw': round(_yaw(r), 3)}

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            dets = []
            for kind, (stamp, items) in self.dets.items():
                if now - stamp < 0.6:
                    dets += [dict(i, excluded=(kind == 'excluded')) for i in items]
            if now - self.vo_time > 2.0:
                tracking = 'no data'
            else:
                tracking = {1: 'ok', 2: 'lost'}.get(self.vo_state, 'starting')
            state = {
                'tracking': tracking,
                'mode': 'localization' if self.localization_mode else 'mapping',
                'localized': self.localized,
                'detections': sorted(dets, key=lambda d: d['dist']),
                'map_version': self.map_version,
                'zones_version': self.zones_version,
                'path_version': self.path_version,
                'plan_version': self.plan_version,
                'obstacles_version': self.obstacles_version,
                'nav_goal': self.nav_goal,
                'floor': self.floor,
                'vslam_filter': self.vslam_filter,
            }
            if self.nav_instruction and now - self.nav_instruction[0] < 2.0:
                state['nav'] = self.nav_instruction[1]
            else:
                state['nav'] = None
            if now - self.battery_time < 3.0:
                state['robot'] = {
                    'mode': self.robot_mode if now - self.robot_mode_time < 3.0 else 'unknown',
                    'battery': self.battery,
                    'hazards': self.hazards,
                    'max_speed': self.max_drive_speed,
                    'max_turn': self.max_drive_turn,
                    'message': self.robot_msg[1] if self.robot_msg and now - self.robot_msg[0] < 6.0 else None,
                }
            else:
                state['robot'] = None
        state['nav_available'] = self.nav_available()
        state['pose'] = self.pose()
        with self.lock:
            state['explore'] = self.explore[1] if self.explore and self.pub_explore.get_subscription_count() else None
            state['floor_grid'] = self.floor_grid
        return state

    def set_floor_grid(self, on):
        with self.lock:
            self.floor_grid = bool(on)
        self.pub_floor_grid.publish(Bool(data=bool(on)))

    def send_zone_command(self, cmd):
        self.pub_zone_cmd.publish(String(data=json.dumps(cmd)))


def build_app(ros, auth, www):

    def authed(request):
        return auth.valid(request.cookies.get(SESSION_COOKIE))

    @web.middleware
    async def guard(request, handler):
        path = request.path
        if path in ('/login', '/favicon.ico') or path.startswith('/static/'):
            return await handler(request)
        if not authed(request):
            if path.startswith('/api/') or path in ('/ws', '/stream.mjpg'):
                raise web.HTTPUnauthorized()
            raise web.HTTPFound('/login')
        if request.method in ('POST', 'DELETE', 'PUT') and path.startswith('/api/') \
                and request.headers.get(CSRF_HEADER) != 'roomba-web':
            raise web.HTTPForbidden(text='missing request header')
        return await handler(request)

    async def index(_request):
        return web.FileResponse(os.path.join(www, 'index.html'), headers={'Cache-Control': 'no-cache'})

    async def login_page(_request):
        return web.FileResponse(os.path.join(www, 'login.html'), headers={'Cache-Control': 'no-cache'})

    async def login(request):
        form = await request.post()
        password = str(form.get('password', ''))
        ok = await asyncio.get_running_loop().run_in_executor(None, auth.check, password)
        if not ok:
            await asyncio.sleep(1.0)  # slow down guessing
            raise web.HTTPFound('/login?error=1')
        resp = web.HTTPFound('/')
        resp.set_cookie(SESSION_COOKIE, auth.new_session(), httponly=True, samesite='Strict',
                        max_age=7 * 24 * 3600, path='/')
        raise resp

    async def logout(request):
        auth.revoke(request.cookies.get(SESSION_COOKIE))
        resp = web.HTTPFound('/login')
        resp.del_cookie(SESSION_COOKIE, path='/')
        raise resp

    streams = {'n': 0}

    async def stream(request):
        if streams['n'] >= MAX_STREAMS:
            raise web.HTTPServiceUnavailable(text='too many camera viewers')
        resp = web.StreamResponse(headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-cache, no-store'})
        await resp.prepare(request)
        streams['n'] += 1
        last = -1
        try:
            while True:
                with ros.lock:
                    seq, jpg = ros.jpeg_seq, ros.jpeg
                if jpg is not None and seq != last:
                    last = seq
                    await resp.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                     + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                await asyncio.sleep(0.02)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            streams['n'] -= 1
        return resp

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=10.0)
        await ws.prepare(request)

        async def reader():
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break

        task = asyncio.ensure_future(reader())
        try:
            while not ws.closed:
                await ws.send_str(json.dumps(ros.snapshot()))
                await asyncio.sleep(0.1)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            task.cancel()
        return ws

    async def api_state(_request):
        return web.json_response(ros.snapshot())

    async def api_map_json(_request):
        with ros.lock:
            meta = ros.map_meta
        if meta is None:
            raise web.HTTPNotFound(text='no map yet')
        return web.json_response(meta)

    async def api_map_png(_request):
        with ros.lock:
            png = ros.map_png
        if png is None:
            raise web.HTTPNotFound(text='no map yet')
        return web.Response(body=png, content_type='image/png', headers={'Cache-Control': 'no-cache'})

    async def api_path(_request):
        with ros.lock:
            return web.json_response({'version': ros.path_version, 'points': ros.path})

    async def api_zones(_request):
        with ros.lock:
            return web.json_response({'version': ros.zones_version, 'zones': ros.zones})

    async def api_add_zone(request):
        try:
            body = await request.json()
            pts = [[float(p[0]), float(p[1])] for p in body['points']]
        except (ValueError, KeyError, TypeError, IndexError):
            raise web.HTTPBadRequest(text='expected {"points": [[x, y], ...]}')
        if not 3 <= len(pts) <= 64:
            raise web.HTTPBadRequest(text='a zone needs 3 to 64 corners')
        kind = body.get('kind') or 'exclusion'
        if kind not in ('exclusion', 'keepout'):
            raise web.HTTPBadRequest(text='kind must be "exclusion" or "keepout"')
        name = body.get('name')
        with ros.lock:
            if name and any(z['name'] == name for z in ros.zones):
                raise web.HTTPConflict(text=f'zone "{name}" already exists')
        ros.send_zone_command({'op': 'add', 'name': name, 'points': pts, 'kind': kind})
        return web.json_response({'ok': True})

    async def api_delete_zone(request):
        ros.send_zone_command({'op': 'delete', 'name': request.match_info['name']})
        return web.json_response({'ok': True})

    async def api_goal(request):
        try:
            body = await request.json()
            x, y = float(body['x']), float(body['y'])
        except (ValueError, KeyError, TypeError):
            raise web.HTTPBadRequest(text='expected {"x": .., "y": ..}')
        if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > 1000 or abs(y) > 1000:
            raise web.HTTPBadRequest(text='goal out of range')
        if not ros.nav_available():
            raise web.HTTPServiceUnavailable(text='navigation is not running (start ~/run_nav.sh)')
        ros.nav_requests.put(('goal', (x, y)))
        return web.json_response({'ok': True})

    async def api_cancel(_request):
        ros.nav_requests.put(('cancel', None))
        return web.json_response({'ok': True})

    async def api_plan(_request):
        with ros.lock:
            return web.json_response({'version': ros.plan_version, 'points': ros.plan})

    async def api_obstacles(_request):
        with ros.lock:
            return web.json_response({'version': ros.obstacles_version, 'points': ros.obstacles})

    async def api_drive(request):
        try:
            body = await request.json()
            v, w = float(body['v']), float(body['w'])
        except (ValueError, KeyError, TypeError):
            raise web.HTTPBadRequest(text='expected {"v": .., "w": ..}')
        if not (math.isfinite(v) and math.isfinite(w)):
            raise web.HTTPBadRequest(text='bad velocity')
        if not ros.robot_available():
            raise web.HTTPServiceUnavailable(text='the robot driver is not running')
        ros.set_drive(v, w)
        return web.json_response({'ok': True})

    async def api_robot_mode(request):
        try:
            mode = ROBOT_MODES[(await request.json())['mode']]
        except (ValueError, KeyError, TypeError):
            raise web.HTTPBadRequest(text='expected {"mode": "safe" | "passive"}')
        if ros.mode_client is None or not ros.mode_client.service_is_ready():
            raise web.HTTPServiceUnavailable(text='the robot driver is not running')
        ros.robot_requests.put(('mode', mode))
        return web.json_response({'ok': True})

    async def api_robot_stop(_request):
        ros.robot_requests.put(('stop', None))
        ros.nav_requests.put(('cancel', None))
        return web.json_response({'ok': True})

    async def api_explore(request):
        try:
            action = (await request.json())['action']
        except (ValueError, KeyError, TypeError):
            raise web.HTTPBadRequest(text='expected {"action": "start" | "stop" | "home"}')
        if action not in ('start', 'stop', 'home'):
            raise web.HTTPBadRequest(text='action must be start, stop or home')
        if ros.pub_explore.get_subscription_count() == 0:
            raise web.HTTPServiceUnavailable(text='exploration is not running (robot.launch.py with nav)')
        if action == 'start':
            ros.nav_requests.put(('cancel', None))   # don't fight a goal sent from the map
        ros.pub_explore.publish(String(data=action))
        return web.json_response({'ok': True})

    async def api_floor_grid(request):
        try:
            on = bool((await request.json())['on'])
        except (ValueError, KeyError, TypeError):
            raise web.HTTPBadRequest(text='expected {"on": true | false}')
        ros.set_floor_grid(on)
        return web.json_response({'ok': True, 'on': on})

    app = web.Application(middlewares=[guard], client_max_size=256 * 1024)

    async def revalidate_static(request, response):
        # Without this, browsers kept running an old app.js after an update (the HTML is
        # no-cache, so new buttons appeared without their code). ETags keep reloads cheap.
        if request.path.startswith('/static/'):
            response.headers['Cache-Control'] = 'no-cache'
    app.on_response_prepare.append(revalidate_static)
    app.router.add_get('/', index)
    app.router.add_get('/login', login_page)
    app.router.add_post('/login', login)
    app.router.add_post('/logout', logout)
    app.router.add_get('/stream.mjpg', stream)
    app.router.add_get('/ws', ws_handler)
    app.router.add_get('/api/state', api_state)
    app.router.add_get('/api/map.json', api_map_json)
    app.router.add_get('/api/map.png', api_map_png)
    app.router.add_get('/api/path', api_path)
    app.router.add_get('/api/zones', api_zones)
    app.router.add_post('/api/zones', api_add_zone)
    app.router.add_delete('/api/zones/{name}', api_delete_zone)
    app.router.add_post('/api/goal', api_goal)
    app.router.add_post('/api/goal/cancel', api_cancel)
    app.router.add_get('/api/plan', api_plan)
    app.router.add_get('/api/obstacles', api_obstacles)
    app.router.add_post('/api/drive', api_drive)
    app.router.add_post('/api/robot/mode', api_robot_mode)
    app.router.add_post('/api/robot/stop', api_robot_stop)
    app.router.add_post('/api/explore', api_explore)
    app.router.add_post('/api/floor_grid', api_floor_grid)
    app.router.add_static('/static/', www, show_index=False)
    return app


def main():
    rclpy.init()
    ros = RosState()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(ros)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    www = os.path.join(get_package_share_directory('roomba_web'), 'www')
    auth = Auth(ros.get_logger())
    ros.get_logger().info(f'Web app on http://0.0.0.0:{ros.port}/')
    try:
        web.run_app(build_app(ros, auth, www), host='0.0.0.0', port=ros.port, print=None,
                    access_log=None)
    finally:
        executor.shutdown()
        ros.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
