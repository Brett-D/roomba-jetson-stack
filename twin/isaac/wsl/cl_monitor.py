#!/usr/bin/env python3
"""Closed-loop run recorder (WSL, sim time): logs what the evaluation needs and starts the explorer.

  cl_monitor.py --out <run>/monitor --ros-args -p use_sim_time:=true

Writes (1 file each, flushed as it goes):
  poses.csv      2 Hz: sim t, ground truth (Isaac map frame), RTAB-Map map->base_link, EKF odom->base_link,
                 cuVSLAM vslam_odom->base_link, cuVSLAM tracking state
  events.jsonl   explorer status changes, OI mode, bumps / cliffs (from /roomba/sensors), cuVSLAM status
                 changes, vslam_filter status changes, Isaac status (RTF, bump count) every 10 s
  detections.csv person detector 2D detections (t, n, boxes) and Isaac's people positions (people.csv)
  map_rtabmap.npz / map_costmap.npz   latest /rtabmap/map and /global_costmap/costmap (on exit, and every 60 s)
  summary.json   on exit
Starts exploration ("start" on /explorer/command) once the OI is in Safe mode, Nav2 answers and RTAB-Map has
published a map, after --settle s of sim time (optionally after a 360 deg look-around); stops recording --after-done s
after the explorer is done. --assist: logged interventions when the robot is stuck (see check_stuck).
"""
import argparse
import csv
import json
import math
import os
import signal
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

try:
    from roomba_msgs.msg import SensorState
except ImportError:
    SensorState = None
try:
    from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
except ImportError:
    VisualSlamStatus = None
try:
    from vision_msgs.msg import Detection2DArray
except ImportError:
    Detection2DArray = None


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def rdp(P, eps):
    """Ramer-Douglas-Peucker simplification of a closed polygon (N x 2)."""
    def rec(a, b):
        seg = P[b] - P[a]
        L = np.hypot(*seg) or 1e-9
        d = np.abs(np.cross(seg, P[a + 1:b] - P[a])) / L if b > a + 1 else np.array([])
        if d.size and d.max() > eps:
            k = a + 1 + int(d.argmax())
            return rec(a, k)[:-1] + rec(k, b)
        return [a, b]
    if np.allclose(P[0], P[-1]) or len(P) < 4:
        far = int(np.argmax(np.hypot(*(P - P[0]).T)))
        idx = rec(0, far)[:-1] + rec(far, len(P) - 1)
    else:
        idx = rec(0, len(P) - 1)
    idx = list(dict.fromkeys(idx))
    if len(idx) > 1 and np.allclose(P[idx[0]], P[idx[-1]]):
        idx = idx[:-1]
    return P[idx]


class Monitor(Node):
    def __init__(self, a):
        super().__init__('cl_monitor')
        self.a = a
        os.makedirs(a.out, exist_ok=True)
        self.t_wall0 = time.time()
        self.f_pose = open(os.path.join(a.out, 'poses.csv'), 'w', newline='', buffering=1)
        self.w_pose = csv.writer(self.f_pose)
        self.w_pose.writerow(['t', 'wall', 'gt_x', 'gt_y', 'gt_yaw', 'map_x', 'map_y', 'map_yaw', 'odom_x', 'odom_y',
                              'odom_yaw', 'vo_x', 'vo_y', 'vo_yaw', 'vo_state', 'wheel_x', 'wheel_y', 'wheel_yaw', 'cmd_v', 'cmd_w'])
        self.f_ev = open(os.path.join(a.out, 'events.jsonl'), 'w', buffering=1)
        self.f_det = open(os.path.join(a.out, 'detections.csv'), 'w', newline='', buffering=1)
        self.w_det = csv.writer(self.f_det)
        self.w_det.writerow(['t', 'n', 'boxes'])
        self.f_ppl = open(os.path.join(a.out, 'people.csv'), 'w', newline='', buffering=1)
        self.w_ppl = csv.writer(self.f_ppl)
        self.w_ppl.writerow(['t', 'people'])
        self.tf = Buffer(cache_time=Duration(seconds=30))
        self.tfl = TransformListener(self.tf, self)
        rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.gt = self.vo = self.wheel = None
        self.cmd_vel = (0.0, 0.0)
        self.vo_state = None
        self.explorer = {}
        self.mode = None
        self.maps = {}
        self.counts = {'bump_events': 0, 'cliff_events': 0, 'detections': 0, 'det_frames': 0, 'det_msgs': 0,
                       'vo_state_changes': 0, 'tracking_lost': 0, 'assists': 0}
        self.prev_bump = False
        self.prev_cliff = False
        self.started_at = None
        self.done_at = None
        self.isaac = {}
        self.filter_state = None
        self.create_subscription(Odometry, '/ground_truth/odom', lambda m: setattr(self, 'gt', m), rel)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', lambda m: setattr(self, 'vo', m), 10)
        self.create_subscription(Odometry, '/wheel/odom', lambda m: setattr(self, 'wheel', m), 10)
        from geometry_msgs.msg import Twist
        self.create_subscription(Twist, '/cmd_vel', lambda m: setattr(self, 'cmd_vel', (m.linear.x, m.angular.z)), 10)
        self.create_subscription(String, '/explorer/status', self.on_explorer, latched)
        self.create_subscription(String, '/roomba/mode', self.on_mode, latched)
        self.create_subscription(String, '/vslam_filter/status', self.on_filter, 10)
        self.create_subscription(String, '/isaac/status', self.on_isaac, rel)
        self.create_subscription(String, '/isaac/people', self.on_people, rel)
        self.create_subscription(OccupancyGrid, '/rtabmap/map', lambda m: self.maps.__setitem__('rtabmap', m), latched)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', lambda m: self.maps.__setitem__('costmap', m),
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        if SensorState:
            self.create_subscription(SensorState, '/roomba/sensors', self.on_sensors, 10)
        if VisualSlamStatus:
            self.create_subscription(VisualSlamStatus, '/visual_slam/status', self.on_vslam, 10)
        if Detection2DArray:
            self.create_subscription(Detection2DArray, '/perception/person_detector/detections_2d', self.on_det, 10)
        self.cmd = self.create_publisher(String, '/explorer/command', 10)
        self.zone_pub = self.create_publisher(String, '/perception/ignore_zones/command', 10)   # node perception/ignore_zones
        self.zone_ok = False
        self.create_subscription(String, '/perception/ignore_zones', self.on_zones,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.look = None            # startup look-around: None (not yet), 'running', 'done'
        self.hist = []              # (t, gt x, gt y) for the stuck detector
        self.assist_until = None
        self.assists = []
        if a.assist:
            from geometry_msgs.msg import Twist as _Twist
            from rclpy.clock import Clock, ClockType
            self.Twist = _Twist
            self.teleop = self.create_publisher(_Twist, '/isaac/teleop_cmd_vel', 10)
            # Isaac's teleop override times out after 0.5 s WALL time: publish on a wall-clock timer
            self.create_timer(0.1, self.assist_tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        if a.look_around:
            from nav2_msgs.action import Spin
            from rclpy.action import ActionClient
            self.Spin = Spin
            self.spin_client = ActionClient(self, Spin, '/spin')
        self.create_timer(0.5, self.on_tick)
        self.create_timer(60.0, self.save_maps)
        self.event('monitor_start', explorer_autostart=a.autostart, look_around=a.look_around)

    def start_look_around(self, t):
        """One slow 360 deg turn with Nav2's Spin behaviour before exploring (see --look-around)."""
        self.look = 'running'
        self.look_t = t
        goal = self.Spin.Goal()
        goal.target_yaw = 2 * math.pi
        goal.time_allowance.sec = 90
        self.event('look_around_sent')

        def done(fut):
            self.look = 'done'
            self.event('look_around_done', status=getattr(fut.result(), 'status', None))

        def accepted(fut):
            h = fut.result()
            if not h or not h.accepted:
                self.look = 'done'
                self.event('look_around_rejected')
                return
            h.get_result_async().add_done_callback(done)
        if not self.spin_client.wait_for_server(timeout_sec=2.0):
            self.look = 'done'
            self.event('look_around_no_server')
            return
        self.spin_client.send_goal_async(goal).add_done_callback(accepted)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def event(self, ev, **kw):
        kw.update(ev=ev, t=round(self.now(), 3), wall=round(time.time() - self.t_wall0, 1))
        self.f_ev.write(json.dumps(kw) + '\n')

    def on_zones(self, m):
        try:
            z = json.loads(m.data)
        except ValueError:
            return
        zones = z.get('zones', z) if isinstance(z, dict) else z
        ok = any(isinstance(q, dict) and q.get('name') == 'stairwell' for q in (zones or []))
        if ok and not self.zone_ok:
            self.event('keepout_active')
        self.zone_ok = ok

    def on_explorer(self, m):
        try:
            s = json.loads(m.data)
        except ValueError:
            return
        key = (s.get('state'), s.get('message'))
        if key != (self.explorer.get('state'), self.explorer.get('message')):
            self.event('explorer', **{k: s[k] for k in s if k in ('state', 'message', 'goals', 'reached', 'failed',
                                                                    'frontiers', 'blacklisted', 'bumps')})
        if s.get('state') == 'done' and self.done_at is None:
            self.done_at = self.now()
        self.explorer = s

    def on_mode(self, m):
        if m.data != self.mode:
            self.event('mode', mode=m.data)
        self.mode = m.data

    def on_filter(self, m):
        try:
            s = json.loads(m.data)
        except ValueError:
            return
        st = s.get('state')
        if st != self.filter_state:
            self.event('vslam_filter', **{k: v for k, v in s.items() if isinstance(v, (str, int, float, bool))})
            if self.filter_state is not None and st != 'idle':
                self.counts['tracking_lost'] += 1
        self.filter_state = st

    def on_vslam(self, m):
        st = int(m.vo_state)
        if st != self.vo_state:
            self.event('cuvslam', vo_state=st)
            if self.vo_state is not None:
                self.counts['vo_state_changes'] += 1
        self.vo_state = st

    def on_sensors(self, m):
        bump = bool(m.bump_left or m.bump_right)
        cliff = bool(m.cliff_left or m.cliff_front_left or m.cliff_front_right or m.cliff_right)
        if bump and not self.prev_bump:
            self.counts['bump_events'] += 1
            self.event('bump', left=bool(m.bump_left), right=bool(m.bump_right))
        if cliff and not self.prev_cliff:
            self.counts['cliff_events'] += 1
            self.event('cliff', fl=bool(m.cliff_front_left), fr=bool(m.cliff_front_right), l=bool(m.cliff_left),
                       r=bool(m.cliff_right))
        self.prev_bump, self.prev_cliff = bump, cliff

    def on_isaac(self, m):
        try:
            self.isaac = json.loads(m.data)
        except ValueError:
            pass

    def on_people(self, m):
        try:
            d = json.loads(m.data)
            self.w_ppl.writerow([round(d['t'], 3), json.dumps(d['people'])])
        except (ValueError, KeyError):
            pass

    def on_det(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        boxes = [[round(d.bbox.center.position.x, 1), round(d.bbox.center.position.y, 1), round(d.bbox.size_x, 1),
                  round(d.bbox.size_y, 1), round(d.results[0].hypothesis.score, 3) if d.results else None,
                  d.results[0].hypothesis.class_id if d.results else None] for d in m.detections]
        self.counts['det_msgs'] += 1
        if boxes:
            self.counts['det_frames'] += 1
            self.counts['detections'] += len(boxes)
        self.w_det.writerow([round(t, 3), len(boxes), json.dumps(boxes)])

    def lookup(self, target, source):
        try:
            tr = self.tf.lookup_transform(target, source, Time())
            q = tr.transform.rotation
            return tr.transform.translation.x, tr.transform.translation.y, yaw_of(q)
        except Exception:  # noqa: BLE001
            return None

    def send_keepout(self):
        """Keep-out zone around the stairwell, as a user would draw it in the web app (ignore_zones 'add' op, map
        frame = the twin's frame): the depth camera looks through the balusters, the void below reads as free floor
        and the explorer drives into the railing otherwise (README, findings)."""
        fp = json.load(open(self.a.keepout_plan))
        pts = np.array(fp['stair_void']['polygon'], float)
        keep = rdp(pts, 0.03)
        while len(keep) > 60:
            keep = keep[::2]
        cmd = {'op': 'add', 'kind': 'keepout', 'name': 'stairwell', 'points': [[float(x), float(y)] for x, y in keep]}
        self.zone_pub.publish(String(data=json.dumps(cmd)))
        self.event('keepout_sent', points=len(keep))

    def assist_tick(self):
        """Intervention ("a person pulls the robot back"): reverse at 0.1 m/s for --assist-s sim seconds through
        Isaac's teleop override, which drives the wheels directly (encoders, IMU and cameras stay consistent)."""
        if self.assist_until is None:
            return
        tw = self.Twist()
        if self.now() < self.assist_until:
            tw.linear.x = -0.1
            self.teleop.publish(tw)
        else:
            self.teleop.publish(tw)
            self.assist_until = None
            self.event('assist_done')

    def check_stuck(self, t, gt):
        if gt is None:
            return
        self.hist.append((t, gt[0], gt[1]))
        while self.hist and t - self.hist[0][0] > self.a.stuck_s + 5:
            self.hist.pop(0)
        busy = self.explorer.get('state') in ('exploring', 'returning') and self.started_at is not None
        if not busy or self.assist_until is not None or t - self.hist[0][0] < self.a.stuck_s:
            return
        if self.assists and t - self.assists[-1] < self.a.stuck_s + 20:
            return
        recent = [h for h in self.hist if t - h[0] <= self.a.stuck_s]
        span = max(math.hypot(h[1] - recent[0][1], h[2] - recent[0][2]) for h in recent)
        if span < 0.05:
            self.assists.append(t)
            self.assist_until = t + self.a.assist_s
            self.counts['assists'] = len(self.assists)
            self.event('assist', x=round(gt[0], 3), y=round(gt[1], 3), reason=f'no motion for {self.a.stuck_s:.0f} s',
                       explorer=self.explorer.get('message'))

    def on_tick(self):
        t = self.now()
        if t <= 0:
            return
        if getattr(self, 't_first', None) is None:
            self.t_first = t           # the sim may have run for a while before this stack started
        gt = (self.gt.pose.pose.position.x, self.gt.pose.pose.position.y, yaw_of(self.gt.pose.pose.orientation)) if self.gt else None
        mp = self.lookup('map', 'base_link')
        od = self.lookup('odom', 'base_link')
        vo = (self.vo.pose.pose.position.x, self.vo.pose.pose.position.y, yaw_of(self.vo.pose.pose.orientation)) if self.vo else None

        def f(x):
            return ['', '', ''] if x is None else [round(x[0], 4), round(x[1], 4), round(x[2], 4)]
        wh = (self.wheel.pose.pose.position.x, self.wheel.pose.pose.position.y, yaw_of(self.wheel.pose.pose.orientation)) if self.wheel else None
        self.w_pose.writerow([round(t, 3), round(time.time() - self.t_wall0, 2)] + f(gt) + f(mp) + f(od) + f(vo) +
                             [self.vo_state if self.vo_state is not None else ''] + f(wh) +
                             [round(self.cmd_vel[0], 3), round(self.cmd_vel[1], 3)])
        if int(t * 2) % 20 == 0 and self.isaac:
            self.event('isaac', **{k: self.isaac[k] for k in ('rtf', 'bumps', 'teleports', 'x', 'y', 'pushes_undone')
                                         if k in self.isaac})
        if self.a.assist:
            self.check_stuck(t, gt)
        # start exploring once everything is up
        if self.a.autostart and self.started_at is None:
            ready = (self.mode in ('safe', 'full') and 'rtabmap' in self.maps and mp is not None
                     and t - self.t_first > self.a.settle)
            if ready and self.a.keepout_plan and not self.zone_ok and t - self.t_first < self.a.settle + 30:
                if t - getattr(self, 'zone_sent_t', -1e9) > 3.0:
                    self.zone_sent_t = t
                    self.send_keepout()
            elif ready and self.a.look_around and self.look is None:
                self.start_look_around(t)
            elif ready and (self.look in (None, 'done') or (self.look == 'running' and t - self.look_t > 90)):
                self.cmd.publish(String(data='start'))
                self.started_at = t
                self.event('explorer_start_sent')
        elif self.started_at is not None and self.explorer.get('state') in (None, 'idle') and t - self.started_at > 10 \
                and not getattr(self, 'resent', False):
            self.cmd.publish(String(data='start'))
            self.resent = True
            self.event('explorer_start_resent')
        if self.done_at is not None and t - self.done_at > self.a.after_done:
            self.finish('explorer done')
        if self.a.max_sim and self.started_at is not None and t - self.started_at > self.a.max_sim:
            self.finish(f'time limit {self.a.max_sim} s sim')

    def save_maps(self):
        for k, m in self.maps.items():
            g = np.array(m.data, np.int8).reshape(m.info.height, m.info.width)
            o = m.info.origin
            np.savez_compressed(os.path.join(self.a.out, f'map_{k}.npz'), grid=g, resolution=m.info.resolution,
                                origin=[o.position.x, o.position.y, yaw_of(o.orientation)], frame=m.header.frame_id,
                                t=self.now())

    def finish(self, why):
        if getattr(self, 'finished', False):
            return
        self.finished = True
        self.save_maps()
        summary = {'reason': why, 'sim_t_end': round(self.now(), 2), 'wall_s': round(time.time() - self.t_wall0, 1),
                   'explorer_started_at': self.started_at, 'explorer_done_at': self.done_at,
                   'explorer_final': self.explorer, 'mode_final': self.mode, 'counts': self.counts, 'isaac': self.isaac}
        with open(os.path.join(self.a.out, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)
        self.event('finish', reason=why)
        with open(os.path.join(self.a.out, 'FINISHED'), 'w') as f:
            f.write(why + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--settle', type=float, default=20.0, help='sim s after the monitor starts before exploring')
    ap.add_argument('--after-done', type=float, default=15.0)
    ap.add_argument('--max-sim', type=float, default=0.0, help='stop recording after this many sim s of exploring')
    ap.add_argument('--no-autostart', dest='autostart', action='store_false')
    ap.add_argument('--look-around', action='store_true',
                    help='turn 360 deg in place (Nav2 Spin) before starting the explorer: works around the explorer '
                         'marking a frontier behind the start as visited without ever facing it (README, findings)')
    ap.add_argument('--assist', action='store_true',
                    help='if the robot has not moved for --stuck-s sim s while exploring, reverse it --assist-s s '
                         '(a logged intervention: Nav2 cannot leave a pose whose footprint touches lethal cost)')
    ap.add_argument('--stuck-s', type=float, default=40.0)
    ap.add_argument('--assist-s', type=float, default=3.0)
    ap.add_argument('--keepout-plan', default='',
                    help='floorplan.json: add its stair void as a keep-out zone (web-app style) before exploring')
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = Monitor(a)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    try:
        while rclpy.ok() and not stop:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.finish('stopped' if stop else 'ended')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
