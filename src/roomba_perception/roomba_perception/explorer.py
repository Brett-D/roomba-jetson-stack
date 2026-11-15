#!/usr/bin/env python3
"""Frontier exploration: drives the robot to the edge of the known map until nothing is left.

A frontier is free, known space next to unknown space in Nav2's global costmap (which
already includes RTAB-Map's grid, live obstacles and keep-out zones). Frontier cells are
grouped; only groups connected to the robot through passable space count. Each gets a goal
parked in clear, known space (goal_standoff_m from the unknown, away from obstacles) near
it, facing it, so the controller isn't asked to drive into clutter at the map's edge. The
explorer drives to the nearest one and repeats. Spots reached are not revisited and goals
that fail are blacklisted for the session. When no frontier is left (or on request) it
returns to where it started. Nav2 answers that never come don't stall it: a goal request not
answered within goal_response_timeout_s counts as rejected (a late acceptance is cancelled), and
a trip home with no result within return_timeout_s as failed.

Safety: exploring stops if the Roomba leaves safe/full mode (cliff, wheel drop, charger:
the firmware drops it to passive), and it returns home when the battery runs low. While
visual tracking is recovering it pauses. During exploration the controller's in-place
turn speed can be lowered (rotate_speed_rps) to keep visual tracking. With auto_keepout,
a cliff-sensor stop (stairs, or deep carpet that reads like a drop; not a wheel drop) adds
a keep-out zone just ahead of the robot, in exploring, navigation or manual driving.

Subscribes: /global_costmap/costmap (OccupancyGrid), /roomba/mode (String), /battery_state,
            /roomba/sensors, /vslam_filter/status (String JSON),
            ~/command (String: "start" | "stop" | "home")
Publishes:  ~/status (String JSON, transient local), /perception/ignore_zones/command
Uses:       navigate_to_pose action, /controller_server parameters, TF map -> base frame
"""
import json
import math
import time

import cv2
import numpy as np
import rclpy
import rclpy.executors
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

ROTATE_PARAM = 'FollowPath.rotate_to_heading_angular_vel'
# Cliff sensor directions on the Roomba (rad, left positive) for automatic keep-out zones.
CLIFF_ANGLES = {'cliff_left': 1.05, 'cliff_front_left': 0.35, 'cliff_front_right': -0.35, 'cliff_right': -1.05}
# Light-bumper (IR proximity) directions, used to tell where on the bumper a hit was.
LIGHT_ANGLES = {'light_bump_left_signal': 1.2, 'light_bump_front_left_signal': 0.65,
                'light_bump_center_left_signal': 0.2, 'light_bump_center_right_signal': -0.2,
                'light_bump_front_right_signal': -0.65, 'light_bump_right_signal': -1.2}

try:
    from roomba_msgs.msg import SensorState
except ImportError:   # driver messages not built: no automatic keep-out zones
    SensorState = None


def bump_zone(x, y, yaw, bearing, contact=0.18, depth=0.06, width=0.3):
    """Thin keep-out strip (map frame) on the surface the bumper touched, `contact` m from the
    robot centre along `bearing`, extending `depth` m into the obstacle."""
    a = yaw + bearing
    ca, sa = math.cos(a), math.sin(a)
    h = width / 2.0
    corners = ((contact, -h), (contact + depth, -h), (contact + depth, h), (contact, h))
    return [[round(x + u * ca - v * sa, 3), round(y + u * sa + v * ca, 3)] for u, v in corners]


def bump_bearing(msg, light_min=200):
    """Direction of a bumper hit relative to the robot heading (rad, + = left).

    The switches alone are coarse (and this robot's right switch is masked as faulty, so a head-on
    hit reads as left), so the light bumpers' signals at the moment of contact pick the direction
    when they see something (deep carpet pile reads ~100-200); otherwise fall back to the switches.
    """
    sig = {k: getattr(msg, k, 0) for k in LIGHT_ANGLES}
    total = sum(v for v in sig.values() if v >= light_min)
    if total > 0:
        return math.atan2(sum(v * math.sin(LIGHT_ANGLES[k]) for k, v in sig.items() if v >= light_min),
                          sum(v * math.cos(LIGHT_ANGLES[k]) for k, v in sig.items() if v >= light_min))
    left, right = bool(msg.bump_left), bool(msg.bump_right)
    return 0.5 if left and not right else -0.5 if right and not left else 0.0


def cliff_zone(x, y, yaw, bearing, distance=0.42, size=0.35):
    """Square keep-out polygon (map frame) centred `distance` ahead of the robot along `bearing`."""
    a = yaw + bearing
    cx, cy = x + distance * math.cos(a), y + distance * math.sin(a)
    h = size / 2.0
    corners = ((h, h), (-h, h), (-h, -h), (h, -h))
    return [[round(cx + u * math.cos(a) - v * math.sin(a), 3), round(cy + u * math.sin(a) + v * math.cos(a), 3)]
            for u, v in corners]


def find_frontiers(grid, free_max, min_cells):
    """Frontier groups in an occupancy grid (int array, -1 unknown, 0..100 cost).

    Returns a list of (cells_yx array, size). Cells outside the grid count as unknown.
    """
    free = (grid >= 0) & (grid <= free_max)
    unknown = np.pad(grid < 0, 1, constant_values=True)
    near_unknown = (unknown[:-2, 1:-1] | unknown[2:, 1:-1] | unknown[1:-1, :-2] | unknown[1:-1, 2:])
    frontier = (free & near_unknown).astype(np.uint8)
    count, labels = cv2.connectedComponents(frontier, connectivity=8)
    groups = []
    for i in range(1, count):
        cells = np.argwhere(labels == i)
        if len(cells) >= min_cells:
            groups.append((cells, len(cells)))
    return groups


def frontier_goals(grid, robot_rc, free_max, pass_max, clear_max, min_cells, standoff_cells, search_cells):
    """Reachable frontier groups with a goal cell parked in clear, known space next to each.

    grid: int costmap (-1 unknown, 0..100). robot_rc: (row, col) of the robot.
    Returns (goals, seed) where goals is a list of (goal_rc, centroid_rc, size) and seed the
    robot's free cell (None if the robot isn't next to passable space).
    """
    h, w = grid.shape
    passable = ((grid >= 0) & (grid <= pass_max)).astype(np.uint8)
    r, c = robot_rc
    seed = None
    for rad in range(0, 7):   # the robot's own cell can be inflated: use the nearest passable one
        r0, r1, c0, c1 = max(0, r - rad), min(h, r + rad + 1), max(0, c - rad), min(w, c + rad + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        hits = np.argwhere(passable[r0:r1, c0:c1])
        if len(hits):
            k = int(np.argmin(((hits + [r0, c0] - [r, c]) ** 2).sum(axis=1)))
            seed = tuple(hits[k] + [r0, c0])
            break
    if seed is None:
        return [], None
    _, labels = cv2.connectedComponents(passable, connectivity=8)
    reach = labels == labels[seed]
    # Distance (cells) to the nearest unknown cell; outside the grid counts as unknown.
    known = np.pad((grid >= 0).astype(np.uint8), 1)
    d_unknown = cv2.distanceTransform(known, cv2.DIST_L2, 3)[1:-1, 1:-1]
    clear = reach & (grid >= 0) & (grid <= clear_max) & (d_unknown >= standoff_cells)
    clear_rc = np.argwhere(clear)
    goals = []
    if not len(clear_rc):
        return goals, seed
    for cells, size in find_frontiers(grid, free_max, min_cells):
        if not reach[cells[:, 0], cells[:, 1]].any():
            continue          # only seen through a gap the robot can't drive through
        cy, cx = cells.mean(axis=0)
        d2 = (clear_rc[:, 0] - cy) ** 2 + (clear_rc[:, 1] - cx) ** 2
        i = int(np.argmin(d2))
        if d2[i] > search_cells ** 2:
            continue          # no clear spot near it to look from
        goals.append((tuple(clear_rc[i]), (cy, cx), size))
    return goals, seed


class Explorer(Node):

    def __init__(self):
        super().__init__('explorer')
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        self.base_frame = p('base_frame', 'base_link').value
        self.min_cells = int(p('min_frontier_cells', 10).value)         # 0.5 m of frontier at 5 cm
        self.free_max = int(p('free_cost_max', 40).value)               # cells cheaper than this count as free
        self.pass_max = int(p('passable_cost_max', 98).value)           # below "inscribed": the centre may cross
        self.clear_max = int(p('goal_cost_max', 15).value)              # goals: well away from obstacles
        self.standoff = float(p('goal_standoff_m', 0.3).value)          # goals this far inside known space
        self.goal_search = float(p('goal_search_radius_m', 0.8).value)  # ... and within this of the frontier
        self.min_goal_dist = float(p('min_goal_distance_m', 0.4).value)
        self.blacklist_radius = float(p('blacklist_radius_m', 0.75).value)
        self.visited_radius = float(p('visited_radius_m', 0.6).value)
        self.goal_timeout = float(p('goal_timeout_s', 60.0).value)
        # Nav2 answers a goal request at once; one it never answers (server restarting, request
        # lost) counts as rejected after this long, so exploring doesn't stall waiting for it.
        self.response_timeout = float(p('goal_response_timeout_s', 10.0).value)
        # Driving home may take a while, but a result that never comes counts as a failed trip
        # home after this long (0 = wait for it forever).
        self.return_timeout = float(p('return_timeout_s', 300.0).value)
        self.low_battery = float(p('low_battery_fraction', 0.2).value)
        self.rotate_speed = float(p('rotate_speed_rps', 0.7).value)     # 0 = leave the controller alone
        self.return_home = bool(p('return_home', True).value)
        self.home_tolerance = float(p('home_tolerance_m', 0.5).value)
        # A robot briefly boxed in (stale obstacles, tracking recovery) sees no reachable
        # frontier; only call exploring finished after this long with none.
        self.finish_confirm = float(p('finish_confirm_s', 20.0).value)
        # When a cliff sensor stops the robot (stairs, or deep carpet that reads like a drop),
        # add a keep-out zone there so neither exploring nor navigation tries it again.
        self.auto_keepout = bool(p('auto_keepout', True).value) and SensorState is not None
        self.cliff_seen = {}         # sensor name -> monotonic time it last read a cliff
        # Bumper hits leave a thin, temporary keep-out strip on the touched surface so the planner
        # stops sending the robot back into a wall the camera can't see (the last ~30 cm, or below
        # the obstacle band). Margin ~ the Roomba's radius: it may still drive right up to the
        # wall (it's a vacuum); it just won't plan through that spot. Never saved with the map.
        self.bump_marks = bool(p('bump_marks', True).value) and SensorState is not None
        self.bump_mark_ttl = float(p('bump_mark_ttl_s', 600.0).value)
        self.bump_mark_margin = float(p('bump_mark_margin_m', 0.15).value)
        self.bump_mark_spacing = float(p('bump_mark_spacing_m', 0.15).value)
        self.bump_mark_width = float(p('bump_mark_width_m', 0.3).value)
        # Also mark when a front light bumper reads at least this (0-4095; 0 = switches only): thin
        # railing posts set it off right before, or instead of, the switches.
        self.bump_light_min = int(p('bump_light_signal', 0).value)
        # A frontier whose way in keeps hitting something the camera can't see (railing posts,
        # glass) is skipped after this many bumps (0 = never).
        self.bumps_per_goal = int(p('bumps_per_goal', 2).value)
        # Areas that failed (blocked, bumped, timed out) are skipped, but a door shut or a chair
        # moved since: before giving up, retry them - again only after some goal has been reached.
        self.retry_failed = bool(p('retry_failed_before_finish', True).value)
        self.reached_at_retry = -1
        self.goal_bumps = 0
        self.bumped = False
        self.bump_points = []        # (x, y, monotonic) of recent marks

        self.grid = None
        self.robot_mode = None
        self.battery = None
        self.tracking_ok = True
        self.state = 'idle'          # idle | exploring | returning | done | stopped
        self.message = 'Ready'
        self.logged_message = None   # last message written to the log
        self.home = None
        self.goal = None             # (x, y, sent monotonic)
        self.goal_handle = None
        self.goal_seq = 0
        self.blacklist = []
        self.reached = 0
        self.failed = 0
        self.frontier_count = 0
        self.saved_rotate = None
        self.sending = None          # (seq, sent monotonic) of the goal request Nav2 hasn't answered
        self.no_frontier_since = None
        self.resume_after_home = False   # went home because nothing was reachable: look again there
        self.final_check = False

        self.tf_buffer = Buffer()
        # spin_thread=True would spin this node from a second executor, which races the
        # action client's wait set and kills the TF thread (as in roomba_web).
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_params = self.create_client(GetParameters, '/controller_server/get_parameters')
        self.set_params = self.create_client(SetParameters, '/controller_server/set_parameters')

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self._on_grid, latched)
        self.create_subscription(String, '/roomba/mode', self._on_mode, latched)
        self.create_subscription(BatteryState, '/battery_state', self._on_battery, 2)
        self.create_subscription(String, '/vslam_filter/status', self._on_filter, 2)
        self.create_subscription(String, '~/command', self._on_command, 10)
        self.pub_zone_cmd = self.create_publisher(String, '/perception/ignore_zones/command', 10)
        if self.auto_keepout or self.bump_marks:
            self.create_subscription(SensorState, '/roomba/sensors', self._on_sensors, 10)
        self.pub_status = self.create_publisher(String, '~/status', latched)
        self.create_timer(1.0, self._tick)
        self._publish_status()

    # ---- inputs ----
    def _on_grid(self, msg):
        self.grid = msg

    def _on_mode(self, msg):
        previous, self.robot_mode = self.robot_mode, msg.data
        if previous in ('safe', 'full') and msg.data == 'passive' and self.auto_keepout:
            # Sensors arrive at ~5 Hz: decide once the frames around the stop are in.
            timer = None

            def check():
                self.destroy_timer(timer)
                self._cliff_keepout()
            timer = self.create_timer(0.6, check)
        if self.state in ('exploring', 'returning') and msg.data not in ('safe', 'full'):
            self._stop(f'Stopped: the Roomba switched to {msg.data} mode (cliff, wheel drop or carpet?)')

    def _on_sensors(self, msg):
        now = time.monotonic()
        for name in CLIFF_ANGLES:
            if getattr(msg, name, False):
                self.cliff_seen[name] = now
        if msg.wheel_drop_left or msg.wheel_drop_right:
            self.cliff_seen['wheel_drop'] = now
        light = self.bump_light_min > 0 and any(
            getattr(msg, k, 0) >= self.bump_light_min for k, a in LIGHT_ANGLES.items() if abs(a) < 1.0)
        bumped = bool(msg.bump_left or msg.bump_right) or light
        if bumped and not self.bumped and self.bump_marks and self.robot_mode in ('safe', 'full') \
                and not (msg.wheel_drop_left or msg.wheel_drop_right):
            self._mark_bump(bump_bearing(msg, self.bump_light_min or 200), now)
            self._bumped_on_goal()
        self.bumped = bumped

    def _bumped_on_goal(self):
        if self.state != 'exploring' or self.goal is None or self.bumps_per_goal <= 0:
            return
        self.goal_bumps += 1
        if self.goal_bumps >= self.bumps_per_goal:
            self.get_logger().warn(f'Bumped {self.goal_bumps} times on the way to a frontier '
                                   '(railing or glass?): skipping it')
            self._blacklist(self.goal[0], self.goal[1])
            self.failed += 1
            self._cancel_goal()
            self.message = "Kept bumping into something the camera can't see: trying another area"

    def _mark_bump(self, bearing, now):
        pose = self._pose()
        if pose is None:
            return
        points = bump_zone(pose[0], pose[1], pose[2], bearing, width=self.bump_mark_width)
        cx = sum(p[0] for p in points) / 4.0
        cy = sum(p[1] for p in points) / 4.0
        self.bump_points = [b for b in self.bump_points if now - b[2] < self.bump_mark_ttl]
        if any(math.hypot(cx - bx, cy - by) < self.bump_mark_spacing for bx, by, _ in self.bump_points):
            return   # already marked here
        self.bump_points.append((cx, cy, now))
        name = time.strftime('bump_%H%M%S') + f'_{len(self.bump_points)}'
        self.pub_zone_cmd.publish(String(data=json.dumps(
            {'op': 'add', 'kind': 'keepout', 'name': name, 'points': points,
             'margin': self.bump_mark_margin, 'ttl_s': self.bump_mark_ttl})))
        self.get_logger().info(f'Bumper hit at {math.degrees(bearing):+.0f} deg: marked {name} '
                               f'for {self.bump_mark_ttl / 60:.0f} min')

    def _cliff_keepout(self):
        """The firmware just left safe mode: if a cliff sensor fired, keep out of that spot."""
        now = time.monotonic()
        if now - self.cliff_seen.get('wheel_drop', -1e9) < 3.0:
            return   # lifted or a wheel dropped: every cliff sensor reads a drop, not a place to avoid
        fired = [n for n in CLIFF_ANGLES if now - self.cliff_seen.get(n, -1e9) < 1.5]
        if not fired:
            return   # charger or a mode request
        pose = self._pose()
        if pose is None:
            self.get_logger().warn(f'Cliff ({", ".join(fired)}) but no robot position: no keep-out zone added')
            return
        bearing = math.atan2(sum(math.sin(CLIFF_ANGLES[n]) for n in fired),
                             sum(math.cos(CLIFF_ANGLES[n]) for n in fired))
        points = cliff_zone(pose[0], pose[1], pose[2], bearing)
        name = time.strftime('cliff_%H%M%S')
        self.pub_zone_cmd.publish(String(data=json.dumps(
            {'op': 'add', 'kind': 'keepout', 'name': name, 'points': points})))
        self.get_logger().info(f'Cliff ({", ".join(fired)}): added keep-out zone {name}')
        self.message = f'Cliff sensor stop: added keep-out zone {name}'

    def _on_battery(self, msg):
        if msg.percentage == msg.percentage:   # not NaN
            self.battery = msg.percentage

    def _on_filter(self, msg):
        try:
            self.tracking_ok = json.loads(msg.data).get('state', 'idle') == 'idle'
        except ValueError:
            pass

    def _on_command(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == 'start':
            self._start()
        elif cmd == 'stop':
            if self.state in ('exploring', 'returning'):
                self._stop('Stopped')
        elif cmd == 'home':
            if self.home is not None:
                self._go_home('Returning to the start point')
        else:
            self.get_logger().warn(f'Unknown command "{msg.data}" (start | stop | home)')

    # ---- state changes ----
    def _start(self):
        if self.state in ('exploring', 'returning'):
            return
        if self.robot_mode not in ('safe', 'full'):
            self._set('stopped', 'Enable driving (safe mode) first')
            return
        pose = self._pose()
        if pose is None:
            self._set('stopped', 'Robot position unknown (no map -> base TF yet)')
            return
        if not self.nav.server_is_ready():
            self._set('stopped', 'Navigation is not running')
            return
        if self.battery is not None and self.battery < self.low_battery:
            self._set('stopped', f'Battery at {self.battery * 100:.0f}%: charge the Roomba first')
            return
        if self.home is None or self.state in ('done', 'idle'):
            self.home = pose[:2]
            self.blacklist = []
            self.visited = []
            self.reached = self.failed = 0
            self.reached_at_retry = -1
        self.no_frontier_since = None
        self.resume_after_home = self.final_check = False
        self._slow_turns(True)
        self._set('exploring', 'Looking for unmapped areas…')
        self.get_logger().info(f'Exploration started; home at ({pose[0]:.2f}, {pose[1]:.2f})')

    def _stop(self, why):
        self.resume_after_home = self.final_check = False
        self._cancel_goal()
        self._slow_turns(False)
        self._set('stopped', why)
        self.get_logger().info(why)

    def _go_home(self, why, resume=False):
        self.resume_after_home = resume
        if self.home is None or not self.return_home:
            self._finish('Exploration finished')
            return
        self._cancel_goal()
        self._set('returning', why)
        self._send_goal(self.home[0], self.home[1], None)

    def _finish(self, why):
        self._slow_turns(False)
        self._set('done', why)
        self.get_logger().info(why)

    def _set(self, state, message):
        self.state, self.message = state, message
        self._publish_status()

    # ---- main loop ----
    def _tick(self):
        now = time.monotonic()
        if self.sending is not None and now - self.sending[1] > self.response_timeout:
            self._no_response()
        elif self.state == 'exploring':
            if self.battery is not None and self.battery < self.low_battery:
                self._go_home(f'Battery at {self.battery * 100:.0f}%: returning to the start point')
            elif not self.tracking_ok:
                self.message = 'Paused: visual tracking is recovering'
            elif self.goal is not None and now - self.goal[2] > self.goal_timeout:
                self.get_logger().warn('Frontier goal timed out; skipping it')
                self._blacklist(self.goal[0], self.goal[1])
                self.failed += 1
                self._cancel_goal()
            elif self.goal is None and self.sending is None:
                self._next_frontier()
        elif (self.state == 'returning' and self.goal is not None and self.return_timeout > 0
              and now - self.goal[2] > self.return_timeout):
            self.get_logger().warn(f'Driving back to the start point timed out after {self.return_timeout:.0f} s')
            self._cancel_goal()
            self._returned(False)
        self._publish_status()

    def _next_frontier(self):
        if self.grid is None:
            self.message = 'Waiting for the costmap…'
            return
        pose = self._pose()
        if pose is None:
            self.message = 'Waiting for the robot position…'
            return
        info = self.grid.info
        grid = np.asarray(self.grid.data, dtype=np.int16).reshape(info.height, info.width)
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        robot_rc = (int((pose[1] - oy) / res), int((pose[0] - ox) / res))
        goals, seed = frontier_goals(grid, robot_rc, self.free_max, self.pass_max, self.clear_max,
                                     self.min_cells, self.standoff / res, self.goal_search / res)
        if seed is None:
            self.message = 'Robot is not in free space on the costmap; waiting…'
            return
        best = None
        candidates = 0
        for (gy, gx), (cy, cx), size in goals:
            x, y = ox + (gx + 0.5) * res, oy + (gy + 0.5) * res
            fx, fy = ox + (cx + 0.5) * res, oy + (cy + 0.5) * res
            if self._blacklisted(x, y) or self._visited(x, y):
                continue
            d = math.hypot(x - pose[0], y - pose[1])
            if d < self.min_goal_dist:
                # Already standing here and the camera still can't fill it in: don't come back.
                self.visited.append((x, y))
                continue
            candidates += 1
            # Prefer near frontiers, but let bigger ones win over slightly nearer small ones.
            score = d - 0.01 * size
            if best is None or score < best[0]:
                best = (score, x, y, math.atan2(fy - y, fx - x))   # look at the unknown
        self.frontier_count = candidates
        if best is None:
            now = time.monotonic()
            if self.no_frontier_since is None:
                self.no_frontier_since = now
            if now - self.no_frontier_since < self.finish_confirm:
                self.message = 'No reachable unmapped areas right now; checking again…'
                return
            self.no_frontier_since = None
            if self.retry_failed and self.blacklist and self.reached != self.reached_at_retry:
                self.reached_at_retry = self.reached
                self.get_logger().info(f'No unmapped areas left except {len(self.blacklist)} that failed '
                                       'earlier: trying those again')
                self.message = 'Retrying areas that failed earlier…'
                self.blacklist = []
                return
            if self.final_check:
                self.final_check = False
                self._finish('Exploration finished: no reachable unmapped areas left'
                             + (' - back at the start point' if self._near_home() else ''))
                return
            self._go_home('Exploration finished: no reachable unmapped areas left', resume=True)
            return
        self.no_frontier_since = None
        _, x, y, yaw = best
        self.message = f'Heading to an unmapped area {math.hypot(x - pose[0], y - pose[1]):.1f} m away'
        self._send_goal(x, y, yaw)

    def _blacklist(self, x, y):
        self.blacklist.append((x, y))

    def _blacklisted(self, x, y):
        return any(math.hypot(x - bx, y - by) < self.blacklist_radius for bx, by in self.blacklist)

    def _visited(self, x, y):
        return any(math.hypot(x - vx, y - vy) < self.visited_radius for vx, vy in self.visited)

    # ---- navigation ----
    def _send_goal(self, x, y, yaw):
        pose = self._pose()
        if yaw is None:
            yaw = pose[2] if pose is not None else 0.0
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.goal_seq += 1
        seq = self.goal_seq
        self.goal = (x, y, time.monotonic())
        self.goal_bumps = 0
        self.sending = (seq, self.goal[2])
        future = self.nav.send_goal_async(goal)
        future.add_done_callback(lambda f: self._on_goal_response(f, seq))

    def _on_goal_response(self, future, seq):
        if self.sending is not None and self.sending[0] == seq:
            self.sending = None
        handle = future.result()
        if seq != self.goal_seq:          # superseded, cancelled or given up on meanwhile
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            self.get_logger().warn('Navigation rejected the goal')
            self._refused()
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self._on_result(f, seq))

    def _no_response(self):
        """Nav2 never answered the goal request: stop waiting and count the goal as rejected."""
        seq, self.sending = self.sending[0], None
        if seq != self.goal_seq:
            return    # cancelled or superseded meanwhile: nothing left to do
        self.goal_seq += 1   # an answer that still comes is then for a superseded goal (cancelled)
        self.get_logger().warn(f'Navigation did not answer the goal request within '
                               f'{self.response_timeout:.0f} s; giving up on it')
        self._refused()

    def _refused(self):
        """The current goal was rejected (or never answered): skip that spot, or end the trip home."""
        goal, self.goal = self.goal, None
        if self.state == 'returning':
            self._returned(False)
        elif goal is not None:
            self._blacklist(goal[0], goal[1])

    def _on_result(self, future, seq):
        if seq != self.goal_seq:
            return
        status = future.result().status
        goal, self.goal, self.goal_handle = self.goal, None, None
        if self.state == 'returning':
            # Nav2 can report success without the robot moving (a one-pose plan while it was
            # boxed in at a wall), so check that it really is back.
            self._returned(status == GoalStatus.STATUS_SUCCEEDED and self._near_home())
            return
        if self.state != 'exploring':
            return
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.reached += 1
            if goal is not None:
                # Looked from here once; if the camera couldn't fill it in, don't come back.
                self.visited.append((goal[0], goal[1]))
        elif goal is not None:
            # Unreachable or no progress: don't try this spot again.
            self.failed += 1
            self._blacklist(goal[0], goal[1])

    def _returned(self, home):
        """The trip home ended, at the start point (home) or not."""
        why = self.message   # why it went home (finished, low battery, request)
        if self.resume_after_home:
            # The map looks different from elsewhere: check once more before finishing.
            self.resume_after_home = False
            self.final_check = True
            self._set('exploring', 'Back at the start point; checking for unmapped areas again…' if home
                      else 'Could not reach the start point; checking for unmapped areas again from here…')
            return
        self._finish(f'{why} - back at the start point' if home
                     else f'{why} - could not drive back to the start point')

    def _cancel_goal(self):
        self.goal_seq += 1
        handle, self.goal_handle, self.goal = self.goal_handle, None, None
        if handle is not None:
            handle.cancel_goal_async()

    def _near_home(self):
        pose = self._pose()
        return (pose is not None and self.home is not None
                and math.hypot(pose[0] - self.home[0], pose[1] - self.home[1]) <= self.home_tolerance)

    def _pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _slow_turns(self, on):
        """Lower the controller's in-place turn speed while exploring; restore it after."""
        if self.rotate_speed <= 0.0 or not self.set_params.service_is_ready():
            return
        if on:
            if self.saved_rotate is None and self.get_params.service_is_ready():
                req = GetParameters.Request(names=[ROTATE_PARAM])
                self.get_params.call_async(req).add_done_callback(self._on_saved_rotate)
            self._set_rotate(self.rotate_speed)
        elif self.saved_rotate is not None:
            self._set_rotate(self.saved_rotate)

    def _on_saved_rotate(self, future):
        try:
            values = future.result().values
            if values and values[0].type == ParameterType.PARAMETER_DOUBLE:
                self.saved_rotate = values[0].double_value
        except Exception as e:  # noqa: BLE001 - restoring is best effort
            self.get_logger().warn(f'Could not read {ROTATE_PARAM}: {e}')

    def _set_rotate(self, value):
        param = Parameter(name=ROTATE_PARAM,
                          value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)))
        self.set_params.call_async(SetParameters.Request(parameters=[param]))

    def _publish_status(self):
        if self.message != self.logged_message and self.state in ('exploring', 'returning'):
            self.get_logger().info(f'[{self.state}] {self.message}')
        self.logged_message = self.message
        self.pub_status.publish(String(data=json.dumps({
            'state': self.state, 'message': self.message, 'reached': self.reached,
            'failed': self.failed, 'frontiers': self.frontier_count,
            'home': [round(v, 2) for v in self.home] if self.home else None,
            'goal': [round(self.goal[0], 2), round(self.goal[1], 2)] if self.goal else None})))


def main():
    rclpy.init()
    node = Explorer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node._cancel_goal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
