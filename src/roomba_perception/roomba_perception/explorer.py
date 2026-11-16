#!/usr/bin/env python3
"""Frontier exploration: drives the robot to the edge of the known map until nothing is left.

A frontier is free, known space next to unknown space in Nav2's global costmap (which
already includes RTAB-Map's grid, live obstacles and keep-out zones). Frontier cells are
grouped; only groups connected to the robot through space Nav2 can plan the robot's centre
through (below "inscribed") count; groups longer than frontier_split_m are cut into pieces. Each
gets a goal parked in clear, known space
(goal_standoff_m from the unknown, away from obstacles) near it, facing it, so the controller
isn't asked to drive into clutter at the map's edge; where there is no such spot (narrow
passages), the best reachable cell near it that is still below inscribed (goal_cost_weight
trades distance against cost). The explorer drives to the nearest one and repeats. At the first
spot it reaches (away from the dock, which the light bumpers would take for an obstacle while
turning on it), and before deciding nothing is left, it looks around (Nav2 spin in steps); at other
spots it reached it turns to face the unknown it came to see (Nav2's goal checker ignores the
heading). Timeouts run on the node clock (sim time under use_sim_time). Spots
reached are not revisited; goals that fail are skipped until the map around them opens up
(a door opened, a person walked away) or, once, before finishing (when it also drops its bump
marks: someone walking into the robot leaves them too). Frontiers it can't reach
right now get one closer look from the nearest reachable spot before it gives up on them.
When no frontier is left it returns to where it started and checks once more there; it ends
there (or says why not). Nav2 answers that never come don't stall it: a goal request not
answered within goal_response_timeout_s counts as rejected (a late acceptance is cancelled), and
a trip home with no result within return_timeout_s as failed.

Safety: exploring stops if the Roomba leaves safe/full mode (cliff, wheel drop, charger:
the firmware drops it to passive), and it returns home when the battery runs low. While
visual tracking is recovering it pauses. During exploration the controller's in-place
turn speed can be lowered (rotate_speed_rps) to keep visual tracking. With auto_keepout,
a cliff-sensor stop (stairs, or deep carpet that reads like a drop; not a wheel drop) adds
a keep-out zone just ahead of the robot, in exploring, navigation or manual driving. Bump and
cliff marks never cover the robot itself (mark_clear_radius_m around it stays free), so it can
still turn and back away.
Drop-offs floor_obstacles reports (a stair opening the map shows as floor) are remembered: no
frontier or goal within drop_clear_m of one, and no way through.
A bump with a person seen in front is not marked; a bump mark lasts bump_mark_short_ttl_s
unless the same spot is bumped again; boxed in by its own marks (the area it can reach tiny
while the map has more), it removes them one at a time.

Subscribes: /global_costmap/costmap (OccupancyGrid), /roomba/mode (String), /battery_state,
            /roomba/sensors, /vslam_filter/status (String JSON),
            /perception/drops (PointCloud2),
            /perception/person_detector/detections_3d (Detection3DArray),
            ~/command (String: "start" | "stop" | "home")
Publishes:  ~/status (String JSON, transient local), /perception/ignore_zones/command,
            ~/markers (MarkerArray, map frame, transient local; only while someone listens): how it
            decides where to go - frontier groups, the spots it considered with distance and score
            (or why it skipped them), the chosen goal with an arrow from where it was chosen, spots
            to look from, failed and visited spots, keep-out marks it sent, home;
            ~/drops (PointCloud2, map frame, 1 Hz): the drop-off cells it remembers, for the
            global costmap's drop_layer (marked again right after a costmap clear)
Uses:       navigate_to_pose and spin actions, /controller_server parameters, TF map -> base frame
"""
import json
import math
import time

import cv2
import numpy as np
import rclpy
import rclpy.executors
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import BatteryState, PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

ROTATE_PARAM = 'FollowPath.rotate_to_heading_angular_vel'
# Cliff sensor directions on the Roomba (rad, left positive) for automatic keep-out zones.
CLIFF_ANGLES = {'cliff_left': 1.05, 'cliff_front_left': 0.35, 'cliff_front_right': -0.35, 'cliff_right': -1.05}
# Light-bumper (IR proximity) directions, used to tell where on the bumper a hit was.
LIGHT_ANGLES = {'light_bump_left_signal': 1.2, 'light_bump_front_left_signal': 0.65,
                'light_bump_center_left_signal': 0.2, 'light_bump_center_right_signal': -0.2,
                'light_bump_front_right_signal': -0.65, 'light_bump_right_signal': -1.2}
# ~/markers colours (r, g, b, a; exact in float32) and goal labels.
GREEN, BLUE, GREY, PURPLE = (0.0, 1.0, 0.0, 1.0), (0.0, 0.5, 1.0, 1.0), (0.5, 0.5, 0.5, 0.75), (0.75, 0.0, 1.0, 1.0)
ORANGE, RED_AREA, CYAN = (1.0, 0.5, 0.0, 0.75), (1.0, 0.0, 0.0, 0.25), (0.0, 1.0, 1.0, 0.75)
YELLOW, RED, WHITE = (1.0, 1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)
MAGENTA = (1.0, 0.0, 1.0, 0.75)
OPTION_COLORS = {'chosen': GREEN, 'option': BLUE, 'skipped': GREY}
PAUSED = 'Paused: visual tracking is recovering'
LOOK_REJECTS = 30   # look-around turns Nav2 may reject in a row (its behaviours still inactive) before it gives up
GOAL_LABELS = {'frontier': 'goal: unmapped area', 'peek': 'goal: closer look', 'home': 'goal: start point'}
DROP_CELL_M = 0.05  # lattice (m) of the drop-off cells it remembers, in the map frame
DROP_TF_WAIT_S = 1.0  # a drop cloud waits this long (s after its stamp) for the transform at its time
MAX_PENDING_DROPS = 10  # drop clouds waiting for their transform at most (the oldest go first)

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


def _nearest(rc, cy, cx, limit_cells):
    """Index of the cell in rc (N x 2 row, col) nearest to (cy, cx), or None beyond limit_cells."""
    if not len(rc):
        return None
    d2 = (rc[:, 0] - cy) ** 2 + (rc[:, 1] - cx) ** 2
    i = int(np.argmin(d2))
    return None if d2[i] > limit_cells ** 2 else i


def _best_spot(rc, penalty, cy, cx, limit_cells):
    """Index of the cell in rc with the lowest distance (cells) to (cy, cx) plus its cost penalty,
    within limit_cells, or None."""
    if not len(rc):
        return None
    d2 = (rc[:, 0] - cy) ** 2 + (rc[:, 1] - cx) ** 2
    score = np.where(d2 <= limit_cells ** 2, np.sqrt(d2) + penalty, np.inf)
    i = int(np.argmin(score))
    return None if score[i] == np.inf else i


def frontier_pieces(cells, split_cells):
    """A frontier group's cells (np.argwhere order) cut into ceil(extent / split_cells) pieces of
    equal count along its longer side (split_cells <= 0 or a shorter group: whole)."""
    n = len(cells)
    span_r = int(cells[:, 0].max() - cells[:, 0].min())
    span_c = int(cells[:, 1].max() - cells[:, 1].min())
    ext = max(span_r, span_c) + 1
    if split_cells <= 0 or ext <= split_cells:
        return [cells]
    k = min(math.ceil(ext / split_cells), n)
    if span_r > span_c:
        cells = cells[np.lexsort((cells[:, 1], cells[:, 0]))]
    else:
        cells = cells[np.lexsort((cells[:, 0], cells[:, 1]))]
    return [cells[j * n // k:(j + 1) * n // k] for j in range(k)]


def reachable(grid, passable, seed):
    """The passable cells connected to seed (bool array). 8-connected, except that a diagonal step
    between two lethal cells (keep-out marks, walls) is no way through: Nav2's planner (navfn) only
    spreads to the 4 neighbours, and in Isaac run 9 a ring of bump marks with diagonal gaps trapped
    the robot while the explorer saw a way out. Past inscribed corners it still steps diagonally
    (that inflation changes as the robot gets closer and sees better)."""
    n, labels = cv2.connectedComponents(passable, connectivity=4)
    lethal = grid >= 100
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    # diagonal neighbours (r, c) - (r + 1, c +/- 1), unless both corner cells are lethal
    for a, b, blocked in ((labels[:-1, :-1], labels[1:, 1:], lethal[:-1, 1:] & lethal[1:, :-1]),
                          (labels[:-1, 1:], labels[1:, :-1], lethal[:-1, :-1] & lethal[1:, 1:])):
        ok = (a > 0) & (b > 0) & (a != b) & ~blocked
        for x, y in set(zip(a[ok].tolist(), b[ok].tolist())):
            rx, ry = root(x), root(y)
            if rx != ry:
                parent[rx] = ry
    mine = root(labels[seed])
    same = np.array([root(k) == mine for k in range(n)], dtype=bool)
    same[0] = False
    return same[labels]


def frontier_goals(grid, robot_rc, free_max, pass_max, clear_max, min_cells, standoff_cells, search_cells,
                   cost_weight=0.0, peek_cells=0.0, groups=None, split_cells=0.0):
    """Reachable frontier groups with a goal cell parked in clear, known space next to each.

    grid: int costmap (-1 unknown, 0..100; 99 inscribed and 100 lethal: Nav2 never plans the
    robot's centre there). robot_rc: (row, col) of the robot. A group's goal is the reachable
    cell nearest to it with cost <= clear_max and standoff_cells from unknown space, within
    search_cells; failing that (narrow places), with cost_weight > 0, the reachable cell below
    inscribed (cost <= pass_max, half the standoff) with the lowest distance + cost * cost_weight
    (cells). Groups the robot can't reach get, with peek_cells > 0, a spot to look at them from:
    chosen the same way, within peek_cells. Groups longer than split_cells (> 0) are cut into
    pieces (frontier_pieces), a spot each: one spot failing (a railing the camera misses in the
    way) doesn't drop a room's whole far side. groups: find_frontiers(grid, free_max, min_cells)
    if the caller has it already.
    Returns (goals, seed, peeks, reach_cells): goals and peeks are lists of (goal_rc, piece
    centroid_rc, group size), seed the robot's free cell (None if the robot isn't next to passable
    space), reach_cells how many cells it can reach (see reachable).
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
        return [], None, [], 0
    reach = reachable(grid, passable, seed)
    # Distance (cells) to the nearest unknown cell; outside the grid counts as unknown.
    known = np.pad((grid >= 0).astype(np.uint8), 1)
    d_unknown = cv2.distanceTransform(known, cv2.DIST_L2, 3)[1:-1, 1:-1]
    clear = reach & (grid >= 0) & (grid <= clear_max) & (d_unknown >= standoff_cells)
    clear_rc = np.argwhere(clear)
    if cost_weight > 0:
        spot = reach & (grid >= 0) & (grid <= pass_max) & (d_unknown >= standoff_cells / 2)
        spot_rc = np.argwhere(spot)
        penalty = grid[spot] * cost_weight
    else:
        spot_rc, penalty = clear_rc[:0], np.zeros(0)
    goals, peeks = [], []
    for cells, size in (find_frontiers(grid, free_max, min_cells) if groups is None else groups):
        for piece in frontier_pieces(cells, split_cells):
            cy, cx = piece.mean(axis=0)
            if reach[piece[:, 0], piece[:, 1]].any():
                out, limit = goals, search_cells
            elif peek_cells > 0:
                out, limit = peeks, peek_cells     # only seen through a gap the robot can't drive through
            else:
                continue
            i = _nearest(clear_rc, cy, cx, limit)
            if i is not None:
                out.append((tuple(clear_rc[i]), (cy, cx), size))
                continue
            i = _best_spot(spot_rc, penalty, cy, cx, limit)
            if i is not None:
                out.append((tuple(spot_rc[i]), (cy, cx), size))
            # else: no spot near it to look from
    return goals, seed, peeks, int(np.count_nonzero(reach))


def drop_blocked(grid, res, ox, oy, drops, clear_cells):
    """grid with the known cells within clear_cells of a drop-off cell (map-frame lattice
    DROP_CELL_M) made lethal: no frontier, goal or way through there. drops: sorted (i, j)."""
    grid = grid.copy()
    h, w = grid.shape
    k = int(clear_cells)
    k2 = clear_cells * clear_cells
    offsets = np.array([(dr, dc) for dr in range(-k, k + 1) for dc in range(-k, k + 1) if dr * dr + dc * dc <= k2],
                       dtype=np.int64).reshape(-1, 2)
    for i, j in drops:
        c = math.floor(((i + 0.5) * DROP_CELL_M - ox) / res)
        r = math.floor(((j + 0.5) * DROP_CELL_M - oy) / res)
        rr, cc = r + offsets[:, 0], c + offsets[:, 1]
        ok = (rr >= 0) & (cc >= 0) & (rr < h) & (cc < w)
        rr, cc = rr[ok], cc[ok]
        known = grid[rr, cc] >= 0
        grid[rr[known], cc[known]] = 100
    return grid


def _rotation(q):
    """Rotation matrix rows (x, y) of a quaternion (x, y, z, w)."""
    x, y, z, w = q
    return ((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)))


def _mark_centre(points):
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def _duration_text(ttl):
    return f'{ttl / 60:.0f} min' if ttl >= 120 else f'{ttl:.0f} s'


def _cloud_xyz(msg):
    """x, y, z (float32 fields) of every point of a PointCloud2, as an N x 3 float32 array."""
    off = {f.name: f.offset for f in msg.fields if f.datatype == 7}
    if not all(k in off for k in ('x', 'y', 'z')):
        return np.zeros((0, 3), np.float32)
    fmt = '>f4' if msg.is_bigendian else '<f4'
    dt = np.dtype({'names': ['x', 'y', 'z'], 'formats': [fmt] * 3, 'offsets': [off['x'], off['y'], off['z']],
                   'itemsize': msg.point_step})
    data = bytes(msg.data)
    rows = [np.frombuffer(data[r * msg.row_step:r * msg.row_step + msg.width * msg.point_step], dtype=dt)
            for r in range(msg.height)]
    a = np.concatenate(rows) if rows else np.zeros(0, dt)
    return np.stack([a['x'], a['y'], a['z']], axis=1)


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
        # No such spot (a narrow passage): the reachable cell below inscribed with the lowest distance
        # + cost x this (cells per cost unit; 0 = skip such frontiers).
        self.cost_weight = float(p('goal_cost_weight', 0.05).value)
        # Frontiers it can't reach get one look from the nearest reachable spot within this.
        self.peek_search = float(p('peek_search_m', 3.0).value)         # 0 = never
        # Frontier groups longer than this get a goal spot per piece of about this length (a
        # room's far side seen from the doorway); 0 = one spot per group.
        self.frontier_split = float(p('frontier_split_m', 2.0).value)
        # Drop-offs floor_obstacles reported (/perception/drops, e.g. a stair opening the camera
        # looks down into while the map shows floor) are remembered; frontier and goal cells within
        # this of one are left out, so it neither aims into the void nor reaches through it
        # (0 = ignore drops). A cell counts after drop points in this many clouds.
        self.drop_clear = float(p('drop_clear_m', 0.3).value)
        self.drop_min_hits = int(p('drop_min_hits', 2).value)
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
        # Trips home when nothing is left: a failed one (a TF hiccup, a bump mark in the way) is
        # tried again, up to this many in all, before it says it could not drive back.
        self.home_attempts = max(1, int(p('return_home_attempts', 3).value))
        # A robot briefly boxed in (stale obstacles, tracking recovery) sees no reachable
        # frontier; only call exploring finished after this long with none.
        self.finish_confirm = float(p('finish_confirm_s', 20.0).value)
        # Look around (Nav2 spin, in steps) before the first goal and before deciding nothing is
        # left: started facing a wall or with the dock behind it, the camera has seen little.
        self.look_steps = int(p('look_around_steps', 4).value)          # 0 = never
        self.look_step = math.radians(float(p('look_around_step_deg', 90.0).value))
        self.look_timeout = float(p('look_around_timeout_s', 20.0).value)   # time allowance per 90 deg of turn
        # Pause after each look-around turn (s): visual tracking catches up while it stands.
        self.look_pause = float(p('look_around_pause_s', 0.0).value)
        # Nav2's goal checker ignores the final heading here (yaw_goal_tolerance 6.28), so at a spot it
        # reached it turns to face the unknown it came to see when it is off by more than this
        # (0 = never).
        self.face_turn_min = math.radians(float(p('face_frontier_min_deg', 45.0).value))
        # Failed spots are skipped until the map within blacklist_radius_m of them opens up by at
        # least this many cells (a door opened, a person walked away: 100+; one flickering railing
        # post with its inflation: ~37; 0 = never).
        self.reopen_cells = int(p('blacklist_reopen_cells', 40).value)
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
        # A mark not confirmed by another bump at the same spot (within bump_mark_spacing_m) lasts
        # only this long: people walking into the robot leave single hits (0 = all bump_mark_ttl_s).
        self.bump_mark_short_ttl = float(p('bump_mark_short_ttl_s', 60.0).value)
        self.bump_mark_margin = float(p('bump_mark_margin_m', 0.15).value)
        # A confirmed mark (bumped twice at the same spot: a railing, glass) keeps the robot's
        # centre this far off: radius 0.17 + 0.1 m clearance (+ 1 cm) - keep-out isn't inflated.
        self.bump_mark_confirmed_margin = float(p('bump_mark_confirmed_margin_m', 0.28).value)
        self.bump_mark_spacing = float(p('bump_mark_spacing_m', 0.15).value)
        self.bump_mark_width = float(p('bump_mark_width_m', 0.3).value)
        # Bump and cliff marks leave the robot's own footprint (plus clearance) free, so Nav2 can
        # still turn it and back it away (a mark over the robot blocked every move). Footprint
        # 0.18 m + a mask/costmap cell diagonal (the grids don't line up) + slack: 0.28 m.
        self.mark_clear = float(p('mark_clear_radius_m', 0.28).value)
        # ... and carry an "avoid" band this wide beyond their margin: keep-out cells aren't
        # inflated, so the planner routed the robot's centre along their edge, where the
        # controller's footprint check then refused to move ("collision ahead").
        self.mark_soft = float(p('mark_soft_band_m', 0.22).value)
        # Also mark when a front light bumper reads at least this (0-4095; 0 = switches only): thin
        # railing posts set it off right before, or instead of, the switches.
        self.bump_light_min = int(p('bump_light_signal', 0).value)
        # A frontier whose way in keeps hitting something the camera can't see (railing posts,
        # glass) is skipped after this many bumps (0 = never).
        self.bumps_per_goal = int(p('bumps_per_goal', 2).value)
        # A bump while a person was detected in front within this distance (in the last
        # person_bump_window_s) is a person's leg: not marked, not counted (0 = mark all).
        self.person_bump_dist = float(p('person_bump_dist_m', 1.2).value)
        self.person_bump_window = float(p('person_bump_window_s', 3.0).value)
        # Boxed in by its own bump marks (Nav2 plans 4-connected, so a ring of marks with only
        # diagonal gaps closes it in): when the area it can reach is below this while the map has
        # more free space elsewhere and bump marks lie within trap_radius_m, it removes them one at
        # a time (unconfirmed first, rows along a railing last), trap_wait_s apart (0 = never).
        self.trap_area = float(p('trap_area_m2', 1.0).value)
        self.trap_radius = float(p('trap_radius_m', 1.5).value)
        self.trap_wait = float(p('trap_wait_s', 3.0).value)
        # Areas that failed (blocked, bumped, timed out) are skipped, but a door shut or a chair
        # moved since: before giving up, retry them - again only after some goal has been reached.
        self.retry_failed = bool(p('retry_failed_before_finish', True).value)
        # Learned goal ranking (docs/learned-skills.md) exists only in the Rust explorer.
        if p('goal_ranking', 'rule').value != 'rule':
            self.get_logger().warn('goal_ranking is only supported by the Rust explorer; using the rule')
        p('goal_ranking_model', '')
        # So is the learned unstuck skill (unstuck_skill: true).
        if p('unstuck_skill', False).value:
            self.get_logger().warn('unstuck_skill is only supported by the Rust explorer; exploring without it')
        for name, default in (('unstuck_time_limit_s', 15.0), ('unstuck_call_timeout_s', 25.0),
                              ('unstuck_no_progress_s', 30.0), ('unstuck_clearance_m', 0.30)):
            p(name, default)
        self.reached_at_retry = -1
        self.goal_bumps = 0
        self.bumped = False
        self.bump_points = []        # (x, y, monotonic) of recent marks
        self.persons = []            # (time, x, y) in the map frame of people the detector saw lately
        self.drop_hits = {}          # drop-off cell (DROP_CELL_M lattice, map frame) -> clouds with drop points there
        self.pending_drops = []      # (stamp ns, frame, points) of drop clouds waiting for their transform
        self.drops_skipped = 0
        self.trap_next = -math.inf   # no trap-breaking mark removal before this time
        self.look_until = -math.inf  # next look-around turn not before this time (look_around_pause_s)

        self.grid = None
        self.robot_mode = None
        self.battery = None
        self.tracking_ok = True
        self.state = 'idle'          # idle | exploring | returning | done | stopped
        self.message = 'Ready'
        self.logged_message = None   # last message written to the log
        self.home = None
        self.goal = None             # (x, y, sent (ROS time), kind 'frontier' | 'peek' | 'home', yaw)
        self.goal_message = ''       # what it said when it sent the goal (shown again after a pause)
        self.turn = None             # turn (rad) to face the unknown at the spot just reached
        self.goal_handle = None
        self.goal_seq = 0
        self.blacklist = []          # (x, y, passable cells around it when it failed)
        self.visited = []
        self.peeked = []
        self.reached = 0
        self.failed = 0
        self.frontier_count = 0
        self.saved_rotate = None
        self.sending = None          # (seq, sent monotonic) of the goal request Nav2 hasn't answered
        self.no_frontier_since = None
        self.resume_after_home = False   # went home because nothing was reachable: look again there
        self.home_failures = 0       # trips home that failed since something was last reached
        self.look = None             # [steps left, stop as soon as a frontier shows up] while looking around
        self.looked_here = False
        self.look_pending = False    # look around once the first spot is reached
        self.look_rejects = 0        # look-around turns rejected in a row
        self.spin = None             # (seq, sent monotonic, accepted) of the look-around turn in progress
        self.spin_allowance = 0.0    # its time allowance (s)
        self.spin_handle = None
        self.spin_seq = 0
        # ~/markers: what the last frontier search saw (resolution, frontier cells, spots
        # considered (x, y, kind, label), spots to look from), where the goal was chosen from,
        # the keep-out marks sent (kind, name, points, monotonic, ttl s or 0) and the last
        # markers published.
        self.viz = None
        self.goal_from = None
        self.marks_sent = []
        self.markers_last = None

        self.tf_buffer = Buffer()
        # spin_thread=True would spin this node from a second executor, which races the
        # action client's wait set and kills the TF thread (as in roomba_web).
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spinner = ActionClient(self, Spin, 'spin')
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
        self.create_subscription(PointCloud2, '/perception/drops', self._on_drops, 2)
        self.create_subscription(Detection3DArray, '/perception/person_detector/detections_3d', self._on_persons, 5)
        self.pub_status = self.create_publisher(String, '~/status', latched)
        # How it decides where to go (only built while someone listens): see _marker_specs.
        self.pub_markers = self.create_publisher(MarkerArray, '~/markers', latched)
        # The drop-offs it remembers, every second, for the global costmap's drop_layer: a costmap
        # clear (navigation recoveries) wipes that layer, and this marks it again right after.
        self.pub_drops = self.create_publisher(PointCloud2, '~/drops', 2)
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
        now = self._now()
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
            bearing = bump_bearing(msg, self.bump_light_min or 200)
            if self._person_in_front(now):
                self.get_logger().info(f'Bumper hit at {math.degrees(bearing):+.0f} deg with a person in front: '
                                       'not marked')
            else:
                self._mark_bump(bearing, now)
                self._bumped_on_goal()
        self.bumped = bumped

    def _on_persons(self, msg):
        """People the person detector reports: kept person_bump_window_s in the map frame."""
        points = [(d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z)
                  for d in msg.detections]
        self._persons(msg.header.frame_id, points)

    def _persons(self, frame, points):
        if self.person_bump_dist <= 0 or not points:
            return
        tf = self._lookup(frame)
        if tf is None:
            return
        (tx, ty, _), q = tf
        now = self._now()
        r = _rotation(q)
        for px, py, pz in points:
            x = r[0][0] * px + r[0][1] * py + r[0][2] * pz + tx
            y = r[1][0] * px + r[1][1] * py + r[1][2] * pz + ty
            self.persons.append((now, x, y))
        self.persons = [p for p in self.persons if now - p[0] <= self.person_bump_window][-100:]

    def _person_in_front(self, now):
        """Whether someone was seen in front of the robot within person_bump_dist_m lately."""
        if self.person_bump_dist <= 0:
            return False
        self.persons = [p for p in self.persons if now - p[0] <= self.person_bump_window]
        if not self.persons:
            return False
        pose = self._pose()
        if pose is None:
            return False
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        return any(math.hypot(px - x, py - y) <= self.person_bump_dist and (px - x) * c + (py - y) * s > 0.0
                   for _, px, py in self.persons)

    def _on_drops(self, msg):
        stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        self._drops(msg.header.frame_id, stamp, _cloud_xyz(msg))

    def _drops(self, frame, stamp, points):
        """Drop points floor_obstacles reports (float32 N x 3), in `frame` at `stamp` (ns, the depth
        frame's time): remembered as drop-off cells (a cell counts once per cloud), placed with the
        transform at that time; a cloud whose transform isn't there yet waits for it (up to
        DROP_TF_WAIT_S), then is skipped."""
        if self.drop_clear <= 0 or not len(points):
            return
        self.pending_drops.append((stamp, frame, points))
        del self.pending_drops[:-MAX_PENDING_DROPS]
        self._apply_drops()

    def _apply_drops(self):
        """The waiting drop clouds whose transform is there now."""
        if not self.pending_drops:
            return
        now = self._now()
        keep = []
        for stamp, frame, points in self.pending_drops:
            tf = self._lookup(frame, stamp)
            if tf is not None:
                self._remember_drops(tf, points)
            elif now - stamp * 1e-9 > DROP_TF_WAIT_S:
                self.drops_skipped += 1
                if self.drops_skipped == 1:
                    self.get_logger().warn('Drop-off points without a transform at their time: skipped')
            else:
                keep.append((stamp, frame, points))
        self.pending_drops = keep

    def _remember_drops(self, tf, points):
        (tx, ty, _), q = tf
        r = _rotation(q)
        p = np.asarray(points, dtype=np.float32).astype(np.float64)
        x = r[0][0] * p[:, 0] + r[0][1] * p[:, 1] + r[0][2] * p[:, 2] + tx
        y = r[1][0] * p[:, 0] + r[1][1] * p[:, 1] + r[1][2] * p[:, 2] + ty
        ok = np.isfinite(x) & np.isfinite(y)
        cells = set(zip(np.floor(x[ok] / DROP_CELL_M).astype(np.int64).tolist(),
                        np.floor(y[ok] / DROP_CELL_M).astype(np.int64).tolist()))
        for c in cells:
            self.drop_hits[c] = self.drop_hits.get(c, 0) + 1

    def _drop_cells(self):
        """The drop-off cells (DROP_CELL_M lattice), sorted."""
        return sorted(c for c, n in self.drop_hits.items() if n >= self.drop_min_hits)

    def _lookup(self, frame, stamp=0):
        """Transform frame -> map at stamp (ns; 0: the latest): ((x, y, z), (qx, qy, qz, qw)) or None."""
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, frame, Time(nanoseconds=stamp))
        except TransformException:
            return None
        tr, q = t.transform.translation, t.transform.rotation
        return (tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w)

    def _bumped_on_goal(self):
        if self.state != 'exploring' or self.goal is None or self.bumps_per_goal <= 0:
            return
        self.goal_bumps += 1
        if self.goal_bumps >= self.bumps_per_goal:
            self.get_logger().warn(f'Bumped {self.goal_bumps} times on the way to a frontier '
                                   '(railing or glass?): skipping it')
            self._blacklist(self.goal[0], self.goal[1], reopen=False)
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
        # A bump at the same spot again (within bump_mark_spacing_m of a mark's centre) confirms
        # the mark there: something the camera can't see (railing, glass), not someone walking by.
        # Hits around the robot at one spot (people's legs as it turns) are different spots.
        active = self._bump_marks(now)

        def near(m, d):
            mx, my = _mark_centre(m[2])
            return math.hypot(cx - mx, cy - my) < d
        same = [m[1] for m in active if near(m, self.bump_mark_spacing)]
        for name in same:
            self._confirm_mark(name, now, pose)
        self.bump_points = [b for b in self.bump_points if now - b[2] < self.bump_mark_ttl]
        if same:
            return   # already marked here
        ttl = self.bump_mark_ttl if self.bump_mark_short_ttl <= 0 else self.bump_mark_short_ttl
        self.bump_points.append((cx, cy, now))
        name = time.strftime('bump_%H%M%S') + f'_{len(self.bump_points)}'
        cmd = {'op': 'add', 'kind': 'keepout', 'name': name, 'points': points,
               'margin': self.bump_mark_margin, 'ttl_s': ttl}
        self._clear_robot(cmd, pose)
        self.pub_zone_cmd.publish(String(data=json.dumps(cmd)))
        self._sent_mark('bump', name, points, now, ttl, cmd, self.bump_mark_short_ttl <= 0)
        self.get_logger().info(f'Bumper hit at {math.degrees(bearing):+.0f} deg: marked {name} '
                               f'for {_duration_text(ttl)}')

    def _confirm_mark(self, name, now, pose):
        """Bumped again at mark `name`: something solid (a railing): make it last bump_mark_ttl_s
        from now and keep planned paths bump_mark_confirmed_margin_m from it (the robot's free
        circle moves to where it stands now)."""
        k = next(i for i, m in enumerate(self.marks_sent) if m[1] == name)
        kind, _, points, sent, ttl, cmd, confirmed = self.marks_sent[k]
        if confirmed:
            return
        if cmd is not None:
            self.pub_zone_cmd.publish(String(data=json.dumps({'op': 'delete', 'name': name})))
            cmd = dict(cmd)
            cmd['margin'] = max(cmd['margin'], self.bump_mark_confirmed_margin)
            cmd['ttl_s'] = self.bump_mark_ttl
            self._clear_robot(cmd, pose)
            self.pub_zone_cmd.publish(String(data=json.dumps(cmd)))
            sent, ttl = now, self.bump_mark_ttl
        self.marks_sent[k] = (kind, name, points, sent, ttl, cmd, True)
        if cmd is not None:
            self.get_logger().info(f'Bumped at {name} again: keeping it for {_duration_text(self.bump_mark_ttl)}')

    def _sent_mark(self, kind, name, points, now, ttl, cmd=None, confirmed=True):
        self.marks_sent.append((kind, name, points, now, ttl, cmd, confirmed))
        del self.marks_sent[:-50]

    def _clear_robot(self, cmd, pose):
        """Keep-out command: leave the robot's footprint free (it is standing there), and keep
        planned paths a robot radius away from the mark (a soft band beyond its margin)."""
        if self.mark_clear > 0:
            cmd['clear'] = [round(pose[0], 3), round(pose[1], 3), self.mark_clear]
        if self.mark_soft > 0:
            cmd['soft'] = self.mark_soft

    def _cliff_keepout(self):
        """The firmware just left safe mode: if a cliff sensor fired, keep out of that spot."""
        now = self._now()
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
        cmd = {'op': 'add', 'kind': 'keepout', 'name': name, 'points': points}
        self._clear_robot(cmd, pose)
        self.pub_zone_cmd.publish(String(data=json.dumps(cmd)))
        self._sent_mark('cliff', name, points, now, 0.0)
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
            self.peeked = []
            self.reached = self.failed = 0
            self.reached_at_retry = -1
        self.no_frontier_since = None
        self.resume_after_home = self.looked_here = False
        self.home_failures = 0
        # Look around once at the first spot reached, not here: it may be on the dock.
        self.look_pending = self.look_steps > 0
        self.look = self.turn = None
        self._slow_turns(True)
        self._set('exploring', 'Looking for unmapped areas…')
        self.get_logger().info(f'Exploration started; home at ({pose[0]:.2f}, {pose[1]:.2f})')

    def _stop(self, why):
        self.resume_after_home = False
        self._cancel_goal()
        self._cancel_spin()
        self._slow_turns(False)
        self._set('stopped', why)
        self.get_logger().info(why)

    def _go_home(self, why, resume=False):
        self.resume_after_home = resume
        self._cancel_spin()
        if self.home is None or not self.return_home:
            self._finish('Exploration finished')
            return
        self._cancel_goal()
        self._set('returning', why)
        self._send_goal(self.home[0], self.home[1], None, 'home')

    def _finish(self, why):
        self._cancel_spin()
        self._slow_turns(False)
        self._set('done', why)
        self.get_logger().info(why)

    def _set(self, state, message):
        self.state, self.message = state, message
        self._publish_status()

    # ---- main loop ----
    def _tick(self):
        now = self._now()
        self._apply_drops()
        if self.sending is not None and now - self.sending[1] > self.response_timeout:
            self._no_response()
        elif self.spin is not None and now - self.spin[1] > (
                self.spin_allowance + 5.0 if self.spin[2] else self.response_timeout):
            self.get_logger().warn('Looking around: the turn did not finish' if self.spin[2]
                                   else 'Looking around: navigation did not answer the turn request')
            self._cancel_spin()
            self._look_done()
        elif self.state == 'exploring':
            if self.battery is not None and self.battery < self.low_battery:
                self._go_home(f'Battery at {self.battery * 100:.0f}%: returning to the start point')
            elif not self.tracking_ok:
                self.message = PAUSED
            elif self.message == PAUSED:
                # Tracking is back: say what it is doing again (it kept the goal meanwhile).
                self.message = self.goal_message if self.goal is not None else 'Looking for unmapped areas…'
            elif self.goal is not None and now - self.goal[2] > self.goal_timeout:
                self.get_logger().warn('Frontier goal timed out; skipping it')
                self._blacklist(self.goal[0], self.goal[1])
                self.failed += 1
                self._cancel_goal()
            elif self.goal is None and self.sending is None and self.spin is None:
                self._next_frontier()
        elif (self.state == 'returning' and self.goal is not None and self.return_timeout > 0
              and now - self.goal[2] > self.return_timeout):
            self.get_logger().warn(f'Driving back to the start point timed out after {self.return_timeout:.0f} s')
            self._cancel_goal()
            self._returned(False)
        self._publish_status()
        self._publish_markers()
        self._publish_drops()

    def _publish_drops(self):
        """~/drops: every drop-off cell it remembers (centres, map frame), as a PointCloud2."""
        cells = self._drop_cells() if self.drop_clear > 0 else []
        if not cells:
            return
        pts = np.array([((i + 0.5) * DROP_CELL_M, (j + 0.5) * DROP_CELL_M, 0.0) for i, j in cells], dtype=np.float32)
        msg = PointCloud2(height=1, width=len(pts), is_dense=True, is_bigendian=False, point_step=12,
                          row_step=12 * len(pts))
        msg.header.frame_id, msg.header.stamp = self.map_frame, self.get_clock().now().to_msg()
        msg.fields = [PointField(name=n, offset=4 * k, datatype=PointField.FLOAT32, count=1) for k, n in enumerate('xyz')]
        msg.data = pts.tobytes()
        self.pub_drops.publish(msg)

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
        # Drop-offs it was told about stay out of bounds even where the map shows floor.
        drops = self._drop_cells() if self.drop_clear > 0 else []
        if drops:
            grid = drop_blocked(grid, res, ox, oy, drops, self.drop_clear / res)
        viz = self.pub_markers.get_subscription_count() > 0
        groups = find_frontiers(grid, self.free_max, self.min_cells) if viz else None
        goals, seed, peeks, reach_cells = frontier_goals(grid, robot_rc, self.free_max, self.pass_max, self.clear_max,
                                                         self.min_cells, self.standoff / res, self.goal_search / res,
                                                         self.cost_weight, self.peek_search / res, groups,
                                                         self.frontier_split / res)
        if seed is None:
            self.message = 'Robot is not in free space on the costmap; waiting…'
            return
        free_cells = int(np.count_nonzero((grid >= 0) & (grid <= self.free_max)))
        if self._break_trap(reach_cells, free_cells, res, pose):
            return
        if self.turn is not None:
            turn, self.turn = self.turn, None
            self.message = 'Turning to face the unmapped area…'
            self._send_spin(turn)
            return
        if self.look is not None and not self.look[1]:
            self.message = 'Looking around…'
            self._look_step()
            return
        self._reopen(grid, res, ox, oy)
        best = None
        candidates = 0
        options = []                    # for ~/markers: (x, y, kind, label)
        for (gy, gx), (cy, cx), size in goals:
            x, y = ox + (gx + 0.5) * res, oy + (gy + 0.5) * res
            fx, fy = ox + (cx + 0.5) * res, oy + (cy + 0.5) * res
            if self._blacklisted(x, y) or self._visited(x, y):
                options.append((x, y, 'skipped', 'failed earlier' if self._blacklisted(x, y) else 'looked from here'))
                continue
            d = math.hypot(x - pose[0], y - pose[1])
            if d < self.min_goal_dist:
                # Already standing here and the camera still can't fill it in: don't come back.
                self.visited.append((x, y))
                options.append((x, y, 'skipped', 'here already'))
                continue
            candidates += 1
            # Prefer near frontiers, but let bigger ones win over slightly nearer small ones.
            score = d - 0.01 * size
            options.append((x, y, 'option', f'{d:.1f} m, score {score:.2f}'))
            if best is None or score < best[0]:
                best = (score, x, y, math.atan2(fy - y, fx - x))   # look at the unknown
                chosen = len(options) - 1
        if viz:
            if best is not None:
                x, y, _, text = options[chosen]
                options[chosen] = (x, y, 'chosen', f'chosen: {text}')
            frontiers = []
            for cells, _ in groups:
                xs = (ox + (cells[:, 1] + 0.5) * res).tolist()
                ys = (oy + (cells[:, 0] + 0.5) * res).tolist()
                frontiers.append([(px, py, 0.0) for px, py in zip(xs, ys)])
            self.viz = (res, frontiers, options,
                        [(ox + (gx + 0.5) * res, oy + (gy + 0.5) * res) for (gy, gx), _, _ in peeks])
        self.frontier_count = candidates
        if best is not None:
            self.look = None
            self.no_frontier_since = None
            _, x, y, yaw = best
            self.message = f'Heading to an unmapped area {math.hypot(x - pose[0], y - pose[1]):.1f} m away'
            self._send_goal(x, y, yaw)
            return
        now = self._now()
        if self.no_frontier_since is None:
            self.no_frontier_since = now
        if self.look is None and not self.looked_here and self.look_steps > 0:
            self.look = [self.look_steps, True]
            self.message = 'No unmapped areas in view: looking around…'
        if self.look is not None:
            self._look_step()
            return
        peek = self._peek(peeks, pose, res, ox, oy)
        if peek is not None:
            d, x, y, yaw = peek
            self.no_frontier_since = None
            self.message = f"Taking a closer look at an area it can't reach yet, {d:.1f} m away"
            self._send_goal(x, y, yaw, 'peek')
            return
        if now - self.no_frontier_since < self.finish_confirm:
            self.message = 'No reachable unmapped areas right now; checking again…'
            return
        self.no_frontier_since = None
        marks = self._bump_marks(now, isolated=True)
        if self.retry_failed and (self.blacklist or marks) and self.reached != self.reached_at_retry:
            self.reached_at_retry = self.reached
            if self.blacklist:
                self.get_logger().info(f'No unmapped areas left except {len(self.blacklist)} that failed '
                                       'earlier: trying those again')
            self._drop_bump_marks(marks)
            self.message = 'Retrying areas that failed earlier…'
            self.blacklist = []
            return
        # Finish at the start point: go back there and check once more (the map looks different
        # from there), unless already there; a failed trip there is tried again up to
        # return_home_attempts in all: without the isolated bump marks (a person's; marks in a row
        # are a railing or glass the camera misses, and route the trip around it), and before the
        # last trip without any (people walking into it in a hallway leave rows too).
        why = 'Exploration finished: no reachable unmapped areas left'
        if self._near_home():
            self._finish(f'{why} - back at the start point')
        elif self.home_failures >= self.home_attempts:
            self._finish(f'{why} - could not drive back to the start point')
        else:
            if self.home_failures:
                self._drop_bump_marks(self._bump_marks(now, isolated=self.home_failures < self.home_attempts - 1))
            self._go_home(why, resume=True)

    def _bump_marks(self, now, isolated=False):
        """Bump marks still in force; isolated: only those with no other one within twice the mark
        width (a railing or glass wall gets bumped and marked along its length; someone walking
        into the robot leaves single marks)."""
        marks = [m for m in self.marks_sent if m[0] == 'bump' and (m[4] <= 0 or now - m[3] < m[4])]
        if not isolated:
            return marks
        centres = [(sum(p[0] for p in m[2]) / len(m[2]), sum(p[1] for p in m[2]) / len(m[2])) for m in marks]
        near = 2.0 * self.bump_mark_width
        return [m for i, (m, (x, y)) in enumerate(zip(marks, centres))
                if not any(math.hypot(x - cx, y - cy) < near for j, (cx, cy) in enumerate(centres) if j != i)]

    def _drop_bump_marks(self, marks):
        """Delete these bump marks: someone walking into the robot leaves them too, and they can
        close a hallway for bump_mark_ttl_s (what is really there gets bumped and marked again).
        Before finishing only isolated ones; before trying the trip home again isolated ones, and
        all of them before the last trip."""
        if not marks:
            return
        self._delete_marks(marks)
        self.bump_points = []
        self.get_logger().info(f'Removed {len(marks)} bump mark(s) (a person walking into the robot leaves '
                               'them too): checking again')

    def _break_trap(self, reach_cells, free_cells, res, pose):
        """Boxed in by its own bump marks (see trap_area_m2): removes the one that matters least
        (unconfirmed before confirmed, rows along a railing last, oldest first), at most one every
        trap_wait_s. True when it removed one."""
        now = self._now()
        if self.trap_area <= 0 or now < self.trap_next:
            return False
        reach = reach_cells * res * res
        free = free_cells * res * res
        if reach >= self.trap_area or free - reach < self.trap_area:
            return False
        active = self._bump_marks(now)
        near = [m for m in active
                if math.hypot(_mark_centre(m[2])[0] - pose[0], _mark_centre(m[2])[1] - pose[1]) <= self.trap_radius]
        if not near:
            return False
        pick = near[0]
        for m in near[1:]:
            a = (self._in_row(m, active) * 2 + m[6], m[3])
            b = (self._in_row(pick, active) * 2 + pick[6], pick[3])
            if a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]):
                pick = m
        self._delete_marks([pick])
        self.get_logger().info(f'Boxed in by its own bump marks ({reach:.1f} m2 to move in): removed {pick[1]}')
        self.message = 'Boxed in by its own bump marks: clearing a way out…'
        self.trap_next = now + self.trap_wait
        return True

    def _in_row(self, mark, active):
        """Whether a bump mark lies in a row of marks (a railing or glass wall marked along its
        length): it has neighbours within twice the mark width, all roughly on one line through it."""
        cx, cy = _mark_centre(mark[2])
        dirs = []
        for m in active:
            if m[1] == mark[1]:
                continue
            x, y = _mark_centre(m[2])
            if math.hypot(x - cx, y - cy) < 2.0 * self.bump_mark_width:
                dirs.append((x - cx, y - cy))
        if not dirs:
            return False
        return all(abs(a[0] * b[1] - a[1] * b[0]) <= 0.5 * math.hypot(*a) * math.hypot(*b)
                   for k, a in enumerate(dirs) for b in dirs[k + 1:])

    def _delete_marks(self, marks):
        """Send a delete for these marks and forget them."""
        for m in marks:
            self.pub_zone_cmd.publish(String(data=json.dumps({'op': 'delete', 'name': m[1]})))
        self.marks_sent = [m for m in self.marks_sent if m not in marks]

    def _peek(self, peeks, pose, res, ox, oy):
        """The nearest spot to take a closer look at a frontier it can't reach, not tried yet:
        (distance, x, y, yaw) or None."""
        best = None
        for (gy, gx), (cy, cx), size in peeks:
            x, y = ox + (gx + 0.5) * res, oy + (gy + 0.5) * res
            fx, fy = ox + (cx + 0.5) * res, oy + (cy + 0.5) * res
            if self._blacklisted(x, y) or any(math.hypot(x - px, y - py) < self.visited_radius
                                              for px, py in self.peeked):
                continue
            d = math.hypot(x - pose[0], y - pose[1])
            if d < self.min_goal_dist:
                self.peeked.append((x, y))    # looking from here already (looked around)
                continue
            if best is None or d < best[0]:
                best = (d, x, y, math.atan2(fy - y, fx - x))
        if best is not None:
            self.peeked.append((best[1], best[2]))
        return best

    def _look_step(self):
        """The next turn of the look-around, or its end."""
        if self.look[0] > 0 and self._now() < self.look_until:
            return   # standing still between turns (look_around_pause_s)
        if self.look[0] <= 0 or not self.spinner.server_is_ready():
            self._look_done()
            return
        self.look[0] -= 1
        self._send_spin(self.look_step)

    def _send_spin(self, angle):
        """A turn in place by angle (rad, + = left) with Nav2's spin behaviour; it may take
        look_around_timeout_s per 90 deg (gentle turns on carpet go at half the commanded rate or less)."""
        if not self.spinner.server_is_ready():
            return
        self.spin_seq += 1
        seq = self.spin_seq
        self.spin = (seq, self._now(), False)
        t = self.look_timeout * max(abs(angle) / (math.pi / 2), 1.0)
        self.spin_allowance = t
        allowance = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
        future = self.spinner.send_goal_async(Spin.Goal(target_yaw=float(angle), time_allowance=allowance))
        future.add_done_callback(lambda f: self._on_spin_response(f, seq))

    def _look_done(self):
        self.look = None
        self.looked_here = True

    def _on_spin_response(self, future, seq):
        handle = future.result()
        if self.spin is None or self.spin[0] != seq:     # given up on or cancelled meanwhile
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            self.spin = None
            self.look_rejects += 1
            if self.look_rejects == 1:
                self.get_logger().warn('Looking around: navigation rejected the turn')
            if self.look is not None:
                if self.look_rejects >= LOOK_REJECTS:
                    self._look_done()
                else:
                    self.look[0] += 1        # Nav2 still starting? that turn again next tick
            return
        self.look_rejects = 0
        self.spin = (seq, self.spin[1], True)
        self.spin_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self._on_spin_result(f, seq))

    def _on_spin_result(self, future, seq):
        if self.spin is None or self.spin[0] != seq:
            return
        self.spin, self.spin_handle = None, None
        if future.result().status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn('Looking around: the turn stopped early (obstacle?)')
            if self.look is not None:
                self._look_done()
        elif self.look is not None and self.look_pause > 0:
            self.look_until = self._now() + self.look_pause

    def _cancel_spin(self):
        self.look = self.turn = None
        handle, self.spin_handle, self.spin = self.spin_handle, None, None
        if handle is not None:
            handle.cancel_goal_async()

    def _patch(self, grid, res, ox, oy, x, y):
        """Passable cells (bool, square of blacklist_radius_m) around (x, y); outside the grid: not."""
        k = int(self.blacklist_radius / res)
        r0, c0 = int((y - oy) / res) - k, int((x - ox) / res) - k
        h, w = grid.shape
        out = np.zeros((2 * k + 1, 2 * k + 1), bool)
        rr0, rr1 = max(r0, 0), min(r0 + 2 * k + 1, h)
        cc0, cc1 = max(c0, 0), min(c0 + 2 * k + 1, w)
        if rr0 < rr1 and cc0 < cc1:
            sub = grid[rr0:rr1, cc0:cc1]
            out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = (sub >= 0) & (sub <= self.pass_max)
        return out

    def _grid_array(self):
        info = self.grid.info
        if len(self.grid.data) != info.width * info.height:
            return None
        return (np.asarray(self.grid.data, dtype=np.int16).reshape(info.height, info.width),
                info.resolution, info.origin.position.x, info.origin.position.y)

    def _blacklist(self, x, y, reopen=True):
        """Skip this spot until the map around it opens up (see _reopen); reopen False: only the
        retry before finishing (the camera can't see what was in the way)."""
        patch = None
        if reopen and self.grid is not None and self.reopen_cells > 0:
            g = self._grid_array()
            if g is not None:
                patch = self._patch(*g, x, y)
        self.blacklist.append((x, y, patch))

    def _reopen(self, grid, res, ox, oy):
        """Failed spots where the map has opened up since (a door opened, a person walked away):
        try them again."""
        if self.reopen_cells <= 0:
            return
        keep = []
        for x, y, patch in self.blacklist:
            if patch is not None:
                now_patch = self._patch(grid, res, ox, oy, x, y)
                if now_patch.shape == patch.shape and int(np.count_nonzero(now_patch & ~patch)) >= self.reopen_cells:
                    self.get_logger().info(f'The map opened up around an area that failed earlier '
                                           f'({x:.1f}, {y:.1f}): trying it again')
                    continue
            keep.append((x, y, patch))
        self.blacklist = keep

    def _blacklisted(self, x, y):
        return any(math.hypot(x - b[0], y - b[1]) < self.blacklist_radius for b in self.blacklist)

    def _visited(self, x, y):
        return any(math.hypot(x - vx, y - vy) < self.visited_radius for vx, vy in self.visited)

    # ---- navigation ----
    def _send_goal(self, x, y, yaw, kind='frontier'):
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
        self.goal = (x, y, self._now(), kind, yaw)
        self.goal_message = self.message
        self.goal_from = (pose[0], pose[1]) if pose is not None else None
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
            self.looked_here = False         # somewhere new: look around here before giving up
            if self.look_pending:
                self.look_pending = False
                self.look = [self.look_steps, False]
            elif goal is not None and self.face_turn_min > 0:
                pose = self._pose()
                if pose is not None:
                    d = goal[4] - pose[2]
                    d = math.atan2(math.sin(d), math.cos(d))
                    if abs(d) > self.face_turn_min:
                        self.turn = d        # face the unknown it came to see
            if goal is not None and goal[3] == 'peek':
                return
            self.reached += 1
            self.home_failures = 0
            if goal is not None:
                # Looked from here once; if the camera couldn't fill it in, don't come back.
                self.visited.append((goal[0], goal[1]))
        elif goal is not None and goal[3] != 'peek':
            # Unreachable or no progress: skip this spot (until the map around it opens up).
            self.failed += 1
            self._blacklist(goal[0], goal[1])

    def _returned(self, home):
        """The trip home ended, at the start point (home) or not."""
        why = self.message   # why it went home (finished, low battery, request)
        if self.resume_after_home:
            # The map looks different from here: check once more before finishing.
            self.resume_after_home = False
            self.home_failures = 0 if home else self.home_failures + 1
            self.looked_here = False
            self.no_frontier_since = None
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

    def _shutdown(self):
        """Cancel whatever the robot is doing for the explorer (goal, look-around turn)."""
        self._cancel_goal()
        self._cancel_spin()

    def _near_home(self):
        pose = self._pose()
        return (pose is not None and self.home is not None
                and math.hypot(pose[0] - self.home[0], pose[1] - self.home[1]) <= self.home_tolerance)

    def _now(self):
        """The node clock in seconds: ROS time (sim time under use_sim_time), so timeouts about the
        robot's progress don't fire early when a simulation runs slower than real time."""
        return self.get_clock().now().nanoseconds * 1e-9

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

    def _marker_specs(self):
        """How the explorer decides where to go, for ~/markers: the frontier groups of its last
        search, the spots it considered there (distance and score, the one it chose, or why it
        skipped them), spots to look at frontiers it can't reach from, failed spots (and when they
        are retried), visited spots, the keep-out marks it sent, home, and the goal with an arrow
        from where it was chosen. A list of (ns, id, type, position, scale, colour, points, text)."""
        specs, labels = [], []
        if self.viz is not None:
            res, frontiers, options, peeks = self.viz
            for k, points in enumerate(frontiers):
                specs.append(('frontiers', k, Marker.CUBE_LIST, (0.0, 0.0, 0.0), (res, res, 0.01), ORANGE, points, ''))
            for k, (x, y, kind, text) in enumerate(options):
                specs.append(('options', k, Marker.SPHERE, (x, y, 0.05), (0.12, 0.12, 0.12), OPTION_COLORS[kind], [], ''))
                labels.append((x, y, text))
            for k, (x, y) in enumerate(peeks):
                specs.append(('peeks', k, Marker.SPHERE, (x, y, 0.05), (0.12, 0.12, 0.12), PURPLE, [], ''))
                labels.append((x, y, "can't reach: look from here"))
        d = 2.0 * self.blacklist_radius
        for k, (x, y, patch) in enumerate(self.blacklist):
            specs.append(('failed', k, Marker.CYLINDER, (x, y, 0.01), (d, d, 0.02), RED_AREA, [], ''))
            labels.append((x, y, 'failed: retry if the map opens up' if patch is not None
                           else 'failed: retry before finishing'))
        if self.visited:
            specs.append(('visited', 0, Marker.SPHERE_LIST, (0.0, 0.0, 0.0), (0.08, 0.08, 0.08), GREY,
                          [(x, y, 0.05) for x, y in self.visited], ''))
        drops = self._drop_cells()
        if drops:
            specs.append(('drops', 0, Marker.CUBE_LIST, (0.0, 0.0, 0.0), (DROP_CELL_M, DROP_CELL_M, 0.01), MAGENTA,
                          [((i + 0.5) * DROP_CELL_M, (j + 0.5) * DROP_CELL_M, 0.01) for i, j in drops], ''))
        now = self._now()
        marks = [m for m in self.marks_sent if m[4] <= 0 or now - m[3] < m[4]]
        for k, (kind, name, points, *_) in enumerate(marks):
            specs.append(('marks', k, Marker.LINE_STRIP, (0.0, 0.0, 0.0), (0.03, 0.0, 0.0),
                          YELLOW if kind == 'bump' else RED, [(x, y, 0.03) for x, y in points + points[:1]], ''))
            labels.append((points[0][0], points[0][1], name))
        if self.home is not None:
            specs.append(('home', 0, Marker.CYLINDER, (self.home[0], self.home[1], 0.02), (0.3, 0.3, 0.04), CYAN, [], ''))
            labels.append((self.home[0], self.home[1], 'home'))
        if self.goal is not None:
            x, y, _, kind = self.goal[:4]
            if self.goal_from is not None:
                specs.append(('goal', 0, Marker.ARROW, (0.0, 0.0, 0.0), (0.03, 0.08, 0.1), GREEN,
                              [(self.goal_from[0], self.goal_from[1], 0.1), (x, y, 0.1)], ''))
            else:
                specs.append(('goal', 0, Marker.SPHERE, (x, y, 0.1), (0.2, 0.2, 0.2), GREEN, [], ''))
            labels.append((x, y, GOAL_LABELS[kind]))
        for k, (x, y, text) in enumerate(labels):
            specs.append(('labels', k, Marker.TEXT_VIEW_FACING, (x, y, 0.3), (0.0, 0.0, 0.1), WHITE, [], text))
        return specs

    def _publish_markers(self):
        """~/markers (after a DELETEALL), when they changed and someone listens."""
        if self.pub_markers.get_subscription_count() == 0:
            self.markers_last = None
            return
        specs = self._marker_specs()
        if specs == self.markers_last:
            return
        self.markers_last = specs
        stamp = self.get_clock().now().to_msg()
        clear = Marker(action=Marker.DELETEALL)
        clear.header.frame_id, clear.header.stamp = self.map_frame, stamp
        out = MarkerArray(markers=[clear])
        for ns, mid, kind, pos, scale, color, points, text in specs:
            m = Marker(ns=ns, id=mid, type=kind, action=Marker.ADD, text=text)
            m.header.frame_id, m.header.stamp = self.map_frame, stamp
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pos)
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
            m.color.r, m.color.g, m.color.b, m.color.a = color
            m.points = [Point(x=float(px), y=float(py), z=float(pz)) for px, py, pz in points]
            out.markers.append(m)
        self.pub_markers.publish(out)


def main():
    rclpy.init()
    node = Explorer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node._shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
