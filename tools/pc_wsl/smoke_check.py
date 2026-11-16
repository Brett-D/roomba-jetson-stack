"""Checks a running smoke stack: Nav2 lifecycle nodes active, map / keep-out / driver topics, then
drives NavigateToPose goals through the test house and reports each result."""
import argparse
import math
import sys
import time

import rclpy
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav2_msgs.msg import CostmapFilterInfo
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

LIFECYCLE = ['map_server', 'controller_server', 'planner_server', 'behavior_server', 'bt_navigator',
             'velocity_smoother', 'collision_monitor']


class Check(Node):
    def __init__(self):
        super().__init__('smoke_check')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.got = {}
        self.create_subscription(OccupancyGrid, '/rtabmap/map_filled', lambda m: self.mark('map_filled', m), latched)
        self.create_subscription(CostmapFilterInfo, '/costmap_filter_info', lambda m: self.mark('filter_info', m), latched)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', lambda m: self.mark('global_costmap', m), 1)
        self.wheel = []
        self.create_subscription(Odometry, '/wheel/odom', lambda m: self.wheel.append(time.time()), 50)
        self.odom = None
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def mark(self, k, m):
        self.got.setdefault(k, m)

    def on_odom(self, m):
        self.odom = (m.pose.pose.position.x, m.pose.pose.position.y)

    def spin_for(self, secs):
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def state(self, name):
        cli = self.create_client(GetState, f'/{name}/get_state')
        try:
            if not cli.wait_for_service(timeout_sec=1.0):
                return 'no service'
            fut = cli.call_async(GetState.Request())
            end = time.time() + 2.0
            while not fut.done() and time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
            return fut.result().current_state.label if fut.done() else 'timeout'
        finally:
            self.destroy_client(cli)

    def go(self, x, y, timeout):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0
        t0 = time.time()
        fut = self.nav.send_goal_async(goal)
        while not fut.done() and time.time() - t0 < 10:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not fut.done() or not fut.result().accepted:
            return 'rejected', time.time() - t0
        res = fut.result().get_result_async()
        while not res.done() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not res.done():
            return 'timeout', time.time() - t0
        return {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}.get(res.result().status, str(res.result().status)), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goals', default='6.0,2.0;0.0,0.0', help='x,y;x,y in map (through the door at x=4)')
    ap.add_argument('--loops', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=120.0)
    ap.add_argument('--wait-active', type=float, default=60.0)
    ap.add_argument('--driver', action='store_true', help='also check /wheel/odom from roomba_driver')
    a = ap.parse_args()
    rclpy.init()
    n = Check()
    t0 = time.time()
    states = {}
    while time.time() - t0 < a.wait_active:
        states = {k: n.state(k) for k in LIFECYCLE}
        if all(v == 'active' for v in states.values()):
            break
        n.spin_for(1.0)
    print(f'lifecycle after {time.time() - t0:.1f}s: {states}', flush=True)
    ok = all(v == 'active' for v in states.values())
    n.spin_for(3.0)
    print(f'topics: map_filled={"map_filled" in n.got} filter_info={"filter_info" in n.got} '
          f'global_costmap={"global_costmap" in n.got}', flush=True)
    ok &= all(k in n.got for k in ('map_filled', 'filter_info', 'global_costmap'))
    if a.driver:
        k = len(n.wheel)
        n.spin_for(5.0)
        rate = (len(n.wheel) - k) / 5.0
        print(f'/wheel/odom rate {rate:.1f} Hz', flush=True)
        ok &= rate > 20
    if not n.nav.wait_for_server(timeout_sec=20):
        print('navigate_to_pose server missing', flush=True)
        sys.exit(1)
    goals = [tuple(float(v) for v in g.split(',')) for g in a.goals.split(';')]
    for loop in range(a.loops):
        for (x, y) in goals:
            status, dt = n.go(x, y, a.timeout)
            pose = n.odom or (float('nan'), float('nan'))
            err = math.hypot(pose[0] - x, pose[1] - y)
            print(f'loop {loop} goal ({x:.1f},{y:.1f}): {status} in {dt:.1f}s, end error {err:.2f} m', flush=True)
            ok &= status == 'SUCCEEDED'
    print('SMOKE OK' if ok else 'SMOKE FAILED', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
