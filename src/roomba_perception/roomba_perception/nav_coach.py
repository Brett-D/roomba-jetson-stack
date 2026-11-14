#!/usr/bin/env python3
"""Turns Nav2's velocity commands into instructions a person can follow.

A person carrying the camera acts as the robot. Nav2's controller output (cmd_vel_nav)
passes the collision monitor (cmd_vel); this node classifies the final command into
forward / bear left / bear right / turn left / turn right / back up / stop, holds each
instruction briefly to avoid flicker, and reports goal progress and results.
With robot_mode true (on the Roomba) the same states are worded as what the robot is doing.

Subscribes: /cmd_vel_nav, /cmd_vel (geometry_msgs/Twist), /collision_monitor_state,
            /perception/obstacles (stop zone check), /navigate_to_pose/_action/status|feedback
Publishes:  /nav_coach/instruction (std_msgs/String JSON)
"""
import json
import time

import numpy as np
import rclpy
import rclpy.executors
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav2_msgs.msg import CollisionMonitorState
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


def _quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

INSTRUCTIONS = {
    'idle': ('', 'Tap the map to set a goal'),
    'forward': ('⬆', 'Go forward'),
    'bear_left': ('↖', 'Forward, bear left'),
    'bear_right': ('↗', 'Forward, bear right'),
    'turn_left': ('⟲', 'Turn left'),
    'turn_right': ('⟳', 'Turn right'),
    'back': ('⬇', 'Back up slowly'),
    'wait': ('⏸', 'Wait…'),
    'stop_obstacle': ('✋', 'STOP – obstacle ahead'),
    'arrived': ('✅', 'You have arrived'),
    'failed': ('⚠', 'No path found'),
    'canceled': ('⏹', 'Navigation stopped'),
}

# The robot drives itself: describe what it is doing instead of instructing a person.
ROBOT_TEXT = {
    'idle': 'Tap the map to send the robot',
    'forward': 'Driving forward',
    'bear_left': 'Driving, bearing left',
    'bear_right': 'Driving, bearing right',
    'turn_left': 'Turning left',
    'turn_right': 'Turning right',
    'back': 'Backing up',
    'wait': 'Planning…',
    'stop_obstacle': 'Stopped – obstacle ahead',
    'arrived': 'Arrived',
    'failed': 'Could not reach the goal',
    'canceled': 'Navigation stopped',
}


class NavCoach(Node):

    def __init__(self):
        super().__init__('nav_coach')
        p = self.declare_parameter
        self.min_lin = float(p('min_linear_mps', 0.04).value)
        self.min_ang = float(p('min_angular_rps', 0.15).value)
        self.bear_ratio = float(p('bear_angular_rps', 0.35).value)
        self.hold_s = float(p('hold_s', 0.6).value)
        self.texts = ({k: (arrow, ROBOT_TEXT[k]) for k, (arrow, _) in INSTRUCTIONS.items()}
                      if bool(p('robot_mode', False).value) else INSTRUCTIONS)

        self.cmd_nav = (0.0, Twist())
        self.cmd_out = (0.0, Twist())
        self.cm_stop = (0.0, False)
        self.active = False
        self.result = (0.0, None)
        self.distance = None
        self.current = 'idle'
        self.since = 0.0
        self.pending = (None, 0.0)

        self.create_subscription(Twist, '/cmd_vel_nav', lambda m: self._set('cmd_nav', m), 10)
        self.create_subscription(Twist, '/cmd_vel', lambda m: self._set('cmd_out', m), 10)
        self.create_subscription(CollisionMonitorState, '/collision_monitor_state', self._on_cm, 10)
        # Obstacles in the stop zone directly ahead (the controller may simply stop
        # without commanding anything, so the collision monitor alone would stay silent).
        self.stop_dist = float(p('stop_distance_m', 0.45).value)
        self.stop_half_width = float(p('stop_half_width_m', 0.22).value)
        self.stop_min_points = int(p('stop_min_points', 15).value)
        self.base_frame = p('base_frame', 'camera_link').value
        self.ahead = (0.0, False)
        self.tf_buffer = Buffer()
        # spin_thread=True spins this node from a second executor; under load it races the main
        # executor's wait set and the main loop stops delivering callbacks (seen live: the
        # obstacle stream froze and the collision monitor stopped the robot). The main
        # executor also processes /tf here, so lookups must not block.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.create_subscription(PointCloud2, '/perception/obstacles', self._on_obstacles, 2)
        self.create_subscription(GoalStatusArray, '/navigate_to_pose/_action/status', self._on_status, 10)
        self.create_subscription(NavigateToPose.Impl.FeedbackMessage, '/navigate_to_pose/_action/feedback',
                                 self._on_feedback, 10)
        self.pub = self.create_publisher(String, '/nav_coach/instruction', 10)
        self.create_timer(0.2, self._tick)

    def _set(self, name, msg):
        setattr(self, name, (time.monotonic(), msg))

    def _on_cm(self, msg):
        self.cm_stop = (time.monotonic(), msg.action_type == CollisionMonitorState.STOP)

    def _on_obstacles(self, msg):
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, Time())
        except TransformException:
            return
        pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 3) if msg.width else np.zeros((0, 3))
        r = tf.transform.rotation
        t = tf.transform.translation
        R = _quat_to_matrix(r.x, r.y, r.z, r.w)
        base = pts @ R.T + np.array([t.x, t.y, t.z])
        ahead = np.count_nonzero((base[:, 0] > 0.0) & (base[:, 0] < self.stop_dist)
                                 & (np.abs(base[:, 1]) < self.stop_half_width))
        self.ahead = (time.monotonic(), ahead >= self.stop_min_points)

    def _on_status(self, msg):
        if not msg.status_list:
            return
        latest = max(msg.status_list, key=lambda s: (s.goal_info.stamp.sec, s.goal_info.stamp.nanosec))
        st = latest.status
        was_active = self.active
        self.active = st in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING)
        if was_active and not self.active:
            code = {GoalStatus.STATUS_SUCCEEDED: 'arrived', GoalStatus.STATUS_CANCELED: 'canceled',
                    GoalStatus.STATUS_ABORTED: 'failed'}.get(st)
            self.result = (time.monotonic(), code)
            self.distance = None

    def _on_feedback(self, msg):
        self.distance = msg.feedback.distance_remaining

    def _classify(self, now):
        if not self.active:
            t, code = self.result
            return code if code and now - t < 15.0 else 'idle'
        t_cm, stop = self.cm_stop
        if stop and now - t_cm < 1.0:
            return 'stop_obstacle'
        t, cmd = self.cmd_out
        vx, wz = (cmd.linear.x, cmd.angular.z) if now - t <= 0.6 else (0.0, 0.0)
        turning_in_place = abs(vx) < self.min_lin and abs(wz) >= self.min_ang
        t_ahead, blocked = self.ahead
        if blocked and now - t_ahead < 1.0 and not turning_in_place and vx > -self.min_lin:
            return 'stop_obstacle'
        if now - t > 0.6:
            return 'wait'
        t_nav, nav = self.cmd_nav
        if abs(vx) < self.min_lin and abs(wz) < self.min_ang and now - t_nav < 0.6 \
                and (abs(nav.linear.x) >= self.min_lin or abs(nav.angular.z) >= self.min_ang):
            return 'stop_obstacle'  # collision monitor zeroed a real command
        if vx <= -self.min_lin:
            return 'back'
        if vx >= self.min_lin:
            if wz > self.bear_ratio:
                return 'bear_left'
            if wz < -self.bear_ratio:
                return 'bear_right'
            return 'forward'
        if wz >= self.min_ang:
            return 'turn_left'
        if wz <= -self.min_ang:
            return 'turn_right'
        return 'wait'

    def _tick(self):
        now = time.monotonic()
        code = self._classify(now)
        # Stops and results apply immediately; movement changes must persist hold_s (no flicker).
        immediate = code in ('stop_obstacle', 'arrived', 'failed', 'canceled', 'idle')
        if code != self.current:
            if immediate:
                self.current, self.since = code, now
            elif self.pending[0] != code:
                self.pending = (code, now)
            elif now - self.pending[1] >= self.hold_s:
                self.current, self.since = code, now
        else:
            self.pending = (None, 0.0)
        arrow, text = self.texts[self.current]
        self.pub.publish(String(data=json.dumps({
            'code': self.current, 'arrow': arrow, 'text': text, 'active': self.active,
            'distance_m': round(self.distance, 2) if self.distance is not None else None,
        })))


def main():
    rclpy.init()
    node = NavCoach()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
