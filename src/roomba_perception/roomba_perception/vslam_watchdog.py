#!/usr/bin/env python3
"""Motion sanity filter and recovery supervisor for the visual-inertial odometry (Basalt or cuVSLAM).

The raw odometry (in its own `vslam_odom` frame) occasionally glitches: single-frame
jumps of metres, or IMU-only runaway after visual tracking fails. Nothing carrying the camera
can teleport, so this node is the only publisher of `odom -> base_frame` (camera_link when
hand-held, base_link on the robot) and only publishes physically plausible motion:

  - Every new pose's rotation is checked against the gyro integrated over the same interval
    (the gyro measures rotation directly and does not glitch), and its translation against
    speed, vertical-speed and acceleration limits scaled by the time since the last pose plus
    a jitter margin. Without IMU data a plain turn-rate limit is used instead. With wheel
    odometry the translation must also agree with the wheels to within a few centimetres.
  - Roll/pitch are pulled slowly toward the accelerometer's gravity direction, so the odom
    frame stays level even if cuVSLAM's frame tilts (tilt turns walking into climbing).
  - A violating pose is never published. The node holds the pose where the body plausibly is
    (last position advanced by the wheel odometry if available, else by the current velocity;
    orientation advanced by the gyro) and re-anchors the VIO frame there, so subsequent VIO
    motion continues from that point.
  - Sustained violations (or lost visual tracking) trigger a full recovery: pause RTAB-Map,
    reset the VIO, re-anchor at the last trustworthy position carried forward by the wheels
    (position, if available) and the gyro (heading), resume. With wheel odometry the published
    pose keeps following the wheels during the reset, so a driving robot never freezes in place.
    Without a trustworthy pose it starts a new RTAB-Map session instead.

Default limits fit a person carrying the camera (hand-held cameras sway hard); robot.launch.py
sets robot limits, wheel_odom_topic and planar output for the Roomba.

With publish_tf false (robot.launch.py fusion:=ekf) a robot_localization EKF owns
odom -> base_frame and /odom instead. This node then only supervises the VIO (same checks,
recoveries and /vslam_filter/status) and forwards the poses it accepted while visual tracking
was OK on fused_vio_topic, for the EKF to fuse differentially. Rejected, IMU-only and
wheel-bridged poses are not forwarded, and the forwarded poses stay continuous across VIO
resets (same re-anchoring as the published pose).

Subscribes: /visual_slam/tracking/odometry (vslam_odom), /visual_slam/status,
            /camera/imu/data_raw (gyro), optional wheel odometry (wheel_odom_topic)
Publishes:  TF odom -> base_frame (filtered), TF odom -> vslam_odom (current anchor),
            /odom (nav_msgs/Odometry, filtered), /vslam_filter/status (std_msgs/String JSON);
            with publish_tf false only fused_vio_topic (nav_msgs/Odometry) and the status
Calls:      /visual_slam/reset, <rtabmap_ns>/pause|resume|trigger_new_map
"""
import collections
import json
import math
import time

import numpy as np
import rclpy
import rclpy.executors
from geometry_msgs.msg import TransformStamped
from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
from isaac_ros_visual_slam_interfaces.srv import Reset
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from roomba_perception.scene_filter import quat_rotate

VO_OK = 1
VO_FAILED = 2
Q_IDENTITY = (0.0, 0.0, 0.0, 1.0)
IDENTITY = (np.zeros(3), Q_IDENTITY)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_inv(q):
    return (-q[0], -q[1], -q[2], q[3])


def compose(a, b):
    """Pose composition a * b for poses (translation ndarray, quaternion)."""
    return a[0] + np.array(quat_rotate(a[1], b[0])), quat_mul(a[1], b[1])


def invert(a):
    qi = quat_inv(a[1])
    return -np.array(quat_rotate(qi, a[0])), qi


def angle_between(q1, q2):
    d = abs(sum(x * y for x, y in zip(q1, q2)))
    return 2.0 * math.acos(min(1.0, d))


def quat_normalize(q):
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def rotvec_to_quat(r):
    a = float(np.linalg.norm(r))
    s = 0.5 if a < 1e-9 else math.sin(0.5 * a) / a
    return (float(r[0]) * s, float(r[1]) * s, float(r[2]) * s, math.cos(0.5 * a))


def quat_to_rotvec(q):
    x, y, z, w = q
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return np.array([2.0 * x, 2.0 * y, 2.0 * z])
    return np.array([x, y, z]) * (2.0 * math.atan2(n, w) / n)


def quat_yaw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class VslamWatchdog(Node):

    def __init__(self):
        super().__init__('vslam_watchdog')
        p = self.declare_parameter
        # Physical motion limits (person carrying the camera by default; a hand-held camera
        # swinging during steps and turns reaches ~10 m/s^2).
        self.max_speed = float(p('max_speed_mps', 1.5).value)
        self.max_vspeed = float(p('max_vertical_speed_mps', 0.8).value)
        self.max_accel = float(p('max_accel_mps2', 10.0).value)
        self.max_turn = float(p('max_turn_rate_rps', 8.0).value)  # only used without IMU
        self.margin_pos = float(p('jitter_margin_m', 0.02).value)
        self.margin_rot = float(p('jitter_margin_rad', 0.03).value)
        # Allowed disagreement between cuVSLAM's rotation and the gyro per frame.
        self.gyro_tol = float(p('gyro_tolerance_rad', 0.05).value)
        self.gyro_tol_rel = float(p('gyro_tolerance_ratio', 0.15).value)
        # Pull roll/pitch toward the accelerometer's gravity direction (fraction per frame), so
        # the odom frame stays level: tilt from tracking errors would otherwise turn walking
        # into climbing/descending and corrupt floor/obstacle separation in the map.
        self.level_gain = float(p('level_gain', 0.02).value)
        self.burst_count = int(p('violations_for_reset', 8).value)
        self.burst_window = float(p('violation_window_s', 2.0).value)
        # Recovery and global sanity.
        self.max_fail_s = float(p('max_tracking_loss_s', 1.0).value)
        self.boundary = float(p('boundary_radius_m', 20.0).value)
        self.max_height = float(p('max_height_m', 3.0).value)
        self.cooldown = float(p('reset_cooldown_s', 5.0).value)
        self.recover_timeout = float(p('recovery_timeout_s', 6.0).value)
        self.new_map_always = bool(p('new_map_on_recovery', False).value)
        self.odom_frame = p('odom_frame', 'odom').value
        self.vslam_frame = p('vslam_odom_frame', 'vslam_odom').value
        self.base_frame = p('base_frame', 'camera_link').value
        imu_topic = p('imu_topic', '/camera/imu/data_raw').value
        rtab_ns = p('rtabmap_ns', '/rtabmap/rtabmap').value
        # Ground robot: wheel odometry (nav_msgs/Odometry, e.g. /wheel/odom from roomba_driver)
        # bridges glitches and tracking loss. Filtered frames move by what the wheels measured
        # instead of a constant-velocity guess, odom -> base keeps following the wheels while the
        # VIO is being reset, and recovery continues from where the wheels say the robot went.
        wheel_topic = p('wheel_odom_topic', '').value
        self.wheel_max_age = float(p('wheel_max_age_s', 0.25).value)
        # Allowed VIO-vs-wheel disagreement per frame (wheel slip, stamp jitter, VIO noise) and
        # over a longer window (catches gross drift that stays under the per-frame tolerance).
        # Loose on purpose: on carpet the Roomba's wheels slipped to 2.5x the real distance when
        # reversing, so the wheels may only overrule the VIO when it is clearly broken.
        self.wheel_tol = float(p('wheel_tolerance_m', 0.03).value)
        self.wheel_tol_rel = float(p('wheel_tolerance_ratio', 0.5).value)
        self.wheel_window = float(p('wheel_window_s', 0.5).value)
        self.wheel_window_tol = float(p('wheel_window_tolerance_m', 0.08).value)
        self.wheel_window_rel = float(p('wheel_window_tolerance_ratio', 0.6).value)
        # Ground robot: publish x, y and heading only (z = 0, no roll/pitch).
        self.planar = bool(p('planar', False).value)
        # false: an EKF publishes odom -> base_frame; accepted VIO poses go to fused_vio_topic.
        self.publish_tf = bool(p('publish_tf', True).value)
        fused_topic = p('fused_vio_topic', '/vslam_filter/vio').value
        # Per-pose variances of the forwarded VIO (see config/ekf_robot.yaml for the reasoning).
        self.vio_pos_var = float(p('vio_position_variance', 0.002).value)
        self.vio_yaw_var = float(p('vio_yaw_variance', 0.0005).value)

        self.cli_reset = self.create_client(Reset, '/visual_slam/reset')
        self.cli_pause = self.create_client(Empty, rtab_ns + '/pause')
        self.cli_resume = self.create_client(Empty, rtab_ns + '/resume')
        self.cli_new_map = self.create_client(Empty, rtab_ns + '/trigger_new_map')
        self.tf_pub = TransformBroadcaster(self)
        self.pub_odom = (self.create_publisher(Odometry, '/odom', 20) if self.publish_tf
                         else self.create_publisher(Odometry, fused_topic, 20))
        self.forwarded = 0
        self.pub_status = self.create_publisher(String, '/vslam_filter/status', 2)

        self.offset = IDENTITY            # odom <- vslam_odom
        self.last = None                  # (stamp, pos, quat) last published pose in odom
        self.vel = np.zeros(3)            # smoothed velocity of accepted motion (odom frame)
        self.violations = collections.deque()
        self.filtered_total = 0
        self.history = collections.deque(maxlen=300)  # (monotonic, stamp, pos, quat) accepted
        self.pending_anchor = None        # (stamp, pos, quat) to place cuVSLAM at after a reset

        # Gyro in base_frame: (stamp, angular velocity) samples, ~20 s at 200 Hz.
        self.imu_buf = collections.deque(maxlen=4000)
        self.imu_rot = None               # rotation base_frame <- IMU frame (static TF)
        self.gyro_bias = np.zeros(3)      # residual bias, learned while tracking is good
        self.acc_lp = None                # low-passed specific force in base_frame (gravity "up")
        self.last_imu = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state = 'idle'
        self.state_since = time.monotonic()
        self.last_recovery = 0.0
        self.recoveries = 0
        self.fail_since = None
        self.ok_count = 0
        self.last_vo_ok = 0.0
        self.anchor = None
        self.last_log = 0.0
        # Wheel odometry: (stamp, x, y, yaw) in the wheel odometry frame, ~40 s at 66 Hz.
        self.wheel_buf = collections.deque(maxlen=2600)
        self.recent = collections.deque(maxlen=60)   # published (stamp, pos, quat), ~4 s
        self.wheel_twist = (0.0, 0.0)
        self.bridge = None                # (stamp, pos, quat) dead-reckoning base during recovery
        self.bridge_t = 0.0
        self.bridged = 0
        self.wheel_followed = 0
        self.last_wheel = 0.0
        if wheel_topic:
            self.create_subscription(Odometry, wheel_topic, self._on_wheel, 50)

        self.create_subscription(VisualSlamStatus, '/visual_slam/status', self._on_status, 10)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self._on_odom, 20)
        self.create_subscription(Imu, imu_topic, self._on_imu, qos_profile_sensor_data)
        self.create_timer(0.1, self._tick)
        self.create_timer(1.0, self._publish_status)

    # ---------- gyro ----------
    def _on_imu(self, msg):
        if self.imu_rot is None:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, Time())
            except TransformException:
                return
            r = tf.transform.rotation
            self.imu_rot = (r.x, r.y, r.z, r.w)
            # Only the static extrinsic was needed; stop deserializing all of /tf.
            for sub in (getattr(self.tf_listener, 'tf_sub', None),
                        getattr(self.tf_listener, 'tf_static_sub', None)):
                if sub is not None:
                    self.destroy_subscription(sub)
            self.get_logger().info('Gyro connected: rotations are checked against the IMU')
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.imu_buf and t <= self.imu_buf[-1][0]:
            return
        w = msg.angular_velocity
        self.imu_buf.append((t, np.array(quat_rotate(self.imu_rot, (w.x, w.y, w.z)))))
        a = msg.linear_acceleration
        a_c = np.array(quat_rotate(self.imu_rot, (a.x, a.y, a.z)))
        if np.linalg.norm(a_c) > 1.0:
            # ~1 s time constant at 200 Hz: averages out steps and hand sway.
            self.acc_lp = a_c if self.acc_lp is None else 0.995 * self.acc_lp + 0.005 * a_c
        self.last_imu = time.monotonic()

    def _gyro_delta(self, t0, t1):
        """Body rotation from t0 to t1 integrated from the gyro (q1 = q0 * delta), or None
        when the IMU buffer doesn't cover the interval."""
        buf = self.imu_buf
        if len(buf) < 2 or t1 <= t0 or buf[-1][0] < t1 - 0.03:
            return None
        seg = []
        for sample in reversed(buf):
            seg.append(sample)
            if sample[0] <= t0:
                break
        if seg[-1][0] > t0 + 0.01:
            return None
        seg.reverse()
        q = Q_IDENTITY
        for i, (t, w) in enumerate(seg):
            ta = max(t, t0)
            tb = min(seg[i + 1][0] if i + 1 < len(seg) else t1, t1)
            if tb > ta:
                q = quat_mul(q, rotvec_to_quat((w - self.gyro_bias) * (tb - ta)))
        return quat_normalize(q)

    def _learn_gyro_bias(self, q_vo, q_gyro, dt):
        """While tracking is good, cuVSLAM's rotation is accurate on average: the leftover
        difference to the gyro over slow frames is gyro bias. Learned slowly (~30 s) with
        per-frame outliers clipped, so tracking glitches can't drag it away."""
        rv_g = quat_to_rotvec(q_gyro)
        if dt <= 0.0 or dt > 0.2 or np.linalg.norm(rv_g) > 0.1:
            return
        resid = (rv_g - quat_to_rotvec(q_vo)) / dt
        n = float(np.linalg.norm(resid))
        if n > 0.02:
            resid *= 0.02 / n
        self.gyro_bias = self.gyro_bias + 0.002 * resid
        n = float(np.linalg.norm(self.gyro_bias))
        if n > 0.05:  # ~3 deg/s: anything larger is not bias
            self.gyro_bias *= 0.05 / n

    def _level(self, quat):
        """Nudges roll/pitch of quat (odom <- base) toward the measured gravity direction."""
        if self.acc_lp is None or self.level_gain <= 0.0:
            return quat
        n = float(np.linalg.norm(self.acc_lp))
        if abs(n - 9.81) > 0.8:  # accelerating hard: the average isn't gravity right now
            return quat
        up = np.array(quat_rotate(quat, self.acc_lp / n))  # measured "up" in odom
        axis = np.cross(up, (0.0, 0.0, 1.0))
        s = float(np.linalg.norm(axis))
        if s < 1e-6:
            return quat
        ang = math.atan2(s, float(up[2]))
        corr = rotvec_to_quat(axis / s * ang * self.level_gain)
        return quat_normalize(quat_mul(corr, quat))

    def tilt_deg(self, quat):
        """Angle between odom's up and measured gravity, for status/tests."""
        if self.acc_lp is None:
            return None
        up = np.array(quat_rotate(quat, self.acc_lp / np.linalg.norm(self.acc_lp)))
        return math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))

    # ---------- wheel odometry ----------
    def _on_wheel(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.wheel_buf and t <= self.wheel_buf[-1][0]:
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.wheel_buf.append((t, p.x, p.y, quat_yaw((q.x, q.y, q.z, q.w))))
        self.wheel_twist = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        self.last_wheel = time.monotonic()

    def _wheel_pose(self, t):
        """Wheel odometry (x, y, yaw) interpolated at t, or None if not covered."""
        buf = self.wheel_buf
        if not buf or t < buf[0][0] or t > buf[-1][0] + self.wheel_max_age:
            return None
        if t >= buf[-1][0]:
            return buf[-1][1:]
        for i in range(len(buf) - 1, 0, -1):
            a = buf[i - 1]
            if a[0] <= t:
                b = buf[i]
                f = (t - a[0]) / (b[0] - a[0])
                return (a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]),
                        a[3] + f * wrap_angle(b[3] - a[3]))
        return None

    def _wheel_delta(self, t0, t1):
        """Body motion from t0 to t1 measured by the wheels: (translation in the t0 body frame,
        yaw change), or None."""
        p0, p1 = self._wheel_pose(t0), self._wheel_pose(t1)
        if p0 is None or p1 is None:
            return None
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        c, s = math.cos(p0[2]), math.sin(p0[2])
        return np.array([c * dx + s * dy, -s * dx + c * dy, 0.0]), wrap_angle(p1[2] - p0[2])

    def _wheel_position(self, t0, pos0, quat0, t1):
        """Position at t1 starting from (pos0, quat0) at t0 and moving as the wheels did."""
        d = self._wheel_delta(t0, t1)
        if d is None:
            return None
        # The wheels move the robot along the floor: rotate by heading only.
        return np.asarray(pos0) + np.array(quat_rotate(yaw_quat(quat_yaw(quat0)), d[0]))

    def _dead_reckon(self, base, t):
        """Pose at t from base (stamp, pos, quat): wheels for position, gyro (or wheels) for
        heading. None without wheel odometry covering the interval."""
        bt, bpos, bquat = base
        d = self._wheel_delta(bt, t)
        if d is None:
            return None
        q_g = self._gyro_delta(bt, t)
        rot = q_g if q_g is not None else yaw_quat(d[1])
        pos = np.asarray(bpos) + np.array(quat_rotate(yaw_quat(quat_yaw(bquat)), d[0]))
        return pos, quat_normalize(quat_mul(bquat, rot))

    def _publish_bridge(self):
        """While the VIO is reset, keep odom -> base moving with the wheels."""
        if self.bridge is None or not self.wheel_buf:
            return
        t = self.wheel_buf[-1][0]
        if t <= self.bridge_t:
            return
        reckoned = self._dead_reckon(self.bridge, t)
        if reckoned is None:
            return
        self.bridge_t = t
        self.bridged += 1
        msg = Odometry()
        ns = int(round(t * 1e9))
        msg.header.stamp.sec, msg.header.stamp.nanosec = ns // 1_000_000_000, ns % 1_000_000_000
        cov = [0.0] * 36
        for i, v in enumerate((0.05, 0.05, 0.05, 0.05, 0.05, 0.1)):
            cov[i * 7] = v
        msg.pose.covariance = cov
        msg.twist.twist.linear.x, msg.twist.twist.angular.z = self.wheel_twist
        self._publish(msg, *reckoned)

    # ---------- tracking status ----------
    def _on_status(self, msg):
        now = time.monotonic()
        if msg.vo_state == VO_OK:
            self.ok_count += 1
            self.last_vo_ok = now
        else:
            self.ok_count = 0
        if self.state != 'idle':
            return
        if msg.vo_state == VO_FAILED:
            if self.fail_since is None:
                self.fail_since = now
            elif now - self.fail_since > self.max_fail_s:
                self._recover(f'visual tracking lost for {now - self.fail_since:.1f} s')
        else:
            self.fail_since = None

    # ---------- motion filter ----------
    def _check_motion(self, t, pos, quat):
        """Returns (reason, kind, gyro_delta). reason is set if moving from the last pose to
        (pos, quat) is impossible; kind is 'pos' for translation and 'rot' for rotation
        violations. gyro_delta is the gyro-measured rotation over the interval (or None)."""
        lt, lpos, lquat = self.last
        dt = t - lt
        if dt <= 0.0:
            return 'non-increasing stamp', 'pos', None
        q_g = self._gyro_delta(lt, t)
        if q_g is not None:
            gyro_angle = angle_between(q_g, Q_IDENTITY)
            err = angle_between(quat_mul(quat_inv(lquat), quat), q_g)
            if err > self.gyro_tol + self.gyro_tol_rel * gyro_angle:
                return (f'rotation off by {math.degrees(err):.0f} deg from gyro '
                        f'({math.degrees(gyro_angle / dt):.0f} deg/s measured)'), 'rot', q_g
        else:
            turn = angle_between(lquat, quat) / dt
            if turn > self.max_turn + self.margin_rot / dt:
                return f'turn rate {math.degrees(turn):.0f} deg/s', 'rot', None
        v = (pos - lpos) / dt
        speed = float(np.linalg.norm(v))
        # Ground robot: the wheels measure how far it really moved. VIO drift that stays under
        # the speed limits (e.g. IMU-only runaway just after tracking fails) disagrees with them.
        wheel_pos = self._wheel_position(lt, lpos, lquat, t) if self.wheel_buf else None
        if wheel_pos is not None:
            if self.fail_since is not None:
                # Visual tracking is lost, so the VIO is integrating the IMU alone: the wheels
                # are the better estimate from the first frame on. Not counted as a glitch: real
                # tracking loss is handled by max_tracking_loss_s.
                return 'visual tracking lost, following the wheels', 'lost', q_g
            moved = float(np.linalg.norm(wheel_pos[:2] - lpos[:2]))
            off = float(np.linalg.norm(pos[:2] - wheel_pos[:2]))
            if off > self.wheel_tol + self.wheel_tol_rel * moved:
                return f'{off * 100:.0f} cm off the wheel odometry', 'pos', q_g
            # Slow drift passes the per-frame test, so also compare over a longer window.
            ref = next((r for r in reversed(self.recent) if t - r[0] >= self.wheel_window), None)
            ref_pos = self._wheel_position(*ref, t) if ref is not None else None
            if ref_pos is not None:
                moved = float(np.linalg.norm(ref_pos[:2] - ref[1][:2]))
                off = float(np.linalg.norm(pos[:2] - ref_pos[:2]))
                if off > self.wheel_window_tol + self.wheel_window_rel * moved:
                    return (f'drifted {off * 100:.0f} cm from the wheel odometry in '
                            f'{t - ref[0]:.1f} s'), 'pos', q_g
        if speed > self.max_speed + self.margin_pos / dt:
            return f'speed {speed:.1f} m/s', 'pos', q_g
        if abs(v[2]) > self.max_vspeed + self.margin_pos / dt:
            return f'vertical speed {v[2]:+.1f} m/s', 'pos', q_g
        dv = float(np.linalg.norm(v - self.vel))
        if dv > self.max_accel * dt + 2.0 * self.margin_pos / dt:
            return f'acceleration {dv / dt:.0f} m/s^2', 'pos', q_g
        return None, None, q_g

    def _on_odom(self, msg):
        if self.state != 'idle':
            return  # hold the last pose while cuVSLAM is being reset
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last is not None and t <= self.last[0]:
            return
        pp, pq = msg.pose.pose.position, msg.pose.pose.orientation
        raw = (np.array([pp.x, pp.y, pp.z]), (pq.x, pq.y, pq.z, pq.w))
        if self.pending_anchor is not None:
            # First pose after a reset: continue from the last good position, carried forward
            # over the time tracking was broken by the wheels (position) and the gyro (heading).
            anchor = self.pending_anchor
            self.pending_anchor = None
            reckoned = self._dead_reckon(anchor, t)
            if reckoned is None:
                at, apos, aquat = anchor
                q_g = self._gyro_delta(at, t)
                if q_g is None:
                    self.get_logger().warn('No gyro data across the recovery; heading may be off')
                    q_g = Q_IDENTITY
                reckoned = (np.array(apos), quat_normalize(quat_mul(aquat, q_g)))
            self.offset = compose(reckoned, invert(raw))
        pos, quat = compose(self.offset, raw)

        if self.last is None:
            reason, kind, q_g = None, None, None
        else:
            reason, kind, q_g = self._check_motion(t, pos, quat)
        if reason:
            now = time.monotonic()
            # Keep the body where it plausibly is and continue cuVSLAM's motion from there.
            # Rotation comes from the gyro when cuVSLAM's disagrees (or the last heading
            # without IMU); a translation-only glitch keeps cuVSLAM's gyro-consistent rotation.
            lt, lpos, lquat = self.last
            if kind == 'rot':
                new_q = quat_normalize(quat_mul(lquat, q_g)) if q_g is not None else lquat
            else:
                new_q = quat
            # With wheel odometry the body moved by what the wheels measured, not by a guess.
            wheel_pos = self._wheel_position(lt, lpos, lquat, t)
            pred = (wheel_pos if wheel_pos is not None else lpos + self.vel * (t - lt), new_q)
            self.vel = 0.8 * self.vel  # don't coast on a stale velocity through long glitches
            self.offset = compose(pred, invert(raw))
            pos, quat = pred
            if kind == 'lost':
                self.wheel_followed += 1
                if now - self.last_log > 5.0:
                    self.last_log = now
                    self.get_logger().info('Visual tracking lost: following the wheel odometry')
            else:
                self.filtered_total += 1
                self.violations.append(t)
                while self.violations and t - self.violations[0] > self.burst_window:
                    self.violations.popleft()
                if now - self.last_log > 1.0:
                    self.last_log = now
                    self.get_logger().warn(f'Filtered impossible motion ({reason}); holding position '
                                           f'[{self.filtered_total} filtered so far]')
            if len(self.violations) >= self.burst_count:
                self.violations.clear()
                self._recover(f'{self.burst_count} impossible motions within {self.burst_window:.0f} s')
                if self.state != 'idle':
                    return
        accepted = not reason
        if accepted and self.last is not None:
            dt = t - self.last[0]
            self.vel = 0.7 * self.vel + 0.3 * (pos - self.last[1]) / dt
            if q_g is not None:
                self._learn_gyro_bias(quat_mul(quat_inv(self.last[2]), quat), q_g, dt)

        leveled = self._level(quat)
        if leveled is not quat:
            quat = leveled
            self.offset = compose((pos, quat), invert(raw))

        if math.hypot(pos[0], pos[1]) > self.boundary or abs(pos[2]) > self.max_height:
            self._recover(f'position ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) m outside boundary')
            return
        # Only in-bounds, accepted poses from while visual tracking was OK can become a recovery
        # anchor (IMU-only poses after tracking fails drift).
        if accepted and self.fail_since is None and time.monotonic() - self.last_vo_ok < 0.5:
            self.history.append((time.monotonic(), t, pos, quat))
        self.last = (t, pos, quat)
        self.recent.append(self.last)
        self._publish(msg, pos, quat, vio_ok=accepted and self.fail_since is None)

    def _publish(self, msg, pos, quat, vio_ok=False):
        stamp = msg.header.stamp
        if self.planar:
            # Ground robot: it can't leave the floor, so publish x, y and heading only.
            pos = np.array([pos[0], pos[1], 0.0])
            quat = yaw_quat(quat_yaw(quat))
        if not self.publish_tf:
            # EKF mode: forward only VIO poses that passed every check with tracking OK.
            if vio_ok:
                self._forward(stamp, pos, quat)
            return
        base = TransformStamped()
        base.header.stamp, base.header.frame_id, base.child_frame_id = stamp, self.odom_frame, self.base_frame
        (base.transform.translation.x, base.transform.translation.y,
         base.transform.translation.z) = map(float, pos)
        (base.transform.rotation.x, base.transform.rotation.y,
         base.transform.rotation.z, base.transform.rotation.w) = quat
        anchor = TransformStamped()
        anchor.header.stamp, anchor.header.frame_id, anchor.child_frame_id = stamp, self.odom_frame, self.vslam_frame
        (anchor.transform.translation.x, anchor.transform.translation.y,
         anchor.transform.translation.z) = map(float, self.offset[0])
        (anchor.transform.rotation.x, anchor.transform.rotation.y,
         anchor.transform.rotation.z, anchor.transform.rotation.w) = self.offset[1]
        self.tf_pub.sendTransform([base, anchor])

        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, self.odom_frame, self.base_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, pos)
        (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = quat
        odom.pose.covariance = msg.pose.covariance
        odom.twist = msg.twist  # body-frame velocity from cuVSLAM
        self.pub_odom.publish(odom)

    def _forward(self, stamp, pos, quat):
        odom = Odometry()
        # The pose lives in this node's anchored VIO frame; the EKF only uses differences
        # between consecutive poses (differential mode), so the frame name is nominal.
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, self.odom_frame, self.base_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, pos)
        (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = quat
        cov = [0.0] * 36
        for i, v in enumerate((self.vio_pos_var, self.vio_pos_var, self.vio_pos_var,
                               self.vio_yaw_var, self.vio_yaw_var, self.vio_yaw_var)):
            cov[i * 7] = v
        odom.pose.covariance = cov
        self.pub_odom.publish(odom)
        self.forwarded += 1

    def _publish_status(self):
        self.pub_status.publish(String(data=json.dumps({
            'state': self.state, 'filtered': self.filtered_total, 'recoveries': self.recoveries,
            'speed_mps': round(float(np.linalg.norm(self.vel)), 2),
            'gyro': time.monotonic() - self.last_imu < 1.0,
            'gyro_bias_dps': round(math.degrees(float(np.linalg.norm(self.gyro_bias))), 2),
            'tilt_deg': (round(self.tilt_deg(self.last[2]), 1)
                         if self.last is not None and self.acc_lp is not None else None),
            'wheels': bool(self.wheel_buf) and self._wheel_pose(self.wheel_buf[-1][0]) is not None
                      and time.monotonic() - self.last_wheel < 1.0,
            'bridged': self.bridged,
            'wheel_followed': self.wheel_followed,
            'output': 'tf' if self.publish_tf else 'ekf',
            'forwarded': self.forwarded})))

    def _last_good_pose(self):
        """Most recent trustworthy (stamp, pos, quat) from before the failure started."""
        cutoff = time.monotonic() - 0.3
        for mono, stamp, pos, quat in reversed(self.history):
            if mono <= cutoff:
                return stamp, pos, quat
        return None

    # ---------- recovery sequence ----------
    def _recover(self, reason):
        now = time.monotonic()
        if self.state != 'idle' or now - self.last_recovery < self.cooldown:
            return
        if not self.cli_reset.service_is_ready():
            self.get_logger().error(f'Cannot recover ({reason}): /visual_slam/reset unavailable')
            self.last_recovery = now
            return
        self.recoveries += 1
        self.last_recovery = now
        self.anchor = None if self.new_map_always else self._last_good_pose()
        # With wheels, keep the robot's pose moving from the anchor until the VIO is back.
        self.bridge = self.anchor if self.wheel_buf else None
        self.bridge_t = 0.0
        how = 'continuing from last good pose' if self.anchor else 'starting a new map session'
        self.get_logger().warn(f'Recovery #{self.recoveries}: {reason} ({how})')
        self._enter('pausing')
        self._call(self.cli_pause, Empty.Request(), lambda _ok: self._after_pause())

    def _after_pause(self):
        self._enter('resetting')
        self._call(self.cli_reset, Reset.Request(), lambda _ok: self._after_reset())

    def _after_reset(self):
        # cuVSLAM restarts in a fresh vslam_odom: the first pose it reports after the reset is
        # placed at the anchor (heading advanced by the gyro), see _on_odom.
        self.offset = IDENTITY
        self.pending_anchor = self.anchor
        self.last = None
        self.recent.clear()
        self.vel = np.zeros(3)
        self._enter('waiting')

    def _tick(self):
        if self.state == 'idle':
            return
        self._publish_bridge()
        elapsed = time.monotonic() - self.state_since
        if self.state == 'waiting':
            # Healthy again: several consecutive OK statuses after the reset.
            if self.ok_count >= 5 or elapsed > self.recover_timeout:
                if self.ok_count < 5:
                    self.get_logger().warn('Tracking not healthy yet; resuming anyway')
                if self.anchor is None:
                    self._enter('new_map')
                    self._call(self.cli_new_map, Empty.Request(), lambda _ok: self._resume())
                else:
                    self._resume()
        elif elapsed > self.recover_timeout:
            self.get_logger().error(f'Recovery stuck in "{self.state}"; resuming mapping')
            self._resume()

    def _resume(self):
        self._enter('resuming')
        self._call(self.cli_resume, Empty.Request(), lambda _ok: self._done())

    def _done(self):
        self.history.clear()
        self.violations.clear()
        self.fail_since = None
        self.bridge = None
        how = 'same map, continued from last good pose' if self.anchor else 'new map session started'
        self._enter('idle')
        self.get_logger().info(f'Recovery #{self.recoveries} complete: {how}')

    def _enter(self, state):
        self.state = state
        self.state_since = time.monotonic()
        if state == 'waiting':
            self.ok_count = 0

    def _call(self, client, request, then):
        """Calls a service asynchronously; continues even if it is unavailable or fails."""
        if not client.service_is_ready():
            self.get_logger().warn(f'{client.srv_name} unavailable, skipping')
            then(False)
            return

        def done(future):
            try:
                future.result()
                then(True)
            except Exception as e:  # noqa: BLE001 - keep the sequence moving
                self.get_logger().error(f'{client.srv_name} failed: {e}')
                then(False)

        client.call_async(request).add_done_callback(done)


def main():
    rclpy.init()
    node = VslamWatchdog()
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
