#!/usr/bin/env python3
"""Prepares the wheel odometry and the camera gyro for the robot_localization EKF (fusion:=ekf).

robot_localization takes each sensor's covariance from its messages. The Roomba driver reports
fixed covariances that are far too confident for carpet, and raw 200 Hz gyro samples with
their bias would make the fused heading drift while the robot stands still. This node:

  /wheel/odom -> /ekf/wheel_odom   twist scaled by the measured calibration (wheels read +5 %
                                   forward and ~40 % too much rotation on carpet), covariances
                                   that grow with speed (slip/scrub), larger when reversing
                                   (slips) and vy = 0 for the non-holonomic base.
  /camera/imu/data_raw -> /ekf/imu gyro averaged over imu_average samples (200 -> 50 Hz), minus
                                   a bias learned while the robot stands still (wheels report
                                   zero motion, or, without wheel data, the gyro is quiet), with
                                   a covariance that also covers the residual bias. Orientation
                                   and acceleration are marked unavailable (covariance[0] = -1);
                                   the frame stays the IMU's, robot_localization rotates the rate
                                   into base_link through TF (camera driver + mount transforms).

Gyro noise model, from this IMU's own calibration (imu_calibrate: 3 h static Allan analysis,
~/.config/roomba/imu_noise.yaml). That file stores the values multiplied by the inflation
named in its header ("raw values x10.0", for cuVSLAM), so they are divided back to the measured
raw values here. Per published (averaged) sample at f_out = calibration_frequency / imu_average:
  white  = N * sqrt(f_out)                         N: raw gyro noise density, rad/s/sqrt(Hz)
  bias   = sqrt(sb0^2 + K^2 * dt_still)            K: raw gyro random walk, rad/s^2/sqrt(Hz);
           dt_still: time since the bias was last updated at standstill; sb0: uncertainty of
           that estimate, N / sqrt(2 * bias_time_constant_s) for the running average, or
           bias_turn_on_sd until the first standstill of this run (bias from the calibration
           recording or zero; turn-on and temperature shift it by a few 0.01 deg/s)
  motion = gyro_motion_sd_ratio * |rate|           scale factor / vibration / g-sensitivity: a
           static test can't measure these, so this is an allowance (default 2 %)
  sd = sqrt(white^2 + bias^2 + motion^2)
Initial bias: gyro_static_bias from imu_noise.yaml if present, else the mean of the newest
static recording ~/maps/imu_raw_*.npz written by imu_calibrate. Without imu_noise.yaml the old
fixed gyro_sd (0.03 rad/s) is used.

Variances are documented in config/ekf_robot.yaml together with the EKF settings.
Publishes /ekf/inputs_status (std_msgs/String JSON, 1 Hz): gyro bias, gyro sd, wheel data age.
"""
import collections
import glob
import json
import math
import os
import re
import time

import numpy as np
import rclpy
import rclpy.executors
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String

IMU_NOISE_FILE = '~/.config/roomba/imu_noise.yaml'
IMU_RAW_GLOB = '~/maps/imu_raw_*.npz'


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def load_imu_noise(path):
    """(values dict, inflation) from imu_calibrate's imu_noise.yaml, or (None, None)."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return None, None
    values, inflation = {}, None
    with open(path) as f:
        for line in f:
            m = re.search(r'raw values x([0-9.]+)', line)
            if line.lstrip().startswith('#'):
                if m:
                    inflation = float(m.group(1))
                continue
            key, sep, value = line.partition(':')
            if not sep:
                continue
            value = value.strip()
            if value.startswith('['):
                values[key.strip()] = [float(v) for v in value.strip('[]').split(',')]
            else:
                values[key.strip()] = float(value)
    return values, inflation


class EkfInputs(Node):

    def __init__(self):
        super().__init__('ekf_inputs')
        p = self.declare_parameter
        self.lin_scale = float(p('wheel_linear_scale', 0.952).value)
        self.ang_scale = float(p('wheel_angular_scale', 0.71).value)
        self.vx_sd0 = float(p('wheel_vx_sd', 0.01).value)
        self.vx_sd_rel = float(p('wheel_vx_sd_ratio', 0.1).value)
        self.vx_sd_rel_rev = float(p('wheel_vx_sd_ratio_reverse', 0.5).value)
        self.vy_sd = float(p('wheel_vy_sd', 0.01).value)
        self.wz_sd0 = float(p('wheel_wz_sd', 0.01).value)
        self.wz_sd_rel = float(p('wheel_wz_sd_ratio', 0.4).value)
        self.gyro_sd_fallback = float(p('gyro_sd', 0.03).value)
        self.motion_ratio = float(p('gyro_motion_sd_ratio', 0.02).value)
        self.turn_on_sd = float(p('bias_turn_on_sd', 0.001).value)
        self.average = max(1, int(p('imu_average', 4).value))
        self.still_s = float(p('still_time_s', 1.0).value)
        self.bias_tau = float(p('bias_time_constant_s', 5.0).value)
        self.wheel_max_age = float(p('wheel_max_age_s', 0.5).value)
        noise_file = p('imu_noise_file', IMU_NOISE_FILE).value
        raw_glob = p('imu_raw_glob', IMU_RAW_GLOB).value

        self.bias = np.zeros(3)
        self._load_calibration(noise_file, raw_glob)
        quiet = float(p('quiet_gyro_rate', 0.0).value)
        # Standstill test without wheels: per-sample sd of the 200 Hz gyro is N * sqrt(rate);
        # 6 sigma of that (but at least 0.005 rad/s) still counts as "not turning".
        self.quiet_rate = quiet if quiet > 0.0 else (
            max(0.005, 6.0 * self.noise_n * math.sqrt(self.imu_rate)) if self.noise_n else 0.02)

        self.pub_wheel = self.create_publisher(Odometry, '/ekf/wheel_odom', 20)
        self.pub_imu = self.create_publisher(Imu, '/ekf/imu', 50)
        self.pub_status = self.create_publisher(String, '/ekf/inputs_status', 2)
        self.create_subscription(Odometry, p('wheel_topic', '/wheel/odom').value, self._on_wheel, 20)
        self.create_subscription(Imu, p('imu_topic', '/camera/imu/data_raw').value, self._on_imu,
                                 qos_profile_sensor_data)
        self.create_timer(1.0, self._status)

        self.bias_ready = False             # learned at a standstill during this run
        self.bias_samples = 0
        self.bias_stamp = None              # stamp of the last standstill bias update
        self.last_sd = None
        self.acc = []                       # gyro samples being averaged
        self.window = collections.deque()   # (stamp, gyro) for the quiet test, ~still_s long
        self.wheel_still_since = None       # stamp from which the wheels report standstill
        self.last_wheel_stamp = None
        self.last_wheel_mono = 0.0

    def _load_calibration(self, noise_file, raw_glob):
        log = self.get_logger()
        values, inflation = load_imu_noise(noise_file)
        self.noise_n = self.noise_k = None
        self.imu_rate = 200.0
        if values and 'gyro_noise_density' in values:
            if inflation is None:
                inflation = 1.0
                log.warn(f'{noise_file}: no "raw values xN" in the header; assuming raw values')
            self.noise_n = values['gyro_noise_density'] / inflation
            self.noise_k = values.get('gyro_random_walk', 0.0) / inflation
            self.imu_rate = values.get('calibration_frequency', 200.0)
            f_out = self.imu_rate / self.average
            log.info(f'Gyro noise from {os.path.expanduser(noise_file)} (stored x{inflation:g}, '
                     f'divided back): N = {self.noise_n:.4g} rad/s/sqrt(Hz), K = {self.noise_k:.4g} '
                     f'rad/s^2/sqrt(Hz), {self.imu_rate:g} Hz -> white sd {self.noise_n * math.sqrt(f_out):.2g} '
                     f'rad/s at {f_out:.1f} Hz; motion allowance {self.motion_ratio:.0%} of rate, '
                     f'turn-on bias sd {self.turn_on_sd:g} rad/s')
        else:
            log.warn(f'{noise_file} not found: fixed gyro sd {self.gyro_sd_fallback} rad/s (uncalibrated default)')
        if values and isinstance(values.get('gyro_static_bias'), list):
            self.bias = np.array(values['gyro_static_bias'][:3])
            log.info(f'Initial gyro bias from {noise_file}: {np.round(np.degrees(self.bias), 4)} deg/s')
            return
        raws = sorted(glob.glob(os.path.expanduser(raw_glob)))
        if raws:
            try:
                with np.load(raws[-1]) as d:
                    g = d['gyro']
                    hours = (d['t'][-1] - d['t'][0]) / 3600.0
                self.bias = g.mean(axis=0)
                log.info(f'Initial gyro bias = mean of the {hours:.1f} h static calibration recording '
                         f'{raws[-1]}: {np.round(np.degrees(self.bias), 4)} deg/s (relearned at standstill)')
                return
            except (OSError, KeyError, ValueError) as e:
                log.warn(f'Could not read {raws[-1]}: {e}')
        log.info('No static gyro bias available: starting from zero (learned at the first standstill)')

    def _gyro_sd(self, t, rate):
        if self.noise_n is None:
            return self.gyro_sd_fallback * (1.0 if self.bias_ready else 10.0)
        white = self.noise_n * math.sqrt(self.imu_rate / self.average)
        if self.bias_ready:
            sb0 = self.noise_n / math.sqrt(2.0 * self.bias_tau)
            dt = max(0.0, t - self.bias_stamp)
        else:
            sb0, dt = self.turn_on_sd, 0.0
        bias = math.sqrt(sb0 * sb0 + self.noise_k * self.noise_k * dt)
        motion = self.motion_ratio * rate
        return math.sqrt(white * white + bias * bias + motion * motion)

    def _on_wheel(self, msg):
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id or 'base_link'
        vx = msg.twist.twist.linear.x * self.lin_scale
        wz = msg.twist.twist.angular.z * self.ang_scale
        out.twist.twist.linear.x = vx
        out.twist.twist.angular.z = wz
        rel = self.vx_sd_rel_rev if vx < 0.0 else self.vx_sd_rel
        cov = [0.0] * 36
        cov[0] = self.vx_sd0 ** 2 + (rel * vx) ** 2
        cov[7] = self.vy_sd ** 2
        cov[14] = cov[21] = cov[28] = 1e3   # z, roll, pitch rates: unused (two_d_mode)
        cov[35] = self.wz_sd0 ** 2 + (self.wz_sd_rel * wz) ** 2
        out.twist.covariance = cov
        # Pose is not fused (integrated wheel pose drifts without bound); keep it marked unknown.
        out.pose.covariance = [1e3 if i % 7 == 0 else 0.0 for i in range(36)]
        self.pub_wheel.publish(out)

        t = _stamp(msg)
        moving = abs(msg.twist.twist.linear.x) > 1e-3 or abs(msg.twist.twist.angular.z) > 1e-3
        if moving:
            self.wheel_still_since = None
        elif self.wheel_still_since is None:
            self.wheel_still_since = t
        self.last_wheel_stamp = t
        self.last_wheel_mono = time.monotonic()

    def _still(self, t):
        """True if the robot is known to be standing still at stamp t."""
        if self.last_wheel_stamp is not None and time.monotonic() - self.last_wheel_mono < self.wheel_max_age:
            return self.wheel_still_since is not None and t - self.wheel_still_since >= self.still_s
        # No wheel data (Roomba asleep, or replay without it): a quiet gyro for still_s.
        if len(self.window) < 20 or t - self.window[0][0] < 0.9 * self.still_s:
            return False
        g = np.array([w for _, w in self.window])
        # Before the first bias estimate allow up to ~3 deg/s offset (a MEMS bias, not a turn).
        mean_tol = self.quiet_rate if self.bias_ready else 0.05
        return bool(np.all(np.abs(g.mean(axis=0) - self.bias) < mean_tol)
                    and np.all(g.std(axis=0) < self.quiet_rate))

    def _on_imu(self, msg):
        t = _stamp(msg)
        w = msg.angular_velocity
        g = np.array([w.x, w.y, w.z])
        self.window.append((t, g))
        while self.window and t - self.window[0][0] > self.still_s:
            self.window.popleft()
        if self._still(t):
            # First estimate: plain mean of the still window; then a slow running average.
            if not self.bias_ready:
                self.bias = np.array([s for _, s in self.window]).mean(axis=0)
                self.bias_ready = True
            else:
                a = min(1.0, 1.0 / (self.imu_rate * self.bias_tau))
                self.bias = (1.0 - a) * self.bias + a * g
            self.bias_stamp = t
            self.bias_samples += 1
        self.acc.append((msg, g))
        if len(self.acc) < self.average:
            return
        last = self.acc[-1][0]
        mean = np.mean([s for _, s in self.acc], axis=0) - self.bias
        self.acc = []
        out = Imu()
        out.header = last.header
        out.orientation_covariance[0] = -1.0
        out.linear_acceleration_covariance[0] = -1.0
        out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z = map(float, mean)
        sd = self._gyro_sd(_stamp(last), float(np.linalg.norm(mean)))
        self.last_sd = sd
        v = sd * sd
        out.angular_velocity_covariance = [v, 0.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, v]
        self.pub_imu.publish(out)

    def _status(self):
        age = time.monotonic() - self.last_wheel_mono if self.last_wheel_stamp is not None else None
        self.pub_status.publish(String(data=json.dumps({
            'gyro_bias_dps': [round(math.degrees(b), 4) for b in self.bias],
            'bias_ready': self.bias_ready, 'bias_samples': self.bias_samples,
            'gyro_sd_rad_s': round(self.last_sd, 6) if self.last_sd is not None else None,
            'calibrated': self.noise_n is not None,
            'wheel_age_s': round(age, 2) if age is not None else None})))


def main():
    rclpy.init()
    node = EkfInputs()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
