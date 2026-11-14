#!/usr/bin/env python3
"""Measures the camera IMU's noise (Allan deviation) and writes cuVSLAM IMU parameters.

The camera must stay completely still for the whole recording (default 3 h after a 10 min
warm-up). Results use the Kalibr IMU noise model that cuVSLAM expects:
  noise density  N: Allan deviation line of slope -1/2, evaluated at tau = 1 s (fitted between
                    the IMU's low-pass filter region and the bias-instability minimum)
  random walk    K: Allan deviation line of slope +1/2, evaluated at tau = 3 s (fitted on the
                    long-tau tail only)
Raw values are multiplied by `inflation` (default 10) before use. A static test can't capture
hand-held vibration, motion blur or temperature changes; replaying recorded walks, ~10x the
static noise tracked best for cuVSLAM and matched the hand-tuned Basalt settings (x3 tracked no
better in good light and trusts the IMU more where vision is weak).

  ros2 run roomba_perception imu_calibrate --ros-args -p duration_s:=10800.0
  ros2 run roomba_perception imu_calibrate --ros-args -p analyze_file:=/path/imu_raw.npz

Outputs (in output_dir, default ~/maps):
  imu_raw_<date>.npz     recorded samples
  imu_allan_<date>.png   Allan deviation plot (if matplotlib is available)
  ~/.config/roomba/imu_noise.yaml     cuVSLAM parameters, loaded by vslam.launch.py
  ~/.config/roomba/basalt_calib.json  Basalt calibration with these noise values, loaded by
                                      vslam.launch.py instead of the packaged one
"""
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Imu

CONFIG_FILE = os.path.expanduser('~/.config/roomba/imu_noise.yaml')
BASALT_CALIB_FILE = os.path.expanduser('~/.config/roomba/basalt_calib.json')


def allan_deviation(x, rate, n_taus=80):
    """Overlapping Allan deviation of a rate signal x sampled at `rate` Hz."""
    n = x.size
    max_m = n // 9
    ms = np.unique(np.logspace(0, math.log10(max_m), n_taus).astype(int))
    theta = np.concatenate([[0.0], np.cumsum(x)]) / rate  # integrated signal
    taus, devs = [], []
    for m in ms:
        tau = m / rate
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        avar = np.sum(d * d) / (2.0 * tau * tau * (n - 2 * m + 1))
        taus.append(tau)
        devs.append(math.sqrt(avar))
    return np.array(taus), np.array(devs)


def fit_line(taus, devs, slope, at_tau):
    """Fits log(dev) = slope*log(tau) + b where the local slope is closest to `slope`."""
    lt, ld = np.log10(taus), np.log10(devs)
    local = np.gradient(ld, lt)
    i = int(np.argmin(np.abs(local - slope)))
    b = ld[i] - slope * lt[i]
    return 10 ** (slope * math.log10(at_tau) + b), taus[i], abs(local[i] - slope)


def noise_density(taus, devs):
    """White noise N: the -1/2 slope region between the IMU's internal low-pass filter (which
    bends the curve at very short tau, here up to ~0.02 s) and the bias-instability minimum."""
    i_min = int(np.argmin(devs))
    sel = (taus >= 0.1) & (taus <= taus[i_min])
    if np.count_nonzero(sel) < 3:
        sel = taus <= taus[i_min]
    return fit_line(taus[sel], devs[sel], -0.5, 1.0)


def random_walk(taus, devs):
    """Bias random walk K, read only from the long-tau tail after the bias-instability minimum.
    The +1/2 line lies below the curve, so K = min(dev * sqrt(3/tau)) over the tail; if the
    tail hasn't turned up within the recording this is an upper bound (conservative).
    Searching the whole curve instead picked up the low-pass filter's rise at tau < 0.02 s and
    reported a gyro random walk ~10^4 times too large."""
    i_min = int(np.argmin(devs))
    k = devs[i_min:] * np.sqrt(3.0 / taus[i_min:])
    j = int(np.argmin(k))
    return float(k[j]), taus[i_min + j]


def analyze(path, inflation, out_png=None):
    data = np.load(path)
    t, gyro, accel = data['t'], data['gyro'], data['accel']
    rate = (t.size - 1) / (t[-1] - t[0])
    hours = (t[-1] - t[0]) / 3600.0
    result = {'samples': int(t.size), 'rate_hz': round(rate, 2), 'hours': round(hours, 2), 'axes': {}}
    curves = {}
    for name, sig in (('gyro', gyro), ('accel', accel)):
        Ns, Ks = [], []
        for axis in range(3):
            taus, devs = allan_deviation(sig[:, axis] - sig[:, axis].mean(), rate)
            N, tau_n, err_n = noise_density(taus, devs)
            K, tau_k = random_walk(taus, devs)
            Ns.append(N)
            Ks.append(K)
            curves[f'{name}_{"xyz"[axis]}'] = (taus, devs)
            result['axes'][f'{name}_{"xyz"[axis]}'] = {
                'noise_density': N, 'random_walk': K, 'bias_instability': float(devs.min() / 0.664),
                'tau_n': float(tau_n), 'tau_k': float(tau_k), 'slope_fit_error_n': round(float(err_n), 3)}
        result[f'{name}_noise_density'] = float(np.mean(Ns))
        result[f'{name}_random_walk'] = float(np.mean(Ks))
    params = {
        'gyro_noise_density': result['gyro_noise_density'] * inflation,
        'gyro_random_walk': result['gyro_random_walk'] * inflation,
        'accel_noise_density': result['accel_noise_density'] * inflation,
        'accel_random_walk': result['accel_random_walk'] * inflation,
        'calibration_frequency': round(rate, 1),
    }
    if out_png:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            for key, (taus, devs) in curves.items():
                axs[0 if key.startswith('gyro') else 1].loglog(taus, devs, label=key)
            axs[0].set_title('Gyro Allan deviation (rad/s)')
            axs[1].set_title('Accel Allan deviation (m/s^2)')
            for a in axs:
                a.set_xlabel('tau (s)')
                a.grid(True, which='both', alpha=0.3)
                a.legend()
            fig.tight_layout()
            fig.savefig(out_png, dpi=110)
        except ImportError:
            pass
    return result, params


def write_params(params, result, inflation):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    stamp = datetime.datetime.now().isoformat(timespec='seconds')
    with open(CONFIG_FILE, 'w') as f:
        f.write(f'# cuVSLAM IMU noise for the MYNT D1000, measured {stamp}\n')
        f.write(f'# {result["hours"]} h static recording at {result["rate_hz"]} Hz; raw values x{inflation}\n')
        for k, v in params.items():
            f.write(f'{k}: {v:.8g}\n')


def write_basalt_calib(params):
    """Writes the packaged Basalt calibration with the measured noise; returns the path or None
    when basalt_vio_ros isn't installed."""
    try:
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
        base = os.path.join(get_package_share_directory('basalt_vio_ros'), 'config', 'basalt_calib.json')
    except (ImportError, PackageNotFoundError):
        return None
    with open(base) as f:
        calib = json.load(f)
    v = calib['value0']
    v['gyro_noise_std'] = [params['gyro_noise_density']] * 3
    v['accel_noise_std'] = [params['accel_noise_density']] * 3
    v['gyro_bias_std'] = [params['gyro_random_walk']] * 3
    v['accel_bias_std'] = [params['accel_random_walk']] * 3
    v['imu_update_rate'] = params['calibration_frequency']
    os.makedirs(os.path.dirname(BASALT_CALIB_FILE), exist_ok=True)
    with open(BASALT_CALIB_FILE, 'w') as f:
        json.dump(calib, f, indent=2)
    return BASALT_CALIB_FILE


class Recorder(Node):

    def __init__(self):
        super().__init__('imu_calibrate')
        p = self.declare_parameter
        self.duration = float(p('duration_s', 10800.0).value)
        self.warmup = float(p('warmup_s', 600.0).value)
        self.out_dir = os.path.expanduser(p('output_dir', '~/maps').value)
        self.inflation = float(p('inflation', 10.0).value)
        self.analyze_file = p('analyze_file', '').value
        self.start = time.monotonic()
        self.t, self.g, self.a = [], [], []
        self.last_log = 0.0
        self.done = False
        if not self.analyze_file:
            self.create_subscription(Imu, '/camera/imu/data_raw', self._on_imu, QoSProfile(depth=400))
            self.get_logger().info(f'Warm-up {self.warmup / 60:.0f} min, then recording '
                                   f'{self.duration / 3600:.1f} h. Keep the camera completely still.')

    def _on_imu(self, msg):
        now = time.monotonic()
        elapsed = now - self.start
        if elapsed < self.warmup or self.done:
            return
        self.t.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        w, a = msg.angular_velocity, msg.linear_acceleration
        self.g.append((w.x, w.y, w.z))
        self.a.append((a.x, a.y, a.z))
        if now - self.last_log > 600:
            self.last_log = now
            self.get_logger().info(f'Recorded {(elapsed - self.warmup) / 60:.0f} of {self.duration / 60:.0f} min '
                                   f'({len(self.t)} samples)')
        if elapsed - self.warmup >= self.duration:
            self.done = True

    def finish(self):
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        if self.analyze_file:
            raw = self.analyze_file
        else:
            if len(self.t) < 200 * 600:
                self.get_logger().error(f'Only {len(self.t)} samples; need at least 10 minutes. Not analysing.')
                return False
            os.makedirs(self.out_dir, exist_ok=True)
            raw = os.path.join(self.out_dir, f'imu_raw_{stamp}.npz')
            np.savez_compressed(raw, t=np.array(self.t, dtype=np.float64),
                                gyro=np.array(self.g, dtype=np.float64), accel=np.array(self.a, dtype=np.float64))
            self.get_logger().info(f'Saved {raw}')
        png = os.path.join(self.out_dir, f'imu_allan_{stamp}.png')
        result, params = analyze(raw, self.inflation, png)
        write_params(params, result, self.inflation)
        basalt = write_basalt_calib(params)
        self.get_logger().info('Allan analysis: ' + json.dumps(
            {k: v for k, v in result.items() if k != 'axes'}))
        self.get_logger().info(f'Wrote {CONFIG_FILE} (used by vslam.launch.py on next start): {params}')
        if basalt:
            self.get_logger().info(f'Wrote {basalt} (Basalt, used by vslam.launch.py on next start)')
        return True


def main():
    rclpy.init()
    node = Recorder()
    ok = False
    try:
        if not node.analyze_file:
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.5)
        ok = node.finish()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node.get_logger().warn('Interrupted; analysing what was recorded')
        ok = node.finish()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
