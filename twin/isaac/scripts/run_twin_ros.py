"""Run the twin with ROS 2 (PC-local this phase): house + Roomba + D1000 sensors + walking people.

  . twin\\isaac\\ros_env.ps1
  python run_twin_ros.py --seconds 60 --people 4 [--drive path] [--record E:\\twin\\runs\\drive1]

Publishes like the real robot (topic names / types / frame ids / encodings / rates from the
fresh_map8 recording):
  /camera/left/image_rect_color   sensor_msgs/Image bgr8   640x360  15 Hz  camera_left_optical_frame
  /camera/left/image_rect_mono    sensor_msgs/Image mono8                  camera_left_optical_frame
  /camera/right/image_rect_mono   sensor_msgs/Image mono8                  camera_right_optical_frame
  /camera/depth/image_rect_raw    sensor_msgs/Image 16UC1 mm, 0 = invalid (outside 0.3-7 m), registered to left
  /camera/{left,right,depth}/camera_info   real K/P (right P[3] = -fx * 0.118378 m)
  /camera/imu/data_raw            sensor_msgs/Imu 200 Hz   camera_imu_frame
  /tf_static                      base_link -> camera_link -> optical frames -> camera_imu_frame
  /wheel/odom                     nav_msgs/Odometry 30 Hz (wheel_odom -> base_link, from wheel joints)
  /ground_truth/odom              nav_msgs/Odometry 30 Hz (map -> base_link, simulator truth)
  /clock                          rosgraph_msgs/Clock (sim time)
Subscribes /cmd_vel (geometry_msgs/Twist): differential drive like roomba_driver (wheel base 0.235 m,
wheel speeds scaled to <= 0.4 m/s, 0.5 s timeout). --drive path follows the RTAB-Map robot path
with pure pursuit when no /cmd_vel is arriving. --record saves left/depth frames + poses.
ROS 2 runs through Isaac Sim's bundled Jazzy (isaacsim.ros2.core) with rclpy in this process.
"""
import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('--seconds', type=float, default=60.0, help='simulated seconds to run')
ap.add_argument('--people', type=int, default=4)
ap.add_argument('--look', default='remodel', choices=['remodel', 'scan', 'clean'])
ap.add_argument('--drive', default='none', choices=['none', 'path'])
ap.add_argument('--path-start', type=int, default=0, help='index into the RTAB-Map path to start from')
ap.add_argument('--speed', type=float, default=0.2)
ap.add_argument('--record', default=None, help='directory for captured frames')
ap.add_argument('--record-every', type=int, default=3, help='save every Nth published image frame')
ap.add_argument('--drive-log', default=None, help='CSV of the path drive (1 Hz: pose, path index, stuck flag)')
ap.add_argument('--unstick', action='store_true',
                help='path survey: when stuck for 5 s, log it and teleport ~0.8 m further along the path')
args, _ = ap.parse_known_args()

# Bundled ROS 2 Jazzy: must be on the DLL path before rclpy / the bridge load.
ROS = os.path.join(os.path.dirname(sys.executable), '..', 'Lib', 'site-packages', 'isaacsim', 'exts',
                   'isaacsim.ros2.core', 'jazzy')
ROS = os.path.abspath(ROS)
os.environ.setdefault('ROS_DISTRO', 'jazzy')
os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
os.environ.setdefault('ROS_DOMAIN_ID', '69')
os.environ.setdefault('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
os.environ['PATH'] = os.path.join(ROS, 'lib') + os.pathsep + os.environ['PATH']

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting'})

os.add_dll_directory(os.path.join(ROS, 'lib'))
sys.path.append(os.path.join(ROS, 'rclpy'))

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import rclpy  # noqa: E402
from builtin_interfaces.msg import Time as RosTime  # noqa: E402
from geometry_msgs.msg import TransformStamped, Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, Imu  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

import twin_common as tc  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager  # noqa: E402
from isaacsim.sensors.experimental.physics import IMUSensor  # noqa: E402
from twin_scene import TwinScene, app_update  # noqa: E402

rep.settings.set_render_rtx_realtime(antialiasing='DLAA')

# ---------------- scene ----------------
sc = TwinScene(look=args.look, seed=7)
sc.add_lights()
path = sc.path
x0, y0, yaw0 = path[args.path_start]
sc.add_robot(float(x0), float(y0), float(yaw0))
if args.people > 0:
    sc.enable_people_runtime()
    sc.bake_navmesh()
    sc.spawn_people(args.people)
W, H = tc.CAM['width'], tc.CAM['height']
rp_l = rep.create.render_product(sc.cam_paths['left'], (W, H))
rp_r = rep.create.render_product(sc.cam_paths['right'], (W, H))
a_rgb_l = rep.AnnotatorRegistry.get_annotator('rgb')
a_dep_l = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
a_rgb_r = rep.AnnotatorRegistry.get_annotator('rgb')
a_rgb_l.attach(rp_l)
a_dep_l.attach(rp_l)
a_rgb_r.attach(rp_r)

SimulationManager.set_physics_dt(1.0 / 200.0)
tl = omni.timeline.get_timeline_interface()
tl.play()
app_update(2)
art = Articulation(sc.robot_paths['base_link'])
dof = art.dof_names
LI, RI = dof.index('left_wheel_joint'), dof.index('right_wheel_joint')
imu = IMUSensor(sc.imu_path) if sc.imu_path else None

# ---------------- ROS ----------------
rclpy.init()
node = rclpy.create_node('isaac_twin')
rel = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
pub = {
    'color': node.create_publisher(Image, '/camera/left/image_rect_color', rel),
    'lmono': node.create_publisher(Image, '/camera/left/image_rect_mono', rel),
    'rmono': node.create_publisher(Image, '/camera/right/image_rect_mono', rel),
    'depth': node.create_publisher(Image, '/camera/depth/image_rect_raw', rel),
    'linfo': node.create_publisher(CameraInfo, '/camera/left/camera_info', rel),
    'rinfo': node.create_publisher(CameraInfo, '/camera/right/camera_info', rel),
    'dinfo': node.create_publisher(CameraInfo, '/camera/depth/camera_info', rel),
    'imu': node.create_publisher(Imu, '/camera/imu/data_raw', QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)),
    'wodom': node.create_publisher(Odometry, '/wheel/odom', rel),
    'gt': node.create_publisher(Odometry, '/ground_truth/odom', rel),
    'clock': node.create_publisher(Clock, '/clock', QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)),
    'tf_static': node.create_publisher(TFMessage, '/tf_static', latched),
}
cmd = {'v': 0.0, 'w': 0.0, 't': -1e9}


def on_cmd(msg):
    cmd.update(v=msg.linear.x, w=msg.angular.z, t=time.monotonic())


node.create_subscription(Twist, '/cmd_vel', on_cmd, 10)


def stamp(t):
    s = RosTime()
    s.sec = int(t)
    s.nanosec = int(round((t - int(t)) * 1e9)) % 1_000_000_000
    return s


def camera_info(frame, tx):
    m = CameraInfo()
    m.header.frame_id = frame
    m.width, m.height = W, H
    m.distortion_model = 'plumb_bob'
    m.d = [0.0] * 5
    fx, fy, cx, cy = tc.CAM['fx'], tc.CAM['fy'], tc.CAM['cx'], tc.CAM['cy']
    m.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    m.p = [fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return m


INFO = {'linfo': camera_info('camera_left_optical_frame', -0.0),
        'rinfo': camera_info('camera_right_optical_frame', -tc.CAM['fx'] * tc.CAM['baseline_m']),
        'dinfo': camera_info('camera_left_optical_frame', -0.0)}


def image(frame, enc, arr, t):
    m = Image()
    m.header.stamp = stamp(t)
    m.header.frame_id = frame
    m.height, m.width = arr.shape[:2]
    m.encoding = enc
    m.is_bigendian = 0
    m.step = arr.strides[0]
    m.data = np.ascontiguousarray(arr).tobytes()
    return m


def tf(parent, child, xyz, xyzw, t):
    s = TransformStamped()
    s.header.stamp = stamp(t)
    s.header.frame_id, s.child_frame_id = parent, child
    s.transform.translation.x, s.transform.translation.y, s.transform.translation.z = xyz
    q = s.transform.rotation
    q.x, q.y, q.z, q.w = xyzw
    return s


# same transforms the real robot publishes (camera_mount_tf + D1000 driver)
TF_STATIC = TFMessage(transforms=[
    tf('base_link', 'camera_link', (0.153, 0.0, 0.13), (0, 0, 0, 1), 0.0),
    tf('camera_link', 'camera_left_optical_frame', (0.0, 0.059188953399658205, 0.0), (-0.5, 0.5, -0.5, 0.5), 0.0),
    tf('camera_link', 'camera_right_optical_frame', (0.0, -0.059188953399658205, 0.0), (-0.5, 0.5, -0.5, 0.5), 0.0),
    tf('camera_left_optical_frame', 'camera_imu_frame', (0.0477928465820604, -0.0020221539578519996, -0.0012811425097398002),
       (0.9999874724368888, 0.0021644046093707487, 0.0034723934920541906, -0.002883193612270431), 0.0),
])
pub['tf_static'].publish(TF_STATIC)


IMU_RNG = np.random.default_rng(0)
IMU_ACC_STD, IMU_GYRO_STD = 0.018, 0.0011


def imu_cb(dt, ctx):
    if imu is None:
        return
    r = imu.get_sensor_reading()
    if not r.is_valid:
        return
    t = SimulationManager.get_simulation_time()
    m = Imu()
    m.header.stamp = stamp(t)
    m.header.frame_id = 'camera_imu_frame'
    # white noise matched to the real D1000 at rest (fresh_map8: accel std ~0.018 m/s^2, gyro ~0.001 rad/s)
    gn, an = IMU_RNG.normal(0, IMU_GYRO_STD, 3), IMU_RNG.normal(0, IMU_ACC_STD, 3)
    m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (
        r.angular_velocity_x + gn[0], r.angular_velocity_y + gn[1], r.angular_velocity_z + gn[2])
    m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (
        r.linear_acceleration_x + an[0], r.linear_acceleration_y + an[1], r.linear_acceleration_z + an[2])
    m.orientation_covariance[0] = -1.0     # the D1000 publishes no orientation
    pub['imu'].publish(m)


imu_sub = SimulationManager.register_callback(imu_cb, SimulationEvent.PHYSICS_POST_STEP)


def pose():
    p, q = art.get_world_poses()
    p = np.asarray(p.numpy() if hasattr(p, 'numpy') else p)[0]
    q = np.asarray(q.numpy() if hasattr(q, 'numpy') else q)[0]       # wxyz
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return p, q, yaw


def odom_msg(frame, t, x, y, z, qwxyz, v, w):
    m = Odometry()
    m.header.stamp = stamp(t)
    m.header.frame_id = frame
    m.child_frame_id = 'base_link'
    m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = float(x), float(y), float(z)
    o = m.pose.pose.orientation
    o.w, o.x, o.y, o.z = (float(c) for c in qwxyz)
    m.twist.twist.linear.x, m.twist.twist.angular.z = float(v), float(w)
    return m


def wheel_targets(v, w):
    """roomba_rs kinematics::twist_to_wheel_speeds, then m/s -> rad/s."""
    b, vmax, r = tc.DRIVE['wheel_base_m'], tc.DRIVE['max_wheel_speed_m_s'], tc.DRIVE['wheel_radius_m']
    left, right = v - w * b / 2, v + w * b / 2
    peak = max(abs(left), abs(right))
    if peak > vmax > 0:
        left, right = left * vmax / peak, right * vmax / peak
    return left / r, right / r


class PurePursuit:
    def __init__(self, pts, start, lookahead=0.45):
        self.pts, self.i, self.L = pts, start, lookahead

    def __call__(self, x, y, yaw, v):
        while self.i < len(self.pts) - 1 and math.hypot(self.pts[self.i][0] - x, self.pts[self.i][1] - y) < self.L:
            self.i += 1
        if self.i >= len(self.pts) - 1 and math.hypot(self.pts[-1][0] - x, self.pts[-1][1] - y) < 0.2:
            return 0.0, 0.0
        tx, ty = self.pts[self.i][0], self.pts[self.i][1]
        alpha = math.atan2(ty - y, tx - x) - yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        if abs(alpha) > 1.2:                      # rotate in place towards the next waypoint
            return 0.0, math.copysign(0.8, alpha)
        return v, max(-0.8, min(0.8, 2 * v * math.sin(alpha) / self.L))


pp = PurePursuit(path, args.path_start) if args.drive == 'path' else None
rec_dir = args.record
if rec_dir:
    for s in ('left', 'depth'):
        os.makedirs(os.path.join(rec_dir, s), exist_ok=True)
    rec_csv = open(os.path.join(rec_dir, 'ground_truth.csv'), 'w', newline='')
    rec_w = csv.writer(rec_csv)
    rec_w.writerow(['t', 'x', 'y', 'yaw', 'left_png'])

# wheel odometry state
wo = {'x': 0.0, 'y': 0.0, 'th': 0.0, 'pl': None, 'pr': None, 't': None}
t_start = SimulationManager.get_simulation_time()
wall0 = time.monotonic()
n_upd = n_img = 0
import cv2  # noqa: E402

drive_log = None
if args.drive_log:
    os.makedirs(os.path.dirname(os.path.abspath(args.drive_log)), exist_ok=True)
    drive_log = csv.writer(open(args.drive_log, 'w', newline='', buffering=1))
    drive_log.writerow(['t', 'x', 'y', 'yaw_deg', 'path_idx', 'v_cmd', 'w_cmd', 'moved_5s_m', 'stuck'])
hist = []      # (t, x, y) for the stuck detector
stuck_events = []


def teleport_ahead():
    p0, _, _ = pose()
    j = pp.i
    while j < len(path) - 1 and math.hypot(path[j][0] - p0[0], path[j][1] - p0[1]) < 0.8:
        j += 1
    x_, y_, yaw_ = path[j]
    art.set_world_poses(np.array([[x_, y_, float(p0[2])]], np.float32),
                        np.array([[math.cos(yaw_ / 2), 0, 0, math.sin(yaw_ / 2)]], np.float32))
    art.set_velocities(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))
    pp.i = j
    hist.clear()
    return j
print('LOOP_START', flush=True)

while app.is_running():
    t = SimulationManager.get_simulation_time()
    if t - t_start >= args.seconds:
        break
    rclpy.spin_once(node, timeout_sec=0.0)
    p, q, yaw = pose()
    if time.monotonic() - cmd['t'] < 0.5:
        v, w = cmd['v'], cmd['w']
    elif pp is not None:
        v, w = pp(p[0], p[1], yaw, args.speed)
    else:
        v, w = 0.0, 0.0
    vl, vr = wheel_targets(v, w)
    tgt = np.zeros((1, len(dof)), np.float32)
    tgt[0, LI], tgt[0, RI] = vl, vr
    art.set_dof_velocity_targets(tgt)
    app_update(1)
    n_upd += 1
    if drive_log is not None and n_upd % 60 == 0:      # 1 Hz drive log
        p1, _, yaw1 = pose()
        t1 = SimulationManager.get_simulation_time()
        hist.append((t1, p1[0], p1[1], yaw1))
        old = [h for h in hist if t1 - h[0] >= 5.0]
        moved = math.hypot(p1[0] - old[-1][1], p1[1] - old[-1][2]) if old else float('nan')
        turned = abs((yaw1 - old[-1][3] + math.pi) % (2 * math.pi) - math.pi) if old else float('nan')
        # stuck: no progress while driving, or no rotation while turning in place
        stuck = int(bool(old) and ((moved < 0.05 and abs(v) > 0) or (abs(v) == 0 and abs(w) > 0.3 and turned < 0.1)))
        drive_log.writerow([f'{t1:.2f}', f'{p1[0]:.3f}', f'{p1[1]:.3f}', f'{math.degrees(yaw1):.1f}',
                            pp.i if pp else -1, f'{v:.2f}', f'{w:.2f}', f'{moved:.3f}', stuck])
        if n_upd % 600 == 0:
            print(f'DRIVE t {t1:.0f} idx {pp.i if pp else -1} pose {p1[0]:.2f} {p1[1]:.2f} stuck {stuck}', flush=True)
        if stuck and args.unstick and pp is not None:
            j = teleport_ahead()
            stuck_events.append({'t': round(t1, 1), 'x': round(float(p1[0]), 3), 'y': round(float(p1[1]), 3),
                                 'path_idx': j})
            print(f'STUCK t {t1:.0f} at {p1[0]:.2f} {p1[1]:.2f} -> teleported to path idx {j}', flush=True)
        if pp is not None and pp.i >= len(path) - 1 and math.hypot(path[-1][0] - p1[0], path[-1][1] - p1[1]) < 0.25:
            print('PATH_DONE', flush=True)
            break
    t = SimulationManager.get_simulation_time()
    ck = Clock()
    ck.clock = stamp(t)
    pub['clock'].publish(ck)
    if n_upd % 2 == 0:       # 30 Hz odometry
        p, q, yaw = pose()
        pub['gt'].publish(odom_msg('map', t, p[0], p[1], p[2], q, v, w))
        jp = art.get_dof_positions()
        jp = np.asarray(jp.numpy() if hasattr(jp, 'numpy') else jp)[0]
        if wo['pl'] is not None:
            r_ = tc.DRIVE['wheel_radius_m']
            dl, dr = (jp[LI] - wo['pl']) * r_, (jp[RI] - wo['pr']) * r_
            dth = (dr - dl) / tc.DRIVE['wheel_base_m']
            ds = (dl + dr) / 2
            wo['x'] += ds * math.cos(wo['th'] + dth / 2)
            wo['y'] += ds * math.sin(wo['th'] + dth / 2)
            wo['th'] += dth
            dt = max(t - wo['t'], 1e-6)
            qz = (math.cos(wo['th'] / 2), 0.0, 0.0, math.sin(wo['th'] / 2))
            pub['wodom'].publish(odom_msg('wheel_odom', t, wo['x'], wo['y'], 0.0, qz, ds / dt, dth / dt))
        wo['pl'], wo['pr'], wo['t'] = jp[LI], jp[RI], t
    if n_upd % 4 == 0:       # 15 Hz camera, like the real D1000 at output_scale 0.5
        rgb_l = np.asarray(a_rgb_l.get_data())
        rgb_r = np.asarray(a_rgb_r.get_data())
        dep = np.asarray(a_dep_l.get_data())
        if rgb_l.size == 0 or rgb_r.size == 0 or dep.size == 0:
            continue
        bgr = np.ascontiguousarray(rgb_l[:, :, 2::-1])
        mono_l = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mono_r = cv2.cvtColor(np.ascontiguousarray(rgb_r[:, :, 2::-1]), cv2.COLOR_BGR2GRAY)
        mm = np.where(np.isfinite(dep) & (dep >= tc.CAM['depth_min_m']) & (dep <= tc.CAM['depth_max_m']),
                      dep * 1000.0, 0).astype(np.uint16)
        pub['color'].publish(image('camera_left_optical_frame', 'bgr8', bgr, t))
        pub['lmono'].publish(image('camera_left_optical_frame', 'mono8', mono_l, t))
        pub['rmono'].publish(image('camera_right_optical_frame', 'mono8', mono_r, t))
        pub['depth'].publish(image('camera_left_optical_frame', '16UC1', mm, t))
        for k, m in INFO.items():
            m.header.stamp = stamp(t)
            pub[k].publish(m)
        n_img += 1
        if rec_dir and n_img % args.record_every == 0:
            name = f'{int(round(t * 1e9))}.png'
            cv2.imwrite(os.path.join(rec_dir, 'left', name), bgr)
            cv2.imwrite(os.path.join(rec_dir, 'depth', name), mm)
            p, q, yaw = pose()
            rec_w.writerow([f'{t:.4f}', f'{p[0]:.4f}', f'{p[1]:.4f}', f'{yaw:.4f}', name])

sim_s = SimulationManager.get_simulation_time() - t_start
wall_s = time.monotonic() - wall0
p, q, yaw = pose()
print(f'RUN_DONE sim {sim_s:.1f} s wall {wall_s:.1f} s rtf {sim_s / max(wall_s, 1e-6):.2f} '
      f'images {n_img} final_pose {p[0]:.2f} {p[1]:.2f} {math.degrees(yaw):.0f} path_idx {pp.i if pp else None}',
      flush=True)
if rec_dir:
    rec_csv.close()
if args.drive_log:
    import json as _json
    with open(os.path.splitext(args.drive_log)[0] + '_stuck.json', 'w') as f:
        _json.dump(stuck_events, f, indent=1)
    print('STUCK_EVENTS', len(stuck_events), flush=True)
SimulationManager.deregister_callback(imu_sub)
tl.stop()
node.destroy_node()
rclpy.shutdown()
app.close()
