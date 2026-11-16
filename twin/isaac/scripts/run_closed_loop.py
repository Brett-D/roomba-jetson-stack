"""Isaac Sim side of the closed loop, with the GUI window (to watch) and a follow camera.

  run_closed_loop.ps1                    # everything (Zenoh router + WSL stack + this)
  python run_closed_loop.py [--drive path|ros|stop] [--people 2] [--ros]

The house (remodelled upstairs + downstairs), the Roomba 550 + D1000, walking people. The robot either
follows the recorded RTAB-Map path (--drive path, a demo without the stack) or is driven by the robot's
real stack in WSL (--drive ros): its Rust roomba_driver talks the Open Interface to
wsl/isaac_roomba.py, which turns drive commands into /isaac/roomba/wheel_cmd and streams the sensor
values this script publishes on /isaac/roomba/state.

ROS 2 (Isaac Sim's bundled Jazzy, rmw_zenoh_cpp, client of the WSL router tcp/127.0.0.1:7447),
started with --ros or later from the control file. Publishes like the D1000 driver (15 Hz images,
200 Hz IMU, camera_info, camera TF) plus /clock, /ground_truth/odom, /isaac/people, /isaac/roomba/state.

Control file (polled every second, default E:\\twin\\run\\isaac_control.json), e.g.
  {"drive": "ros", "ros": true}    {"drive": "path"}    {"teleport": [0, 0, 0], "id": 3}   {"quit": true}
Status: one line in E:\\twin\\isaac_NOW.txt (what is on screen, since when) and E:\\twin\\run\\isaac_state.json.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('--drive', default='path', choices=['path', 'ros', 'stop'])
ap.add_argument('--people', type=int, default=2)
ap.add_argument('--ros', action='store_true', help='start ROS 2 right away (else when the control file asks)')
ap.add_argument('--speed', type=float, default=0.25, help='path-following speed (m/s)')
ap.add_argument('--control', default=r'E:\twin\run\isaac_control.json')
ap.add_argument('--status', default=r'E:\twin\isaac_NOW.txt')
ap.add_argument('--state', default=r'E:\twin\run\isaac_state.json')
ap.add_argument('--viewer', default='file', choices=['file', 'kit', 'none'],
                help="file: headless, live view frames to --view-file for twin_viewer.py (the Kit GUI window crashes "
                     "in the NVIDIA Vulkan driver on this PC); kit: Isaac Sim's own window")
ap.add_argument('--view-file', default=r'E:\twin\run\view.jpg')
ap.add_argument('--view-every', type=int, default=2, help='app updates (1/60 s sim) per view frame')
ap.add_argument('--record', default='', help='record the follow camera (1920x1080) + robot camera (640x360) at 30 fps '
                                           'of sim time: <name>_follow.mp4, <name>_robotcam.mp4, <name>_frames.jsonl')
ap.add_argument('--render-hz', type=int, default=60, help='app updates per simulated second')
ap.add_argument('--vp-width', type=int, default=1280, help='GUI viewport render width (16:9)')
ap.add_argument('--aa', default='DLSS', choices=['DLSS', 'DLAA'],
                help='DLSS renders at a lower internal resolution (much faster); DLAA at full resolution')
ap.add_argument('--dlss-mode', type=int, default=1, help='0 performance, 1 balanced, 2 quality')
ap.add_argument('--light-scale', type=float, default=2.5,
                help='room/dome light multiplier (the raw renders are ~3x darker than the real D1000 auto-exposed '
                     'frames, so this is the robot-camera realism setting; dataset generation is separate)')
ap.add_argument('--view-gain', type=float, default=2.0,
                help='extra exposure for the GUI viewport and the recorded follow camera only (RTX film ISO x gain). '
                     'RTX tonemapping is global (per-camera exposure and per-render-product overrides are ignored), '
                     'so the robot cameras are mapped back to the --light-scale look with a per-channel LUT '
                     'calibrated at start-up: what the stack sees does not change. 1 = off')
ap.add_argument('--robot-view', type=int, default=1, help='Kit viewer: second viewport with the robot camera')
ap.add_argument('--start', default='dock', help="'dock' (map origin, the recorded path's start) or x,y,yaw")
args, _ = ap.parse_known_args()
os.makedirs(os.path.dirname(args.control), exist_ok=True)

# Bundled ROS 2 Jazzy on the DLL path before anything loads rclpy. Zenoh client of the WSL router.
ROS = os.path.abspath(os.path.join(os.path.dirname(sys.executable), '..', 'Lib', 'site-packages', 'isaacsim', 'exts',
                                   'isaacsim.ros2.core', 'jazzy'))
os.environ.setdefault('ROS_DISTRO', 'jazzy')
os.environ.setdefault('AMENT_PREFIX_PATH', ROS)     # rmw_zenoh finds its default config there
os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_zenoh_cpp')
os.environ.setdefault('ZENOH_CONFIG_OVERRIDE', 'mode="client";connect/endpoints=["tcp/127.0.0.1:7447"]')
os.environ.setdefault('ROS_DOMAIN_ID', '60')
os.environ['PATH'] = os.path.join(ROS, 'lib') + os.pathsep + os.environ['PATH']

# MSI Afterburner / RivaTuner Statistics Server registers an implicit Vulkan layer (VK_LAYER_RTSS) that
# hooks every swapchain: with it, Isaac Sim's GUI window crashed ~3 s after start inside the NVIDIA Vulkan driver
# (integer divide by zero in nvoglv64.dll creating the swapchain). Disable it (and Steam's overlay layers) for
# this process only; RTSS/Afterburner themselves are left alone.
for _k in ('DISABLE_RTSS_LAYER', 'DISABLE_VK_LAYER_VALVE_steam_overlay_1', 'DISABLE_VK_LAYER_VALVE_steam_fossilize_1'):
    os.environ.setdefault(_k, '1')
T_START = time.time()


def status(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    try:
        with open(args.status, 'w', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError:
        pass
    print('STATUS', line, flush=True)


status('Isaac Sim is starting (window opens in ~1-2 min) ...')

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': args.viewer != 'kit', 'renderer': 'RaytracedLighting', 'width': 1600, 'height': 900,
                     'window_width': 1920, 'window_height': 1080})

os.add_dll_directory(os.path.join(ROS, 'lib'))
sys.path.append(os.path.join(ROS, 'rclpy'))

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.physx  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

import twin_common as tc  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager  # noqa: E402
from isaacsim.sensors.experimental.physics import IMUSensor  # noqa: E402
from twin_scene import TwinScene, app_update  # noqa: E402

rep.settings.set_render_rtx_realtime(antialiasing=args.aa)
if args.aa == 'DLSS':
    import carb.settings
    carb.settings.get_settings().set('/rtx/post/dlss/execMode', args.dlss_mode)

# ---------------------------------------------------------------- scene
sc = TwinScene(look='remodel', seed=11)
sc.add_lights()
if args.light_scale != 1.0:
    from pxr import Usd, UsdLux
    for root in ('/World/House', '/World/HouseF1', '/World'):
        pr = sc.stage.GetPrimAtPath(root)
        if not pr:
            continue
        for q in Usd.PrimRange(pr):
            if q.HasAPI(UsdLux.LightAPI) and not q.GetCustomDataByKey('twin:scaled'):
                a_ = UsdLux.LightAPI(q).GetIntensityAttr()
                if a_ and a_.Get() is not None:
                    a_.Set(float(a_.Get()) * args.light_scale)
                    q.SetCustomDataByKey('twin:scaled', True)
path = sc.path
if args.start == 'dock':
    x0, y0, yaw0 = (float(v) for v in path[0])
else:
    x0, y0, yaw0 = (float(v) for v in args.start.split(','))
sc.add_robot(x0, y0, yaw0)
status('Isaac Sim: house loaded, adding people ...')
if args.people > 0:
    try:
        sc.enable_people_runtime()
        sc.bake_navmesh()
        sc.spawn_people(args.people)
    except Exception as e:  # noqa: BLE001
        print('PEOPLE_FAILED', e, flush=True)
stage = sc.stage

# follow camera (collision-aware boom behind the robot) as the main viewport's camera
follow = UsdGeom.Camera.Define(stage, '/World/FollowCam')
follow.CreateFocalLengthAttr(16.0)
follow.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200))
follow_op = UsdGeom.Xformable(follow).AddTransformOp()
try:
    from omni.kit.viewport.utility import get_active_viewport
    _vp = get_active_viewport()
    _vp.camera_path = '/World/FollowCam'
    # render the GUI viewport at 1280x720 (upscaled to the window): at the window's native ~2400x1350 the
    # RTX real-time viewport alone kept the GPU ~70 % busy and the sim at ~0.2x real time
    _vp.resolution = (args.vp_width, args.vp_width * 9 // 16)
except Exception as e:  # noqa: BLE001
    print('VIEWPORT_CAM_FAILED', e, flush=True)

VW, VH = 1280, 720
W, H = tc.CAM['width'], tc.CAM['height']
view = {}
if args.viewer == 'file':
    view['rp'] = rep.create.render_product('/World/FollowCam', (VW, VH))
    view['ann'] = rep.AnnotatorRegistry.get_annotator('rgb')
    view['ann'].attach(view['rp'])
# the robot's left camera (live-view inset now, ROS images later)
rp_left = rep.create.render_product(sc.cam_paths['left'], (W, H))
ann_left = {'rgb': rep.AnnotatorRegistry.get_annotator('rgb'), 'dep': rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')}
ann_left['rgb'].attach(rp_left)
ann_left['dep'].attach(rp_left)
robot_rps = [rp_left]          # rendered only every CAM_EVERY-th update (RTF), see the main loop
video = {'w': None}
CAM_LUT = {'lut': None}


def robot_rgb(a):
    """A robot camera's RGB(A) as the stack must see it: the --view-gain exposure boost undone (see calibrate_lut)."""
    img = np.asarray(a.get_data())
    lut = CAM_LUT['lut']
    if lut is None or img.ndim != 3 or img.size == 0:
        return img
    out = img.copy()
    for c in range(3):
        out[:, :, c] = lut[c][img[:, :, c]]
    return out


def calibrate_lut(n_settle=30):
    """Render the left camera at the base film ISO and at ISO x view-gain from the same pose; per channel and
    8-bit level of the bright image, the median base level -> LUT (monotone, gaps interpolated)."""
    import carb.settings
    st = carb.settings.get_settings()
    key = '/rtx/post/tonemap/filmIso'
    iso0 = float(st.get(key) or 100.0)

    def grab(iso):
        st.set(key, iso)
        for _ in range(n_settle):
            app_update(1)
        return np.asarray(ann_left['rgb'].get_data())[:, :, :3].astype(np.int32)
    base = grab(iso0)
    bright = grab(iso0 * args.view_gain)
    lut = np.zeros((3, 256), np.uint8)
    xs = np.arange(256)
    model = np.clip(xs * args.view_gain ** (-1 / 2.2), 0, 255)       # sRGB-like fallback outside the data
    for c in range(3):
        b, a = bright[:, :, c].ravel(), base[:, :, c].ravel()
        known, vals = [0], [0.0]
        for v in range(1, 256):
            sel = a[b == v]
            if sel.size >= 20:
                known.append(v)
                vals.append(float(np.median(sel)))
        f = np.interp(xs, known, vals)
        hi = known[-1]
        if hi < 255:                                                   # above the brightest seen level
            f[hi + 1:] = np.maximum(f[hi], model[hi + 1:] * (vals[-1] / max(model[hi], 1e-6)))
        lut[c] = np.clip(np.maximum.accumulate(np.round(f)), 0, 255).astype(np.uint8)
    comp = np.stack([lut[c][bright[:, :, c]] for c in range(3)], -1).astype(np.int32)
    err = float(np.abs(comp - base).mean())
    print(f'CAMERA_LUT view gain x{args.view_gain}: base mean {base.mean():.1f}, bright {bright.mean():.1f}, '
          f'compensated {comp.mean():.1f}, mean abs error {err:.2f} levels', flush=True)
    CAM_LUT['lut'] = lut
    st.set(key, iso0 * args.view_gain)


class Recorder:
    """Frames per sim step (30 fps of sim time) piped to ffmpeg: smooth playback whatever the real-time factor."""
    RW, RH = 1920, 1080

    def __init__(self, base):
        import subprocess
        os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
        self.base = base
        self.rp = rep.create.render_product('/World/FollowCam', (self.RW, self.RH))
        self.ann = rep.AnnotatorRegistry.get_annotator('rgb')
        self.ann.attach(self.rp)

        def enc(path, w, h):
            return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                                     '-s', f'{w}x{h}', '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast',
                                     '-crf', '14', '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
        self.main = enc(base + '_follow.mp4', self.RW, self.RH)
        self.rob = enc(base + '_robotcam.mp4', W, H)
        self.meta = open(base + '_frames.jsonl', 'w')
        self.n = 0
        self.skipped = 0

    def frame(self, t, p, yaw):
        img = np.asarray(self.ann.get_data())
        rob = robot_rgb(ann_left['rgb'])
        if img.size == 0 or rob.size == 0:
            self.skipped += 1
            return
        self.main.stdin.write(np.ascontiguousarray(img[:, :, 2::-1]).tobytes())
        self.rob.stdin.write(np.ascontiguousarray(rob[:, :, 2::-1]).tobytes())
        self.meta.write(json.dumps({'i': self.n, 't': round(t, 4), 'x': round(float(p[0]), 3), 'y': round(float(p[1]), 3),
                                    'yaw': round(yaw, 3), 'drive': S['drive'], 'teleports': S['teleports'],
                                    'bumps': S['bumps'], 'people': [[round(a, 2), round(b, 2)] for a, b in people_xy()]}) + '\n')
        self.n += 1

    def close(self):
        for e in (self.main, self.rob):
            e.stdin.close()
            e.wait()
        self.meta.close()
        self.ann.detach()
        self.rp.destroy()
        print('RECORDED', self.base, self.n, 'frames, skipped', self.skipped, flush=True)

SimulationManager.set_physics_dt(1.0 / 200.0)
tl = omni.timeline.get_timeline_interface()
if args.render_hz != 60:
    tl.set_time_codes_per_second(args.render_hz)
    tl.set_target_framerate(args.render_hz)
CAM_EVERY = max(1, args.render_hz // 15)
tl.play()
app_update(2 if args.render_hz == 60 else 12)      # a new frame rate is applied lazily and resets the physics view
art = Articulation(sc.robot_paths['base_link'])
dof = art.dof_names
LI, RI = dof.index('left_wheel_joint'), dof.index('right_wheel_joint')
imu = IMUSensor(sc.imu_path) if sc.imu_path else None
sq = omni.physx.get_physx_scene_query_interface()
R_BODY = 0.17
WR, WB = tc.DRIVE['wheel_radius_m'], tc.DRIVE['wheel_base_m']


def npy(x):
    return np.asarray(x.numpy() if hasattr(x, 'numpy') else x)


def pose():
    p, q = art.get_world_poses()
    p, q = npy(p)[0], npy(q)[0]       # wxyz
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return p, q, yaw


def teleport(x, y, yaw):
    p, _, _ = pose()
    art.set_world_poses(np.array([[x, y, float(p[2])]], np.float32),
                        np.array([[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]], np.float32))
    art.set_velocities(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))
    S['prev_pose'] = None          # not a push (see the main loop)


def people_xy():
    try:
        from omni.metropolis.pipeline.agent import AgentsManager
        out = []
        for a in AgentsManager.get_instance().get_runtime_agent_instances():
            p = a.get_world_position()
            if p is not None:
                out.append((float(p[0]), float(p[1])))
        return out
    except Exception:  # noqa: BLE001
        return []


def disable_people_colliders():
    """The behaviour runtime gives each walking person a kinematic BodyCollider (created after play). Kinematic
    bodies shove the dynamic Roomba: a person walking into it moved it 0.6-1 m and spun it (cl_run3, cl_run6), and
    undoing those pushes every step (cl_run8) jolted the IMU and broke the VIO. So switch the body colliders off
    (their avoidance triggers stay); the bumper model in sense() still feels people's legs. Returns how many."""
    from pxr import Usd, UsdPhysics
    n_ = 0
    for pp in sc.people:
        root = sc.stage.GetPrimAtPath(pp['skelroot'])
        if not root:
            continue
        for q in Usd.PrimRange(root):
            if q.GetName() == 'BodyCollider' and q.HasAPI(UsdPhysics.CollisionAPI):
                attr = UsdPhysics.CollisionAPI(q).GetCollisionEnabledAttr()
                if attr.Get() is not False:
                    UsdPhysics.CollisionAPI(q).CreateCollisionEnabledAttr(False)
                    n_ += 1
    return n_


# ---------------------------------------------------------------- bumper / light bumpers / cliffs
LIGHT_ANG = [math.radians(a) for a in (65, 38, 12, -12, -38, -65)]   # L, FL, CL, CR, FR, R
CLIFF_ANG = [math.radians(a) for a in (60, 20, -20, -60)]            # L, FL, FR, R


def ray(origin, d, length):
    h = sq.raycast_closest(origin, d, length)
    return h['distance'] if h.get('hit') else None


def sense(p, yaw, ppl):
    """Bump left/right, 6 light-bumper signals (0-4095, ~235 at 1-2 cm), 4 cliff flags, wall signal."""
    x, y = float(p[0]), float(p[1])
    light, bl, br = [], 0, 0
    for a in LIGHT_ANG:
        c, s = math.cos(yaw + a), math.sin(yaw + a)
        o = (x + c * (R_BODY + 0.004), y + s * (R_BODY + 0.004), 0.045)
        d = ray(o, (c, s, 0.0), 0.12)
        dp = min((math.hypot(px - o[0], py - o[1]) - 0.15 for px, py in ppl
                  if (px - x) * c + (py - y) * s > 0), default=None)   # people (pushes undone in the loop): legs ~0.15 m
        if dp is not None and dp < 0.12:
            d = dp if d is None else min(d, dp)
        sig = 0 if d is None else int(max(0.0, 260.0 - 2500.0 * max(d, 0.0)))
        light.append(sig)
        if d is not None and d < 0.008:          # touching: the bumper closes
            if a > 0.2:
                bl = 1
            elif a < -0.2:
                br = 1
            else:
                bl = br = 1
    cliffs = []
    for a in CLIFF_ANG:
        c, s = math.cos(yaw + a), math.sin(yaw + a)
        o = (x + c * (R_BODY - 0.02), y + s * (R_BODY - 0.02), 0.03)
        cliffs.append(1 if ray(o, (0.0, 0.0, -1.0), 0.08) is None else 0)
    c, s = math.cos(yaw - 1.4), math.sin(yaw - 1.4)
    dw = ray((x + c * (R_BODY + 0.004), y + s * (R_BODY + 0.004), 0.045), (c, s, 0.0), 0.08)
    wall = 0 if dw is None else int(max(0.0, 200.0 - 2500.0 * dw))
    return bl, br, light, cliffs, wall


# ---------------------------------------------------------------- driving
class PurePursuit:
    def __init__(self, pts, start, lookahead=0.45):
        self.pts, self.i, self.L = pts, start, lookahead

    def __call__(self, x, y, yaw, v):
        while self.i < len(self.pts) - 1 and math.hypot(self.pts[self.i][0] - x, self.pts[self.i][1] - y) < self.L:
            self.i += 1
        if self.i >= len(self.pts) - 1 and math.hypot(self.pts[-1][0] - x, self.pts[-1][1] - y) < 0.25:
            self.i = 0                           # loop the demo
        tx, ty = self.pts[self.i][0], self.pts[self.i][1]
        alpha = (math.atan2(ty - y, tx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(alpha) > 1.0:
            return 0.0, math.copysign(0.7, alpha)
        return v, max(-0.8, min(0.8, 2 * v * math.sin(alpha) / self.L))


def wheel_targets(v, w):
    left, right = v - w * WB / 2, v + w * WB / 2
    peak = max(abs(left), abs(right))
    vmax = tc.DRIVE['max_wheel_speed_m_s']
    if peak > vmax > 0:
        left, right = left * vmax / peak, right * vmax / peak
    return left, right


def clean_path(pts, clearance=0.27, search=0.7):
    """Move recorded-path waypoints that are too close to plan walls/furniture (doorways where plan and map
    disagree by 0.1-0.3 m) to the nearest point with clearance, then smooth: the demo drive gets stuck less."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    res, (ox, oy) = sc.meta['resolution'], sc.meta['origin'][:2]
    d = ndimage.distance_transform_edt(~sc.obstacles) * res
    ok = np.argwhere(d >= clearance)
    tree = cKDTree(ok)
    out = []
    for x, y, yaw in pts:
        r, c = int((y - oy) / res), int((x - ox) / res)
        if 0 <= r < d.shape[0] and 0 <= c < d.shape[1] and d[r, c] < clearance:
            dist, k = tree.query((r, c), distance_upper_bound=search / res)
            if np.isfinite(dist):
                r, c = ok[k]
                x, y = ox + (c + 0.5) * res, oy + (r + 0.5) * res
        out.append([x, y, yaw])
    a = np.array(out)
    sm = a.copy()
    for i in range(len(a)):
        lo, hi = max(0, i - 2), min(len(a), i + 3)
        sm[i, :2] = a[lo:hi, :2].mean(0)
    return [tuple(v) for v in sm]


drive_path = clean_path(path)
pp = PurePursuit(drive_path, 0)
S = {'drive': args.drive, 'ros': False, 'wheel_cmd': (0.0, 0.0), 'wheel_cmd_t': -1e9, 'cmd_vel': (0.0, 0.0),
     'cmd_vel_t': -1e9, 'stuck_hist': [], 'teleports': 0, 'control_mtime': 0.0, 'control_id': None,
     'bumps': 0, 'last_bump': 0, 'cliff_events': 0}

# ---------------------------------------------------------------- ROS (deferred)
R = {}


def enable_ros():
    """rclpy on Zenoh + all publishers, render products for the robot cameras."""
    if S['ros']:
        return
    import rclpy
    from builtin_interfaces.msg import Time as RosTime
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, Imu
    from std_msgs.msg import Float64MultiArray, String
    from tf2_msgs.msg import TFMessage

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
        'gt': node.create_publisher(Odometry, '/ground_truth/odom', rel),
        'clock': node.create_publisher(Clock, '/clock', QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)),
        'tf_static': node.create_publisher(TFMessage, '/tf_static', latched),
        'state': node.create_publisher(Float64MultiArray, '/isaac/roomba/state', QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)),
        'people': node.create_publisher(String, '/isaac/people', rel),
        'sim': node.create_publisher(String, '/isaac/status', rel),
    }

    def on_wheel(m):
        if len(m.data) >= 2:
            S['wheel_cmd'] = (float(m.data[0]), float(m.data[1]))
            S['wheel_cmd_t'] = time.monotonic()

    def on_cmd(m):
        S['cmd_vel'] = (m.linear.x, m.angular.z)
        S['cmd_vel_t'] = time.monotonic()

    node.create_subscription(Float64MultiArray, '/isaac/roomba/wheel_cmd', on_wheel, 10)
    node.create_subscription(Twist, '/isaac/teleop_cmd_vel', on_cmd, 10)

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

    def tf(parent, child, xyz, xyzw):
        s = TransformStamped()
        s.header.frame_id, s.child_frame_id = parent, child
        s.transform.translation.x, s.transform.translation.y, s.transform.translation.z = xyz
        q = s.transform.rotation
        q.x, q.y, q.z, q.w = xyzw
        return s

    # what the D1000 driver publishes (base_link -> camera_link comes from robot.launch.py's camera_mount_tf)
    pub['tf_static'].publish(TFMessage(transforms=[
        tf('camera_link', 'camera_left_optical_frame', (0.0, 0.059188953399658205, 0.0), (-0.5, 0.5, -0.5, 0.5)),
        tf('camera_link', 'camera_right_optical_frame', (0.0, -0.059188953399658205, 0.0), (-0.5, 0.5, -0.5, 0.5)),
        tf('camera_left_optical_frame', 'camera_imu_frame', (0.0477928465820604, -0.0020221539578519996, -0.0012811425097398002),
           (0.9999874724368888, 0.0021644046093707487, 0.0034723934920541906, -0.002883193612270431)),
    ]))
    rp_r = rep.create.render_product(sc.cam_paths['right'], (W, H))
    ann = {'rgb_l': ann_left['rgb'], 'dep': ann_left['dep'], 'rgb_r': rep.AnnotatorRegistry.get_annotator('rgb')}
    ann['rgb_r'].attach(rp_r)
    R['rp_r'] = rp_r
    robot_rps.append(rp_r)
    imu_rng = np.random.default_rng(0)

    def imu_cb(dt, ctx):
        t = SimulationManager.get_simulation_time()
        ck = Clock()
        ck.clock = stamp(t)
        pub['clock'].publish(ck)
        if imu is None:
            return
        r = imu.get_sensor_reading()
        if not r.is_valid:
            return
        m = Imu()
        m.header.stamp = stamp(t)
        m.header.frame_id = 'camera_imu_frame'
        gn, an = imu_rng.normal(0, 0.0011, 3), imu_rng.normal(0, 0.018, 3)
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (
            r.angular_velocity_x + gn[0], r.angular_velocity_y + gn[1], r.angular_velocity_z + gn[2])
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (
            r.linear_acceleration_x + an[0], r.linear_acceleration_y + an[1], r.linear_acceleration_z + an[2])
        m.orientation_covariance[0] = -1.0
        pub['imu'].publish(m)

    R.update(rclpy=rclpy, node=node, pub=pub, stamp=stamp, Image=Image, Odometry=Odometry, String=String,
             Float64MultiArray=Float64MultiArray, info={'linfo': camera_info('camera_left_optical_frame', 0.0),
                                                       'rinfo': camera_info('camera_right_optical_frame', -tc.CAM['fx'] * tc.CAM['baseline_m']),
                                                       'dinfo': camera_info('camera_left_optical_frame', 0.0)},
             ann=ann, imu_sub=SimulationManager.register_callback(imu_cb, SimulationEvent.PHYSICS_POST_STEP))
    S['ros'] = True
    print('ROS_ENABLED', os.environ.get('RMW_IMPLEMENTATION'), os.environ.get('ROS_DOMAIN_ID'), flush=True)


def publish_images(t):
    import cv2
    a = R['ann']
    rgb_l, rgb_r, dep = robot_rgb(a['rgb_l']), robot_rgb(a['rgb_r']), np.asarray(a['dep'].get_data())
    if rgb_l.size == 0 or rgb_r.size == 0 or dep.size == 0:
        return False
    bgr = np.ascontiguousarray(rgb_l[:, :, 2::-1])
    mono_l = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mono_r = cv2.cvtColor(np.ascontiguousarray(rgb_r[:, :, 2::-1]), cv2.COLOR_BGR2GRAY)
    mm = np.where(np.isfinite(dep) & (dep >= tc.CAM['depth_min_m']) & (dep <= tc.CAM['depth_max_m']),
                  dep * 1000.0, 0).astype(np.uint16)
    Image, stamp, pub = R['Image'], R['stamp'], R['pub']

    def img(frame, enc, arr):
        m = Image()
        m.header.stamp = stamp(t)
        m.header.frame_id = frame
        m.height, m.width = arr.shape[:2]
        m.encoding = enc
        m.step = arr.strides[0]
        m.data = np.ascontiguousarray(arr).tobytes()
        return m
    pub['color'].publish(img('camera_left_optical_frame', 'bgr8', bgr))
    pub['lmono'].publish(img('camera_left_optical_frame', 'mono8', mono_l))
    pub['rmono'].publish(img('camera_right_optical_frame', 'mono8', mono_r))
    pub['depth'].publish(img('camera_left_optical_frame', '16UC1', mm))
    for k, m in R['info'].items():
        m.header.stamp = stamp(t)
        pub[k].publish(m)
    return True


# ---------------------------------------------------------------- control file
def poll_control():
    try:
        mt = os.path.getmtime(args.control)
    except OSError:
        return
    if mt == S['control_mtime']:
        return
    S['control_mtime'] = mt
    try:
        c = json.load(open(args.control))
    except Exception as e:  # noqa: BLE001
        print('CONTROL_BAD', e, flush=True)
        return
    if c.get('ros') and not S['ros']:
        try:
            enable_ros()
            announce()
        except Exception as e:  # noqa: BLE001
            print('ROS_FAILED', repr(e), flush=True)
    if c.get('drive') in ('path', 'ros', 'stop') and c['drive'] != S['drive']:
        S['drive'] = c['drive']
        announce()
    if 'teleport' in c and c.get('id') != S['control_id']:
        x, y, yaw = (float(v) for v in c['teleport'])
        teleport(x, y, yaw)
        pp.i = int(np.argmin([math.hypot(px - x, py - y) for px, py, _ in path]))
    S['control_id'] = c.get('id')
    if 'record' in c and c['record'] != S.get('record_base'):
        if S.get('rec'):
            S['rec'].close()
            S['rec'] = None
        if c['record']:
            S['rec'] = Recorder(c['record'])
        S['record_base'] = c['record']
        announce()
    if c.get('quit'):
        S['quit'] = True


def live_view(t, rtf):
    """Follow camera + robot camera inset + HUD -> JPEG for twin_viewer.py (and the video)."""
    import cv2
    img = np.asarray(view['ann'].get_data())
    if img.size == 0:
        return
    frame = np.ascontiguousarray(img[:, :, 2::-1])
    rob = robot_rgb(ann_left['rgb'])
    if rob.size:
        inset = cv2.resize(np.ascontiguousarray(rob[:, :, 2::-1]), (448, 252))
        frame[VH - 262:VH - 10, VW - 458:VW - 10] = inset
        cv2.rectangle(frame, (VW - 459, VH - 263), (VW - 9, VH - 9), (255, 255, 255), 1)
        cv2.putText(frame, 'robot camera (D1000 left, 0.13 m)', (VW - 452, VH - 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    hud = (f"Isaac Sim twin   sim {t:7.1f} s   RTF {rtf:4.2f}   {DRIVE_SHORT[S['drive']]}   people {len(sc.people)}"
           f"   bumps {S['bumps']}")
    cv2.rectangle(frame, (0, 0), (VW, 30), (0, 0, 0), -1)
    cv2.putText(frame, hud, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if args.viewer == 'file':
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            try:
                with open(args.view_file + '.tmp', 'wb') as f:
                    f.write(buf.tobytes())
                os.replace(args.view_file + '.tmp', args.view_file)
            except OSError:
                pass
    if video['w'] is not None:
        video['w'].write(frame)


class ImagePanel:
    """An omni.ui window showing a numpy image (fed from the robot camera's annotators: no extra rendering)."""

    def __init__(self, title, w, h):
        import omni.ui as ui
        self.ui = ui
        self.w, self.h = w, h
        self.prov = ui.ByteImageProvider()
        self.win = ui.Window(title, width=w + 20, height=h + 40)
        with self.win.frame:
            ui.ImageWithProvider(self.prov, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)

    def set(self, rgba):
        rgba = np.ascontiguousarray(rgba, np.uint8)
        try:
            self.prov.set_data_array(rgba, [rgba.shape[1], rgba.shape[0]])
        except Exception:  # noqa: BLE001  (older omni.ui)
            self.prov.set_bytes_data(rgba.flatten().tolist(), [rgba.shape[1], rgba.shape[0]])


panels = {}


def make_panels():
    """Robot camera and its depth (what the stack receives), docked right of the follow-camera viewport."""
    import omni.ui as ui
    panels['rgb'] = ImagePanel('Robot camera (D1000 left)', W, H)
    panels['depth'] = ImagePanel('Robot depth (0.3-7 m, turbo: near = blue, far = red)', W, H)
    app_update(3)
    vp = ui.Workspace.get_window('Viewport')
    if vp is not None:
        panels['rgb'].win.dock_in(vp, ui.DockPosition.RIGHT, 0.36)
        app_update(2)
        panels['depth'].win.dock_in(panels['rgb'].win, ui.DockPosition.BOTTOM, 0.5)
    for k in ('Stage', 'Property', 'Content', 'Console', 'Render Settings'):
        w_ = ui.Workspace.get_window(k)
        if w_ is not None:
            w_.visible = False


def update_panels():
    import cv2
    rgb = robot_rgb(ann_left['rgb'])
    dep = np.asarray(ann_left['dep'].get_data())
    if rgb.size:
        panels['rgb'].set(rgb[:, :, :4] if rgb.shape[2] == 4 else np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]))
    if dep.size:
        d = dep.astype(np.float32)
        ok = np.isfinite(d) & (d >= tc.CAM['depth_min_m']) & (d <= tc.CAM['depth_max_m'])
        u8 = np.clip((np.where(ok, d, 0) - 0.3) / (7.0 - 0.3) * 255, 0, 255).astype(np.uint8)
        col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        col[~ok] = 0
        panels['depth'].set(np.dstack([col[:, :, ::-1], np.full(d.shape, 255, np.uint8)]))


DRIVE_SHORT = {'path': 'demo: following the recorded path', 'ros': 'driven by the robot stack (WSL)', 'stop': 'stopped'}
DRIVE_TEXT = {'path': 'Roomba following the recorded robot path (demo, stack not driving)',
              'ros': "Roomba driven by the robot's real stack in WSL (Nav2 + explorer)",
              'stop': 'Roomba standing still'}


def announce():
    where = ("Isaac Sim GUI window" if args.viewer == 'kit' else
             "Live view window 'Isaac Sim twin - live' (Isaac Sim renders headless; its own GUI window crashes in the "
             "NVIDIA Vulkan driver here)")
    rec = f" Recording video ({os.path.basename(S['record_base'])})." if S.get('rec') else ''
    status(f"{where} OPEN: remodelled house, {len(sc.people)} people walking; follow camera, robot camera and robot depth panels. "
           f"{DRIVE_TEXT[S['drive']]}. ROS {'on (Zenoh, domain ' + os.environ['ROS_DOMAIN_ID'] + ')' if S['ros'] else 'off'}.{rec}")


if args.ros or args.drive == 'ros':
    try:
        enable_ros()
    except Exception as e:  # noqa: BLE001
        print('ROS_FAILED', repr(e), flush=True)
announce()
if args.record:
    S['rec'] = Recorder(args.record)
    S['record_base'] = args.record
    announce()

# ---------------------------------------------------------------- main loop
boom = {'yaw': yaw0}
n = 0
wall0, sim0 = time.monotonic(), SimulationManager.get_simulation_time()
rtf_hist = []
tm = {'update': 0.0, 'n': 0}
last_ctrl = 0.0
last_state = 0.0
wheel_dist = [0.0, 0.0]
prev_jp = None
if args.view_gain != 1.0:
    try:
        calibrate_lut()
    except Exception as e:  # noqa: BLE001
        print('CAMERA_LUT_FAILED', repr(e), flush=True)
while app.is_running() and not S.get('quit'):
    now = time.monotonic()
    if now - last_ctrl > 1.0:
        last_ctrl = now
        poll_control()
    p, q, yaw = pose()
    if S['drive'] == 'path':
        v, w = pp(p[0], p[1], yaw, args.speed)
        vl, vr = wheel_targets(v, w)
        # demo: hop past places where the plan and the recorded path disagree (doorways)
        S['stuck_hist'].append((now, p[0], p[1], yaw))
        S['stuck_hist'] = [h for h in S['stuck_hist'] if now - h[0] < 8.0]
        h0 = S['stuck_hist'][0]
        if now - h0[0] > 6.0 and math.hypot(p[0] - h0[1], p[1] - h0[2]) < 0.05 and abs(yaw - h0[3]) < 0.1:
            j = pp.i
            while j < len(drive_path) - 1 and math.hypot(drive_path[j][0] - p[0], drive_path[j][1] - p[1]) < 0.8:
                j += 1
            teleport(*drive_path[j])
            pp.i = j
            S['stuck_hist'].clear()
            S['teleports'] += 1
    elif S['drive'] == 'ros':
        if now - S['cmd_vel_t'] < 0.5:        # manual teleop overrides
            vl, vr = wheel_targets(*S['cmd_vel'])
        elif now - S['wheel_cmd_t'] < 2.0:    # the OI keeps the last drive command (bridge alive)
            vl, vr = S['wheel_cmd']
        else:
            vl = vr = 0.0
    else:
        vl = vr = 0.0
    tgt = np.zeros((1, len(dof)), np.float32)
    tgt[0, LI], tgt[0, RI] = vl / WR, vr / WR
    art.set_dof_velocity_targets(tgt)
    t_py = time.perf_counter()
    render_cams = (n + 1) % CAM_EVERY == 0 or S.get('rec') is not None
    for rp_ in robot_rps:
        try:
            rp_.hydra_texture.set_updates_enabled(render_cams)
        except Exception:  # noqa: BLE001
            pass
    t_up = time.perf_counter()
    app_update(1)
    tm['update'] += time.perf_counter() - t_up
    tm['n'] += 1
    n += 1
    t = SimulationManager.get_simulation_time()
    if n % CAM_EVERY == 0:
        S['cam_t'] = t
    if n <= 5:
        print('SIM_T', n, round(t, 4), flush=True)
    p, q, yaw = pose()
    jp = npy(art.get_dof_positions())[0]
    # Safety net for pushes by people (their body colliders are switched off, see disable_people_colliders): if the
    # robot moved or turned clearly more than its wheels did while a person is close, put it back. Counted in the
    # status as pushes_undone; it should stay 0 (cl_run8 undid 437 pushes before the colliders were switched off).
    if S.get('prev_pose') is not None and prev_jp is not None and S['drive'] == 'ros':
        pp_, pq_, pyaw_ = S['prev_pose']
        dl, dr = float(jp[LI] - prev_jp[LI]) * WR, float(jp[RI] - prev_jp[RI]) * WR
        moved = math.hypot(float(p[0] - pp_[0]), float(p[1] - pp_[1]))
        turned = abs((yaw - pyaw_ + math.pi) % (2 * math.pi) - math.pi)
        if moved > abs(dl + dr) / 2 + 0.012 or turned > abs(dr - dl) / WB + 0.04:
            if any(math.hypot(px - float(pp_[0]), py - float(pp_[1])) < 0.9 for px, py in people_xy()):
                art.set_world_poses(np.array([pp_], np.float32), np.array([pq_], np.float32))
                art.set_velocities(np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32))
                p, q, yaw = pp_, pq_, pyaw_
                S['pushes_undone'] = S.get('pushes_undone', 0) + 1
    S['prev_pose'] = (np.array(p, np.float32), np.array(q, np.float32), yaw)
    if sc.people and n % 60 == 0:
        k_ = disable_people_colliders()
        if k_:
            S['people_colliders_off'] = S.get('people_colliders_off', 0) + k_
            print('PEOPLE_COLLIDERS_OFF', k_, 'at update', n, flush=True)
    # follow camera: 1.7 m behind, 1.25 m up, pulled in front of walls
    boom['yaw'] += ((yaw - boom['yaw'] + math.pi) % (2 * math.pi) - math.pi) * 0.06
    by = boom['yaw']
    back = (-math.cos(by), -math.sin(by), 0.0)
    o = (float(p[0]), float(p[1]), 1.1)
    hb = sq.raycast_closest(o, back, 1.8)
    L = max(0.35, (hb['distance'] - 0.2) if hb.get('hit') else 1.7)
    eye = Gf.Vec3d(o[0] + back[0] * L, o[1] + back[1] * L, 1.25 + 0.15 * (1.7 - L))
    target = Gf.Vec3d(float(p[0]) + math.cos(by) * 0.6, float(p[1]) + math.sin(by) * 0.6, 0.15)
    follow_op.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
    # wheel travel for the encoders
    if prev_jp is not None:
        wheel_dist[0] += float(jp[LI] - prev_jp[LI]) * WR
        wheel_dist[1] += float(jp[RI] - prev_jp[RI]) * WR
    prev_jp = jp
    if S['ros']:
        try:
            R['rclpy'].spin_once(R['node'], timeout_sec=0.0)
            ppl = people_xy()
            bl, br, light, cliffs, wall = sense(p, yaw, ppl)
            if (bl or br) and not S['last_bump']:
                S['bumps'] += 1
            S['last_bump'] = bl or br
            m = R['Float64MultiArray']()
            m.data = [t, wheel_dist[0], wheel_dist[1], float(bl), float(br), 0.0, *[float(c) for c in cliffs],
                      *[float(s) for s in light], float(wall), 0.0, float(n)]
            R['pub']['state'].publish(m)
            if n % 2 == 0:
                od = R['Odometry']()
                od.header.stamp = R['stamp'](t)
                od.header.frame_id = 'map'
                od.child_frame_id = 'base_link'
                od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = (float(c) for c in p)
                o_ = od.pose.pose.orientation
                o_.w, o_.x, o_.y, o_.z = (float(c) for c in q)
                R['pub']['gt'].publish(od)
            if n % CAM_EVERY == 2 and 'cam_t' in S:      # read back two updates after the render
                publish_images(S['cam_t'])
            if n % 12 == 0:
                R['pub']['people'].publish(R['String'](data=json.dumps({'t': t, 'people': [[round(a, 3), round(b, 3)] for a, b in ppl]})))
        except Exception as e:  # noqa: BLE001
            print('ROS_STEP_ERROR', repr(e), flush=True)
    if S.get('rec') and n % max(1, args.render_hz // 30) == 0:
        try:
            S['rec'].frame(t, p, yaw)
        except Exception as e:  # noqa: BLE001
            print('REC_ERROR', repr(e), flush=True)
    if args.viewer == 'kit':
        if n == 20:
            try:
                make_panels()
            except Exception as e:  # noqa: BLE001
                print('PANELS_FAILED', repr(e), flush=True)
        elif panels and n % CAM_EVERY == 2:
            try:
                update_panels()
            except Exception as e:  # noqa: BLE001
                print('PANEL_ERROR', repr(e), flush=True)
                panels.clear()
    if view and n % args.view_every == 0:
        try:
            live_view(t, S.get('rtf', 0.0))
        except Exception as e:  # noqa: BLE001
            print('VIEW_ERROR', repr(e), flush=True)
    if n % 60 == 0:
        wall_s = time.monotonic() - wall0
        rtf = (t - sim0) / max(wall_s, 1e-6)
        S['rtf'] = rtf
        wall0, sim0 = time.monotonic(), t
        S['ms_update'] = round(1000 * tm['update'] / max(tm['n'], 1), 1)
        S['ms_frame'] = round(1000 * wall_s / max(tm['n'], 1), 1)
        tm.update(update=0.0, n=0)
        st = {'t_sim': round(t, 2), 'rtf': round(rtf, 3), 'drive': S['drive'], 'ros': S['ros'], 'x': round(float(p[0]), 3),
              'y': round(float(p[1]), 3), 'yaw_deg': round(math.degrees(yaw), 1), 'path_idx': pp.i, 'teleports': S['teleports'],
              'bumps': S['bumps'], 'people': len(sc.people), 'wall_clock': time.strftime('%H:%M:%S'),
              'uptime_s': round(time.time() - T_START), 'ms_frame': S.get('ms_frame'), 'ms_update': S.get('ms_update'),
              'pushes_undone': S.get('pushes_undone', 0), 'people_colliders_off': S.get('people_colliders_off', 0)}
        try:
            with open(args.state + '.tmp', 'w') as f:
                json.dump(st, f)
            os.replace(args.state + '.tmp', args.state)
        except OSError:
            pass
        if S['ros']:
            try:
                R['pub']['sim'].publish(R['String'](data=json.dumps(st)))
            except Exception:  # noqa: BLE001
                pass
status(f'Isaac Sim window CLOSED (was open since {time.strftime("%H:%M", time.localtime(T_START))}).')
try:
    if S['ros']:
        SimulationManager.deregister_callback(R['imu_sub'])
        R['node'].destroy_node()
        R['rclpy'].shutdown()
except Exception:  # noqa: BLE001
    pass
if video['w'] is not None:
    video['w'].release()
if S.get('rec'):
    S['rec'].close()
tl.stop()
app.close()
