"""Shared constants and small helpers for the Isaac Sim twin scripts (need a running SimulationApp)."""
import json
import math
import os

from pxr import Gf, Usd, UsdGeom, UsdSemantics

TWIN = os.environ.get('TWIN', r'E:\twin')
ASSETS = os.path.join(TWIN, 'assets')
USD_DIR = os.path.join(ASSETS, 'usd')
HOUSE_USD = os.path.join(USD_DIR, 'house.usd')
ROBOT_USD = os.path.join(USD_DIR, 'roomba550_d1000.usd')
REPO_ISAAC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(REPO_ISAAC, 'robot', 'roomba550_d1000.urdf')
POSES_TXT = os.path.join(ASSETS, 'house', 'house_poses.txt')
GRID_YAML = os.path.join(ASSETS, 'house', 'house.yaml')

# Real MYNT EYE D1000-120 (fresh_map8 camera_info + /tf_static), see README "Robot facts"
CAM = {
    'width': 640, 'height': 360,
    'fx': 349.20001220703125, 'fy': 349.20001220703125,
    'cx': 318.03509521484375, 'cy': 174.52203369140625,
    'baseline_m': 0.11837790679931642,       # -P[3]/fx of the right camera_info
    'rate_hz': 15.0,                          # measured: 15.0 Hz images, 199.7 Hz IMU
    'imu_rate_hz': 200.0,
    'depth_min_m': 0.3, 'depth_max_m': 7.0,
    'mount_xyz': (0.153, 0.0, 0.13),          # base_link -> camera_link
}
DRIVE = {'wheel_base_m': 0.235, 'wheel_radius_m': 0.036, 'max_wheel_speed_m_s': 0.4}
SEM_CLASSES = ['floor', 'wall', 'furniture', 'ceiling', 'person', 'railing', 'robot']


def add_semantic_label(prim, label, taxonomy='class'):
    attr = UsdSemantics.LabelsAPI.Apply(prim, taxonomy).GetLabelsAttr()
    cur = list(attr.Get() or [])
    if label not in cur:
        cur.append(label)
    attr.Set(cur)


def load_path_poses(path=POSES_TXT):
    """RTAB-Map optimized base_link poses (map frame): list of (x, y, yaw)."""
    out = []
    for ln in open(path):
        if not ln.strip() or ln.startswith('#'):
            continue
        _, x, y, z, qx, qy, qz, qw, _id = map(float, ln.split())
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        out.append((x, y, yaw))
    return out


def camera_aperture(width=CAM['width'], height=CAM['height'], fx=CAM['fx'], fy=CAM['fy'],
                    cx=CAM['cx'], cy=CAM['cy'], focal_mm=1.0):
    """USD pinhole camera attributes reproducing the real K (focal in mm, apertures in mm, offsets in mm).

    pixel size = focalLength / fx. USD film-back x points right and y up, so the principal point is
    cx = W/2 - hOffset/px and cy = H/2 + vOffset/px (verified by projecting a target in test_robot.py).
    fx == fy for the D1000, so square pixels are assumed."""
    px = focal_mm / fx
    return {
        'focalLength': focal_mm,
        'horizontalAperture': width * px,
        'verticalAperture': height * px,
        'horizontalApertureOffset': (width / 2.0 - cx) * px,
        'verticalApertureOffset': (cy - height / 2.0) * px,
    }


def set_pinhole(cam_prim, **kw):
    """Apply camera_aperture() values (stage units are metres, camera lengths are in tenths of a scene unit
    -> USD expects mm when metersPerUnit is 1, so pass mm)."""
    a = camera_aperture(**kw)
    cam = UsdGeom.Camera(cam_prim)
    cam.CreateFocalLengthAttr(a['focalLength'])
    cam.CreateHorizontalApertureAttr(a['horizontalAperture'])
    cam.CreateVerticalApertureAttr(a['verticalAperture'])
    cam.CreateHorizontalApertureOffsetAttr(a['horizontalApertureOffset'])
    cam.CreateVerticalApertureOffsetAttr(a['verticalApertureOffset'])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 50.0))
    cam.CreateProjectionAttr('perspective')
    return a


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
