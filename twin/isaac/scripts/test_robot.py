"""Robot asset check: camera intrinsics (projected target vs real K), depth, drive kinematics.

  python test_robot.py   -> prints ROBOT_TEST {...} and saves E:\\twin\\screens\\robot_test_left.png
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting'})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

import twin_common as tc  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

info = json.load(open(os.path.splitext(tc.ROBOT_USD)[0] + '_info.json'))
settings = carb.settings.get_settings()
ctx = omni.usd.get_context()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
UsdLux.DomeLight.Define(stage, '/World/Dome').CreateIntensityAttr(1500)
ground = UsdGeom.Mesh.Define(stage, '/World/Ground')
ground.CreatePointsAttr([(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)])
ground.CreateFaceVertexCountsAttr([4])
ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
ground.CreateDisplayColorAttr([(0.5, 0.5, 0.5)])
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

robot = stage.DefinePrim('/World/Roomba', 'Xform')
robot.GetReferences().AddReference(tc.ROBOT_USD)
root = '/World/Roomba'
p = {k: (v.replace(info['root'], root, 1) if v else v) for k, v in info['paths'].items()}
left_cam = info['cameras']['left'].replace(info['root'], root, 1)

# calibration target: red sphere at (x=0.5, y=-0.2, z=2.0) in the left optical frame
opt = UsdGeom.Xformable(stage.GetPrimAtPath(p['camera_left_optical_frame'])).ComputeLocalToWorldTransform(0)
target_opt = Gf.Vec3d(0.5, -0.2, 2.0)
target_w = opt.Transform(target_opt)
sph = UsdGeom.Sphere.Define(stage, '/World/Target')
sph.CreateRadiusAttr(0.03)
sph.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])
UsdGeom.Xformable(sph).AddTranslateOp().Set(target_w)
tc.add_semantic_label(sph.GetPrim(), 'target')
expected_uv = (tc.CAM['fx'] * 0.5 / 2.0 + tc.CAM['cx'], tc.CAM['fy'] * -0.2 / 2.0 + tc.CAM['cy'])

rp = rep.create.render_product(left_cam, (tc.CAM['width'], tc.CAM['height']))
rgb_a = rep.AnnotatorRegistry.get_annotator('rgb')
dep_a = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
sem_a = rep.AnnotatorRegistry.get_annotator('semantic_segmentation', init_params={'colorize': False})
for a in (rgb_a, dep_a, sem_a):
    a.attach(rp)
for _ in range(8):
    rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
rgb = np.asarray(rgb_a.get_data())[:, :, :3]
depth = np.asarray(dep_a.get_data())
sem = sem_a.get_data()
ids = {int(k): v for k, v in sem['info']['idToLabels'].items()}
tid = [k for k, v in ids.items() if 'target' in str(v)]
mask = np.isin(np.asarray(sem['data']), tid)
vv, uu = np.nonzero(mask)
measured_uv = (float(uu.mean()) + 0.5, float(vv.mean()) + 0.5) if len(uu) else None
Image.fromarray(rgb).save(r'E:\twin\screens\robot_test_left.png')
d_center = float(np.median(depth[mask])) if len(uu) else None

# ---- drive test ----
SimulationManager.set_physics_dt(1.0 / 200.0)
tl = omni.timeline.get_timeline_interface()
tl.play()
from isaacsim.core.experimental.prims import Articulation  # noqa: E402

app.update()
art = Articulation(p['base_link'])
names = art.dof_names
li, ri = names.index('left_wheel_joint'), names.index('right_wheel_joint')


def pose():
    pos, quat = art.get_world_poses()
    pos = np.asarray(pos.numpy() if hasattr(pos, 'numpy') else pos)[0]
    q = np.asarray(quat.numpy() if hasattr(quat, 'numpy') else quat)[0]   # wxyz
    yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return pos, yaw


def drive(v, w, seconds):
    vl = (v - w * tc.DRIVE['wheel_base_m'] / 2) / tc.DRIVE['wheel_radius_m']
    vr = (v + w * tc.DRIVE['wheel_base_m'] / 2) / tc.DRIVE['wheel_radius_m']
    tgt = np.zeros((1, len(names)), np.float32)
    tgt[0, li], tgt[0, ri] = vl, vr
    t0 = SimulationManager.get_simulation_time()
    n = 0
    while SimulationManager.get_simulation_time() - t0 < seconds:
        art.set_dof_velocity_targets(tgt)
        app.update()
        n += 1
    return n, SimulationManager.get_simulation_time() - t0


for _ in range(30):
    app.update()
p0, y0 = pose()
n1, s1 = drive(0.2, 0.0, 3.0)
p1, y1 = pose()
n2, s2 = drive(0.0, 0.8, 3.0)
p2, y2 = pose()
tl.stop()
res = {
    'expected_uv': [round(x, 2) for x in expected_uv], 'measured_uv': [round(x, 2) for x in measured_uv] if measured_uv else None,
    'uv_error_px': round(math.dist(expected_uv, measured_uv), 2) if measured_uv else None,
    'depth_at_target_m': round(d_center, 4) if d_center else None, 'expected_depth_m': round(2.0 - 0.03, 3),
    'start_pose': [round(float(x), 4) for x in p0], 'start_z_base': round(float(p0[2]), 4),
    'straight_3s_0p2': {'moved_m': round(float(np.linalg.norm(p1[:2] - p0[:2])), 3), 'expected_m': 0.6,
                        'yaw_drift_deg': round(math.degrees(y1 - y0), 2), 'updates': n1, 'sim_s': round(s1, 3)},
    'turn_3s_0p8': {'yaw_deg': round(math.degrees((y2 - y1 + math.pi) % (2 * math.pi) - math.pi), 1),
                    'expected_deg': round(math.degrees(2.4) - 360 * round(2.4 / (2 * math.pi)), 1),
                    'translation_m': round(float(np.linalg.norm(p2[:2] - p1[:2])), 3)},
    'physics_dt': SimulationManager.get_physics_dt(),
    'settings': {k: settings.get(k) for k in ['/app/player/useFixedTimeStepping', '/app/runLoops/main/rateLimitFrequency',
                                               '/app/runLoops/main/rateLimitEnabled', '/rtx/rendermode',
                                               '/rtx/post/dlss/execMode', '/rtx/post/aa/op']},
}
print('ROBOT_TEST ' + json.dumps(res))
app.close()
