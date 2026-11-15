"""Check animated people: house + lights + robot + IRA people wandering; render robot-eye frames.

  python test_people.py --n 4 --seconds 8
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=4)
ap.add_argument('--seconds', type=float, default=8.0)
ap.add_argument('--pose', type=float, nargs=3, default=[1.42, -0.17, 25.0], help='x y yaw_deg')
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting'})
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
from PIL import Image  # noqa: E402

import twin_common as tc  # noqa: E402
from twin_scene import TwinScene, app_update  # noqa: E402

rep.settings.set_render_rtx_realtime(antialiasing='DLAA')
sc = TwinScene(look='scan', seed=3)
sc.add_lights()
x, y, yaw = args.pose[0], args.pose[1], math.radians(args.pose[2])
sc.add_robot(x, y, yaw)
sc.enable_people_runtime()
t0 = time.time()
ok = sc.bake_navmesh()
t_nav = time.time() - t0
# people in front of the robot, 1.5-5 m, within +-35 deg
d = sc.free_xy - np.array([x, y])
r = np.linalg.norm(d, axis=1)
ang = np.arctan2(d[:, 1], d[:, 0]) - yaw
ang = (ang + np.pi) % (2 * np.pi) - np.pi
cand = sc.free_xy[(r > 1.5) & (r < 5.0) & (np.abs(ang) < math.radians(35))]
pos = [tuple(cand[i]) for i in sc.rng.sample(range(len(cand)), min(args.n, len(cand)))]
sc.spawn_people(len(pos), positions=pos)
from pxr import Usd, UsdGeom, Gf
for pp in sc.people:
    prim = sc.stage.GetPrimAtPath(pp['path'])
    nm = sum(1 for q in Usd.PrimRange(prim) if q.GetTypeName() == 'Mesh')
    bb = UsdGeom.BBoxCache(0, ['default', 'render']).ComputeWorldBound(prim).ComputeAlignedRange()
    print('PERSON', pp['path'], 'meshes', nm, 'loaded', prim.IsLoaded(), 'bbox', bb.GetMin(), bb.GetMax(), flush=True)
if os.environ.get('PEOPLE_LOOK'):
    sc.set_look(os.environ['PEOPLE_LOOK'])
obs = UsdGeom.Camera.Define(sc.stage, '/World/ObsCam')
obs.CreateFocalLengthAttr(14.0)
oxf = UsdGeom.Xformable(obs)
oxf.AddTranslateOp().Set(Gf.Vec3d(x - 2.5 * math.cos(yaw), y - 2.5 * math.sin(yaw), 1.8))
oxf.AddRotateXYZOp().Set(Gf.Vec3f(75, 0, math.degrees(yaw) - 90))
rp2 = rep.create.render_product('/World/ObsCam', (640, 360))
rgb2 = rep.AnnotatorRegistry.get_annotator('rgb')
rgb2.attach(rp2)
rp = rep.create.render_product(sc.cam_paths['left'], (640, 360))
rgb = rep.AnnotatorRegistry.get_annotator('rgb')
rgb.attach(rp)
tl = omni.timeline.get_timeline_interface()
tl.play()
frames, track = [], []
n_upd = int(args.seconds * 60)
for k in range(n_upd):
    app_update(1)
    if k % 60 == 30:
        rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
        frames.append(np.asarray(rgb.get_data())[:, :, :3].copy())
        Image.fromarray(np.asarray(rgb2.get_data())[:, :, :3]).save(rf'E:\twin\work\people_obs_{k:04d}.png')
        track.append([[round(a, 2), round(b, 2)] for a, b in sc.person_positions()])
tl.stop()
os.makedirs(r'E:\twin\screens', exist_ok=True)
if frames:
    sel = frames[:: max(1, len(frames) // 4)][:4]
    Image.fromarray(np.vstack([np.hstack(sel[:2]), np.hstack(sel[2:4])]) if len(sel) == 4 else np.hstack(sel)).save(
        r'E:\twin\screens\people_robot_view.png')
moved = [round(math.dist(track[0][i], track[-1][i]), 2) for i in range(len(track[0]))] if len(track) > 1 else []
print('PEOPLE_TEST ' + json.dumps({'navmesh': ok, 'navmesh_s': round(t_nav, 1), 'spawned': len(sc.people),
                                   'start': pos and [list(map(lambda v: round(float(v), 2), p)) for p in pos],
                                   'track_first': track[0] if track else None, 'track_last': track[-1] if track else None,
                                   'moved_m': moved}))
app.close()
