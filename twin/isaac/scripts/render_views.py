"""Render the house twin for inspection: top-down and oblique overviews, and robot-eye views at the
poses where real frames were captured, side by side with the real D1000 image.

  python render_views.py [--look remodel] [--n 8] [--prefix remodel_]
"""
import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval'))

ap = argparse.ArgumentParser()
ap.add_argument('--look', default='remodel', choices=['remodel', 'scan', 'clean'])
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--real', default=r'E:\twin\real_frames\left')
ap.add_argument('--out', default=r'E:\twin\screens')
ap.add_argument('--prefix', default=None)
ap.add_argument('--dome', type=float, default=300.0)
ap.add_argument('--floor1', action='store_true', help='also render the stacked downstairs (top-down, cutaway, stair views)')
ap.add_argument('--skip-real', action='store_true', help='only the overview renders')
args, _ = ap.parse_known_args()
prefix = args.prefix if args.prefix is not None else f'{args.look}_'

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting'})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

import twin_common as tc  # noqa: E402
from camera_model import apply as camera_model  # noqa: E402
from twin_scene import TwinScene  # noqa: E402

rep.settings.set_render_rtx_realtime(antialiasing='DLAA')   # DLAA/DLSS carry the denoiser; FXAA leaves path-tracing noise
sc = TwinScene(look=args.look)
stage = sc.stage
sc.add_lights(dome=args.dome)
sc.add_robot(0.0, 0.0, 0.0)
t_op, r_op = sc.robot_t, sc.robot_r
left_cam = sc.cam_paths['left']
ceilings = [p for p in stage.Traverse() if 'ceiling' in p.GetName().lower() and p.GetTypeName() == 'Mesh']
ceilings += [stage.GetPrimAtPath('/World/House/Ceiling')] if stage.GetPrimAtPath('/World/House/Ceiling') else []

grid_c = np.array(sc.meta['origin'][:2]) + np.array(sc.grid.shape[::-1]) * sc.meta['resolution'] / 2
top = UsdGeom.Camera.Define(stage, '/World/TopCam')
top.CreateFocalLengthAttr(18.0)
top.CreateHorizontalApertureAttr(20.955)
top.CreateVerticalApertureAttr(20.955)
top.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100))
UsdGeom.Xformable(top).AddTranslateOp().Set(Gf.Vec3d(float(grid_c[0]), float(grid_c[1]), 22.0))
obl = UsdGeom.Camera.Define(stage, '/World/ObliqueCam')
obl.CreateFocalLengthAttr(14.0)
obl.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100))
ox = UsdGeom.Xformable(obl)
ox.AddTranslateOp().Set(Gf.Vec3d(float(grid_c[0]) - 8.0, float(grid_c[1]) - 11.0, 10.0))
ox.AddRotateXYZOp().Set(Gf.Vec3f(52, 0, -32))


def grab(cam, w, h, frames=12):
    rp = rep.create.render_product(cam, (w, h))
    a = rep.AnnotatorRegistry.get_annotator('rgb')
    a.attach(rp)
    for _ in range(frames):
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
    img = np.asarray(a.get_data())[:, :, :3].copy()
    a.detach()
    rp.destroy()
    return img


t_op.Set(Gf.Vec3d(0, 0, 0))
r_op.Set(0.0)
for c in ceilings:
    UsdGeom.Imageable(c).MakeInvisible()
Image.fromarray(grab('/World/TopCam', 1400, 1400, 24)).save(os.path.join(args.out, f'{prefix}topdown.png'))
Image.fromarray(grab('/World/ObliqueCam', 1600, 900, 24)).save(os.path.join(args.out, f'{prefix}oblique.png'))
for c in ceilings:
    UsdGeom.Imageable(c).MakeVisible()


def look_at_cam(path, eye, target, focal=14.0):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100))
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(cam).AddTransformOp().Set(m)
    return path


if args.floor1 and stage.GetPrimAtPath('/World/HouseF1'):
    import json
    f1 = json.load(open(os.path.join(os.path.dirname(tc.FLOORPLAN_JSON), 'floorplan_floor1.json')))
    P = np.vstack([np.array(r['polygon']) for r in f1['rooms']])
    c1 = (P.min(0) + P.max(0)) / 2
    z1 = f1['floor_z_m']
    up = stage.GetPrimAtPath('/World/House')
    for c in ceilings:
        UsdGeom.Imageable(c).MakeInvisible()
    UsdGeom.Imageable(up).MakeInvisible()
    top1 = UsdGeom.Camera.Define(stage, '/World/TopCamF1')
    top1.CreateFocalLengthAttr(18.0)
    top1.CreateHorizontalApertureAttr(20.955)
    top1.CreateVerticalApertureAttr(20.955)
    top1.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100))
    UsdGeom.Xformable(top1).AddTranslateOp().Set(Gf.Vec3d(float(c1[0]), float(c1[1]), z1 + 26.0))
    Image.fromarray(grab('/World/TopCamF1', 1400, 1400, 24)).save(os.path.join(args.out, f'{prefix}floor1_topdown.png'))
    UsdGeom.Imageable(up).MakeVisible()
    # cutaway of both floors (all ceilings hidden), seen from outside
    look_at_cam('/World/StackCam', (c1[0] - 14.0, c1[1] - 15.0, 9.0), (c1[0], c1[1], z1 + 1.0))
    Image.fromarray(grab('/World/StackCam', 1600, 900, 24)).save(os.path.join(args.out, f'{prefix}stack_oblique.png'))
    for c in ceilings:
        UsdGeom.Imageable(c).MakeVisible()
    # stair views: from the upstairs top step looking down, and from the entry looking up the stair
    up_plan = json.load(open(tc.FLOORPLAN_JSON))
    top_step = np.array(up_plan['stair_void']['top_step'])
    vc = np.array(up_plan['stair_void']['polygon']).mean(0)
    ts = top_step.mean(0)
    d = (ts - vc) / np.linalg.norm(ts - vc)
    look_at_cam('/World/StairTopCam', (ts[0] + d[0] * 1.2, ts[1] + d[1] * 1.2, 1.6), (vc[0], vc[1], -2.0), 12.0)
    Image.fromarray(grab('/World/StairTopCam', 1280, 720, 24)).save(os.path.join(args.out, f'{prefix}stair_from_top.png'))
    fd = [np.array(dr['segment']).mean(0) for dr in f1['doors'] if dr['id'] == 'front']
    eye = fd[0] + (vc - fd[0]) / np.linalg.norm(vc - fd[0]) * 1.2 if fd else vc
    look_at_cam('/World/EntryCam', (eye[0], eye[1], z1 + 1.5), (ts[0], ts[1], -0.8), 12.0)
    Image.fromarray(grab('/World/EntryCam', 1280, 720, 24)).save(os.path.join(args.out, f'{prefix}entry_up_stair.png'))

if args.skip_real:
    print('VIEWS_DONE overviews only')
    app.close()
    sys.exit(0)

poses = []
for ln in open(tc.POSES_TXT):
    if ln.strip() and not ln.startswith('#'):
        s, x, y, z, qx, qy, qz, qw, i = map(float, ln.split())
        poses.append((s, x, y, math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))))
poses = np.array(poses)
files = sorted(glob.glob(os.path.join(args.real, '*.png')))
cands = []
for f in files:
    s = int(os.path.splitext(os.path.basename(f))[0]) / 1e9
    j = int(np.argmin(np.abs(poses[:, 0] - s)))
    if abs(poses[j, 0] - s) < 0.15:
        cands.append((f, poses[j]))
pick = [cands[i] for i in np.linspace(0, len(cands) - 1, min(args.n, len(cands))).astype(int)] if cands else []
rp = rep.create.render_product(left_cam, (tc.CAM['width'], tc.CAM['height']))
ann = rep.AnnotatorRegistry.get_annotator('rgb')
ann.attach(rp)
rng = np.random.default_rng(0)
for k, (f, (s, x, y, yaw)) in enumerate(pick):
    t_op.Set(Gf.Vec3d(x, y, 0.0))
    r_op.Set(math.degrees(yaw))
    for _ in range(12):
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
    sim = np.asarray(ann.get_data())[:, :, :3].copy()
    simc = camera_model(np.ascontiguousarray(sim[:, :, ::-1]), rng)[:, :, ::-1]
    real = np.asarray(Image.open(f).convert('RGB'))
    canvas = Image.new('RGB', (1920 + 20, 360 + 24), (30, 30, 30))
    for j, (im, title) in enumerate(((real, f'REAL D1000 left  {os.path.basename(f)}'),
                                     (sim, f'SIM {args.look}  x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.0f}'),
                                     (simc, 'SIM + D1000 camera model'))):
        canvas.paste(Image.fromarray(im), (650 * j, 24))
        ImageDraw.Draw(canvas).text((650 * j + 6, 5), title, fill=(255, 255, 255))
    canvas.save(os.path.join(args.out, f'{prefix}sim_vs_real_{k:02d}.png'))
print('VIEWS_DONE', len(pick))
app.close()
