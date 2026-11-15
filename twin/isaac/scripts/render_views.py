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
