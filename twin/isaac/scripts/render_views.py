"""Render the house twin for inspection: a top-down overview, and robot-eye views at the poses where
real frames were captured (side by side with the real D1000 image).

  python render_views.py --n 8
Writes E:\\twin\\screens\\house_topdown.png, house_oblique.png, sim_vs_real_XX.png
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--real', default=r'E:\twin\real_frames\left')
ap.add_argument('--out', default=r'E:\twin\screens')
ap.add_argument('--dome', type=float, default=300.0)
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting'})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

import twin_common as tc  # noqa: E402

rep.settings.set_render_rtx_realtime(antialiasing='DLAA')   # DLAA/DLSS carry the denoiser; FXAA leaves path-tracing noise
from twin_scene import TwinScene  # noqa: E402

sc = TwinScene(look='scan')
stage = sc.stage
sc.add_lights(dome=args.dome)
house = sc.house
look = house.GetVariantSets().GetVariantSet('look')
sc.add_robot(0.0, 0.0, 0.0)
t_op, r_op = sc.robot_t, sc.robot_r
left_cam = sc.cam_paths['left']
ceiling = UsdGeom.Imageable(stage.GetPrimAtPath('/World/House/Ceiling'))

# overview cameras
top = UsdGeom.Camera.Define(stage, '/World/TopCam')
top.CreateFocalLengthAttr(18.0)
top.CreateHorizontalApertureAttr(20.955)
top.CreateVerticalApertureAttr(20.955)
top.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100))
UsdGeom.Xformable(top).AddTranslateOp().Set(Gf.Vec3d(2.0, 1.5, 22.0))
obl = UsdGeom.Camera.Define(stage, '/World/ObliqueCam')
obl.CreateFocalLengthAttr(14.0)
obl.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100))
ox = UsdGeom.Xformable(obl)
ox.AddTranslateOp().Set(Gf.Vec3d(-6.0, -9.0, 9.0))
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
ceiling.MakeInvisible()        # overview cameras look down through the ceiling
for lk in ('scan', 'clean'):
    look.SetVariantSelection(lk)
    Image.fromarray(grab('/World/TopCam', 1280, 1280, 20)).save(os.path.join(args.out, f'house_topdown_{lk}.png'))
    Image.fromarray(grab('/World/ObliqueCam', 1600, 900, 20)).save(os.path.join(args.out, f'house_oblique_{lk}.png'))

ceiling.MakeVisible()
# robot-eye views at real frame poses
poses = []
for ln in open(tc.POSES_TXT):
    if ln.strip() and not ln.startswith('#'):
        s, x, y, z, qx, qy, qz, qw, i = map(float, ln.split())
        poses.append((s, x, y, math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))))
poses = np.array(poses)
files = sorted(glob.glob(os.path.join(args.real, '*.png')))
stamps = np.array([int(os.path.splitext(os.path.basename(f))[0]) / 1e9 for f in files])
cands = []
for f, s in zip(files, stamps):
    j = int(np.argmin(np.abs(poses[:, 0] - s)))
    if abs(poses[j, 0] - s) < 0.15:     # node captured within 150 ms of this frame
        cands.append((f, poses[j]))
print('frames with a matching graph node:', len(cands))
pick = [cands[i] for i in np.linspace(0, len(cands) - 1, min(args.n, len(cands))).astype(int)] if cands else []
rp = rep.create.render_product(left_cam, (tc.CAM['width'], tc.CAM['height']))
ann = rep.AnnotatorRegistry.get_annotator('rgb')
ann.attach(rp)
meta = []
for k, (f, (s, x, y, yaw)) in enumerate(pick):
    t_op.Set(Gf.Vec3d(x, y, 0.0))
    r_op.Set(math.degrees(yaw))
    sims = {}
    for lk in ('scan', 'clean'):
        look.SetVariantSelection(lk)
        for _ in range(10):
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
        sims[lk] = np.asarray(ann.get_data())[:, :, :3].copy()
    real = np.asarray(Image.open(f).convert('RGB'))
    canvas = Image.new('RGB', (1920 + 20, 360 + 24), (30, 30, 30))
    canvas.paste(Image.fromarray(real), (0, 24))
    canvas.paste(Image.fromarray(sims['scan']), (650, 24))
    canvas.paste(Image.fromarray(sims['clean']), (1300, 24))
    d = ImageDraw.Draw(canvas)
    d.text((6, 5), f'REAL D1000 left  {os.path.basename(f)}', fill=(255, 255, 255))
    d.text((656, 5), f'SIM look=scan  x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.0f}', fill=(255, 255, 255))
    d.text((1306, 5), 'SIM look=clean (grid blocks, same pose)', fill=(255, 255, 255))
    out = os.path.join(args.out, f'sim_vs_real_{k:02d}.png')
    canvas.save(out)
    meta.append({'real': f, 'x': x, 'y': y, 'yaw_deg': math.degrees(yaw), 'out': out})
tc.save_json(meta, os.path.join(args.out, 'sim_vs_real_views.json'))
print('VIEWS_DONE', len(meta))
app.close()
