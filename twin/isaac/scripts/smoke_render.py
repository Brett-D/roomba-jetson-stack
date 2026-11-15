"""Headless RTX smoke test: start Isaac Sim, build a tiny scene, render one frame, save it.

  E:\\twin\\isaacsim-env\\Scripts\\python.exe smoke_render.py --out E:\\twin\\screens\\smoke_render.png
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=r'E:\twin\screens\smoke_render.png')
args, _ = ap.parse_known_args()

t0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'width': 640, 'height': 360})
t_start = time.time() - t0

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux  # noqa: E402

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdLux.DomeLight.Define(stage, '/World/Dome').CreateIntensityAttr(800)
sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
sun.CreateIntensityAttr(2500)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40, 20, 0))
floor = UsdGeom.Cube.Define(stage, '/World/Floor')
floor.CreateSizeAttr(1.0)
UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3f(10, 10, 0.02))
floor.CreateDisplayColorAttr([(0.6, 0.55, 0.5)])
for i, c in enumerate([(0.8, 0.1, 0.1), (0.1, 0.7, 0.2), (0.1, 0.3, 0.9)]):
    s = UsdGeom.Sphere.Define(stage, f'/World/Ball{i}')
    s.CreateRadiusAttr(0.25)
    UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(2.0, -0.8 + 0.8 * i, 0.26))
    s.CreateDisplayColorAttr([c])

cam = rep.create.camera(position=(0, 0, 0.13), look_at=(2.0, 0, 0.26))
rp = rep.create.render_product(cam, (640, 360))
rgb = rep.AnnotatorRegistry.get_annotator('rgb')
rgb.attach(rp)
for _ in range(20):
    rep.orchestrator.step(rt_subframes=4)
img = np.asarray(rgb.get_data())[:, :, :3]
os.makedirs(os.path.dirname(args.out), exist_ok=True)
Image.fromarray(img).save(args.out)

import torch  # noqa: E402

info = {
    'startup_s': round(t_start, 1),
    'total_s': round(time.time() - t0, 1),
    'image_mean': float(img.mean()),
    'image_std': float(img.std()),
    'kit_version': app.app.get_build_version() if hasattr(app, 'app') else None,
    'renderer': carb.settings.get_settings().get('/renderer/active'),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'torch': torch.__version__,
}
print('SMOKE_RESULT ' + json.dumps(info), flush=True)
app.close()
sys.exit(0 if img.std() > 1.0 else 1)
