"""Pick render settings: render one robot-eye view of the house with several anti-aliasing /
sub-frame settings and report the image noise (Immerkaer sigma) and time per frame."""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting'})
import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

import twin_common as tc  # noqa: E402

sys.path.insert(0, os.path.join(tc.REPO_ISAAC, 'eval'))
from sim_vs_real import noise_sigma  # noqa: E402

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
stage.DefinePrim('/World/House', 'Xform').GetReferences().AddReference(tc.HOUSE_USD)
UsdLux.DomeLight.Define(stage, '/World/Dome').CreateIntensityAttr(1000)
info = json.load(open(os.path.splitext(tc.ROBOT_USD)[0] + '_info.json'))
robot = stage.DefinePrim('/World/Roomba', 'Xform')
robot.GetReferences().AddReference(tc.ROBOT_USD)
rx = UsdGeom.Xformable(robot)
rx.AddTranslateOp().Set(Gf.Vec3d(1.42, -0.17, 0))
rx.AddRotateZOp().Set(25.0)
cam = info['cameras']['left'].replace(info['root'], '/World/Roomba', 1)
rp = rep.create.render_product(cam, (640, 360))
ann = rep.AnnotatorRegistry.get_annotator('rgb')
ann.attach(rp)
s = carb.settings.get_settings()
res = []
for aa, sub, extra in [('FXAA', 4, {}), ('FXAA', 32, {}), ('DLAA', 4, {}), ('DLSS', 4, {}),
                       ('DLAA', 16, {}), ('TAA', 16, {}),
                       ('FXAA', 4, {'/rtx/rendermode': 'RaytracedLighting'})]:
    rep.settings.set_render_rtx_realtime(antialiasing=aa)
    for k, v in extra.items():
        s.set(k, v)
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=sub, delta_time=0.0)
    t0 = time.time()
    rep.orchestrator.step(rt_subframes=sub, delta_time=0.0)
    dt = time.time() - t0
    img = np.asarray(ann.get_data())[:, :, :3]
    g = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)
    name = f'{aa}_{sub}' + ('_' + '_'.join(str(v) for v in extra.values()) if extra else '')
    Image.fromarray(img).save(os.path.join(tc.TWIN, 'work', f'render_{name}.png'))
    res.append({'name': name, 'rendermode': s.get('/rtx/rendermode'), 'noise_sigma': round(noise_sigma(g), 2),
                'mean': round(float(g.mean()), 1), 'sec_per_frame': round(dt, 3)})
    print('RENDER_TEST', json.dumps(res[-1]), flush=True)
app.close()
