"""Synthetic training data from the robot's real viewpoint (Replicator annotators + domain randomisation).

  python gen_dataset.py --frames 1500 --out E:\\twin\\datasets\\v1

Scene: house twin (look scan|clean), ceiling lights, IRA people wandering on a NavMesh, procedural
railings / baluster rows (class railing), optional furniture from the Isaac ArchVis library.
Camera: real D1000 left intrinsics (640x360, fx=fy=349.2, cx=318.04, cy=174.52), height 0.13 m
(tape-measured lens height; jittered 0.11-0.16 m), pitch +-4 deg, roll +-2 deg, positions on drivable
free space near the recorded robot path, often aimed near a person.
Randomised per frame: camera pose, light intensity / colour temperature / on-off, dome light,
railings; every --look-every frames: look (scan / clean), floor / wall / furniture colours,
furniture placement. People walk and idle continuously (different characters / clothing).

Output (YOLO layout + extras):
  images/NNNNNN.png      RGB (sim, no camera-noise model; see README for augmentation)
  labels/NNNNNN.txt      YOLO: class cx cy w h (0 person, 1 railing), tight visible boxes
  depth/NNNNNN.png       uint16 mm, 0 outside the D1000's 0.3-7 m validity
  semantic/NNNNNN.png    uint8 class id (see classes.json)
  meta/NNNNNN.json       camera pose, lights, look, per-box visibility and distance
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--frames', type=int, default=1500)
ap.add_argument('--out', default=r'E:\twin\datasets\v1')
ap.add_argument('--people', type=int, default=8)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--look-every', type=int, default=25)
ap.add_argument('--furniture', type=int, default=0, help='extra ArchVis furniture assets to scatter (0 = none)')
ap.add_argument('--subframes', type=int, default=6)
ap.add_argument('--look', default='remodel', choices=['remodel', 'mix'],
                help="house: 'remodel' (floor plan, default) or 'mix' (v1: RTAB-Map scan mesh / grid blocks)")
ap.add_argument('--cameras', type=int, default=4, help='camera rigs rendered per Replicator step (batched)')
ap.add_argument('--worker', type=int, default=0, help='index of this generator process (file prefix, seed offset)')
ap.add_argument('--camera-model', type=int, default=1,
                help='1: images/ get the D1000 camera model (eval/camera_model.py), raw RGB -> images_raw/*.jpg')
ap.add_argument('--rail-prob', type=float, default=0.35, help='chance per camera of extra procedural railings')
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting'})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.client  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt  # noqa: E402

import house_lib  # noqa: E402
import twin_common as tc  # noqa: E402
from twin_scene import TwinScene, app_update, assets_root  # noqa: E402

CLASSES = {'unlabelled': 0, 'floor': 1, 'wall': 2, 'furniture': 3, 'ceiling': 4, 'person': 5, 'railing': 6,
           'robot': 7, 'stairs': 8, 'door': 9, 'window': 10}
YOLO = {'person': 0, 'railing': 1}
SEED = args.seed * 1000 + args.worker
rng = random.Random(SEED)
np_rng = np.random.default_rng(SEED)
for d in ('images', 'labels', 'depth', 'semantic', 'meta') + (('images_raw',) if args.camera_model else ()):
    os.makedirs(os.path.join(args.out, d), exist_ok=True)
sys.path.insert(0, os.path.join(tc.REPO_ISAAC, 'eval'))
from camera_model import apply as camera_model  # noqa: E402

rep.settings.set_render_rtx_realtime(antialiasing='DLAA')
sc = TwinScene(look='remodel' if args.look == 'remodel' else 'scan', seed=SEED)
stage = sc.stage
lights = sc.add_lights()
sc.enable_people_runtime()
sc.bake_navmesh()
sc.spawn_people(args.people)
occ = sc.obstacles
res, (ox, oy) = sc.meta['resolution'], sc.meta['origin'][:2]
free = sc.free_xy

# ---------------- camera rigs (same optical chain as the robot asset), rendered as one batch ----------------
rigs = []
for k in range(args.cameras):
    rig = UsdGeom.Xform.Define(stage, f'/World/SDG/Rig{k}')
    rx = UsdGeom.Xformable(rig)
    ops = (rx.AddTranslateOp(), rx.AddRotateZOp(), rx.AddRotateYOp(), rx.AddRotateXOp())
    optical = UsdGeom.Xform.Define(stage, f'/World/SDG/Rig{k}/Optical')
    UsdGeom.Xformable(optical).AddOrientOp().Set(Gf.Quatf(0.5, -0.5, 0.5, -0.5))
    cam = UsdGeom.Camera.Define(stage, f'/World/SDG/Rig{k}/Optical/Camera')
    UsdGeom.Xformable(cam).AddOrientOp().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))
    tc.set_pinhole(cam.GetPrim())
    rp = rep.create.render_product(str(cam.GetPath()), (tc.CAM['width'], tc.CAM['height']))
    ann = {
        'rgb': rep.AnnotatorRegistry.get_annotator('rgb'),
        'depth': rep.AnnotatorRegistry.get_annotator('distance_to_image_plane'),
        'sem': rep.AnnotatorRegistry.get_annotator('semantic_segmentation', init_params={'colorize': False}),
        'bbox': rep.AnnotatorRegistry.get_annotator('bounding_box_2d_tight', init_params={'semanticTypes': ['class']}),
    }
    for a in ann.values():
        a.attach(rp)
    rigs.append({'ops': ops, 'ann': ann})

# ---------------- railings (procedural baluster rows) ----------------
RAIL_COLORS = [(0.05, 0.05, 0.05), (0.92, 0.92, 0.9), (0.35, 0.2, 0.1), (0.55, 0.55, 0.58), (0.2, 0.12, 0.07)]
rails = []
for i in range(2 * args.cameras):
    m = UsdGeom.Mesh.Define(stage, f'/World/SDG/Railing_{i}')
    m.CreateSubdivisionSchemeAttr('none')
    mat = UsdShade.Material.Define(stage, f'/World/SDG/Looks/rail_{i}')
    sh = UsdShade.Shader.Define(stage, f'/World/SDG/Looks/rail_{i}/S')
    sh.CreateIdAttr('UsdPreviewSurface')
    col = sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)
    rough = sh.CreateInput('roughness', Sdf.ValueTypeNames.Float)
    met = sh.CreateInput('metallic', Sdf.ValueTypeNames.Float)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    tc.add_semantic_label(m.GetPrim(), 'railing')
    rails.append({'mesh': m, 'col': col, 'rough': rough, 'met': met})


def box(cx, cy, cz, sx, sy, sz, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dz in (-sz / 2, sz / 2):
        for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)):
            pts.append((cx + c * dx - s * dy, cy + s * dx + c * dy, cz + dz))
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return pts, faces


def build_railing(r, x, y, yaw, length, height, spacing, post_w, round_posts):
    pts, idx = [], []

    def add(b):
        p, f = b
        base = len(pts)
        pts.extend(p)
        for q in f:
            idx.extend(base + i for i in q)
    n = max(2, int(length / spacing) + 1)
    c, s = math.cos(yaw), math.sin(yaw)
    for k in range(n):
        d = -length / 2 + k * length / (n - 1)
        add(box(x + c * d, y + s * d, height / 2, post_w, post_w, height, yaw + (math.pi / 4 if round_posts else 0)))
    add(box(x, y, height + 0.03, length + 0.06, 0.06, 0.05, yaw))          # handrail
    if rng.random() < 0.5:
        add(box(x, y, 0.06, length, 0.05, 0.04, yaw))                      # bottom rail / stair stringer
    m = r['mesh']
    m.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.GetFaceVertexCountsAttr().Set(Vt.IntArray([4] * (len(idx) // 4)))
    m.GetFaceVertexIndicesAttr().Set(Vt.IntArray(idx))
    a = np.array(pts)
    m.GetExtentAttr().Set([Gf.Vec3f(*a.min(0).tolist()), Gf.Vec3f(*a.max(0).tolist())])


for r in rails:
    r['mesh'].CreatePointsAttr()
    r['mesh'].CreateFaceVertexCountsAttr()
    r['mesh'].CreateFaceVertexIndicesAttr()
    r['mesh'].CreateExtentAttr()

# ---------------- furniture from the ArchVis library ----------------
furniture = []
if args.furniture > 0:
    root = assets_root()
    cats = ['Chairs', 'CoffeeTables', 'Sofas', 'EndTables', 'Bookshelves', 'Tables', 'BarStools']
    for cat in cats:
        if len(furniture) >= args.furniture:
            break
        base = f'{root}/NVIDIA/Assets/ArchVis/Residential/Furniture/{cat}'
        rr, entries = omni.client.list(base)
        usds = [e.relative_path for e in entries if e.relative_path.endswith('.usd')] if rr == omni.client.Result.OK else []
        if not usds:
            continue
        pick = rng.choice(usds)
        p = stage.DefinePrim(f'/World/SDG/Furniture_{len(furniture)}', 'Xform')
        p.GetReferences().AddReference(f'{base}/{pick}')
        xf = UsdGeom.Xformable(p)
        furniture.append({'prim': p, 't': xf.AddTranslateOp(), 'r': xf.AddRotateZOp(), 'asset': f'{cat}/{pick}'})
        tc.add_semantic_label(p, 'furniture')
    app_update(10)
    for f in furniture:
        bb = UsdGeom.BBoxCache(0, ['default', 'render']).ComputeWorldBound(f['prim']).ComputeAlignedRange()
        size = bb.GetSize()
        f['scale'] = 0.01 if max(size) > 20 else 1.0          # some ArchVis assets are authored in cm
        if f['scale'] != 1.0:
            UsdGeom.Xformable(f['prim']).AddScaleOp().Set(Gf.Vec3f(f['scale']))
        f['radius'] = 0.5 * math.hypot(size[0], size[1]) * f['scale']
    print('FURNITURE', [(f['asset'], round(f['radius'], 2)) for f in furniture], flush=True)

# ---------------- helpers ----------------


def cell(x, y):
    return int((y - oy) / res), int((x - ox) / res)


def line_clear(x0, y0, x1, y1, skip_end=0.4):
    d = math.hypot(x1 - x0, y1 - y0)
    n = int(d / (res * 0.5))
    for k in range(1, n):
        t = k / n
        if d * (1 - t) < skip_end:
            break
        r_, c_ = cell(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        if 0 <= r_ < occ.shape[0] and 0 <= c_ < occ.shape[1] and occ[r_, c_]:
            return False
    return True


def people_xy():
    try:
        from omni.metropolis.pipeline.agent import AgentsManager
        out = []
        for a in AgentsManager.get_instance().get_runtime_agent_instances():
            p = a.get_world_position()
            if p is not None:
                out.append((p[0], p[1]))
        return out
    except Exception:  # noqa: BLE001
        return []


def sample_camera():
    pp = people_xy()
    if pp and rng.random() < 0.75:
        for _ in range(40):
            px, py = rng.choice(pp)
            d = rng.uniform(0.9, 6.5) if rng.random() < 0.8 else rng.uniform(6.5, 9.0)
            b = rng.uniform(-math.pi, math.pi)
            cx_, cy_ = px + d * math.cos(b), py + d * math.sin(b)
            k = np.argmin(np.hypot(free[:, 0] - cx_, free[:, 1] - cy_))
            if math.hypot(free[k, 0] - cx_, free[k, 1] - cy_) > 0.08:
                continue
            if not line_clear(cx_, cy_, px, py):
                continue
            if min(math.hypot(qx - cx_, qy - cy_) for qx, qy in pp) < 0.6:     # not inside / against a person
                continue
            yaw = math.atan2(py - cy_, px - cx_) + math.radians(rng.uniform(-38, 38))
            return cx_, cy_, yaw, 'person'
    x, y, yaw = sc.path[rng.randrange(len(sc.path))]
    for _ in range(20):
        jx, jy = x + rng.gauss(0, 0.15), y + rng.gauss(0, 0.15)
        k = np.argmin(np.hypot(free[:, 0] - jx, free[:, 1] - jy))
        if math.hypot(free[k, 0] - jx, free[k, 1] - jy) < 0.08:
            x, y = jx, jy
            break
    return x, y, yaw + math.radians(rng.uniform(-30, 30)), 'path'


def place_in_view(cx_, cy_, yaw, dmin, dmax, clearance):
    for _ in range(30):
        d = rng.uniform(dmin, dmax)
        a = yaw + math.radians(rng.uniform(-40, 40))
        x, y = cx_ + d * math.cos(a), cy_ + d * math.sin(a)
        r_, c_ = cell(x, y)
        if not (0 <= r_ < occ.shape[0] and 0 <= c_ < occ.shape[1]):
            continue
        k = int(math.ceil(clearance / res))
        if occ[max(0, r_ - k):r_ + k + 1, max(0, c_ - k):c_ + k + 1].any():
            continue
        if sc.look == 'remodel':
            if np.min(np.hypot(free[:, 0] - x, free[:, 1] - y)) > 0.3:     # indoors, drivable area
                continue
        elif sc.grid[r_, c_] != house_lib.FREE:
            continue
        return x, y
    return None


def house_shader(name):
    return UsdShade.Shader(stage.GetPrimAtPath(f'/World/House/Looks/{name}/Surface'))


def randomize_look():
    if sc.look == 'remodel':
        return 'remodel', sc.randomize_remodel(rng)
    look = 'scan' if rng.random() < 0.5 else 'clean'
    sc.set_look(look)
    tones = [(0.36, 0.33, 0.29), (0.55, 0.5, 0.42), (0.25, 0.22, 0.2), (0.6, 0.58, 0.55), (0.42, 0.3, 0.2),
             (0.3, 0.32, 0.34), (0.7, 0.66, 0.6)]
    floor = rng.choice(tones)
    house_shader('floor').GetInput('diffuseColor').Set(Gf.Vec3f(*(np.clip(np.array(floor) * rng.uniform(0.7, 1.3), 0, 1))))
    house_shader('floor').GetInput('roughness').Set(rng.uniform(0.4, 1.0))
    wall = np.clip(np.array([rng.uniform(0.55, 0.95)] * 3) * np.array([1, rng.uniform(0.92, 1.05), rng.uniform(0.8, 1.02)]), 0, 1)
    house_shader('blocks_wall').GetInput('diffuseColor').Set(Gf.Vec3f(*wall))
    house_shader('blocks_furniture').GetInput('diffuseColor').Set(Gf.Vec3f(*np_rng.uniform(0.1, 0.7, 3)))
    return look, {}


LIGHT_BASE = 60000.0     # ceiling sphere light intensity giving ~real D1000 brightness (median Y ~60)


def randomize_lights():
    if sc.look == 'remodel':
        on = 0
        scale = rng.uniform(0.4, 3.0)
        temp = rng.uniform(2700, 6500)
        for L in lights:
            e = rng.random() < 0.85
            on += e
            L.GetIntensityAttr().Set(sc.light_base[str(L.GetPath())] * scale * rng.uniform(0.7, 1.3) if e else 0.0)
            L.GetColorTemperatureAttr().Set(temp + rng.uniform(-300, 300))
        sun = rng.choice([0.0, rng.uniform(500, 6000)])
        sc.sun.GetIntensityAttr().Set(sun)
        sc.sun_rot.Set(Gf.Vec3f(rng.uniform(-75, -25), 0.0, rng.uniform(0, 360)))
        dome = rng.uniform(50, 900)
        sc.dome.GetIntensityAttr().Set(dome)
        return {'room_light_scale': round(scale, 3), 'color_temp_k': round(temp), 'lights_on': on,
                'sun': round(sun), 'dome': round(dome, 1)}
    on = 0
    scale = rng.uniform(0.25, 2.5)
    temp = rng.uniform(2700, 6500)
    for L in lights:
        e = rng.random() < 0.8
        on += e
        L.GetIntensityAttr().Set(LIGHT_BASE * scale * rng.uniform(0.7, 1.3) if e else 0.0)
        L.GetColorTemperatureAttr().Set(temp + rng.uniform(-300, 300))
    dome = rng.uniform(50, 600)
    sc.dome.GetIntensityAttr().Set(dome)
    return {'ceiling_scale': round(scale, 3), 'color_temp_k': round(temp), 'lights_on': on, 'dome': round(dome, 1)}


def randomize_props(cx_, cy_, yaw, k=0):
    placed = []
    pool = rails[2 * k:2 * k + 2]
    for r in pool:
        UsdGeom.Imageable(r['mesh']).MakeInvisible()
    nr = (1 if rng.random() < 0.6 else 2) if rng.random() < args.rail_prob else 0
    for r in pool[:nr]:
        p = place_in_view(cx_, cy_, yaw, 0.7, 4.0, 0.1)
        if p is None:
            continue
        L = rng.uniform(0.8, 3.0)
        params = dict(length=L, height=rng.uniform(0.75, 1.05), spacing=rng.uniform(0.09, 0.16),
                      post_w=rng.uniform(0.015, 0.045), round_posts=rng.random() < 0.5)
        ryaw = yaw + math.pi / 2 + math.radians(rng.uniform(-60, 60))
        build_railing(r, p[0], p[1], ryaw, **params)
        r['col'].Set(Gf.Vec3f(*rng.choice(RAIL_COLORS)))
        metal = rng.random() < 0.5
        r['met'].Set(0.8 if metal else 0.0)
        r['rough'].Set(rng.uniform(0.2, 0.5) if metal else rng.uniform(0.4, 0.9))
        UsdGeom.Imageable(r['mesh']).MakeVisible()
        placed.append({'x': round(p[0], 2), 'y': round(p[1], 2), **{k: (round(v, 3) if isinstance(v, float) else v)
                                                                    for k, v in params.items()}})
    for f in (furniture if k == 0 else []):
        p = place_in_view(cx_, cy_, yaw, 1.0, 5.0, max(0.2, f['radius'] * 0.6)) if rng.random() < 0.6 else None
        if p is None:
            UsdGeom.Imageable(f['prim']).MakeInvisible()
            continue
        f['t'].Set(Gf.Vec3d(p[0], p[1], 0.0))
        f['r'].Set(rng.uniform(0, 360))
        UsdGeom.Imageable(f['prim']).MakeVisible()
    return placed


def label_map(info):
    """Replicator semantic id -> our class id."""
    lut = {}
    for k, v in info.get('idToLabels', {}).items():
        name = v.get('class', '') if isinstance(v, dict) else str(v)
        cls = 0
        for part in name.split(','):
            part = part.strip()
            if part in CLASSES:
                cls = CLASSES[part]
                break
        lut[int(k)] = cls
    return lut


# ---------------- main loop ----------------
tl = omni.timeline.get_timeline_interface()
tl.play()
app_update(120)                         # let the agents start walking
stats = {'frames': 0, 'person_boxes': 0, 'railing_boxes': 0, 'frames_with_person': 0, 'frames_with_railing': 0,
         'looks': {}, 'pose_mode': {'person': 0, 'path': 0}, 'cameras_per_step': args.cameras, 'worker': args.worker}
look, look_info = randomize_look()
t0 = time.time()
steps = (args.frames + args.cameras - 1) // args.cameras
i = -1
for step in range(steps):
    if step and step % max(1, args.look_every // args.cameras) == 0:
        look, look_info = randomize_look()
    light_info = randomize_lights()
    shots = []
    for k, rig in enumerate(rigs):
        cx_, cy_, yaw, mode = sample_camera()
        h = float(np.clip(rng.gauss(0.13, 0.01), 0.11, 0.16))
        pitch, roll = rng.uniform(-4, 4), rng.uniform(-2, 2)
        op_t, op_yaw, op_pitch, op_roll = rig['ops']
        op_t.Set(Gf.Vec3d(cx_, cy_, h))
        op_yaw.Set(math.degrees(yaw))
        op_pitch.Set(pitch)
        op_roll.Set(roll)
        shots.append((cx_, cy_, yaw, mode, h, pitch, roll, randomize_props(cx_, cy_, yaw, k)))
    app_update(3)                        # people keep moving between samples
    rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=False)
    for k, rig in enumerate(rigs):
        cx_, cy_, yaw, mode, h, pitch, roll, rails_info = shots[k]
        ann = rig['ann']
        i += 1
        rgb = np.asarray(ann['rgb'].get_data())[:, :, :3]
        depth = np.asarray(ann['depth'].get_data())
        sem = ann['sem'].get_data()
        bbox = ann['bbox'].get_data()
        lut = label_map(sem['info'])
        sem_ids = np.asarray(sem['data'])
        if sem_ids.size:
            lut_arr = np.zeros(max(max(lut, default=0), int(sem_ids.max())) + 1, np.uint8)
            for k_, v_ in lut.items():
                lut_arr[k_] = v_
            sem_cls = lut_arr[sem_ids]
        else:
            sem_cls = None
        name = f'w{args.worker}_{i:06d}'
        bgr_raw = np.ascontiguousarray(rgb[:, :, ::-1])
        if args.camera_model:
            cv2.imwrite(os.path.join(args.out, 'images_raw', name + '.jpg'), bgr_raw, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(args.out, 'images', name + '.png'), camera_model(bgr_raw, np_rng))
        else:
            cv2.imwrite(os.path.join(args.out, 'images', name + '.png'), bgr_raw)
        mm = np.where(np.isfinite(depth) & (depth >= 0.3) & (depth <= 7.0), depth * 1000.0, 0).astype(np.uint16)
        cv2.imwrite(os.path.join(args.out, 'depth', name + '.png'), mm)
        if sem_cls is not None:
            cv2.imwrite(os.path.join(args.out, 'semantic', name + '.png'), sem_cls)
        W, H = tc.CAM['width'], tc.CAM['height']
        lines, boxes = [], []
        bl = {int(k): (v.get('class', '') if isinstance(v, dict) else str(v)) for k, v in bbox['info'].get('idToLabels', {}).items()}
        for b in np.asarray(bbox['data']):
            cls_name = bl.get(int(b['semanticId']), '')
            cls_name = next((c for c in cls_name.split(',') if c.strip() in YOLO), None)
            if cls_name is None:
                continue
            x1, y1, x2, y2 = float(b['x_min']), float(b['y_min']), float(b['x_max']) + 1, float(b['y_max']) + 1
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < 3 or y2 - y1 < 3 or (x2 - x1) * (y2 - y1) > 0.85 * W * H:
                continue
            occ_r = float(b['occlusionRatio']) if 'occlusionRatio' in b.dtype.names else -1.0
            vis = 1.0 - occ_r if occ_r >= 0 else None          # -1 = not computed by this Replicator build
            reg = mm[int(y1):int(y2), int(x1):int(x2)]
            if sem_cls is not None:
                msk = sem_cls[int(y1):int(y2), int(x1):int(x2)] == CLASSES[cls_name.strip()]
                vals = reg[msk & (reg > 0)]
            else:
                vals = reg[reg > 0]
            dist = float(np.median(vals)) / 1000.0 if vals.size else None
            rec = {'cls': cls_name.strip(), 'box': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                   'visible_frac': round(vis, 3) if vis is not None else None, 'distance_m': round(dist, 2) if dist else None}
            boxes.append(rec)
            # YOLO label: keep boxes with >= 30 % of the object visible and >= 6 px tall
            if (vis is None or vis >= 0.30) and (y2 - y1) >= 6:
                c = YOLO[rec['cls']]
                lines.append(f'{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
        with open(os.path.join(args.out, 'labels', name + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        persons = [b for b in boxes if b['cls'] == 'person']
        railings = [b for b in boxes if b['cls'] == 'railing']
        meta = {'frame': i, 'name': name, 'look': look, 'look_info': look_info, 'pose_mode': mode,
                'camera': {'x': round(cx_, 3), 'y': round(cy_, 3), 'z': round(h, 3), 'yaw_deg': round(math.degrees(yaw), 2),
                           'pitch_deg': round(pitch, 2), 'roll_deg': round(roll, 2)},
                'lights': light_info, 'railings': rails_info, 'persons': persons, 'railing_boxes': railings,
                'people_world_xy': [[round(a, 2), round(b, 2)] for a, b in people_xy()]}
        with open(os.path.join(args.out, 'meta', name + '.json'), 'w') as f:
            json.dump(meta, f)
        stats['frames'] += 1
        stats['person_boxes'] += sum(1 for ln in lines if ln.startswith('0 '))
        stats['railing_boxes'] += sum(1 for ln in lines if ln.startswith('1 '))
        stats['frames_with_person'] += any(ln.startswith('0 ') for ln in lines)
        stats['frames_with_railing'] += any(ln.startswith('1 ') for ln in lines)
        stats['looks'][look] = stats['looks'].get(look, 0) + 1
        stats['pose_mode'][mode] += 1
        if i % 200 == 0:
            el = time.time() - t0
            print(f'FRAME {i} {el:.0f}s {el / (i + 1):.2f}s/frame persons {stats["frames_with_person"]} '
                  f'railings {stats["frames_with_railing"]}', flush=True)
tl.stop()
stats['seconds'] = round(time.time() - t0, 1)
stats['classes_yolo'] = YOLO
stats['classes_semantic'] = CLASSES
stats['furniture_assets'] = [f['asset'] for f in furniture]
stats['people'] = [p['asset'].split('/Characters/')[-1] for p in sc.people]
with open(os.path.join(args.out, 'classes.json'), 'w') as f:
    json.dump({'yolo': YOLO, 'semantic': CLASSES}, f, indent=1)
with open(os.path.join(args.out, f'stats_w{args.worker}.json'), 'w') as f:
    json.dump(stats, f, indent=1)
with open(os.path.join(args.out, 'data.yaml'), 'w') as f:
    f.write(f'path: {args.out}\ntrain: images\nval: images\nnames:\n  0: person\n  1: railing\n')
print('DATASET_DONE ' + json.dumps(stats), flush=True)
app.close()
