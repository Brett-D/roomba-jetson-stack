"""Build the remodelled house USD from the digitised floor plan, registered into the RTAB-Map map frame.

  python build_remodel.py --with-floor1   # upstairs -> E:\\twin\\assets\\usd\\house_remodel.usd + floorplan.json/.svg
  python build_remodel.py --floor1        # FLOOR 1  -> house_remodel_f1.usd + floorplan_floor1.json/.svg (stacked below)

/Remodel  (xform: translate t, rotateZ angle, scale (s, s, 1) = plan metres -> map metres)
  Looks/        UsdPreviewSurface materials (procedural textures in assets\\remodel\\textures)
  Walls, Trim, Glass, Baseboards, Ceilings, Stairs, Railings, Skirt, Fixtures   merged meshes
  Floors/<room>  one mesh per room (material randomisable per room)
  Doors/<id>/Leaf<k>   hinged leaves, rotateZ op; attributes twin:open_yaw / twin:closed_yaw
  Furniture/<role>_<k>  ArchVis references (+ invisible box colliders) or procedural beds
  Lights/       one sphere light per room (more in the hallway), chandelier light over the stair well
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))

ap = argparse.ArgumentParser()
ap.add_argument('--plan', default=None, help='default: floorplan_upstairs_px.json (floorplan_floor1_px.json with --floor1)')
ap.add_argument('--image', default=None, help='default: floorplan_upstairs.png (floorplan_downstairs.png with --floor1)')
ap.add_argument('--floor1', action='store_true', help='build the downstairs (FLOOR 1) house, stacked below the upstairs')
ap.add_argument('--upstairs-plan', default=os.path.join(HERE, 'floorplan_upstairs_px.json'))
ap.add_argument('--upstairs-image', default=r'E:\twin\assets\floorplan_upstairs.png')
ap.add_argument('--registration', default=r'E:\twin\assets\remodel\registration.json')
ap.add_argument('--catalog', default=r'E:\twin\assets\remodel\asset_catalog.json')
ap.add_argument('--out', default=None, help='default: E:\\twin\\assets\\usd\\house_remodel[_f1].usd')
ap.add_argument('--plan-out', default=r'E:\twin\assets\remodel')
ap.add_argument('--seed', type=int, default=3)
ap.add_argument('--no-grid-furniture', action='store_true')
ap.add_argument('--with-floor1', action='store_true',
                help='the FLOOR 1 house (build_floor1.py) is stacked below: no placeholder foyer floor, slab fascia only')
args, _ = ap.parse_known_args()
F1 = args.floor1
args.plan = args.plan or os.path.join(HERE, 'floorplan_floor1_px.json' if F1 else 'floorplan_upstairs_px.json')
args.image = args.image or (r'E:\twin\assets\floorplan_downstairs.png' if F1 else args.upstairs_image)
args.out = args.out or (r'E:\twin\assets\usd\house_remodel_f1.usd' if F1 else r'E:\twin\assets\usd\house_remodel.usd')
OUT_TAG = '_floor1' if F1 else ''

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})

import cv2  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

import furnish as fu  # noqa: E402
import house_lib  # noqa: E402
import plan_lib as pl  # noqa: E402
import remodel_lib as rl  # noqa: E402
from twin_common import add_semantic_label  # noqa: E402

rng = np.random.default_rng(args.seed)
plan = json.load(open(args.plan))
reg = json.load(open(args.registration))
best = reg['best']
T = {'mirror': best['mirror'], 'angle': best['angle'], 't': np.array(best['t']), 'scale': best['scale']}
gray = pl.load_gray(args.image)
wmask = pl.wall_mask(gray)
for x0_, y0_, x1_, y1_ in plan.get('mask_out', []):
    wmask[y0_:y1_, x0_:x1_] = False
if F1:
    # same drawing scale as upstairs; floor-1 pixels map into the upstairs plan frame via the stack offset
    up_shape = pl.load_gray(args.upstairs_image).shape
    frame = pl.PlanFrame(reg['px_per_m'], up_shape[1], up_shape[0])
    OFF = np.array(plan['stack']['offset_px_from_upstairs'], float)
    frame.cx += OFF[0]
    frame.cy += OFF[1]
    rl.WALL_H = plan['stack']['wall_height_m']
    FLOOR_Z = plan['stack']['floor_z_m']
    up_plan = json.load(open(args.upstairs_plan))
    hole_px = (np.asarray(pl.stair_void_polygon(up_plan), float) + OFF).tolist()    # double-height entry
else:
    frame = pl.PlanFrame(reg['px_per_m'], gray.shape[1], gray.shape[0])
    FLOOR_Z = 0.0
S = frame.s
shape = wmask.shape
void_px = None if F1 else pl.stair_void_polygon(plan)
void_mask = rl.poly_mask(shape, void_px)
void_m = frame.m(void_px) if void_px else None
hole_mask = rl.poly_mask(shape, hole_px) if F1 else void_mask
rooms_in = [r for r in plan['rooms'] if not r.get('outdoor')]
els = []        # merged-mesh elements

# ---------------- walls (drawn wall pixels) ----------------
for x0, y0, x1, y1 in pl.wall_boxes(wmask, frame):
    els.append(rl.box(((x0 + x1) / 2, (y0 + y1) / 2, rl.WALL_H / 2), (x1 - x0, y1 - y0, rl.WALL_H), 0, 'wall', 'wall'))
T_WALL = 5 / S
# openings
door_specs = []
for d in plan['doors']:
    a, b = frame.m(d['a']), frame.m(d['b'])
    t = d.get('wall_t', 5) / S
    if d['type'] == 'glass':
        els += rl.window_elements(a, b, t, 0.0, rl.DOOR_H)
        els += [e for e in rl.door_frame_elements(a, b, t) if e['label'] == 'wall']
        continue
    if d['type'] == 'opening':          # cased opening: header only
        els += [e for e in rl.door_frame_elements(a, b, t) if e['label'] == 'wall']
        continue
    if d['type'] == 'garage':           # overhead door, closed
        els += [e for e in rl.door_frame_elements(a, b, t, head=2.13) if e['label'] == 'wall']
        els.append(rl.seg_box(a, b, 0.05, 0.0, 2.13, 'door', 'door', True))
        continue
    els += rl.door_frame_elements(a, b, t)
    sw = {'left': (-1, 0), 'right': (1, 0), 'up': (0, 1), 'down': (0, -1)}[d['swing']]    # plan coords
    door_specs.append({'id': d['id'], 'type': d['type'], 'a': a.tolist(), 'b': b.tolist(), 'closed': bool(d.get('closed')),
                       'leaves': rl.door_leaves(a, b, d['type'], np.array(sw, float))})
for w in plan['windows']:
    els += rl.window_elements(frame.m(w['a']), frame.m(w['b']), T_WALL, w['sill_m'], w['head_m'])


def auto_window_segments(plan, wmask, shape):
    """Gaps in the exterior wall line (outline of the indoor rooms) that are not doors -> window segments (px)."""
    cfg = plan['auto_windows']
    U = np.any([rl.poly_mask(shape, r['poly']) for r in plan['rooms'] if not r.get('outdoor')], axis=0).astype(np.uint8)
    U = cv2.morphologyEx(U, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(U, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    near = cv2.dilate(wmask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    dm = np.zeros(shape, np.uint8)
    for d in plan['doors']:
        cv2.line(dm, tuple(int(round(v)) for v in d['a']), tuple(int(round(v)) for v in d['b']), 1, 13)
    segs = []
    for c in cs:
        P = c[:, 0, :]
        gap = ~near[P[:, 1], P[:, 0]] & ~dm[P[:, 1], P[:, 0]].astype(bool)
        if gap.all() or not gap.any():
            continue
        k0 = int(np.argmin(gap))                       # start the circular scan on a wall pixel
        P, gap = np.roll(P, -k0, 0), np.roll(gap, -k0)
        k = 0
        while k < len(P):
            if not gap[k]:
                k += 1
                continue
            k1 = k
            while k1 < len(P) and gap[k1]:
                k1 += 1
            run = P[max(k - 1, 0):min(k1 + 1, len(P))]
            if len(run) >= cfg['min_len_px']:
                ap_ = cv2.approxPolyDP(run.reshape(-1, 1, 2).astype(np.int32), 1.5, False)[:, 0, :]
                for a_, b_ in zip(ap_[:-1], ap_[1:]):
                    if np.linalg.norm(b_ - a_) >= 4:
                        segs.append((a_.tolist(), b_.tolist()))
            k = k1
    return segs


auto_windows = []
if 'auto_windows' in plan:
    for a_, b_ in auto_window_segments(plan, wmask, shape):
        auto_windows.append({'a': a_, 'b': b_, 'sill_m': plan['auto_windows']['sill_m'], 'head_m': plan['auto_windows']['head_m']})
        els += rl.window_elements(frame.m(a_), frame.m(b_), T_WALL, plan['auto_windows']['sill_m'], plan['auto_windows']['head_m'])
    print('AUTO_WINDOWS', len(auto_windows))

# ---------------- floors, ceilings, baseboards ----------------
floor_meshes = []
claimed = np.zeros(shape, bool)
for i, r in enumerate(plan['rooms']):
    m = rl.poly_mask(shape, r['poly'])
    m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~void_mask   # run under the walls
    if r.get('outdoor'):
        m &= ~np.any([rl.poly_mask(shape, q['poly']) for q in rooms_in], axis=0)
    if F1:                      # nested rooms (storage under the stair inside the entry): first listed wins
        m &= ~claimed
        claimed |= m
    fl = [rl.box(((x0 + x1) / 2, (y0 + y1) / 2, -0.125 - (0.03 if r.get('outdoor') else 0)), (x1 - x0, y1 - y0, 0.25),
                 0, f'floor_{i}', 'floor') for x0, y0, x1, y1 in rl.mask_rects_m(m, frame)]
    floor_meshes.append((f'R{i:02d}_{r["name"].replace(" ", "_")}', r, fl))
    if not r.get('outdoor'):
        cm = rl.poly_mask(shape, r['poly'])
        if F1:
            cm &= ~cv2.dilate(hole_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        cz = rl.WALL_H - 0.02 if F1 else rl.WALL_H + 0.01        # F1: just below the upstairs slab
        els += [rl.box(((x0 + x1) / 2, (y0 + y1) / 2, cz), (x1 - x0, y1 - y0, 0.02), 0, 'ceiling', 'ceiling',
                       False) for x0, y0, x1, y1 in rl.mask_rects_m(cm, frame)]
indoor = np.any([rl.poly_mask(shape, r['poly']) for r in rooms_in], axis=0)
base = cv2.dilate(wmask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~wmask & indoor & ~void_mask
els += [rl.box(((x0 + x1) / 2, (y0 + y1) / 2, 0.045), (x1 - x0, y1 - y0, 0.09), 0, 'trim', 'wall', False)
        for x0, y0, x1, y1 in rl.mask_rects_m(base, frame)]

# ---------------- stairwell: steps, skirt, foyer floor, railings ----------------
st = plan.get('stair')
if st is not None:
    top_first = st.get('edges_from') == 'top'
    outer = frame.m(st['outer_edge'] if top_first else st['outer_edge'][::-1])     # from the top step down
    inner = frame.m(st['inner_edge'] if top_first else st['inner_edge'][::-1])
    els += rl.stair_elements(outer, inner, st['steps'], st['rise_m'], -rl.FLOOR_TO_FLOOR)
    vpoly = np.asarray(void_m)
    top_a, top_b = frame.m(st['top_step'][0]), frame.m(st['top_step'][1])
    for a, b in zip(vpoly[:-1], vpoly[1:]):
        if np.linalg.norm((a + b) / 2 - (top_a + top_b) / 2) < 0.3:
            continue
        # skirt down to the placeholder foyer floor, or only the slab edge when FLOOR 1 is built below
        e = rl.seg_box(a, b, 0.12, -0.3 if args.with_floor1 else -rl.FLOOR_TO_FLOOR, -0.02, 'wall', 'wall', True)
        if e:
            els.append(e)
    if not args.with_floor1:
        lo, hi = vpoly.min(0) - 1.0, vpoly.max(0) + 1.0
        els.append(rl.box(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, -rl.FLOOR_TO_FLOOR - 0.05), (hi[0] - lo[0], hi[1] - lo[1], 0.1),
                          0, 'foyer_floor', 'floor'))
rail_out = []
for rail in pl.railing_polylines(plan):
    ptsm = frame.m(rail['pts'])
    z0 = -0.03 if rail.get('outdoor') else 0.0
    rail_out.append({'name': rail['name'], 'pts_m': ptsm, 'els': rl.railing_elements(ptsm, z0=z0)})

# ---------------- fixtures ----------------
for f in plan['fixtures']:
    els += rl.fixture_elements(f, frame)

# ---------------- furniture ----------------
catalog = fu.Catalog(args.catalog)
blobs = []
path_map = house_lib.load_poses(r'E:\twin\assets\house\house_poses.txt')[:, 1:3]
path_px = None if F1 else fu.frame_px(frame, fu.apply_inv(T, path_map))     # the recording is upstairs only
if not args.no_grid_furniture and not F1:
    grid, meta = house_lib.load_grid(r'E:\twin\assets\house\house.yaml')
    occ = house_lib.clean_occupied(grid, 4)
    xyz = house_lib.load_ply_xyz(r'E:\twin\assets\house\house_cloud.ply')
    heights = house_lib.cell_heights(occ, meta, xyz, default=0.8)
    blobs = fu.blobs_from_grid(occ, meta, heights, T, frame, plan, wmask, void_px)
items = fu.furnish(plan, frame, wmask, void_px, catalog, rng, blobs, path_px)
for r in plan['rooms']:
    print('ROOM', r['name'], r['type'], [(i['role'], i['source']) for i in items if i.get('room_id') == id(r)])
for it in items:
    if it['role'] == 'bed':
        bx = it.get('back_dir')
        yaw = math.atan2(bx[0], -bx[1]) if bx else it['yaw'] + math.pi / 2
        els += rl.bed_elements(it['c'], tuple(fu.BED_SIZES[it['bed']]), yaw, name='bed')

# ================= USD =================
stage = Usd.Stage.CreateNew(args.out)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, '/Remodel')
stage.SetDefaultPrim(root.GetPrim())
rx = UsdGeom.Xformable(root)
rx.AddTranslateOp().Set(Gf.Vec3d(float(T['t'][0]), float(T['t'][1]), float(FLOOR_Z)))
rx.AddRotateZOp().Set(float(T['angle']))
rx.AddScaleOp().Set(Gf.Vec3f(float(T['scale']), float(T['scale']), 1.0))
UsdGeom.Scope.Define(stage, '/Remodel/Looks')
TEX = os.path.relpath(r'E:\twin\assets\remodel\textures', os.path.dirname(args.out)).replace('\\', '/')


def material(name, color=(0.8, 0.8, 0.8), tex=None, rough=0.8, metal=0.0, opacity=1.0):
    p = f'/Remodel/Looks/{name}'
    mat = UsdShade.Material.Define(stage, p)
    sh = UsdShade.Shader.Define(stage, p + '/Surface')
    sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metal)
    if opacity < 1:
        sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(opacity)
        sh.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.5)
    if tex:
        st_ = UsdShade.Shader.Define(stage, p + '/st')
        st_.CreateIdAttr('UsdPrimvarReader_float2')
        st_.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tx = UsdShade.Shader.Define(stage, p + '/Tex')
        tx.CreateIdAttr('UsdUVTexture')
        tx.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(f'{TEX}/{tex}.png')
        tx.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        tx.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
        tx.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
        tx.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(color[0], color[1], color[2], 1.0))
        tx.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(st_.ConnectableAPI(), 'result')
        tx.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), 'rgb')
    else:
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    return mat


FLOOR_TEX = {'bedroom_primary': 'carpet_beige', 'bedroom': 'carpet_gray', 'hallway': 'carpet_taupe', 'bathroom': 'tile_white',
             'closet': 'carpet_gray', 'balcony': 'stone_gray', 'kitchen': 'tile_beige', 'dining': 'wood_oak',
             'living': 'wood_maple', 'office': 'wood_walnut', 'entry': 'tile_white', 'family': 'carpet_beige',
             'laundry': 'tile_gray', 'garage': 'stone_gray'}
if F1:
    FLOOR_TEX['hallway'] = 'tile_gray'
MATS = {
    'wall': material('wall', (0.93, 0.91, 0.87), 'paint_white', 0.9),
    'ceiling': material('ceiling', (0.97, 0.97, 0.96), 'paint_white', 0.95),
    'trim': material('trim', (0.95, 0.95, 0.93), None, 0.45),
    'glass': material('glass', (0.85, 0.9, 0.95), None, 0.03, 0.0, 0.12),
    'ceramic': material('ceramic', (0.95, 0.95, 0.94), None, 0.15),
    'water': material('water', (0.7, 0.8, 0.85), None, 0.05),
    'cabinet': material('cabinet', (1, 1, 1), 'wood_walnut', 0.5),
    'counter': material('counter', (0.9, 0.9, 0.9), 'stone_gray', 0.25),
    'metal': material('metal', (0.6, 0.6, 0.62), None, 0.3, 1.0),
    'rail_wood': material('rail_wood', (1, 1, 1), 'wood_walnut', 0.45),
    'rail_metal': material('rail_metal', (0.05, 0.05, 0.05), None, 0.45, 0.6),
    'stair_wood': material('stair_wood', (1, 1, 1), 'wood_oak', 0.5),
    'foyer_floor': material('foyer_floor', (1, 1, 1), 'wood_oak', 0.45),
    'bed_frame': material('bed_frame', (1, 1, 1), 'wood_walnut', 0.55),
    'bed_sheet': material('bed_sheet', (0.92, 0.92, 0.9), None, 0.85),
    'duvet': material('duvet', (0.45, 0.52, 0.62), 'carpet_gray', 0.9),
    'pillow': material('pillow', (0.95, 0.95, 0.93), None, 0.9),
    'door': material('door', (0.94, 0.94, 0.92), None, 0.45),
}
for i, r in enumerate(plan['rooms']):
    MATS[f'floor_{i}'] = material(f'floor_{i}', (1, 1, 1), FLOOR_TEX[r['type']], 0.85 if 'carpet' in FLOOR_TEX[r['type']] else 0.35)


def write_mesh(path, elements, mat, label, collide):
    pts, counts, idx, uvs = [], [], [], []
    for e in elements:
        P, faces = rl.element_mesh(e)
        base = len(pts)
        pts.extend(P.tolist())
        for f in faces:
            a, b, c = P[f[0]], P[f[1]], P[f[2]]
            n = np.cross(b - a, c - a)
            counts.append(len(f))
            idx.extend(base + i for i in f)
            uvs.extend(rl.planar_uv(P[i], n) for i in f)
    if not pts:
        return None
    m = UsdGeom.Mesh.Define(stage, path)
    A = np.asarray(pts, np.float32)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(A))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    m.CreateSubdivisionSchemeAttr('none')
    m.CreateExtentAttr([Gf.Vec3f(*A.min(0).tolist()), Gf.Vec3f(*A.max(0).tolist())])
    pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uvs, np.float32)))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(MATS[mat])
    add_semantic_label(m.GetPrim(), label)
    if collide:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('none')
    return m


groups = {}
for e in els:
    groups.setdefault((e['mat'], e['label'], e['collide']), []).append(e)
UsdGeom.Scope.Define(stage, '/Remodel/Geometry')
mesh_count = 0
for (mat, label, col), g in groups.items():
    name = f'{label}_{mat}' + ('' if col else '_vis')
    write_mesh(f'/Remodel/Geometry/{name}', g, mat, label, col)
    mesh_count += 1
# one mesh pair per railing so each railing is its own instance (boxes / instance masks per railing)
UsdGeom.Scope.Define(stage, '/Remodel/Railings')
for r in rail_out:
    for mat in ('rail_wood', 'rail_metal'):
        write_mesh(f'/Remodel/Railings/{r["name"]}_{mat}', [e for e in r['els'] if e['mat'] == mat], mat, 'railing', True)
UsdGeom.Scope.Define(stage, '/Remodel/Floors')
for name, r, fl in floor_meshes:
    write_mesh(f'/Remodel/Floors/{name}', fl, fl[0]['mat'], 'floor', True)

# ---------------- doors ----------------
UsdGeom.Scope.Define(stage, '/Remodel/Doors')
for d in door_specs:
    UsdGeom.Xform.Define(stage, f'/Remodel/Doors/{d["id"]}')
    for k, lf in enumerate(d['leaves']):
        p = f'/Remodel/Doors/{d["id"]}/Leaf{k}'
        x = UsdGeom.Xform.Define(stage, p)
        xf = UsdGeom.Xformable(x)
        xf.AddTranslateOp().Set(Gf.Vec3d(lf['hinge'][0], lf['hinge'][1], 0.0))
        closed = math.degrees(lf['yaw'])
        opened = closed + lf['open_sign'] * lf['open_deg']
        xf.AddRotateZOp().Set(closed if d['closed'] else opened)
        x.GetPrim().CreateAttribute('twin:closed_yaw_deg', Sdf.ValueTypeNames.Double).Set(closed)
        x.GetPrim().CreateAttribute('twin:open_yaw_deg', Sdf.ValueTypeNames.Double).Set(opened)
        leaf = rl.box((lf['width'] / 2, 0.0, lf['height'] / 2), (lf['width'], lf['thick'], lf['height']), 0, 'door', 'door')
        write_mesh(p + '/Slab', [leaf], 'door', 'door', True)

# ---------------- furniture ----------------
UsdGeom.Scope.Define(stage, '/Remodel/Furniture')
fcount = {}
for it in items:
    if it['role'] == 'bed' or not it['asset']:
        continue
    k = fcount.get(it['role'], 0)
    fcount[it['role']] = k + 1
    a = catalog.get(it['asset_key'])
    p = f'/Remodel/Furniture/{it["role"]}_{k:02d}'
    x = UsdGeom.Xform.Define(stage, p)
    xf = UsdGeom.Xformable(x)
    xf.AddTranslateOp().Set(Gf.Vec3d(it['c'][0], it['c'][1], 0.0))
    sx, sy, sz = a['size_m']
    long_y = sy > sx
    bx = it.get('back_dir')
    yaw = math.atan2(-bx[0], bx[1]) if bx else it['yaw']
    xf.AddRotateZOp().Set(math.degrees(yaw) + (90.0 if long_y else 0.0))
    w, dd = it['size']
    fit = min(w / max(sx, sy), dd / max(min(sx, sy), 1e-3))
    fit = float(np.clip(fit, 0.75, 1.25))
    mpu = a['meters_per_unit']
    ast = stage.DefinePrim(p + '/Asset', 'Xform')
    ax = UsdGeom.Xformable(ast)
    cmin, cmax = np.array(a['min']), np.array(a['max'])
    off = -np.array([(cmin[0] + cmax[0]) / 2, (cmin[1] + cmax[1]) / 2, cmin[2]]) * fit
    ax.AddTranslateOp().Set(Gf.Vec3d(*off.tolist()))
    ax.AddScaleOp().Set(Gf.Vec3f(mpu * fit, mpu * fit, mpu * fit))
    ast.GetReferences().AddReference(a['url'])
    col = UsdGeom.Cube.Define(stage, p + '/Collider')
    col.CreateSizeAttr(1.0)
    cx_ = UsdGeom.Xformable(col)
    cx_.AddTranslateOp().Set(Gf.Vec3d(0, 0, sz * fit / 2))
    cx_.AddScaleOp().Set(Gf.Vec3f(sx * fit, sy * fit, sz * fit))
    col.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    add_semantic_label(x.GetPrim(), 'furniture')

# ---------------- lights ----------------
UsdGeom.Scope.Define(stage, '/Remodel/Lights')
nl = 0
for i, r in enumerate(rooms_in):
    m = rl.poly_mask(shape, r['poly']) & ~void_mask
    area = m.sum() / S / S
    ys, xs = np.nonzero(m)
    n = 1 if area < 14 else int(round(area / 10))
    pts = np.stack([xs, ys], 1).astype(np.float32)
    if n > 1:
        _, _, cen = cv2.kmeans(pts, n, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3,
                               cv2.KMEANS_PP_CENTERS)
    else:
        cen = pts.mean(0, keepdims=True)
    for c in cen:
        cm = frame.m(c)
        L = UsdLux.SphereLight.Define(stage, f'/Remodel/Lights/L{nl:02d}_{r["type"]}')
        L.CreateRadiusAttr(0.07)
        L.CreateIntensityAttr(30000.0 if r['type'] != 'closet' else 12000.0)
        L.CreateEnableColorTemperatureAttr(True)
        L.CreateColorTemperatureAttr(3200.0)
        UsdGeom.Xformable(L).AddTranslateOp().Set(Gf.Vec3d(float(cm[0]), float(cm[1]), rl.WALL_H - 0.12))
        nl += 1
ch = None
if not F1:
    wc = frame.m(st['well_center'])
    L = UsdLux.SphereLight.Define(stage, '/Remodel/Lights/Chandelier')
    L.CreateRadiusAttr(0.15)
    L.CreateIntensityAttr(25000.0)
    L.CreateEnableColorTemperatureAttr(True)
    L.CreateColorTemperatureAttr(2900.0)
    UsdGeom.Xformable(L).AddTranslateOp().Set(Gf.Vec3d(float(wc[0]), float(wc[1]), 0.6))
    ch = catalog.get('Lighting/Chandelier_Spiral')
if ch:
    p = '/Remodel/Furniture/chandelier'
    x = UsdGeom.Xform.Define(stage, p)
    UsdGeom.Xformable(x).AddTranslateOp().Set(Gf.Vec3d(float(wc[0]), float(wc[1]), rl.WALL_H - ch['size_m'][2] - 0.25))
    ast = stage.DefinePrim(p + '/Asset', 'Xform')
    UsdGeom.Xformable(ast).AddTranslateOp().Set(Gf.Vec3d(*(-np.array([(ch['min'][0] + ch['max'][0]) / 2, (ch['min'][1] + ch['max'][1]) / 2, ch['min'][2]])).tolist()))
    UsdGeom.Xformable(ast).AddScaleOp().Set(Gf.Vec3f(ch['meters_per_unit']))
    ast.GetReferences().AddReference(ch['url'])
    add_semantic_label(x.GetPrim(), 'furniture')
root.GetPrim().SetCustomDataByKey('twin:source', os.path.basename(args.image) + (
    f', stacked {FLOOR_Z:+.2f} m below the upstairs via the staircase anchor' if F1 else ', registered to the RTAB-Map grid'))
stage.GetRootLayer().Save()

# ================= 2D plan for the Jetson twin (map frame) =================


def to_map(P):
    return pl.apply_T(T, np.asarray(P, float))


def fixture_corners_px(f):
    if 'rect' in f:
        x0, y0, x1, y1 = f['rect']
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    r = 0.35 * S if f['type'] == 'toilet' else 0.9 * S
    c = np.array(f['center'], float)
    return c + np.array([[-r, -r], [r, -r], [r, r], [-r, r]]) / 2


wall_polys = []
for e in els:
    if e['label'] == 'wall' and e['kind'] == 'box' and e['c'][2] > 0 and e['s'][2] > 2.0:
        cx, cy = e['c'][:2]
        sx, sy = e['s'][:2]
        co, si = math.cos(e['yaw']), math.sin(e['yaw'])
        corners = [(cx + co * dx - si * dy, cy + si * dx + co * dy) for dx, dy in
                   ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))]
        wall_polys.append(to_map(corners).round(3).tolist())
fp = {
    'frame': 'map (RTAB-Map), metres, z up; plan registered with rotation/translation/scale below',
    'floor': 'FLOOR 1 (downstairs)' if F1 else 'FLOOR 2 (upstairs, where the robot was mapped)',
    'floor_z_m': FLOOR_Z,
    'registration': {k: best[k] for k in best},
    'scale_fit': reg['scale'],
    'assumptions': {'wall_height_m': rl.WALL_H, 'door_height_m': rl.DOOR_H, 'floor_to_floor_m': rl.FLOOR_TO_FLOOR},
    'stack': plan.get('stack'),
    'rooms': [{'name': r['name'], 'type': r['type'], 'outdoor': r.get('outdoor', False),
               'dims_ft': r.get('dims_ft'), 'polygon': to_map(frame.m(r['poly'])).round(3).tolist()} for r in plan['rooms']],
    'walls': wall_polys,
    'doors': [{'id': d['id'], 'type': d['type'], 'segment': to_map(frame.m([d['a'], d['b']])).round(3).tolist(),
               'swing': d.get('swing'), 'closable': True, 'default': 'closed' if d.get('closed') else 'open'} for d in plan['doors']],
    'windows': [{'segment': to_map(frame.m([w['a'], w['b']])).round(3).tolist(), 'sill_m': w['sill_m'], 'head_m': w['head_m']}
                for w in plan['windows'] + auto_windows],
    'stair_void': ({'polygon': to_map(void_m).round(3).tolist(), 'drop_m': rl.FLOOR_TO_FLOOR, 'steps': st['steps'],
                    'top_step': to_map(frame.m(st['top_step'])).round(3).tolist()} if not F1 else None),
    'double_height_area': to_map(frame.m(hole_px)).round(3).tolist() if F1 else None,
    'railings': [{'name': r['name'], 'polyline': to_map(r['pts_m']).round(3).tolist(), 'height_m': rl.RAIL_H} for r in rail_out],
    'fixtures': [{'type': f['type'], 'room': f['room'], 'footprint': to_map(frame.m(fixture_corners_px(f))).round(3).tolist()}
                 for f in plan['fixtures']],
    'furniture': [{'role': it['role'], 'room': it['room'], 'source': it['source'], 'asset': it.get('asset_key'),
                   'bed': it.get('bed'), 'footprint': to_map(it['footprint_m']).round(3).tolist(),
                   'height_m': it.get('height')} for it in items],
}
os.makedirs(args.plan_out, exist_ok=True)
json.dump(fp, open(os.path.join(args.plan_out, f'floorplan{OUT_TAG}.json'), 'w'), indent=1)

allp = np.vstack([np.array(r['polygon']) for r in fp['rooms']])
mn, mx = allp.min(0) - 0.5, allp.max(0) + 0.5
SC = 60.0


def sv(p):
    p = np.asarray(p)
    return ' '.join(f'{(x - mn[0]) * SC:.1f},{(mx[1] - y) * SC:.1f}' for x, y in p)


svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{(mx[0] - mn[0]) * SC:.0f}" height="{(mx[1] - mn[1]) * SC:.0f}" '
       f'viewBox="0 0 {(mx[0] - mn[0]) * SC:.0f} {(mx[1] - mn[1]) * SC:.0f}">',
       f'<!-- map frame: x = {mn[0]:.3f} + px/{SC}, y = {mx[1]:.3f} - py/{SC} (metres) -->',
       '<rect width="100%" height="100%" fill="white"/>']
COLORS = {'bedroom_primary': '#f3e9d2', 'bedroom': '#e9eef5', 'hallway': '#f2f2f2', 'bathroom': '#dff3f3', 'closet': '#eee7f3',
          'balcony': '#e6efe1', 'kitchen': '#f6efd9', 'dining': '#efe3d3', 'living': '#f1e6da', 'office': '#e3e8ef',
          'entry': '#f4f4f4', 'family': '#efe6dc', 'laundry': '#e4eef0', 'garage': '#e2e2e2'}
for r in fp['rooms']:
    svg.append(f'<polygon points="{sv(r["polygon"])}" fill="{COLORS.get(r["type"], "#eee")}" stroke="#999" stroke-width="1">'
               f'<title>{r["name"]}</title></polygon>')
if fp['stair_void']:
    svg.append(f'<polygon points="{sv(fp["stair_void"]["polygon"])}" fill="#9fd0ff" stroke="#3a7bd5"><title>stair void (drop-off)</title></polygon>')
if fp['double_height_area']:
    svg.append(f'<polygon points="{sv(fp["double_height_area"])}" fill="none" stroke="#3a7bd5" stroke-dasharray="6,4"><title>open to the upstairs</title></polygon>')
for wpoly in wall_polys:
    svg.append(f'<polygon points="{sv(wpoly)}" fill="#222"/>')
for f in fp['furniture']:
    svg.append(f'<polygon points="{sv(f["footprint"])}" fill="#c8a27a" fill-opacity="0.7" stroke="#7a5230"><title>{f["role"]} ({f["source"]})</title></polygon>')
for r in fp['railings']:
    svg.append(f'<polyline points="{sv(r["polyline"])}" fill="none" stroke="#ff8c00" stroke-width="4" stroke-dasharray="3,3"/>')
for d in fp['doors']:
    svg.append(f'<polyline points="{sv(d["segment"])}" stroke="{"#00a" if d["type"] != "glass" else "#0aa"}" stroke-width="5"/>')
for w in fp['windows']:
    svg.append(f'<polyline points="{sv(w["segment"])}" stroke="#4fc3f7" stroke-width="5"/>')
for r in fp['rooms']:
    c = np.mean(r['polygon'], axis=0)
    svg.append(f'<text x="{(c[0] - mn[0]) * SC:.0f}" y="{(mx[1] - c[1]) * SC:.0f}" font-size="14" text-anchor="middle" '
               f'font-family="sans-serif" fill="#333">{r["name"]}</text>')
if not F1:
    svg.append(f'<polyline points="{sv(path_map)}" fill="none" stroke="#0a0" stroke-width="2" opacity="0.6"/>')
svg.append('</svg>')
open(os.path.join(args.plan_out, f'floorplan{OUT_TAG}.svg'), 'w').write('\n'.join(svg))

stats = {'elements': len(els), 'merged_meshes': mesh_count, 'rooms': len(plan['rooms']), 'doors': len(door_specs),
         'windows': len(plan['windows']) + len(auto_windows), 'lights': nl + (0 if F1 else 1), 'grid_blobs': len(blobs),
         'furniture': {'grid': sum(1 for i in items if i['source'] == 'grid'),
                       'template': sum(1 for i in items if i['source'] == 'template'),
                       'by_role': {r: sum(1 for i in items if i['role'] == r) for r in sorted({i['role'] for i in items})}},
         'out': args.out}
json.dump(stats, open(os.path.join(args.plan_out, f'remodel{OUT_TAG}_stats.json'), 'w'), indent=1)
print('REMODEL_STATS ' + json.dumps(stats))
app.close()
