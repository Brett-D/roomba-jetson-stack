"""Build the house USD from the RTAB-Map exports (map frame, z up, metres).

  python build_house.py --house-dir E:\\twin\\assets\\house --out E:\\twin\\assets\\usd\\house.usd

Default prim /House, with a variant set "look" = scan | clean:
  /House/Scan/<class>_<material>   textured scan mesh (rtabmap-export --mesh --texture) split by a
                                   heuristic class (wall / furniture / ceiling) for semantic labels;
                                   its floor faces are dropped (bumpy) in favour of /House/Floor.
                                   Visible in look=scan. Visual only.
  /House/Blocks/wall|furniture     occupancy-grid cells extruded to the height of the scan points
                                   above them (cells whose points reach > --wall-min-top become
                                   full-height walls). Colliders in both looks; visible in look=clean
                                   (a clean, re-materialisable stand-in for domain randomisation).
  /House/Floor                     flat floor at z=0 covering the grid; collider; label floor
  /House/Ceiling                   flat ceiling at --ceiling-height; visual; label ceiling
Grid, cloud and mesh share RTAB-Map's map frame, so robot poses from house_poses.txt and the 2D
twin on the Jetson line up with this stage.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import house_lib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--house-dir', default=r'E:\twin\assets\house')
ap.add_argument('--mesh', default='house_mesh.obj')
ap.add_argument('--cloud', default='house_cloud.ply')
ap.add_argument('--grid', default='house.yaml')
ap.add_argument('--out', default=r'E:\twin\assets\usd\house.usd')
ap.add_argument('--ceiling-height', type=float, default=2.5)
ap.add_argument('--wall-min-top', type=float, default=1.3,
                help='cells whose scan points reach this high are treated as walls (to the ceiling)')
ap.add_argument('--min-wall-cells', type=int, default=4)
ap.add_argument('--look', default='scan', choices=['scan', 'clean'])
ap.add_argument('--max-edge', type=float, default=0.16, help='drop scan triangles with a longer edge (m)')
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

from twin_common import add_semantic_label  # noqa: E402

t0 = time.time()
out_dir = os.path.dirname(os.path.abspath(args.out))
tex_dir = os.path.join(out_dir, 'house_textures')
os.makedirs(tex_dir, exist_ok=True)

stage = Usd.Stage.CreateNew(args.out)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, '/House')
stage.SetDefaultPrim(root.GetPrim())
UsdGeom.Scope.Define(stage, '/House/Looks')


def preview_material(name, texture=None, color=(0.6, 0.6, 0.6), roughness=0.85):
    mat = UsdShade.Material.Define(stage, f'/House/Looks/{name}')
    sh = UsdShade.Shader.Define(stage, f'/House/Looks/{name}/Surface')
    sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    if texture:
        st = UsdShade.Shader.Define(stage, f'/House/Looks/{name}/st')
        st.CreateIdAttr('UsdPrimvarReader_float2')
        st.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tx = UsdShade.Shader.Define(stage, f'/House/Looks/{name}/Tex')
        tx.CreateIdAttr('UsdUVTexture')
        tx.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture)
        tx.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        tx.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('clamp')
        tx.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('clamp')
        tx.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), 'result')
        tx.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), 'rgb')
    else:
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    return mat


def make_collider(prim):
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')


def define_mesh(path, pts, counts, idx):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(counts, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(idx, np.int32)))
    m.CreateSubdivisionSchemeAttr('none')
    m.CreateExtentAttr([Gf.Vec3f(*np.min(pts, 0).tolist()), Gf.Vec3f(*np.max(pts, 0).tolist())])
    return m


def quad(path, x0, y0, x1, y1, z, up=True):
    pts = np.array([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], np.float32)
    m = define_mesh(path, pts, [4], [0, 1, 2, 3] if up else [0, 3, 2, 1])
    m.CreateNormalsAttr([Gf.Vec3f(0, 0, 1 if up else -1)] * 4)
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(x0, y0), Gf.Vec2f(x1, y0), Gf.Vec2f(x1, y1), Gf.Vec2f(x0, y1)])   # 1 UV unit = 1 m
    return m


stats = {}
grid, meta = house_lib.load_grid(os.path.join(args.house_dir, args.grid))
res = meta['resolution']
ox, oy = meta['origin'][0], meta['origin'][1]
gh, gw = grid.shape
x0, y0, x1, y1 = ox, oy, ox + gw * res, oy + gh * res

# ---------------- scan mesh (look=scan) ----------------
obj_path = os.path.join(args.house_dir, args.mesh)
scan = UsdGeom.Scope.Define(stage, '/House/Scan')
if os.path.exists(obj_path):
    m = house_lib.parse_obj(obj_path)
    pts, faces = m['points'], m['faces']
    labels = house_lib.classify_faces(pts, faces, wall_min_top=args.wall_min_top)
    # stretched triangles bridging scan gaps (edge > --max-edge) read as flying shards: drop them
    tri = pts[faces]
    edge = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
    labels[edge > args.max_edge] = -1
    mats = {}
    for mi, name in enumerate(m['material_names']):
        tex = m['materials'].get(name)
        rel = None
        if tex and os.path.exists(tex):
            shutil.copyfile(tex, os.path.join(tex_dir, os.path.basename(tex)))
            rel = './house_textures/' + os.path.basename(tex)
        mats[mi] = preview_material(f'scan_{mi}', rel)
    untextured = preview_material('scan_untextured', None, (0.5, 0.5, 0.5))
    counts = {}
    for ci, cname in enumerate(house_lib.CLASSES):
        if cname == 'floor':
            counts['floor_dropped'] = int((labels == ci).sum())
            continue
        for mi in sorted(set(m['face_mat'].tolist())):
            sel = np.nonzero((labels == ci) & (m['face_mat'] == mi))[0]
            if len(sel) == 0:
                continue
            f = faces[sel]
            used, inv = np.unique(f.reshape(-1), return_inverse=True)
            mesh = define_mesh(f'/House/Scan/{cname}_{max(mi, 0)}', pts[used], np.full(len(f), 3), inv)
            mesh.CreateDoubleSidedAttr(True)
            if m['face_uv'] is not None and mi >= 0:
                st = m['uvs'][m['face_uv'][sel].reshape(-1)]
                pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                                             UsdGeom.Tokens.faceVarying)
                pv.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(st, np.float32)))
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats[mi])
            else:
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(untextured)
            add_semantic_label(mesh.GetPrim(), cname)
            counts[cname] = counts.get(cname, 0) + len(sel)
    counts['dropped_long_edge'] = int((labels == -1).sum())
    stats['scan'] = {'faces': int(len(faces)), 'vertices': int(len(pts)), 'per_class': counts,
                     'textures': len(m['material_names']),
                     'bounds_min': pts.min(0).round(3).tolist(), 'bounds_max': pts.max(0).round(3).tolist()}
else:
    print(f'WARNING: no scan mesh at {obj_path}')

# ---------------- height-aware blocks from the grid (colliders; look=clean) ----------------
occ = house_lib.clean_occupied(grid, args.min_wall_cells)
cloud_path = os.path.join(args.house_dir, args.cloud)
if os.path.exists(cloud_path):
    xyz = house_lib.load_ply_xyz(cloud_path)
    heights = house_lib.cell_heights(occ, meta, xyz, default=1.0)
else:
    heights = np.where(occ, args.ceiling_height, 0.0)
wall_cells = occ & (heights >= args.wall_min_top)
furn_cells = occ & ~wall_cells
UsdGeom.Scope.Define(stage, '/House/Blocks')
blocks = {}
for name, cells in (('wall', wall_cells), ('furniture', furn_cells)):
    if name == 'wall':
        rects = house_lib.greedy_rectangles(cells)
        hs = [args.ceiling_height] * len(rects)
    else:
        # quantise furniture heights to 10 cm so neighbouring cells merge into boxes
        q = np.round(np.clip(heights, 0.2, args.wall_min_top) * 10) / 10
        rects, hs = [], []
        for hv in np.unique(q[cells]):
            rr = house_lib.greedy_rectangles(cells & (q == hv))
            rects += rr
            hs += [float(hv)] * len(rr)
    if not rects:
        continue
    bp, bc, bi = house_lib.box_mesh(rects, meta, args.ceiling_height, z0=0.0, heights=hs)
    mesh = define_mesh(f'/House/Blocks/{name}', bp, bc, bi)
    color = (0.82, 0.80, 0.76) if name == 'wall' else (0.45, 0.33, 0.25)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(preview_material(f'blocks_{name}', None, color, 0.8))
    make_collider(mesh.GetPrim())
    add_semantic_label(mesh.GetPrim(), name)
    blocks[name] = {'cells': int(cells.sum()), 'boxes': len(rects)}

# ---------------- floor + ceiling ----------------
floor = quad('/House/Floor', x0, y0, x1, y1, 0.0, up=True)
UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(preview_material('floor', None, (0.36, 0.33, 0.29), 0.9))
make_collider(floor.GetPrim())
add_semantic_label(floor.GetPrim(), 'floor')
fmat = UsdShade.Material.Define(stage, '/House/Looks/floor_physics')
pm = UsdPhysics.MaterialAPI.Apply(fmat.GetPrim())
pm.CreateStaticFrictionAttr(0.9)
pm.CreateDynamicFrictionAttr(0.8)
pm.CreateRestitutionAttr(0.0)
UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(fmat, UsdShade.Tokens.weakerThanDescendants, 'physics')
ceil = quad('/House/Ceiling', x0, y0, x1, y1, args.ceiling_height, up=False)
UsdShade.MaterialBindingAPI.Apply(ceil.GetPrim()).Bind(preview_material('ceiling', None, (0.9, 0.9, 0.88), 0.9))
add_semantic_label(ceil.GetPrim(), 'ceiling')

# ---------------- look variants ----------------
vset = root.GetPrim().GetVariantSets().AddVariantSet('look')
for look in ('scan', 'clean'):
    vset.AddVariant(look)
    vset.SetVariantSelection(look)
    with vset.GetVariantEditContext():
        vis_scan = UsdGeom.Tokens.inherited if look == 'scan' else UsdGeom.Tokens.invisible
        vis_blocks = UsdGeom.Tokens.invisible if look == 'scan' else UsdGeom.Tokens.inherited
        UsdGeom.Imageable(stage.GetPrimAtPath('/House/Scan')).CreateVisibilityAttr().Set(vis_scan)
        for name in blocks:
            UsdGeom.Imageable(stage.GetPrimAtPath(f'/House/Blocks/{name}')).CreateVisibilityAttr().Set(vis_blocks)
vset.SetVariantSelection(args.look)

root.GetPrim().SetCustomDataByKey('twin:grid_origin', Gf.Vec3d(*meta['origin']))
root.GetPrim().SetCustomDataByKey('twin:grid_resolution', res)
root.GetPrim().SetCustomDataByKey('twin:ceiling_height', args.ceiling_height)
root.GetPrim().SetCustomDataByKey('twin:frame', 'rtabmap map')
stage.GetRootLayer().Save()

stats.update({'blocks': blocks, 'occupied_cells': int((grid == house_lib.OCCUPIED).sum()),
              'kept_cells': int(occ.sum()), 'ceiling_height_m': args.ceiling_height,
              'grid': {'shape': list(grid.shape), 'resolution': res, 'origin': meta['origin']},
              'seconds': round(time.time() - t0, 1), 'out': args.out})
with open(os.path.splitext(args.out)[0] + '_stats.json', 'w') as f:
    json.dump(stats, f, indent=1)
print('HOUSE_STATS ' + json.dumps(stats))
app.close()
