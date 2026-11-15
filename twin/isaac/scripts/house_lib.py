"""Pure numpy helpers for the house twin: RTAB-Map grid -> wall boxes, OBJ parsing, face labels.

No Isaac Sim imports here so the logic can be tested with plain Python.
"""
import os
import re

import numpy as np
from PIL import Image

FREE, UNKNOWN, OCCUPIED = 0, 1, 2


def load_grid(yaml_path):
    """Load a map_server style pgm/yaml pair. Returns (labels[row, col] with row 0 = min y, meta)."""
    meta = {}
    with open(yaml_path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                meta[k.strip()] = v.strip()
    res = float(meta['resolution'])
    origin = [float(x) for x in re.findall(r'-?[\d.]+(?:e-?\d+)?', meta['origin'])][:3]
    img = np.array(Image.open(os.path.join(os.path.dirname(yaml_path), meta['image'])))
    negate = int(meta.get('negate', 0))
    occ_t, free_t = float(meta.get('occupied_thresh', 0.65)), float(meta.get('free_thresh', 0.196))
    p = img.astype(np.float32) / 255.0
    p = p if negate else 1.0 - p
    labels = np.full(img.shape, UNKNOWN, np.uint8)
    labels[p > occ_t] = OCCUPIED
    labels[p < free_t] = FREE
    labels = labels[::-1]          # image row 0 is max y; flip so row index grows with y
    return labels, {'resolution': res, 'origin': origin}


def _components(mask):
    """4-connected components; returns label image and sizes (label 0 = background)."""
    lab = np.zeros(mask.shape, np.int32)
    sizes = [0]
    h, w = mask.shape
    cur = 0
    for r0, c0 in zip(*np.nonzero(mask)):
        if lab[r0, c0]:
            continue
        cur += 1
        stack = [(r0, c0)]
        lab[r0, c0] = cur
        n = 0
        while stack:
            r, c = stack.pop()
            n += 1
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = cur
                    stack.append((rr, cc))
        sizes.append(n)
    return lab, np.array(sizes)


def clean_occupied(labels, min_cells=4):
    """Occupied mask with specks (components smaller than min_cells, 8-neighbour dilated) removed."""
    occ = labels == OCCUPIED
    # join diagonal neighbours so thin diagonal walls count as one component
    grown = occ.copy()
    grown[1:, 1:] |= occ[:-1, :-1]
    grown[:-1, :-1] |= occ[1:, 1:]
    grown[1:, :-1] |= occ[:-1, 1:]
    grown[:-1, 1:] |= occ[1:, :-1]
    lab, sizes = _components(grown)
    keep = sizes >= min_cells
    keep[0] = False
    return occ & keep[lab]


def greedy_rectangles(mask):
    """Cover a boolean mask with axis-aligned rectangles (row0, col0, rows, cols), greedily."""
    m = mask.copy()
    rects = []
    h, w = m.shape
    for r in range(h):
        c = 0
        while c < w:
            if not m[r, c]:
                c += 1
                continue
            c1 = c
            while c1 < w and m[r, c1]:
                c1 += 1
            r1 = r + 1
            while r1 < h and m[r1, c:c1].all():
                r1 += 1
            m[r:r1, c:c1] = False
            rects.append((r, c, r1 - r, c1 - c))
            c = c1
    return rects


def box_mesh(rects, meta, height, z0=0.0, heights=None):
    """Closed boxes (no bottom faces) for grid rectangles -> (points[N,3], counts[M], indices[K]).
    `heights` (optional, per rect) overrides the common `height`."""
    res = meta['resolution']
    ox, oy = meta['origin'][0], meta['origin'][1]
    pts, idx = [], []
    for k, (r, c, nr, nc) in enumerate(rects):
        top = height if heights is None else heights[k]
        x0, x1 = ox + c * res, ox + (c + nc) * res
        y0, y1 = oy + r * res, oy + (r + nr) * res
        b = len(pts)
        pts += [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top)]
        for q in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)):
            idx += [b + i for i in q]
    n_faces = len(idx) // 4
    return np.array(pts, np.float32), np.full(n_faces, 4, np.int32), np.array(idx, np.int32)


def parse_obj(path):
    """Minimal OBJ reader for RTAB-Map exports (v, vt, f v/vt, usemtl, mtllib).

    Returns dict(points, uvs, faces[F,3] vertex idx, face_uv[F,3] or None, face_mat[F], materials
    {name: texture path or None}, vertex_colors or None).
    """
    v, vc, vt, faces, fuv, fmat = [], [], [], [], [], []
    mats, mat_names, cur = {}, [], -1
    base = os.path.dirname(path)
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                v.append((float(p[1]), float(p[2]), float(p[3])))
                if len(p) >= 7:
                    vc.append((float(p[4]), float(p[5]), float(p[6])))
            elif line.startswith('vt '):
                p = line.split()
                vt.append((float(p[1]), float(p[2])))
            elif line.startswith('f '):
                p = line.split()[1:]
                vi = [int(x.split('/')[0]) for x in p]
                ti = [int(x.split('/')[1]) if '/' in x and x.split('/')[1] else 0 for x in p]
                for k in range(1, len(vi) - 1):     # fan-triangulate
                    faces.append((vi[0], vi[k], vi[k + 1]))
                    fuv.append((ti[0], ti[k], ti[k + 1]))
                    fmat.append(cur)
            elif line.startswith('usemtl'):
                name = line.split(None, 1)[1].strip()
                if name not in mat_names:
                    mat_names.append(name)
                cur = mat_names.index(name)
            elif line.startswith('mtllib'):
                mats.update(parse_mtl(os.path.join(base, line.split(None, 1)[1].strip())))
    faces = np.array(faces, np.int64)
    faces = np.where(faces > 0, faces - 1, faces + len(v))
    fuv = np.array(fuv, np.int64)
    has_uv = len(vt) > 0 and (fuv > 0).all()
    return {
        'points': np.array(v, np.float64),
        'uvs': np.array(vt, np.float32) if vt else None,
        'faces': faces,
        'face_uv': (fuv - 1) if has_uv else None,
        'face_mat': np.array(fmat, np.int32),
        'material_names': mat_names,
        'materials': {n: mats.get(n) for n in mat_names},
        'vertex_colors': np.array(vc, np.float32) if len(vc) == len(v) and vc else None,
    }


def parse_mtl(path):
    out, cur = {}, None
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('newmtl'):
                cur = s.split(None, 1)[1]
                out[cur] = None
            elif s.startswith('map_Kd') and cur:
                out[cur] = os.path.join(os.path.dirname(path), s.split(None, 1)[1])
    return out


CLASSES = ['floor', 'wall', 'furniture', 'ceiling']


def classify_faces(points, faces, cell=0.10, wall_min_top=1.40, floor_max_z=0.08):
    """Heuristic semantic label per face of a scan mesh (index into CLASSES).

    floor     upward-facing and within floor_max_z of z=0
    ceiling   downward-facing above 1.8 m
    wall      near-vertical faces in xy columns whose surface reaches above wall_min_top
              (walls go up to the ceiling; tables, sofas and counters stop well below 1.4 m)
    furniture everything else
    """
    tri = points[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    cen = tri.mean(axis=1)
    lo = points[:, :2].min(axis=0)
    ij = np.floor((points[:, :2] - lo) / cell).astype(np.int64)
    shape = ij.max(axis=0) + 1
    top = np.full(shape, -np.inf)
    np.maximum.at(top, (ij[:, 0], ij[:, 1]), points[:, 2])
    # spread the column heights to the 8 neighbours so a wall's foot sees its own top
    padded = np.pad(top, 1, constant_values=-np.inf)
    top3 = np.max([padded[1 + di:shape[0] + 1 + di, 1 + dj:shape[1] + 1 + dj]
                   for di in (-1, 0, 1) for dj in (-1, 0, 1)], axis=0)
    cij = np.clip(np.floor((cen[:, :2] - lo) / cell).astype(np.int64), 0, shape - 1)
    col_top = top3[cij[:, 0], cij[:, 1]]
    lab = np.full(len(faces), CLASSES.index('furniture'), np.int32)
    vertical = np.abs(n[:, 2]) < 0.6
    lab[vertical & (col_top > wall_min_top)] = CLASSES.index('wall')
    lab[(cen[:, 2] > wall_min_top) & ~vertical & (n[:, 2] > -0.5)] = CLASSES.index('wall')
    lab[(n[:, 2] < -0.5) & (cen[:, 2] > 1.8)] = CLASSES.index('ceiling')
    lab[(np.abs(n[:, 2]) > 0.7) & (cen[:, 2] < floor_max_z)] = CLASSES.index('floor')
    return lab


def load_ply_xyz(path):
    """x, y, z of a binary little-endian PCL PLY (as written by rtabmap-export --cloud)."""
    with open(path, 'rb') as f:
        header = b''
        while not header.endswith(b'end_header\n'):
            header += f.readline()
        lines = header.decode('ascii').splitlines()
        n = int(next(l for l in lines if l.startswith('element vertex')).split()[-1])
        props = []
        in_vertex = False
        for l in lines:
            if l.startswith('element'):
                in_vertex = l.startswith('element vertex')
            elif l.startswith('property') and in_vertex:
                _, t, name = l.split()
                props.append((name, {'float': '<f4', 'uchar': 'u1', 'double': '<f8', 'int': '<i4'}[t]))
        data = np.frombuffer(f.read(n * np.dtype(props).itemsize), dtype=np.dtype(props), count=n)
    return np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)


def cell_heights(occ, meta, xyz, default=1.0, nbhd=1):
    """Max cloud z above each occupied cell (3x3 neighbourhood); `default` where no points landed."""
    res = meta['resolution']
    ox, oy = meta['origin'][0], meta['origin'][1]
    h, w = occ.shape
    c = np.floor((xyz[:, 0] - ox) / res).astype(np.int64)
    r = np.floor((xyz[:, 1] - oy) / res).astype(np.int64)
    ok = (c >= 0) & (c < w) & (r >= 0) & (r < h) & (xyz[:, 2] > 0.03) & (xyz[:, 2] < 3.0)
    top = np.full((h, w), -1.0, np.float32)
    np.maximum.at(top, (r[ok], c[ok]), xyz[ok, 2])
    pad = np.pad(top, nbhd, constant_values=-1.0)
    top_n = np.max([pad[nbhd + dr:nbhd + dr + h, nbhd + dc:nbhd + dc + w]
                    for dr in range(-nbhd, nbhd + 1) for dc in range(-nbhd, nbhd + 1)], axis=0)
    out = np.where(top_n > 0, top_n, default).astype(np.float32)
    return np.where(occ, out, 0.0)


def free_space_samples(labels, meta, path_xy, robot_radius=0.20, max_dist_from_path=1.0):
    """Cell centres where the robot fits: free, >= robot_radius from occupied/unknown-heavy cells,
    and within max_dist_from_path of the recorded path (keeps samples inside the house)."""
    res = meta['resolution']
    ox, oy = meta['origin'][0], meta['origin'][1]
    occ = labels == OCCUPIED
    rad = int(np.ceil(robot_radius / res))
    blocked = np.zeros_like(occ)
    rr, cc = np.nonzero(occ)
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            if dr * dr + dc * dc <= rad * rad:
                r2 = np.clip(rr + dr, 0, occ.shape[0] - 1)
                c2 = np.clip(cc + dc, 0, occ.shape[1] - 1)
                blocked[r2, c2] = True
    ok = (labels == FREE) & ~blocked
    rows, cols = np.nonzero(ok)
    xy = np.stack([ox + (cols + 0.5) * res, oy + (rows + 0.5) * res], axis=1)
    if path_xy is not None and len(path_xy):
        from math import ceil
        near = np.zeros(len(xy), bool)
        for i in range(0, len(xy), 4096):
            d = np.linalg.norm(xy[i:i + 4096, None, :] - path_xy[None, :, :], axis=2).min(axis=1)
            near[i:i + 4096] = d <= max_dist_from_path
        xy = xy[near]
    return xy


def load_poses(path):
    """RTAB-Map --poses_format 11: stamp x y z qx qy qz qw id -> array [N, 9]."""
    rows = [list(map(float, ln.split())) for ln in open(path) if ln.strip() and not ln.startswith('#')]
    return np.array(rows)
