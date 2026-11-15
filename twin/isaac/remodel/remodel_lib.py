"""Plan -> 3D house elements (numpy only). Everything is in PLAN metres (x right, y up, z up);
build_remodel.py places the whole plan in RTAB-Map's map frame with the registration transform.

An element is a dict: {'kind': 'box', 'c': (x, y, z), 's': (sx, sy, sz), 'yaw': rad, 'mat': key,
'label': semantic class, 'collide': bool} or {'kind': 'prism', 'poly': [(x, y), ...], 'z0', 'z1', ...}.
"""
import math

import cv2
import numpy as np

WALL_H = 2.44            # 8 ft ceilings (listing plans show standard heights; assumption)
DOOR_H = 2.03            # 6'8" doors
FLOOR_TO_FLOOR = 2.90    # upstairs floor to the foyer floor below (assumption, see README)
RAIL_H = 0.92
BALUSTER_SPACING = 0.115
TEX_TILE_M = 1.2


def box(c, s, yaw=0.0, mat='wall', label='wall', collide=True, name=None):
    return {'kind': 'box', 'c': tuple(float(v) for v in c), 's': tuple(float(v) for v in s), 'yaw': float(yaw),
            'mat': mat, 'label': label, 'collide': collide, 'name': name}


def prism(poly, z0, z1, mat, label, collide=True, name=None):
    return {'kind': 'prism', 'poly': [tuple(map(float, p)) for p in poly], 'z0': float(z0), 'z1': float(z1),
            'mat': mat, 'label': label, 'collide': collide, 'name': name}


def seg_box(a, b, t, z0, z1, mat, label, collide=True, extend=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-4:
        return None
    m = (a + b) / 2
    return box((m[0], m[1], (z0 + z1) / 2), (L + 2 * extend, t, z1 - z0), math.atan2(d[1], d[0]), mat, label, collide)


# ---------------------------------------------------------------- mesh conversion

def element_mesh(e):
    """-> (points[N,3], faces[list of index lists], uvs per face-vertex in metres (world planar))"""
    if e['kind'] == 'box':
        cx, cy, cz = e['c']
        sx, sy, sz = e['s']
        c, s = math.cos(e['yaw']), math.sin(e['yaw'])
        pts = []
        for dz in (-sz / 2, sz / 2):
            for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)):
                pts.append((cx + c * dx - s * dy, cy + s * dx + c * dy, cz + dz))
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        return np.array(pts), faces
    poly = np.asarray(e['poly'], float)
    if signed_area(poly) < 0:
        poly = poly[::-1]
    n = len(poly)
    pts = [(x, y, e['z0']) for x, y in poly] + [(x, y, e['z1']) for x, y in poly]
    tri = ear_clip(poly)
    faces = [(t[2], t[1], t[0]) for t in tri] + [(n + t[0], n + t[1], n + t[2]) for t in tri]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    return np.array(pts), faces


def planar_uv(p, normal, tile=TEX_TILE_M):
    ax = int(np.argmax(np.abs(normal)))
    if ax == 2:
        return (p[0] / tile, p[1] / tile)
    if ax == 0:
        return (p[1] / tile, p[2] / tile)
    return (p[0] / tile, p[2] / tile)


def signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ear_clip(poly):
    """Triangulate a simple CCW polygon -> list of index triples."""
    idx = list(range(len(poly)))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[(k - 1) % n], idx[k], idx[(k + 1) % n]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 1e-12:
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if point_in_tri(poly[j], a, b, c):
                    inside = True
                    break
            if not inside:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def point_in_tri(p, a, b, c):
    def s(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


# ---------------------------------------------------------------- raster helpers (plan pixels)

def poly_mask(shape, poly_px):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly_px) * 8).astype(np.int32)], 1, shift=3)
    return m.astype(bool)


def mask_rects_m(mask, frame):
    """Greedy rectangles of a pixel mask -> [(x0, y0, x1, y1)] in plan metres."""
    from house_lib import greedy_rectangles
    out = []
    for r, c, nr, nc in greedy_rectangles(mask):
        a = frame.m([c, r + nr])
        b = frame.m([c + nc, r])
        out.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
    return out


# ---------------------------------------------------------------- stairs

def resample(poly, n):
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, poly[:, 0]), np.interp(t, cum, poly[:, 1])], 1)


def stair_elements(outer_m, inner_m, steps, rise, bottom_z):
    """Curved stair as solid step prisms. outer/inner go from the top step (landing) to the far end."""
    o = resample(outer_m, steps + 1)
    i = resample(inner_m, steps + 1)
    els = []
    for k in range(steps):
        poly = [o[k], o[k + 1], i[k + 1], i[k]]
        z_top = -(k + 1) * rise
        els.append(prism(poly, bottom_z, z_top, 'stair_wood', 'stairs', True, name=f'step_{k:02d}'))
    return els


# ---------------------------------------------------------------- railings

def railing_elements(pts_m, z0=0.0, spacing=BALUSTER_SPACING, post=0.022, height=RAIL_H, mat='rail_wood',
                     baluster_mat='rail_metal', newels=True):
    pts = np.asarray(pts_m, float)
    els = []
    for a, b in zip(pts[:-1], pts[1:]):
        e = seg_box(a, b, 0.065, z0 + height - 0.02, z0 + height + 0.035, mat, 'railing')
        if e:
            els.append(e)
        e = seg_box(a, b, 0.04, z0, z0 + 0.035, mat, 'railing')
        if e:
            els.append(e)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    for d in np.arange(spacing / 2, L[-1], spacing):
        k = min(int(np.searchsorted(L, d, side='right')) - 1, len(pts) - 2)
        f = (d - L[k]) / max(L[k + 1] - L[k], 1e-9)
        p = pts[k] + (pts[k + 1] - pts[k]) * f
        yaw = math.atan2(*(pts[k + 1] - pts[k])[::-1])
        els.append(box((p[0], p[1], z0 + height / 2), (post, post, height), yaw, baluster_mat, 'railing'))
    if newels:
        for p in (pts[0], pts[-1]):
            els.append(box((p[0], p[1], z0 + 0.52), (0.09, 0.09, 1.04), 0.0, mat, 'railing'))
    return els


# ---------------------------------------------------------------- openings

def window_elements(a, b, t, sill, head, wall_h=WALL_H):
    els = []
    for z0, z1 in ((0.0, sill), (head, wall_h)):
        if z1 - z0 > 1e-3:
            e = seg_box(a, b, t, z0, z1, 'wall', 'wall')
            if e:
                els.append(e)
    g = seg_box(a, b, 0.012, sill, head, 'glass', 'window', True)
    if g:
        els.append(g)
    # frame: sill board, head, two jambs
    d = np.asarray(b, float) - np.asarray(a, float)
    u = d / max(np.linalg.norm(d), 1e-9)
    for z in (sill, head):
        e = seg_box(a, b, t + 0.03, z - 0.025, z + 0.025, 'trim', 'window')
        if e:
            els.append(e)
    for p in (np.asarray(a) + u * 0.025, np.asarray(b) - u * 0.025):
        els.append(box((p[0], p[1], (sill + head) / 2), (0.05, t + 0.03, head - sill), math.atan2(d[1], d[0]),
                       'trim', 'window'))
    return els


def door_frame_elements(a, b, t, head=DOOR_H, wall_h=WALL_H):
    els = []
    e = seg_box(a, b, t, head, wall_h, 'wall', 'wall')
    if e:
        els.append(e)
    d = np.asarray(b, float) - np.asarray(a, float)
    u = d / max(np.linalg.norm(d), 1e-9)
    yaw = math.atan2(d[1], d[0])
    for p in (np.asarray(a) + u * 0.035, np.asarray(b) - u * 0.035):
        els.append(box((p[0], p[1], head / 2), (0.07, t + 0.03, head), yaw, 'trim', 'door', True))
    e = seg_box(a, b, t + 0.03, head, head + 0.07, 'trim', 'door', True, extend=0.035)
    if e:
        els.append(e)
    return els


def door_leaves(a, b, kind, swing_normal, open_deg=88.0, thick=0.04, head=DOOR_H):
    """Leaf specs (hinge point, closed yaw, width, open angle sign) for USD door prims."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    yaw = math.atan2(d[1], d[0])
    leaves = []
    if kind == 'double':
        halves = [(a, yaw, L / 2 - 0.04, 1), (b, yaw + math.pi, L / 2 - 0.04, -1)]
    else:
        halves = [(a, yaw, L - 0.08, 1)]
    for hinge, y, w, sgn in halves:
        u = np.array([math.cos(y), math.sin(y)])
        h = hinge + u * 0.04
        # rotate towards the swing side: sign of the rotation that moves u towards swing_normal
        cross = u[0] * swing_normal[1] - u[1] * swing_normal[0]
        leaves.append({'hinge': h.tolist(), 'yaw': y, 'width': w, 'open_sign': 1.0 if cross > 0 else -1.0,
                       'open_deg': open_deg, 'thick': thick, 'height': head - 0.01})
    return leaves


# ---------------------------------------------------------------- fixtures (procedural)

def fixture_elements(f, frame):
    t = f['type']
    els = []
    if 'rect' in f:
        x0, y0, x1, y1 = f['rect']
        p0, p1 = frame.m([x0, y1]), frame.m([x1, y0])
        cx, cy = (p0 + p1) / 2
        sx, sy = abs(p1[0] - p0[0]), abs(p1[1] - p0[1])
    else:
        cx, cy = frame.m(f['center'])
    if t == 'tub':
        els.append(box((cx, cy, 0.28), (sx, sy, 0.56), 0, 'ceramic', 'furniture'))
        els.append(box((cx, cy, 0.565), (sx - 0.12, sy - 0.12, 0.01), 0, 'water', 'furniture', False))
    elif t == 'tub_freestanding':
        a, b = f['size_m'][0] / 2, f['size_m'][1] / 2
        yaw = math.radians(-f.get('yaw_deg', 0))
        ang = np.linspace(0, 2 * np.pi, 28, endpoint=False)
        pts = [(cx + a * math.cos(q) * math.cos(yaw) - b * math.sin(q) * math.sin(yaw),
                cy + a * math.cos(q) * math.sin(yaw) + b * math.sin(q) * math.cos(yaw)) for q in ang]
        els.append(prism(pts, 0.0, 0.6, 'ceramic', 'furniture'))
    elif t == 'shower':
        els.append(box((cx, cy, 0.03), (sx, sy, 0.06), 0, 'ceramic', 'furniture'))
        els.append(box((cx, cy - sy / 2 + 0.005, 1.0), (sx, 0.01, 1.9), 0, 'glass', 'furniture'))
    elif t in ('vanity', 'vanity_double'):
        long_x = sx >= sy
        els.append(box((cx, cy, 0.42), (sx, sy, 0.84), 0, 'cabinet', 'furniture'))
        els.append(box((cx, cy, 0.855), (sx + 0.02, sy + 0.02, 0.03), 0, 'counter', 'furniture'))
        n = 2 if t == 'vanity_double' else 1
        for k in range(n):
            f_ = (k + 0.5) / n
            px = cx - sx / 2 + sx * f_ if long_x else cx
            py = cy - sy / 2 + sy * f_ if not long_x else cy
            els.append(box((px, py, 0.875), (0.42 if long_x else 0.34, 0.34 if long_x else 0.42, 0.012), 0, 'ceramic',
                           'furniture', False))
    elif t == 'toilet':
        yaw = math.radians(-f.get('yaw_deg', 0))
        c, s = math.cos(yaw), math.sin(yaw)
        els.append(box((cx, cy, 0.2), (0.38, 0.52, 0.4), yaw, 'ceramic', 'furniture'))
        # tank behind the bowl: local (0, -0.22) rotated by yaw
        els.append(box((cx + s * 0.22, cy - c * 0.22, 0.6), (0.48, 0.2, 0.4), yaw, 'ceramic', 'furniture'))
    elif t == 'closet_shelf':
        els.append(box((cx, cy, 1.72), (sx, sy, 0.025), 0, 'trim', 'furniture', False))
        els.append(box((cx, cy, 1.62), (max(sx - 0.1, 0.02) if sx > sy else 0.025,
                                         max(sy - 0.1, 0.02) if sy >= sx else 0.025, 0.025), 0, 'metal', 'furniture', False))
    return els


# ---------------------------------------------------------------- procedural bed

def bed_elements(c, size, yaw, mat_frame='bed_frame', mat_sheet='bed_sheet', name='bed'):
    """Bed with frame, mattress, duvet, pillows and headboard; c = footprint centre (x, y)."""
    W, L = size            # W across, L along (headboard at -L/2 along the local y axis)
    co, si = math.cos(yaw), math.sin(yaw)

    def at(dx, dy):
        return (c[0] + co * dx - si * dy, c[1] + si * dx + co * dy)
    els = [box((*at(0, 0), 0.17), (W, L, 0.34), yaw, mat_frame, 'furniture', True, name),
           box((*at(0, 0.02), 0.44), (W - 0.06, L - 0.08, 0.2), yaw, mat_sheet, 'furniture', True, name),
           box((*at(0, 0.18), 0.56), (W - 0.02, L - 0.4, 0.05), yaw, 'duvet', 'furniture', False, name),
           box((*at(0, -L / 2 + 0.03), 0.6), (W + 0.04, 0.06, 1.2), yaw, mat_frame, 'furniture', True, name)]
    n = 2 if W > 1.2 else 1
    for k in range(n):
        dx = -W / 2 + W * (k + 0.5) / n
        els.append(box((*at(dx, -L / 2 + 0.3), 0.6), (W / n - 0.12, 0.38, 0.12), yaw, 'pillow', 'furniture', False, name))
    return els
