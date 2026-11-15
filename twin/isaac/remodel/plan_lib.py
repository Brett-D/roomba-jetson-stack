"""Vector floor plan: load the hand-digitised plan (pixels), fit the scale from the room dimensions,
convert to metres, register to the RTAB-Map occupancy grid and export JSON / SVG.

Pure numpy / OpenCV / SciPy (no Isaac Sim) so it can run anywhere, including the Jetson.
"""
import json
import math
import os

import cv2
import numpy as np
from scipy import ndimage

from plan_extract import load_gray, wall_mask

FT = 0.3048
IN = 0.0254


def ft_in(v):
    return v[0] * FT + v[1] * IN


# ---------------------------------------------------------------- scale

def fit_scale(plan):
    """px per metre from rooms with known interior dims; tries both W x D assignments per room."""
    per_room, all_s = [], []
    for r in plan['rooms']:
        if 'inner' not in r or 'dims_ft' not in r:
            continue
        x0, y0, x1, y1 = r['inner']
        wpx, hpx = x1 - x0, y1 - y0
        a, b = ft_in(r['dims_ft'][0]), ft_in(r['dims_ft'][1])
        c1 = (wpx / a, hpx / b)
        c2 = (wpx / b, hpx / a)
        s = c1 if abs(c1[0] - c1[1]) <= abs(c2[0] - c2[1]) else c2
        per_room.append({'room': r['name'], 'dims_m': [round(a, 3), round(b, 3)], 'px': [wpx, hpx],
                         'px_per_m': [round(s[0], 2), round(s[1], 2)]})
        all_s += list(s)
    all_s = np.array(all_s)
    return float(np.median(all_s)), {'rooms': per_room, 'median': round(float(np.median(all_s)), 3),
                                     'std': round(float(all_s.std()), 3),
                                     'rel_spread_pct': round(float(all_s.std() / np.median(all_s) * 100), 2),
                                     'min': round(float(all_s.min()), 2), 'max': round(float(all_s.max()), 2)}


class PlanFrame:
    """Pixel -> plan metres (x right, y up), origin at the image centre."""

    def __init__(self, px_per_m, w, h):
        self.s, self.cx, self.cy = px_per_m, w / 2.0, h / 2.0

    def m(self, p):
        p = np.asarray(p, float)
        return np.stack([(p[..., 0] - self.cx) / self.s, (self.cy - p[..., 1]) / self.s], axis=-1)

    def len(self, px):
        return px / self.s


# ---------------------------------------------------------------- walls

def wall_boxes(mask, frame, openings_px=()):
    """Greedy axis-aligned boxes covering the wall mask, in plan metres: [(x0, y0, x1, y1)]."""
    from house_lib import greedy_rectangles
    m = mask.copy()
    rects = greedy_rectangles(m)            # (row0, col0, rows, cols) in pixels
    out = []
    for r, c, nr, nc in rects:
        a = frame.m([c, r + nr])
        b = frame.m([c + nc, r])
        out.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
    return out


def plan_points(plan, mask, frame, step_px=1.0):
    """Points (plan metres) on walls, railings and fixtures: what a depth camera would see."""
    ys, xs = np.nonzero(mask)
    pts = [frame.m(np.stack([xs, ys], 1).astype(float))]
    for rl in railing_polylines(plan):
        if rl.get('outdoor'):
            continue
        pts.append(frame.m(densify(rl['pts'], step_px)))
    return np.vstack(pts)


def densify(pts, step):
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.array(out)


def arc_points(center, radius, a_from, a_to, clockwise, step_deg=3.0):
    """Arc in image pixel coordinates (y down). Angles in degrees, image convention."""
    if clockwise:          # clockwise on screen = increasing angle with y down
        while a_to <= a_from:
            a_to += 360
        angs = np.arange(a_from, a_to + 1e-6, step_deg)
    else:
        while a_to >= a_from:
            a_to -= 360
        angs = np.arange(a_from, a_to - 1e-6, -step_deg)
    t = np.radians(angs)
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], 1)


def railing_polylines(plan):
    out = []
    for rl in plan.get('railings', []):
        if 'pts' in rl:
            out.append({'name': rl['name'], 'pts': rl['pts'], 'outdoor': rl.get('outdoor', False)})
        else:
            a = rl['arc']
            c = a['center']
            af = math.degrees(math.atan2(a['from_pt'][1] - c[1], a['from_pt'][0] - c[0]))
            at = math.degrees(math.atan2(a['to_pt'][1] - c[1], a['to_pt'][0] - c[0]))
            pts = arc_points(c, a['radius'], af, at, a.get('clockwise', True)).tolist() + list(rl.get('then', []))
            out.append({'name': rl['name'], 'pts': pts, 'outdoor': rl.get('outdoor', False)})
    return out


def stair_void_polygon(plan):
    """Drop-off polygon (pixels): explicit stair['void_poly'], else outer stair edge + the well circle the long way round."""
    st = plan['stair']
    if 'void_poly' in st:
        return [list(q) for q in st['void_poly']] + [list(st['void_poly'][0])]
    c, r = st['well_center'], st['well_radius']
    outer = list(st['outer_edge'])
    a_end = math.degrees(math.atan2(st['top_step'][1][1] - c[1], st['top_step'][1][0] - c[0]))
    a_start = math.degrees(math.atan2(st['inner_edge'][-1][1] - c[1], st['inner_edge'][-1][0] - c[0]))
    # from the stair bottom inner point go counter-clockwise on screen (bottom -> right -> top) to the stair top
    circ = arc_points(c, r, a_end, a_start, clockwise=False).tolist()
    return outer + circ + [outer[0]]


# ---------------------------------------------------------------- registration

def rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def register(plan_pts, grid_occ, meta, path_xy=None, floor_poly_m=None, void_poly_m=None,
             res=0.05, angles=np.arange(0, 360, 1.0), mirror=(False, True), top=5, free_mask=None):
    """Coarse exhaustive search over rotation (+ mirror) with FFT cross-correlation for translation,
    then ICP refinement (rigid + uniform scale). Score = mean blurred-occupancy under plan points.
    Returns the best transform (plan metres -> map metres) and diagnostics."""
    from scipy.signal import fftconvolve
    ox, oy = meta['origin'][:2]
    gres = meta['resolution']
    occ = grid_occ.astype(np.float32)
    soft = ndimage.gaussian_filter(occ, 1.5)
    soft /= soft.max()
    H, W = occ.shape
    results = []
    for mir in mirror:
        P = plan_pts.copy()
        if mir:
            P[:, 0] = -P[:, 0]
        for a in angles:
            Q = P @ rot(math.radians(a)).T
            mn = Q.min(0)
            ij = np.floor((Q - mn) / gres).astype(int)
            k = np.zeros((ij[:, 1].max() + 1, ij[:, 0].max() + 1), np.float32)
            np.add.at(k, (ij[:, 1], ij[:, 0]), 1.0)
            k = (k > 0).astype(np.float32)
            corr = fftconvolve(soft, k[::-1, ::-1], mode='full')
            r_, c_ = np.unravel_index(np.argmax(corr), corr.shape)
            # kernel origin cell (0,0) lands on grid cell (r_ - (kh-1), c_ - (kw-1))
            gr, gc = r_ - (k.shape[0] - 1), c_ - (k.shape[1] - 1)
            t = np.array([ox + gc * gres, oy + gr * gres]) - mn
            score = corr[r_, c_] / k.sum()
            results.append({'mirror': mir, 'angle': float(a), 't': t, 'score': float(score)})
    results.sort(key=lambda d: -d['score'])
    best = []
    for cand in results[:top * 4]:
        if any(c['mirror'] == cand['mirror'] and abs(((c['angle'] - cand['angle'] + 180) % 360) - 180) < 4 for c in best):
            continue
        best.append(cand)
        if len(best) >= top:
            break
    occ_pts = np.argwhere(grid_occ)[:, ::-1] * gres + np.array([ox, oy]) + gres / 2
    refined = []
    for cand in best:
        T = icp(plan_pts, occ_pts, cand, mirror=cand['mirror'])
        T.update(evaluate(T, plan_pts, occ_pts, path_xy, floor_poly_m, void_poly_m, grid=(grid_occ, meta, free_mask)))
        refined.append(T)
    refined.sort(key=lambda d: -d['rank_score'])
    return refined, results[:10]


def apply_T(T, P):
    P = np.asarray(P, float).copy()
    if T['mirror']:
        P[..., 0] = -P[..., 0]
    return T['scale'] * (P @ rot(math.radians(T['angle'])).T) + T['t']


def icp(plan_pts, occ_pts, init, mirror, iters=40, max_d=0.4, scale=True):
    from scipy.spatial import cKDTree
    T = {'mirror': mirror, 'angle': init['angle'], 't': np.array(init['t'], float), 'scale': 1.0}
    tree = cKDTree(occ_pts)
    for _ in range(iters):
        Q = apply_T(T, plan_pts)
        d, j = tree.query(Q, distance_upper_bound=max_d)
        ok = np.isfinite(d)
        if ok.sum() < 50:
            break
        A = apply_T({**T, 'angle': 0.0, 't': np.zeros(2), 'scale': 1.0}, plan_pts[ok])   # mirrored plan pts
        B = occ_pts[j[ok]]
        ma, mb = A.mean(0), B.mean(0)
        Ac, Bc = A - ma, B - mb
        U, S, Vt = np.linalg.svd(Ac.T @ Bc)
        R = (U @ Vt).T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = (U @ Vt).T
        s = (S.sum() / (Ac ** 2).sum()) if scale else 1.0
        s = float(np.clip(s, 0.9, 1.1))
        ang = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        t = mb - s * (R @ ma)
        T = {'mirror': mirror, 'angle': ang, 't': t, 'scale': s}
        max_d = max(0.15, max_d * 0.9)
    return T


def point_in_poly(pts, poly):
    from matplotlib.path import Path
    return Path(np.asarray(poly)).contains_points(np.asarray(pts))


def evaluate(T, plan_pts, occ_pts, path_xy, floor_poly_m, void_poly_m, tol=0.15, grid=None):
    from scipy.spatial import cKDTree
    Q = apply_T(T, plan_pts)
    d_plan, _ = cKDTree(occ_pts).query(Q)
    d_occ, _ = cKDTree(Q).query(occ_pts)
    out = {'plan_pts_near_occ_pct': round(float((d_plan < tol).mean() * 100), 1),
           'occ_near_plan_pct': round(float((d_occ < tol).mean() * 100), 1),
           'occ_median_dist_m': round(float(np.median(d_occ)), 3)}
    if grid is not None and grid[2] is not None:
        occ_g, meta_g, free_g = grid
        # 'mapped' = cells within 0.5 m of observed free space; plan walls there should have been seen
        mapped = ndimage.binary_dilation(free_g, iterations=int(0.5 / meta_g['resolution']))
        gres = meta_g['resolution']
        ij = np.floor((Q - np.array(meta_g['origin'][:2])) / gres).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < occ_g.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < occ_g.shape[0])
        inm = np.zeros(len(Q), bool)
        inm[ok] = mapped[ij[ok, 1], ij[ok, 0]]
        out['plan_pts_in_mapped_area_pct'] = round(float(inm.mean() * 100), 1)
        out['mapped_plan_pts_near_occ_pct'] = round(float((d_plan[inm] < tol).mean() * 100), 1) if inm.any() else None
        out['mapped_plan_pts_median_dist_m'] = round(float(np.median(d_plan[inm])), 3) if inm.any() else None
    rank = out['occ_near_plan_pct'] + 0.5 * (out.get('mapped_plan_pts_near_occ_pct') or 0)
    if path_xy is not None and floor_poly_m is not None:
        inside = np.zeros(len(path_xy), bool)
        for poly in floor_poly_m:
            inside |= point_in_poly(path_xy, apply_T(T, np.asarray(poly)))
        out['path_on_plan_floor_pct'] = round(float(inside.mean() * 100), 1)
        d_path, _ = cKDTree(Q).query(path_xy)
        out['path_min_wall_clear_m'] = round(float(d_path.min()), 3)
        out['path_poses_closer_than_0p15_to_wall'] = int((d_path < 0.15).sum())
        rank += 0.5 * out['path_on_plan_floor_pct']
        if void_poly_m is not None:
            v = point_in_poly(path_xy, apply_T(T, np.asarray(void_poly_m)))
            out['path_in_stair_void'] = int(v.sum())
            rank -= 2.0 * v.mean() * 100
    out['rank_score'] = round(float(rank), 2)
    out['t'] = [round(float(v), 4) for v in T['t']]
    out['angle'] = round(float(T['angle']), 3)
    out['scale'] = round(float(T['scale']), 4)
    out['mirror'] = bool(T['mirror'])
    return out


def dump_plan(plan, path):
    """Write a plan JSON compactly: one line per room / door / window / fixture / railing / stair key."""
    def c(v):
        return json.dumps(v, separators=(', ', ': '))
    out = ['{']
    keys = list(plan)
    for k_i, k in enumerate(keys):
        v = plan[k]
        end = ',' if k_i < len(keys) - 1 else ''
        if isinstance(v, list) and v and all(isinstance(x, (dict, str)) for x in v):
            out.append(f' "{k}": [')
            out += [f'  {c(x)}' + (',' if j < len(v) - 1 else '') for j, x in enumerate(v)]
            out.append(f' ]{end}')
        elif isinstance(v, dict):
            out.append(f' "{k}": {{')
            ks = list(v)
            out += [f'  "{kk}": {c(v[kk])}' + (',' if j < len(ks) - 1 else '') for j, kk in enumerate(ks)]
            out.append(f' }}{end}')
        else:
            out.append(f' "{k}": {c(v)}{end}')
    out.append('}')
    with open(path, 'w', newline='\n') as f:
        f.write('\n'.join(out) + '\n')
