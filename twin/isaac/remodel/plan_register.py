"""Fit the scale of the digitised floor plan, register it to the RTAB-Map grid and write
the plan in the map frame (JSON + SVG) plus an overlay image.

  python plan_register.py --plan floorplan_upstairs_px.json --image E:\\twin\\assets\\floorplan_upstairs.png
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))
import house_lib  # noqa: E402
import plan_lib as pl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--plan', default=os.path.join(HERE, 'floorplan_upstairs_px.json'))
ap.add_argument('--image', default=r'E:\twin\assets\floorplan_upstairs.png')
ap.add_argument('--grid', default=r'E:\twin\assets\house\house.yaml')
ap.add_argument('--poses', default=r'E:\twin\assets\house\house_poses.txt')
ap.add_argument('--out', default=r'E:\twin\assets\remodel')
ap.add_argument('--screens', default=r'E:\twin\screens')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

plan = json.load(open(args.plan))
gray = pl.load_gray(args.image)
wmask = pl.wall_mask(gray)
s, scale_rep = pl.fit_scale(plan)
frame = pl.PlanFrame(s, gray.shape[1], gray.shape[0])
print('SCALE', json.dumps(scale_rep))

P = pl.plan_points(plan, wmask, frame)
P = P[::2]
floor_polys = [frame.m(r['poly']) for r in plan['rooms'] if not r.get('outdoor')]
void_px = pl.stair_void_polygon(plan)
void_m = frame.m(void_px)

grid, meta = house_lib.load_grid(args.grid)
occ = house_lib.clean_occupied(grid, 4)
poses = house_lib.load_poses(args.poses)
path_xy = poses[:, 1:3]
cands, coarse = pl.register(P, occ, meta, path_xy, floor_polys, void_m, free_mask=(grid == house_lib.FREE))
for c in cands:
    print('CAND', json.dumps(c))
best = cands[0]
T = {'mirror': best['mirror'], 'angle': best['angle'], 't': np.array(best['t']), 'scale': best['scale']}

# ---------------- overlay ----------------
res, (ox, oy) = meta['resolution'], meta['origin'][:2]
h, w = grid.shape
S = 4
g = np.kron(grid, np.ones((S, S), np.uint8))


def to_px(xy):
    xy = np.asarray(xy, float)
    return np.stack([(xy[..., 0] - ox) / res * S, (h - (xy[..., 1] - oy) / res) * S], -1).astype(np.int32)


img = np.full((h * S, w * S, 3), 255, np.uint8)
gflip = g[::-1]
img[gflip == house_lib.UNKNOWN] = (225, 225, 225)
img[gflip == house_lib.OCCUPIED] = (70, 70, 70)
for poly in floor_polys:
    cv2.polylines(img, [to_px(pl.apply_T(T, poly))], True, (200, 160, 60), 1, cv2.LINE_AA)
cv2.fillPoly(img, [to_px(pl.apply_T(T, void_m))], (255, 210, 160))
Q = pl.apply_T(T, pl.plan_points(plan, wmask, frame))
for q in to_px(Q)[::1]:
    if 0 <= q[0] < w * S and 0 <= q[1] < h * S:
        img[q[1], q[0]] = (0, 0, 230)
for rl in pl.railing_polylines(plan):
    cv2.polylines(img, [to_px(pl.apply_T(T, frame.m(rl['pts'])))], False, (0, 140, 255), 2, cv2.LINE_AA)
cv2.polylines(img, [to_px(path_xy)], False, (0, 170, 0), 2, cv2.LINE_AA)
for r in plan['rooms']:
    c = to_px(pl.apply_T(T, frame.m(np.mean(r['poly'], axis=0))))
    cv2.putText(img, r['name'], (int(c[0]) - 30, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 60, 0), 1, cv2.LINE_AA)
cv2.putText(img, f"plan->map: rot {best['angle']:.1f} deg, scale {best['scale']:.3f}, mirror {best['mirror']}; "
            f"grid occ within 15cm of plan {best['occ_near_plan_pct']}%, mapped plan walls near occ {best.get('mapped_plan_pts_near_occ_pct')}%, "
            f"path on plan floor {best.get('path_on_plan_floor_pct')}%", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
os.makedirs(args.screens, exist_ok=True)
cv2.imwrite(os.path.join(args.screens, 'remodel_plan_registration_overlay.png'), img)

json.dump({'scale': scale_rep, 'px_per_m': s, 'frame': {'cx': frame.cx, 'cy': frame.cy},
           'best': best, 'candidates': cands,
           'coarse_top': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in coarse]},
          open(os.path.join(args.out, 'registration.json'), 'w'), indent=1)
print('BEST', json.dumps(best))
