"""Validate the remodelled plan against the RTAB-Map grid and the recorded robot path.

  python validate_remodel.py   -> E:\\twin\\assets\\remodel\\validation.json, E:\\twin\\screens\\remodel_overlay.png

Rasterises floorplan.json (map frame) at the grid resolution and reports:
  wall coverage   plan wall cells in the mapped area that have an occupied grid cell within 0.10 / 0.15 m
  grid explained  occupied grid cells within 0.15 m of any plan element (walls, railings, furniture, fixtures)
  wall IoU        IoU of plan walls vs occupied cells near walls, both dilated by 0.10 m, inside the mapped area
  path clearance  recorded base_link poses whose robot disc (r = 0.17 m) touches a plan element
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))
import house_lib  # noqa: E402

FP = r'E:\twin\assets\remodel\floorplan.json'
fp = json.load(open(FP))
grid, meta = house_lib.load_grid(r'E:\twin\assets\house\house.yaml')
occ = house_lib.clean_occupied(grid, 4)
res, (ox, oy) = meta['resolution'], meta['origin'][:2]
H, W = grid.shape
path = house_lib.load_poses(r'E:\twin\assets\house\house_poses.txt')[:, 1:3]


def raster(polys, filled=True, width=1):
    m = np.zeros((H, W), np.uint8)
    for p in polys:
        q = np.round(((np.asarray(p) - [ox, oy]) / res) * 8).astype(np.int32)
        if filled:
            cv2.fillPoly(m, [q], 1, shift=3)
        else:
            cv2.polylines(m, [q], False, 1, width, shift=3)
    return m.astype(bool)


walls = raster(fp['walls'])
rails = raster([r['polyline'] for r in fp['railings']], filled=False, width=2)
furn = raster([f['footprint'] for f in fp['furniture']] + [f['footprint'] for f in fp.get('fixtures', [])])
void = raster([fp['stair_void']['polygon']])
doors_closed = raster([d['segment'] for d in fp['doors'] if d['default'] == 'closed'], filled=False, width=2)
elements = walls | rails | furn | doors_closed
mapped = ndimage.binary_dilation(grid == house_lib.FREE, iterations=int(0.5 / res))
d_occ = ndimage.distance_transform_edt(~occ) * res
d_elem = ndimage.distance_transform_edt(~elements) * res
wm = walls & mapped
occ_near_walls = occ & (ndimage.distance_transform_edt(~walls) * res <= 0.3) & mapped
k = int(round(0.10 / res))
A = ndimage.binary_dilation(walls, iterations=k) & mapped
B = ndimage.binary_dilation(occ_near_walls, iterations=k)
out = {
    'wall_cells_in_mapped_area': int(wm.sum()),
    'wall_coverage_10cm_pct': round(float((d_occ[wm] <= 0.10).mean() * 100), 1),
    'wall_coverage_15cm_pct': round(float((d_occ[wm] <= 0.15).mean() * 100), 1),
    'grid_occupied_explained_15cm_pct': round(float((d_elem[occ] <= 0.15).mean() * 100), 1),
    'grid_occupied_explained_by_walls_only_pct': round(float(((ndimage.distance_transform_edt(~walls) * res)[occ] <= 0.15).mean() * 100), 1),
    'wall_iou_10cm': round(float((A & B).sum() / max((A | B).sum(), 1)), 3),
}
ij = np.floor((path - [ox, oy]) / res).astype(int)
ok = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
clear = np.full(len(path), np.inf)
clear[ok] = d_elem[ij[ok, 1], ij[ok, 0]]
inv = np.zeros(len(path), bool)
inv[ok] = void[ij[ok, 1], ij[ok, 0]]
out.update({'path_poses': len(path), 'path_poses_robot_disc_hits_element': int((clear < 0.17).sum()),
            'path_poses_in_stair_void': int(inv.sum()), 'path_min_clearance_m': round(float(clear.min()), 3),
            'path_median_clearance_m': round(float(np.median(clear)), 3)})
by_kind = {}
for name, m in (('walls', walls), ('railings', rails), ('furniture_fixtures', furn)):
    d = ndimage.distance_transform_edt(~m) * res
    c = np.full(len(path), np.inf)
    c[ok] = d[ij[ok, 1], ij[ok, 0]]
    by_kind[name] = int((c < 0.17).sum())
out['path_hits_by_kind'] = by_kind
json.dump(out, open(os.path.join(os.path.dirname(FP), 'validation.json'), 'w'), indent=1)
print('VALIDATION ' + json.dumps(out))

S = 4
img = np.full((H, W, 3), 255, np.uint8)
img[grid == house_lib.UNKNOWN] = (228, 228, 228)
img[occ] = (90, 90, 90)
img[void] = (255, 215, 170)
img[furn] = (120, 170, 220)
img[walls] = (30, 30, 200)
img[walls & occ] = (0, 0, 120)
img[rails] = (0, 140, 255)
img = cv2.resize(img[::-1], None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
pp = np.stack([(path[:, 0] - ox) / res * S, (H - (path[:, 1] - oy) / res) * S], 1).astype(np.int32)
cv2.polylines(img, [pp], False, (0, 160, 0), 2, cv2.LINE_AA)
for i in np.nonzero(clear < 0.17)[0]:
    cv2.circle(img, tuple(pp[i]), 5, (255, 0, 255), 1)
for r in fp['rooms']:
    c = np.mean(r['polygon'], axis=0)
    cv2.putText(img, r['name'], (int((c[0] - ox) / res * S) - 30, int((H - (c[1] - oy) / res) * S)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 40, 0), 1, cv2.LINE_AA)
lines = [f"remodel vs RTAB-Map grid: wall coverage {out['wall_coverage_10cm_pct']}% (10 cm) / {out['wall_coverage_15cm_pct']}% (15 cm), "
         f"wall IoU {out['wall_iou_10cm']}, grid occupied explained {out['grid_occupied_explained_15cm_pct']}%",
         f"path: {out['path_poses_robot_disc_hits_element']} of {out['path_poses']} poses touch a plan element "
         f"(walls {by_kind['walls']}, railings {by_kind['railings']}, furniture/fixtures {by_kind['furniture_fixtures']}), "
         f"{out['path_poses_in_stair_void']} in the stair void",
         'red/dark-red: plan walls (dark = on occupied cells), grey: grid occupied, orange: railings, blue: furniture/fixtures, '
         'peach: stair void, green: robot path, magenta: contacts']
for k_, t in enumerate(lines):
    cv2.putText(img, t, (10, 22 + 20 * k_), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
cv2.imwrite(r'E:\twin\screens\remodel_overlay.png', img)
