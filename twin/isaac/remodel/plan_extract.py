"""Floor-plan raster -> axis-aligned wall rectangles (pixel units) + residual (diagonal/curved) wall pixels.

Used to digitise the floor plans; the result is reviewed and completed by hand in
floorplan_*.json (doors, windows, rooms, curved/diagonal walls, stair void, balconies).

  python plan_extract.py floorplan.png overlay.png rects.json
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image


def load_gray(path):
    im = Image.open(path).convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return np.array(bg.convert('L'))


def wall_mask(gray, thr=70, min_area=40):
    m = (gray < thr).astype(np.uint8)
    # walls are >= 4 px thick: an opening with a 3x3 kernel removes text and tick marks
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def axis_rects(mask, min_len=8, max_thick=9):
    """Maximal axis-aligned bars: horizontal (long in x) and vertical (long in y)."""
    rects = []
    for axis in (0, 1):
        m = mask if axis == 0 else mask.T
        used = np.zeros_like(m)
        h, w = m.shape
        for r in range(h):
            c = 0
            while c < w:
                if not m[r, c] or used[r, c]:
                    c += 1
                    continue
                c1 = c
                while c1 < w and m[r, c1]:
                    c1 += 1
                if c1 - c >= min_len:
                    r1 = r + 1
                    while r1 < h and r1 - r < max_thick and m[r1, c:c1].mean() > 0.9:
                        r1 += 1
                    if r1 - r >= 2:
                        used[r:r1, c:c1] = True
                        rects.append((axis, r, r1, c, c1))
                c = c1
    out = []
    for axis, r0, r1, c0, c1 in rects:
        if axis == 0:
            out.append({'x0': c0, 'x1': c1, 'y0': r0, 'y1': r1, 'dir': 'h'})
        else:
            out.append({'x0': r0, 'x1': r1, 'y0': c0, 'y1': c1, 'dir': 'v'})
    return out


if __name__ == '__main__':
    g = load_gray(sys.argv[1])
    wm = wall_mask(g)
    rects = axis_rects(wm)
    cov = np.zeros_like(wm)
    for r in rects:
        cov[r['y0']:r['y1'], r['x0']:r['x1']] = True
    resid = wm & ~cv2.dilate(cov.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    print(json.dumps({'n_rects': len(rects), 'wall_px': int(wm.sum()), 'residual_px': int(resid.sum())}))
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    vis = (vis * 0.5 + 127).astype(np.uint8)
    for r in rects:
        cv2.rectangle(vis, (r['x0'], r['y0']), (r['x1'] - 1, r['y1'] - 1), (0, 0, 255) if r['dir'] == 'h' else (255, 0, 0), -1)
    vis[resid] = (0, 200, 0)
    cv2.imwrite(sys.argv[2], cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    json.dump(rects, open(sys.argv[3], 'w'))
