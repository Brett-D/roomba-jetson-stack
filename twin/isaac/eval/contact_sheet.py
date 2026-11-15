"""Contact sheet of dataset frames with YOLO boxes (green person, orange railing) + semantic overlay."""
import argparse
import glob
import os

import cv2
import numpy as np

PAL = np.array([[0, 0, 0], [60, 120, 60], [170, 170, 170], [30, 90, 200], [220, 220, 120], [0, 0, 255],
                [0, 165, 255], [255, 0, 255], [40, 90, 140], [200, 0, 200], [255, 230, 120]], np.uint8)   # BGR per class id

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('out')
ap.add_argument('--n', type=int, default=12)
ap.add_argument('--start', type=int, default=0)
ap.add_argument('--with-semantic', action='store_true')
a = ap.parse_args()
imgs = sorted(glob.glob(os.path.join(a.root, 'images', '*.png')))
pick = imgs[a.start::max(1, (len(imgs) - a.start) // a.n)][:a.n]
tiles = []
for f in pick:
    stem = os.path.splitext(os.path.basename(f))[0]
    im = cv2.imread(f)
    H, W = im.shape[:2]
    lab = os.path.join(a.root, 'labels', stem + '.txt')
    for ln in (open(lab).read().split('\n') if os.path.exists(lab) else []):
        p = ln.split()
        if len(p) < 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        x1, y1, x2, y2 = int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0) if c == 0 else (0, 165, 255), 2)
    row = [im]
    if a.with_semantic:
        s = cv2.imread(os.path.join(a.root, 'semantic', stem + '.png'), cv2.IMREAD_UNCHANGED)
        if s is not None:
            row.append(PAL[np.clip(s, 0, len(PAL) - 1)])
        d = cv2.imread(os.path.join(a.root, 'depth', stem + '.png'), cv2.IMREAD_UNCHANGED)
        if d is not None:
            row.append(cv2.applyColorMap((np.clip(d / 7000.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    tiles.append(np.hstack(row))
cols = 1 if a.with_semantic else 3
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
if sheet.shape[1] > 1920:
    sheet = cv2.resize(sheet, (1920, int(sheet.shape[0] * 1920 / sheet.shape[1])))
cv2.imwrite(a.out, sheet)
print('wrote', a.out, sheet.shape)
