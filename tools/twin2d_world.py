#!/usr/bin/env python3
"""Builds the 2D digital twin's world (the house floor plan) from a saved RTAB-Map database.

  tools/twin2d_world.py --db /data/twin/source/vslam.db --out /data/twin/twin2d/world

1. Copies the database (rtabmap-export may write to it; the source stays untouched) and runs
   `rtabmap-export --map --poses --opt 2`: the 2D occupancy grid RTAB-Map saved with the map
   (map frame, 5 cm cells) and the robot's trajectory. Kept as house_raw.pgm/.yaml, house_poses.txt.
2. Cleans the grid conservatively into house.pgm/.yaml (same frame, size and cells):
   - free "spikes": 1-2 cell wide free rays RTAB-Map traced through gaps into unknown space
     (glass, doorways, railings) are removed (morphological opening of the free space with a 3x3
     square; nothing a 0.34 m robot fits through is lost);
   - small unknown holes (< --hole_m2) enclosed by free space become free (floor the camera never
     happened to see, e.g. under the robot or behind a chair leg);
   - occupied cells are kept as they are (furniture legs are real obstacles).
   The twin treats every cell that isn't free (occupied or unknown) as solid, full-height wall.
3. Writes house_preview.png (raw | cleaned, 1 m grid with coordinates, trajectory in red, start in
   green) to pick door, railing and obstacle coordinates for scenario files.
PGM values: 254 free, 0 occupied, 205 unknown (map_server convention, row 0 = top = max y).
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import yaml

FREE, OCC, UNK = 254, 0, 205


def read_map(yaml_path):
    with open(yaml_path) as f:
        info = yaml.safe_load(f)
    img = cv2.imread(os.path.join(os.path.dirname(yaml_path), info['image']), cv2.IMREAD_UNCHANGED)
    return info, img


def write_map(out_dir, name, info, img, note):
    cv2.imwrite(os.path.join(out_dir, name + '.pgm'), img)
    meta = {'image': name + '.pgm', 'resolution': info['resolution'], 'origin': info['origin'],
            'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196}
    with open(os.path.join(out_dir, name + '.yaml'), 'w') as f:
        f.write(f'# {note}\n')
        yaml.safe_dump(meta, f, default_flow_style=None, sort_keys=False)


def clean(img, res, hole_m2):
    free = (img == FREE).astype(np.uint8)
    opened = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = img.copy()
    spikes = (free == 1) & (opened == 0)
    out[spikes] = UNK
    # Unknown pockets enclosed by free space (4-connected components not touching the border and
    # bordered only by free cells) -> free.
    unk = (out == UNK).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(unk, connectivity=4)
    max_cells = hole_m2 / (res * res)
    filled = 0
    h, w = out.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area > max_cells or x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        comp = (labels == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((3, 3), np.uint8)) & (1 - comp)
        if np.all(out[ring == 1] == FREE):
            out[comp == 1] = FREE
            filled += area
    return out, int(spikes.sum()), int(filled)


def preview(raw, cleaned, info, poses, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = info['resolution']
    ox, oy = info['origin'][0], info['origin'][1]
    h, w = raw.shape
    extent = [ox, ox + w * res, oy, oy + h * res]
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for ax, img, title in ((axes[0], raw, 'RTAB-Map grid (raw)'), (axes[1], cleaned, 'twin world (cleaned)')):
        ax.imshow(img, cmap='gray', extent=extent, interpolation='nearest', vmin=0, vmax=255)
        if poses is not None and len(poses):
            ax.plot(poses[:, 0], poses[:, 1], 'r.', ms=2)
        ax.plot([0], [0], 'gs', ms=8)
        ax.set_xticks(np.arange(np.floor(extent[0]), extent[1] + 1, 1.0))
        ax.set_yticks(np.arange(np.floor(extent[2]), extent[3] + 1, 1.0))
        ax.grid(True, color='tab:blue', alpha=0.35, lw=0.6)
        ax.tick_params(labelsize=7)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default='/data/twin/source/vslam.db')
    ap.add_argument('--out', default='/data/twin/twin2d/world')
    ap.add_argument('--work', default='', help='scratch dir for the database copy (default: <out>/.work, deleted)')
    ap.add_argument('--hole_m2', type=float, default=0.25, help='largest enclosed unknown hole that becomes free')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = a.work or tempfile.mkdtemp(prefix='.work', dir=a.out)
    try:
        db = os.path.join(work, 'house.db')
        shutil.copyfile(a.db, db)
        os.chmod(db, 0o644)
        subprocess.run(['rtabmap-export', '--map', '--poses', '--opt', '2', '--output', 'house',
                        '--output_dir', work, db], check=True, stdout=subprocess.DEVNULL)
        info, raw = read_map(os.path.join(work, 'house.yaml'))
        shutil.copyfile(os.path.join(work, 'house_poses.txt'), os.path.join(a.out, 'house_poses.txt'))
    finally:
        if not a.work:
            shutil.rmtree(work, ignore_errors=True)
    write_map(a.out, 'house_raw', info, raw, f'rtabmap-export --map --opt 2 of {a.db}')
    cleaned, spikes, filled = clean(raw, info['resolution'], a.hole_m2)
    write_map(a.out, 'house', info, cleaned,
              f'twin world: house_raw with {spikes} free spike cells removed, {filled} hole cells filled '
              '(tools/twin2d_world.py); not free = solid wall')
    poses = np.loadtxt(os.path.join(a.out, 'house_poses.txt'), usecols=(1, 2), ndmin=2)
    preview(raw, cleaned, info, poses, os.path.join(a.out, 'house_preview.png'))
    res = info['resolution']
    print(f'grid {raw.shape[1]}x{raw.shape[0]} cells at {res} m, origin {info["origin"][:2]}; '
          f'free {np.sum(raw == FREE) * res * res:.1f} m2 raw, {np.sum(cleaned == FREE) * res * res:.1f} m2 cleaned '
          f'({spikes} spike cells removed, {filled} hole cells filled)')


if __name__ == '__main__':
    main()
