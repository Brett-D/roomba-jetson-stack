#!/usr/bin/env python3
"""Side-by-side map image and metrics table for compare_pipelines.sh.

  python3 compare_render.py OUT_DIR NAME LABEL_A PREFIX_A LABEL_B PREFIX_B

Reads PREFIX_{grid.npz,map_track.npy,.json,_perf.json} written by compare_monitor.py and
perf_sample.py and writes OUT_DIR/NAME_side_by_side.png (both RTAB-Map grids at the same scale
and extent, path in green, start/end marked), OUT_DIR/NAME_overlay.png (obstacles of A in red, B
in blue, both in black) and OUT_DIR/NAME_metrics.md. Both maps are drawn in the robot's start
frame (start pose at the origin, heading to the right), so they line up at the start and the
overlay shows where they disagree.
"""
import json
import os
import sqlite3
import sys

import cv2
import numpy as np

PX = 4  # pixels per cell


def load(prefix):
    out = {'metrics': {}, 'perf': {}}
    for key, path in (('metrics', prefix + '.json'), ('perf', prefix + '_perf.json')):
        if os.path.isfile(path):
            with open(path) as f:
                out[key] = json.load(f)
    if os.path.isfile(prefix + '_grid.npz'):
        g = np.load(prefix + '_grid.npz')
        out['grid'] = (g['data'], float(g['resolution']), tuple(g['origin']))
    if os.path.isfile(prefix + '_map_track.npy'):
        out['track'] = np.load(prefix + '_map_track.npy')
    if os.path.isfile(prefix + '.db'):
        # Loop closures as stored in the final graph (RTAB-Map Link types: 1 global loop
        # closure, 2 local space/proximity closure), counted once per node pair.
        try:
            con = sqlite3.connect(f'file:{prefix}.db?mode=ro', uri=True)
            for t, key in ((1, 'db_loop_links'), (2, 'db_proximity_links')):
                out['metrics'][key] = con.execute(
                    "SELECT COUNT(DISTINCT MIN(from_id, to_id) || '-' || MAX(from_id, to_id)) "
                    "FROM Link WHERE type = ?", (t,)).fetchone()[0]
            out['metrics']['db_nodes'] = con.execute('SELECT COUNT(*) FROM Node').fetchone()[0]
            con.close()
        except sqlite3.Error as e:
            print(f'{prefix}.db: {e}')
    return out


def start_pose(run):
    """The robot's first pose in the run's map frame (x, y, yaw); identity without a track."""
    if 'track' in run and len(run['track']):
        return tuple(float(v) for v in run['track'][0, 1:4])
    return 0.0, 0.0, 0.0


def to_start(run, x, y):
    """Map-frame points -> the run's start frame (robot's start pose at the origin, facing +x).
    Each pipeline's map frame has its own origin/heading (Basalt aligns to gravity with an
    arbitrary yaw, cuVSLAM starts at identity); in the start frame both runs line up."""
    x0, y0, a = start_pose(run)
    c, s = np.cos(a), np.sin(a)
    dx, dy = np.asarray(x) - x0, np.asarray(y) - y0
    return c * dx + s * dy, -s * dx + c * dy


def bounds(runs):
    xs, ys, res = [], [], 0.05
    for r in runs:
        if 'grid' in r:
            d, res, (ox, oy) = r['grid']
            cx = np.array([ox, ox + d.shape[1] * res, ox, ox + d.shape[1] * res])
            cy = np.array([oy, oy, oy + d.shape[0] * res, oy + d.shape[0] * res])
            sx, sy = to_start(r, cx, cy)
            xs += list(sx)
            ys += list(sy)
        if 'track' in r:
            sx, sy = to_start(r, r['track'][:, 1], r['track'][:, 2])
            xs += [sx.min(), sx.max()]
            ys += [sy.min(), sy.max()]
    if not xs:
        return None
    return min(xs) - 0.5, min(ys) - 0.5, max(xs) + 0.5, max(ys) + 0.5, res


def raster(run, b):
    """The run's grid resampled (nearest cell) onto the common start-frame canvas b."""
    x0, y0, x1, y1, res = b
    w, h = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    out = np.full((h, w), -1, np.int16)
    if 'grid' in run:
        d, r, (ox, oy) = run['grid']
        sx0, sy0, a = start_pose(run)
        c, s = np.cos(a), np.sin(a)
        v, u = np.mgrid[0:h, 0:w]
        px, py = x0 + (u + 0.5) * res, y0 + (v + 0.5) * res      # start frame
        mx, my = sx0 + c * px - s * py, sy0 + s * px + c * py      # map frame
        gi, gj = np.floor((my - oy) / r).astype(int), np.floor((mx - ox) / r).astype(int)
        ok = (gi >= 0) & (gi < d.shape[0]) & (gj >= 0) & (gj < d.shape[1])
        out[ok] = d[gi[ok], gj[ok]]
    return out


def to_px(b, x, y, h):
    return int((x - b[0]) / b[4] * PX), int((h - (y - b[1]) / b[4]) * PX)


def draw(run, b, label):
    g = raster(run, b)
    h, w = g.shape
    img = np.full((h, w, 3), 160, np.uint8)
    img[g == 0] = (255, 255, 255)
    img[g >= 65] = (0, 0, 0)
    img = cv2.resize(cv2.flip(img, 0), (w * PX, h * PX), interpolation=cv2.INTER_NEAREST)
    if 'track' in run and len(run['track']) > 1:
        sx, sy = to_start(run, run['track'][:, 1], run['track'][:, 2])
        pts = np.array([to_px(b, x, y, h) for x, y in zip(sx, sy)], np.int32)
        cv2.polylines(img, [pts], False, (0, 170, 0), 2)
        cv2.circle(img, tuple(pts[0]), 7, (0, 200, 0), 2)
        cv2.drawMarker(img, tuple(pts[-1]), (0, 0, 220), cv2.MARKER_CROSS, 14, 2)
    # Start pose heading (start frame +x).
    o = to_px(b, 0.0, 0.0, h)
    cv2.arrowedLine(img, o, to_px(b, 0.4, 0.0, h), (200, 0, 200), 2, tipLength=0.3)
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    scale = int(1.0 / b[4] * PX)
    cv2.line(img, (10, img.shape[0] - 12), (10 + scale, img.shape[0] - 12), (0, 0, 255), 3)
    cv2.putText(img, '1 m', (14 + scale, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return img


def overlay(a, bb, b):
    ga, gb = raster(a, b), raster(bb, b)
    h, w = ga.shape
    img = np.full((h, w, 3), 160, np.uint8)
    img[(ga == 0) | (gb == 0)] = (255, 255, 255)
    oa, ob = ga >= 65, gb >= 65
    img[oa & ~ob] = (40, 40, 230)
    img[ob & ~oa] = (230, 120, 30)
    img[oa & ob] = (0, 0, 0)
    return cv2.resize(cv2.flip(img, 0), (w * PX, h * PX), interpolation=cv2.INTER_NEAREST)


ROWS = [
    ('Replayed duration (s)', 'metrics', 'odom_duration_s'),
    ('Visual tracking losses', 'metrics', 'tracking_losses'),
    ('Time with tracking lost (s)', 'metrics', 'tracking_failed_s'),
    ('Watchdog recoveries (VIO resets)', 'metrics', 'watchdog_recoveries'),
    ('Watchdog filtered poses', 'metrics', 'watchdog_filtered'),
    ('Loop closures accepted', 'metrics', 'loop_closures'),
    ('Proximity closures accepted', 'metrics', 'proximity_closures'),
    ('Loop / proximity links in final DB', None, None),
    ('RTAB-Map nodes (DB)', 'metrics', 'db_nodes'),
    ('Path length, map (m)', 'metrics', 'map_path_m'),
    ('Largest odom step (m)', 'metrics', 'odom_max_step_m'),
    ('End vs start, map (m)', 'metrics', 'map_end_vs_start_m'),
    ('End vs start, map (deg)', 'metrics', 'map_end_vs_start_yaw_deg'),
    ('End vs start, odom (m)', 'metrics', 'odom_end_vs_start_m'),
    ('End vs start, odom (deg)', 'metrics', 'odom_end_vs_start_yaw_deg'),
    ('Map known area (m²)', 'metrics', 'grid_known_area_m2'),
    ('Map free area (m²)', 'metrics', 'grid_free_area_m2'),
    ('Map occupied area (m²)', 'metrics', 'grid_occupied_area_m2'),
    ('Known / occupied cells', None, None),
    ('CPU total (% of 6 cores)', 'perf', 'cpu_total_pct'),
    ('CPU cores used', 'perf', 'cpu_cores_used'),
    ('GPU load mean / max (%)', None, None),
    ('RAM used (MB, incl. GPU)', 'perf', 'ram_mb_mean'),
]


def cell(run, section, key):
    v = run[section].get(key) if section else None
    return '–' if v is None else str(v)


def main():
    out_dir, name, la, pa, lb, pb = sys.argv[1:7]
    a, bb = load(pa), load(pb)
    b = bounds([a, bb])
    if b is None:
        print('No grids or tracks to render')
    else:
        ia, ib = draw(a, b, la), draw(bb, b, lb)
        sep = np.full((ia.shape[0], 8, 3), 255, np.uint8)
        cv2.imwrite(os.path.join(out_dir, f'{name}_side_by_side.png'), np.hstack([ia, sep, ib]))
        cv2.imwrite(os.path.join(out_dir, f'{name}_overlay.png'), overlay(a, bb, b))
    lines = [f'# Pipeline comparison: {name}', '',
             f'| Metric | {la} | {lb} |', '|---|---|---|']
    for title, section, key in ROWS:
        if title.startswith('Loop / proximity'):
            va = f"{a['metrics'].get('db_loop_links', '–')} / {a['metrics'].get('db_proximity_links', '–')}"
            vb = f"{bb['metrics'].get('db_loop_links', '–')} / {bb['metrics'].get('db_proximity_links', '–')}"
        elif title.startswith('Known / occupied'):
            va = f"{a['metrics'].get('grid_known_cells', '–')} / {a['metrics'].get('grid_occupied_cells', '–')}"
            vb = f"{bb['metrics'].get('grid_known_cells', '–')} / {bb['metrics'].get('grid_occupied_cells', '–')}"
        elif title.startswith('GPU'):
            va = f"{a['perf'].get('gpu_pct_mean', '–')} / {a['perf'].get('gpu_pct_max', '–')}"
            vb = f"{bb['perf'].get('gpu_pct_mean', '–')} / {bb['perf'].get('gpu_pct_max', '–')}"
        else:
            va, vb = cell(a, section, key), cell(bb, section, key)
        lines.append(f'| {title} | {va} | {vb} |')
    for label, run in ((la, a), (lb, bb)):
        procs = run['perf'].get('processes', {})
        top = ', '.join(f'{k} {v}%' for k, v in list(procs.items())[:12])
        lines += ['', f'**{label} per-process CPU (% of one core):** {top or "–"}']
    lines += ['', f'Images: `{name}_side_by_side.png` (start frame: robot starts at the magenta arrow '
              f'facing right; green path, circle = start, red cross = end), `{name}_overlay.png` (obstacles: red only {la}, blue only {lb}, black both).',
              '"End vs start" is the loop error only if the drive ended where it started.']
    with open(os.path.join(out_dir, f'{name}_metrics.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
