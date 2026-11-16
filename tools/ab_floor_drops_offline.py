#!/usr/bin/env python3
"""Exact A/B of the floor_obstacles drop-off detection (negative obstacles): roomba_perception's
floor_obstacles.drop_points against roomba_rs::floor::drop_points (examples/floor_drops_replay.rs)
on the same inputs, every output coordinate compared bit for bit.
  tools/ab_floor_drops_offline.py [--cases N] [--rust BIN]
Cases: a camera 0.11 m (robot) or 0.4 m above the floor, tilted at random, seeing floor, a drop-off
(stair treads, the stairwell's far wall, the floor below) at random distances, walls and random
speckles above and below the floor, with depth noise; parameters varied around the robot's.
Build the replay first: cd src/roomba_rs && cargo build --release --examples
"""
import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys

import numpy as np

from roomba_perception.floor_obstacles import drop_points, floor_axes

CARGO_TARGET = os.environ.get('CARGO_TARGET_DIR') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src/roomba_rs/target')


def bits(x):
    return struct.unpack('<Q', struct.pack('<d', float(x)))[0]


def unbits(b):
    return struct.unpack('<d', struct.pack('<Q', b))[0]


def case(rng):
    """Optical-frame points (x right, y down, z forward) of one view, the world 'up' in that frame,
    and the floor height relative to the camera."""
    cam_h = rng.choice([0.11, 0.11, 0.4])
    pitch = rng.uniform(-0.35, 0.15)          # rad, + = looking down
    roll = rng.uniform(-0.15, 0.15)
    # up in the optical frame for a camera pitched down by `pitch` and rolled by `roll`
    up = np.array([math.sin(roll) * math.cos(pitch), -math.cos(roll) * math.cos(pitch), math.sin(pitch)])
    up /= np.linalg.norm(up)
    fwd, side = floor_axes(up)
    pts = []
    edge = rng.uniform(0.4, 2.2)              # where the floor ends ahead
    width = rng.uniform(0.6, 2.0)
    drop = rng.uniform(0.1, 3.0)              # how far below the floor
    noise = rng.uniform(0.0, 0.03)
    for _ in range(rng.randint(300, 2500)):
        f = rng.uniform(0.3, 4.0)
        s = rng.uniform(-1.5, 1.5)
        kind = rng.random()
        if f < edge or abs(s) > width / 2:
            hgt = -cam_h + rng.gauss(0.0, noise)                       # floor
        elif kind < 0.5:
            hgt = -cam_h - drop + rng.gauss(0.0, noise)                # below the drop
        else:
            hgt = -cam_h - rng.uniform(0.0, drop)                      # stairwell wall / treads
        if rng.random() < 0.03:
            hgt = rng.uniform(-cam_h - 1.0, 1.0)                       # speckles anywhere
        if rng.random() < 0.05:
            hgt = rng.uniform(-cam_h + 0.05, 0.8)                      # walls, furniture
        p = f * fwd + s * side + hgt * up
        if rng.random() < 0.02:
            p = p * rng.choice([1.0, 1.0 + 1e-15, 0.5])                # near-duplicates, odd rays
        pts.append(p)
    pts = np.array(pts, dtype=np.float64)
    pts = pts[pts[:, 2] > 0.05]
    floor_h = -cam_h if rng.random() < 0.7 else -cam_h + rng.gauss(0.0, 0.01)
    params = dict(below=rng.choice([0.12, 0.12, 0.08, 0.2]), max_range=rng.choice([2.0, 2.0, 1.5, 3.0]),
                  lift=rng.choice([0.1, 0.1, 0.05]), cell=rng.choice([0.05, 0.05, 0.1]),
                  min_points=rng.choice([6, 6, 1, 3, 10]))
    return pts, up, fwd, side, floor_h, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=400)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--rust', default=os.path.join(CARGO_TARGET, 'release/examples/floor_drops_replay'))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    lines, expected = [], []
    for _ in range(args.cases):
        pts, up, fwd, side, floor_h, prm = case(rng)
        h = pts @ up
        out = drop_points(pts, h, floor_h, up, prm['below'], prm['max_range'], prm['lift'], fwd, side,
                          prm['cell'], prm['min_points'])
        expected.append([[bits(v) for v in q] for q in out])
        lines.append(json.dumps({
            'pts': [[bits(v) for v in p] for p in pts], 'h': [bits(v) for v in h], 'floor_h': bits(floor_h),
            'up': [bits(v) for v in up], 'fwd': [bits(v) for v in fwd], 'side': [bits(v) for v in side],
            'below': bits(prm['below']), 'max_range': bits(prm['max_range']), 'lift': bits(prm['lift']),
            'cell': bits(prm['cell']), 'min_points': prm['min_points']}))
    res = subprocess.run([args.rust], input='\n'.join(lines) + '\n', capture_output=True, text=True)
    if res.returncode:
        sys.exit(f'rust replay failed: {res.stderr[-2000:]}')
    got = [json.loads(line) for line in res.stdout.splitlines()]
    assert len(got) == len(expected), (len(got), len(expected))
    points = sum(len(e) for e in expected)
    diff = [k for k, (a, b) in enumerate(zip(expected, got)) if a != b]
    print(f'cases {len(expected)}, input points {sum(len(json.loads(x)["pts"]) for x in lines)}, '
          f'drop points {points} (cases with any: {sum(1 for e in expected if e)})')
    for k in diff[:5]:
        a, b = expected[k], got[k]
        print(f'DIFF case {k}: python {len(a)} points, rust {len(b)}; first '
              f'{[unbits(v) for v in a[0]] if a else None} vs {[unbits(v) for v in b[0]] if b else None}')
    print('RESULT: identical' if not diff else f'RESULT: {len(diff)} cases differ')
    sys.exit(1 if diff else 0)


if __name__ == '__main__':
    main()
