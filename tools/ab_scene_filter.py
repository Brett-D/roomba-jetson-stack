#!/usr/bin/env python3
"""Exact A/B of roomba_rs::scene_filter (examples/scene_filter_replay.rs) against
roomba_perception/scene_filter.py: point_in_polygon on random polygons (also with points on edges and
vertices), ZoneSet.load_json + containing on random zone messages (keep-out and too-small zones
skipped, missing names/frames), and GridLineOfSight.blocked on random grids (rotated origins,
float32 resolutions) and rays (short, long, off the map, custom skips).
  tools/ab_scene_filter.py [--cases N] [--seed S] [--rust BIN] [--work DIR]
"""
import argparse
import json
import math
import os
import random
import subprocess
from types import SimpleNamespace

import numpy as np

from roomba_perception.scene_filter import GridLineOfSight, ZoneSet, point_in_polygon

ap = argparse.ArgumentParser()
ap.add_argument('--cases', type=int, default=3000)
ap.add_argument('--seed', type=int, default=5)
ap.add_argument('--rust', default=os.path.expanduser('~/rs_lanes/tfmap_target/release/examples/scene_filter_replay'))
ap.add_argument('--work', default='/data/rs_lanes/tfmap')
args = ap.parse_args()
rng = random.Random(args.seed)
cases, expected = [], []


def poly():
    n = rng.randint(3, 9)
    cx, cy = rng.uniform(-5, 5), rng.uniform(-5, 5)
    if rng.random() < 0.3:   # grid-aligned integer polygons: many points exactly on edges
        return [[rng.randint(-3, 3), rng.randint(-3, 3)] for _ in range(n)]
    return [[cx + rng.uniform(-2, 2), cy + rng.uniform(-2, 2)] for _ in range(n)]


def queries(p, n=30):
    q = [[rng.uniform(-8, 8), rng.uniform(-8, 8)] for _ in range(n)]
    q += [list(v) for v in p[:3]] + [[(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2]]
    return q + [[rng.randint(-3, 3), rng.randint(-3, 3)] for _ in range(10)]


for k in range(args.cases):
    kind = k % 3
    if kind == 0:
        p = poly()
        q = queries(p)
        cases.append({'pip': p, 'q': q})
        expected.append([point_in_polygon(x, y, p) for x, y in q])
    elif kind == 1:
        zones = []
        for i in range(rng.randint(0, 5)):
            z = {'points': poly() if rng.random() < 0.9 else [[0, 0], [1, 1]]}
            if rng.random() < 0.8:
                z['name'] = rng.choice(['tv', f'zone_{i}', 'K\u00fcche'])
            if rng.random() < 0.5:
                z['kind'] = rng.choice(['exclusion', 'keepout'])
            zones.append(z)
        msg = {'zones': zones}
        if rng.random() < 0.8:
            msg['frame'] = rng.choice(['map', 'odom'])
        text = json.dumps(msg)
        zs = ZoneSet()
        zs.load_json(text)
        q = queries(zones[0]['points']) if zones else [[0.0, 0.0]]
        cases.append({'zones': text, 'q': q})
        expected.append({'frame': zs.frame, 'names': [zs.containing(x, y) for x, y in q]})
    else:
        w, h = rng.randint(1, 80), rng.randint(1, 80)
        res = float(np.float32(rng.choice([0.05, 0.03, 0.1])))
        data = [rng.choice([-1, 0, 0, 0, 30, 65, 100]) for _ in range(w * h)]
        yaw = rng.choice([0.0, rng.uniform(-math.pi, math.pi)])
        q = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
        ox, oy = rng.uniform(-3, 3), rng.uniform(-3, 3)
        los = GridLineOfSight(occupied_threshold=65)
        info = SimpleNamespace(width=w, height=h, resolution=res,
                               origin=SimpleNamespace(position=SimpleNamespace(x=ox, y=oy),
                                                      orientation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3])))
        los.update(SimpleNamespace(info=info, data=data, header=SimpleNamespace(frame_id='map')))
        rays = []
        for _ in range(20):
            s = [rng.uniform(-4, 4), rng.uniform(-4, 4)]
            e = [s[0] + rng.uniform(-5, 5), s[1] + rng.uniform(-5, 5)]
            skips = [0.15, 0.35] if rng.random() < 0.7 else [rng.uniform(0, 0.5), rng.uniform(0, 0.5)]
            rays.append(s + e + skips)
        cases.append({'grid': {'w': w, 'h': h, 'res': res, 'ox': ox, 'oy': oy, 'q': q, 'data': data}, 'threshold': 65,
                      'rays': rays})
        expected.append([los.blocked(r[:2], r[2:4], r[4], r[5]) for r in rays])

path = os.path.join(args.work, 'scene_cases.jsonl')
with open(path, 'w') as f:
    for c in cases:
        f.write(json.dumps(c) + '\n')
out = subprocess.run([args.rust, path], capture_output=True, text=True, check=True).stdout.splitlines()
os.remove(path)
names = ('point_in_polygon', 'ZoneSet', 'GridLineOfSight')
checked, diffs = [0, 0, 0], [0, 0, 0]
for k, (exp, line) in enumerate(zip(expected, out)):
    got = json.loads(line)
    n = len(exp['names']) if isinstance(exp, dict) else len(exp)
    checked[k % 3] += n
    if got != exp:
        diffs[k % 3] += 1
        if sum(diffs) <= 5:
            print('DIFF', names[k % 3], json.dumps(cases[k])[:300], '\n  py', exp, '\n  rs', got)
assert len(out) == len(expected)
print(', '.join(f'{names[i]}: {checked[i]} results, {diffs[i]} cases differ' for i in range(3)))
