"""Summarise a generated dataset (labels + meta + semantic) into <dataset>/stats_summary.json.

  python dataset_stats.py E:\\twin\\datasets\\v2 [--sem-every 10]

Merges the per-worker stats_w*.json, counts YOLO boxes per class, person distance / box height
percentiles (from meta), look / pose-mode shares and the semantic pixel share (every Nth frame).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def pct(a, qs=(10, 50, 90)):
    return [round(float(np.percentile(a, q)), 2) for q in qs] if len(a) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--sem-every', type=int, default=10)
    a = ap.parse_args()

    workers = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.data, 'stats_w*.json')))]
    labels = sorted(glob.glob(os.path.join(a.data, 'labels', '*.txt')))
    n_box = {0: 0, 1: 0}
    f_with = {0: 0, 1: 0}
    for p in labels:
        cls = [int(ln.split()[0]) for ln in open(p) if ln.strip()]
        for c in (0, 1):
            n_box[c] += cls.count(c)
            f_with[c] += c in cls
    dist, hpx, looks, modes = [], [], {}, {}
    for p in glob.glob(os.path.join(a.data, 'meta', '*.json')):
        m = json.load(open(p))
        looks[m['look']] = looks.get(m['look'], 0) + 1
        modes[m['pose_mode']] = modes.get(m['pose_mode'], 0) + 1
        for b in m['persons']:
            hpx.append(b['box'][3] - b['box'][1])
            if b['distance_m']:
                dist.append(b['distance_m'])
    classes = json.load(open(os.path.join(a.data, 'classes.json')))['semantic']
    names = {v: k for k, v in classes.items()}
    hist = np.zeros(256, np.int64)
    sems = sorted(glob.glob(os.path.join(a.data, 'semantic', '*.png')))[::a.sem_every]
    for p in sems:
        s = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        hist += np.bincount(s.ravel(), minlength=256)[:256]
    share = {names.get(i, 'unlabelled' if i == 0 else str(i)): round(float(hist[i] / max(hist.sum(), 1)), 4)
             for i in np.nonzero(hist)[0]}
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(a.data) for f in fs)
    out = {
        'frames': len(labels), 'workers': len(workers),
        'generation_seconds_per_worker': [w.get('seconds') for w in workers],
        'person_boxes': n_box[0], 'railing_boxes': n_box[1],
        'frames_with_person': f_with[0], 'frames_with_railing': f_with[1],
        'person_distance_m_p10_50_90': pct(dist), 'person_box_height_px_p10_50_90': pct(hpx),
        'looks': looks, 'pose_mode': modes, 'semantic_pixel_share': dict(sorted(share.items(), key=lambda kv: -kv[1])),
        'size_gb': round(size / 1e9, 2),
    }
    with open(os.path.join(a.data, 'stats_summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
