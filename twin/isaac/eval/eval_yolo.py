"""Baseline evaluation of the Jetson's YOLO11n person detector (same ONNX, same pre/post-processing).

  python eval_yolo.py --model E:\\twin\\assets\\yolo11n_384x640.onnx \\
      --synthetic E:\\twin\\datasets\\v1 --real E:\\twin\\real_frames\\left --out E:\\twin\\eval

Synthetic set (YOLO format: images/*.png, labels/*.txt, class 0 = person; optional meta/*.json with
per-box visibility and distance): precision / recall at the deployed threshold (conf 0.45, IoU 0.5),
recall by box height, distance and visibility, AP50 from a low-threshold sweep.
Real frames: detections per frame at conf 0.45 and overlays of the frames with detections.
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from yolo_onnx import YoloOnnx, iou, match

H_BINS = [(0, 40), (40, 80), (80, 160), (160, 1000)]
D_BINS = [(0, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 100)]
V_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def load_labels(path, w, h):
    boxes = []
    if os.path.exists(path):
        for ln in open(path):
            p = ln.split()
            if len(p) >= 5 and int(p[0]) == 0:
                cx, cy, bw, bh = (float(x) for x in p[1:5])
                boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def ap50(scores, tps, n_gt):
    if n_gt == 0 or not scores:
        return 0.0
    order = np.argsort(-np.array(scores))
    tp = np.array(tps)[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mprec = np.concatenate([[0], prec, [0]])
    mrec = np.concatenate([[0], rec, [1]])
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mprec[idx + 1]).sum())


def bin_name(b):
    return f'{b[0]}-{b[1]}'


def eval_synthetic(model, root, out, min_h=10, image_list=None, tag='synthetic'):
    if image_list:
        imgs = [ln.strip() for ln in open(image_list) if ln.strip()]
    else:
        imgs = sorted(glob.glob(os.path.join(root, 'images', '*.png')) + glob.glob(os.path.join(root, 'images', '*.jpg')))
    tp_d = fp_d = n_gt = 0
    sweep_scores, sweep_tp = [], []
    by = {'height_px': {bin_name(b): [0, 0] for b in H_BINS},
          'distance_m': {bin_name(b): [0, 0] for b in D_BINS},
          'visible_frac': {bin_name(b): [0, 0] for b in V_BINS}}
    misses, fps_list = [], []
    clean, trunc_n = [0, 0], [0, 0]
    t0 = time.time()
    for f in imgs:
        bgr = cv2.imread(f)
        H, W = bgr.shape[:2]
        stem = os.path.splitext(os.path.basename(f))[0]
        gts = load_labels(os.path.join(root, 'labels', stem + '.txt'), W, H)
        meta_p = os.path.join(root, 'meta', stem + '.json')
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        pmeta = meta.get('persons', [])
        gts = [g for g in gts if g[3] - g[1] >= min_h]
        # meta lists every person box (also the ones too occluded to label): match by box
        pm = []
        for g in gts:
            best = max(pmeta, key=lambda m: iou(g, m['box']), default=None)
            pm.append(best if best is not None and iou(g, best['box']) > 0.8 else {})
        n_gt += len(gts)
        sem_p = os.path.join(root, 'semantic', stem + '.png')
        sem = cv2.imread(sem_p, cv2.IMREAD_UNCHANGED) if os.path.exists(sem_p) else None
        dets = model.detect(bgr)
        tp, used = match(dets, gts)
        for g, u in zip(gts, used):
            # 'clean' subset: upright, not cut by the image border, silhouette fills >= 35 % of its box
            x1, y1, x2, y2 = (int(round(v)) for v in g)
            fill = float((sem[y1:y2, x1:x2] == 5).mean()) if sem is not None and x2 > x1 and y2 > y1 else 0.0
            trunc = x1 <= 1 or y1 <= 1 or x2 >= W - 1 or y2 >= H - 1
            if not trunc and fill >= 0.35 and (y2 - y1) >= 40:
                clean[0] += 1
                clean[1] += int(u)
            if trunc:
                trunc_n[0] += 1
                trunc_n[1] += int(u)
        tp_d += sum(tp)
        fp_d += len(tp) - sum(tp)
        if len(tp) - sum(tp):
            fps_list.append(stem)
        for g, u, m in zip(gts, used, pm):
            hb = g[3] - g[1]
            for b in H_BINS:
                if b[0] <= hb < b[1]:
                    by['height_px'][bin_name(b)][0] += 1
                    by['height_px'][bin_name(b)][1] += int(u)
            if m.get('distance_m') is not None:
                for b in D_BINS:
                    if b[0] <= m['distance_m'] < b[1]:
                        by['distance_m'][bin_name(b)][0] += 1
                        by['distance_m'][bin_name(b)][1] += int(u)
            if m.get('visible_frac') is not None:
                for b in V_BINS:
                    if b[0] <= m['visible_frac'] < b[1]:
                        by['visible_frac'][bin_name(b)][0] += 1
                        by['visible_frac'][bin_name(b)][1] += int(u)
            if not u:
                misses.append(stem)
        low = model.detect(bgr, conf=0.01)
        ltp, _ = match(low, gts)
        sweep_scores += [d[4] for d in low]
        sweep_tp += ltp
    res = {
        'images': len(imgs), 'gt_persons': n_gt, 'min_box_height_px': min_h,
        'conf': model.conf, 'iou': 0.5,
        'tp': tp_d, 'fp': fp_d, 'fn': n_gt - tp_d,
        'precision': round(tp_d / max(tp_d + fp_d, 1), 4), 'recall': round(tp_d / max(n_gt, 1), 4),
        'ap50': round(ap50(sweep_scores, sweep_tp, n_gt), 4),
        'recall_by': {k: {b: {'n': v[0], 'recall': round(v[1] / v[0], 3) if v[0] else None}
                          for b, v in d.items()} for k, d in by.items()},
        'recall_clean_subset': {'n': clean[0], 'recall': round(clean[1] / clean[0], 4) if clean[0] else None,
                                'definition': 'not truncated, >= 40 px tall, silhouette fills >= 35 % of the box'},
        'recall_truncated': {'n': trunc_n[0], 'recall': round(trunc_n[1] / trunc_n[0], 4) if trunc_n[0] else None},
        'frames_with_misses': len(set(misses)), 'frames_with_fp': len(fps_list),
        'seconds': round(time.time() - t0, 1), 'provider': model.provider,
    }
    res['model'] = model.path
    res['image_list'] = image_list
    with open(os.path.join(out, f'{tag}_eval.json'), 'w') as f:
        json.dump(res, f, indent=1)
    return res


def eval_real(model, folder, out, grid_n=24, tag='real'):
    files = sorted(glob.glob(os.path.join(folder, '*.png')))
    counts, with_det, scores = [], [], []
    for f in files:
        bgr = cv2.imread(f)
        d = model.detect(bgr)
        counts.append(len(d))
        scores += [x[4] for x in d]
        if d:
            with_det.append((f, d))
    res = {'frames': len(files), 'frames_with_person': len(with_det),
           'detections': int(sum(counts)), 'max_per_frame': int(max(counts) if counts else 0),
           'score_median': round(float(np.median(scores)), 3) if scores else None,
           'files_with_person': [os.path.basename(f) for f, _ in with_det]}
    tiles = []
    for f, d in with_det[:grid_n]:
        im = cv2.imread(f)
        for (x1, y1, x2, y2, s, _) in d:
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, f'{s:.2f}', (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        tiles.append(cv2.resize(im, (320, 180)))
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        cv2.imwrite(os.path.join(out, f'{tag}_detections_grid.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(os.path.join(out, f'{tag}_eval.json'), 'w') as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=r'E:\twin\assets\yolo11n_384x640.onnx')
    ap.add_argument('--synthetic')
    ap.add_argument('--real')
    ap.add_argument('--out', default=r'E:\twin\eval')
    ap.add_argument('--conf', type=float, default=0.45)
    ap.add_argument('--list', help='evaluate only the images listed in this file (e.g. splits/val.txt)')
    ap.add_argument('--tag', default='', help='prefix for the output json names')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = YoloOnnx(args.model, conf=args.conf)
    print('provider', model.provider, 'input', model.in_w, 'x', model.in_h, model.dtype.__name__)
    if args.synthetic:
        r = eval_synthetic(model, args.synthetic, args.out, image_list=args.list, tag=(args.tag or '') + 'synthetic')
        print('SYNTHETIC', json.dumps({k: v for k, v in r.items() if k != 'recall_by'}))
        print(json.dumps(r['recall_by'], indent=1))
    if args.real:
        r = eval_real(model, args.real, args.out, tag=(args.tag or '') + 'real')
        print('REAL', json.dumps({k: v for k, v in r.items() if k != 'files_with_person'}))


if __name__ == '__main__':
    main()
