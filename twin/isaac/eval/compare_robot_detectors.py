"""Stock COCO detector vs the fine-tuned person / railing detectors, as the robot's person_detector
runs them (yolo_onnx.py: letterbox 384x640, conf 0.45, NMS 0.5; the classes each robot profile
publishes, see src/roomba_bringup/config/person_models.yaml).

  python compare_robot_detectors.py --out E:\\twin\\eval\\hookup

- Synthetic people (v2 and v2-AE val, never trained on): people found (recall), correct detections
  (precision) of the people outputs (person boxes >= 10 px; for the stock profile a cat / dog box on
  a person counts as found, as the explorer treats every people detection alike).
- Real frames without people (every 2 s), where every people detection is a false alarm:
  fresh_map8 held-out = its last 30 % (312 frames, never trained on); fresh_map8 trained-on = its first
  727, the background images of the fine-tuned models' hard-negative round (not a fair test);
  fresh_map7 = an independent recording of the same house earlier that day (643 frames, never trained
  on; extracted with every 30th left colour frame like fresh_map8).
  Each false alarm is saved as a crop (crops/<model>_<frame>_<k>.jpg) for labelling.
- Railing detections (fine-tuned models only; published as objects, never as people) are counted.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from eval_yolo import load_labels
from yolo_onnx import YoloOnnx, match

MODELS = {
    # name: (onnx, class ids the robot runs, people class ids, railing class id)
    'stock_yolo11n_coco': (r'E:\twin\assets\yolo11n_384x640.onnx', (0, 15, 16), (0, 15, 16), None),
    'ft_yolo11s_v2aeneg': (r'E:\twin\assets\models\yolo11s_person_railing_v2aeneg_384x640.onnx', (0, 1), (0,), 1),
    'ft_yolo11n_v2aeneg': (r'E:\twin\assets\models\yolo11n_person_railing_v2aeneg_384x640.onnx', (0, 1), (0,), 1),
}


def synthetic(model, people, root, image_list, min_h=10):
    imgs = [ln.strip() for ln in open(image_list) if ln.strip()]
    tp = fp = n_gt = other = 0
    for f in imgs:
        bgr = cv2.imread(f)
        h, w = bgr.shape[:2]
        stem = os.path.splitext(os.path.basename(f))[0]
        gts = [g for g in load_labels(os.path.join(root, 'labels', stem + '.txt'), w, h) if g[3] - g[1] >= min_h]
        dets = model.detect(bgr)
        pd = [d for d in dets if d[5] in people]
        other += len(dets) - len(pd)
        t, _ = match(pd, gts)
        tp += sum(t)
        fp += len(t) - sum(t)
        n_gt += len(gts)
    return {'images': len(imgs), 'people': n_gt, 'found': tp, 'false': fp,
            'recall': round(tp / max(n_gt, 1), 4), 'precision': round(tp / max(tp + fp, 1), 4),
            'other_class_dets': other}


def real(model, people, railing, files, name, crops):
    alarm_frames, alarms, rail_frames, rails, near = [], 0, 0, 0, 0
    for f in files:
        bgr = cv2.imread(f)
        dets = model.detect(bgr)
        pd = [d for d in dets if d[5] in people]
        rd = [d for d in dets if railing is not None and d[5] == railing]
        low = [d for d in model.detect(bgr, conf=0.25) if d[5] in people]
        near += bool(low)
        rails += len(rd)
        rail_frames += bool(rd)
        if pd:
            alarms += len(pd)
            stem = os.path.splitext(os.path.basename(f))[0]
            alarm_frames.append({'frame': stem, 'dets': [[int(v) for v in d[:4]] + [round(d[4], 3), d[5]] for d in pd]})
            for k, (x1, y1, x2, y2, s, c) in enumerate(pd):
                im = bgr.copy()
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(im, f'{c}:{s:.2f}', (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imwrite(os.path.join(crops, f'{name}_{stem}_{k}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {'frames': len(files), 'false_alarm_frames': len(alarm_frames), 'false_alarms': alarms,
            'frames_with_people_score_over_0.25': near, 'railing_frames': rail_frames, 'railing_dets': rails,
            'alarms': alarm_frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'E:\twin\eval\hookup')
    ap.add_argument('--real', default=r'E:\twin\real_frames\left')
    ap.add_argument('--heldout', default=r'E:\twin\datasets\v2ae\splits\real_heldout.txt')
    ap.add_argument('--independent', default=r'E:\twin\real_frames_fm7\left')
    ap.add_argument('--models', nargs='*', default=list(MODELS))
    args = ap.parse_args()
    crops = os.path.join(args.out, 'crops')
    os.makedirs(crops, exist_ok=True)
    every = sorted(glob.glob(os.path.join(args.real, '*.png')))
    held = {os.path.normcase(ln.strip()) for ln in open(args.heldout) if ln.strip()}
    sets = {'real_heldout': [f for f in every if os.path.normcase(f) in held],
            'real_trained_on': [f for f in every if os.path.normcase(f) not in held],
            'real_fm7': sorted(glob.glob(os.path.join(args.independent, '*.png')))}
    results = {}
    for name in args.models:
        path, classes, people, railing = MODELS[name]
        model = YoloOnnx(path, conf=0.45, nms=0.5, classes=classes)
        r = {'onnx': path, 'classes': list(classes), 'people_classes': list(people)}
        for tag, root in (('v2ae_val', r'E:\twin\datasets\v2ae'), ('v2_val', r'E:\twin\datasets\v2')):
            r[tag] = synthetic(model, people, root, os.path.join(root, 'splits', 'val.txt'))
        for tag, files in sets.items():
            r[tag] = real(model, people, railing, files, f'{name}_{tag}', crops)
        results[name] = r
        print(name, json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'alarms'} if isinstance(v, dict) else v)
                                for k, v in r.items()}), flush=True)
    with open(os.path.join(args.out, 'compare_robot_detectors.json'), 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
