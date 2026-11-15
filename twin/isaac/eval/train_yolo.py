"""Fine-tune YOLO11 (n or s) on a synthetic set (person + railing), export a 384x640 ONNX.

  E:\\twin\\eval-env\\Scripts\\python.exe train_yolo.py --data E:\\twin\\datasets\\v1 --epochs 5
  E:\\twin\\eval-env\\Scripts\\python.exe train_yolo.py --data E:\\twin\\datasets\\v2 --model yolo11s.pt \\
      --name yolo11s_synth_v2 --epochs 40 --batch 64 --cache ram

Split (deterministic, 10 % val): v1 = random frames; v2-style names (w<worker>_<index>) are split in blocks of
100 consecutive frames per worker so the 4 cameras of one capture step never straddle train/val.
Trains from the COCO-pretrained weights, then exports ONNX with the Jetson engine's input (1x3x384x640).
Nothing is deployed.
"""
import argparse
import glob
import os
import random
import re
import shutil

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'E:\twin\datasets\v1')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--project', default=r'E:\twin\runs\train')
    ap.add_argument('--name', default='yolo11n_synth_v1')
    ap.add_argument('--model', default='yolo11n.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cache', default='', help="'' (off), 'ram' or 'disk'")
    ap.add_argument('--patience', type=int, default=100)
    a = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(a.data, 'images', '*.png')))
    rnd = random.Random(0)
    blk = [re.match(r'w(\d+)_(\d+)\.png$', os.path.basename(p)) for p in imgs]
    if imgs and all(blk):
        groups = sorted({(int(m.group(1)), int(m.group(2)) // 100) for m in blk})
        vg = set(rnd.sample(groups, max(1, len(groups) // 10)))
        val = {p for p, m in zip(imgs, blk) if (int(m.group(1)), int(m.group(2)) // 100) in vg}
    else:
        val = set(rnd.sample(imgs, max(1, len(imgs) // 10)))
    split_dir = os.path.join(a.data, 'splits')
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(p for p in imgs if p not in val) + '\n')
    with open(os.path.join(split_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(sorted(val)) + '\n')
    yaml_path = os.path.join(split_dir, 'data_split.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {a.data}\ntrain: {os.path.join(split_dir, 'train.txt')}\nval: {os.path.join(split_dir, 'val.txt')}\n"
                "names:\n  0: person\n  1: railing\n")

    os.makedirs(a.project, exist_ok=True)
    model = YOLO(a.model)
    model.train(data=yaml_path, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, rect=False, project=a.project,
                name=a.name, exist_ok=True, workers=a.workers, seed=0, plots=True, patience=a.patience,
                cache=(a.cache or False),
                # camera-gap augmentations (see README "sim vs real"): colour cast / exposure / blur-ish scale jitter
                hsv_h=0.03, hsv_s=0.6, hsv_v=0.5, degrees=2.0, translate=0.1, scale=0.4, fliplr=0.5, mosaic=1.0)
    metrics = model.val(data=yaml_path, imgsz=a.imgsz, batch=a.batch, project=a.project, name=a.name + '_val', exist_ok=True)
    print('VAL', {k: round(float(v), 4) for k, v in metrics.results_dict.items()})
    best = os.path.join(a.project, a.name, 'weights', 'best.pt')
    onnx = YOLO(best).export(format='onnx', imgsz=(384, 640), opset=17, simplify=True, dynamic=False)
    dst = os.path.join(a.project, a.name, f'{os.path.splitext(os.path.basename(a.model))[0]}_synth_384x640.onnx')
    shutil.copyfile(onnx, dst)
    print('EXPORTED', dst)


if __name__ == '__main__':      # Windows: dataloader workers re-import this module
    main()
