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
    ap.add_argument('--resume', action='store_true', help='continue an interrupted run from its last.pt')
    ap.add_argument('--negatives', default=None,
                    help='folder of real frames without people (e.g. E:\\twin\\real_frames\\left): the first --neg-frac '
                         '(by time) are added to training as background images, the rest are held out (real_heldout.txt)')
    ap.add_argument('--neg-frac', type=float, default=0.7)
    ap.add_argument('--lr0', type=float, default=0.01)
    ap.add_argument('--warmup-epochs', type=float, default=3.0)
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
    def write_if_changed(path, text):      # parallel runs on the same dataset share the split files
        if not os.path.exists(path) or open(path).read() != text:
            with open(path, 'w') as f:
                f.write(text)
    write_if_changed(os.path.join(split_dir, 'train.txt'), '\n'.join(p for p in imgs if p not in val) + '\n')
    write_if_changed(os.path.join(split_dir, 'val.txt'), '\n'.join(sorted(val)) + '\n')
    yaml_path = os.path.join(split_dir, 'data_split.yaml')
    train_txt = os.path.join(split_dir, 'train.txt')
    if a.negatives:
        neg = sorted(glob.glob(os.path.join(a.negatives, '*.png')))
        k = int(len(neg) * a.neg_frac)
        write_if_changed(os.path.join(split_dir, 'real_negatives_train.txt'), '\n'.join(neg[:k]) + '\n')
        write_if_changed(os.path.join(split_dir, 'real_heldout.txt'), '\n'.join(neg[k:]) + '\n')
        train_txt = os.path.join(split_dir, 'train_plus_real_neg.txt')
        write_if_changed(train_txt, '\n'.join([p for p in imgs if p not in val] + neg[:k]) + '\n')
        yaml_path = os.path.join(split_dir, 'data_split_neg.yaml')
    write_if_changed(yaml_path, f"path: {a.data}\ntrain: {train_txt}\n"
                                f"val: {os.path.join(split_dir, 'val.txt')}\nnames:\n  0: person\n  1: railing\n")

    os.makedirs(a.project, exist_ok=True)
    last = os.path.join(a.project, a.name, 'weights', 'last.pt')
    if a.resume and os.path.exists(last):
        model = YOLO(last)
        model.train(resume=True)
    else:
        model = YOLO(a.model)
        model.train(data=yaml_path, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, rect=False, project=a.project,
                    name=a.name, exist_ok=True, workers=a.workers, seed=0, plots=True, patience=a.patience,
                    cache=(a.cache or False), lr0=a.lr0, warmup_epochs=a.warmup_epochs,
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
