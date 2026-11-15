"""YOLO11n ONNX person detector, identical pre/post-processing to the Jetson node
(roomba_perception/person_detector.py): letterbox to 384x640 with grey 114, BGR->RGB, /255,
class 0 only, conf 0.45, OpenCV NMS 0.5, boxes clipped to the image and >= 4 px.
"""
import os

import cv2
import numpy as np
import onnxruntime as ort


class YoloOnnx:
    def __init__(self, path, conf=0.45, nms=0.5, classes=(0,), providers=None):
        # CPU by default (yolo11n at 384x640 is ~10 ms); TWIN_ORT_CUDA=1 tries CUDA (needs matching CUDA/cuDNN DLLs)
        avail = ort.get_available_providers()
        want = ('CUDAExecutionProvider', 'CPUExecutionProvider') if os.environ.get('TWIN_ORT_CUDA') else ('CPUExecutionProvider',)
        providers = providers or [p for p in want if p in avail]
        self.path = path
        self.sess = ort.InferenceSession(path, providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.in_h, self.in_w = int(inp.shape[2]), int(inp.shape[3])
        self.dtype = np.float16 if 'float16' in inp.type else np.float32
        self.conf, self.nms, self.class_ids = conf, nms, list(classes)
        self.provider = self.sess.get_providers()[0]

    def preprocess(self, bgr):
        h, w = bgr.shape[:2]
        r = min(self.in_w / w, self.in_h / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else bgr
        px, py = (self.in_w - nw) // 2, (self.in_h - nh) // 2
        canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = img
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) * (1.0 / 255.0)
        return x.astype(self.dtype), r, px, py

    def raw(self, bgr):
        x, r, px, py = self.preprocess(bgr)
        out = self.sess.run(None, {self.input_name: x})[0][0].astype(np.float32)   # (4 + nc, N)
        return out, r, px, py

    def detect(self, bgr, conf=None):
        """[(x1, y1, x2, y2, score, cls)] in image pixels."""
        conf = self.conf if conf is None else conf
        out, r, px, py = self.raw(bgr)
        scores = out[4:][self.class_ids]
        best = scores.max(axis=0)
        keep = best > conf
        if not np.any(keep):
            return []
        cls = np.array(self.class_ids)[scores[:, keep].argmax(axis=0)]
        cx, cy, w, h = out[0, keep], out[1, keep], out[2, keep], out[3, keep]
        x1 = (cx - w / 2 - px) / r
        y1 = (cy - h / 2 - py) / r
        bw, bh = w / r, h / r
        sc = best[keep]
        boxes = np.stack([x1, y1, bw, bh], axis=1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), sc.tolist(), conf, self.nms)
        H, W = bgr.shape[:2]
        dets = []
        for i in np.array(idx).flatten():
            bx1 = int(np.clip(boxes[i, 0], 0, W - 1))
            by1 = int(np.clip(boxes[i, 1], 0, H - 1))
            bx2 = int(np.clip(boxes[i, 0] + boxes[i, 2], 0, W - 1))
            by2 = int(np.clip(boxes[i, 1] + boxes[i, 3], 0, H - 1))
            if bx2 - bx1 >= 4 and by2 - by1 >= 4:
                dets.append((bx1, by1, bx2, by2, float(sc[i]), int(cls[i])))
        return dets


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(dets, gts, thr=0.5):
    """Greedy (by score) one-to-one matching. Returns (tp flags per det, matched gt flags)."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i][4])
    used = [False] * len(gts)
    tp = [False] * len(dets)
    for i in order:
        best, bj = thr, -1
        for j, g in enumerate(gts):
            if not used[j]:
                v = iou(dets[i], g)
                if v >= best:
                    best, bj = v, j
        if bj >= 0:
            used[bj] = True
            tp[i] = True
    return tp, used
