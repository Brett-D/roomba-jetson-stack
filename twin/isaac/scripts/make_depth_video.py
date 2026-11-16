"""3-panel depth heat-map clip: follow camera (left, 1280x1080 centre crop) + the robot's left camera + its depth image
as a heat map (near = red, far = blue, outside the D1000's 0.3-7 m black) with a range legend.

  python make_depth_video.py E:\\twin\\videos\\raw\\depthclip_01 --label depth_heatmap --seconds 25

Inputs: <base>_follow.mp4 / _robotcam.mp4 / _frames.jsonl (run_closed_loop.py recording, 30 fps of sim time) and
<base>_depth/<stamp_ns>.png (save_depth.py: the 16UC1 depth the stack received, 15 Hz of sim time). Each video
frame shows the newest depth image not newer than it.
"""
import argparse
import bisect
import glob
import json
import os
import subprocess

import cv2
import numpy as np

W, H = 1920, 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX
DMIN, DMAX = 0.3, 7.0


def heat(mm):
    d = mm.astype(np.float32) / 1000.0
    ok = (d >= DMIN) & (d <= DMAX)
    v = np.clip((DMAX - d) / (DMAX - DMIN), 0, 1)          # near -> 1 -> red end of turbo
    col = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[~ok] = 0
    return col


def legend(w, h):
    img = np.zeros((h, w, 3), np.uint8)
    bar_w, x0, y0 = w - 120, 60, 70
    ramp = np.linspace(DMIN, DMAX, bar_w)
    row = heat((ramp * 1000).astype(np.uint16)[None, :])
    img[y0:y0 + 34, x0:x0 + bar_w] = np.repeat(row, 34, axis=0)
    for m in (0.3, 1, 2, 3, 4, 5, 6, 7):
        x = x0 + int((m - DMIN) / (DMAX - DMIN) * (bar_w - 1))
        cv2.line(img, (x, y0 + 34), (x, y0 + 44), (255, 255, 255), 1)
        cv2.putText(img, f'{m:g}', (x - 10, y0 + 66), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, 'depth (m)   near = red, far = blue', (x0, y0 - 18), FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0 + 92), (x0 + 34, y0 + 116), (0, 0, 0), -1)
    cv2.rectangle(img, (x0, y0 + 92), (x0 + 34, y0 + 116), (120, 120, 120), 1)
    cv2.putText(img, 'black: outside the D1000 range (0.3-7 m)', (x0 + 48, y0 + 111), FONT, 0.55, (230, 230, 230), 1,
                cv2.LINE_AA)
    return img


class AutoGain:
    """Viewing aid for dark renders: gamma + slowly adapting gain towards a target mean brightness."""

    def __init__(self, target, gamma=0.8):
        self.target, self.gamma, self.g = target, gamma, None
        self.lut = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.float32)

    def __call__(self, img):
        x = self.lut[img]
        g = float(np.clip(self.target / max(float(x.mean()), 1.0), 0.8, 4.0))
        self.g = g if self.g is None else 0.95 * self.g + 0.05 * g
        return np.clip(x * self.g, 0, 255).astype(np.uint8)


class Reader:
    """Frames of an MP4 through an ffmpeg pipe (the Isaac venv's OpenCV has no video I/O)."""

    def __init__(self, path, w, h, start_frame):
        self.w, self.h = w, h
        self.p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-ss', f'{start_frame / 30:.4f}', '-i', path, '-f', 'rawvideo',
                                   '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)

    def read(self):
        b = self.p.stdout.read(self.w * self.h * 3)
        if len(b) < self.w * self.h * 3:
            return False, None
        return True, np.frombuffer(b, np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        self.p.kill()


def label(img, text, y=28):
    cv2.rectangle(img, (0, 0), (img.shape[1], y + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (12, y), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('--label', default='depth_heatmap')
    ap.add_argument('--caption', default='Isaac Sim digital twin: autonomous exploration')
    ap.add_argument('--seconds', type=float, default=25.0)
    ap.add_argument('--start', type=float, default=-1.0, help='start (s into the recording); -1: pick a moving stretch')
    ap.add_argument('--out', default=r'E:\twin\videos\web')
    ap.add_argument('--lift', type=float, default=105.0, help='target mean brightness of the follow view (0 = off)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames = [json.loads(ln) for ln in open(a.base + '_frames.jsonl') if ln.strip()]
    depth = sorted((int(os.path.splitext(os.path.basename(p))[0]) * 1e-9, p) for p in glob.glob(a.base + '_depth/*.png'))
    dts = [t for t, _ in depth]
    n = int(a.seconds * 30)
    xy = np.array([[f['x'], f['y']] for f in frames])
    if a.start < 0:
        # the stretch with the most robot motion (and depth coverage)
        mv = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
        best, i0 = -1, 0
        for i in range(0, max(1, len(frames) - n), 15):
            if not dts or frames[i]['t'] < dts[0]:
                continue
            sc = mv[i:i + n].sum() - 5 * (mv[i:i + n] > 0.2).sum()
            if sc > best:
                best, i0 = sc, i
    else:
        i0 = int(a.start * 30)
    cf = Reader(a.base + '_follow.mp4', W, H, i0)
    cr = Reader(a.base + '_robotcam.mp4', 640, 360, i0)
    leg = legend(640, 360)
    cap_lines = [a.caption, 'Roomba 550 + MYNT EYE D1000, lens 13 cm up', 'depth = what the robot stack receives',
                 '(registered to the left camera, 15 Hz)']
    for k, t in enumerate(cap_lines):
        cv2.putText(leg, t, (60, 230 + 30 * k), FONT, 0.6 if k else 0.68, (255, 255, 255) if k == 0 else (200, 200, 200), 1,
                    cv2.LINE_AA)
    tmp = os.path.join(a.out, a.label + '_master.mp4')
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                            '-r', '30', '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt',
                            'yuv420p', tmp], stdin=subprocess.PIPE)
    cache = {}
    ag_f, ag_r = AutoGain(a.lift), AutoGain(80.0)
    for i in range(i0, min(i0 + n, len(frames))):
        ok1, f = cf.read()
        ok2, r = cr.read()
        if not (ok1 and ok2):
            break
        t = frames[i]['t']
        j = bisect.bisect_right(dts, t) - 1
        if j < 0:
            continue
        if j not in cache:
            cache = {j: heat(cv2.imread(depth[j][1], cv2.IMREAD_UNCHANGED))}
        out = np.zeros((H, W, 3), np.uint8)
        if a.lift > 0:
            f, r = ag_f(f), ag_r(r)
        out[:, :1280] = f[:, 320:1600]
        rr = r.copy()
        label(rr, 'robot camera (D1000 left)')
        out[0:360, 1280:] = rr
        dd = cache[j].copy()
        label(dd, 'robot depth heat map')
        out[360:720, 1280:] = dd
        out[720:, 1280:] = leg
        cv2.line(out, (1279, 0), (1279, H), (40, 40, 40), 2)
        cv2.line(out, (1280, 360), (W, 360), (40, 40, 40), 2)
        cv2.line(out, (1280, 720), (W, 720), (40, 40, 40), 2)
        enc.stdin.write(out.tobytes())
    enc.stdin.close()
    enc.wait()
    cf.close()
    cr.close()
    base = os.path.join(a.out, a.label)
    common = ['-an', '-r', '30', '-pix_fmt', 'yuv420p']
    for crf in (23, 26, 29):
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf),
                        '-profile:v', 'high', '-movflags', '+faststart'] + common + [base + '.mp4'], check=True)
        if os.path.getsize(base + '.mp4') < 9.5e6:
            break
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libvpx-vp9', '-crf', '36', '-b:v', '0',
                    '-row-mt', '1', '-cpu-used', '2'] + common + [base + '.webm'], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(min(8.0, a.seconds / 2)), '-i', base + '.mp4',
                    '-frames:v', '1', '-q:v', '3', base + '.jpg'], check=True)
    os.remove(tmp)
    for ext in ('mp4', 'webm', 'jpg'):
        print(base + '.' + ext, f'{os.path.getsize(base + "." + ext) / 1e6:.2f} MB')
    print('start frame', i0, 'sim t', frames[i0]['t'])


if __name__ == '__main__':
    main()
