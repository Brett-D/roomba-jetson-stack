"""Web-ready videos from a run_closed_loop.py recording (follow camera 1920x1080 + robot camera, 30 fps of sim time).

  python make_web_video.py E:\\twin\\videos\\raw\\path_replay_01 --label path_replay --caption "..."

Writes to E:\\twin\\videos\\web\\:
  <label>_hero_loop.mp4/.webm/.jpg   10-15 s loop (end cross-faded into the start), no teleports, people nearby
  <label>_highlights.mp4/.webm/.jpg  60-120 s cut: interesting stretches at 1x, long ones at 2-4x (labelled)
H.264 (yuv420p, +faststart, CRF ~21, no audio) and VP9 WebM; picture-in-picture of the robot's own camera.
"""
import argparse
import json
import math
import os
import subprocess

import numpy as np

FONT = 'C\\:/Windows/Fonts/segoeui.ttf'
FONT_B = 'C\\:/Windows/Fonts/segoeuib.ttf'


def run(cmd):
    print(' '.join(cmd[:6]), '...', flush=True)
    subprocess.run(cmd, check=True)


def probe_duration(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def esc(text):
    return text.replace('\\', '\\\\').replace(':', '\\:').replace("'", "\u2019").replace('%', '\\%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('--label', required=True, help='file name prefix, honest about what it shows (path_replay, autonomous)')
    ap.add_argument('--caption', required=True)
    ap.add_argument('--out', default=r'E:\twin\videos\web')
    ap.add_argument('--hero', type=float, default=12.0)
    ap.add_argument('--target', type=float, default=90.0, help='highlight cut length (s)')
    ap.add_argument('--fade', type=float, default=1.0)
    ap.add_argument('--lift', type=float, default=1.0, help='gamma lift for dark renders (viewing aid, e.g. 1.5)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames = [json.loads(ln) for ln in open(a.base + '_frames.jsonl') if ln.strip()]
    n = len(frames)
    dur = min(probe_duration(a.base + '_follow.mp4'), probe_duration(a.base + '_robotcam.mp4'))
    n = min(n, int(dur * 30))
    frames = frames[:n]
    xy = np.array([[f['x'], f['y']] for f in frames])
    tele = np.array([f['teleports'] for f in frames])
    jump = np.zeros(n, bool)
    jump[1:] = (np.diff(tele) > 0) | (np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.2)
    speed = np.zeros(n)
    speed[15:] = np.linalg.norm(xy[15:] - xy[:-15], axis=1) / 0.5
    # interest per frame: people close to the robot (and in front of the follow camera), moving robot
    interest = np.zeros(n)
    for i, f in enumerate(frames):
        c, s = math.cos(f['yaw']), math.sin(f['yaw'])
        for px, py in f['people']:
            dx, dy = px - f['x'], py - f['y']
            d = math.hypot(dx, dy)
            ahead = dx * c + dy * s
            if d < 4.5 and ahead > -1.0:
                interest[i] += 1.0 / max(d, 0.8)
    moving = speed > 0.05
    print(f'{n} frames ({n / 30:.1f} s), jumps {int(jump.sum())}, moving {moving.mean():.0%}, people-near {np.mean(interest > 0):.0%}')

    # ---------------------------------------------------------------- composite master (PiP + caption)
    master = a.base + '_master.mp4'
    pip_w, pip_h, m = 512, 288, 28
    cap = esc(a.caption)
    lift = f"eq=gamma={a.lift:.2f}," if a.lift != 1.0 else ""
    vf = (f"[1:v]{lift}scale={pip_w}:{pip_h}:flags=lanczos,pad={pip_w + 4}:{pip_h + 4}:2:2:color=white@0.9[pip];"
          f"[0:v]{lift}null[bg];[bg][pip]overlay=W-w-{m}:H-h-{m}[v1];"
          f"[v1]drawbox=x=iw-{pip_w + 4 + m}:y=ih-{pip_h + 4 + m + 30}:w={pip_w + 4}:h=30:color=black@0.55:t=fill,"
          f"drawtext=fontfile='{FONT}':text='Robot camera (MYNT EYE D1000, 13 cm above the floor)':fontcolor=white:"
          f"fontsize=19:x=W-{pip_w + m}:y=H-{pip_h + 4 + m + 24},"
          f"drawbox=x=0:y=0:w=iw:h=58:color=black@0.45:t=fill,"
          f"drawtext=fontfile='{FONT_B}':text='{cap}':fontcolor=white:fontsize=28:x=28:y=14[v]")
    run(['ffmpeg', '-y', '-loglevel', 'error', '-i', a.base + '_follow.mp4', '-i', a.base + '_robotcam.mp4',
         '-filter_complex', vf, '-map', '[v]', '-frames:v', str(n), '-c:v', 'libx264', '-preset', 'medium', '-crf', '15',
         '-pix_fmt', 'yuv420p', master])

    # ---------------------------------------------------------------- hero loop
    L = int(round((a.hero + a.fade) * 30))
    best, best_i = -1e9, 0
    for i in range(0, n - L, 15):
        if jump[i:i + L].any() or not moving[i:i + L].mean() > 0.7:
            continue
        sc = interest[i:i + L].sum() + 2.0 * moving[i:i + L].mean()
        if sc > best:
            best, best_i = sc, i
    s0 = best_i / 30
    F = a.fade
    hero = os.path.join(a.out, f'{a.label}_hero_loop')
    # loop: the last F s cross-fade into the first F s, so the clip ends where it starts
    xf = (f"[0:v]trim=start={s0:.3f}:duration={a.hero + F:.3f},setpts=PTS-STARTPTS,split[a][b];"
          f"[a]trim=start=0:duration={F},setpts=PTS-STARTPTS[head];"
          f"[b]trim=start={F}:duration={a.hero},setpts=PTS-STARTPTS[body];"
          f"[body]split[body1][body2];[body1]trim=start=0:duration={a.hero - F},setpts=PTS-STARTPTS[main];"
          f"[body2]trim=start={a.hero - F}:duration={F},setpts=PTS-STARTPTS[tail];"
          f"[tail][head]xfade=transition=fade:duration={F}:offset=0[blend];[blend][main]concat=n=2:v=1[v]")
    enc_h264(master, hero + '.mp4', xf, crf=23, maxrate='5M')
    if os.path.getsize(hero + '.mp4') > 8e6:
        enc_h264(master, hero + '.mp4', xf, crf=26, maxrate='4M')
    enc_webm(master, hero + '.webm', xf, crf=36)
    poster(hero + '.mp4', hero + '.jpg', a.hero / 2)

    # ---------------------------------------------------------------- highlight cut
    segs = plan_cut(n, jump, interest, moving, a.target)
    parts, labels = [], []
    for k, (i0, i1, sp) in enumerate(segs):
        dur_s = (i1 - i0) / 30
        lab = '' if sp == 1 else (f"drawbox=x=iw-190:y=78:w=160:h=46:color=black@0.55:t=fill,"
                                  f"drawtext=fontfile='{FONT_B}':text='{sp}x speed':fontcolor=white:fontsize=30:x=W-176:y=84,")
        parts.append(f"[0:v]trim=start={i0 / 30:.3f}:duration={dur_s:.3f},setpts=(PTS-STARTPTS)/{sp},fps=30,{lab}"
                     f"format=yuv420p[s{k}]")
        labels.append(f'[s{k}]')
    fc = ';'.join(parts) + ';' + ''.join(labels) + f'concat=n={len(segs)}:v=1[c];[c]fade=t=in:st=0:d=0.6[v]'
    cut = os.path.join(a.out, f'{a.label}_highlights')
    enc_h264(master, cut + '.mp4', fc, crf=22, maxrate='6M')
    if os.path.getsize(cut + '.mp4') > 40e6:
        enc_h264(master, cut + '.mp4', fc, crf=25, maxrate='4M')
    enc_webm(master, cut + '.webm', fc, crf=36)
    poster(cut + '.mp4', cut + '.jpg', None, interest_time=segs)
    for p in (hero + '.mp4', hero + '.webm', cut + '.mp4', cut + '.webm'):
        print(f'{p}: {probe_duration(p):.1f} s, {os.path.getsize(p) / 1e6:.1f} MB')
    with open(os.path.join(a.out, f'{a.label}_segments.json'), 'w') as f:
        json.dump({'source': a.base, 'hero_start_s': s0, 'cut_segments': [{'start_s': i0 / 30, 'end_s': i1 / 30, 'speed': sp}
                                                                        for i0, i1, sp in segs]}, f, indent=1)


def plan_cut(n, jump, interest, moving, target):
    """Split at jumps (teleports); stretches with people nearby play at 1x, the rest at 2-4x, until ~target s."""
    bounds = [0] + [i for i in np.nonzero(jump)[0]] + [n]
    segs = []
    for a0, a1 in zip(bounds[:-1], bounds[1:]):
        a0 += 15                     # skip 0.5 s after a jump
        if a1 - a0 < 60:
            continue
        # sub-split into 1x (people near) and fast parts (nobody around)
        busy = interest[a0:a1] > 0.15
        i = a0
        while i < a1:
            j = i
            while j < a1 and busy[j - a0] == busy[i - a0]:
                j += 1
            if j - i >= 45 and moving[i:j].mean() > 0.3:
                segs.append([i, j, 1 if busy[i - a0] else 4])
            i = j
    if not segs:
        segs = [[0, n, 2]]
    total = sum((j - i) / 30 / sp for i, j, sp in segs)
    # too long: speed up the 1x parts to 2x, then drop the least interesting fast parts
    if total > target:
        for s_ in segs:
            if s_[2] == 1 and (s_[1] - s_[0]) / 30 > 8:
                s_[2] = 2
        total = sum((j - i) / 30 / sp for i, j, sp in segs)
    while total > target and len(segs) > 2:
        k = min(range(len(segs)), key=lambda q: interest[segs[q][0]:segs[q][1]].mean() + (1 if segs[q][2] == 1 else 0))
        total -= (segs[k][1] - segs[k][0]) / 30 / segs[k][2]
        segs.pop(k)
    return [tuple(s_) for s_ in segs]


def enc_h264(src, dst, fc, crf, maxrate):
    run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-filter_complex', fc, '-map', '[v]', '-an', '-c:v', 'libx264',
         '-preset', 'slow', '-crf', str(crf), '-maxrate', maxrate, '-bufsize', '12M', '-pix_fmt', 'yuv420p',
         '-profile:v', 'high', '-movflags', '+faststart', '-r', '30', dst])


def enc_webm(src, dst, fc, crf):
    run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-filter_complex', fc, '-map', '[v]', '-an', '-c:v', 'libvpx-vp9',
         '-crf', str(crf), '-b:v', '0', '-row-mt', '1', '-deadline', 'good', '-cpu-used', '2', '-pix_fmt', 'yuv420p',
         '-r', '30', dst])


def poster(video, jpg, t, interest_time=None):
    if t is None:
        t = min(5.0, probe_duration(video) / 3)
    run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{t:.2f}', '-i', video, '-frames:v', '1', '-q:v', '3', jpg])


if __name__ == '__main__':
    main()
