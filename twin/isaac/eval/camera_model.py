"""D1000-like camera model for synthetic frames (closes the sim-vs-real gap measured by sim_vs_real.py).

Real left frames (fresh_map8) vs raw sim: ~15x lower Laplacian variance (soft optics, 0.5x
downscale, rectification resampling), lower contrast, lifted blacks (no crushed shadows), green
white balance (B:G:R ~ 0.55:1:0.86), and auto-exposure: real frames sit at a mean brightness of
~62 (quartiles 35 / 85) whatever the lighting, while the remodelled house renders at ~20.
apply() reproduces those with randomised strength:

  python camera_model.py --in E:\\twin\\datasets\\v1\\images --out E:\\twin\\datasets\\v1\\images_d1000
"""
import argparse
import glob
import os

import cv2
import numpy as np


def apply(bgr, rng=None, strength=1.0, auto_exposure=True):
    rng = rng or np.random.default_rng()
    img = bgr.astype(np.float32)
    gain = 1.0
    if auto_exposure:
        # AE towards the real brightness distribution (mean 62, quartiles 35 / 85), limited gain range
        target = float(np.clip(rng.normal(62, 22), 25, 110))
        gain = float(np.clip(target / max(float(img.mean()), 1.0), 0.6, 6.0))
        img *= gain
    # optics + resampling softness, occasional motion blur while turning
    sigma = rng.uniform(1.4, 2.2) * strength
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    if rng.random() < 0.3:
        k = int(rng.integers(3, 9))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        ang = rng.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
        ker = cv2.warpAffine(ker, M, (k, k))
        ker /= ker.sum()
        img = cv2.filter2D(img, -1, ker)
    # white balance: green-tinted like the D1000 (per-channel gains on B, G, R)
    gains = np.array([rng.uniform(0.6, 0.78), rng.uniform(1.05, 1.25), rng.uniform(0.64, 0.78)], np.float32)
    img *= gains ** strength
    # lower contrast around the mean and lift the black level
    m = 0.5 * (img.reshape(-1, 3).mean(0) + img.mean()) if auto_exposure else img.mean()    # keep part of the cast
    img = m + (img - m) * (rng.uniform(0.45, 0.65) if auto_exposure else rng.uniform(0.7, 0.85)) + rng.uniform(6, 14) * strength
    # sensor noise (signal-dependent, grows with the AE gain) before quantisation
    img += rng.normal(0, 0.6 if auto_exposure else 1.0, img.shape) * np.sqrt(np.clip(img, 1, 255) / 40.0) * min(1.0 + 0.1 * (gain - 1.0), 1.4)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-ae', action='store_true')
    ap.add_argument('--procs', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.inp, '*.png')) + glob.glob(os.path.join(a.inp, '*.jpg')))
    from concurrent.futures import ProcessPoolExecutor
    chunks = [(files[i::a.procs], a.out, a.seed + i, not a.no_ae) for i in range(a.procs)]
    with ProcessPoolExecutor(a.procs) as ex:
        print('WROTE', sum(ex.map(_run, chunks)))


def _run(job):
    files, out, seed, ae = job
    rng = np.random.default_rng(seed)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0] + '.png'
        cv2.imwrite(os.path.join(out, name), apply(cv2.imread(f), rng, auto_exposure=ae))
    return len(files)


if __name__ == '__main__':
    main()
