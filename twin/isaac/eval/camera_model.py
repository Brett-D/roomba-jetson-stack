"""D1000-like camera model for synthetic frames (closes the sim-vs-real gap measured by sim_vs_real.py).

Real left frames (fresh_map8) vs raw sim: ~15x lower Laplacian variance (soft optics, 0.5x
downscale, rectification resampling), lower contrast, lifted blacks (no crushed shadows), green
white balance (B:G:R ~ 0.55:1:0.86). apply() reproduces those with randomised strength:

  python camera_model.py --in E:\\twin\\datasets\\v1\\images --out E:\\twin\\datasets\\v1\\images_d1000
"""
import argparse
import glob
import os

import cv2
import numpy as np


def apply(bgr, rng=None, strength=1.0):
    rng = rng or np.random.default_rng()
    img = bgr.astype(np.float32)
    # optics + resampling softness, occasional motion blur while turning
    sigma = rng.uniform(1.0, 1.8) * strength
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
    gains = np.array([rng.uniform(0.6, 0.8), rng.uniform(1.05, 1.25), rng.uniform(0.85, 1.0)], np.float32)
    img *= gains ** strength
    # lower contrast around the mean and lift the black level
    m = img.mean()
    img = m + (img - m) * rng.uniform(0.7, 0.85) + rng.uniform(6, 14) * strength
    # sensor noise (signal-dependent) before quantisation
    img += rng.normal(0, 1.0, img.shape) * np.sqrt(np.clip(img, 1, 255) / 40.0)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for f in sorted(glob.glob(os.path.join(a.inp, '*.png'))):
        cv2.imwrite(os.path.join(a.out, os.path.basename(f)), apply(cv2.imread(f), rng))


if __name__ == '__main__':
    main()
