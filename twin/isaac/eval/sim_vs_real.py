"""Compare image statistics of real D1000 frames vs synthetic frames.

  python sim_vs_real.py --real E:\\twin\\real_frames\\left --sim E:\\twin\\datasets\\v1\\images --out E:\\twin\\eval

Per image: mean brightness (Y), contrast (std Y), saturation (HSV S mean), noise sigma (Immerkaer's
fast estimator on Y), sharpness (variance of Laplacian), clipped highlights / crushed shadows,
colour cast (mean B, G, R). Prints medians and IQRs for both sets and writes stats.json + a
histogram figure.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def noise_sigma(gray):
    """Immerkaer 1996: sigma = sqrt(pi/2) / (6 (W-2)(H-2)) * sum |I * N|."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = gray.shape
    conv = cv2.filter2D(gray.astype(np.float32), -1, k)
    return float(np.sqrt(np.pi / 2) * np.abs(conv[1:-1, 1:-1]).sum() / (6.0 * (w - 2) * (h - 2)))


def stats(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return {
        'brightness': float(y.mean()),
        'contrast': float(y.std()),
        'saturation': float(hsv[:, :, 1].mean()),
        'noise_sigma': noise_sigma(y),
        'sharpness_lapvar': float(cv2.Laplacian(y, cv2.CV_64F).var()),
        'clipped_hi_pct': float((y >= 250).mean() * 100),
        'crushed_lo_pct': float((y <= 5).mean() * 100),
        'mean_b': float(bgr[:, :, 0].mean()), 'mean_g': float(bgr[:, :, 1].mean()),
        'mean_r': float(bgr[:, :, 2].mean()),
    }


def summarize(rows):
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        out[k] = {'median': round(float(np.median(v)), 3), 'p25': round(float(np.percentile(v, 25)), 3),
                  'p75': round(float(np.percentile(v, 75)), 3), 'mean': round(float(v.mean()), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--sim', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max', type=int, default=1500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = {}
    rows_by = {}
    for name, d in (('real', args.real), ('sim', args.sim)):
        files = sorted(glob.glob(os.path.join(d, '*.png')) + glob.glob(os.path.join(d, '*.jpg')))[:args.max]
        rows = [r for r in (stats(f) for f in files) if r]
        rows_by[name] = rows
        res[name] = {'n': len(rows), **summarize(rows)}
    res['ratio_sim_over_real_median'] = {k: round(res['sim'][k]['median'] / res['real'][k]['median'], 3)
                                         for k in rows_by['real'][0] if res['real'][k]['median']}
    with open(os.path.join(args.out, 'sim_vs_real_stats.json'), 'w') as f:
        json.dump(res, f, indent=1)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        keys = ['brightness', 'contrast', 'saturation', 'noise_sigma', 'sharpness_lapvar', 'clipped_hi_pct']
        fig, axs = plt.subplots(2, 3, figsize=(13, 7))
        for ax, k in zip(axs.flat, keys):
            for name, c in (('real', 'tab:blue'), ('sim', 'tab:orange')):
                v = np.array([r[k] for r in rows_by[name]])
                if k == 'sharpness_lapvar':
                    v = np.log10(v + 1)
                ax.hist(v, bins=40, alpha=0.55, color=c, label=name, density=True)
            ax.set_title(k if k != 'sharpness_lapvar' else 'log10 laplacian var (sharpness)')
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'sim_vs_real_hist.png'), dpi=90)
    except ImportError:
        pass
    for k in rows_by['real'][0]:
        print(f"{k:18s} real {res['real'][k]['median']:9.3f}  sim {res['sim'][k]['median']:9.3f}")


if __name__ == '__main__':
    main()
