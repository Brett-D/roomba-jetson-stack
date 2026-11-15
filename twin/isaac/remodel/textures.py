"""Procedural, tileable PBR-ish textures for the remodelled house (no downloads needed).

  python textures.py --out E:\\twin\\assets\\remodel\\textures
Writes <name>.png (sRGB albedo, 1024^2, tileable). One texture tile covers TEX_TILE_M metres
(see remodel_lib.TEX_TILE_M), set via the UV scale in the house builder.
"""
import argparse
import os

import numpy as np
from PIL import Image

N = 1024


def periodic_noise(rng, n, sigma_px, aniso=(1.0, 1.0)):
    """Tileable noise: white noise low-passed in the Fourier domain."""
    w = rng.standard_normal((n, n))
    fy = np.fft.fftfreq(n)[:, None] * aniso[1]
    fx = np.fft.fftfreq(n)[None, :] * aniso[0]
    f = np.exp(-(fx ** 2 + fy ** 2) * (2 * np.pi * sigma_px) ** 2 / 2)
    out = np.real(np.fft.ifft2(np.fft.fft2(w) * f))
    return (out - out.mean()) / (out.std() + 1e-9)


def to_img(rgb):
    return Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))


def carpet(rng, base):
    n = 0.5 * periodic_noise(rng, N, 1.0) + 0.35 * periodic_noise(rng, N, 6) + 0.25 * periodic_noise(rng, N, 40)
    return np.clip(np.array(base)[None, None] * (1 + 0.09 * n[..., None]), 0, 1)


def wood(rng, base, planks=8, length_px=640):
    img = np.zeros((N, N, 3))
    ph = N // planks
    grain = periodic_noise(rng, N, 1.5, aniso=(0.05, 1.0))
    ring = periodic_noise(rng, N, 12, aniso=(0.08, 1.0))
    for p in range(planks):
        tone = 1 + 0.12 * rng.standard_normal()
        hue = np.array(base) * tone * (1 + 0.04 * rng.standard_normal(3))
        sl = slice(p * ph, (p + 1) * ph)
        img[sl] = hue[None, None]
        off = rng.integers(0, N)
        for x in range(off % length_px, N + length_px, length_px):      # end joints (tileable)
            img[sl, x % N:(x % N) + 3] *= 0.75
        img[p * ph:p * ph + 3] *= 0.7                                    # seam
    img *= (1 + 0.07 * grain[..., None] + 0.1 * ring[..., None])
    return np.clip(img, 0, 1)


def tile(rng, base, grout, tiles=4):
    n = periodic_noise(rng, N, 3)
    img = np.array(base)[None, None] * (1 + 0.03 * n[..., None]) * np.ones((N, N, 1))
    t = N // tiles
    for k in range(tiles):
        img[k * t:k * t + 6] = grout
        img[:, k * t:k * t + 6] = grout
    for i in range(tiles):
        for j in range(tiles):
            img[i * t + 6:(i + 1) * t, j * t + 6:(j + 1) * t] *= 1 + 0.03 * rng.standard_normal()
    return np.clip(img, 0, 1)


def paint(rng, base):
    n = 0.6 * periodic_noise(rng, N, 2) + 0.4 * periodic_noise(rng, N, 30)
    return np.clip(np.array(base)[None, None] * (1 + 0.025 * n[..., None]), 0, 1)


def stone(rng, base):
    n = 0.5 * periodic_noise(rng, N, 2) + 0.5 * periodic_noise(rng, N, 25)
    return np.clip(np.array(base)[None, None] * (1 + 0.12 * n[..., None]), 0, 1)


TEXTURES = {
    'carpet_beige': lambda r: carpet(r, (0.62, 0.55, 0.45)),
    'carpet_gray': lambda r: carpet(r, (0.48, 0.47, 0.46)),
    'carpet_taupe': lambda r: carpet(r, (0.45, 0.39, 0.33)),
    'carpet_blue': lambda r: carpet(r, (0.30, 0.34, 0.42)),
    'wood_oak': lambda r: wood(r, (0.62, 0.45, 0.28)),
    'wood_walnut': lambda r: wood(r, (0.36, 0.23, 0.14)),
    'wood_maple': lambda r: wood(r, (0.78, 0.63, 0.45)),
    'tile_white': lambda r: tile(r, (0.88, 0.88, 0.86), (0.6, 0.6, 0.58)),
    'tile_gray': lambda r: tile(r, (0.55, 0.55, 0.56), (0.35, 0.35, 0.35), tiles=3),
    'tile_beige': lambda r: tile(r, (0.8, 0.74, 0.64), (0.55, 0.5, 0.45), tiles=3),
    'paint_white': lambda r: paint(r, (0.9, 0.89, 0.86)),
    'stone_gray': lambda r: stone(r, (0.5, 0.49, 0.47)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'E:\twin\assets\remodel\textures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(7)
    for name, fn in TEXTURES.items():
        to_img(fn(rng)).save(os.path.join(a.out, name + '.png'))
        print('wrote', name)


if __name__ == '__main__':
    main()
