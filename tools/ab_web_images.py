#!/usr/bin/env python3
"""Checks the Rust web server's images against roomba_web/server.py's OpenCV calls.

Camera preview: recorded mono frames (and synthetic frames of other sizes) shrunk with
cv2.resize(INTER_AREA) and encoded with cv2.imencode('.jpg', quality 70); map: occupancy grids
(random and map-like) coloured, flipped and encoded with cv2.imencode('.png') exactly as
server.py's _on_map. The Rust side is examples/web_images.rs. Reports byte equality of the JPEG
and PNG files, pixel equality of the resized frames and of the decoded PNGs.

  tools/ab_web_images.py BAG OUT_DIR RUST_EXAMPLE [--frames 200]
"""
import argparse
import os
import subprocess

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from sensor_msgs.msg import Image


def preview_size(w, h, pw=480):
    return (pw, int(h * pw / w)) if w > pw else (w, h)


def map_image(grid):
    img = np.zeros(grid.shape + (4,), np.uint8)
    g = grid.astype(np.int16)
    img[g >= 0] = (70, 60, 52, 255)
    img[g >= 65] = (240, 232, 225, 255)
    return np.ascontiguousarray(np.flipud(img))


def write_frame(out, fid, img, step=None):
    h, w = img.shape
    step = step or w
    buf = np.zeros((h, step), np.uint8)
    buf[:, :w] = img
    dw, dh = preview_size(w, h)
    buf.tofile(os.path.join(out, f'{fid}_{w}x{h}_{step}_{dw}x{dh}.gray'))
    view = buf[:, :w]
    small = cv2.resize(view, (dw, dh), interpolation=cv2.INTER_AREA) if (dw, dh) != (w, h) else view
    ok, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    np.ascontiguousarray(small).tofile(os.path.join(out, f'{fid}.py.area'))
    jpg.tofile(os.path.join(out, f'{fid}.py.jpg'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('out')
    ap.add_argument('example')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--every', type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(7)

    r = SequentialReader()
    r.open(StorageOptions(uri=a.bag, storage_id='mcap'), ConverterOptions('', ''))
    r.set_filter(StorageFilter(topics=['/camera/left/image_rect_mono']))
    n = seen = 0
    while r.has_next() and n < a.frames:
        _, data, _ = r.read_next()
        seen += 1
        if seen % a.every:
            continue
        m = deserialize_message(data, Image)
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width]
        write_frame(a.out, f'bag{n:04d}', img, m.step)
        n += 1
    bag_frames = n
    # Other camera sizes: fractional and integer area ratios, padded rows.
    sizes = [(1280, 720), (960, 540), (1920, 1080), (752, 480), (641, 361), (800, 600), (481, 300), (480, 270), (320, 240)]
    for i, (w, h) in enumerate(sizes):
        base = cv2.GaussianBlur(rng.integers(0, 256, (h, w), dtype=np.uint8), (0, 0), 3)
        noisy = np.clip(base.astype(int) + rng.integers(-40, 40, (h, w)), 0, 255).astype(np.uint8)
        write_frame(a.out, f'syn{i:02d}', noisy, step=w + (8 if i % 2 else 0))
    # Grids: random values over the whole int8 range, and map-like rooms.
    grids = []
    for i in range(30):
        w, h = int(rng.integers(1, 700)), int(rng.integers(1, 700))
        grids.append(rng.integers(-128, 128, (h, w), dtype=np.int16).astype(np.int8))
    for i in range(20):
        w, h = int(rng.integers(100, 900)), int(rng.integers(100, 900))
        g = np.full((h, w), -1, np.int8)
        for _ in range(int(rng.integers(3, 12))):
            x0, y0 = int(rng.integers(0, w - 10)), int(rng.integers(0, h - 10))
            x1, y1 = int(rng.integers(x0 + 5, w)), int(rng.integers(y0 + 5, h))
            g[y0:y1, x0:x1] = 0
            g[y0, x0:x1] = g[y1 - 1, x0:x1] = 100
            g[y0:y1, x0] = g[y0:y1, x1 - 1] = 100
        g[rng.random((h, w)) < 0.01] = 100
        grids.append(g)
    for i, g in enumerate(grids):
        h, w = g.shape
        g.tofile(os.path.join(a.out, f'grid{i:02d}_{w}x{h}.grid'))
        ok, png = cv2.imencode('.png', map_image(g))
        assert ok
        png.tofile(os.path.join(a.out, f'grid{i:02d}.py.png'))

    print(subprocess.run([a.example, a.out], check=True, capture_output=True, text=True).stdout.strip())

    def read(name):
        with open(os.path.join(a.out, name), 'rb') as f:
            return f.read()

    stats = {'frames': 0, 'area_equal': 0, 'jpeg_equal': 0, 'grids': 0, 'png_equal': 0, 'png_pixels_equal': 0}
    worst = []
    for name in sorted(os.listdir(a.out)):
        if name.endswith('.gray'):
            fid = name.split('_')[0]
            stats['frames'] += 1
            pa, ra = read(f'{fid}.py.area'), read(f'{fid}.rs.area')
            if pa == ra:
                stats['area_equal'] += 1
            else:
                d = np.abs(np.frombuffer(pa, np.uint8).astype(int) - np.frombuffer(ra, np.uint8).astype(int))
                worst.append((fid, int(d.max()), int((d > 0).sum())))
            stats['jpeg_equal'] += read(f'{fid}.py.jpg') == read(f'{fid}.rs.jpg')
        elif name.endswith('.grid'):
            gid = name.split('_')[0]
            stats['grids'] += 1
            pp, rp = read(f'{gid}.py.png'), read(f'{gid}.rs.png')
            stats['png_equal'] += pp == rp
            dp = cv2.imdecode(np.frombuffer(pp, np.uint8), cv2.IMREAD_UNCHANGED)
            dr = cv2.imdecode(np.frombuffer(rp, np.uint8), cv2.IMREAD_UNCHANGED)
            stats['png_pixels_equal'] += dp.shape == dr.shape and bool((dp == dr).all())
    print(f'{bag_frames} recorded frames + {len(sizes)} synthetic sizes; {len(grids)} grids')
    print(stats)
    for fid, mx, cnt in worst[:10]:
        print(f'  {fid}: resized pixels differ (max {mx}, {cnt} pixels)')


if __name__ == '__main__':
    main()
