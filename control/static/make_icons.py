#!/usr/bin/env python3
"""Draws the control page's home-screen icons (icon-192.png, icon-512.png) next to this file:
a HUD-green power symbol on a dark cockpit panel with four screws. Needs Pillow."""
import math
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def icon(size):
    s = size * 4  # drawn 4x, then scaled down for smooth edges
    img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=s // 6, fill=(43, 46, 50, 255))
    m = s // 14
    d.rounded_rectangle([m, m, s - 1 - m, s - 1 - m], radius=s // 9, fill=(4, 12, 7, 255))
    for cx in (m * 1.9, s - m * 1.9):
        for cy in (m * 1.9, s - m * 1.9):
            r = s * 0.028
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(125, 131, 136, 255))
    green = (61, 255, 143, 255)
    c, r, w = s / 2, s * 0.25, int(s * 0.055)
    # The ring with a gap at the top, then the bar through the gap.
    gap = 50
    d.arc([c - r, c - r, c + r, c + r], start=-90 + gap / 2, end=270 - gap / 2, fill=green, width=w)
    for a in (-90 + gap / 2, 270 - gap / 2):
        x, y = c + (r - w / 2) * math.cos(math.radians(a)), c + (r - w / 2) * math.sin(math.radians(a))
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=green)
    d.rounded_rectangle([c - w / 2, c - r * 1.18, c + w / 2, c - r * 0.12], radius=w // 2, fill=green)
    return img.resize((size, size), Image.LANCZOS)


if __name__ == '__main__':
    for size in (192, 512):
        icon(size).save(os.path.join(HERE, f'icon-{size}.png'), optimize=True)
