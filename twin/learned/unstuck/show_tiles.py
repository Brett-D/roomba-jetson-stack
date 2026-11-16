"""Draw a grid of generated stuck tiles (top view) to check the generator."""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import scenarios as S


def draw(ax, t: S.Tile):
    for x0, y0, x1, y1 in t.holes:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="#f4b6c2"))
    for p in t.prims:
        poly = p.polygon()
        if p.z0 >= S.BLOCK_Z:
            col = "#cfd8dc"  # high (robot passes under)
        elif p.z0 > S.BODY_TOP:
            col = "#ff9800"  # overhang at camera height
        elif p.z1 < S.STEP_BLOCK:
            col = "#e0f2f1"  # drivable rug
        elif p.depth == "none":
            col = "#9c27b0"  # invisible to depth
        else:
            col = "#37474f"
        ax.fill(poly[:, 0], poly[:, 1], color=col, lw=0)
    for m in t.marks:
        m = np.asarray(m)
        ax.fill(m[:, 0], m[:, 1], color="#e53935", alpha=0.5, lw=0)
    sx, sy, syaw = t.start
    ax.add_patch(plt.Circle((sx, sy), S.ROBOT_R, fill=False, color="#1e88e5", lw=1.5))
    ax.plot([sx, sx + 0.25 * math.cos(syaw)], [sy, sy + 0.25 * math.sin(syaw)], color="#1e88e5", lw=1.5)
    ax.plot(*t.goal, marker="*", color="#43a047", ms=10)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(t.kind, fontsize=8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="tiles.png")
    a = ap.parse_args()
    tiles = S.load(a.tiles)[: a.n] if a.tiles else S.generate(a.n, a.seed)
    cols = 6
    rows = math.ceil(len(tiles) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.3))
    for ax, t in zip(np.ravel(axs), tiles):
        draw(ax, t)
    for ax in np.ravel(axs)[len(tiles):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(a.out)
