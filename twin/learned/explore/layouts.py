"""House layouts for the exploration environment (0.1 m grid, numpy).

- procedural houses: rectangular or L-shaped outline, rooms by recursive splitting, doors (closable)
  and open-plan openings, hallways, furniture against walls, tables as legs only;
- the real house's floor plans (remodelled twin): floorplan.json (upstairs, where the robot was mapped) and
  floorplan_floor1.json (downstairs), rasterised from the vector plans in the map frame.

A layout is a dict: occ [H, W] uint8 (1 = wall / furniture / outside), door [H, W] int16 (0 = none,
k = door k-1), door_closed0 [D] bool, start (x, y) metres in grid coordinates (x = col * res),
origin (map-frame position of cell (0, 0) corner), name.
"""

from __future__ import annotations

import json
import math

import numpy as np

RES = 0.1


def _rect(a, r0, c0, r1, c1, v):
    a[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = v


def _clearance_ok(occ, r, c, rad):
    H, W = occ.shape
    r0, r1, c0, c1 = max(r - rad, 0), min(r + rad + 1, H), max(c - rad, 0), min(c + rad + 1, W)
    return not occ[r0:r1, c0:c1].any()


def gen_house(rng, size=160):
    H = W = size
    occ = np.ones((H, W), np.uint8)
    door = np.zeros((H, W), np.int16)
    hw = int(rng.uniform(80, size - 8))
    hh = int(rng.uniform(70, size - 8))
    r0 = int(rng.integers(3, H - hh - 3))
    c0 = int(rng.integers(3, W - hw - 3))
    _rect(occ, r0 + 2, c0 + 2, r0 + hh - 2, c0 + hw - 2, 0)  # exterior walls 0.2 m
    if rng.random() < 0.4:  # L shape: cut a corner
        ch, cw = int(hh * rng.uniform(0.3, 0.5)), int(hw * rng.uniform(0.3, 0.5))
        if rng.random() < 0.5:
            _rect(occ, r0, c0, r0 + ch, c0 + cw, 1)
            _rect(occ, r0 + ch, c0 + 2, r0 + ch + 2, c0 + cw, 1)
            _rect(occ, r0 + 2, c0 + cw, r0 + ch, c0 + cw + 2, 1)
        else:
            _rect(occ, r0 + hh - ch, c0 + hw - cw, r0 + hh, c0 + hw, 1)
            _rect(occ, r0 + hh - ch - 2, c0 + hw - cw, r0 + hh - ch, c0 + hw - 2, 1)
            _rect(occ, r0 + hh - ch, c0 + hw - cw - 2, r0 + hh - 2, c0 + hw - cw, 1)
    doors = []

    def split(rr0, cc0, rr1, cc1, depth):
        h, w = rr1 - rr0, cc1 - cc0
        if depth > 5 or (h < 45 and w < 45) or (h < 60 and w < 60 and rng.random() < 0.3):
            return
        vertical = w > h if abs(w - h) > 10 else rng.random() < 0.5
        if vertical:
            if w < 50:
                return
            cut = cc0 + int(w * rng.uniform(0.35, 0.65))
            if rng.random() < 0.15:  # hallway strip
                cut2 = cut + int(rng.uniform(10, 14))
                if cut2 < cc1 - 22:
                    _wall_v(cut, rr0, rr1)
                    _wall_v(cut2, rr0, rr1)
                    split(rr0, cc0, rr1, cut, depth + 1)
                    split(rr0, cut2 + 1, rr1, cc1, depth + 1)
                    return
            _wall_v(cut, rr0, rr1)
            split(rr0, cc0, rr1, cut, depth + 1)
            split(rr0, cut + 1, rr1, cc1, depth + 1)
        else:
            if h < 50:
                return
            cut = rr0 + int(h * rng.uniform(0.35, 0.65))
            _wall_h(cut, cc0, cc1)
            split(rr0, cc0, cut, cc1, depth + 1)
            split(cut + 1, cc0, rr1, cc1, depth + 1)

    def _opening(n, lo, hi):
        # door width 0.8-1.0 m (or a 1.5-3 m open-plan opening)
        if hi - lo < 12:
            return None
        width = int(rng.uniform(8, 11)) if rng.random() < 0.8 else int(rng.uniform(15, 30))
        width = min(width, hi - lo - 4)
        s = int(rng.integers(lo + 2, hi - width - 1))
        return s, s + width, width <= 11

    def _wall_v(c, ra, rb):
        occ[ra:rb, c] = np.where(occ[ra:rb, c] == 0, 1, occ[ra:rb, c])
        for _ in range(1 if rng.random() < 0.85 else 2):
            o = _opening(c, ra, rb)
            if o:
                s, e, is_door = o
                occ[s:e, c] = 0
                if is_door:
                    doors.append([(r, c) for r in range(s, e)])

    def _wall_h(r, ca, cb):
        occ[r, ca:cb] = np.where(occ[r, ca:cb] == 0, 1, occ[r, ca:cb])
        for _ in range(1 if rng.random() < 0.85 else 2):
            o = _opening(r, ca, cb)
            if o:
                s, e, is_door = o
                occ[r, s:e] = 0
                if is_door:
                    doors.append([(r, c) for c in range(s, e)])

    split(r0 + 2, c0 + 2, r0 + hh - 2, c0 + hw - 2, 0)
    for k, cells in enumerate(doors):
        for r, c in cells:
            door[r, c] = k + 1
    # keep a free margin around doors for furniture placement
    near_door = np.zeros_like(occ, bool)
    for cells in doors:
        for r, c in cells:
            near_door[max(r - 7, 0):r + 8, max(c - 7, 0):c + 8] = True
    # furniture
    free = occ == 0
    n_items = int(rng.integers(hw * hh // 900, hw * hh // 350 + 1))
    for _ in range(n_items):
        kind = rng.random()
        for _try in range(10):
            r, c = int(rng.integers(r0, r0 + hh)), int(rng.integers(c0, c0 + hw))
            if kind < 0.6:  # box furniture (bed, sofa, dresser, cabinet)
                a, b = int(rng.uniform(4, 20)), int(rng.uniform(4, 14))
                if rng.random() < 0.5:
                    a, b = b, a
                region = (slice(r, r + a), slice(c, c + b))
                if r + a >= H or c + b >= W or not free[region].all() or near_door[region].any():
                    continue
                # prefer against a wall
                touch = occ[max(r - 1, 0):r + a + 1, max(c - 1, 0):c + b + 1].any()
                if not touch and rng.random() < 0.6:
                    continue
                occ[region] = 1
                break
            else:  # table / chairs: legs only
                a, b = int(rng.uniform(7, 14)), int(rng.uniform(7, 18))
                region = (slice(r, r + a), slice(c, c + b))
                if r + a >= H or c + b >= W or not free[region].all() or near_door[region].any():
                    continue
                for rr, cc in ((r, c), (r, c + b - 1), (r + a - 1, c), (r + a - 1, c + b - 1)):
                    occ[rr, cc] = 1
                break
        free = occ == 0
    closed0 = rng.random(len(doors)) < 0.1
    return finish_layout(occ, door, closed0, rng, name="procedural")


def reachable(occ, start_rc, rad=2):
    """Cells the robot centre can reach (8-connected) with clearance rad cells."""
    from scipy.ndimage import binary_dilation, label

    blocked = binary_dilation(occ > 0, structure=_disk(rad))
    lab, _ = label(~blocked, structure=np.ones((3, 3)))
    r, c = start_rc
    if lab[r, c] == 0:
        return np.zeros_like(blocked)
    return lab == lab[r, c]


def _disk(rad):
    y, x = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    return (x * x + y * y) <= rad * rad + 0.01


def finish_layout(occ, door, closed0, rng, name, start=None, origin=(0.0, 0.0)):
    from scipy.ndimage import binary_dilation

    blocked = binary_dilation(occ > 0, structure=_disk(4))
    if start is None:
        cand = np.argwhere(~blocked)
        if len(cand) == 0:
            return None
        start = tuple(cand[rng.integers(len(cand))])
    closed = door * 0
    for k in np.nonzero(closed0)[0]:
        closed |= door == k + 1
    reach = reachable(np.maximum(occ, (closed > 0).astype(np.uint8)), start)
    if reach.sum() < 2000 and name == "procedural":
        return None
    return {
        "occ": occ,
        "door": door,
        "door_closed0": np.asarray(closed0, bool),
        "start": (start[1] * RES + RES / 2, start[0] * RES + RES / 2),
        "origin": origin,
        "name": name,
    }


def generate(n, seed, size=160):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        lay = gen_house(rng, size)
        if lay is not None:
            out.append(lay)
    return out


# ---------------------------------------------------------------- the real floor plans


def _fill_poly(grid, poly, origin, v, pad=0.0):
    from matplotlib.path import Path

    H, W = grid.shape
    poly = np.asarray(poly, float)
    x0, y0 = poly.min(0) - pad - RES
    x1, y1 = poly.max(0) + pad + RES
    c0, c1 = max(int((x0 - origin[0]) / RES), 0), min(int((x1 - origin[0]) / RES) + 1, W)
    r0, r1 = max(int((y0 - origin[1]) / RES), 0), min(int((y1 - origin[1]) / RES) + 1, H)
    if c0 >= c1 or r0 >= r1:
        return
    cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    px = origin[0] + (cc + 0.5) * RES
    py = origin[1] + (rr + 0.5) * RES
    inside = Path(poly).contains_points(np.stack([px.ravel(), py.ravel()], 1), radius=-1e-9).reshape(px.shape)
    if pad > 0:
        d = np.full(px.shape, np.inf)
        n = len(poly)
        for i in range(n - 1 if len(poly) > 1 else 1):
            a, b = poly[i], poly[(i + 1) % n]
            ab = b - a
            t = np.clip(((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / max(ab @ ab, 1e-12), 0, 1)
            d = np.minimum(d, np.hypot(px - a[0] - t * ab[0], py - a[1] - t * ab[1]))
        inside |= d <= pad
    sub = grid[r0:r1, c0:c1]
    sub[inside] = v


def _polyline(grid, pts, origin, v, pad):
    pts = np.asarray(pts, float)
    for i in range(len(pts) - 1):
        _fill_poly(grid, pts[i:i + 2], origin, v, pad)


def from_floorplan(path, start_xy=None, start_room=None, size=256):
    d = json.load(open(path))
    pts = np.concatenate([np.asarray(w) for w in d["walls"]])
    lo, hi = pts.min(0), pts.max(0)
    span = hi - lo
    origin = (float(lo[0] - (size * RES - span[0]) / 2), float(lo[1] - (size * RES - span[1]) / 2))
    occ = np.ones((size, size), np.uint8)  # outside everything = blocked
    for room in d["rooms"]:
        if room.get("outdoor") or room["type"] in ("balcony",):
            continue
        _fill_poly(occ, room["polygon"], origin, 0)
    for w in d["walls"]:
        _fill_poly(occ, w, origin, 1)
    door = np.zeros((size, size), np.int16)
    closed0 = []
    for k, dr in enumerate(d["doors"]):
        tmp = np.zeros_like(occ)
        _polyline(tmp, dr["segment"], origin, 1, pad=0.12)
        cells = (tmp > 0) & (occ > 0) | (tmp > 0)
        occ[cells] = 0
        door[cells] = k + 1
        closed0.append(dr.get("default", "open") == "closed" or dr["type"] == "glass")
    for f in d.get("furniture", []):
        _fill_poly(occ, f["footprint"], origin, 1)
    for f in d.get("fixtures", []):
        _fill_poly(occ, f["footprint"], origin, 1)
    if d.get("stair_void"):
        _fill_poly(occ, d["stair_void"]["polygon"], origin, 1)
    for r in d.get("railings", []):
        _polyline(occ, r["polyline"], origin, 1, pad=0.05)
    if start_room:
        room = [r for r in d["rooms"] if r["name"] == start_room][0]
        start_xy = tuple(np.asarray(room["polygon"]).mean(0))
    sx, sy = start_xy
    sr, sc = int((sy - origin[1]) / RES), int((sx - origin[0]) / RES)
    # nearest cell with 0.4 m clearance
    from scipy.ndimage import binary_dilation

    blocked = binary_dilation(occ > 0, structure=_disk(4))
    cand = np.argwhere(~blocked)
    i = np.argmin((cand[:, 0] - sr) ** 2 + (cand[:, 1] - sc) ** 2)
    lay = finish_layout(occ, door, np.array(closed0, bool), None, name=path.split("\\")[-1].split("/")[-1], start=tuple(cand[i]), origin=origin)
    return lay


def plan_layouts(dirpath="E:/twin/assets/remodel"):
    up = from_floorplan(f"{dirpath}/floorplan.json", start_xy=(0.0, 0.0))
    up["name"] = "plan_upstairs"
    down = from_floorplan(f"{dirpath}/floorplan_floor1.json", start_room="Entry")
    down["name"] = "plan_downstairs"
    return [up, down]


if __name__ == "__main__":
    import sys
    import time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    lays = generate(10, 1)
    print(f"10 houses in {time.time() - t0:.1f}s")
    lays += plan_layouts()
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    for ax, lay in zip(axs.ravel(), lays):
        img = np.where(lay["occ"] > 0, 0.0, 1.0)
        img = np.where(lay["door"] > 0, 0.6, img)
        ax.imshow(img, cmap="gray", origin="lower")
        ax.plot(lay["start"][0] / RES, lay["start"][1] / RES, "r*", ms=12)
        st = lay["start"]
        reach = reachable(lay["occ"], (int(st[1] / RES), int(st[0] / RES)))
        ax.set_title(f"{lay['name']} doors {len(lay['door_closed0'])} reach {reach.sum() * RES * RES:.0f} m2")
    fig.tight_layout()
    out = sys.argv[1] if len(sys.argv) > 1 else "layouts.png"
    fig.savefig(out, dpi=80)
    print(out)
