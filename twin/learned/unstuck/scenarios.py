"""Procedural "stuck" situations for the unstuck skill (tile generator, pure numpy).

Every tile is a 4 x 4 m patch of floor with a perimeter wall and one stuck situation built around a
robot pose: the robot starts at the tile centre region already stuck (nose to a wall inside the
camera's 0.3 m blind zone, wedged in a corner, between chair legs, under an overhang at camera
height, at an invisible railing in front of a stair void, against a threshold, inside a ring of
bump keep-out marks, ...). The geometry is described by primitives with a height range and how
the sensors see them:

  depth  'solid' (the D1000 depth fan sees it), 'none' (thin posts / glass: invisible) or
         'sparse' (decided per tile: seen or not)
  ir     the light bumpers (IR) see it
  block  counts as an obstacle for the ground-truth clearance (anything the body, camera housing
         or payload can hit: lower edge under 0.155 m and top over 0.02 m), plus floor holes

The same tiles feed the Isaac Lab scene (collision mesh, depth / IR ray-cast meshes), the
ground-truth clearance field, the initial costmap keep-out marks and the episode goal.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

TILE = 4.0  # m
GRID_RES = 0.05  # m, ground-truth / costmap cell
GRID_N = int(round(TILE / GRID_RES))  # 80
ROBOT_R = 0.17
BODY_TOP = 0.092
BLOCK_Z = 0.155  # camera housing top 0.145, payload top 0.152
STEP_BLOCK = 0.02  # thresholds this high or higher can't be driven over
SUCCESS_CLEARANCE = 0.30


@dataclass
class Prim:
    kind: str  # box | cyl | poly
    p: tuple  # box: (cx, cy, sx, sy, yaw); cyl: (cx, cy, r); poly: (((x, y), ...),)
    z0: float
    z1: float
    depth: str = "solid"  # solid | none | sparse
    ir: bool = True

    def blocks(self) -> bool:
        return self.z0 < BLOCK_Z and self.z1 >= STEP_BLOCK

    def blocks_body(self) -> bool:
        return self.z0 < BODY_TOP and self.z1 >= STEP_BLOCK

    def polygon(self, cyl_sections: int = 12) -> np.ndarray:
        if self.kind == "box":
            cx, cy, sx, sy, yaw = self.p
            c, s = math.cos(yaw), math.sin(yaw)
            pts = np.array([[-sx / 2, -sy / 2], [sx / 2, -sy / 2], [sx / 2, sy / 2], [-sx / 2, sy / 2]])
            return pts @ np.array([[c, s], [-s, c]]) + [cx, cy]
        if self.kind == "cyl":
            cx, cy, r = self.p
            a = np.linspace(0, 2 * math.pi, cyl_sections, endpoint=False)
            return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
        return np.asarray(self.p[0], dtype=float)

    def transformed(self, rot: float, dx: float = 0.0, dy: float = 0.0) -> "Prim":
        c, s = math.cos(rot), math.sin(rot)

        def tf(x, y):
            return (c * x - s * y + dx, s * x + c * y + dy)

        if self.kind == "box":
            cx, cy, sx, sy, yaw = self.p
            q = (*tf(cx, cy), sx, sy, yaw + rot)
        elif self.kind == "cyl":
            cx, cy, r = self.p
            q = (*tf(cx, cy), r)
        else:
            q = (tuple(tf(x, y) for x, y in self.p[0]),)
        return Prim(self.kind, q, self.z0, self.z1, self.depth, self.ir)


@dataclass
class Tile:
    kind: str
    prims: list = field(default_factory=list)
    holes: list = field(default_factory=list)  # axis-aligned (x0, y0, x1, y1), tile frame
    marks: list = field(default_factory=list)  # keep-out polygons already in the costmap
    start: tuple = (0.0, 0.0, 0.0)
    goal: tuple = (1.5, 0.0)
    source: str = "procedural"

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "source": self.source,
            "prims": [[p.kind, p.p, p.z0, p.z1, p.depth, p.ir] for p in self.prims],
            "holes": self.holes,
            "marks": [np.asarray(m).tolist() for m in self.marks],
            "start": self.start,
            "goal": self.goal,
        }

    @staticmethod
    def from_json(d: dict) -> "Tile":
        def prim(e):
            kind, p, z0, z1, depth, ir = e
            if kind == "poly":
                p = (tuple(tuple(q) for q in p[0]),)
            return Prim(kind, tuple(p), z0, z1, depth, ir)

        return Tile(
            d["kind"],
            [prim(e) for e in d["prims"]],
            [tuple(h) for h in d["holes"]],
            [np.asarray(m) for m in d["marks"]],
            tuple(d["start"]),
            tuple(d["goal"]),
            d.get("source", "procedural"),
        )


# ---------------------------------------------------------------- rasterising


def _cell_centres():
    c = (np.arange(GRID_N) + 0.5) * GRID_RES - TILE / 2
    return np.meshgrid(c, c, indexing="xy")  # X[row, col], Y[row, col]; row = y index


def _poly_mask(poly: np.ndarray, X, Y, pad: float) -> np.ndarray:
    from matplotlib.path import Path

    path = Path(poly)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    inside = path.contains_points(pts, radius=0.0).reshape(X.shape)
    if pad > 0:
        # distance from cell centres to the polygon's edges
        d = np.full(X.shape, np.inf)
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            ab = b - a
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / max(ab @ ab, 1e-12), 0, 1)
            d = np.minimum(d, np.hypot(X - a[0] - t * ab[0], Y - a[1] - t * ab[1]))
        inside |= d <= pad
    return inside


def rasterise(tile: Tile, which: str = "block") -> np.ndarray:
    """Boolean [GRID_N, GRID_N] (row = y, col = x, tile frame) of blocking cells (+ holes)."""
    X, Y = _cell_centres()
    occ = np.zeros(X.shape, dtype=bool)
    for p in tile.prims:
        if which == "block" and not p.blocks():
            continue
        if which == "body" and not p.blocks_body():
            continue
        if p.kind == "cyl":
            cx, cy, r = p.p
            occ |= np.hypot(X - cx, Y - cy) <= r + GRID_RES / 2
        else:
            occ |= _poly_mask(p.polygon(), X, Y, GRID_RES / 2)
    if which != "body":
        for x0, y0, x1, y1 in tile.holes:
            occ |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    return occ


def clearance_field(occ: np.ndarray) -> np.ndarray:
    """Distance (m) from each cell centre to the nearest blocking cell centre."""
    from scipy.ndimage import distance_transform_edt

    if not occ.any():
        return np.full(occ.shape, 9.0, dtype=np.float32)
    return (distance_transform_edt(~occ) * GRID_RES).astype(np.float32)


def lookup(field_: np.ndarray, x: float, y: float) -> float:
    c = int(np.clip((x + TILE / 2) / GRID_RES, 0, GRID_N - 1))
    r = int(np.clip((y + TILE / 2) / GRID_RES, 0, GRID_N - 1))
    return float(field_[r, c])


# ---------------------------------------------------------------- building blocks


def wall(x0, y0, x1, y1, t=0.1, h=2.4, depth="solid"):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L = math.hypot(x1 - x0, y1 - y0)
    return Prim("box", (cx, cy, L, t, math.atan2(y1 - y0, x1 - x0)), 0.0, h, depth)


def box(cx, cy, sx, sy, yaw=0.0, z0=0.0, z1=0.8, depth="solid", ir=True):
    return Prim("box", (cx, cy, sx, sy, yaw), z0, z1, depth, ir)


def legs(cx, cy, sx, sy, yaw, r, h, depth):
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for u, v in ((sx / 2, sy / 2), (-sx / 2, sy / 2), (-sx / 2, -sy / 2), (sx / 2, -sy / 2)):
        out.append(Prim("cyl", (cx + c * u - s * v, cy + s * u + c * v, r), 0.0, h, depth))
    return out


def chair(rng, cx, cy, yaw, side=None):
    side = side or rng.uniform(0.38, 0.5)
    r = rng.uniform(0.012, 0.025)
    depth = "sparse" if r < 0.02 else "solid"
    seat_z = rng.uniform(0.42, 0.48)
    out = legs(cx, cy, side, side, yaw, r, seat_z, depth)
    out.append(box(cx, cy, side + 0.04, side + 0.04, yaw, seat_z, seat_z + 0.05))
    if rng.random() < 0.6:  # backrest
        c, s = math.cos(yaw), math.sin(yaw)
        out.append(box(cx - c * side / 2, cy - s * side / 2, 0.04, side, yaw, seat_z, seat_z + 0.45))
    return out


def railing(x0, y0, x1, y1, rng, h=0.92, depth="none"):
    L = math.hypot(x1 - x0, y1 - y0)
    spacing = rng.uniform(0.09, 0.13)
    n = max(2, int(L / spacing) + 1)
    r = rng.uniform(0.007, 0.015)
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(Prim("cyl", (x0 + t * (x1 - x0), y0 + t * (y1 - y0), r), 0.0, h, depth))
    out.append(wall(x0, y0, x1, y1, t=0.05, h=h + 0.04, depth=depth))
    out[-1].z0 = h  # top rail only
    return out


def bump_strip(x, y, yaw, bearing, width=0.5):
    """explorer.rs bump_zone (contact 0.18 m, depth 0.06 m) as a polygon."""
    a = yaw + bearing
    ca, sa = math.cos(a), math.sin(a)
    h = width / 2
    return np.array([[x + u * ca - v * sa, y + u * sa + v * ca] for u, v in ((0.18, -h), (0.24, -h), (0.24, h), (0.18, h))])


# ---------------------------------------------------------------- scenario kinds (canonical frame)
# Canonical: robot near the origin facing +x. The tile is then rotated by k * 90 deg.


def _clutter(rng, n, avoid=1.0):
    out = []
    for _ in range(n):
        for _ in range(20):
            x, y = rng.uniform(-1.7, 1.7, 2)
            if math.hypot(x, y) > avoid:
                break
        kind = rng.integers(3)
        if kind == 0:
            out.append(box(x, y, rng.uniform(0.2, 0.9), rng.uniform(0.2, 0.6), rng.uniform(0, math.pi), 0, rng.uniform(0.3, 1.0)))
        elif kind == 1:
            out += chair(rng, x, y, rng.uniform(0, 2 * math.pi))
        else:
            out.append(Prim("cyl", (x, y, rng.uniform(0.1, 0.25)), 0, rng.uniform(0.3, 0.9)))
    return out


def gen_nose_wall(rng):
    gap = rng.uniform(0.0, 0.12)
    x = ROBOT_R + gap + 0.05
    t = Tile("nose_wall")
    t.prims.append(wall(x, -rng.uniform(0.6, 1.9), x, rng.uniform(0.6, 1.9)))
    if rng.random() < 0.5:
        sy = rng.choice([-1, 1]) * (ROBOT_R + rng.uniform(0.01, 0.15) + 0.05)
        t.prims.append(wall(x, sy, x - rng.uniform(0.6, 1.8), sy))
    t.prims += _clutter(rng, rng.integers(0, 3))
    t.start = (0.0, 0.0, rng.uniform(-0.45, 0.45))
    return t


def gen_corner(rng):
    a = math.radians(rng.uniform(45, 100))  # opening angle
    t = Tile("corner")
    # apex ahead of the robot; walls symmetric about the x axis (+ jitter)
    d_apex = (ROBOT_R + rng.uniform(0.0, 0.04)) / math.sin(a / 2) + 0.05
    ax = d_apex
    jitter = rng.uniform(-0.25, 0.25)
    for sgn in (-1, 1):
        ang = math.pi + sgn * (a / 2) + jitter
        L = rng.uniform(1.0, 1.9)
        t.prims.append(wall(ax, 0, ax + L * math.cos(ang), L * math.sin(ang)))
    t.prims += _clutter(rng, rng.integers(0, 2), avoid=1.3)
    t.start = (0.0, 0.0, jitter + rng.uniform(-0.3, 0.3))
    return t


def gen_chair_legs(rng):
    t = Tile("chair_legs")
    # a table with chairs around it; the robot inside the leg forest
    tx, ty = rng.uniform(-0.3, 0.3, 2)
    tw, tl = rng.uniform(0.7, 1.1), rng.uniform(0.9, 1.6)
    tyaw = rng.uniform(0, math.pi)
    r = rng.uniform(0.02, 0.035)
    t.prims += legs(tx, ty, tl - 0.1, tw - 0.1, tyaw, r, 0.72, "solid")
    t.prims.append(box(tx, ty, tl, tw, tyaw, 0.72, 0.76))
    c, s = math.cos(tyaw), math.sin(tyaw)
    for u, v, ya in ((tl / 2 + 0.15, 0, math.pi), (-tl / 2 - 0.15, 0, 0), (0.2, tw / 2 + 0.15, -math.pi / 2), (-0.25, -tw / 2 - 0.15, math.pi / 2)):
        if rng.random() < 0.8:
            t.prims += chair(rng, tx + c * u - s * v, ty + s * u + c * v, tyaw + ya + rng.uniform(-0.3, 0.3))
    if rng.random() < 0.5:
        t.prims.append(wall(-1.2, -1.9, -1.2, 1.9))
    t.start = (rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), rng.uniform(-math.pi, math.pi))
    return t


def gen_gap(rng):
    t = Tile("narrow_gap")
    g = rng.uniform(0.36, 0.5)
    depth_a, depth_b = rng.uniform(0.5, 1.0, 2)
    L = rng.uniform(0.8, 1.6)
    x0 = rng.uniform(-0.3, 0.2)
    t.prims.append(box(x0 + L / 2, g / 2 + depth_a / 2, L, depth_a, 0, 0, rng.uniform(0.4, 0.9)))
    t.prims.append(box(x0 + L / 2, -g / 2 - depth_b / 2, L, depth_b, 0, 0, rng.uniform(0.4, 0.9)))
    if rng.random() < 0.5:  # dead end
        t.prims.append(wall(x0 + L + 0.05, -1, x0 + L + 0.05, 1))
    t.prims += _clutter(rng, rng.integers(0, 2), avoid=1.5)
    yaw = rng.uniform(-0.35, 0.35)
    t.start = (x0 + rng.uniform(0.0, 0.25), rng.uniform(-0.05, 0.05), yaw)
    return t


def gen_threshold(rng):
    t = Tile("threshold")
    h = rng.uniform(0.02, 0.035)
    x = ROBOT_R + rng.uniform(0.0, 0.03) + 0.03
    door = rng.uniform(0.75, 0.95)
    t.prims.append(box(x, 0, 0.06, door, 0, 0, h))
    off = rng.uniform(-0.15, 0.15)
    t.prims.append(wall(x, door / 2 + off, x, 1.9))
    t.prims.append(wall(x, -door / 2 + off, x, -1.9))
    t.prims[0] = box(x, off, 0.06, door, 0, 0, h)
    if rng.random() < 0.5:  # a drivable rug behind
        t.prims.append(box(-0.8, rng.uniform(-0.5, 0.5), 1.2, 1.0, rng.uniform(0, 1), 0, rng.uniform(0.006, 0.014)))
    t.prims += _clutter(rng, rng.integers(0, 2), avoid=1.3)
    t.start = (0.0, 0.0, rng.uniform(-0.6, 0.6))
    return t


def gen_railing(rng):
    t = Tile("railing")
    x = ROBOT_R + rng.uniform(0.0, 0.1) + 0.015
    y0, y1 = -rng.uniform(0.8, 1.9), rng.uniform(0.8, 1.9)
    t.prims += railing(x, y0, x, y1, rng)
    if rng.random() < 0.6:  # stair void behind the railing
        t.holes.append((x + 0.08, y0, 1.95, y1))
    if rng.random() < 0.5:  # earlier bump marks along the railing
        for yy in np.arange(y0 + 0.3, y1 - 0.3, rng.uniform(0.4, 0.8)):
            t.marks.append(bump_strip(x - 0.21, yy, 0.0, 0.0))
    if rng.random() < 0.4:
        t.prims.append(wall(x, y1, -1.5, y1))
    t.prims += _clutter(rng, rng.integers(0, 2), avoid=1.3)
    t.start = (0.0, 0.0, rng.uniform(-0.7, 0.7))
    return t


def gen_overhang(rng):
    t = Tile("overhang")
    zu = rng.uniform(0.095, 0.145)
    edge = rng.uniform(0.0, 0.15)  # front edge of the furniture relative to the robot centre
    depth = rng.uniform(0.6, 0.9)
    width = rng.uniform(0.9, 2.0)
    off = rng.uniform(-0.4, 0.4)
    t.prims.append(box(edge + depth / 2, off, depth, width, 0, zu, zu + rng.uniform(0.2, 0.4)))
    # legs at the corners (feet), visible
    t.prims += legs(edge + depth / 2, off, depth - 0.06, width - 0.06, 0, 0.025, zu, "solid")
    if rng.random() < 0.5:
        t.prims.append(wall(edge + depth + 0.05, -1.9, edge + depth + 0.05, 1.9))
    t.prims += _clutter(rng, rng.integers(0, 2), avoid=1.4)
    t.start = (0.0, 0.0, rng.uniform(-0.5, 0.5))
    return t


def gen_door(rng):
    t = Tile("door_leaf")
    w = rng.uniform(0.75, 0.9)
    x = rng.uniform(0.35, 0.6)
    t.prims.append(wall(x, w / 2, x, 1.9))
    t.prims.append(wall(x, -w / 2, x, -1.9))
    phi = math.radians(rng.uniform(60, 120))  # leaf angle from the wall, hinge at (x, w/2), opening toward -x
    hx, hy = x - 0.05, w / 2
    ex, ey = hx - w * math.sin(phi), hy - w * math.cos(phi)
    t.prims.append(wall(hx, hy, ex, ey, t=0.04, h=2.0))
    # robot wedged between the leaf and the wall
    mx, my = (hx + ex) / 2, (hy + ey) / 2
    nx, ny = -math.cos(phi), math.sin(phi)  # leaf normal (toward the wall side)
    sx, sy = mx + nx * (ROBOT_R + rng.uniform(0.02, 0.06)), my + ny * (ROBOT_R + rng.uniform(0.02, 0.06))
    t.prims += _clutter(rng, rng.integers(0, 2), avoid=1.4)
    t.start = (sx, sy, math.atan2(hy - sy, hx - sx) + rng.uniform(-0.5, 0.5))
    return t


def gen_bump_ring(rng):
    t = Tile("bump_ring")
    n = rng.integers(4, 9)
    yaw0 = rng.uniform(-math.pi, math.pi)
    bearings = np.sort(rng.uniform(-math.pi, math.pi, n))
    for b in bearings:
        t.marks.append(bump_strip(0, 0, yaw0, b, width=rng.uniform(0.4, 0.6)))
        if rng.random() < 0.35:  # this one was real: an invisible post / glass there
            a = yaw0 + b
            d = ROBOT_R + rng.uniform(0.01, 0.05)
            t.prims.append(Prim("cyl", (d * math.cos(a), d * math.sin(a), rng.uniform(0.01, 0.03)), 0, 0.9, "none"))
    t.prims += _clutter(rng, rng.integers(1, 4), avoid=0.8)
    t.start = (0.0, 0.0, yaw0)
    return t


def gen_cliff(rng):
    t = Tile("cliff_edge")
    x = ROBOT_R + rng.uniform(0.02, 0.12)
    t.holes.append((x, -rng.uniform(0.7, 1.9), 1.95, rng.uniform(0.7, 1.9)))
    if rng.random() < 0.5:
        y = rng.choice([-1, 1]) * rng.uniform(0.3, 0.8)
        t.prims += railing(x - 0.02, y, x - 0.02, 1.9 * np.sign(y), rng)
    t.prims += _clutter(rng, rng.integers(0, 2), avoid=1.3)
    t.start = (0.0, 0.0, rng.uniform(-0.6, 0.6))
    return t


def gen_clutter(rng):
    t = Tile("clutter")
    # boxes in a rough ring with 1-2 exits
    n = rng.integers(5, 9)
    angs = np.sort(rng.uniform(0, 2 * math.pi, n))
    for a in angs:
        d = ROBOT_R + rng.uniform(0.04, 0.25)
        s = rng.uniform(0.15, 0.45)
        t.prims.append(box((d + s / 2) * math.cos(a), (d + s / 2) * math.sin(a), s, rng.uniform(0.15, 0.5), a, 0, rng.uniform(0.08, 0.6),
                           depth="solid" if rng.random() < 0.8 else "none"))
    t.prims += _clutter(rng, rng.integers(0, 3), avoid=1.2)
    t.start = (0.0, 0.0, rng.uniform(-math.pi, math.pi))
    return t


GENERATORS = {
    "nose_wall": gen_nose_wall,
    "corner": gen_corner,
    "chair_legs": gen_chair_legs,
    "narrow_gap": gen_gap,
    "threshold": gen_threshold,
    "railing": gen_railing,
    "overhang": gen_overhang,
    "door_leaf": gen_door,
    "bump_ring": gen_bump_ring,
    "cliff_edge": gen_cliff,
    "clutter": gen_clutter,
}


# ---------------------------------------------------------------- finishing a tile


def _perimeter():
    h = TILE / 2 - 0.05
    return [wall(-h, -h, h, -h, h=0.6), wall(h, -h, h, h, h=0.6), wall(h, h, -h, h, h=0.6), wall(-h, h, -h, -h, h=0.6)]


def _rotate_tile(t: Tile, k: int) -> Tile:
    rot = k * math.pi / 2
    c, s = round(math.cos(rot)), round(math.sin(rot))
    prims = [p.transformed(rot) for p in t.prims]

    def rh(h):
        x0, y0, x1, y1 = h
        pts = [(c * x - s * y, s * x + c * y) for x, y in ((x0, y0), (x1, y1))]
        return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))

    marks = [np.asarray(m) @ np.array([[c, s], [-s, c]]) for m in t.marks]
    sx, sy, syaw = t.start
    return Tile(t.kind, prims, [rh(h) for h in t.holes], marks, (c * sx - s * sy, s * sx + c * sy, syaw + rot), t.goal, t.source)


def finish(t: Tile, rng, rotate=True) -> Tile | None:
    """Rotate, add the perimeter, resolve 'sparse', check the start pose, pick a goal."""
    if rotate:
        t = _rotate_tile(t, int(rng.integers(4)))
    for p in t.prims:
        if p.depth == "sparse":
            p.depth = "solid" if rng.random() < 0.5 else "none"
    t.prims += _perimeter()
    body = clearance_field(rasterise(t, "body"))
    sx, sy, syaw = t.start
    # back off along -heading until the body doesn't overlap anything
    for _ in range(12):
        if lookup(body, sx, sy) >= ROBOT_R + 0.01:
            break
        sx, sy = sx - 0.025 * math.cos(syaw), sy - 0.025 * math.sin(syaw)
    else:
        return None
    for x0, y0, x1, y1 in t.holes:  # wheels / casters must be on the floor
        dx = max(x0 - sx, 0.0, sx - x1)
        dy = max(y0 - sy, 0.0, sy - y1)
        if math.hypot(dx, dy) < ROBOT_R:
            return None
    clear = clearance_field(rasterise(t, "block"))
    c0 = lookup(clear, sx, sy)
    if c0 >= SUCCESS_CLEARANCE - 0.02:
        return None  # not stuck
    # goal: a clear spot 1.0-2.3 m away
    ys, xs = np.nonzero(clear >= 0.4)
    gx = (xs + 0.5) * GRID_RES - TILE / 2
    gy = (ys + 0.5) * GRID_RES - TILE / 2
    d = np.hypot(gx - sx, gy - sy)
    ok = (d > 1.0) & (d < 2.3) & (np.abs(gx) < 1.7) & (np.abs(gy) < 1.7)
    if not ok.any():
        return None
    i = rng.choice(np.nonzero(ok)[0])
    t.start = (float(sx), float(sy), float(math.atan2(math.sin(syaw), math.cos(syaw))))
    t.goal = (float(gx[i]), float(gy[i]))
    return t


DEFAULT_WEIGHTS = {
    "nose_wall": 1.0,
    "corner": 1.0,
    "chair_legs": 1.2,
    "narrow_gap": 1.0,
    "threshold": 0.8,
    "railing": 1.2,
    "overhang": 1.0,
    "door_leaf": 0.8,
    "bump_ring": 1.0,
    "cliff_edge": 0.8,
    "clutter": 1.0,
}


def generate(n: int, seed: int, weights: dict | None = None) -> list[Tile]:
    rng = np.random.default_rng(seed)
    w = weights or DEFAULT_WEIGHTS
    kinds = list(w)
    prob = np.array([w[k] for k in kinds], dtype=float)
    prob /= prob.sum()
    out = []
    while len(out) < n:
        k = kinds[rng.choice(len(kinds), p=prob)]
        t = finish(GENERATORS[k](rng), rng)
        if t is not None:
            out.append(t)
    return out


def save(tiles: list[Tile], path: str):
    with open(path, "w") as f:
        json.dump([t.to_json() for t in tiles], f)


def load(path: str) -> list[Tile]:
    with open(path) as f:
        return [Tile.from_json(d) for d in json.load(f)]


if __name__ == "__main__":
    import collections
    import time

    t0 = time.time()
    tiles = generate(200, 1)
    print(f"{len(tiles)} tiles in {time.time() - t0:.1f} s", collections.Counter(t.kind for t in tiles))
