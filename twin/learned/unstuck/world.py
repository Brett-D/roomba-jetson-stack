"""Lay out stuck tiles on a grid and build what the Isaac Lab scene and the env need.

- collision mesh (floor with holes, every primitive, a catch floor 0.35 m down)
- depth mesh (what the D1000 depth fan sees: floor + primitives with depth 'solid')
- IR mesh (what the light bumpers see: every primitive with ir)
- per-tile tensors: ground-truth clearance field, initial keep-out marks, holes, start, goal
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

import scenarios as S

CATCH_Z = -0.35


def _prim_mesh(p: S.Prim) -> trimesh.Trimesh:
    h = p.z1 - p.z0
    if p.kind == "box" and p.z0 <= 0.0 and p.z1 < S.STEP_BLOCK:
        # rugs / low transitions: bevelled edges (1:2), as real rug edges are; a vertical 1 cm step
        # stops the caster sphere in PhysX (face contact normal) although the robot climbs it
        cx, cy, sx, sy, yaw = p.p
        k = 2.0 * h
        bury = 0.01  # start the bevel under the floor so wheels meet the slope, not the toe edge
        pts = []
        for z, ex, ey in ((p.z0 - bury, sx / 2 + 2 * bury, sy / 2 + 2 * bury), (p.z1, max(sx / 2 - k, 0.01), max(sy / 2 - k, 0.01))):
            pts += [(-ex, -ey, z), (ex, -ey, z), (ex, ey, z), (-ex, ey, z)]
        m = trimesh.convex.convex_hull(np.array(pts))
        T = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
        T[:3, 3] = (cx, cy, 0.0)
        m.apply_transform(T)
        return m
    if p.kind == "box":
        cx, cy, sx, sy, yaw = p.p
        m = trimesh.creation.box(extents=(max(sx, 0.005), max(sy, 0.005), h))
        T = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
        T[:3, 3] = (cx, cy, p.z0 + h / 2)
        m.apply_transform(T)
        return m
    if p.kind == "cyl":
        cx, cy, r = p.p
        m = trimesh.creation.cylinder(radius=r, height=h, sections=8)
        m.apply_translation((cx, cy, p.z0 + h / 2))
        return m
    from shapely.geometry import Polygon

    poly = Polygon(p.p[0])
    if not poly.is_valid:
        poly = poly.buffer(0)
    m = trimesh.creation.extrude_polygon(poly, h)
    m.apply_translation((0, 0, p.z0))
    return m


def _floor_rects(holes, half):
    rects = [(-half, -half, half, half)]
    for hx0, hy0, hx1, hy1 in holes:
        out = []
        for x0, y0, x1, y1 in rects:
            if hx1 <= x0 or hx0 >= x1 or hy1 <= y0 or hy0 >= y1:
                out.append((x0, y0, x1, y1))
                continue
            ix0, iy0, ix1, iy1 = max(x0, hx0), max(y0, hy0), min(x1, hx1), min(y1, hy1)
            if iy0 > y0:
                out.append((x0, y0, x1, iy0))
            if iy1 < y1:
                out.append((x0, iy1, x1, y1))
            if ix0 > x0:
                out.append((x0, iy0, ix0, iy1))
            if ix1 < x1:
                out.append((ix1, iy0, x1, iy1))
        rects = out
    return rects


def _floor_mesh(holes):
    ms = []
    for x0, y0, x1, y1 in _floor_rects(holes, S.TILE / 2):
        m = trimesh.creation.box(extents=(x1 - x0, y1 - y0, 0.04))
        m.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, -0.02))
        ms.append(m)
    return ms


@dataclass
class TileWorld:
    tiles: list
    centers: np.ndarray  # [T, 2]
    collision: trimesh.Trimesh
    depth: trimesh.Trimesh
    ir: trimesh.Trimesh
    clearance: np.ndarray  # [T, N, N] float32
    marks: np.ndarray  # [T, N, N] bool (initial keep-out)
    holes: np.ndarray  # [T, H, 4] (tile frame; empty = nan)
    starts: np.ndarray  # [T, 3]
    goals: np.ndarray  # [T, 2]
    kinds: list


def _mark_raster(marks):
    X, Y = S._cell_centres()
    occ = np.zeros(X.shape, dtype=bool)
    for m in marks:
        occ |= S._poly_mask(np.asarray(m, dtype=float), X, Y, 0.0)
    return occ


def build(tiles: list, spacing: float = S.TILE) -> TileWorld:
    T = len(tiles)
    cols = int(math.ceil(math.sqrt(T)))
    centers = np.array([((i % cols) - (cols - 1) / 2, (i // cols) - (cols - 1) / 2) for i in range(T)]) * spacing
    col, dep, ir = [], [], []
    clear = np.zeros((T, S.GRID_N, S.GRID_N), np.float32)
    marks = np.zeros((T, S.GRID_N, S.GRID_N), bool)
    hmax = max(1, max(len(t.holes) for t in tiles))
    holes = np.full((T, hmax, 4), np.nan, np.float32)
    for i, t in enumerate(tiles):
        cx, cy = centers[i]
        floor = _floor_mesh(t.holes)
        prims = [(p, _prim_mesh(p)) for p in t.prims]
        for m in floor:
            m.apply_translation((cx, cy, 0))
        for _, m in prims:
            m.apply_translation((cx, cy, 0))
        col += floor + [m for p, m in prims if p.z1 >= S.STEP_BLOCK]  # rugs: no collider (see docs)
        dep += floor + [m for p, m in prims if p.depth == "solid"]
        ir += [m for p, m in prims if p.ir]
        clear[i] = S.clearance_field(S.rasterise(t, "block"))
        marks[i] = _mark_raster(t.marks)
        for k, h in enumerate(t.holes):
            holes[i, k] = h
    ext = np.abs(centers).max() + spacing
    catch = trimesh.creation.box(extents=(2 * ext, 2 * ext, 0.05))
    catch.apply_translation((0, 0, CATCH_Z - 0.025))
    collision = trimesh.util.concatenate(col + [catch])
    depth = trimesh.util.concatenate(dep + [catch])
    irm = trimesh.util.concatenate(ir)
    return TileWorld(
        tiles,
        centers.astype(np.float32),
        collision,
        depth,
        irm,
        clear,
        marks,
        holes,
        np.array([t.start for t in tiles], np.float32),
        np.array([t.goal for t in tiles], np.float32),
        [t.kind for t in tiles],
    )


_CACHE: dict = {}


def load_world(path: str) -> TileWorld:
    if path not in _CACHE:
        _CACHE[path] = build(S.load(path))
    return _CACHE[path]


if __name__ == "__main__":
    import sys
    import time

    t0 = time.time()
    w = build(S.generate(64, 5))
    print(f"built in {time.time() - t0:.1f}s; tris collision {len(w.collision.faces)} depth {len(w.depth.faces)} ir {len(w.ir.faces)}")
    sys.exit(0)
