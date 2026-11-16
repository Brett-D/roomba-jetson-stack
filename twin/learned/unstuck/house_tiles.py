"""Held-out stuck tiles cut from the real house's floor plans (remodelled twin) (vector floor plans, map frame).

Stuck poses are placed where the real robot gets stuck: nose to a wall, in room corners, at the
stair / balcony railings (posts invisible to the depth camera, stair void behind), against
furniture and bathroom fixtures, at door jambs. Each tile is the 4 x 4 m of the plan around the
pose (walls, doors' openings, furniture, fixtures, railings as thin posts, the stair void as
cliff holes), in map-aligned axes.

  python house_tiles.py --plans E:/twin/assets/remodel --out data/tiles_house.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

import scenarios as S

HALF = S.TILE / 2 - 0.1


def _polys(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]


FIXTURE_Z = {
    "toilet": (0.0, 0.40),
    "vanity": (0.0, 0.86),
    "vanity_double": (0.0, 0.86),
    "tub": (0.0, 0.55),
    "tub_freestanding": (0.0, 0.55),
    "shower": (0.0, 0.06),  # curb
    "closet_shelf": (1.0, 1.05),
}


class Plan:
    def __init__(self, path):
        d = json.load(open(path))
        self.name = os.path.basename(path)
        self.walls = [Polygon(w).buffer(0) for w in d["walls"]]
        self.rooms = [(r, Polygon(r["polygon"]).buffer(0)) for r in d["rooms"] if not r.get("outdoor") and r["type"] != "balcony"]
        self.floor = unary_union([p for _, p in self.rooms])
        self.wall_u = unary_union(self.walls)
        self.furn = []
        for f in d.get("furniture", []):
            h = float(f.get("height_m", 0.8))
            z0 = 0.0
            if f.get("role") in ("lamp", "plant"):
                h = min(h, 0.6)
            self.furn.append((Polygon(f["footprint"]).buffer(0), z0, h, f.get("role", "")))
        for f in d.get("fixtures", []):
            z0, z1 = FIXTURE_Z.get(f["type"], (0.0, 0.8))
            self.furn.append((Polygon(f["footprint"]).buffer(0), z0, z1, f["type"]))
        self.void = Polygon(d["stair_void"]["polygon"]).buffer(0) if d.get("stair_void") else None
        self.railings = [np.asarray(r["polyline"], float) for r in d.get("railings", [])]
        self.doors = [np.asarray(dr["segment"], float) for dr in d["doors"]]
        solid = [self.wall_u] + [p for p, z0, z1, _ in self.furn if z0 < S.BLOCK_Z and z1 >= S.STEP_BLOCK]
        if self.void is not None:
            solid.append(self.void)
        self.solid = unary_union(solid)
        # outside the house (not a room, not a wall) is treated as solid wall
        self.outside = box(*self.floor.buffer(3).bounds).difference(self.floor.union(self.wall_u))

    def free_at(self, x, y, r=S.ROBOT_R + 0.01):
        p = Point(x, y)
        return self.floor.contains(p) and p.distance(self.solid) >= r and p.distance(self.outside) >= r

    def tile(self, cx, cy, yaw, kind, rng) -> S.Tile | None:
        tb = box(cx - HALF, cy - HALF, cx + HALF, cy + HALF)
        prims = []

        def add_poly(geom, z0, z1, depth="solid"):
            for g in _polys(geom.intersection(tb)):
                if g.area < 1e-4:
                    continue
                g = g.simplify(0.01)
                pts = tuple((float(x - cx), float(y - cy)) for x, y in list(g.exterior.coords)[:-1])
                if len(pts) >= 3:
                    prims.append(S.Prim("poly", (pts,), z0, z1, depth))

        add_poly(self.wall_u, 0.0, 2.44)
        add_poly(self.outside, 0.0, 2.44)
        for p, z0, z1, role in self.furn:
            add_poly(p, z0, z1)
        for rl in self.railings:
            line = LineString(rl)
            if line.distance(Point(cx, cy)) > 3.0:
                continue
            L = line.length
            spacing = rng.uniform(0.1, 0.12)
            for s in np.arange(0.0, L + 1e-6, spacing):
                q = line.interpolate(s)
                if tb.contains(q):
                    prims.append(S.Prim("cyl", (q.x - cx, q.y - cy, 0.012), 0.0, 0.92, "none"))
            for i in range(len(rl) - 1):
                seg = LineString(rl[i:i + 2]).intersection(tb)
                if seg.is_empty or seg.length < 0.05:
                    continue
                (x0, y0), (x1, y1) = list(seg.coords)[0], list(seg.coords)[-1]
                w = S.wall(x0 - cx, y0 - cy, x1 - cx, y1 - cy, t=0.05, h=0.96, depth="none")
                w.z0 = 0.9
                prims.append(w)
        holes = []
        if self.void is not None and self.void.intersects(tb):
            v = self.void.intersection(tb)
            step = 0.1
            for yy in np.arange(cy - HALF, cy + HALF, step):
                row = []
                for xx in np.arange(cx - HALF, cx + HALF, step):
                    if v.contains(Point(xx + step / 2, yy + step / 2)):
                        row.append(xx)
                # merge runs
                i = 0
                while i < len(row):
                    j = i
                    while j + 1 < len(row) and abs(row[j + 1] - row[j] - step) < 1e-6:
                        j += 1
                    holes.append((float(row[i] - cx), float(yy - cy), float(row[j] + step - cx), float(yy + step - cy)))
                    i = j + 1
        t = S.Tile(kind, prims, holes, [], (0.0, 0.0, float(yaw)), (1.5, 0.0), source=self.name)
        return S.finish(t, rng, rotate=False)


def _edges(poly: Polygon):
    c = list(poly.exterior.coords)
    for i in range(len(c) - 1):
        yield np.array(c[i]), np.array(c[i + 1])


def stuck_poses(plan: Plan, rng, n_per_kind=12):
    out = []

    def facing(p, n, gap):
        """Robot centre gap + R off the surface point p along n (into free space), facing -n."""
        c = p + n * (S.ROBOT_R + gap)
        return c[0], c[1], math.atan2(-n[1], -n[0]) + rng.uniform(-0.35, 0.35)

    def surface_poses(polys, kind, n_target):
        tries = 0
        got = 0
        polys = [p for p in polys if p.area > 0.01]
        while got < n_target and tries < 400 and polys:
            tries += 1
            poly = polys[rng.integers(len(polys))]
            edges = [e for e in _edges(poly) if np.linalg.norm(e[1] - e[0]) > 0.4]
            if not edges:
                continue
            a, b = edges[rng.integers(len(edges))]
            p = a + (b - a) * rng.uniform(0.2, 0.8)
            t = (b - a) / np.linalg.norm(b - a)
            for n in (np.array([-t[1], t[0]]), np.array([t[1], -t[0]])):
                x, y, yaw = facing(p, n, rng.uniform(0.0, 0.1))
                if plan.free_at(x, y, S.ROBOT_R - 0.005) and not poly.contains(Point(p + n * 0.05)):
                    out.append((x, y, yaw, kind))
                    got += 1
                    break

    surface_poses(plan.walls, "house_wall", n_per_kind)
    surface_poses([p for p, z0, z1, role in plan.furn if z0 < S.BLOCK_Z and z1 > 0.3], "house_furniture", n_per_kind)
    # room corners
    got = 0
    for _ in range(300):
        if got >= n_per_kind:
            break
        r, poly = plan.rooms[rng.integers(len(plan.rooms))]
        c = list(poly.exterior.coords)[:-1]
        i = rng.integers(len(c))
        v = np.array(c[i])
        a = np.array(c[i - 1]) - v
        b = np.array(c[(i + 1) % len(c)]) - v
        if np.linalg.norm(a) < 0.5 or np.linalg.norm(b) < 0.5:
            continue
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        ang = math.acos(np.clip(a @ b, -1, 1))
        if ang > math.radians(135) or ang < math.radians(30):
            continue
        bis = (a + b) / np.linalg.norm(a + b)
        d = (S.ROBOT_R + rng.uniform(0.0, 0.05) + 0.08) / math.sin(ang / 2)
        x, y = v + bis * d
        if plan.free_at(x, y, S.ROBOT_R - 0.005):
            out.append((x, y, math.atan2(-bis[1], -bis[0]) + rng.uniform(-0.3, 0.3), "house_corner"))
            got += 1
    # railings
    got = 0
    for _ in range(300):
        if got >= n_per_kind or not plan.railings:
            break
        rl = plan.railings[rng.integers(len(plan.railings))]
        line = LineString(rl)
        s = rng.uniform(0.1, 0.9) * line.length
        p = np.array(line.interpolate(s).coords[0])
        q = np.array(line.interpolate(min(s + 0.05, line.length)).coords[0])
        t = (q - p) / max(np.linalg.norm(q - p), 1e-6)
        for n in (np.array([-t[1], t[0]]), np.array([t[1], -t[0]])):
            x, y, yaw = facing(p, n, rng.uniform(0.0, 0.1))
            pt = Point(x, y)
            void_ok = plan.void is None or (not plan.void.contains(pt) and pt.distance(plan.void) >= 0.1)
            if void_ok and plan.floor.buffer(0.05).contains(pt) and pt.distance(plan.wall_u) >= S.ROBOT_R and pt.distance(plan.outside) >= S.ROBOT_R:
                out.append((x, y, yaw, "house_railing"))
                got += 1
                break
    # door jambs
    got = 0
    for _ in range(200):
        if got >= n_per_kind:
            break
        seg = plan.doors[rng.integers(len(plan.doors))]
        end = seg[rng.integers(2)]
        t = (seg[1] - seg[0]) / np.linalg.norm(seg[1] - seg[0])
        n = np.array([-t[1], t[0]]) * rng.choice([-1, 1])
        p = end + n * 0.25 + (seg.mean(0) - end) / np.linalg.norm(seg.mean(0) - end) * 0.05
        yaw = math.atan2(end[1] - p[1], end[0] - p[0]) + rng.uniform(-0.4, 0.4)
        c = p - np.array([math.cos(yaw), math.sin(yaw)]) * 0.02
        if plan.free_at(c[0], c[1], S.ROBOT_R - 0.02):
            out.append((c[0], c[1], yaw, "house_door"))
            got += 1
    return out


if __name__ == "__main__":
    import collections

    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="E:/twin/assets/remodel")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tiles_house.json"))
    ap.add_argument("--n", type=int, default=14)
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    tiles = []
    for f in ("floorplan.json", "floorplan_floor1.json"):
        plan = Plan(os.path.join(a.plans, f))
        for x, y, yaw, kind in stuck_poses(plan, rng, a.n):
            t = plan.tile(x, y, yaw, kind, rng)
            if t is not None:
                tiles.append(t)
    S.save(tiles, a.out)
    print(len(tiles), dict(collections.Counter(t.kind for t in tiles)), "->", a.out)
