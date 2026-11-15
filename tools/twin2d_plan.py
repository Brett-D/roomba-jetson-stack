#!/usr/bin/env python3
"""Builds a 2D digital-twin world from a vector floor plan (instead of RTAB-Map's grid).

  tools/twin2d_plan.py --plan /data/twin/isaac/remodel/floorplan.json --out /data/twin/twin2d/world_plan

The plan is JSON in the map frame (metres). The reader is deliberately tolerant about the layout:
lists under keys such as walls, doors, doorways/openings, rooms/spaces, stairs/stair_openings/
drop_offs, railings, obstacles/furniture (or GeoJSON-like "features" whose properties name the
type, or "levels"/"floors" holding those), with points as [x, y] or {"x", "y"}, polylines under
points/polyline/vertices/coordinates, segments as start/end, a/b, p1/p2, from/to or x1..y2, and
doors also as centre + width + angle. Unrecognised entries are listed in <name>_parse.json.

Output (same conventions as tools/twin2d_world.py, so twin2d.launch.py world:=<out>/<name>.yaml works):
  <name>.pgm/.yaml    5 cm grid aligned to --like (default the RTAB-Map world, so cells line up):
                      free floor 254, walls 0, outside the house 205 (the twin treats both as solid).
                      Free floor = the rooms (if the plan has rooms) or everything enclosed by the
                      walls; walls are drawn at their thickness; doorways and doors are cut open.
  <name>_overlay.yaml the plan's objects for the twin (`twin_overlay:` in the YAML): every door as a
                      closable door (open unless the plan says closed; scenarios refer to them by
                      name), stair openings / drop-offs as cliffs (the camera looks down into them),
                      railings as railings the depth camera can't see (bumpers and light bumpers do).
  <name>_preview.png  the grid with doors, railings, cliffs, rooms and a 1 m grid.
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import yaml

FREE, OCC, UNK = 254, 0, 205

CATEGORIES = {
    'wall': ('walls', 'wall_segments', 'wall', 'exterior_walls', 'interior_walls', 'partitions'),
    'door': ('doors', 'door'),
    'doorway': ('doorways', 'openings', 'doorway', 'opening', 'passages', 'archways'),
    'room': ('rooms', 'spaces', 'areas', 'room', 'space', 'zones'),
    'cliff': ('stairs', 'stair_openings', 'stair_opening', 'drop_offs', 'dropoffs', 'drop_off', 'cliffs',
              'voids', 'floor_openings', 'stairwells', 'stairwell'),
    'railing': ('railings', 'railing', 'rails', 'balustrades', 'guardrails', 'handrails'),
    'obstacle': ('obstacles', 'furniture', 'fixtures'),
}
KIND_WORDS = {
    'wall': ('wall', 'partition'),
    'door': ('door',),
    'doorway': ('doorway', 'opening', 'passage', 'archway'),
    'room': ('room', 'space', 'area'),
    'cliff': ('stair', 'drop', 'cliff', 'void', 'well'),
    'railing': ('rail', 'balustrade', 'banister'),
    'obstacle': ('obstacle', 'furniture', 'fixture'),
}


def pt(p):
    if isinstance(p, dict):
        return (float(p['x']), float(p['y']))
    return (float(p[0]), float(p[1]))


def points_of(obj):
    """The points of an entry: a polyline/polygon or a segment; None if it has no geometry."""
    if isinstance(obj, list) and obj and isinstance(obj[0], (list, tuple, dict)):
        try:
            return [pt(q) for q in obj]
        except (KeyError, TypeError, ValueError, IndexError):
            return None
    if not isinstance(obj, dict):
        return None
    geom = obj.get('geometry')
    if isinstance(geom, dict) and 'coordinates' in geom:
        c = geom['coordinates']
        if geom.get('type') == 'Polygon':
            c = c[0]
        return [pt(q) for q in c]
    for key in ('points', 'polyline', 'polygon', 'vertices', 'coords', 'coordinates', 'path', 'line',
                'segment', 'outline', 'boundary', 'corners'):
        if key in obj and isinstance(obj[key], list):
            c = obj[key]
            if c and isinstance(c[0], list) and c[0] and isinstance(c[0][0], list):
                c = c[0]
            return [pt(q) for q in c]
    for a, b in (('start', 'end'), ('a', 'b'), ('p1', 'p2'), ('from', 'to'), ('p0', 'p1')):
        if a in obj and b in obj:
            return [pt(obj[a]), pt(obj[b])]
    if all(k in obj for k in ('x1', 'y1', 'x2', 'y2')):
        return [(float(obj['x1']), float(obj['y1'])), (float(obj['x2']), float(obj['y2']))]
    centre = obj.get('center', obj.get('centre', obj.get('position')))
    width = obj.get('width', obj.get('opening_width'))
    if centre is not None and width is not None:
        c = pt(centre)
        ang = float(obj.get('angle', obj.get('orientation', obj.get('yaw', 0.0))))
        if abs(ang) > 2 * math.pi + 1e-6:
            ang = math.radians(ang)
        h = float(width) / 2.0
        return [(c[0] - h * math.cos(ang), c[1] - h * math.sin(ang)), (c[0] + h * math.cos(ang), c[1] + h * math.sin(ang))]
    return None


def kind_of(obj):
    if not isinstance(obj, dict):
        return None
    props = obj.get('properties') if isinstance(obj.get('properties'), dict) else {}
    for src in (props, obj):
        for key in ('type', 'kind', 'category', 'class', 'layer'):
            v = src.get(key)
            if not isinstance(v, str) or v.lower() in ('feature', 'featurecollection'):
                continue
            low = v.lower()
            # "stair_opening" before "opening", "doorway" / "opening" before "door"
            for cat in ('cliff', 'railing', 'doorway'):
                if any(w in low for w in KIND_WORDS[cat]):
                    return cat
            for cat, words in KIND_WORDS.items():
                if any(w in low for w in words):
                    return cat
    return None


def collect(plan):
    """{category: [entries]} and the keys that weren't understood."""
    found = {c: [] for c in CATEGORIES}
    unknown = []

    def visit(container, where):
        if isinstance(container, dict):
            for key, val in container.items():
                low = key.lower()
                cat = next((c for c, keys in CATEGORIES.items() if low in keys), None)
                if cat and isinstance(val, list):
                    for e in val:
                        found[kind_of(e) or cat].append(e)
                elif cat and isinstance(val, dict):
                    found[cat].append(val)
                elif low in ('features', 'elements', 'objects', 'items', 'entities') and isinstance(val, list):
                    for e in val:
                        k = kind_of(e)
                        if k:
                            found[k].append(e)
                        else:
                            unknown.append(f'{where}.{key}[]: {str(e)[:80]}')
                elif low in ('levels', 'floors', 'storeys', 'stories', 'plan', 'floorplan', 'floor_plan', 'layers'):
                    for k, sub in (val.items() if isinstance(val, dict) else enumerate(val)):
                        visit(sub, f'{where}.{key}.{k}')
                elif low not in ('frame', 'frame_id', 'units', 'unit', 'name', 'description', 'version', 'source',
                                 'notes', 'metadata', 'scale', 'origin', 'resolution', 'crs', 'type', 'bounds', 'id'):
                    unknown.append(f'{where}.{key}')
        elif isinstance(container, list):
            for k, sub in enumerate(container):
                visit(sub, f'{where}[{k}]')

    visit(plan, '')
    # Railings described inside stair entries.
    for s in list(found['cliff']):
        if isinstance(s, dict):
            for key in ('railing', 'railings', 'rail', 'balustrade', 'guardrail'):
                r = s.get(key)
                if isinstance(r, list) and r and isinstance(r[0], dict):
                    found['railing'].extend(r)
                elif r is not None:
                    found['railing'].append(r if isinstance(r, dict) else {'points': r})
    return found, unknown


def name_of(obj, default):
    if isinstance(obj, dict):
        props = obj.get('properties') if isinstance(obj.get('properties'), dict) else {}
        for key in ('name', 'id', 'label'):
            v = obj.get(key, props.get(key))
            if v is not None:
                return str(v)
    return default


def num(obj, keys, default):
    if isinstance(obj, dict):
        props = obj.get('properties') if isinstance(obj.get('properties'), dict) else {}
        for key in keys:
            v = obj.get(key, props.get(key))
            if isinstance(v, (int, float)):
                return float(v)
    return default


def door_closed(obj):
    if not isinstance(obj, dict):
        return False
    state = str(obj.get('state', obj.get('status', ''))).lower()
    if state:
        return state.startswith('close') or state == 'shut'
    if 'closed' in obj:
        return bool(obj['closed'])
    if 'open' in obj and isinstance(obj['open'], bool):
        return not obj['open']
    return False


class Raster:
    def __init__(self, origin, res, w, h):
        self.origin, self.res, self.w, self.h = origin, res, w, h

    def px(self, p):
        return (int(round((p[0] - self.origin[0]) / self.res - 0.5)), int(round(self.h - 1 - ((p[1] - self.origin[1]) / self.res - 0.5))))

    def poly(self, img, pts, value):
        cv2.fillPoly(img, [np.array([self.px(p) for p in pts], np.int32)], value)

    def line(self, img, pts, thickness_m, value):
        t = max(1, int(round(thickness_m / self.res)))
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, self.px(a), self.px(b), value, t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', default='/data/twin/isaac/remodel/floorplan.json')
    ap.add_argument('--out', default='/data/twin/twin2d/world_plan')
    ap.add_argument('--name', default='plan')
    ap.add_argument('--like', default='/data/twin/twin2d/world/house.yaml',
                    help='align cells with this map YAML (empty: grid from the plan bounds)')
    ap.add_argument('--res', type=float, default=0.05)
    ap.add_argument('--wall-thickness', type=float, default=0.10, help='for walls without a thickness')
    ap.add_argument('--railing-height', type=float, default=0.9)
    a = ap.parse_args()

    with open(a.plan) as f:
        plan = json.load(f)
    found, unknown = collect(plan)
    geo = {c: [(e, points_of(e)) for e in es] for c, es in found.items()}
    skipped = [f'{c}: {str(e)[:80]}' for c, es in geo.items() for e, p in es if not p]
    geo = {c: [(e, p) for e, p in es if p] for c, es in geo.items()}
    allpts = [q for es in geo.values() for _, p in es for q in p]
    if not allpts:
        raise SystemExit(f'no geometry understood in {a.plan}; unrecognised: {unknown[:10]}')
    xs, ys = [p[0] for p in allpts], [p[1] for p in allpts]
    lo, hi = (min(xs) - 1.0, min(ys) - 1.0), (max(xs) + 1.0, max(ys) + 1.0)
    res = a.res
    if a.like and os.path.isfile(a.like):
        with open(a.like) as f:
            like = yaml.safe_load(f)
        res = float(like['resolution'])
        lox, loy = like['origin'][0], like['origin'][1]
        img_like = cv2.imread(os.path.join(os.path.dirname(a.like), like['image']), cv2.IMREAD_UNCHANGED)
        hix, hiy = lox + img_like.shape[1] * res, loy + img_like.shape[0] * res
        # Keep the cell alignment, cover both.
        ox = lox - math.ceil(max(0.0, lox - lo[0]) / res) * res
        oy = loy - math.ceil(max(0.0, loy - lo[1]) / res) * res
        w = int(math.ceil((max(hix, hi[0]) - ox) / res))
        h = int(math.ceil((max(hiy, hi[1]) - oy) / res))
    else:
        ox, oy = lo
        w, h = int(math.ceil((hi[0] - lo[0]) / res)), int(math.ceil((hi[1] - lo[1]) / res))
    r = Raster((ox, oy), res, w, h)

    walls = np.zeros((h, w), np.uint8)
    for e, p in geo['wall']:
        t = num(e, ('thickness', 'width', 't'), a.wall_thickness)
        closed = isinstance(e, dict) and (e.get('closed') or (e.get('geometry') or {}).get('type') == 'Polygon')
        r.line(walls, p + ([p[0]] if closed else []), t, 1)
    openings = np.zeros((h, w), np.uint8)
    for e, p in geo['doorway'] + geo['door']:
        if len(p) == 2:
            t = num(e, ('wall_thickness', 'depth'), a.wall_thickness) + 2 * res
            r.line(openings, p, t, 1)
        else:
            r.poly(openings, p, 1)
    if geo['room']:
        free = np.zeros((h, w), np.uint8)
        for _, p in geo['room']:
            r.poly(free, p, 1)
        free = (free.astype(bool) & ~walls.astype(bool)) | openings.astype(bool)
    else:
        # Everything the walls (doors closed) enclose; the outside is what the border reaches.
        barrier = walls.copy()
        for e, p in geo['door']:
            r.line(barrier, p, 0.06, 1)
        open_space = (barrier == 0).astype(np.uint8)
        n, labels = cv2.connectedComponents(open_space, connectivity=4)
        border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
        outside = np.isin(labels, list(border)) & (open_space == 1)
        free = (walls == 0) & ~outside
        free |= openings.astype(bool)
    img = np.full((h, w), UNK, np.uint8)
    img[free] = FREE
    img[(walls == 1) & ~openings.astype(bool)] = OCC

    os.makedirs(a.out, exist_ok=True)
    cv2.imwrite(os.path.join(a.out, a.name + '.pgm'), img)
    overlay = {'name': a.name, 'description': f'from {a.plan} (tools/twin2d_plan.py)', 'doors': [], 'railings': [],
               'obstacles': [], 'cliffs': []}
    for k, (e, p) in enumerate(geo['door']):
        d = {'name': name_of(e, f'door_{k + 1}'), 'closed': door_closed(e), 'thickness': 0.05}
        if len(p) == 2:
            d['segment'] = [list(p[0]), list(p[1])]
        else:
            d['polygon'] = [list(q) for q in p]
        overlay['doors'].append(d)
    for k, (e, p) in enumerate(geo['cliff']):
        overlay['cliffs'].append({'name': name_of(e, f'stairs_{k + 1}'), 'polygon': [list(q) for q in p]})
    for k, (e, p) in enumerate(geo['railing']):
        overlay['railings'].append({'name': name_of(e, f'railing_{k + 1}'), 'points': [list(q) for q in p],
                                    'depth': 'none', 'height_m': num(e, ('height', 'height_m'), a.railing_height)})
    for k, (e, p) in enumerate(geo['obstacle']):
        o = {'name': name_of(e, f'obstacle_{k + 1}'), 'zmin': num(e, ('zmin', 'bottom'), 0.0),
             'zmax': num(e, ('zmax', 'height', 'top'), 0.5)}
        if len(p) >= 3:
            o['polygon'] = [list(q) for q in p]
        else:
            o['segment'] = [list(p[0]), list(p[-1])]
        overlay['obstacles'].append(o)
    with open(os.path.join(a.out, a.name + '_overlay.yaml'), 'w') as f:
        f.write(f'# The objects of {a.plan} for twin2d (tools/twin2d_plan.py); scenarios refer to doors by name.\n')
        yaml.safe_dump(overlay, f, default_flow_style=None, sort_keys=False)
    with open(os.path.join(a.out, a.name + '.yaml'), 'w') as f:
        f.write(f'# twin world from the floor plan {a.plan} (tools/twin2d_plan.py); not free = solid wall\n')
        yaml.safe_dump({'image': a.name + '.pgm', 'resolution': res, 'origin': [round(ox, 6), round(oy, 6), 0.0],
                        'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196,
                        'twin_overlay': a.name + '_overlay.yaml'}, f, default_flow_style=None, sort_keys=False)
    summary = {c: len(es) for c, es in geo.items()}
    start_free = bool(img[r.px((0.0, 0.0))[1], r.px((0.0, 0.0))[0]] == FREE) if 0 <= r.px((0, 0))[0] < w and 0 <= r.px((0, 0))[1] < h else False
    with open(os.path.join(a.out, a.name + '_parse.json'), 'w') as f:
        json.dump({'plan': a.plan, 'understood': summary, 'skipped_without_geometry': skipped,
                   'unrecognised_keys': unknown, 'free_m2': round(float(free.sum()) * res * res, 2),
                   'map_origin_free': start_free, 'grid': [w, h, res, ox, oy]}, f, indent=1)
    preview(img, r, geo, overlay, os.path.join(a.out, a.name + '_preview.png'))
    print(f'{a.name}: {w}x{h} cells at {res} m, free {free.sum() * res * res:.1f} m2; understood {summary}; '
          f'{len(skipped)} entries without geometry, {len(unknown)} unrecognised keys; map origin free: {start_free}')


def preview(img, r, geo, overlay, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ext = [r.origin[0], r.origin[0] + r.w * r.res, r.origin[1], r.origin[1] + r.h * r.res]
    fig, ax = plt.subplots(figsize=(12, 12 * r.h / max(r.w, 1)))
    ax.imshow(img, cmap='gray', extent=ext, interpolation='nearest', vmin=0, vmax=255)
    for _, p in geo['room']:
        q = np.array(p + [p[0]])
        ax.plot(q[:, 0], q[:, 1], color='tab:green', lw=0.6, alpha=0.6)
    for d in overlay['doors']:
        q = np.array(d.get('segment') or d.get('polygon'))
        ax.plot(q[:, 0], q[:, 1], color='saddlebrown', lw=3 if d['closed'] else 1.5)
    for c in overlay['cliffs']:
        q = np.array(c['polygon'] + [c['polygon'][0]])
        ax.plot(q[:, 0], q[:, 1], color='red', lw=1.5)
    for rl in overlay['railings']:
        q = np.array(rl['points'])
        ax.plot(q[:, 0], q[:, 1], color='purple', lw=2)
    ax.plot([0], [0], 'gs', ms=8)
    ax.set_xticks(np.arange(math.floor(ext[0]), ext[1] + 1, 1.0))
    ax.set_yticks(np.arange(math.floor(ext[2]), ext[3] + 1, 1.0))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == '__main__':
    main()
