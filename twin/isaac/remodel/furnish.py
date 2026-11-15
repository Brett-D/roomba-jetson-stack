"""Furnish the plan: furniture from the grid's interior obstacle blobs (mapped rooms) plus
room-type templates (unmapped rooms / missing pieces). Works in plan pixels for placement and
returns items in plan metres.
"""
import json
import math

import cv2
import numpy as np

import plan_lib as pl
import remodel_lib as rl

# catalogue picks by role (ArchVis Residential names); sizes come from asset_catalog.json
ROLE_ASSETS = {
    'nightstand': ['Furniture/EndTables/Ellendale', 'Furniture/EndTables/Festus01'],
    'dresser': ['Furniture/MediaTables/Manchester', 'Furniture/Bookshelves/Fenton', 'Furniture/Bookshelves/Ferguson',
                'Furniture/MediaTables/Soulard'],
    'tall': ['Furniture/Bookshelves/Ferguson', 'Furniture/Bookshelves/Fenton_Var01', 'Furniture/Bookshelves/Dogtown'],
    'armchair': ['Furniture/Chairs/Bevo', 'Furniture/Chairs/Armchair', 'Furniture/Chairs/Meramec_Chair',
                 'Furniture/Chairs/Sappington_Chair'],
    'chair': ['Furniture/Chairs/Chair_Desk', 'Furniture/Chairs/Chair_Array', 'Furniture/Chairs/Chair_Wire'],
    'desk': ['Furniture/Desks/Desk_01'],
    'console': ['Furniture/SofaTables/AftonConsole', 'Furniture/SofaTables/Ellisville', 'Furniture/SofaTables/EurekaConsole'],
    'table': ['Furniture/Tables/Whittershins', 'Furniture/CoffeeTables/Midtown', 'Furniture/CoffeeTables/Make'],
    'sofa': ['Furniture/Sofas/Arnold'],
    'plant': ['Plants/PlantPot', 'Plants/Plant_01', 'Plants/Plant_02'],
    'lamp': ['Lighting/FloorLamp', 'Lighting/BrassFloorLamp', 'Lighting/Candice'],
    'ottoman': ['Furniture/Chairs/Bevo_Ottoman'],
    'media': ['Furniture/MediaTables/CreveCoeur', 'Furniture/MediaTables/Soulard'],
    # downstairs rooms
    'sofa_big': ['Furniture/Sofas/LemaySofa_Left', 'Furniture/Sofas/LemaySofa_Right', 'Furniture/Sofas/Arnold'],
    'coffee': ['Furniture/CoffeeTables/Midtown', 'Furniture/CoffeeTables/Make'],
    'dining_table': ['Furniture/Tables/Whittershins'],
    'dining_chair': ['Furniture/Chairs/Chair_Array', 'Furniture/Chairs/Chair_Wire'],
    'stool': ['Furniture/BarStools/MapleBarStool', 'Furniture/BarStools/bar_stool'],
    'shelf_big': ['Furniture/Bookshelves/Delmar', 'Furniture/Bookshelves/Costello'],
    'ladder': ['Misc/Ladder'],
}
BED_SIZES = {'king': (1.95, 2.12), 'queen': (1.55, 2.08), 'full': (1.40, 1.98), 'twin': (1.0, 1.95)}


class Catalog:
    def __init__(self, path):
        self.items = {f"{c['folder']}/{c['name']}": c for c in json.load(open(path)) if c['size_m'][0] > 0}

    def get(self, key):
        return self.items.get(key)

    def pick(self, role, rng, target=None):
        keys = [k for k in ROLE_ASSETS[role] if k in self.items]
        if not keys:
            return None
        if target is not None:          # closest footprint (long side, short side)
            def err(k):
                s = sorted(self.items[k]['size_m'][:2], reverse=True)
                return abs(s[0] - target[0]) + abs(s[1] - target[1])
            keys.sort(key=err)
            return self.items[keys[0]]
        return self.items[keys[rng.integers(len(keys))]]


def rect_poly_px(c, w, d, yaw):
    """Rotated rectangle corners (plan pixels); yaw in image coordinates."""
    co, si = math.cos(yaw), math.sin(yaw)
    pts = []
    for dx, dy in ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)):
        pts.append((c[0] + co * dx - si * dy, c[1] + si * dx + co * dy))
    return np.array(pts)


class Furnisher:
    def __init__(self, plan, frame, wall_mask, void_px, catalog, rng, path_px=None):
        self.plan, self.frame, self.cat, self.rng = plan, frame, catalog, rng
        self.shape = wall_mask.shape
        s = frame.s
        walls = cv2.dilate(wall_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        self.blocked = walls.copy()
        self.blocked |= rl.poly_mask(self.shape, void_px)
        for rail in pl.railing_polylines(plan):
            cv2.polylines(self.blocked.view(np.uint8), [np.round(np.asarray(rail['pts'])).astype(np.int32)], False, 1,
                          max(2, int(0.15 * s)))
        for f in plan['fixtures']:
            if 'rect' in f:
                x0, y0, x1, y1 = f['rect']
                self.blocked[int(y0) - 2:int(y1) + 2, int(x0) - 2:int(x1) + 2] = True
            else:
                cx, cy = f['center']
                r = int(0.45 * s)
                self.blocked[int(cy) - r:int(cy) + r, int(cx) - r:int(cx) + r] = True
        for d in plan['doors']:                     # door swing / passage clearance
            a, b = np.array(d['a']), np.array(d['b'])
            L = np.linalg.norm(b - a)
            u = (b - a) / max(L, 1e-9)
            n = np.array([-u[1], u[0]])
            depth = min(L, 0.8 * s) + 0.1 * s
            quad = np.array([a - n * depth, b - n * depth, b + n * depth, a + n * depth])
            self.blocked |= rl.poly_mask(self.shape, quad)
        if path_px is not None:
            pm = np.zeros(self.shape, np.uint8)
            cv2.polylines(pm, [np.round(path_px).astype(np.int32)], False, 1, int(2 * 0.19 * s))
            self.blocked |= pm.astype(bool)
        self.items = []
        self.fit_thr = 0.96

    def room_free(self, room):
        return rl.poly_mask(self.shape, room['poly']) & ~self.blocked

    def fits(self, room, poly_px):
        m = rl.poly_mask(self.shape, poly_px)
        if m.sum() == 0:
            return False
        free = self.room_free(room)
        return bool(free[m].mean() > self.fit_thr)

    def commit(self, poly_px, margin_m=0.05):
        m = rl.poly_mask(self.shape, poly_px).astype(np.uint8)
        k = max(1, int(margin_m * self.frame.s))
        self.blocked |= cv2.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)

    def add(self, room, role, c_px, w_m, d_m, yaw_img, asset=None, h_m=None, source='template', bed=None):
        s = self.frame.s
        poly = rect_poly_px(c_px, w_m * s, d_m * s, yaw_img)
        if not self.fits(room, poly):
            return False
        self.commit(poly)
        cm = self.frame.m(c_px)
        self.items.append({'role': role, 'room': room['name'], 'room_id': id(room), 'source': source, 'c': [float(cm[0]), float(cm[1])],
                           # image yaw (y down) -> plan yaw (y up)
                           'yaw': -yaw_img, 'size': [w_m, d_m], 'height': h_m,
                           'asset': asset['url'] if asset else None, 'asset_key': f"{asset['folder']}/{asset['name']}" if asset else None,
                           'bed': bed, 'footprint_m': self.frame.m(poly).tolist()})
        return True

    def wall_slots(self, room, min_len_m):
        """Room polygon edges usable for furniture: (a, b, inward normal) in pixels."""
        P = np.asarray(room['poly'], float)
        n = len(P)
        cen = P.mean(0)
        slots = []
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]
            L = np.linalg.norm(b - a)
            if L < min_len_m * self.frame.s:
                continue
            u = (b - a) / L
            nrm = np.array([-u[1], u[0]])
            mid = (a + b) / 2
            if not rl.poly_mask(self.shape, room['poly'])[int(np.clip(mid[1] + nrm[1] * 6, 0, self.shape[0] - 1)),
                                                          int(np.clip(mid[0] + nrm[0] * 6, 0, self.shape[1] - 1))]:
                nrm = -nrm
            slots.append((a, b, nrm, L))
        slots.sort(key=lambda t: -t[3])
        return slots

    def place_against_wall(self, room, role, w_m, d_m, asset=None, h_m=None, prefer=None, bed=None, tries=12):
        s = self.frame.s
        for a, b, nrm, L in self.wall_slots(room, w_m + 0.2):
            u = (b - a) / L
            yaw = math.atan2(u[1], u[0])
            fracs = (prefer or [0.5]) + list(self.rng.uniform(0.12, 0.88, tries))
            for f in fracs:
                along = a + u * (L * f)
                c = along + nrm * (0.15 * s + d_m * s / 2)     # half wall thickness + gap + half depth
                if self.add(room, role, c, w_m, d_m, yaw, asset, h_m, bed=bed):
                    # direction from the item towards the wall it backs onto (plan coords, y up)
                    self.items[-1]['back_dir'] = [float(-nrm[0]), float(nrm[1])]
                    return self.items[-1], (a, b, nrm, along)
        return None, None


def blobs_from_grid(occ, meta, heights, T, frame, plan, wall_mask, void_px, min_area_m2=0.12):
    """Occupied grid blobs inside plan rooms and away from plan walls -> [(center_px, w_m, d_m, yaw_img, h, room)]."""
    from scipy import ndimage
    res, (ox, oy) = meta['resolution'], meta['origin'][:2]
    ys, xs = np.nonzero(wall_mask)
    wq = pl.apply_T(T, frame.m(np.stack([xs, ys], 1).astype(float)))
    wall_g = np.zeros(occ.shape, bool)
    ij = np.floor((wq - [ox, oy]) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < occ.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < occ.shape[0])
    wall_g[ij[ok, 1], ij[ok, 0]] = True
    wall_g = ndimage.binary_dilation(wall_g, iterations=5)
    for rail in pl.railing_polylines(plan):
        q = pl.apply_T(T, frame.m(pl.densify(rail['pts'], 1.0)))
        ij = np.floor((q - [ox, oy]) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < occ.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < occ.shape[0])
        m = np.zeros(occ.shape, bool)
        m[ij[ok, 1], ij[ok, 0]] = True
        wall_g |= ndimage.binary_dilation(m, iterations=5)
    cand = occ & ~wall_g
    lab, n = ndimage.label(cand, structure=np.ones((3, 3)))
    out = []
    inv = inverse_T(T)
    for k in range(1, n + 1):
        cells = np.argwhere(lab == k)
        area = len(cells) * res * res
        if area < min_area_m2:
            continue
        pts = (cells[:, ::-1] + 0.5) * res + [ox, oy]                 # map metres
        pp = frame_px(frame, apply_inv(inv, pts))                      # plan pixels
        (cx, cy), (w, h), ang = cv2.minAreaRect(pp.astype(np.float32))
        wm, dm = w / frame.s, h / frame.s
        yaw = math.radians(ang)
        if dm > wm:
            wm, dm, yaw = dm, wm, yaw + math.pi / 2
        hh = float(np.percentile(heights[cells[:, 0], cells[:, 1]], 90))
        room = None
        for r in plan['rooms']:
            if not r.get('outdoor') and rl.poly_mask(wall_mask.shape, r['poly'])[int(np.clip(cy, 0, wall_mask.shape[0] - 1)),
                                                                                int(np.clip(cx, 0, wall_mask.shape[1] - 1))]:
                room = r
                break
        if room is None:
            continue
        if room['type'] in ('bathroom', 'closet'):
            continue
        out.append({'c_px': (cx, cy), 'w': wm, 'd': dm, 'yaw_img': yaw, 'h': hh, 'room': room, 'area': area})
    return out


def inverse_T(T):
    return T


def apply_inv(T, X):
    X = np.asarray(X, float)
    P = (X - np.asarray(T['t'])) / T['scale']
    th = math.radians(-T['angle'])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    P = P @ R.T
    if T['mirror']:
        P[:, 0] = -P[:, 0]
    return P


def frame_px(frame, P):
    P = np.asarray(P, float)
    return np.stack([P[:, 0] * frame.s + frame.cx, frame.cy - P[:, 1] * frame.s], 1)


def classify_blob(b):
    L, W, h = b['w'], b['d'], b['h']
    bedroom = b['room']['type'].startswith('bedroom')
    if h >= 1.2 and W < 0.7:
        return 'tall'
    if bedroom and L >= 1.7 and W >= 1.2 and h < 0.9:
        return 'bed'
    if L >= 1.3 and 0.45 <= h < 1.25:
        return 'dresser' if bedroom else 'console'
    if L >= 1.3:
        return 'table'
    if h < 0.75:
        return 'nightstand' if bedroom else 'plant'
    if h < 1.15:
        return 'armchair' if L > 0.6 else 'chair'
    return 'lamp'


def furnish(plan, frame, wall_mask, void_px, catalog, rng, blobs, path_px):
    F = Furnisher(plan, frame, wall_mask, void_px, catalog, rng, path_px)
    rooms = [r for r in plan['rooms'] if not r.get('outdoor')]
    has_bed = set()
    # 1) grid blobs first (what the robot actually saw), clamped to their footprint
    for b in sorted(blobs, key=lambda x: -x['area']):
        role = classify_blob(b)
        room = b['room']
        if role == 'bed':
            size = 'king' if room['type'] == 'bedroom_primary' else 'queen'
            W, L = BED_SIZES[size]
            if F.add(room, 'bed', b['c_px'], W, L, b['yaw_img'] + math.pi / 2, None, 1.2, 'grid', bed=size):
                has_bed.add(id(room))
            continue
        asset = catalog.pick(role, rng, target=(b['w'], b['d']))
        if asset is None:
            continue
        sz = sorted(asset['size_m'][:2], reverse=True)
        w = float(np.clip(sz[0], 0.6 * b['w'], 1.15 * b['w'] + 0.1))
        d = float(np.clip(sz[1], 0.6 * b['d'], 1.15 * b['d'] + 0.1))
        F.add(room, role, b['c_px'], w, d, b['yaw_img'], asset, asset['size_m'][2], 'grid')
    # 2) room templates
    for room in rooms:
        t = room['type']
        if t.startswith('bedroom'):
            size = 'king' if t == 'bedroom_primary' else ('queen' if rng.random() < 0.7 else 'full')
            W, L = BED_SIZES[size]
            if id(room) not in has_bed:
                item = None
                # beds: tolerate a little overlap with door swing / path margins; fall back to smaller beds
                for size_try, thr in ((size, 0.93), ('queen', 0.9), ('full', 0.9), ('twin', 0.88)):
                    W, L = BED_SIZES[size_try]
                    F.fit_thr = thr
                    item, slot = F.place_against_wall(room, 'bed', W, L, None, 1.2, prefer=[0.5, 0.4, 0.6, 0.3, 0.7],
                                                      bed=size_try, tries=30)
                    if item is not None:
                        break
                F.fit_thr = 0.96
                if item is not None:
                    a, b_, nrm, along = slot
                    u = (b_ - a) / np.linalg.norm(b_ - a)
                    ns = catalog.pick('nightstand', rng)
                    if ns:
                        nw, nd = sorted(ns['size_m'][:2], reverse=True)
                        for sgn in (-1, 1):
                            c = along + u * sgn * (W * frame.s / 2 + nw * frame.s / 2 + 0.06 * frame.s) + nrm * (0.15 * frame.s + nd * frame.s / 2)
                            if F.add(room, 'nightstand', c, nw, nd, math.atan2(u[1], u[0]), ns, ns['size_m'][2]):
                                F.items[-1]['back_dir'] = [float(-nrm[0]), float(nrm[1])]
            dr = catalog.pick('dresser', rng)
            if dr:
                F.place_against_wall(room, 'dresser', *sorted(dr['size_m'][:2], reverse=True), dr, dr['size_m'][2])
            if t == 'bedroom_primary':
                for role in ('armchair', 'lamp', 'plant'):
                    a_ = catalog.pick(role, rng)
                    if a_:
                        F.place_against_wall(room, role, *sorted(a_['size_m'][:2], reverse=True), a_, a_['size_m'][2])
            elif rng.random() < 0.6:
                dk, ch = catalog.pick('desk', rng), catalog.pick('chair', rng)
                if dk:
                    item, slot = F.place_against_wall(room, 'desk', *sorted(dk['size_m'][:2], reverse=True), dk, dk['size_m'][2])
                if rng.random() < 0.6:
                    p = catalog.pick('plant', rng)
                    if p:
                        F.place_against_wall(room, 'plant', *sorted(p['size_m'][:2], reverse=True), p, p['size_m'][2])
        elif t in ('hallway', 'entry'):
            for role in ('console', 'plant', 'console', 'plant'):
                a_ = catalog.pick(role, rng)
                if a_:
                    F.place_against_wall(room, role, *sorted(a_['size_m'][:2], reverse=True), a_, a_['size_m'][2])
        elif t in ('living', 'family'):
            sofa = catalog.pick('sofa_big', rng)
            if sofa:
                sw_, sd_ = sorted(sofa['size_m'][:2], reverse=True)
                item, slot = F.place_against_wall(room, 'sofa', sw_, sd_, sofa, sofa['size_m'][2], prefer=[0.5, 0.4, 0.6])
                ct = catalog.pick('coffee', rng)
                if item is not None and ct:
                    a, b_, nrm, along = slot
                    cw, cd = sorted(ct['size_m'][:2], reverse=True)
                    c = along + nrm * (0.15 * frame.s + sd_ * frame.s + 0.45 * frame.s + cd * frame.s / 2)
                    u = (b_ - a) / np.linalg.norm(b_ - a)
                    F.add(room, 'table', c, cw, cd, math.atan2(u[1], u[0]), ct, ct['size_m'][2])
            for role in ('media', 'armchair', 'armchair', 'lamp', 'plant'):
                a_ = catalog.pick(role, rng)
                if a_:
                    F.place_against_wall(room, role, *sorted(a_['size_m'][:2], reverse=True), a_, a_['size_m'][2])
        elif t == 'dining':
            tb = catalog.pick('dining_table', rng)
            free = F.room_free(room).astype(np.uint8)
            if tb is not None and free.any():
                dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
                cy_, cx_ = np.unravel_index(np.argmax(dist), dist.shape)
                tw = max(tb['size_m'][:2])
                c_t = np.array([cx_, cy_], float)
                if F.add(room, 'dining_table', c_t, tw, tw, 0.0, tb, tb['size_m'][2]):
                    ch = catalog.pick('dining_chair', rng)
                    for k in range(4):           # the footprint is committed as a square: chairs on its sides
                        q = 2 * math.pi * k / 4
                        v = np.array([math.cos(q), math.sin(q)])
                        c = c_t + v * (tw / 2 + 0.42) * frame.s
                        cw_, cd_ = sorted(ch['size_m'][:2], reverse=True)
                        if F.add(room, 'chair', c, cw_, cd_, q + math.pi / 2, ch, ch['size_m'][2]):
                            F.items[-1]['back_dir'] = [float(v[0]), float(-v[1])]
            for role in ('console', 'plant'):
                a_ = catalog.pick(role, rng)
                if a_:
                    F.place_against_wall(room, role, *sorted(a_['size_m'][:2], reverse=True), a_, a_['size_m'][2])
        elif t == 'office':
            for role in ('desk', 'tall', 'tall', 'armchair', 'plant'):
                a_ = catalog.pick(role, rng)
                if a_:
                    F.place_against_wall(room, role, *sorted(a_['size_m'][:2], reverse=True), a_, a_['size_m'][2])
        elif t == 'kitchen':
            st_ = catalog.pick('stool', rng)
            isl = [f for f in plan['fixtures'] if f['type'] == 'island' and f.get('room') == room['name']]
            if st_ and isl:
                x0, y0, x1, y1 = isl[0]['rect']
                for fx in (0.2, 0.5, 0.8):
                    c = np.array([x0 + (x1 - x0) * fx, y1 + 0.35 * frame.s])
                    if F.add(room, 'stool', c, *sorted(st_['size_m'][:2], reverse=True), 0.0, st_, st_['size_m'][2]):
                        F.items[-1]['back_dir'] = [0.0, -1.0]
        elif t == 'garage':
            for role in ('shelf_big', 'shelf_big', 'ladder', 'tall'):
                a_ = catalog.pick(role, rng)
                if a_:
                    F.place_against_wall(room, role, *sorted(a_['size_m'][:2], reverse=True), a_, a_['size_m'][2])
    return F.items
