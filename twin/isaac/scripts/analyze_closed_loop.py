"""Analyse a closed-loop run (cl_monitor.py output) against Isaac Sim's ground truth and the floor plan.

  python analyze_closed_loop.py E:\\twin\\runs\\closedloop\\cl_run2 [--twin2d-world E:\\twin\\run\\twin2d_world\\house.yaml]

Writes <run>/analysis/summary.json, results.md, trajectory.png, map_vs_plan.png. The stack's map frame starts at the
robot's start pose, which is the dock = the map origin of the twin (RTAB-Map's frame of the real map), so the
estimates are compared without fitting ("start-aligned"); the SE(2) best fit is reported as well.
"""
import argparse
import csv
import glob
import json
import math
import os
import re

import cv2
import numpy as np


def load_csv(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def fnum(r, k):
    v = r.get(k, '')
    return float(v) if v not in ('', None) else None


def se2_fit(A, B):
    """R, t minimising |R A + t - B| (2D points)."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    return R, cb - R @ ca


def ate(est, gt):
    e = np.linalg.norm(est - gt, axis=1)
    R, t = se2_fit(est, gt)
    ef = np.linalg.norm(est @ R.T + t - gt, axis=1)
    return {'rmse_m': round(float(np.sqrt((e ** 2).mean())), 3), 'max_m': round(float(e.max()), 3),
            'final_m': round(float(e[-1]), 3), 'rmse_fitted_m': round(float(np.sqrt((ef ** 2).mean())), 3),
            'fit_rotation_deg': round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2)}


def path_len(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def poly_mask(shape, polys, origin, res, flip=False):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        q = np.round((np.asarray(p) - origin) / res * 8).astype(np.int32)
        cv2.fillPoly(m, [q], 1, shift=3)
    return m.astype(bool)


def seg_intersect(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--plan', default=r'E:\twin\assets\remodel\floorplan.json')
    ap.add_argument('--twin2d-world', default=r'E:\twin\run\twin2d_world\house.yaml')
    ap.add_argument('--gpu', default='')
    a = ap.parse_args()
    mon = os.path.join(a.run, 'monitor')
    out = os.path.join(a.run, 'analysis')
    os.makedirs(out, exist_ok=True)
    rows = load_csv(os.path.join(mon, 'poses.csv'))
    events = [json.loads(ln) for ln in open(os.path.join(mon, 'events.jsonl')) if ln.strip()]
    summ_mon = json.load(open(os.path.join(mon, 'summary.json'))) if os.path.exists(os.path.join(mon, 'summary.json')) else {}
    fp = json.load(open(a.plan))
    S = {'run': os.path.basename(a.run.rstrip('\\/'))}

    # ---------------------------------------------------------------- timeline / explorer
    t_start = next((e['t'] for e in events if e['ev'] == 'explorer_start_sent'), None)
    ex = [e for e in events if e['ev'] == 'explorer']
    done = next((e for e in ex if e['state'] in ('done', 'stopped')), None)
    ret = next((e for e in ex if e['state'] == 'returning'), None)
    last = ex[-1] if ex else {}
    t_end = done['t'] if done else (rows[-1] and float(rows[-1]['t']))
    S['explorer'] = {'start_sim_s': t_start, 'exploring_ended_sim_s': ret['t'] if ret else (done['t'] if done else None),
                     'end_sim_s': t_end, 'duration_sim_s': round(t_end - t_start, 1) if t_start and t_end else None,
                     'final_state': last.get('state'), 'final_message': last.get('message'),
                     'reached': last.get('reached'), 'failed': last.get('failed'),
                     'paused_for_tracking': sum(1 for e in ex if 'tracking' in (e.get('message') or '')),
                     'monitor_finish': summ_mon.get('reason')}
    isaac = [e for e in events if e['ev'] == 'isaac']
    rtf = [e['rtf'] for e in isaac if e.get('rtf')]
    S['rtf'] = {'median': round(float(np.median(rtf)), 3) if rtf else None, 'min': min(rtf) if rtf else None,
                'max': max(rtf) if rtf else None}
    walls = [e['wall'] for e in events]
    S['wall_time_s'] = round(max(walls) - min(walls), 1) if walls else None

    # ---------------------------------------------------------------- trajectories
    def series(kx, ky):
        P, G, T = [], [], []
        for r in rows:
            g = (fnum(r, 'gt_x'), fnum(r, 'gt_y'))
            e = (fnum(r, kx), fnum(r, ky))
            t = float(r['t'])
            if None in g or None in e or (t_start and t < t_start - 1):
                continue
            if t_end and t > t_end + 1:
                continue
            P.append(e)
            G.append(g)
            T.append(t)
        return np.array(P), np.array(G), np.array(T)
    traj = {}
    for name, (kx, ky) in {'rtabmap_map': ('map_x', 'map_y'), 'ekf_odom': ('odom_x', 'odom_y'),
                           'cuvslam': ('vo_x', 'vo_y'), 'wheel_odom': ('wheel_x', 'wheel_y')}.items():
        P, G, T = series(kx, ky)
        if len(P) > 5:
            traj[name] = (P, G, T)
            S.setdefault('ate', {})[name] = ate(P, G)
    if 'rtabmap_map' in traj:
        P, G, T = traj['rtabmap_map']
        S['distance_gt_m'] = round(path_len(G), 2)
        S['distance_est_m'] = {k: round(path_len(v[0]), 2) for k, v in traj.items()}
    vs = [e for e in events if e['ev'] == 'vslam_filter']
    S['tracking'] = {'vslam_filter_recoveries': max([e.get('recoveries', 0) for e in vs] or [0]),
                     'vslam_filter_rejected': max([e.get('filtered', 0) for e in vs] or [0]),
                     'wheel_bridged': max([e.get('bridged', 0) for e in vs] or [0]),
                     'states': sorted({e.get('state') for e in vs if e.get('state')}),
                     'explorer_pauses_for_tracking': sum(1 for e in ex if 'tracking' in (e.get('message') or ''))}
    # cuVSLAM's own odometry restarts at 0 after each watchdog reset: score it per segment, each aligned to the
    # ground truth at the segment start (pose + heading), as drift per metre driven
    if 'cuvslam' in traj:
        segs = []
        idx = [i for i, r in enumerate(rows) if r['vo_x'] and r['gt_x'] and (not t_start or float(r['t']) >= t_start - 1)
               and (not t_end or float(r['t']) <= t_end + 1)]
        V = np.array([[float(rows[i]['vo_x']), float(rows[i]['vo_y']), float(rows[i]['vo_yaw'])] for i in idx])
        Gp = np.array([[float(rows[i]['gt_x']), float(rows[i]['gt_y']), float(rows[i]['gt_yaw'])] for i in idx])
        cut = np.nonzero(np.linalg.norm(np.diff(V[:, :2], axis=0), axis=1) > 0.5)[0] + 1
        for a_, b_ in zip(np.r_[0, cut], np.r_[cut, len(V)]):
            if b_ - a_ < 6:
                continue
            v, g = V[a_:b_], Gp[a_:b_]
            dyaw = g[0, 2] - v[0, 2]
            c, s_ = math.cos(dyaw), math.sin(dyaw)
            p = (v[:, :2] - v[0, :2]) @ np.array([[c, s_], [-s_, c]]) + g[0, :2]
            e = np.linalg.norm(p - g[:, :2], axis=1)
            segs.append({'n': int(b_ - a_), 'gt_m': round(path_len(g[:, :2]), 2), 'rmse_m': round(float(np.sqrt((e ** 2).mean())), 3),
                         'end_m': round(float(e[-1]), 3)})
        moving = [s_ for s_ in segs if s_['gt_m'] > 1.0]
        S['tracking']['cuvslam_resets'] = len(cut)
        S['tracking']['cuvslam_segments'] = segs[:30]
        if moving:
            S['tracking']['cuvslam_drift_pct_of_distance'] = round(
                100 * sum(s_['end_m'] for s_ in moving) / sum(s_['gt_m'] for s_ in moving), 2)
        S['ate'].pop('cuvslam', None)    # whole-run ATE is meaningless across resets
        S.get('distance_est_m', {}).pop('cuvslam', None)
    rl = os.path.join(a.run, 'logs', 'robot_launch.log')
    if os.path.exists(rl):
        txt = open(rl, errors='replace').read()
        S['tracking']['watchdog_recoveries'] = len(re.findall(r'vslam_watchdog\]: Recovery #\d+:', txt))
        S['tracking']['watchdog_filtered_motions'] = len(re.findall(r'Filtered impossible motion', txt))
        S['nav2'] = {'rpp_collision_ahead_msgs': len(re.findall(r'detected collision ahead', txt)),
                     'behavior_aborted_collision_ahead': len(re.findall(r'Collision Ahead - Exiting', txt)),
                     'explorer_goal_timeouts': len(re.findall(r'Frontier goal timed out', txt)),
                     'return_timeouts': len(re.findall(r'start point timed out', txt))}

    # ---------------------------------------------------------------- contacts
    bumps = [e.get('bumps', 0) for e in isaac]
    # pushed: ground truth moved > 0.15 m within 1 s while the wheel odometry moved < 0.05 m (not the wheels' doing)
    pushes = []
    W = [(float(r['t']), float(r['gt_x']), float(r['gt_y']), float(r['wheel_x']), float(r['wheel_y'])) for r in rows
         if r.get('gt_x') and r.get('wheel_x')]
    for i in range(2, len(W)):
        a0, b0 = W[i - 2], W[i]
        if b0[0] - a0[0] <= 1.1 and math.hypot(b0[1] - a0[1], b0[2] - a0[2]) > 0.15 \
                and math.hypot(b0[3] - a0[3], b0[4] - a0[4]) < 0.05 and (not pushes or b0[0] - pushes[-1] > 3):
            pushes.append(b0[0])
    S['contacts'] = {'isaac_bumper_contacts': (max(bumps) - min(bumps)) if bumps else 0,
                     'pushed_by_person_sim_t': pushes,
                     'pushes_undone_in_isaac': max([e.get('pushes_undone', 0) for e in isaac] or [0]),
                     'driver_bump_events': sum(1 for e in events if e['ev'] == 'bump'),
                     'cliff_events': sum(1 for e in events if e['ev'] == 'cliff'),
                     'oi_modes': [e['mode'] for e in events if e['ev'] == 'mode']}
    # bumps against people (the bumper model feels legs) vs the house
    ppl_rows = load_csv(os.path.join(mon, 'people.csv'))
    if ppl_rows and rows:
        tp = np.array([float(r['t']) for r in ppl_rows])
        gt_rows = [r for r in rows if r.get('gt_x')]
        tg = np.array([float(r['t']) for r in gt_rows])
        near = 0
        for e in events:
            if e['ev'] != 'bump':
                continue
            j, k = int(np.argmin(np.abs(tp - e['t']))), int(np.argmin(np.abs(tg - e['t'])))
            x, y = float(gt_rows[k]['gt_x']), float(gt_rows[k]['gt_y'])
            if any(math.hypot(px - x, py - y) < 0.6 for px, py in json.loads(ppl_rows[j]['people'])):
                near += 1
        S['contacts']['driver_bumps_with_a_person_within_0.6m'] = near

    # ---------------------------------------------------------------- stuck episodes, stairwell
    G = [(float(r['t']), float(r['gt_x']), float(r['gt_y'])) for r in rows if r.get('gt_x')
         and (t_start is None or float(r['t']) >= t_start) and (t_end is None or float(r['t']) <= t_end)]
    stuck, i0 = [], 0
    for i in range(len(G)):
        while G[i][0] - G[i0][0] > 30.0:
            i0 += 1
        if G[i][0] - G[i0][0] >= 29.0 and max(math.hypot(g[1] - G[i0][1], g[2] - G[i0][2]) for g in G[i0:i + 1]) < 0.05:
            if stuck and G[i0][0] <= stuck[-1][1] + 1.0:
                stuck[-1][1] = G[i][0]
            else:
                stuck.append([G[i0][0], G[i][0]])
    S['stuck_episodes'] = [{'from_sim_s': round(a_, 1), 'to_sim_s': round(b_, 1), 'duration_s': round(b_ - a_, 1)}
                           for a_, b_ in stuck]

    def seg_dist(px, py, a_, b_):
        ax, ay = a_
        bx, by = b_
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy or 1e-12
        u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        return math.hypot(px - ax - u * dx, py - ay - u * dy)
    rails = [r['polyline'] for r in fp.get('railings', []) if r.get('polyline')]
    if rails and G:
        dmin = [min(seg_dist(g[1], g[2], p[k], p[k + 1]) for p in rails for k in range(len(p) - 1)) for g in G]
        void = np.array(fp['stair_void']['polygon'], np.float32)
        inside = [cv2.pointPolygonTest(void, (float(g[1]), float(g[2])), False) > 0 for g in G]
        bump_d = []
        for e in events:
            if e['ev'] == 'bump':
                k = min(range(len(G)), key=lambda j: abs(G[j][0] - e['t']))
                bump_d.append(round(dmin[k], 2))
        dt_ = np.diff([g[0] for g in G], prepend=G[0][0])
        S['stairwell'] = {'min_dist_center_to_railing_m': round(min(dmin), 3),
                          'time_center_within_0.3m_of_railing_s': round(float(sum(d for d, m in zip(dt_, dmin) if m < 0.3)), 1),
                          'entered_void': bool(any(inside)),
                          'bumps_dist_to_railing_m': bump_d,
                          'note': 'robot radius 0.17 m: centre < 0.17 m from the railing line = touching it'}

    # ---------------------------------------------------------------- map vs plan
    mp = os.path.join(mon, 'map_rtabmap.npz')
    if os.path.exists(mp):
        z = np.load(mp)
        g = z['grid']
        res = float(z['resolution'])
        org = np.array(z['origin'][:2])
        # pad the grid (unknown) to cover the whole plan, so unmapped rooms show up
        allp = np.vstack([np.array(r['polygon']) for r in fp['rooms']])
        lo = np.minimum(org, allp.min(0) - 0.5)
        hi = np.maximum(org + np.array(g.shape[::-1]) * res, allp.max(0) + 0.5)
        o_ = np.round((org - lo) / res).astype(int)
        G2 = np.full(tuple(np.ceil((hi - lo) / res).astype(int)[::-1]), -1, np.int8)
        G2[o_[1]:o_[1] + g.shape[0], o_[0]:o_[0] + g.shape[1]] = g
        g, org = G2, org - o_ * res
        free = g == 0
        occ = g >= 50
        rooms = [r['polygon'] for r in fp['rooms'] if not r['outdoor']]
        floor = poly_mask(g.shape, rooms, org, res) & ~poly_mask(g.shape, [fp['stair_void']['polygon']], org, res)
        floor_area = floor.sum() * res * res
        # plan floor area (whole upstairs) on its own grid for the denominator
        allp = np.vstack([np.array(p) for p in rooms])
        o2 = allp.min(0) - 1
        shp = tuple((np.ceil((allp.max(0) + 1 - o2) / res)).astype(int)[::-1])
        fl2 = poly_mask(shp, rooms, o2, res) & ~poly_mask(shp, [fp['stair_void']['polygon']], o2, res)
        fl2 &= ~poly_mask(shp, [f['footprint'] for f in fp['furniture'] + fp['fixtures']], o2, res)
        plan_free_m2 = fl2.sum() * res * res
        walls_m = poly_mask(g.shape, fp['walls'], org, res) | poly_mask(g.shape, [f['footprint'] for f in fp['furniture'] + fp['fixtures']], org, res)
        dist_wall = cv2.distanceTransform((~walls_m).astype(np.uint8), cv2.DIST_L2, 5) * res
        S['map'] = {'rtabmap_free_m2': round(float(free.sum() * res * res), 2),
                    'rtabmap_free_on_plan_floor_m2': round(float((free & floor).sum() * res * res), 2),
                    'plan_free_floor_m2': round(float(plan_free_m2), 2),
                    'coverage_of_plan_floor': round(float((free & floor).sum() * res * res / plan_free_m2), 3),
                    'free_outside_plan_floor_m2': round(float((free & ~floor).sum() * res * res), 2),
                    'occupied_within_15cm_of_plan_elements': round(float((dist_wall[occ] <= 0.15).mean()), 3) if occ.any() else None,
                    'note': 'map frame compared without fitting (both start at the dock); see ate.rtabmap_map.fit_rotation_deg'}
        # 2D twin's world (the same map frame): mapped share of its true free area, like twin2d's coverage metric
        if os.path.exists(a.twin2d_world):
            import yaml  # noqa: PLC0415
            wy = yaml.safe_load(open(a.twin2d_world))
            img = cv2.imread(os.path.join(os.path.dirname(a.twin2d_world), wy['image']), cv2.IMREAD_GRAYSCALE)
            wfree = np.flipud(img > 250)
            wres, wo = wy['resolution'], np.array(wy['origin'][:2])
            ys, xs = np.nonzero(wfree)
            wx, wyy = wo[0] + (xs + 0.5) * wres, wo[1] + (ys + 0.5) * wres
            gi = ((wyy - org[1]) / res).astype(int)
            gj = ((wx - org[0]) / res).astype(int)
            ok = (gi >= 0) & (gi < g.shape[0]) & (gj >= 0) & (gj < g.shape[1])
            hit = np.zeros(len(xs), bool)
            hit[ok] = free[gi[ok], gj[ok]]
            S['map']['coverage_of_twin2d_true_free'] = round(float(hit.mean()), 3)
            S['map']['twin2d_true_free_m2'] = round(float(len(xs) * wres * wres), 2)
        # picture
        vis = np.full(g.shape + (3,), 150, np.uint8)
        vis[floor] = (200, 190, 250)
        vis[free] = (255, 255, 255)
        vis[free & ~floor] = (180, 255, 180)
        vis[occ] = (0, 0, 0)
        sc = 6
        vis = cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)

        def px(p):
            return (int((p[0] - org[0]) / res * sc), int((p[1] - org[1]) / res * sc))
        for w in fp['walls']:
            cv2.polylines(vis, [np.array([px(p) for p in w], np.int32)], True, (60, 120, 255), 1)
        for d in fp['doors']:
            cv2.line(vis, px(d['segment'][0]), px(d['segment'][1]), (255, 0, 255), 3)
        for name, col in (('rtabmap_map', (255, 0, 0)), ('wheel_odom', (0, 160, 0))):
            if name in traj:
                P, G, _ = traj[name]
                cv2.polylines(vis, [np.array([px(p) for p in P], np.int32)], False, col, 2)
                cv2.polylines(vis, [np.array([px(p) for p in G], np.int32)], False, (0, 0, 255), 1)
        vis = cv2.flip(vis, 0)
        cv2.putText(vis, 'white: mapped free on plan floor  green: mapped free off the plan  violet: plan floor not mapped',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(vis, 'blue: RTAB-Map pose  green: wheel odometry  red: ground truth  orange: plan walls  magenta: doorways',
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(out, 'map_vs_plan.png'), vis)

    # ---------------------------------------------------------------- person detections vs ground truth
    dets = load_csv(os.path.join(mon, 'detections.csv'))
    ppl = load_csv(os.path.join(mon, 'people.csv'))
    if dets and ppl and rows:
        tp_ = np.array([float(r['t']) for r in ppl])
        P_ = [json.loads(r['people']) for r in ppl]
        gt_t = np.array([float(r['t']) for r in rows if r['gt_x']])
        gt_p = np.array([[float(r['gt_x']), float(r['gt_y']), float(r['gt_yaw'])] for r in rows if r['gt_x']])
        wall_segs = []
        for w in fp['walls']:
            w = np.asarray(w)
            for i in range(len(w)):
                wall_segs.append((w[i], w[(i + 1) % len(w)]))
        wall_segs = [s_ for s_ in wall_segs if np.linalg.norm(s_[1] - s_[0]) > 0.05]
        cnt = {'frames': 0, 'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}
        by_d = {}
        for r in dets:
            t = float(r['t'])
            if t_start and t < t_start:
                continue
            k = int(np.argmin(np.abs(gt_t - t)))
            if abs(gt_t[k] - t) > 1.0:
                continue
            x, y, yaw = gt_p[k]
            cx, cy = x + 0.153 * math.cos(yaw), y + 0.153 * math.sin(yaw)
            j = int(np.argmin(np.abs(tp_ - t)))
            vis_d = []
            for qx, qy in P_[j]:
                dx, dy = qx - cx, qy - cy
                d = math.hypot(dx, dy)
                ang = (math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi
                if d < 0.3 or d > 7.0 or abs(ang) > math.radians(40):
                    continue
                if any(seg_intersect((cx, cy), (qx, qy), s0, s1) for s0, s1 in wall_segs):
                    continue
                vis_d.append(d)
            n = int(r['n'])
            cnt['frames'] += 1
            if vis_d and n:
                cnt['tp'] += 1
            elif vis_d:
                cnt['fn'] += 1
            elif n:
                cnt['fp'] += 1
            else:
                cnt['tn'] += 1
            for d in vis_d:
                b = '0-1.5' if d < 1.5 else '1.5-3' if d < 3 else '3-7'
                by_d.setdefault(b, [0, 0])
                by_d[b][0] += 1
                by_d[b][1] += int(n > 0)
        cnt['frame_recall'] = round(cnt['tp'] / max(cnt['tp'] + cnt['fn'], 1), 3)
        cnt['frame_precision'] = round(cnt['tp'] / max(cnt['tp'] + cnt['fp'], 1), 3)
        cnt['recall_by_distance_m'] = {k: round(v[1] / v[0], 3) for k, v in sorted(by_d.items())}
        cnt['note'] = ('frame level: a person is "visible" if within +-40 deg of the camera axis, 0.3-7 m, line of sight '
                       'not crossing a plan wall (furniture and other occluders ignored); at 0.13 m the camera mostly sees legs')
        S['people_detection'] = cnt

    # ---------------------------------------------------------------- CPU / GPU
    cpuj = os.path.join(a.run, 'logs', 'cpu.jsonl')
    win = None
    if os.path.exists(cpuj):
        C = [json.loads(ln) for ln in open(cpuj) if ln.strip()]
        if C:
            win = (C[0]['wall'] - 15, C[-1]['wall'] + 15)
            names = {}
            for c in C:
                for k, v in c['by_name'].items():
                    names[k] = names.get(k, 0.0) + v / len(C)
            S['wsl_cpu'] = {'stack_cores_median': round(float(np.median([c['stack_pct_one_core'] for c in C])) / 100, 2),
                            'vm_busy_pct_median': float(np.median([c['vm_busy_pct'] for c in C])), 'vm_cpus': C[0]['ncpu'],
                            'top_pct_of_one_core': {k: round(v, 1) for k, v in sorted(names.items(), key=lambda kv: -kv[1])[:10]}}
    import datetime as _dt  # noqa: PLC0415
    gpu = a.gpu or os.path.join(a.run, 'gpu.csv')
    if os.path.exists(gpu):
        u = []
        for r in csv.reader(open(gpu)):
            if len(r) < 2 or not r[1].strip(' %').isdigit():
                continue
            try:
                ts = _dt.datetime.strptime(r[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
            except ValueError:
                continue
            if win is None or win[0] <= ts <= win[1]:
                u.append(float(r[1].strip(' %')))
        if u:
            S['gpu_util_pct'] = {'median': float(np.median(u)), 'mean': round(float(np.mean(u)), 1), 'p90': float(np.percentile(u, 90)),
                                 'samples': len(u)}
    wcpu = os.path.join(a.run, 'win_cpu.csv')
    if os.path.exists(wcpu):
        L = [ln.strip() for ln in open(wcpu) if ln.strip()]
        rows3 = []
        for i in range(0, len(L) - 2, 3):
            try:
                rows3.append((_dt.datetime.fromisoformat(L[i]).timestamp(), float(L[i + 1]), float(L[i + 2])))
            except ValueError:
                continue
        rows3 = [r for r in rows3 if win is None or win[0] <= r[0] <= win[1]]
        if len(rows3) > 2:
            isaac_cores = [(b[2] - a_[2]) / (b[0] - a_[0]) for a_, b in zip(rows3, rows3[1:]) if b[0] > a_[0] and b[2] >= a_[2]]
            S['windows_cpu'] = {'total_load_pct_median': float(np.median([r[1] for r in rows3])),
                                'isaac_cores_median': round(float(np.median(isaac_cores)), 2) if isaac_cores else None,
                                'logical_cpus': os.cpu_count()}

    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(S, f, indent=1)
    # trajectory plot
    if traj:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 7))
        for w in fp['walls']:
            w = np.asarray(w + [w[0]])
            ax.plot(w[:, 0], w[:, 1], color='#999', lw=0.6)
        G = traj[next(iter(traj))][1]
        ax.plot(G[:, 0], G[:, 1], 'k-', lw=2, label='ground truth')
        for name, c in (('rtabmap_map', 'tab:blue'), ('ekf_odom', 'tab:orange'), ('cuvslam', 'tab:green'), ('wheel_odom', 'tab:purple')):
            if name in traj and name in S['ate']:
                P = traj[name][0]
                ax.plot(P[:, 0], P[:, 1], color=c, lw=1.2, label=f"{name} (ATE {S['ate'][name]['rmse_m']} m)")
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=8)
        ax.set_title(f"{S['run']}: trajectories (map frame = twin frame, start-aligned)")
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'trajectory.png'), dpi=110)
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()
