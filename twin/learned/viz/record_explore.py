"""Record an exploration run of the 2D env for the video: the explorer's rule and the learned
ranker side by side (two envs, same house, same start), every decision and every sensing pose.

  python record_explore.py --layout plan_upstairs --yaw 0.5 --out E:/twin/learned/runs/video/explore_up.pkl
  python record_explore.py --layout heldout:17 --out E:/twin/learned/runs/video/explore_h17.pkl

Static conditions (doors as in the plan, no people, full battery). Sensing is applied pose by pose
(the training env batches 12 poses per update), so the map can be revealed along the drive.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPLORE = os.path.join(HERE, "..", "explore")
sys.path.insert(0, EXPLORE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gpu  # noqa: E402
from env2d import RES, ExploreCfg, ExploreEnv  # noqa: E402
from model import Ranker  # noqa: E402


class RecEnv(ExploreEnv):
    rec_on = False

    def _sense_ids(self, ids, poses, valid, chunk=12):
        if not self.rec_on:
            return super()._sense_ids(ids, poses, valid, chunk)
        M = poses.shape[1]
        il = ids.tolist()
        vv = valid.cpu().numpy()
        pp = poses.cpu().numpy()
        for m in range(M):
            if not vv[:, m].any():
                continue
            before = self.belief[ids].clone()
            super()._sense_ids(ids, poses[:, m:m + 1], valid[:, m:m + 1], chunk=1)
            ch = (self.belief[ids] != before).view(len(il), -1)
            for j, e in enumerate(il):
                if vv[j, m]:
                    idx = torch.nonzero(ch[j]).squeeze(-1)
                    self.poselog[e].append((pp[j, m].copy(), idx.cpu().numpy().astype(np.int32),
                                            self.belief[e].view(-1)[idx].cpu().numpy().astype(np.int8)))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="plan_upstairs")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--ckpt", default="E:/twin/learned/runs/explore/ppo_from_cem/ranker_final.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.layout.startswith("heldout:"):
        lay = list(np.load(os.path.join(EXPLORE, "data", "layouts_heldout.npz"), allow_pickle=True)["layouts"])[int(a.layout.split(":")[1])]
        size = 160
    else:
        lays = list(np.load(os.path.join(EXPLORE, "data", "layouts_plan.npz"), allow_pickle=True)["layouts"])
        lay = [l for l in lays if l["name"] == a.layout][0]
        size = 256
    cfg = ExploreCfg()
    cfg.train = False
    env = RecEnv([lay], 2, "cuda", cfg, seed=100, size=size)
    dev = env.dev
    ar = torch.arange(2, device=dev)
    env.reset(ar, torch.zeros(2, dtype=torch.long, device=dev), torch.full((2,), a.yaw, device=dev))
    model = Ranker().to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    env.poselog = [[], []]
    tl = [[], []]  # timeline per env: (t, x, y, yaw, n_diffs_applied_until, decision)
    diffs = [[], []]
    decisions = [[], []]
    belief0 = env.belief.cpu().numpy().copy()
    for e in range(2):
        tl[e].append((0.0, float(env.pos[e, 0]), float(env.pos[e, 1]), float(env.yaw[e]), 0, -1))
    c = env.cfg
    for k in range(400):
        if env.done.all():
            break
        cand = env.candidates()
        mp, gl, ft, mk = env.observe(cand)
        with torch.no_grad():
            logits, _ = model(mp, gl, ft, mk)
        choice = torch.stack([torch.zeros((), dtype=torch.long, device=dev), logits[1].argmax()])
        has = cand["mask"].any(1) & ~env.done
        goal = cand["goal"][ar, choice]
        gr, gc = env._rc(goal)
        R, C, plen = gpu.trace_paths(cand["D"], gr, gc, has, maxlen=4 * env.H)
        known, occ, passable, free, infl, unk_near = env.costmap()
        fr, _ = env.frontier_groups(known, free)
        t0 = env.t.cpu().numpy().copy()
        pos0 = env.pos.cpu().numpy().copy()
        yaw0 = env.yaw.cpu().numpy().copy()
        done0 = env.done.cpu().numpy().copy()
        n_black0 = env.n_black.cpu().numpy().copy()
        for e in range(2):
            env.poselog[e] = []
            if done0[e]:
                continue
            m = cand["mask"][e].cpu().numpy()
            lg = logits[e].cpu().numpy()
            lg = np.where(m, lg, -np.inf)
            pr = np.exp(lg - lg[m].max()) if m.any() else lg
            pr = pr / pr[m].sum() if m.any() else pr
            decisions[e].append({
                "k": k, "t": float(t0[e]), "pos": pos0[e].tolist(), "yaw": float(yaw0[e]),
                "mask": m, "goal": cand["goal"][e].cpu().numpy(), "front": cand["front"][e].cpu().numpy(),
                "size": cand["size"][e].cpu().numpy(), "score": cand["score"][e].cpu().numpy(),
                "path": cand["path"][e].cpu().numpy(), "prob": pr, "choice": int(choice[e]) if bool(has[e]) else -1,
                "frontier": np.nonzero(fr[e].view(-1).cpu().numpy())[0].astype(np.int32),
                "plan": np.stack([R[e, : plen[e]].cpu().numpy(), C[e, : plen[e]].cpu().numpy()], 1) if bool(has[e]) else np.zeros((0, 2), int),
            })
        env.rec_on = True
        env.step(choice)
        env.rec_on = False
        t1 = env.t.cpu().numpy()
        done1 = env.done.cpu().numpy()
        for e in range(2):
            if done0[e]:
                continue
            d = decisions[e][-1]
            d["t_end"] = float(t1[e])
            d["failed"] = bool(env.n_black[e].item() > n_black0[e]) if d["choice"] >= 0 else False
            d["cov"] = float(env.cov[e])
            d["dist"] = float(env.dist[e])
            # sensing poses with times: turning at turn_speed, driving at speed, then the goal overhead
            px, py, pyaw = pos0[e][0], pos0[e][1], yaw0[e]
            durs, rows = [], []
            for pose, idx, val in env.poselog[e]:
                dd = max(math.hypot(pose[0] - px, pose[1] - py) / c.speed, abs(wrap(pose[2] - pyaw)) / c.turn_speed)
                durs.append(dd)
                rows.append((pose, idx, val))
                px, py, pyaw = pose
            total = float(t1[e] - t0[e])
            if d["choice"] < 0:  # finish logic: confirm wait (+ drive home when done)
                if done1[e]:
                    trip = float(env.home_time[e])
                    wait = max(total - trip, 0.0)
                    tl[e].append((float(t0[e]) + wait, float(pos0[e][0]), float(pos0[e][1]), float(yaw0[e]), len(diffs[e]), k))
                    hp = home_path(env, e, pos0[e])
                    seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(hp, axis=0).T))]
                    for (hx, hy), s in zip(hp, seg):
                        frac = s / max(seg[-1], 1e-6)
                        tl[e].append((float(t0[e]) + wait + frac * max(trip - c.goal_overhead_s, 0.0), float(hx), float(hy), float(yaw0[e]), len(diffs[e]), k))
                    tl[e].append((float(t1[e]), float(hp[-1][0]), float(hp[-1][1]), float(yaw0[e]), len(diffs[e]), k))
                    d["home_path"] = hp
                else:
                    tl[e].append((float(t1[e]), float(pos0[e][0]), float(pos0[e][1]), float(yaw0[e]), len(diffs[e]), k))
                continue
            trip = float(env.home_time[e]) if done1[e] else 0.0  # timeout / battery: drove home in the same step
            total -= trip
            S_ = sum(durs)
            scale = 1.0 if S_ <= total else total / max(S_, 1e-6)
            tt = float(t0[e])
            for dd, (pose, idx, val) in zip(durs, rows):
                tt += dd * scale
                diffs[e].append((idx, val))
                tl[e].append((tt, float(pose[0]), float(pose[1]), float(pose[2]), len(diffs[e]), k))
            ex, ey, eyaw = tl[e][-1][1], tl[e][-1][2], tl[e][-1][3]
            tl[e].append((float(t0[e]) + total, ex, ey, eyaw, len(diffs[e]), k))
            if done1[e]:
                hp = home_path(env, e, np.array([ex, ey]))
                seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(hp, axis=0).T))]
                for (hx, hy), s in zip(hp, seg):
                    frac = s / max(seg[-1], 1e-6)
                    tl[e].append((float(t0[e]) + total + frac * max(trip - c.goal_overhead_s, 0.0), float(hx), float(hy), eyaw, len(diffs[e]), k))
                tl[e].append((float(t1[e]), float(hp[-1][0]), float(hp[-1][1]), eyaw, len(diffs[e]), k))
                d["home_path"] = hp
    reach0 = env.reach0[0].cpu().numpy()
    out = {
        "layout": lay["name"] if lay["name"] != "procedural" else a.layout, "yaw": a.yaw, "H": env.H, "res": RES,
        "occ": env.L_occ[0].cpu().numpy(), "door": env.L_door[0].cpu().numpy(), "closed0": env.L_closed0[0].cpu().numpy(),
        "home": env.home[0].cpu().numpy(), "reach0": reach0, "belief0": belief0,
        "timeline": [np.array(tl[e], dtype=np.float64) for e in range(2)],
        "diffs": diffs, "decisions": decisions,
        "final": [{"cov": float(env.cov[e]), "t": float(env.t[e]), "dist": float(env.dist[e]), "revisit": float(env.revisit_dist[e]),
                   "failed": int(env.failed[e]), "finished": bool(env.finished[e])} for e in range(2)],
    }
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    for e, name in enumerate(("rule", "learned")):
        r = out["final"][e]
        print(f"{name:8s} cov {r['cov']:.3f} time {r['t']:.0f}s dist {r['dist']:.0f}m revisit {r['revisit']:.0f}m failed {r['failed']} decisions {len(decisions[e])}")


def home_path(env, e, pos):
    """Path of the drive home (the env only adds its time and distance)."""
    known, occ, passable, free, infl, unk_near = env.costmap()
    passable = passable.clone()
    hr, hc = env._rc(env.home)
    passable[torch.arange(env.N, device=env.dev), hr, hc] = True
    saved = env.pos.clone()
    env.pos[e] = torch.tensor(pos, device=env.dev, dtype=env.pos.dtype)
    seed, _ = env._seed(passable)
    env.pos.copy_(saved)
    D = env.distance_field(seed, passable, infl)
    act = torch.zeros(env.N, dtype=torch.bool, device=env.dev)
    act[e] = True
    if not torch.isfinite(D[e, hr[e], hc[e]]):
        return np.array([pos, env.home[e].cpu().numpy()])
    R, C, plen = gpu.trace_paths(D, hr, hc, act, maxlen=4 * env.H)
    n = int(plen[e])
    rc = np.stack([R[e, :n].cpu().numpy(), C[e, :n].cpu().numpy()], 1)
    return np.stack([rc[:, 1] * RES + RES / 2, rc[:, 0] * RES + RES / 2], 1)


if __name__ == "__main__":
    main()
