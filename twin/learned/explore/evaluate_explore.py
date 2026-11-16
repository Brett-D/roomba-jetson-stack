"""Evaluate exploration goal choice: the explorer's rule vs the learned ranker, on held-out
procedural houses and on the real house's two floors.

  python evaluate_explore.py --theta <cem>/theta_mu.npy [--ckpt <ppo>/ranker_final.pt] --out E:/twin/learned/runs/explore_eval

Static conditions (doors as in the plan, no people, full battery, same start heading for every
controller) and dynamic ones (doors opening / closing, up to 2 people walking, battery 30-100 %).
Metrics per episode: final coverage, time incl. the drive home, distance, time to 90 % coverage,
revisited distance (driven again through 0.5 m cells already driven through), failed goals,
finished (explorer's own finish) and the coverage-over-time curve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from env2d import ExploreCfg, ExploreEnv  # noqa: E402
from model import Ranker, phi  # noqa: E402

GRID_T = np.arange(0, 1801, 30)


def controller(kind, theta=None, model=None):
    if kind == "rule":
        return lambda mp, gl, ft, mk: torch.zeros(mk.shape[0], dtype=torch.long, device=mk.device)
    if kind == "cem":
        th = torch.tensor(theta, device="cuda")

        def f(mp, gl, ft, mk):
            s = (phi(ft, gl) * th).sum(-1)
            return torch.where(mk, s, torch.full_like(s, -1e9)).argmax(1)

        return f

    def g(mp, gl, ft, mk):
        with torch.no_grad():
            logits, _ = model(mp, gl, ft, mk)
        return logits.argmax(1)

    return g


def run(lays, ctl, size, dynamic, seed, repeats=1, max_dec=300):
    cfg = ExploreCfg()
    cfg.train = dynamic
    N = len(lays) * repeats
    env = ExploreEnv(lays, N, "cuda", cfg, seed=seed, size=size)
    env.record = True
    rng = np.random.default_rng(seed)
    lid = torch.arange(len(lays), device="cuda").repeat(repeats)
    yaw = torch.tensor(rng.uniform(-np.pi, np.pi, N), device="cuda", dtype=torch.float32)
    env.reset(torch.arange(N, device="cuda"), lid, yaw)
    if dynamic:
        env.batt[:] = torch.tensor(rng.uniform(0.3, 1.0, N), device="cuda", dtype=torch.float32)
    for _ in range(max_dec):
        if env.done.all():
            break
        cand = env.candidates()
        mp, gl, ft, mk = env.observe(cand)
        env.step(ctl(mp, gl, ft, mk))
    rows = []
    for e in range(N):
        tr = np.array(env.trace[e])  # (t, cov, dist)
        t90 = next((t for t, c, _ in tr if c >= 0.9), None)
        curve = np.interp(GRID_T, tr[:, 0], tr[:, 1], right=tr[-1, 1])
        rows.append({
            "layout": lays[e % len(lays)]["name"] if lays[e % len(lays)]["name"] != "procedural" else f"proc_{e % len(lays)}",
            "coverage": float(env.cov[e]),
            "time_s": float(env.t[e]),
            "dist_m": float(env.dist[e]),
            "t90_s": t90,
            "revisit_frac": float(env.revisit_dist[e] / max(float(env.dist[e]), 1e-6)),
            "failed_goals": int(env.failed[e]),
            "finished": bool(env.finished[e]),
            "home_trip_s": float(env.home_time[e]),
            "curve": curve.round(4).tolist(),
        })
    return rows


def summary(rows):
    r = rows
    t90 = [x["t90_s"] for x in r if x["t90_s"] is not None]
    return {
        "n": len(r),
        "coverage": float(np.mean([x["coverage"] for x in r])),
        "time_s": float(np.mean([x["time_s"] for x in r])),
        "dist_m": float(np.mean([x["dist_m"] for x in r])),
        "reached_90pct": len(t90) / len(r),
        "t90_s_median": float(np.median(t90)) if t90 else None,
        "t90_s_mean": float(np.mean(t90)) if t90 else None,
        "revisit_frac": float(np.mean([x["revisit_frac"] for x in r])),
        "failed_goals": float(np.mean([x["failed_goals"] for x in r])),
        "finished": float(np.mean([x["finished"] for x in r])),
        "coverage_at_5min": float(np.mean([x["curve"][10] for x in r])),
        "coverage_at_10min": float(np.mean([x["curve"][20] for x in r])),
        "auc_30min": float(np.mean([np.mean(x["curve"]) for x in r])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--heldout", default=os.path.join(HERE, "data", "layouts_heldout.npz"))
    ap.add_argument("--plans", default=os.path.join(HERE, "data", "layouts_plan.npz"))
    ap.add_argument("--n_heldout", type=int, default=200)
    ap.add_argument("--plan_repeats", type=int, default=16)
    ap.add_argument("--out", default="E:/twin/learned/runs/explore_eval")
    a = ap.parse_args()
    held = list(np.load(a.heldout, allow_pickle=True)["layouts"])[: a.n_heldout]
    plans = list(np.load(a.plans, allow_pickle=True)["layouts"])
    ctls = {"rule": controller("rule")}
    if a.theta:
        ctls["learned_linear"] = controller("cem", theta=np.load(a.theta))
    if a.ckpt:
        m = Ranker().cuda()
        m.load_state_dict(torch.load(a.ckpt, map_location="cuda"))
        m.eval()
        ctls["learned_net"] = controller("net", model=m)
    os.makedirs(a.out, exist_ok=True)
    results = {}
    t0 = time.time()
    for setname, lays, size, reps in (("heldout", held, 160, 1), ("plan_upstairs", plans[:1], 256, a.plan_repeats), ("plan_downstairs", plans[1:], 256, a.plan_repeats)):
        for dyn in (False, True):
            key = f"{setname}_{'dynamic' if dyn else 'static'}"
            results[key] = {}
            for name, ctl in ctls.items():
                rows = run(lays, ctl, size, dyn, seed=100, repeats=reps)
                results[key][name] = {"summary": summary(rows), "episodes": rows}
                s = results[key][name]["summary"]
                print(f"{key:<26} {name:<15} cov {s['coverage']:.3f} time {s['time_s']:.0f}s dist {s['dist_m']:.0f}m "
                      f"t90 {s['t90_s_median']} reach90 {s['reached_90pct']:.2f} cov@5 {s['coverage_at_5min']:.3f} cov@10 {s['coverage_at_10min']:.3f} "
                      f"revisit {s['revisit_frac']:.3f} fail {s['failed_goals']:.2f} fin {s['finished']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(a.out, "explore_eval.json"), "w") as f:
        json.dump(results, f)
    print("wrote", os.path.join(a.out, "explore_eval.json"))


if __name__ == "__main__":
    main()
