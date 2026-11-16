"""Cross-entropy method (CEM) for the frontier ranker's linear scoring head.

PPO on the full CNN ranker learns slowly here (one decision's effect on the final coverage curve is
small next to the layout noise). CEM optimises the episode objective directly with common random
numbers: every population member explores the same houses (same start, same battery), so members
differ only by their choices. The scorer is linear in the candidate features (+ two battery
interactions); it starts at the explorer's rule (score = -rule_score), so generation 0 contains it.

Fitness per episode (higher is better), T = 1800 s:
  mean coverage over [0, T] (coverage after the end counts as held)  - 0.2 * end time / T
  (the end time includes the drive home), i.e. map fast, finish early, come home.

  python train_cem.py --pop 32 --houses 8 --gens 40
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from env2d import N_FEAT, ExploreCfg, ExploreEnv  # noqa: E402

N_PHI = N_FEAT + 2
SCORE = 11
HOME = 8
NAMES = ["dx", "dy", "dist", "path", "size", "gain", "cos_turn", "sin_turn", "home_dist", "blacklisted_near", "visited_near",
         "rule_score", "rank", "look_x", "look_y", "detour", "batt_x_home", "lowbatt_x_home"]


def phi(feats: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
    b = glob[:, 0:1, None].expand(-1, feats.shape[1], 1)
    hd = feats[..., HOME:HOME + 1]
    return torch.cat([feats, b * hd, (1 - b) * hd], -1)


def rule_theta():
    th = np.zeros(N_PHI, np.float32)
    th[SCORE] = -1.0
    return th


def run_episodes(env: ExploreEnv, thetas: torch.Tensor, layout_ids: torch.Tensor, batt: torch.Tensor, yaw: torch.Tensor, max_dec=200):
    """thetas [N, N_PHI] per env; returns fitness [N] and stats."""
    N = env.N
    ids = torch.arange(N, device=env.dev)
    env.reset(ids, layout_ids, yaw)
    env.batt[:] = batt
    T = env.cfg.max_time_s
    auc = torch.zeros(N, device=env.dev)
    last_t = torch.zeros(N, device=env.dev)
    last_c = env.cov.clone()
    for _ in range(max_dec):
        if env.done.all():
            break
        cand = env.candidates()
        _, glob, feats, mask = env.observe(cand)
        s = (phi(feats, glob) * thetas[:, None, :]).sum(-1)
        s = torch.where(mask, s, torch.full_like(s, -1e9))
        act = s.argmax(1)
        active = ~env.done
        env.step(act)
        # trapezoid-free: coverage is piecewise constant between decisions (new area appears along
        # the way, so this slightly under-counts both controllers equally)
        dt = (env.t.clamp(max=T) - last_t.clamp(max=T))
        auc += torch.where(active, last_c * dt, torch.zeros_like(dt))
        last_t = env.t.clone()
        last_c = env.cov.clone()
    auc += last_c * (T - last_t.clamp(max=T))
    fit = auc / T - 0.2 * env.t.clamp(max=2 * T) / T
    return fit, env.cov.clone(), env.t.clone(), env.dist.clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--houses", type=int, default=8)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--elite", type=float, default=0.25)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--layouts", default=os.path.join(HERE, "data", "layouts_train.npz"))
    ap.add_argument("--out", default="E:/twin/learned/runs/explore_cem")
    ap.add_argument("--run", default=time.strftime("%Y%m%d_%H%M%S"))
    a = ap.parse_args()
    lays = list(np.load(a.layouts, allow_pickle=True)["layouts"])
    cfg = ExploreCfg()
    cfg.train = False  # static doors, no people: common random numbers across the population
    N = a.pop * a.houses
    env = ExploreEnv(lays, N, "cuda", cfg, seed=3, size=160)
    rng = np.random.default_rng(0)
    mu = rule_theta()
    sigma = np.full(N_PHI, a.sigma, np.float32)
    sigma[SCORE] = 0.3
    out = os.path.join(a.out, a.run)
    os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, "cem.csv"), "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["gen", "wall_s", "fit_rule", "fit_mean", "fit_best", "fit_mu", "cov_rule", "cov_mu", "time_rule", "time_mu"])
    t0 = time.time()
    best = (-1e9, mu.copy())
    for gen in range(a.gens):
        pop = mu + sigma * rng.standard_normal((a.pop, N_PHI)).astype(np.float32)
        pop[0] = rule_theta()  # reference: the explorer's rule on the same houses
        pop[1] = mu  # the current mean
        lay = torch.tensor(rng.integers(0, len(lays), a.houses), device="cuda")
        batt = torch.tensor(rng.uniform(0.35, 1.0, a.houses), device="cuda", dtype=torch.float32)
        yaw = torch.tensor(rng.uniform(-np.pi, np.pi, a.houses), device="cuda", dtype=torch.float32)
        th = torch.tensor(np.repeat(pop, a.houses, 0), device="cuda")
        fit, cov, tt, dist = run_episodes(env, th, lay.repeat(a.pop), batt.repeat(a.pop), yaw.repeat(a.pop))
        fit = fit.view(a.pop, a.houses).mean(1).cpu().numpy()
        cov = cov.view(a.pop, a.houses).mean(1).cpu().numpy()
        tt = tt.view(a.pop, a.houses).mean(1).cpu().numpy()
        # rank by improvement over the rule on the same houses
        order = np.argsort(-fit)
        ne = max(2, int(a.elite * a.pop))
        el = pop[order[:ne]]
        mu = 0.7 * el.mean(0) + 0.3 * mu
        sigma = np.maximum(0.7 * el.std(0) + 0.3 * sigma, 0.02).astype(np.float32)
        if fit[1] - fit[0] > best[0]:
            best = (fit[1] - fit[0], pop[1].copy())
        wr.writerow([gen, round(time.time() - t0, 1), fit[0], fit.mean(), fit.max(), fit[1], cov[0], cov[1], tt[0], tt[1]])
        f.flush()
        print(f"gen {gen} {time.time() - t0:.0f}s rule {fit[0]:.4f} mu {fit[1]:.4f} best {fit.max():.4f} cov rule {cov[0]:.3f} mu {cov[1]:.3f} time rule {tt[0]:.0f} mu {tt[1]:.0f}", flush=True)
        np.save(os.path.join(out, "theta_mu.npy"), mu)
    json.dump({"theta": dict(zip(NAMES, map(float, mu))), "sigma": dict(zip(NAMES, map(float, sigma)))}, open(os.path.join(out, "theta.json"), "w"), indent=1)
    print("done", out)


if __name__ == "__main__":
    main()
