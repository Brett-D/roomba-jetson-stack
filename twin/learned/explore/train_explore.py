"""PPO for the frontier ranker on the batched 2D exploration env.

  python train_explore.py --envs 256 --updates 300
Layouts: data/layouts_train.npz (procedural, generated on first use). Logs: E:/twin/learned/runs/explore/<run>.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.distributions import Categorical  # noqa: E402

import layouts as Lg  # noqa: E402
from env2d import ExploreCfg, ExploreEnv  # noqa: E402
from model import Ranker  # noqa: E402


def load_layouts(path, n, seed, size=160):
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return list(d["layouts"])
    t0 = time.time()
    lays = Lg.generate(n, seed, size)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, layouts=np.array(lays, dtype=object))
    print(f"generated {n} layouts in {time.time() - t0:.0f}s -> {path}")
    return lays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--layouts", default=os.path.join(HERE, "data", "layouts_train.npz"))
    ap.add_argument("--n_layouts", type=int, default=2000)
    ap.add_argument("--extra", default="", help="also train on these layouts (npz), e.g. the real floor plans")
    ap.add_argument("--out", default="E:/twin/learned/runs/explore")
    ap.add_argument("--run", default=time.strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--init", default="")
    ap.add_argument("--theta", default="", help="linear prior (theta_mu.npy from train_cem.py); default: the rule")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    lays = load_layouts(a.layouts, a.n_layouts, seed=11)
    env = ExploreEnv(lays, a.envs, dev, ExploreCfg(), seed=1, size=160)
    model = Ranker(theta=np.load(a.theta) if a.theta else None).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    log_dir = os.path.join(a.out, a.run)
    os.makedirs(log_dir, exist_ok=True)
    fcsv = open(os.path.join(log_dir, "metrics.csv"), "w", newline="")
    wr = csv.writer(fcsv)
    wr.writerow(["update", "decisions", "wall_s", "mean_reward", "ep_coverage", "ep_time_s", "ep_dist_m", "ep_failed", "ep_finished", "episodes", "pi_loss", "v_loss", "entropy", "agree_rule", "prior_scale"])
    N, T = a.envs, a.steps
    env.reset(torch.arange(N, device=dev))
    gamma, lam, clip = 0.99, 0.95, 0.2
    t_start = time.time()
    decisions = 0
    ep_stats = []
    for upd in range(a.updates):
        buf = {k: [] for k in ("map", "glob", "feats", "mask", "act", "logp", "val", "rew", "done", "valid")}
        model.eval()
        for t in range(T):
            cand = env.candidates()
            mp, gl, ft, mk = env.observe(cand)
            with torch.no_grad():
                logits, v = model(mp, gl, ft, mk)
            valid = mk.any(1) & ~env.done
            dist = Categorical(logits=torch.where(valid[:, None], logits, torch.zeros_like(logits)))
            act = dist.sample()
            act = torch.where(valid, act, torch.zeros_like(act))
            logp = dist.log_prob(act)
            rew, done, fail = env.step(act)
            for k, x in (("map", mp.half()), ("glob", gl), ("feats", ft), ("mask", mk), ("act", act), ("logp", logp), ("val", v), ("rew", rew), ("done", done), ("valid", valid)):
                buf[k].append(x)
            decisions += int(valid.sum())
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for i in ids.tolist():
                    ep_stats.append((float(env.cov[i]), float(env.t[i]), float(env.dist[i]), int(env.failed[i]), bool(env.finished[i])))
                env.reset(ids)
        with torch.no_grad():
            cand = env.candidates()
            mp, gl, ft, mk = env.observe(cand)
            _, last_v = model(mp, gl, ft, mk)
        B = {k: torch.stack(v) for k, v in buf.items()}
        # GAE (episode ends at done; envs without a decision this step carry the reward)
        adv = torch.zeros(T, N, device=dev)
        last = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else B["val"][t + 1]
            nonterm = 1.0 - B["done"][t].float()
            delta = B["rew"][t] + gamma * nv * nonterm - B["val"][t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + B["val"]
        sel = B["valid"].reshape(-1)
        flat = {k: v.reshape(T * N, *v.shape[2:])[sel] for k, v in B.items()}
        advf = adv.reshape(-1)[sel]
        retf = ret.reshape(-1)[sel]
        advf = (advf - advf.mean()) / (advf.std() + 1e-8)
        model.train()
        M = len(advf)
        mb = max(256, M // 4)
        pl = vl = ent = 0.0
        nmb = 0
        for epoch in range(4):
            perm = torch.randperm(M, device=dev)
            for s in range(0, M, mb):
                j = perm[s:s + mb]
                logits, v = model(flat["map"][j].float(), flat["glob"][j], flat["feats"][j], flat["mask"][j])
                d = Categorical(logits=logits)
                lp = d.log_prob(flat["act"][j])
                ratio = torch.exp(lp - flat["logp"][j])
                s1 = ratio * advf[j]
                s2 = ratio.clamp(1 - clip, 1 + clip) * advf[j]
                p_loss = -torch.min(s1, s2).mean()
                v_loss = 0.5 * ((v - retf[j]) ** 2).mean()
                e = d.entropy().mean()
                loss = p_loss + 0.5 * v_loss - a.ent * e
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                pl += float(p_loss.detach())
                vl += float(v_loss.detach())
                ent += float(e.detach())
                nmb += 1
        with torch.no_grad():
            agree = (flat["act"] == 0).float().mean().item()
        recent = ep_stats[-200:]
        row = [upd, decisions, round(time.time() - t_start, 1), float(B["rew"].mean())]
        if recent:
            r = np.array(recent, dtype=float)
            row += [r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean(), r[:, 4].mean(), len(ep_stats)]
        else:
            row += [0, 0, 0, 0, 0, 0]
        row += [pl / nmb, vl / nmb, ent / nmb, agree, float(model.prior_scale)]
        wr.writerow(row)
        fcsv.flush()
        if upd % 5 == 0 or upd == a.updates - 1:
            print(f"upd {upd} dec {decisions} {row[2]:.0f}s rew {row[3]:.3f} cov {row[4]:.3f} time {row[5]:.0f}s dist {row[6]:.0f}m fail {row[7]:.2f} fin {row[8]:.2f} eps {row[9]} ent {row[12]:.2f} rule_agree {agree:.2f} prior {float(model.prior_scale):.1f}", flush=True)
        if upd % 25 == 24 or upd == a.updates - 1:
            torch.save(model.state_dict(), os.path.join(log_dir, f"ranker_{upd + 1}.pt"))
    torch.save(model.state_dict(), os.path.join(log_dir, "ranker_final.pt"))
    fcsv.close()
    print("done", log_dir)


if __name__ == "__main__":
    main()
