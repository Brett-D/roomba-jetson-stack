"""Evaluate unstuck controllers on held-out stuck tiles: one episode per tile, deterministic.

  python evaluate_unstuck.py --controller classical --tiles data/tiles_eval.json
  python evaluate_unstuck.py --controller policy --ckpt <run>/model_599.pt --tiles data/tiles_house.json

Writes <out>/<name>.json (per episode) and prints success rate, time to free, bumps per episode
per scenario kind. Optional --record N saves N observation rows (.npy) for the ONNX test vectors.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402


def summarise(rows):
    out = {}
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["kind"]].append(r)
        groups["ALL"].append(r)
    for k, rs in sorted(groups.items()):
        succ = [r for r in rs if r["outcome"] == "success"]
        out[k] = {
            "n": len(rs),
            "success": len(succ) / len(rs),
            "cliff_or_fall": sum(r["outcome"] in ("cliff", "fall") for r in rs) / len(rs),
            "time_to_free_s": float(np.median([r["time_s"] for r in succ])) if succ else None,
            "bumps_per_ep": float(np.mean([r["bumps"] for r in rs])),
            "top_contacts_per_ep": float(np.mean([r["top"] for r in rs])),
            "goal_progress_m": float(np.mean([r["goal_progress_m"] for r in rs])),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["classical", "policy", "idle"], default="classical")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tiles", default=os.path.join(HERE, "data", "tiles_eval.json"))
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--out", default="E:/twin/learned/runs/unstuck_eval")
    ap.add_argument("--name", default="")
    ap.add_argument("--record", type=int, default=0)
    a = ap.parse_args()
    import unstuck_env as U
    import world as W

    T = len(W.load_world(a.tiles).tiles)
    cfg = U.UnstuckEnvCfg()
    cfg.scene.num_envs = min(a.num_envs, T)
    cfg.tiles_path = a.tiles
    cfg.eval_mode = True
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 0.02, 5, 5
    name = a.name or f"{a.controller}_{os.path.splitext(os.path.basename(a.tiles))[0]}"
    with launch_simulation(cfg, {"physics": "ovphysx"}):
        env = U.UnstuckEnv(cfg)
        obs, _ = env.reset()
        obs = obs["policy"]
        if a.controller == "classical":
            from baseline import ClassicalRecovery

            ctl = ClassicalRecovery(env)
            ctl.reset(torch.arange(env.num_envs, device=env.device))
            act_fn = ctl.act
        elif a.controller == "policy":
            from export_onnx import load_actor

            actor = load_actor(a.ckpt).to(env.device)
            act_fn = lambda o: actor(o)  # noqa: E731
        else:
            act_fn = lambda o: torch.zeros(o.shape[0], 2, device=o.device)  # noqa: E731
        rows, seen, rec = [], set(), []
        t0 = time.time()
        with torch.no_grad():
            while len(seen) < T and time.time() - t0 < 3600:
                act = act_fn(obs)
                if a.record and len(rec) < a.record and np.random.rand() < 0.05:
                    rec.append(obs[np.random.randint(obs.shape[0])].cpu().numpy())
                o, r, term, trunc, extras = env.step(act)
                obs = o["policy"]
                done = (term | trunc).nonzero().squeeze(-1)
                if len(done) and a.controller == "classical":
                    ctl.reset(done)
                for row in env.eval_log:
                    if row["tile"] not in seen:
                        seen.add(row["tile"])
                        rows.append(row)
                env.eval_log.clear()
        env.close()
    os.makedirs(a.out, exist_ok=True)
    summ = summarise(rows)
    with open(os.path.join(a.out, name + ".json"), "w") as f:
        json.dump({"controller": a.controller, "tiles": a.tiles, "ckpt": a.ckpt, "summary": summ, "episodes": rows}, f, indent=1)
    if rec:
        np.save(os.path.join(a.out, name + "_obs.npy"), np.stack(rec))
    print(f"\n{name}: {len(rows)} episodes in {time.time() - t0:.0f}s")
    print(f"{'kind':<12} {'n':>4} {'success':>8} {'cliff':>6} {'t_free':>7} {'bumps':>6} {'top':>5} {'goal+':>6}")
    for k, s in summ.items():
        tf = f"{s['time_to_free_s']:.1f}" if s["time_to_free_s"] is not None else "-"
        print(f"{k:<12} {s['n']:>4} {s['success']:>8.2f} {s['cliff_or_fall']:>6.2f} {tf:>7} {s['bumps_per_ep']:>6.2f} {s['top_contacts_per_ep']:>5.2f} {s['goal_progress_m']:>6.2f}")
