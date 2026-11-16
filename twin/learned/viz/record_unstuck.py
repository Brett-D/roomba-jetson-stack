"""Record per-step rollout states of the unstuck env (Isaac Lab, PhysX) for the comparison video.

  python record_unstuck.py --controller classical --out E:/twin/learned/runs/video/classical.npz
  python record_unstuck.py --controller policy --ckpt E:/twin/learned/runs/unstuck/main/model_300.pt \
      --out E:/twin/learned/runs/video/policy.npz

Runs one deterministic episode per tile of a pool (all house tiles + the first --per_kind held-out
procedural tiles of every kind, written once to <out dir>/pool_tiles.json) and stores, per 0.1 s
step and env: pose in the tile frame, bumper / camera-top contacts, light-bumper proximities, the
depth fan per column (nearest obstacle return, farthest floor return, blind), collision-monitor
stop, driver reflex, commands, the costmap's lethal cells (packed bits) and the outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UNSTUCK = os.path.join(HERE, "..", "unstuck")
sys.path.insert(0, UNSTUCK)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402


def make_pool(path, per_kind):
    import scenarios as S

    house = S.load(os.path.join(UNSTUCK, "data", "tiles_house.json"))
    ev = S.load(os.path.join(UNSTUCK, "data", "tiles_eval.json"))
    seen, pick = {}, []
    for t in ev:
        if seen.get(t.kind, 0) < per_kind:
            seen[t.kind] = seen.get(t.kind, 0) + 1
            pick.append(t)
    S.save(house + pick, path)
    return len(house) + len(pick)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["classical", "policy"], required=True)
    ap.add_argument("--ckpt", default="E:/twin/learned/runs/unstuck/main/model_300.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_kind", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pool = os.path.join(os.path.dirname(a.out), "pool_tiles.json")
    if not os.path.exists(pool):
        make_pool(pool, a.per_kind)

    import scenarios as S
    import unstuck_env as U
    import world as W

    T = len(W.load_world(pool).tiles)

    class RecEnv(U.UnstuckEnv):
        def snap(self):
            p, yaw = self._pose()
            loc = self._local(p)
            F = self.contact.data.net_forces_w.torch[:, self.base_ci]
            c, s = torch.cos(yaw), torch.sin(yaw)
            fx = c * F[:, 0] + s * F[:, 1]
            fy = -s * F[:, 0] + c * F[:, 1]
            base_hit = torch.hypot(fx, fy) > 1.5
            bearing = torch.atan2(-fy, -fx)
            dist, valid, obst, blind = self.fan_dbg
            el = torch.cos(torch.tensor(U.FAN_ELEV, device=self.device)).repeat(U.FAN_COLS)
            hd = (dist.clamp(max=10.0) * el).view(self.num_envs, U.FAN_COLS, len(U.FAN_ELEV))
            ob = obst.view_as(hd)
            fl = (valid & ~obst).view_as(hd)
            obst_hd = torch.where(ob, hd, torch.full_like(hd, 99.0)).min(-1).values
            free_hd = torch.where(fl, hd, torch.zeros_like(hd)).max(-1).values
            lethal = ((self.t.unsqueeze(-1) - self.mark_t) < U.DECAY_S) | self.keepout
            done = self.reset_terminated | self.reset_time_outs
            row = {
                "tile": self.tile.clone(),
                "xy": loc.clone(),
                "z": p[:, 2].clone(),
                "yaw": yaw.clone(),
                "t": self.t.clone(),
                "base_hit": base_hit,
                "hit_bearing": bearing,
                "bumped": self.bumped.clone(),
                "top": self.top_hit.clone(),
                "ir": self.ir.clone(),
                "obst_hd": obst_hd,
                "free_hd": free_hd,
                "blind": blind.clone(),
                "cm_stop": self.cm_stop.clone(),
                "reflex": self.reflex_steps > 0,
                "act": self.act.clone(),
                "cmd_out": self.cmd_out.clone(),
                "clear": self.clear_now.clone(),
                "done": done.clone(),
                "outcome": self.outcome.clone(),
                "bumps": self.ep_bumps.clone(),
                "tops": self.ep_top.clone(),
            }
            self.rec.append({k: v.cpu().numpy() for k, v in row.items()})
            self.rec_lethal.append(np.packbits(lethal.cpu().numpy(), axis=-1))

        def _get_rewards(self):
            r = super()._get_rewards()
            self.snap()
            return r

    cfg = U.UnstuckEnvCfg()
    cfg.scene.num_envs = T
    cfg.tiles_path = pool
    cfg.eval_mode = True
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 0.02, 5, 5
    with launch_simulation(cfg, {"physics": "ovphysx"}):
        env = RecEnv(cfg)
        env.rec, env.rec_lethal = [], []
        obs, _ = env.reset()
        obs = obs["policy"]
        if a.controller == "classical":
            from baseline import ClassicalRecovery

            ctl = ClassicalRecovery(env)
            ctl.reset(torch.arange(env.num_envs, device=env.device))
            act_fn = ctl.act
        else:
            from export_onnx import load_actor

            actor = load_actor(a.ckpt).to(env.device)
            act_fn = lambda o: actor(o)  # noqa: E731
        env.rec.clear()
        env.rec_lethal.clear()
        env.reset_terminated[:] = False
        env.reset_time_outs[:] = False
        env.snap()
        modes = [np.zeros(T, np.int64)]
        finished = np.zeros(T, bool)
        with torch.no_grad():
            for step in range(170):
                act = act_fn(obs)
                modes.append(ctl.mode.cpu().numpy().copy() if a.controller == "classical" else np.zeros(T, np.int64))
                o, r, term, trunc, _ = env.step(act)
                obs = o["policy"]
                done_ids = (term | trunc).nonzero().squeeze(-1)
                if len(done_ids) and a.controller == "classical":
                    ctl.reset(done_ids)
                finished |= (term | trunc).cpu().numpy()
                if finished.all():
                    break
        env.close()
    rec = env.rec
    out = {k: np.stack([r[k] for r in rec]) for k in rec[0]}
    out["lethal"] = np.stack(env.rec_lethal)
    out["mode"] = np.stack(modes[: len(rec)])
    np.savez_compressed(a.out, **out)
    tiles = S.load(pool)
    first = out["done"].argmax(0)
    res = []
    for e in range(T):
        k = int(first[e]) if out["done"][:, e].any() else len(rec) - 1
        tid = int(out["tile"][0, e])
        res.append({"env": e, "tile": tid, "kind": tiles[tid].kind, "end_step": k,
                    "outcome": ["", "success", "cliff", "fall", "timeout"][int(out["outcome"][k, e])],
                    "time_s": float(out["t"][k, e]), "bumps": int(out["bumps"][k, e])})
    json.dump({"controller": a.controller, "pool": pool, "episodes": res}, open(a.out.replace(".npz", ".json"), "w"), indent=1)
    succ = sum(r["outcome"] == "success" for r in res)
    print(f"{a.controller}: {T} episodes, success {succ / T:.2f}, steps {len(rec)} -> {a.out}")
