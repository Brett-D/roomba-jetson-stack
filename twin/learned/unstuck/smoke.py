"""Smoke test: build the env, drive every robot with a fixed pattern, print sensor statistics."""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--tiles", default=os.path.join(HERE, "data", "tiles_train.json"))
    a = ap.parse_args()
    import unstuck_env as U

    cfg = U.UnstuckEnvCfg()
    cfg.scene.num_envs = a.num_envs
    cfg.tiles_path = a.tiles
    with launch_simulation(cfg, {"physics": "ovphysx"}):
        t0 = time.time()
        env = U.UnstuckEnv(cfg)
        print(f"env built in {time.time() - t0:.1f}s")
        obs, _ = env.reset()
        o = obs["policy"]
        print("obs", tuple(o.shape), "patch lethal frac", o[:, :576].mean().item())
        p0 = env.robot.data.root_pos_w.torch.clone()
        t0 = time.time()
        for i in range(a.steps):
            act = torch.zeros(env.num_envs, 2, device=env.device)
            act[:, 0] = 1.0 if (i // 20) % 2 == 0 else -1.0
            act[:, 1] = 0.3
            obs, rew, term, trunc, extras = env.step(act)
            if i % 10 == 0:
                p = env.robot.data.root_pos_w.torch
                print(
                    f"step {i}: z {p[:, 2].mean():.3f} moved {torch.norm(p[:, :2] - p0[:, :2], dim=-1).mean():.3f} "
                    f"v_enc {obs['policy'][:, 584].mean():.2f} ir {env.ir.mean():.3f} bumped {env.bumped.float().mean():.2f} "
                    f"cm_stop {env.cm_stop.float().mean():.2f} reflex {(env.reflex_steps > 0).float().mean():.2f} "
                    f"patch {obs['policy'][:, :576].mean():.3f} clear {env.clear_now.mean():.2f} "
                    f"rew {rew.mean():.3f} term {term.float().sum():.0f} trunc {trunc.float().sum():.0f}"
                )
        dt = time.time() - t0
        print(f"{a.steps} steps in {dt:.1f}s = {a.steps * env.num_envs / dt:.0f} env-steps/s; log {extras.get('log')}")
        env.close()
