"""Physics / sensor sanity checks on hand-made tiles: drive speed, turn rate, bumper, light bumpers,
costmap patch, a drivable rug (1.2 cm) and a threshold that must stop the robot (3 cm)."""

from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import scenarios as S  # noqa: E402
from isaaclab.app import launch_simulation  # noqa: E402


def tiles():
    out = []
    t = S.Tile("wall_ahead", [S.wall(1.2, -1.5, 1.2, 1.5)], start=(0.0, 0.0, 0.0), goal=(1.0, 1.5))
    out.append(t)
    out.append(S.Tile("rug", [S.box(0.6, 0, 0.5, 2.0, 0, 0, 0.012)], start=(0.0, 0.0, 0.0), goal=(1.5, 0)))
    out.append(S.Tile("threshold", [S.box(0.6, 0, 0.06, 2.0, 0, 0, 0.03)], start=(0.0, 0.0, 0.0), goal=(1.5, 0)))
    out.append(S.Tile("turn", [], start=(0.0, 0.0, 0.0), goal=(1.5, 0)))
    for t in out:
        t.prims += S._perimeter()
    return out


if __name__ == "__main__":
    path = os.path.join(HERE, "data", "tiles_check.json")
    S.save(tiles(), path)
    import unstuck_env as U

    cfg = U.UnstuckEnvCfg()
    cfg.scene.num_envs = 4
    cfg.tiles_path = path
    cfg.eval_mode = True
    cfg.episode_length_s = 60.0
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 0.02, 5, 5
    with launch_simulation(cfg, {"physics": "ovphysx"}):
        env = U.UnstuckEnv(cfg)
        env.reset()
        env.tile_clear[:] = 0.0  # never "succeed"
        p_prev = env.robot.data.root_pos_w.torch.clone()
        y_prev = env.robot.data.heading_w.torch.clone()
        for i in range(70):
            act = torch.zeros(4, 2, device=env.device)
            act[:3, 0] = 1.0
            act[3, 1] = 1.0
            obs, *_ = env.step(act)
            p = env.robot.data.root_pos_w.torch
            y = env.robot.data.heading_w.torch
            if i % 5 == 4:
                v = torch.norm(p[:, :2] - p_prev[:, :2], dim=-1) / (5 * env.step_dt)
                w = torch.atan2(torch.sin(y - y_prev), torch.cos(y - y_prev)) / (5 * env.step_dt)
                p_prev, y_prev = p.clone(), y.clone()
                x = (p[:, 0] - env.tile_center[env.tile][:, 0]).tolist()
                print(f"t={env.t[0]:.1f} x={[round(a, 3) for a in x]} v={[round(a, 3) for a in v.tolist()]} w3={w[3]:.2f} "
                      f"z={[round(a, 3) for a in p[:, 2].tolist()]} bump={env.bumped.int().tolist()} ir0={[round(a, 2) for a in env.ir[0].tolist()]} "
                      f"stop={env.cm_stop.int().tolist()} reflex={env.reflex_steps.tolist()} bumps={env.ep_bumps.int().tolist()}")
            if i == 25:
                patch = obs["policy"][0, :576].reshape(24, 24)
                print("\n".join("".join("#" if v > 0 else "." for v in row) for row in patch.tolist()))
        env.close()
