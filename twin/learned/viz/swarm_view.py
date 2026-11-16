"""Watch the unstuck skill's training environment live: N simulated Roombas at once in Isaac Lab's
Newton GL viewer (kit-less OVPhysX, the physics the policy was trained with), each in its own
generated stuck situation, driven by a training checkpoint.

  . twin\\learned\\env.ps1
  python twin\\learned\\viz\\swarm_view.py                    # 1536 robots, model_0 -> model_50 -> model_300, loops
  python twin\\learned\\viz\\swarm_view.py --ckpts model_300   # final policy only

Same env and physics as train_unstuck.py (unstuck_env.UnstuckEnv, dt 0.02 s, decimation 5, the
training tiles and domain randomisation). Viewer-only additions:
- every robot keeps its own tile (training draws tiles at random, so some held two robots);
- a robot that got free pauses where it is for --freed_hold_s (wheels stopped) before its next
  stuck episode starts, so you can see where it drove (the policy's actions are ignored meanwhile);
- robots are drawn bright cyan with a ring on the floor coloured by state (blue driving, grey idle,
  red bump, green freed) and a fading line of their last ~4 s of driving; walls / furniture are
  semi-transparent; an orange (drop) or grey (15 s up) flash where an episode ended that way.
Actions are sampled from the checkpoint's policy (mean + its learned noise), as during training.

Camera (--tour orbit, default): one smooth loop (--orbit_s) driven by wall-clock time and evaluated
every rendered frame - a slow orbit around the grid at constant angular speed that starts high and
wide over all tiles, sweeps down to a low pass across them and back up, closing seamlessly. The
simulation runs at a constant --speed (0.6 x real time) with every frame paced evenly on the wall
clock. Any mouse or camera key input hands the camera to you for 30 s; it then blends back.
--record OUT.mp4 writes every rendered frame (overlay included) from the framebuffer to an H.264 clip
of --record_s seconds and exits; camera and simulation then advance exactly one frame per video frame.
Viewer keys: middle-drag orbit, Shift + middle-drag pan, scroll zoom, left-drag look, W A S D Q E
fly, F frame everything, Space pause, H hide the overlay, Esc quit. Windowed GPU apps on this PC
need DISABLE_RTSS_LAYER=1 (see swarm_view.ps1 next to this file).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UNSTUCK = os.path.normpath(os.path.join(HERE, "..", "unstuck"))
sys.path.insert(0, UNSTUCK)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402

LABELS = {
    "model_0": "iteration 0 - start of training (after the first of 312 PPO updates)",
    "model_50": "iteration 50 - early in training",
    "model_100": "iteration 100",
    "model_150": "iteration 150",
    "model_200": "iteration 200",
    "model_250": "iteration 250",
    "model_300": "iteration 300 - final; the checkpoint exported to ONNX (not deployed on the robot)",
}
RING = {
    "drive": (0.10, 0.40, 1.00),
    "idle": (0.42, 0.42, 0.46),
    "bump": (1.00, 0.08, 0.05),
    "free": (0.10, 0.95, 0.20),
}
BURST = {"drop": (1.0, 0.55, 0.0), "timeout": (0.55, 0.55, 0.55)}
BODY = (0.0, 0.92, 1.0)
TRAIL = (0.02, 0.12, 0.75)
FLOOR = (0.82, 0.80, 0.76)
TAKEOVER_S = 30.0


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=1536)
    ap.add_argument("--tiles", default=os.path.join(UNSTUCK, "data", "tiles_train.json"))
    ap.add_argument("--run_dir", default="E:/twin/learned/runs/unstuck/main")
    ap.add_argument("--ckpts", default="model_0,model_50,model_300", help="comma-separated checkpoint names in --run_dir")
    ap.add_argument("--loop", type=int, default=1, help="1 = cycle the checkpoints until the window is closed")
    ap.add_argument("--tile_mode", choices=["fixed", "random"], default="fixed", help="fixed: one robot per tile")
    ap.add_argument("--tour", choices=["orbit", "none"], default="orbit")
    ap.add_argument("--orbit_s", type=float, default=100.0, help="wall seconds per camera loop (high and wide -> low pass -> high)")
    ap.add_argument("--orbit_phase", type=float, default=0.0, help="where in the loop the camera starts (0 = high, 0.5 = lowest)")
    ap.add_argument("--speed", type=float, default=0.6, help="simulated / wall time, constant (frames are paced evenly)")
    ap.add_argument("--segment_s", type=float, default=30.0, help="simulated seconds per checkpoint")
    ap.add_argument("--freed_hold_s", type=float, default=1.5, help="simulated seconds a freed robot pauses before restarting")
    ap.add_argument("--solid_opacity", type=float, default=0.6, help="walls / furniture")
    ap.add_argument("--max_s", type=float, default=0.0, help="stop after this many wall seconds (0 = never)")
    ap.add_argument("--panel", action="store_true", help="keep the Newton side panel")
    ap.add_argument("--no_keys", action="store_true", help="hide the controls line (recordings)")
    ap.add_argument("--render_every", type=int, default=1, help="draw every N physics steps (50 Hz / N)")
    ap.add_argument("--hold_s", type=float, default=0.0, help="show the start pose this long before running (recordings)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ready_file", default="", help="written (pid) once the window shows the robots")
    ap.add_argument("--window_title", default="Isaac Lab - Roomba unstuck swarm")
    ap.add_argument("--record", default="", help="write every rendered frame (with the overlay) to this MP4 and exit")
    ap.add_argument("--record_s", type=float, default=45.0, help="video length; the camera and the sim advance per frame")
    ap.add_argument("--split_label", default="", help="half of a before / after split screen: big label (| = new line) + first-try counter")
    return ap.parse_args()


class Policy:
    """RSL-RL actor checkpoint: normalised MLP mean + learned Gaussian noise, as sampled in training."""

    def __init__(self, path: str, device: str):
        from export_onnx import load_actor

        self.actor = load_actor(path).to(device)
        self.std = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]["distribution.std_param"].float()

    @torch.no_grad()
    def __call__(self, obs):
        a = self.actor
        mean = a.mlp((obs - a.mean) / (a.std + 1e-2))
        return (mean + self.std * torch.randn_like(mean)).clamp(-1.0, 1.0)


def tile_meshes(world, scenarios, wmod):
    """Static visual meshes of every tile by category (points [M, 3], triangles [K, 3])."""
    import trimesh
    from shapely.geometry import Polygon

    S = scenarios
    groups: dict[str, list] = {k: [] for k in ("floor", "solid", "railing", "overhang", "high", "rug", "mark", "pit")}
    for t, (cx, cy) in zip(world.tiles, world.centers):
        for m in wmod._floor_mesh(t.holes):
            m.apply_translation((cx, cy, 0.0))
            groups["floor"].append(m)
        for hx0, hy0, hx1, hy1 in t.holes:
            m = trimesh.creation.box(extents=(hx1 - hx0, hy1 - hy0, 0.01))
            m.apply_translation((cx + (hx0 + hx1) / 2, cy + (hy0 + hy1) / 2, wmod.CATCH_Z))
            groups["pit"].append(m)
        for p in t.prims:
            m = wmod._prim_mesh(p)
            m.apply_translation((cx, cy, 0.0))
            if p.z0 >= S.BLOCK_Z:
                groups["high"].append(m)
            elif p.z0 > S.BODY_TOP:
                groups["overhang"].append(m)
            elif p.z1 < S.STEP_BLOCK:
                groups["rug"].append(m)
            elif p.depth == "none":
                groups["railing"].append(m)
            else:
                groups["solid"].append(m)
        for mk in t.marks:
            poly = Polygon(np.asarray(mk, float))
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.area <= 0:
                continue
            m = trimesh.creation.extrude_polygon(poly, 0.003)
            m.apply_translation((cx, cy, 0.001))
            groups["mark"].append(m)
    out = {}
    for k, ms in groups.items():
        if ms:
            m = trimesh.util.concatenate(ms)
            out[k] = (np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32))
    return out


def ring_mesh(r0=0.19, r1=0.31, n=40):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.concatenate([np.stack([r0 * np.cos(ang), r0 * np.sin(ang), np.zeros(n)], -1),
                          np.stack([r1 * np.cos(ang), r1 * np.sin(ang), np.zeros(n)], -1)]).astype(np.float32)
    tri = []
    for i in range(n):
        j = (i + 1) % n
        tri += [(i, n + i, n + j), (i, n + j, j)]
    return pts, np.asarray(tri, np.int32)


def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def main():
    a = parse()
    import scenarios as S
    import unstuck_env as U
    import world as W
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    MESH_STYLE = {  # colour, opacity
        "floor": (FLOOR, None),
        "pit": ((0.10, 0.10, 0.12), None),
        "solid": ((0.30, 0.35, 0.40), a.solid_opacity if a.solid_opacity < 1 else None),
        "railing": ((0.62, 0.20, 0.72), None),
        "overhang": ((1.0, 0.60, 0.10), 0.45),
        "high": ((0.75, 0.80, 0.84), 0.3),
        "rug": ((0.55, 0.80, 0.76), None),
        "mark": ((0.90, 0.20, 0.20), 0.6),
    }
    ckpts = [c.strip() for c in a.ckpts.split(",") if c.strip()]
    world = W.load_world(a.tiles)
    T = len(world.tiles)
    N = a.num_envs
    centers = world.centers
    cols = int(math.ceil(math.sqrt(T)))
    ext = float(np.abs(centers).max() + S.TILE / 2)

    cfg = U.UnstuckEnvCfg()
    cfg.scene.num_envs = N
    cfg.tiles_path = a.tiles
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 0.02, 5, a.render_every  # physics as trained
    cfg.seed = a.seed
    wide_eye = np.array((0.0, -0.98 * ext, 1.28 * ext))
    wide_at = np.array((0.0, 0.04 * ext, 0.0))
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(
            window_width=a.width,
            window_height=a.height,
            eye=tuple(wide_eye),
            lookat=tuple(wide_at),
            background_color=(0.07, 0.08, 0.10),
            enable_live_plots=False,
            enable_markers=False,
            enable_picking=False,
            streaming_view=False,
            show_contacts=False,
        )
    ]

    class ViewEnv(U.UnstuckEnv):
        """Training env + one fixed tile per robot + a viewer-only pause after a robot got free."""

        def __init__(self, cfg, **kw):
            self.fixed_tiles = None
            self.ready = False
            super().__init__(cfg, **kw)
            dev = self.device
            if a.tile_mode == "fixed":
                g = torch.Generator(device="cpu").manual_seed(a.seed)
                perm = torch.randperm(self.T, generator=g) if N <= self.T else torch.arange(N) % self.T
                self.fixed_tiles = perm[:N].to(dev)
            self.end_pos = torch.zeros(N, 3, device=dev)
            self.end_outcome = torch.zeros(N, dtype=torch.long, device=dev)
            self.end_t = torch.zeros(N, device=dev)
            self.ended = torch.zeros(N, dtype=torch.bool, device=dev)
            self.hold_until = torch.full((N,), -1.0, device=dev)
            self.clock = 0.0
            self.clock_render = 0.0  # physics time drawn so far (frame pacing)
            self.force_reset = False
            self.ready = True

        def holding(self):
            """Freed robots waiting for their reset (wheels stopped until the reset happens)."""
            return self.hold_until >= 0

        def _get_dones(self):
            terminated, truncated = super()._get_dones()
            if self.ready and a.freed_hold_s > 0:
                # an expired hold ends the episode even if the robot no longer passes the success test
                expired = (self.hold_until >= 0) & (self.hold_until <= self.clock)
                terminated = terminated | expired
                truncated = truncated & ~terminated
            return terminated, truncated

        def reset(self, *args, **kw):
            self.force_reset = True
            try:
                return super().reset(*args, **kw)
            finally:
                self.force_reset = False

        def step(self, action):
            out = super().step(action)
            self.clock += self.step_dt
            return out

        def _apply_action(self):
            if self.ready and a.freed_hold_s > 0:
                h = self.holding()
                if h.any():  # a freed robot waits with its wheels stopped
                    self.wheel_targets = torch.where(h.unsqueeze(-1), torch.zeros_like(self.wheel_targets), self.wheel_targets)
            super()._apply_action()

        def _reset_idx(self, env_ids):
            ids = torch.as_tensor(self.robot._ALL_INDICES if env_ids is None else env_ids, device=self.device, dtype=torch.long)
            if not self.ready:
                return self._real_reset(ids)
            if self.force_reset or a.freed_hold_s <= 0:
                self._record_end(ids)
                self.hold_until[ids] = -1.0
                return self._real_reset(ids)
            started = self.hold_until[ids] >= 0
            expired = started & (self.hold_until[ids] <= self.clock)
            freed = ~started & (self.outcome[ids] == 1)
            other = ~started & ~freed
            if freed.any():
                self._record_end(ids[freed])
                self.hold_until[ids[freed]] = self.clock + a.freed_hold_s
            if other.any():
                self._record_end(ids[other])
            now = ids[expired | other]
            if len(now):
                self.hold_until[now] = -1.0
                self._real_reset(now)

        def _record_end(self, ids):
            self.end_pos[ids] = self.robot.data.root_pos_w.torch[ids]
            self.end_outcome[ids] = self.outcome[ids]
            self.end_t[ids] = self.t[ids]
            self.ended[ids] = True

        def _real_reset(self, ids):
            if self.fixed_tiles is None:
                return super()._reset_idx(ids)
            real = torch.randint
            n = len(ids)

            def pick(low, high=None, size=None, *args, **kw):
                if high == self.T and size is not None and tuple(size) == (n,):
                    return self.fixed_tiles[ids]
                return real(low, high, size, *args, **kw) if high is not None else real(low, size, *args, **kw)

            torch.randint = pick
            try:
                super()._reset_idx(ids)
            finally:
                torch.randint = real

    with launch_simulation(cfg, {"physics": "ovphysx"}):
        import warp as wp
        from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer

        t_build = time.time()
        env = ViewEnv(cfg)
        vis = next(v for v in env.sim.visualizers if isinstance(v, NewtonGLVisualizer))
        dev = env.device
        print(f"[swarm] env with {N} robots on {T} tiles ready in {time.time() - t_build:.0f} s", flush=True)

        t0 = time.time()
        for name, (pts, tri) in tile_meshes(world, S, W).items():
            col, op = MESH_STYLE[name]
            vis.log_mesh(f"/tiles/{name}", wp.array(pts, dtype=wp.vec3, device=dev), wp.array(tri.reshape(-1), dtype=wp.int32, device=dev),
                         color=col, opacity=op, backface_culling=False)
        rp, rt = ring_mesh()
        vis.log_mesh("/swarm/ring_mesh", wp.array(rp, dtype=wp.vec3, device=dev), wp.array(rt.reshape(-1), dtype=wp.int32, device=dev),
                     hidden=True, backface_culling=False)
        print(f"[swarm] tile meshes in {time.time() - t0:.0f} s", flush=True)

        viewer = vis._viewer
        try:
            viewer.renderer.window.set_caption(a.window_title)
            viewer.renderer.line_width = 6.0
        except Exception as e:  # noqa: BLE001
            print(f"[swarm] window: {e}", flush=True)
        gui = getattr(viewer, "gui", None)
        if gui is not None and not a.panel:
            gui._render_left_panel = lambda *x, **k: None
            gui._render_stats_overlay = lambda *x, **k: None
        # The viewer's model also holds the tiles' collision mesh (one grey static shape): hide it, the
        # coloured tile meshes replace it. Robots: bright body colour. Physics is unaffected.
        for batch in getattr(viewer, "_shape_instances", {}).values():
            if batch.static:
                batch.flags = 0
        try:
            m = viewer.model
            body = m.shape_body.numpy()
            colors = m.shape_color.numpy()
            colors[body >= 0] = BODY
            m.shape_color.assign(colors)
            if hasattr(viewer, "_sync_shape_colors_from_model"):
                viewer._sync_shape_colors_from_model()
            for batch in viewer._shape_instances.values():
                if not batch.static and isinstance(batch.colors, wp.array):
                    batch.colors.fill_(wp.vec3(*BODY))
                    batch.colors_changed = True
        except Exception as e:  # noqa: BLE001
            print(f"[swarm] robot colour: {e}", flush=True)

        # ---- per-robot state for the overlay drawing ----
        K = 21  # trail samples, every 0.2 s of simulated time (4 s)
        state = {
            "bump_until": torch.zeros(N, device=dev),
            "moving": torch.zeros(N, dtype=torch.bool, device=dev),
            "act": torch.zeros(N, device=dev),
            "trail": env.robot.data.root_pos_w.torch.clone().unsqueeze(1).repeat(1, K, 1),
            "bursts": [],
            "t": 0.0,
        }
        ring_q = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(N, 1)
        ring_mat = torch.tensor([0.35, 0.0, 0.0, 0.0], device=dev).repeat(N, 1)
        fade = torch.linspace(0.0, 1.0, K - 1, device=dev).unsqueeze(-1)  # oldest -> newest segment
        floor_c = torch.tensor(FLOOR, device=dev)

        def draw():
            t = state["t"]
            p = env.robot.data.root_pos_w.torch
            height = abs(float(np.asarray(viewer.camera.pos, dtype=np.float32).reshape(-1)[2]))
            s = float(np.clip(height / 40.0, 1.0, 3.0))  # rings grow with distance so the wide shot shows them
            col = torch.tensor(RING["drive"], device=dev).repeat(N, 1)
            col[~state["moving"]] = torch.tensor(RING["idle"], device=dev)
            col[state["bump_until"] > t] = torch.tensor(RING["bump"], device=dev)
            col[env.holding()] = torch.tensor(RING["free"], device=dev)
            xf = torch.cat([p[:, :2], torch.full((N, 1), 0.012, device=dev), ring_q], -1)
            viewer.log_instances("/swarm/rings", "/swarm/ring_mesh", wp.from_torch(xf.contiguous(), dtype=wp.transform),
                                 wp.from_torch(torch.full((N, 3), s, device=dev), dtype=wp.vec3),
                                 wp.from_torch(col.contiguous(), dtype=wp.vec3), wp.from_torch(ring_mat, dtype=wp.vec4))
            # trails: the last 4 s of driving, fading into the floor colour
            tr = state["trail"].clone()
            tr[:, -1] = p
            tr[:, :, 2] = 0.035
            a0, a1 = tr[:, :-1].reshape(-1, 3), tr[:, 1:].reshape(-1, 3)
            base = torch.tensor(TRAIL, device=dev)
            lc = (floor_c + (base - floor_c) * (0.15 + 3.0 * fade).clamp(max=1.0)).unsqueeze(0).expand(N, K - 1, 3).reshape(-1, 3)
            viewer.log_lines("/swarm/trails", wp.from_torch(a0.contiguous(), dtype=wp.vec3), wp.from_torch(a1.contiguous(), dtype=wp.vec3),
                             wp.from_torch(lc.contiguous(), dtype=wp.vec3))
            # flashes where an episode ended with a drop or ran out of time
            keep, P, C, R = [], [], [], []
            for (bt, bp, bc) in state["bursts"]:
                age = t - bt
                if age < 1.6:
                    keep.append((bt, bp, bc))
                    P.append(bp)
                    C.append(bc)
                    R.append(torch.full((bp.shape[0],), 0.08 * s * (1.0 + 1.5 * age), device=dev))
            state["bursts"] = keep
            if P:
                viewer.log_points("/swarm/bursts", wp.from_torch(torch.cat(P).contiguous(), dtype=wp.vec3),
                                  wp.from_torch(torch.cat(R).contiguous()), wp.from_torch(torch.cat(C).contiguous(), dtype=wp.vec3))
            else:
                viewer.log_points("/swarm/bursts", None)

        # ---- camera: one smooth parametric loop driven by wall-clock time, evaluated every rendered frame ----
        # A slow orbit around the grid centre at constant angular speed; radius, height and aim swing
        # smoothly (one cosine per loop) from high and wide over all tiles down to a low pass across
        # them and back up, so the loop closes seamlessly. Mouse / camera keys take over for 30 s, then
        # the camera blends back onto the path over 4 s.
        R_HI, R_LO = 1.30 * ext, 0.55 * ext
        H_HI, H_LO = 1.05 * ext, 9.0
        AIM_LO = 0.55 * R_LO  # at the low point the camera aims at the tiles between it and the centre
        cam = {"t0": time.time() - a.orbit_phase * a.orbit_s, "user_until": 0.0, "blend_from": None, "blend_t0": 0.0}

        def orbit_pose(tw):
            u = ((tw - cam["t0"]) / a.orbit_s) % 1.0
            w = 0.5 - 0.5 * math.cos(2 * math.pi * u)  # 0 high, 1 low, smooth and periodic
            th = -math.pi / 2 + 2 * math.pi * u
            r = R_HI + (R_LO - R_HI) * w
            h = math.exp(math.log(H_HI) + (math.log(H_LO) - math.log(H_HI)) * w)
            d = np.array((math.cos(th), math.sin(th)))
            eye = np.array((*(d * r), h))
            at = np.array((*(d * AIM_LO * w), 0.0))
            return eye, at, w

        def camera_frame(tw):
            if a.tour == "none":
                return
            eye, at, w = orbit_pose(tw)
            cam["w"] = w
            if tw < cam["user_until"]:
                cam["blend_from"] = None
                return
            if cam["blend_from"] is None and cam["user_until"] > 0:
                c = viewer.camera
                e0 = np.asarray(c.pos, dtype=np.float64).reshape(-1)[:3].copy()
                cam["blend_from"] = (e0, e0 + np.asarray(c.get_front(), dtype=np.float64).reshape(-1)[:3] * 20.0)
                cam["blend_t0"] = tw
                cam["user_until"] = 0.0
            if cam["blend_from"] is not None:
                b = smooth((tw - cam["blend_t0"]) / 4.0)
                e0, l0 = cam["blend_from"]
                eye, at = e0 + (eye - e0) * b, l0 + (at - l0) * b
                if b >= 1.0:
                    cam["blend_from"] = None
            vis.set_camera_view(tuple(eye), tuple(at))

        # even frame pacing: every render is scheduled at (physics time drawn) / speed on the wall clock,
        # and the camera uses that scheduled time, so frames and camera steps are uniform
        pace = {"anchor_wall": None, "anchor_sim": 0.0, "frames": [], "report": time.time(), "late": 0}
        render_dt = cfg.sim.dt * a.render_every

        def paced_time():
            sim_t = env.clock_render
            if rec["proc"] is not None:  # recording: one video frame per render, no waiting
                return rec["t0"] + rec["frames"] / rec["fps"]
            if pace["anchor_wall"] is None:
                pace["anchor_wall"], pace["anchor_sim"] = time.time(), sim_t
            target = pace["anchor_wall"] + (sim_t - pace["anchor_sim"]) / max(a.speed, 0.05)
            now = time.time()
            if target > now:
                time.sleep(target - now)
            elif now - target > 0.25:  # can't keep up: re-anchor instead of rushing to catch up
                pace["anchor_wall"], pace["anchor_sim"] = now, sim_t
                pace["late"] += 1
                target = now
            now = time.time()
            pace["frames"].append(now)
            if now - pace["report"] > 20.0 and len(pace["frames"]) > 10:
                f = np.diff(np.array(pace["frames"]))
                print(f"[swarm] frames: {len(f) / (pace['frames'][-1] - pace['frames'][0]):.1f} fps, interval mean "
                      f"{1e3 * f.mean():.1f} ms, p95 {1e3 * np.percentile(f, 95):.1f} ms, max {1e3 * f.max():.1f} ms, "
                      f"re-anchored {pace['late']}x", flush=True)
                pace["frames"], pace["report"], pace["late"] = pace["frames"][-1:], now, 0
            return target  # the camera follows the schedule, so its steps stay uniform even if a frame is late

        # ---- optional recording: every rendered frame, overlay included, straight from the framebuffer ----
        rec = {"proc": None, "frames": 0, "fps": a.speed / render_dt, "t0": time.time(), "n": int(a.record_s * a.speed / render_dt)}
        if a.record:
            import subprocess

            fb_w, fb_h = viewer.renderer._screen_width, viewer.renderer._screen_height
            rec["proc"] = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fb_w}x{fb_h}",
                 "-r", f"{rec['fps']:g}", "-i", "-", "-vf", f"scale={a.width}:{a.height}:flags=lanczos", "-c:v", "libx264", "-preset", "medium",
                 "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.record],
                stdin=subprocess.PIPE)
            print(f"[swarm] recording {rec['n']} frames at {rec['fps']:g} fps ({fb_w}x{fb_h}) to {a.record}", flush=True)
            orig_step = vis.step

            def step_and_grab(dt):
                orig_step(dt)
                if rec["proc"] is None or rec["frames"] >= rec["n"]:
                    return
                img = viewer.get_frame(render_ui=True).numpy()
                rec["proc"].stdin.write(img.tobytes())
                rec["frames"] += 1
                if rec["frames"] % 150 == 0:
                    print(f"[swarm] recorded {rec['frames']}/{rec['n']} frames", flush=True)

            vis.step = step_and_grab
            cam["t0"] = rec["t0"] - a.orbit_phase * a.orbit_s

        def user_input(*args, **kw):
            cam["user_until"] = time.time() + TAKEOVER_S

        try:
            import pyglet

            camera_keys = {getattr(pyglet.window.key, k) for k in ("W", "A", "S", "D", "Q", "E", "F", "UP", "DOWN", "LEFT", "RIGHT")}
            viewer.renderer.window.push_handlers(
                on_mouse_drag=user_input,
                on_mouse_scroll=user_input,
                on_key_press=lambda sym, mod: user_input() if sym in camera_keys else None,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[swarm] input hooks: {e}", flush=True)

        orig_markers = vis._render_markers

        def per_frame(num_envs):
            orig_markers(num_envs)
            try:
                env.clock_render += render_dt
                camera_frame(paced_time())
                draw()
            except Exception as e:  # noqa: BLE001
                print(f"[swarm] frame: {type(e).__name__}: {e}", flush=True)

        vis._render_markers = per_frame

        # ---- overlay ----
        hud = {"lines": [], "big": [], "small": []}

        def overlay(imgui):
            flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.always_auto_resize
                     | imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_focus_on_appearing
                     | imgui.WindowFlags_.no_nav | imgui.WindowFlags_.no_inputs)
            if a.split_label:
                try:
                    base = imgui.get_font_size()
                    h1 = 0.0
                    for win, size, lines, alpha in (("split_big", 2.1, hud["big"], 0.7), ("split_small", 1.3, hud["small"], 0.6)):
                        imgui.set_next_window_pos(imgui.ImVec2(14, 14 + (h1 + 6 if h1 else 0)))
                        imgui.set_next_window_bg_alpha(alpha)
                        imgui.begin(win, None, flags)
                        try:
                            imgui.push_font(None, base * size)
                            for txt, c in lines:
                                imgui.text_colored(imgui.ImVec4(*c, 1.0), txt)
                            imgui.pop_font()
                            h1 = imgui.get_window_size().y
                        finally:
                            imgui.end()
                except Exception as e:  # noqa: BLE001
                    print(f"[swarm] overlay: {e}", flush=True)
                return
            try:
                imgui.set_next_window_pos(imgui.ImVec2(18, 16))
                imgui.set_next_window_bg_alpha(0.6)
                imgui.begin("swarm_hud", None, flags)
                try:
                    imgui.set_window_font_scale(1.45)
                except Exception:  # noqa: BLE001
                    pass
                for txt, c in hud["lines"]:
                    if c:
                        imgui.text_colored(imgui.ImVec4(*c, 1.0), txt)
                    else:
                        imgui.text(txt)
                imgui.end()
                if a.no_keys:
                    return
                imgui.set_next_window_pos(imgui.ImVec2(18, a.height - 44))
                imgui.set_next_window_bg_alpha(0.45)
                imgui.begin("swarm_keys", None, flags)
                imgui.text("Any mouse / camera key takes the camera for 30 s:  middle-drag orbit | Shift+middle pan | scroll zoom | "
                           "left-drag look | WASD fly | F frame all | Space pause | H hide text | Esc quit")
                imgui.end()
            except Exception as e:  # noqa: BLE001
                print(f"[swarm] overlay: {e}", flush=True)

        if gui is not None:
            gui.register_ui_callback(overlay, position="free")

        # ---- run ----
        obs, _ = env.reset()
        obs = obs["policy"]
        if a.ready_file and a.hold_s > 0:
            with open(a.ready_file, "w") as f:
                f.write(f"{os.getpid()}\n")
        hud["lines"] = [("Isaac Lab - the unstuck skill's training environment" + ("" if a.record else ", live"), (1.0, 1.0, 1.0)),
                        (f"{N} simulated Roombas at once (PhysX), each stuck in its own generated 4 x 4 m situation", None)]
        if a.hold_s > 0 and rec["proc"] is not None:
            # recording: a fixed number of held frames, so every recording starts the run on the same frame
            n_hold = min(int(round(a.hold_s * rec["fps"])), max(rec["n"] - rec["frames"], 0))
            for _ in range(n_hold):
                env.sim.render()
        else:
            t_hold = time.time()
            while a.hold_s > 0 and time.time() - t_hold < a.hold_s and vis.is_running():
                env.sim.render()
                time.sleep(1 / 60)
        ready = a.hold_s > 0 or not a.ready_file
        wall0 = time.time()
        seg = 0
        while vis.is_running() and not vis.is_closed:
            name = ckpts[seg % len(ckpts)]
            pol = Policy(os.path.join(a.run_dir, name + ".pt"), dev)
            if seg > 0:
                obs, _ = env.reset()
                obs = obs["policy"]
            env.ended[:] = False
            state["bursts"].clear()
            state["bump_until"].zero_()
            state["trail"][:] = env.robot.data.root_pos_w.torch.unsqueeze(1)
            n_end = n_free = n_drop = 0
            first = torch.zeros(N, dtype=torch.long, device=dev)  # outcome of each robot's first attempt (0 = running)
            first_t = torch.zeros(N, device=dev)
            ts = 0.0
            seg_wall = time.time()
            label = LABELS.get(name, name)
            part = f"  [{seg % len(ckpts) + 1}/{len(ckpts)}]" if len(ckpts) > 1 else ""
            print(f"[swarm] segment {seg}: {name} ({label}), std {pol.std.tolist()}", flush=True)
            k_step = 0
            while vis.is_running() and not vis.is_closed:
                act = pol(obs)
                o, _, _, _, _ = env.step(act)
                obs = o["policy"]
                ts += env.step_dt
                k_step += 1
                state["t"] = ts
                jv = env.robot.data.joint_vel.torch[:, env.wheel_ids]
                v = (jv[:, 0] + jv[:, 1]).abs() * U.WHEEL_R / 2
                w = (jv[:, 0] - jv[:, 1]).abs() * U.WHEEL_R / U.WHEEL_BASE
                state["moving"] = ((v > 0.03) | (w > 0.25) | env.bump_new) & ~env.holding()
                state["act"] = 0.9 * state["act"] + 0.1 * state["moving"].float()
                state["bump_until"] = torch.where(env.bump_new, torch.full_like(state["bump_until"], ts + 0.6), state["bump_until"])
                if k_step % 2 == 0:
                    p = env.robot.data.root_pos_w.torch
                    jump = (p[:, :2] - state["trail"][:, -2, :2]).norm(dim=-1) > 0.4  # reset to the start: restart the trail
                    state["trail"] = torch.cat([state["trail"][:, 1:], p.unsqueeze(1)], 1)
                    state["trail"][jump] = p[jump].unsqueeze(1)
                if env.ended.any():
                    ids = env.ended.nonzero().squeeze(-1)
                    oc = env.end_outcome[ids]
                    for kind, mask in (("drop", (oc == 2) | (oc == 3)), ("timeout", oc == 4)):
                        if mask.any():
                            bp = env.end_pos[ids[mask]].clone()
                            bp[:, 2] = bp[:, 2] + 0.2
                            state["bursts"].append((ts, bp, torch.tensor(BURST[kind], device=dev).repeat(bp.shape[0], 1)))
                    n_end += int(len(ids))
                    n_free += int((oc == 1).sum())
                    n_drop += int(((oc == 2) | (oc == 3)).sum())
                    new = first[ids] == 0
                    first[ids[new]] = oc[new]
                    first_t[ids[new]] = env.end_t[ids[new]]
                    env.ended[:] = False
                if not ready:
                    with open(a.ready_file, "w") as f:
                        f.write(f"{os.getpid()}\n")
                    ready = True
                if k_step % 3 and not rec["proc"]:
                    if a.max_s and time.time() - wall0 > a.max_s:
                        break
                    if a.segment_s > 0 and ts >= a.segment_s and (len(ckpts) > 1 or not a.loop):
                        break
                    continue
                f_free = int((first == 1).sum())
                avg = f"{float(first_t[first == 1].mean()):.1f} s" if f_free else "-"
                pct = f"{100.0 * n_free / n_end:.0f} %" if n_end else "-"
                rate = ts / max(time.time() - seg_wall, 1e-6)
                left = cam["user_until"] - time.time()
                cam_line = (f"camera: yours for {left:.0f} s more, then the automatic orbit resumes" if left > 0
                            else f"camera: slow orbit over all {T} tiles, high and wide down to a low pass and back (loops every {a.orbit_s:.0f} s)"
                            if a.tour != "none" else "camera: free")
                title = "Isaac Lab - the unstuck skill's training environment" + ("" if a.record else ", live")
                f_t = float(first_t[first == 1].mean()) if f_free else float("nan")
                hud["big"] = [
                    *[(t, (1.0, 0.85, 0.3)) for t in a.split_label.split("|")],
                    (f"freed on first try: {f_free} / {N} ({100.0 * f_free / N:.0f} %)", (0.55, 1.0, 0.6)),
                ]
                hud["small"] = [
                    (f"mean time to get free: {f_t:.1f} s" if f_free else "mean time to get free: -", (1.0, 1.0, 1.0)),
                    (f"still trying {int((first == 0).sum())}, out of time (15 s) {int((first == 4).sum())}, "
                     f"drops {int(((first == 2) | (first == 3)).sum())}", (0.85, 0.85, 0.85)),
                    (f"{N} simulated Roombas (Isaac Lab, PhysX),", (0.85, 0.85, 0.85)),
                    ("each stuck in its own 4 x 4 m tile", (0.85, 0.85, 0.85)),
                    ("ring: blue driving, grey idle, red bump, green freed", (0.75, 0.85, 1.0)),
                    ("line: last 4 s of driving", (0.75, 0.85, 1.0)),
                    (f"time {ts:4.1f} s (simulated), shown at {a.speed:.1f} x real time", (0.7, 0.7, 0.7)),
                ]
                hud["lines"] = [
                    (title, (1.0, 1.0, 1.0)),
                    (f"{N} simulated Roombas at once (PhysX), each stuck in its own generated 4 x 4 m situation", None),
                    (f"Policy: {name}  ({label}){part}", (1.0, 0.85, 0.3)),
                    (f"time {ts:5.1f} s   first try: freed {f_free} of {N} ({100.0 * f_free / N:.0f} %) in {avg} on average,"
                     f" drops {int(((first == 2) | (first == 3)).sum())}, out of time (15 s) {int((first == 4).sum())},"
                     f" still trying {int((first == 0).sum())}", None),
                    (f"all tries so far: {n_end} episodes, {pct} freed (a freed robot restarts stuck in its tile)", (0.8, 0.8, 0.8)),
                    ("robots cyan; ring: blue driving, grey idle, red bump, green freed (pauses 1.5 s, viewer only); line = last 4 s of driving",
                     (0.75, 0.85, 1.0)),
                    ("tiles: grey walls / furniture   purple railing (invisible to the depth camera)   yellow overhang   black drop",
                     (0.75, 0.85, 1.0)),
                    (cam_line, (0.6, 1.0, 0.7)),
                ]
                if a.record:
                    hud["lines"].append((f"simulation shown at {a.speed:.1f} x real time", (0.7, 0.7, 0.7)))
                elif not a.no_keys:
                    hud["lines"].append((f"sim speed {rate:.2f} x real time", (0.6, 0.6, 0.6)))
                if a.max_s and time.time() - wall0 > a.max_s:
                    break
                if a.segment_s > 0 and ts >= a.segment_s and (len(ckpts) > 1 or not a.loop):
                    break
                if rec["proc"] is not None and rec["frames"] >= rec["n"]:
                    break
            print(f"[swarm] {name}: first try freed {int((first == 1).sum())}/{N}, drops {int(((first == 2) | (first == 3)).sum())}, "
                  f"out of time {int((first == 4).sum())}, running {int((first == 0).sum())}, mean time to free "
                  f"{float(first_t[first == 1].mean()) if int((first == 1).sum()) else float('nan'):.2f} s; all tries {n_end} episodes, "
                  f"{n_free} freed, {n_drop} drops in {ts:.0f} s sim ({time.time() - seg_wall:.0f} s wall)", flush=True)
            seg += 1
            if a.max_s and time.time() - wall0 > a.max_s:
                break
            if not a.loop and seg >= len(ckpts):
                break
            if rec["proc"] is not None and rec["frames"] >= rec["n"]:
                break
        if rec["proc"] is not None:
            rec["proc"].stdin.close()
            rec["proc"].wait()
            print(f"[swarm] wrote {a.record} ({rec['frames']} frames)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
