"""Batched GPU 2D exploration environment (PyTorch) for learned frontier selection.

Why PyTorch on the GPU and not Isaac Lab: the skill only decides *which* frontier to drive to next;
Nav2 does the driving. One decision = "drive along the planned path to that frontier while the
depth camera maps": a grid problem (ray casting, distance fields, connected components) that
vectorises perfectly over hundreds of houses as [N, H, W] tensors, at ~1000+ decisions per second,
with no physics needed. Isaac Lab would add rigid-body simulation of every wheel turn for no benefit.

Per env (one house each, 0.1 m cells):
  true map (walls, furniture, doors that open / close, people as moving blockers), belief map
  (-1 unknown / 0 free / 1 occupied) built by a 2D ray-cast depth FOV (85 deg, 0.3 m blind zone,
  ~3 m range), Nav2-like global costmap (inscribed 0.17 m, inflation 0.3 m), the explorer's exact
  candidate rules (explorer.rs frontier_goals + next_frontier filters and score), Dijkstra paths on
  the costmap, travel time / distance / battery, blacklist / visited lists, finish + return home.

The policy re-ranks the candidates the rules produce (top K by the rule score); index 0 is always
the rule's own choice, so "rule" = always pick 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import gpu

RES = 0.1
K = 16  # candidates given to the policy
CROP = 64  # map crop cells (0.2 m each -> 12.8 m)
CROP_POOL = 2
N_GLOB = 10
N_FEAT = 16
SQ2 = math.sqrt(2.0)


@dataclass
class ExploreCfg:
    fov_deg: float = 85.0
    rays: int = 48
    range_m: float = 3.0
    range_jitter: float = 0.5  # training: range uniform in range_m +- jitter
    blind_m: float = 0.3
    speed: float = 0.2  # average m/s along paths (RPP 0.25 with accel / slowdowns)
    turn_speed: float = 0.5  # rad/s achieved (0.8 commanded, carpet)
    goal_overhead_s: float = 3.0
    fail_penalty_s: float = 15.0
    max_time_s: float = 1800.0
    battery_minutes: float = 90.0
    low_battery: float = 0.2
    door_toggle_per_s: float = 1.0 / 1200.0
    people_max: int = 2
    # explorer.rs parameters (metres; converted to cells at RES)
    min_frontier_m: float = 0.5  # min_frontier_cells 10 at 0.05 m
    standoff_m: float = 0.3
    goal_search_m: float = 0.8
    min_goal_dist_m: float = 0.4
    blacklist_radius_m: float = 0.75
    visited_radius_m: float = 0.6
    inscribed_m: float = 0.17
    inflation_m: float = 0.3
    size_weight: float = 0.01 / 0.05  # score = d - 0.01 * size_cells_at_0.05m = d - 0.2 * size_m
    retry_failed: bool = True
    # reward
    rew_coverage: float = 10.0
    rew_time: float = 0.01
    rew_fail: float = 1.0
    train: bool = True


def _disk(rad_cells: float, device):
    r = int(math.floor(rad_cells))
    y, x = torch.meshgrid(torch.arange(-r, r + 1, device=device), torch.arange(-r, r + 1, device=device), indexing="ij")
    return ((x * x + y * y).float() <= rad_cells * rad_cells + 1e-6).float()


def dilate(mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """mask [N, H, W] bool -> cells within the kernel of a True cell."""
    k = kernel.shape[-1]
    out = F.conv2d(mask.float().unsqueeze(1), kernel.view(1, 1, k, k), padding=k // 2)
    return out.squeeze(1) > 0.5


class ExploreEnv:
    def __init__(self, layouts: list, n_envs: int, device="cuda", cfg: ExploreCfg | None = None, seed: int = 0, size: int | None = None):
        self.cfg = cfg or ExploreCfg()
        self.dev = torch.device(device)
        self.N = n_envs
        self.g = torch.Generator(device=self.dev)
        self.g.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        H = size or max(l["occ"].shape[0] for l in layouts)
        self.H = self.W = H
        self.layouts = layouts
        L = len(layouts)
        occ = np.ones((L, H, H), np.uint8)
        door = np.zeros((L, H, H), np.int16)
        self.Dmax = max(1, max(len(l["door_closed0"]) for l in layouts))
        closed0 = np.zeros((L, self.Dmax), bool)
        start = np.zeros((L, 2), np.float32)
        origin = np.zeros((L, 2), np.float32)
        for i, l in enumerate(layouts):
            h, w = l["occ"].shape
            occ[i, :h, :w] = l["occ"]
            door[i, :h, :w] = l["door"]
            closed0[i, : len(l["door_closed0"])] = l["door_closed0"]
            start[i] = l["start"]
            origin[i] = l["origin"]
        self.L_occ = torch.tensor(occ, device=self.dev, dtype=torch.bool)
        self.L_door = torch.tensor(door.astype(np.int64), device=self.dev)
        self.L_closed0 = torch.tensor(closed0, device=self.dev)
        self.L_start = torch.tensor(start, device=self.dev)
        self.L_origin = origin
        c = self.cfg
        self.k_insc = _disk(c.inscribed_m / RES, self.dev)
        self.k_infl = _disk(c.inflation_m / RES, self.dev)
        self.k_stand = _disk(c.standoff_m / RES - 1e-3, self.dev)
        self.k_robot = _disk(2.0, self.dev)
        self._fp_off = torch.nonzero(_disk(1.7, self.dev) > 0) - 1
        self.k_person = _disk(2.0, self.dev)
        a = torch.linspace(-math.radians(c.fov_deg) / 2, math.radians(c.fov_deg) / 2, c.rays, device=self.dev)
        self.ray_a = a
        self.steps = torch.arange(1, int((c.range_m + c.range_jitter) / (RES / 2)) + 1, device=self.dev).float() * (RES / 2)
        # 8-neighbour shifts
        self.nb = [(-1, -1, SQ2), (-1, 0, 1.0), (-1, 1, SQ2), (0, -1, 1.0), (0, 1, 1.0), (1, -1, SQ2), (1, 0, 1.0), (1, 1, SQ2)]
        N = n_envs
        z = lambda *s, **kw: torch.zeros(*s, device=self.dev, **kw)  # noqa: E731
        self.lay = z(N, dtype=torch.long)
        self.belief = torch.full((N, H, H), -1, dtype=torch.int8, device=self.dev)
        self.closed = z(N, self.Dmax, dtype=torch.bool)
        self.pos = z(N, 2)
        self.yaw = z(N)
        self.home = z(N, 2)
        self.t = z(N)
        self.dist = z(N)
        self.batt = z(N)
        self.range = z(N)
        self.people = z(N, c.people_max, 2)
        self.n_people = z(N, dtype=torch.long)
        self.BL = 64
        self.black = z(N, self.BL, 2)
        self.n_black = z(N, dtype=torch.long)
        self.visited = z(N, self.BL, 2)
        self.n_visited = z(N, dtype=torch.long)
        self.reached = z(N, dtype=torch.long)
        self.reached_at_retry = torch.full((N,), -1, dtype=torch.long, device=self.dev)
        self.failed = z(N, dtype=torch.long)
        self.reach0 = z(N, H, H, dtype=torch.bool)
        self.reach_area = z(N)
        self.cov = z(N)
        self.done = torch.ones(N, dtype=torch.bool, device=self.dev)
        self.finished = z(N, dtype=torch.bool)
        self.home_time = z(N)
        self.visit_coarse = z(N, H // 5, H // 5, dtype=torch.bool)
        self.revisit_dist = z(N)
        self._gpu = {}
        self.L_reach0 = self._layout_reach()
        self.last_dt = z(N) + 15.0
        self.record = False  # evaluation: keep (t, coverage, distance) after every decision
        self.trace = [[] for _ in range(N)]

    # ------------------------------------------------------------------ helpers
    def _rc(self, xy):
        return (xy[..., 1] / RES).long().clamp(0, self.H - 1), (xy[..., 0] / RES).long().clamp(0, self.W - 1)

    def true_blocked(self):
        occ = self.L_occ[self.lay]
        d = self.L_door[self.lay]
        cl = torch.cat([torch.zeros(self.N, 1, dtype=torch.bool, device=self.dev), self.closed], 1)
        blocked = occ | torch.gather(cl, 1, d.view(self.N, -1)).view_as(d)
        if self.cfg.people_max > 0:
            pm = torch.zeros_like(blocked)
            for p in range(self.cfg.people_max):
                r, c = self._rc(self.people[:, p])
                on = p < self.n_people
                pm[torch.arange(self.N, device=self.dev)[on], r[on], c[on]] = True
            blocked |= dilate(pm, self.k_person)
        return blocked

    # ------------------------------------------------------------------ reset
    def reset(self, ids: torch.Tensor, layout_ids: torch.Tensor | None = None, yaw: torch.Tensor | None = None):
        if len(ids) == 0:
            return
        n = len(ids)
        if layout_ids is None:
            layout_ids = torch.randint(0, len(self.layouts), (n,), device=self.dev, generator=self.g)
        self.lay[ids] = layout_ids
        self.belief[ids] = -1
        self.closed[ids] = self.L_closed0[layout_ids]
        st = self.L_start[layout_ids]
        self.pos[ids] = st
        self.yaw[ids] = yaw if yaw is not None else torch.rand(n, device=self.dev, generator=self.g) * 2 * math.pi - math.pi
        self.home[ids] = st
        self.t[ids] = 0
        self.dist[ids] = 0
        c = self.cfg
        if c.train:
            self.batt[ids] = 0.3 + 0.7 * torch.rand(n, device=self.dev, generator=self.g)
            self.range[ids] = c.range_m + c.range_jitter * (2 * torch.rand(n, device=self.dev, generator=self.g) - 1)
            self.n_people[ids] = torch.randint(0, c.people_max + 1, (n,), device=self.dev, generator=self.g)
        else:
            self.batt[ids] = 1.0
            self.range[ids] = c.range_m
            self.n_people[ids] = 0
        self.n_black[ids] = 0
        self.n_visited[ids] = 0
        self.reached[ids] = 0
        self.reached_at_retry[ids] = -1
        self.failed[ids] = 0
        self.done[ids] = False
        self.finished[ids] = False
        self.home_time[ids] = 0
        self.visit_coarse[ids] = False
        self.revisit_dist[ids] = 0
        for i in ids.tolist():
            self.trace[i] = [(0.0, 0.0, 0.0)]
        self.reach0[ids] = self.L_reach0[layout_ids]
        self.reach_area[ids] = self.reach0[ids].float().sum((1, 2)).clamp(min=1)
        self._place_people(ids)
        # initial look: what is in front of the robot
        poses = torch.stack([self.pos[ids, 0], self.pos[ids, 1], self.yaw[ids]], -1).unsqueeze(1)
        self._sense_ids(ids, poses, torch.ones(n, 1, dtype=torch.bool, device=self.dev))
        self.cov[ids] = self._coverage(ids)

    def _layout_reach(self):
        """Per layout: floor the robot can map from its start with the initial doors (the coverage
        denominator): cells reachable by the robot centre with 0.2 m clearance, plus a 0.2 m band."""
        Ln, H = len(self.layouts), self.H
        out = torch.zeros(Ln, H, H, dtype=torch.bool, device=self.dev)
        for b0 in range(0, Ln, 64):
            sl = slice(b0, min(b0 + 64, Ln))
            occ = self.L_occ[sl]
            d = self.L_door[sl]
            cl = torch.cat([torch.zeros(occ.shape[0], 1, dtype=torch.bool, device=self.dev), self.L_closed0[sl]], 1)
            blocked = occ | torch.gather(cl, 1, d.view(occ.shape[0], -1)).view_as(d)
            free_ok = ~dilate(blocked, self.k_robot)
            r, cc = self._rc(self.L_start[sl])
            R = torch.zeros_like(free_ok)
            R[torch.arange(occ.shape[0], device=self.dev), r, cc] = True
            R &= free_ok
            for it in range(6 * H):
                Rn = F.max_pool2d(R.float().unsqueeze(1), 3, 1, 1).squeeze(1).bool() & free_ok
                if it % 16 == 15 and torch.equal(Rn, R):
                    break
                R = Rn
            out[sl] = dilate(R, self.k_robot) & ~blocked
        return out

    def _place_people(self, ids):
        if self.cfg.people_max == 0 or len(ids) == 0:
            return
        occ = self.L_occ[self.lay[ids]]
        n, P, T = len(ids), self.cfg.people_max, 8
        r = torch.randint(0, self.H, (n, P, T), device=self.dev, generator=self.g)
        c = torch.randint(0, self.W, (n, P, T), device=self.dev, generator=self.g)
        ok = ~occ[torch.arange(n, device=self.dev)[:, None, None], r, c]
        j = ok.float().argmax(-1, keepdim=True)
        pr, pc = r.gather(-1, j).squeeze(-1), c.gather(-1, j).squeeze(-1)
        new = torch.stack([pc.float() * RES + RES / 2, pr.float() * RES + RES / 2], -1)
        self.people[ids] = torch.where(ok.any(-1, keepdim=True), new, self.people[ids])

    def _coverage(self, ids):
        known_free = self.belief[ids] == 0
        return ((known_free & self.reach0[ids]).float().sum((1, 2)) / self.reach_area[ids]).clamp(max=1.0)

    # ------------------------------------------------------------------ sensing
    def _sense_ids(self, ids, poses, valid, chunk=12):
        """poses [n, M, 3] (x, y, yaw) for envs ids; ray-cast the FOV and update the belief."""
        blocked_all = self.true_blocked()[ids]
        n, M, _ = poses.shape
        H = self.H
        rng = self.range[ids]
        st = self.steps
        S = len(st)
        for m0 in range(0, M, chunk):
            P = poses[:, m0:m0 + chunk]
            V = valid[:, m0:m0 + chunk]
            m = P.shape[1]
            ang = P[..., 2:3] + self.ray_a  # [n, m, R]
            dx, dy = torch.cos(ang), torch.sin(ang)
            px = P[..., 0:1, None] + dx[..., None] * st  # [n, m, R, S]
            py = P[..., 1:2, None] + dy[..., None] * st
            c = (px / RES).long()
            r = (py / RES).long()
            inside = (c >= 0) & (c < H) & (r >= 0) & (r < H)
            c, r = c.clamp(0, H - 1), r.clamp(0, H - 1)
            flat = (r * H + c).view(n, -1)
            hit = torch.gather(blocked_all.view(n, -1), 1, flat).view_as(px) | ~inside
            within = st <= rng.view(n, 1, 1, 1)
            hit_in = hit & within
            first = torch.where(hit_in.any(-1), hit_in.float().argmax(-1), torch.full(hit_in.shape[:-1], S, device=self.dev, dtype=torch.long))
            idx = torch.arange(S, device=self.dev)
            free = (idx < first.unsqueeze(-1)) & within & (st >= self.cfg.blind_m) & inside & V[..., None, None]
            occm = (idx == first.unsqueeze(-1)) & (st >= self.cfg.blind_m) & inside & V[..., None, None]
            bflat = self.belief[ids].view(n, -1).clone()
            rows = torch.arange(n, device=self.dev)[:, None].expand_as(flat)
            fm, om = free.view(n, -1), occm.view(n, -1)
            bflat[rows[fm], flat[fm]] = 0
            bflat[rows[om], flat[om]] = 1
            # RTAB-Map clears the robot's own footprint (r 0.17 m) in its grid
            fp = self._fp_off  # [F, 2] (dr, dc)
            pr = (P[..., 1] / RES).long()[..., None] + fp[:, 0]
            pc = (P[..., 0] / RES).long()[..., None] + fp[:, 1]
            okf = V[..., None] & (pr >= 0) & (pr < H) & (pc >= 0) & (pc < H)
            fflat = (pr.clamp(0, H - 1) * H + pc.clamp(0, H - 1)).view(n, -1)
            fmask = okf.view(n, -1)
            frows = torch.arange(n, device=self.dev)[:, None].expand_as(fflat)
            bflat[frows[fmask], fflat[fmask]] = 0
            self.belief[ids] = bflat.view(n, H, H)

    # ------------------------------------------------------------------ costmap, distance field, candidates
    def costmap(self):
        b = self.belief
        known = b >= 0
        occ = b == 1
        insc = dilate(occ, self.k_insc)
        infl = dilate(occ, self.k_infl)
        passable = known & ~insc
        free = known & ~infl
        unk_near = dilate(~known, self.k_stand)
        # outside the grid counts as unknown
        unk_near[:, :3, :] = True
        unk_near[:, -3:, :] = True
        unk_near[:, :, :3] = True
        unk_near[:, :, -3:] = True
        return known, occ, passable, free, infl, unk_near

    def distance_field(self, src_rc, passable, infl, max_iter=None):
        """Geodesic cost-to-go from src over passable cells (8-connected; cells within the
        inflation radius cost 2x, as Nav2's planner prefers the middle of free space)."""
        N, H, W = passable.shape
        inf = float("inf")
        cost = torch.where(passable, 1.0 + infl.float(), torch.full_like(passable, inf, dtype=torch.float32))
        D = torch.full((N, H, W), inf, device=self.dev)
        ar = torch.arange(N, device=self.dev)
        D[ar, src_rc[0], src_rc[1]] = 0.0
        key = ("relax", N)
        if key not in self._gpu:
            self._gpu[key] = gpu.Relaxer(N, H, W, self.dev)
        return self._gpu[key].run(D, cost)

    def _seed(self, passable):
        r, c = self._rc(self.pos)
        N, H, W = passable.shape
        best_r, best_c = r.clone(), c.clone()
        found = passable[torch.arange(N, device=self.dev), r, c]
        for rad in range(1, 4):
            if found.all():
                break
            dr, dc = torch.meshgrid(torch.arange(-rad, rad + 1, device=self.dev), torch.arange(-rad, rad + 1, device=self.dev), indexing="ij")
            dr, dc = dr.reshape(-1), dc.reshape(-1)
            rr = (r[:, None] + dr).clamp(0, H - 1)
            cc = (c[:, None] + dc).clamp(0, W - 1)
            ok = passable[torch.arange(N, device=self.dev)[:, None], rr, cc]
            d2 = (dr * dr + dc * dc).float().expand_as(ok)
            d2 = torch.where(ok, d2, torch.full_like(d2, 1e9))
            j = d2.argmin(1)
            hit = ok.any(1) & ~found
            best_r = torch.where(hit, rr[torch.arange(N, device=self.dev), j], best_r)
            best_c = torch.where(hit, cc[torch.arange(N, device=self.dev), j], best_c)
            found |= ok.any(1)
        return (best_r, best_c), found

    def frontier_groups(self, known, free):
        """explorer.rs find_frontiers: free cells with a 4-neighbour unknown (outside = unknown),
        8-connected groups. Returns per-cell group labels [N, H, W] (0 = none)."""
        N, H, W = known.shape
        unk = F.pad((~known).float(), (1, 1, 1, 1), value=1.0)
        n4 = (unk[:, :-2, 1:-1] + unk[:, 2:, 1:-1] + unk[:, 1:-1, :-2] + unk[:, 1:-1, 2:]) > 0
        fr = free & n4
        lab = torch.where(fr, torch.arange(1, H * W + 1, device=self.dev, dtype=torch.float32).view(1, H, W).expand(N, H, W), torch.zeros(N, H, W, device=self.dev))
        key = ("lab", N)
        if key not in self._gpu:
            self._gpu[key] = gpu.Labeler(N, H, W, self.dev)
        lab = self._gpu[key].run(lab, fr)
        return fr, lab

    def candidates(self):
        """Explorer candidate frontier goals for every env. Returns a dict of tensors."""
        c = self.cfg
        N, H, W = self.N, self.H, self.W
        dev = self.dev
        known, occ, passable, free, infl, unk_near = self.costmap()
        seed, has_seed = self._seed(passable)
        D = self.distance_field(seed, passable, infl)
        reach = torch.isfinite(D)
        clear = reach & free & ~unk_near
        fr, lab = self.frontier_groups(known, free)
        # group statistics over all envs at once
        env_i, rr, cc = torch.nonzero(fr, as_tuple=True)
        keys = env_i * (H * W + 1) + lab[env_i, rr, cc].long()
        ukeys, inv = torch.unique(keys, return_inverse=True)
        G = len(ukeys)
        cnt = torch.zeros(G, device=dev).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float32))
        sr = torch.zeros(G, device=dev).index_add_(0, inv, rr.float())
        sc = torch.zeros(G, device=dev).index_add_(0, inv, cc.float())
        anyreach = torch.zeros(G, device=dev).index_add_(0, inv, reach[env_i, rr, cc].float()) > 0
        genv = ukeys // (H * W + 1)
        cy, cx = sr / cnt.clamp(min=1), sc / cnt.clamp(min=1)
        ok = (cnt >= c.min_frontier_m / RES) & anyreach & has_seed[genv]
        # goal cell: nearest clear cell to the centroid within goal_search
        rad = int(math.ceil(c.goal_search_m / RES))
        o = torch.arange(-rad, rad + 1, device=dev)
        orr, occ_ = torch.meshgrid(o, o, indexing="ij")
        orr, occ_ = orr.reshape(-1), occ_.reshape(-1)
        br = torch.round(cy).long()[:, None] + orr
        bc = torch.round(cx).long()[:, None] + occ_
        inb = (br >= 0) & (br < H) & (bc >= 0) & (bc < W)
        cl = clear[genv[:, None].expand_as(br), br.clamp(0, H - 1), bc.clamp(0, W - 1)] & inb
        d2 = (br.float() - cy[:, None]) ** 2 + (bc.float() - cx[:, None]) ** 2
        d2 = torch.where(cl & (d2 <= (c.goal_search_m / RES) ** 2), d2, torch.full_like(d2, 1e9))
        j = d2.argmin(1)
        ok &= d2.gather(1, j[:, None]).squeeze(1) < 1e8
        gr = br.gather(1, j[:, None]).squeeze(1)
        gc = bc.gather(1, j[:, None]).squeeze(1)
        gx, gy = gc.float() * RES + RES / 2, gr.float() * RES + RES / 2
        fx, fy = cx * RES + RES / 2, cy * RES + RES / 2
        # explorer filters: blacklist, visited, min distance (marks visited)
        px, py = self.pos[genv, 0], self.pos[genv, 1]
        d = torch.hypot(gx - px, gy - py)

        def near(lst, n, radius):
            pts = lst[genv]  # [G, BL, 2]
            valid = torch.arange(self.BL, device=dev)[None] < n[genv][:, None]
            dd = torch.hypot(pts[..., 0] - gx[:, None], pts[..., 1] - gy[:, None])
            return ((dd < radius) & valid).any(1), ((dd < 1.5) & valid).sum(1)

        blk, blk_n = near(self.black, self.n_black, c.blacklist_radius_m)
        vis, vis_n = near(self.visited, self.n_visited, c.visited_radius_m)
        ok &= ~blk & ~vis
        too_close = ok & (d < c.min_goal_dist_m)
        if too_close.any():
            for g in torch.nonzero(too_close).squeeze(-1).tolist():
                e = int(genv[g])
                k = int(self.n_visited[e])
                if k < self.BL:
                    self.visited[e, k] = torch.stack([gx[g], gy[g]])
                    self.n_visited[e] = k + 1
        ok &= ~too_close
        size_m = cnt * RES
        score = d - c.size_weight * size_m
        # top-K per env by score (rule order)
        score_ok = torch.where(ok, score, torch.full_like(score, 1e9))
        order = torch.argsort(genv.float() * 1e6 + score_ok.clamp(-1e5, 1e5))
        genv_s = genv[order]
        ok_s = ok[order]
        first = torch.zeros(N + 1, dtype=torch.long, device=dev)
        first.index_add_(0, genv_s + 1, torch.ones_like(genv_s))
        start_idx = torch.cumsum(first, 0)[:-1]
        rank = torch.arange(len(order), device=dev) - start_idx[genv_s]
        sel = ok_s & (rank < K)
        gi = order[sel]
        ei = genv_s[sel]
        ki = rank[sel]
        out = {
            "mask": torch.zeros(N, K, dtype=torch.bool, device=dev),
            "goal": torch.zeros(N, K, 2, device=dev),
            "front": torch.zeros(N, K, 2, device=dev),
            "size": torch.zeros(N, K, device=dev),
            "score": torch.zeros(N, K, device=dev),
            "path": torch.zeros(N, K, device=dev),
            "blk_n": torch.zeros(N, K, device=dev),
            "vis_n": torch.zeros(N, K, device=dev),
        }
        out["mask"][ei, ki] = True
        out["goal"][ei, ki] = torch.stack([gx[gi], gy[gi]], -1)
        out["front"][ei, ki] = torch.stack([fx[gi], fy[gi]], -1)
        out["size"][ei, ki] = size_m[gi]
        out["score"][ei, ki] = score[gi]
        out["path"][ei, ki] = D[genv[gi], gr[gi], gc[gi]] * RES
        out["blk_n"][ei, ki] = blk_n[gi].float()
        out["vis_n"][ei, ki] = vis_n[gi].float()
        out["D"] = D
        out["seed"] = seed
        out["known"] = known
        out["occ"] = occ
        out["passable"] = passable
        out["infl"] = infl
        self.cand = out
        return out

    # ------------------------------------------------------------------ observation for the policy
    def observe(self, cand):
        N, H = self.N, self.H
        dev = self.dev
        b = self.belief
        onehot = torch.stack([(b == 0), (b == 1), (b < 0)], 1).float()  # free, occ, unknown
        half = CROP * CROP_POOL // 2
        pad = F.pad(onehot, (half, half, half, half), value=0.0)
        pad[:, 2] = F.pad(onehot[:, 2], (half, half, half, half), value=1.0)
        r, c = self._rc(self.pos)
        ar = torch.arange(CROP * CROP_POOL, device=dev)
        rows = (r[:, None] + ar[None]).clamp(0, H + 2 * half - 1)  # crop starts at r - half (+half pad)
        cols = (c[:, None] + ar[None]).clamp(0, H + 2 * half - 1)
        crop = pad[torch.arange(N, device=dev)[:, None, None, None], torch.arange(3, device=dev)[None, :, None, None], rows[:, None, :, None], cols[:, None, None, :]]
        mapo = F.avg_pool2d(crop, CROP_POOL)
        g = cand
        rel = g["goal"] - self.pos[:, None]
        euc = rel.norm(dim=-1)
        ang = torch.atan2(rel[..., 1], rel[..., 0]) - self.yaw[:, None]
        homed = (g["goal"] - self.home[:, None]).norm(dim=-1)
        # unknown area in a 3.1 m square around each frontier centroid (m^2)
        unk = (b < 0).float().unsqueeze(1)
        box = F.avg_pool2d(unk, 31, 1, 15, count_include_pad=False).squeeze(1) * (31 * 31 * RES * RES)
        fr, fc = self._rc(g["front"])
        gain = box[torch.arange(N, device=dev)[:, None], fr, fc]
        look = g["front"] - g["goal"]
        rank = torch.arange(K, device=dev).float()[None].expand(N, K) / K
        feats = torch.stack(
            [rel[..., 0] / 10, rel[..., 1] / 10, euc / 10, g["path"] / 10, g["size"] / 5, gain / 10,
             torch.cos(ang), torch.sin(ang), homed / 10, g["blk_n"] / 3, g["vis_n"] / 3, g["score"] / 10, rank,
             look[..., 0], look[..., 1], (g["path"] - euc) / 5], -1)
        feats = torch.where(g["mask"][..., None], feats, torch.zeros_like(feats))
        known_free = (b == 0).float().sum((1, 2)) * RES * RES
        hd = self.home - self.pos
        glob = torch.stack(
            [self.batt, self.t / self.cfg.max_time_s, torch.cos(self.yaw), torch.sin(self.yaw), hd[:, 0] / 10, hd[:, 1] / 10,
             g["mask"].float().sum(1) / K, self.failed.float() / 10, known_free / 200, self.n_black.float() / 10], -1)
        return mapo, glob, feats, g["mask"]

    # ------------------------------------------------------------------ acting
    def step(self, choice: torch.Tensor):
        """Drive every active env to its chosen candidate (index into the K list). Envs without
        candidates run the explorer's finish logic. Returns reward [N], done [N], failed [N]."""
        c = self.cfg
        N, H = self.N, self.H
        dev = self.dev
        cand = self.cand
        ar = torch.arange(N, device=dev)
        has = cand["mask"].any(1) & ~self.done
        t0 = self.t.clone()
        cov0 = self.cov.clone()
        # doors and people change while time passes (decided per decision)
        if c.door_toggle_per_s > 0 and c.train:
            p = 1.0 - torch.exp(-self.last_dt * c.door_toggle_per_s)
            flip = torch.rand(N, self.Dmax, device=dev, generator=self.g) < p[:, None]
            self.closed ^= flip
        if c.people_max > 0 and c.train:
            mv = torch.nonzero((self.n_people > 0) & ~self.done).squeeze(-1)
            self._place_people(mv)
        fail = torch.zeros(N, dtype=torch.bool, device=dev)
        if has.any():
            ch = choice.clamp(0, K - 1)
            ch = torch.where(cand["mask"][ar, ch], ch, torch.zeros_like(ch))
            goal = cand["goal"][ar, ch]
            front = cand["front"][ar, ch]
            gr, gc = self._rc(goal)
            R, C, plen = gpu.trace_paths(cand["D"], gr, gc, has, maxlen=4 * H)
            lead = torch.zeros(N, dtype=torch.long, device=dev)
            L = R.shape[1]
            idx = torch.arange(L, device=dev)
            # blockers on the true map along the path (closed doors, people)
            blocked = self.true_blocked()
            bp = blocked[ar[:, None], R, C] & (idx[None] > lead[:, None]) & (idx[None] < plen[:, None])
            first_b = torch.where(bp.any(1), bp.float().argmax(1), torch.full((N,), L, device=dev, dtype=torch.long))
            fail = has & (first_b < L)
            stop = torch.where(fail, torch.maximum(first_b - 3, lead), plen - 1)
            moving = has[:, None] & (idx[None] >= lead[:, None]) & (idx[None] <= stop[:, None])
            px = C.float() * RES + RES / 2
            py = R.float() * RES + RES / 2
            segl = torch.hypot(torch.diff(px, dim=1), torch.diff(py, dim=1)) * moving[:, 1:]
            dist = segl.sum(1)
            ahead = torch.minimum(idx[None] + 3, (plen - 1)[:, None]).clamp(min=0)
            hx = px.gather(1, ahead) - px
            hy = py.gather(1, ahead) - py
            head = torch.atan2(hy, hx)
            sidx = torch.arange(0, L, 3, device=dev)
            sv = moving[:, sidx]
            spos = torch.stack([px[:, sidx], py[:, sidx], head[:, sidx]], -1)
            end_pos = torch.stack([px[ar, stop], py[ar, stop]], -1)
            h0 = head[ar, lead]
            hend = head[ar, (stop - 1).clamp(min=0)]
            yaw_end_goal = torch.atan2(front[:, 1] - end_pos[:, 1], front[:, 0] - end_pos[:, 0])
            yaw_end = torch.where(fail, hend, yaw_end_goal)
            turn0 = torch.atan2(torch.sin(h0 - self.yaw), torch.cos(h0 - self.yaw))
            turn1 = torch.atan2(torch.sin(yaw_end - hend), torch.cos(yaw_end - hend))
            fr6 = torch.linspace(0, 1, 6, device=dev)
            rot0 = torch.stack([self.pos[:, 0:1].expand(N, 6), self.pos[:, 1:2].expand(N, 6), self.yaw[:, None] + turn0[:, None] * fr6], -1)
            rot1 = torch.stack([end_pos[:, 0:1].expand(N, 6), end_pos[:, 1:2].expand(N, 6), hend[:, None] + turn1[:, None] * fr6], -1)
            poses = torch.cat([rot0, spos, rot1], 1)
            pvalid = torch.cat([has[:, None].expand(N, 6), sv, has[:, None].expand(N, 6)], 1)
            ids = torch.nonzero(has).squeeze(-1)
            # only the part of the pose list that some env uses
            used = pvalid.any(0).nonzero().max() + 1
            self._sense_ids(ids, poses[ids, :used], pvalid[ids, :used])
            # revisits: distance driven through 0.5 m cells already driven through before
            cr, ccol = (R // 5).clamp(0, H // 5 - 1), (C // 5).clamp(0, H // 5 - 1)
            seen_before = self.visit_coarse[ar[:, None], cr, ccol]
            self.revisit_dist += (segl * seen_before[:, 1:].float()).sum(1)
            rr_ = ar[:, None].expand_as(cr)
            self.visit_coarse[rr_[moving], cr[moving], ccol[moving]] = True
            # time, distance, battery
            dt = dist / c.speed + (turn0.abs() + turn1.abs()) / c.turn_speed + c.goal_overhead_s + fail.float() * c.fail_penalty_s
            dt = torch.where(has, dt, torch.zeros_like(dt))
            self.t += dt
            self.dist += torch.where(has, dist, torch.zeros_like(dist))
            self.pos = torch.where(has[:, None], end_pos, self.pos)
            self.yaw = torch.where(has, yaw_end, self.yaw)
            # explorer bookkeeping: reached -> visited, failed -> blacklist
            kb = self.n_black.clamp(max=self.BL - 1)
            self.black[ar, kb] = torch.where(fail[:, None], goal, self.black[ar, kb])
            self.n_black = (self.n_black + fail.long()).clamp(max=self.BL)
            ok = has & ~fail
            kv = self.n_visited.clamp(max=self.BL - 1)
            self.visited[ar, kv] = torch.where(ok[:, None], goal, self.visited[ar, kv])
            self.n_visited = (self.n_visited + ok.long()).clamp(max=self.BL)
            self.reached += ok.long()
            self.failed += fail.long()
        # no candidates: retry blacklisted areas once (retry_failed_before_finish), else go home
        none = ~has & ~self.done
        if none.any():
            retry = none & (self.n_black > 0) & (self.reached != self.reached_at_retry) & c.retry_failed
            self.reached_at_retry = torch.where(retry, self.reached, self.reached_at_retry)
            self.n_black = torch.where(retry, torch.zeros_like(self.n_black), self.n_black)
            self.t += none.float() * 20.0  # finish_confirm wait
            self._go_home(none & ~retry)
        self.batt -= (self.t - t0) / (c.battery_minutes * 60.0)
        timeout = ~self.done & ((self.t >= c.max_time_s) | (self.batt < c.low_battery))
        self._go_home(timeout, finished=False)
        self.cov = self._coverage(ar)
        dcov = self.cov - cov0
        dt_all = self.t - t0
        self.last_dt = torch.where(dt_all > 0, dt_all, self.last_dt)
        rew = c.rew_coverage * dcov - c.rew_time * dt_all - c.rew_fail * fail.float()
        if self.record:
            tt, cc, dd = self.t.cpu().tolist(), self.cov.cpu().tolist(), self.dist.cpu().tolist()
            act = (dt_all > 0).cpu().tolist()
            for e in range(N):
                if act[e]:
                    self.trace[e].append((tt[e], cc[e], dd[e]))
        return rew, self.done.clone(), fail

    def _go_home(self, mask, finished=True):
        """Drive home (time and distance of the trip) and end the episode."""
        if not mask.any():
            return
        ids = torch.nonzero(mask).squeeze(-1)
        known, occ, passable, free, infl, unk_near = self.costmap()
        hr, hc = self._rc(self.home[ids])
        seed, _ = self._seed(passable)
        sub_pass = passable[ids]
        sub_pass[torch.arange(len(ids), device=self.dev), hr, hc] = True
        D = self.distance_field((seed[0][ids], seed[1][ids]), sub_pass, infl[ids])
        d = D[torch.arange(len(ids), device=self.dev), hr, hc] * RES / 1.5  # cost units ~ 1.5 x metres
        straight = torch.hypot(*(self.home[ids] - self.pos[ids]).unbind(-1))
        d = torch.where(torch.isfinite(d), torch.maximum(d, straight), straight * 1.5)
        tt = d / self.cfg.speed + self.cfg.goal_overhead_s
        self.t[ids] += tt
        self.dist[ids] += d
        self.home_time[ids] = tt
        self.pos[ids] = self.home[ids]
        self.done[ids] = True
        self.finished[ids] = finished
