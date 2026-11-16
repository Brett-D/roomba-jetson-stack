"""Classical recovery baseline: what the robot does today when it is stuck, in the same env.

Nav2 (navigate_robot_bt.xml) + RPP + the driver reflexes, emulated on the policy's own observation
(local costmap patch, goal) so both controllers see exactly the same information:

  FOLLOW    planner + RPP: pick the free straight heading (0.8 m, 0.2 m clearance on the costmap
            patch) closest to the goal bearing; rotate to heading first if it is > 0.6 rad off
            (rotate_to_heading_angular_vel 0.8), else drive 0.25 m/s with pure-pursuit curvature.
            Fails when no heading is free or there is no progress (< 0.1 m in 3 s).
  RECOVERY  RoundRobin as in the BT: Spin +1.57, Spin -3.14, BackUp 0.10 m at 0.05 m/s,
            ClearEntireCostmap (local) + Wait 2 s; Spin aborts if the footprint is already on a
            lethal cell, BackUp if lethal cells are behind; then FOLLOW again.

The collision monitor, velocity smoother and the driver's bump stop / back-up reflex act underneath
(inside the env), exactly as for the learned policy.
"""

from __future__ import annotations

import math

import torch

PATCH = 24
RES = 0.1
FOLLOW, SPIN_L, SPIN_R, BACKUP, WAIT = 0, 1, 2, 3, 4


class ClassicalRecovery:
    def __init__(self, env):
        self.env = env
        dev = env.device
        N = env.num_envs
        self.mode = torch.zeros(N, dtype=torch.long, device=dev)
        self.rr = torch.zeros(N, dtype=torch.long, device=dev)  # next recovery in the round robin
        self.target = torch.zeros(N, device=dev)  # remaining spin (rad) / backup (m) / wait (s)
        self.prog_t = torch.zeros(N, device=dev)
        self.prog_xy = torch.zeros(N, 2, device=dev)
        idx = torch.arange(PATCH, device=dev, dtype=torch.float32)
        u = (PATCH / 2 - 0.5 - idx) * RES
        U, V = torch.meshgrid(u, u, indexing="ij")
        self.cu, self.cv = U.reshape(-1), V.reshape(-1)  # cell centres (forward, left)
        self.headings = torch.linspace(-math.pi, math.pi, 25, device=dev)[:-1]

    def reset(self, ids):
        self.mode[ids] = FOLLOW
        self.rr[ids] = 0
        self.target[ids] = 0
        self.prog_t[ids] = 0
        p = self.env.robot.data.root_pos_w.torch[ids, :2]
        self.prog_xy[ids] = p

    def _free_headings(self, lethal, gdist):
        # [N, H]: straight segment of length L (<= goal distance) clear by 0.2 m of lethal cells
        L = gdist.clamp(0.3, 0.8).unsqueeze(-1)  # [N, 1]
        ch, sh = torch.cos(self.headings), torch.sin(self.headings)  # [H]
        t = (self.cu.unsqueeze(0) * ch.unsqueeze(-1) + self.cv.unsqueeze(0) * sh.unsqueeze(-1))  # [H, C] along
        n = (-self.cu.unsqueeze(0) * sh.unsqueeze(-1) + self.cv.unsqueeze(0) * ch.unsqueeze(-1)).abs()  # [H, C]
        near = (t.unsqueeze(0) > 0.0) & (t.unsqueeze(0) < L.unsqueeze(-1)) & (n.unsqueeze(0) < 0.2)  # [N, H, C]
        own = (self.cu ** 2 + self.cv ** 2) < 0.15 ** 2  # the robot's own cells don't block a heading
        hit = (near & lethal.unsqueeze(1) & ~own).any(-1)
        return ~hit

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        env = self.env
        dev = env.device
        N = obs.shape[0]
        lethal = obs[:, : PATCH * PATCH] > 0.5
        gd = obs[:, PATCH * PATCH + 16] * 3.0
        gb = torch.atan2(obs[:, PATCH * PATCH + 18], obs[:, PATCH * PATCH + 17])
        dt = env.step_dt
        v = torch.zeros(N, device=dev)
        w = torch.zeros(N, device=dev)
        p = env.robot.data.root_pos_w.torch[:, :2]
        yaw_rate = env.robot.data.root_link_ang_vel_w.torch[:, 2]
        fwd_v = obs[:, PATCH * PATCH + 8] * 0.25

        # ---- FOLLOW
        fol = self.mode == FOLLOW
        free = self._free_headings(lethal, gd)
        diff = torch.atan2(torch.sin(self.headings.unsqueeze(0) - gb.unsqueeze(-1)), torch.cos(self.headings.unsqueeze(0) - gb.unsqueeze(-1))).abs()
        diff = torch.where(free, diff, torch.full_like(diff, 99.0))
        best = diff.argmin(-1)
        ok = diff.gather(1, best.unsqueeze(-1)).squeeze(-1) < 90.0
        h = self.headings[best]
        rot = h.abs() > 0.6
        fv = torch.where(rot, 0.0, 0.25)
        fw = torch.where(rot, 0.8 * torch.sign(h), (2 * 0.25 * torch.sin(h) / 0.6).clamp(-0.8, 0.8))
        # progress checker
        self.prog_t = torch.where(fol, self.prog_t + dt, self.prog_t)
        moved = torch.norm(p - self.prog_xy, dim=-1) > 0.1
        self.prog_xy = torch.where((moved | ~fol).unsqueeze(-1), p, self.prog_xy)
        self.prog_t = torch.where(moved | ~fol, torch.zeros_like(self.prog_t), self.prog_t)
        stalled = self.prog_t > 3.0
        fail = fol & (~ok | stalled)
        v = torch.where(fol & ~fail, fv, v)
        w = torch.where(fol & ~fail, fw, w)
        # start the next recovery of the round robin
        nxt = self.rr % 4
        start_mode = torch.tensor([SPIN_L, SPIN_R, BACKUP, WAIT], device=dev)[nxt]
        start_tgt = torch.tensor([1.57, 3.14, 0.10, 2.0], device=dev)[nxt]
        self.mode = torch.where(fail, start_mode, self.mode)
        self.target = torch.where(fail, start_tgt, self.target)
        self.rr = torch.where(fail, self.rr + 1, self.rr)
        self.prog_t = torch.where(fail, torch.zeros_like(self.prog_t), self.prog_t)
        if (fail & (start_mode == WAIT)).any():
            ids = (fail & (start_mode == WAIT)).nonzero().squeeze(-1)
            env.mark_t[ids] = -1e9  # ClearEntireCostmap (local); keep-out zones stay

        # ---- spins (abort if the footprint is on a lethal cell)
        on_lethal = (lethal & ((self.cu ** 2 + self.cv ** 2) < 0.17 ** 2)).any(-1)
        spin = (self.mode == SPIN_L) | (self.mode == SPIN_R)
        sgn = torch.where(self.mode == SPIN_L, 1.0, -1.0)
        self.target = torch.where(spin, self.target - yaw_rate.abs() * dt, self.target)
        end_spin = spin & ((self.target <= 0) | on_lethal)
        w = torch.where(spin & ~end_spin, 0.8 * sgn, w)
        # ---- backup (abort if lethal cells behind)
        bk = self.mode == BACKUP
        behind = (lethal & (self.cu < -0.05) & (self.cu > -0.32) & (self.cv.abs() < 0.17)).any(-1)
        self.target = torch.where(bk, self.target - fwd_v.abs() * dt, self.target)
        end_bk = bk & ((self.target <= 0) | behind)
        v = torch.where(bk & ~end_bk, -0.05, v)
        # ---- wait
        wt = self.mode == WAIT
        self.target = torch.where(wt, self.target - dt, self.target)
        end_wt = wt & (self.target <= 0)
        self.mode = torch.where(end_spin | end_bk | end_wt, torch.full_like(self.mode, FOLLOW), self.mode)

        a = torch.stack([torch.where(v >= 0, v / 0.25, v / 0.15), w / 0.8], -1)
        return a.clamp(-1, 1)
