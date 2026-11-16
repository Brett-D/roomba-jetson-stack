"""Frontier ranker: scores the explorer's candidate frontier goals.

Inputs (all computable by the explorer on the robot, see src/roomba_rs/src/learned/explore.rs):
  map   [B, 3, 64, 64]  robot-centred, map-aligned crop of the global costmap at 0.2 m:
                        fraction free / occupied / unknown per cell
  glob  [B, 10]         battery, time, heading, home offset, #candidates, failures, mapped area, ...
  feats [B, K, 16]      per candidate: offset, distance, path cost, frontier size, unknown area
                        around it, turn, distance to home, failures / visits nearby, rule score, ...
  mask  [B, K]          valid candidates
Output: logits [B, K] (invalid = -1e9).

logits = prior_scale * (theta . phi(feats, glob)) + net(map, glob, feats). theta is a fixed linear
scorer (the explorer's rule: -rule_score, or the CEM-optimised one, train_cem.py); the network
(CNN over the map crop + candidate MLP, zero-initialised output) learns corrections with PPO.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from env2d import N_FEAT, N_GLOB

SCORE_FEAT = 11
HOME_FEAT = 8
N_PHI = N_FEAT + 2


def phi(feats: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
    """Linear-scorer features: the candidate features plus battery x / (1 - battery) x home distance."""
    b = glob[:, 0:1, None].expand(-1, feats.shape[1], 1)
    hd = feats[..., HOME_FEAT:HOME_FEAT + 1]
    return torch.cat([feats, b * hd, (1 - b) * hd], -1)


def rule_theta() -> np.ndarray:
    th = np.zeros(N_PHI, np.float32)
    th[SCORE_FEAT] = -1.0
    return th


class Ranker(nn.Module):
    def __init__(self, theta=None, prior_scale=30.0, n_glob=N_GLOB, n_feat=N_FEAT):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.map_fc = nn.Linear(32 * 4 * 4, 128)
        self.glob_fc = nn.Linear(n_glob, 64)
        self.ctx = nn.Linear(192, 128)
        self.cand1 = nn.Linear(n_feat, 64)
        self.cand2 = nn.Linear(64, 64)
        self.score1 = nn.Linear(192, 128)
        self.score2 = nn.Linear(128, 1)
        self.value1 = nn.Linear(192, 128)
        self.value2 = nn.Linear(128, 1)
        th = rule_theta() if theta is None else np.asarray(theta, np.float32)
        self.register_buffer("theta", torch.tensor(th))
        self.prior_scale = nn.Parameter(torch.tensor(float(prior_scale)))
        nn.init.zeros_(self.score2.weight)
        nn.init.zeros_(self.score2.bias)

    def _ctx(self, map_, glob):
        m = torch.relu(self.map_fc(self.cnn(map_).flatten(1)))
        g = torch.relu(self.glob_fc(glob))
        return torch.relu(self.ctx(torch.cat([m, g], -1)))

    def forward(self, map_, glob, feats, mask):
        ctx = self._ctx(map_, glob)
        ce = torch.relu(self.cand2(torch.relu(self.cand1(feats))))
        K = feats.shape[1]
        h = torch.relu(self.score1(torch.cat([ce, ctx.unsqueeze(1).expand(-1, K, -1)], -1)))
        prior = (phi(feats, glob) * self.theta).sum(-1)
        logits = self.score2(h).squeeze(-1) + self.prior_scale * prior
        logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
        mf = mask.float().unsqueeze(-1)
        pooled = (ce * mf).sum(1) / mf.sum(1).clamp(min=1.0)
        value = self.value2(torch.relu(self.value1(torch.cat([ctx, pooled], -1)))).squeeze(-1)
        return logits, value


class RankerExport(nn.Module):
    """ONNX graph: logits only (mask as float 0/1, so the robot side passes plain f32). Same maths as
    Ranker.forward, written without concatenating broadcast tensors (tract's optimiser mis-splits
    concat + matmul over an expanded operand)."""

    def __init__(self, m: Ranker):
        super().__init__()
        self.m = m

    def forward(self, map_, glob, feats, mask):
        m = self.m
        ctx = m._ctx(map_, glob)  # [B, 128]
        ce = torch.relu(m.cand2(torch.relu(m.cand1(feats))))  # [B, K, 64]
        w = m.score1.weight  # [128, 64 + 128]
        h = torch.relu(torch.matmul(ce, w[:, :64].t()) + (torch.matmul(ctx, w[:, 64:].t()) + m.score1.bias).unsqueeze(1))
        th = m.theta
        hd = feats[..., HOME_FEAT]
        b = glob[:, 0:1]
        prior = (feats * th[:N_FEAT]).sum(-1) + hd * (th[N_FEAT] * b + th[N_FEAT + 1] * (1.0 - b))
        logits = m.score2(h).squeeze(-1) + m.prior_scale * prior
        return torch.where(mask > 0.5, logits, torch.full_like(logits, -1e9))
