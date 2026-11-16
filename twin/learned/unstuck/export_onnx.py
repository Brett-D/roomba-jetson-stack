"""Export the unstuck actor (RSL-RL checkpoint) to ONNX: obs[1, 596] -> action[1, 2] in [-1, 1].

The graph includes the observation normaliser and the clamp, so the robot side only builds the
observation (unstuck_env.py layout) and maps the action: v = a0 * 0.25 if a0 > 0 else a0 * 0.15,
w = a1 * 0.8. Also writes recorded test vectors (inputs + PyTorch outputs) for the Rust tests.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn


class UnstuckActor(nn.Module):
    def __init__(self, sd: dict):
        super().__init__()
        self.register_buffer("mean", sd["obs_normalizer._mean"].reshape(1, -1).float())
        self.register_buffer("std", sd["obs_normalizer._std"].reshape(1, -1).float())
        dims = [sd["mlp.0.weight"].shape[1]]
        i = 0
        layers = []
        while f"mlp.{i}.weight" in sd:
            w = sd[f"mlp.{i}.weight"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(sd[f"mlp.{i}.bias"])
            layers.append(lin)
            if f"mlp.{i + 2}.weight" in sd:
                layers.append(nn.ELU())
            dims.append(w.shape[0])
            i += 2
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs):
        x = (obs - self.mean) / (self.std + 1e-2)
        return torch.clamp(self.mlp(x), -1.0, 1.0)


def load_actor(ckpt: str) -> UnstuckActor:
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    return UnstuckActor(d["actor_state_dict"]).eval()


def export(ckpt: str, out: str, vectors: str | None, samples: np.ndarray | None):
    m = load_actor(ckpt)
    x = torch.zeros(1, m.mean.shape[1])
    torch.onnx.export(m, (x,), out, input_names=["obs"], output_names=["action"], opset_version=17, dynamo=False)
    print("wrote", out, os.path.getsize(out), "bytes")
    if vectors:
        if samples is None:
            rng = np.random.default_rng(0)
            samples = rng.normal(0, 1, (16, m.mean.shape[1])).astype(np.float32)
            samples[:, :576] = (rng.random((16, 576)) < 0.1).astype(np.float32)
        with torch.no_grad():
            y = m(torch.from_numpy(samples.astype(np.float32))).numpy()
        with open(vectors, "w") as f:
            json.dump({"inputs": samples.astype(np.float32).tolist(), "outputs": y.astype(np.float32).tolist()}, f)
        print("wrote", vectors)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="unstuck_policy.onnx")
    ap.add_argument("--vectors", default="")
    ap.add_argument("--samples", default="", help=".npy of recorded observations to use as test vectors")
    a = ap.parse_args()
    s = np.load(a.samples) if a.samples else None
    export(a.ckpt, a.out, a.vectors or None, s)
