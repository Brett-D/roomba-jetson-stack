"""Export the frontier ranker to ONNX (map, glob, feats, mask -> logits) with recorded test vectors.

  python export_ranker.py --ckpt <run>/ranker_final.pt          # PPO-fine-tuned network
  python export_ranker.py --theta <cem>/theta_mu.npy             # CEM linear scorer (network output 0)

Test vectors are real observations from the exploration env (a few decisions in held-out houses)
with the PyTorch logits, for src/roomba_rs/src/learned/explore.rs (tract must match within 1e-5).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from env2d import CROP, K, N_FEAT, N_GLOB, ExploreCfg, ExploreEnv  # noqa: E402
from model import Ranker, RankerExport  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--theta", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "..", "src", "roomba_rs", "models", "explore_ranker.onnx"))
    ap.add_argument("--vectors", default=os.path.join(HERE, "..", "..", "..", "src", "roomba_rs", "models", "explore_vectors.json"))
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    m = Ranker(theta=np.load(a.theta) if a.theta else None)
    if a.ckpt:
        m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    m.eval()
    ex = RankerExport(m).eval()
    args = (torch.zeros(1, 3, CROP, CROP), torch.zeros(1, N_GLOB), torch.zeros(1, K, N_FEAT), torch.ones(1, K))
    torch.onnx.export(ex, args, a.out, input_names=["map", "glob", "feats", "mask"], output_names=["logits"], opset_version=17, dynamo=False)
    print("wrote", os.path.abspath(a.out), os.path.getsize(a.out), "bytes")
    # recorded inputs from the env
    lays = list(np.load(os.path.join(HERE, "data", "layouts_heldout.npz"), allow_pickle=True)["layouts"])[:a.n]
    cfg = ExploreCfg()
    cfg.train = False
    env = ExploreEnv(lays, a.n, "cuda", cfg, seed=5, size=160)
    env.reset(torch.arange(a.n, device="cuda"), torch.arange(a.n, device="cuda"))
    cases = []
    for step in range(4):
        cand = env.candidates()
        mp, gl, ft, mk = env.observe(cand)
        if step in (1, 3):
            for e in range(a.n):
                if not bool(mk[e].any()):
                    continue
                x = (mp[e:e + 1].cpu(), gl[e:e + 1].cpu(), ft[e:e + 1].cpu(), mk[e:e + 1].float().cpu())
                with torch.no_grad():
                    y = ex(*x)[0].numpy()
                    ref = m(x[0], x[1], x[2], x[3] > 0.5)[0][0].numpy()
                assert np.abs(np.where(ref > -1e8, y - ref, 0)).max() < 1e-4, "export graph differs from Ranker.forward"
                cases.append({"map": x[0].numpy().ravel().tolist(), "glob": x[1].numpy().ravel().tolist(),
                              "feats": x[2].numpy().ravel().tolist(), "mask": x[3].numpy().ravel().tolist(),
                              "logits": [float(v) for v in y]})
        env.step(torch.zeros(a.n, dtype=torch.long, device="cuda"))
    json.dump(cases, open(a.vectors, "w"))
    print("wrote", os.path.abspath(a.vectors), len(cases), "cases")
