"""Generate the stuck-tile sets: training, held-out procedural evaluation (different seed)."""

from __future__ import annotations

import argparse
import collections
import os
import time

import scenarios as S

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1536)
    ap.add_argument("--eval", type=int, default=440)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, n, seed in (("tiles_train.json", a.train, 1), ("tiles_eval.json", a.eval, 1001)):
        t0 = time.time()
        tiles = S.generate(n, seed)
        S.save(tiles, os.path.join(a.out, name))
        print(name, len(tiles), f"{time.time() - t0:.0f}s", dict(collections.Counter(t.kind for t in tiles)))
