"""Generate the exploration layout sets: training (procedural), held-out (procedural, other seed),
and the real house's two floors (vector plans in the map frame)."""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import layouts as Lg  # noqa: E402

if __name__ == "__main__":
    out = os.path.join(HERE, "data")
    os.makedirs(out, exist_ok=True)
    for name, n, seed in (("layouts_train.npz", 2000, 11), ("layouts_heldout.npz", 200, 12345)):
        p = os.path.join(out, name)
        if not os.path.exists(p):
            t0 = time.time()
            lays = Lg.generate(n, seed, 160)
            np.savez_compressed(p, layouts=np.array(lays, dtype=object))
            print(name, n, f"{time.time() - t0:.0f}s")
    plans = os.environ.get("TWIN_PLANS", "E:/twin/assets/remodel")
    p = os.path.join(out, "layouts_plan.npz")
    lays = Lg.plan_layouts(plans)
    np.savez_compressed(p, layouts=np.array(lays, dtype=object))
    print("layouts_plan.npz", [(l["name"], l["occ"].shape) for l in lays])
