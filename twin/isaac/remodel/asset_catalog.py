"""Catalogue ArchVis residential assets with their real-world bounding boxes (metres) for furnishing.

  python asset_catalog.py --out E:\\twin\\assets\\remodel\\asset_catalog.json
Downloads only the USD layers (not textures) to measure them.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=r'E:\twin\assets\remodel\asset_catalog.json')
ap.add_argument('--per-folder', type=int, default=8)
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})
import omni.client  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

ROOT = get_assets_root_path()
BASE = f'{ROOT}/NVIDIA/Assets/ArchVis/Residential'
FOLDERS = ['Furniture/Beds', 'Furniture/EndTables', 'Furniture/Bookshelves', 'Furniture/Chairs', 'Furniture/Sofas',
           'Furniture/CoffeeTables', 'Furniture/Desks', 'Furniture/Tables', 'Furniture/MediaTables',
           'Furniture/SofaTables', 'Furniture/Misc', 'Furniture/BarStools', 'Plants', 'Decor', 'Lighting',
           'Fixtures', 'Electronics', 'Misc', 'Entertainment']


def list_usd(url, depth=2):
    out = []
    r, entries = omni.client.list(url)
    if r != omni.client.Result.OK:
        return out
    for e in entries:
        p = f'{url}/{e.relative_path}'
        if e.relative_path.endswith(('.usd', '.usda', '.usdc')) and not e.relative_path.startswith('.'):
            out.append(p)
        elif depth > 0 and (e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN) and not e.relative_path.startswith('.'):
            out += list_usd(p, depth - 1)
    return out


cat = []
t0 = time.time()
for folder in FOLDERS:
    urls = list_usd(f'{BASE}/{folder}')[:args.per_folder]
    for u in urls:
        try:
            st = Usd.Stage.Open(u, Usd.Stage.LoadAll)
            prim = st.GetDefaultPrim() or st.GetPseudoRoot()
            mpu = UsdGeom.GetStageMetersPerUnit(st) or 0.01
            up = UsdGeom.GetStageUpAxis(st)
            rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = rng.GetMin(), rng.GetMax()
            size = [(hi[i] - lo[i]) * mpu for i in range(3)]
            cat.append({'folder': folder, 'url': u, 'name': os.path.splitext(os.path.basename(u))[0],
                        'meters_per_unit': mpu, 'up_axis': up, 'size_m': [round(v, 3) for v in size],
                        'min': [round(v * mpu, 3) for v in lo], 'max': [round(v * mpu, 3) for v in hi]})
            print('ASSET', folder, cat[-1]['name'], cat[-1]['size_m'], up, mpu, flush=True)
        except Exception as e:  # noqa: BLE001
            print('FAIL', u, e, flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(cat, open(args.out, 'w'), indent=1)
print('CATALOG_DONE', len(cat), round(time.time() - t0), 's')
app.close()
