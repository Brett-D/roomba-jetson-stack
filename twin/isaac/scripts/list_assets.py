"""List the Isaac Sim asset folders the twin uses (people, animations, props, materials)."""
import json
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({'headless': True})
import omni.client  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

root = get_assets_root_path()
print('ASSET_ROOT', root)
out = {'root': root}


def ls(rel, depth=1):
    res, entries = omni.client.list(f'{root}/{rel}')
    names = [e.relative_path for e in entries] if res == omni.client.Result.OK else [f'<{res}>']
    out[rel] = names
    print('LIST', rel, len(names), names[:60], flush=True)
    return names


for rel in sys.argv[1:] or ['Isaac', 'Isaac/People', 'Isaac/People/Characters', 'Isaac/People/Animations',
                            'Isaac/People/MotionLibrary', 'Isaac/Props', 'Isaac/Environments',
                            'Isaac/Materials', 'NVIDIA', 'NVIDIA/Materials', 'NVIDIA/Assets']:
    ls(rel)
with open(os.path.join(os.environ.get('TWIN', r'E:\twin'), 'work', 'asset_listing.json'), 'w') as f:
    json.dump(out, f, indent=1)
app.close()
