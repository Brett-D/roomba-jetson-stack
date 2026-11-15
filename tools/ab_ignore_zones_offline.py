#!/usr/bin/env python3
"""Exact A/B of ignore_zones: feeds the same event sequence to the Python node (roomba_perception/
ignore_zones.py, callbacks called directly, time.time() scripted) and to the Rust core
(examples/zones_replay.rs) and compares everything published (zones JSON text, keep-out mask grids,
marker arrays, filter info; stamps excluded), the log lines, and the zones files written (byte for
byte, plus the .bak moved aside by clear_on_start).

Events: TF from a recording (/tf, /tf_static) and clicked points in several frames (map, odom,
base_link, camera frames, unknown and empty frame ids) drawing and closing polygons; text commands
(close/undo/cancel/clear, odd casing, unknown); web-app JSON ops (add/delete/rename/clear with valid
and invalid names, kinds, margins, ttl, points as ints/strings/bad shapes, too few/many points,
out of range, duplicates, unknown ops, broken JSON); explorer bump marks (keep-out, margin 0.15,
ttl 600 s) and cliff marks; marker and expiry timer ticks with time jumps so marks expire; restarts
with the saved file, a hand-edited file (ints, extra keys, unicode names, temporary zones), a corrupt
file, a missing file, clear_on_start; other parameters (map_frame odom, margin, resolution).
  tools/ab_ignore_zones_offline.py VIO_BAG [--fuzz N] [--seed S] [--rust BIN] [--work DIR]
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess

import numpy as np
import rclpy
import rclpy.node
import rclpy.publisher
from geometry_msgs.msg import PointStamped
from rclpy.parameter import Parameter
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

import roomba_perception.ignore_zones as I

ap = argparse.ArgumentParser()
ap.add_argument('bag')
ap.add_argument('--fuzz', type=int, default=3000)
ap.add_argument('--seed', type=int, default=3)
ap.add_argument('--rust', default=os.path.expanduser('~/rs_lanes/tfmap_target/release/examples/zones_replay'))
ap.add_argument('--work', default='/data/rs_lanes/tfmap/ab_zones')
args = ap.parse_args()
rng = random.Random(args.seed)
S = 1_000_000_000
shutil.rmtree(args.work, ignore_errors=True)
for side in ('py', 'rs'):
    os.makedirs(os.path.join(args.work, side))

# ---- Python node harness ----
now = [1794730000.0]
I.time.time = lambda: now[0]
captured, logs = [], []
rclpy.publisher.Publisher.publish = lambda self, msg: captured.append((self.topic_name, msg))


class Log:
    def info(self, m, *a, **k): logs.append(['info', m])
    def warn(self, m, *a, **k): logs.append(['warn', m])
    warning = warn
    def error(self, m, *a, **k): logs.append(['error', m])


rclpy.node.Node.get_logger = lambda self: Log()
overrides = []
_init = rclpy.node.Node.__init__


def _patched_init(self, name, *a, **kw):
    kw.setdefault('parameter_overrides', list(overrides))
    _init(self, name, *a, **kw)


rclpy.node.Node.__init__ = _patched_init
rclpy.init()
node = None
f32 = lambda v: float(np.float32(v))  # noqa: E731 - float32 message fields, as received


def marker_dict(m):
    p, o, s, c = m.pose.position, m.pose.orientation, m.scale, m.color
    return {'ns': m.ns, 'id': m.id, 'type': m.type, 'action': m.action, 'frame': m.header.frame_id,
            'position': [p.x, p.y, p.z], 'orientation': [o.x, o.y, o.z, o.w], 'scale': [s.x, s.y, s.z],
            'color': [f32(c.r), f32(c.g), f32(c.b), f32(c.a)], 'points': [[q.x, q.y, q.z] for q in m.points],
            'text': m.text}


def py_record():
    rec = {'zones': [], 'mask': [], 'markers': [], 'filter_info': [], 'logs': [list(x) for x in logs]}
    for topic, msg in captured:
        if topic == '/perception/ignore_zones':
            rec['zones'].append(msg.data)
        elif topic == '/keepout_filter_mask':
            o = msg.info.origin
            rec['mask'].append({'frame': msg.header.frame_id, 'resolution': f32(msg.info.resolution),
                                'width': msg.info.width, 'height': msg.info.height,
                                'origin': [o.position.x, o.position.y, o.position.z, o.orientation.w],
                                'data': list(msg.data)})
        elif topic == '/perception/ignore_zones/markers':
            rec['markers'].append([marker_dict(m) for m in msg.markers])
        elif topic == '/costmap_filter_info':
            rec['filter_info'].append({'type': msg.type, 'topic': msg.filter_mask_topic, 'base': f32(msg.base),
                                       'multiplier': f32(msg.multiplier), 'frame': msg.header.frame_id})
    del captured[:], logs[:]
    return rec


events, expected, files = [], [], []


def snapshot(name):
    base = os.path.join(args.work, 'py', name)
    files.append([open(p, 'rb').read().decode() if os.path.exists(p) else None for p in (base, base + '.bak')])


def emit(ev, apply):
    apply()
    events.append(ev)
    expected.append(py_record())
    snapshot(current[0])


current = ['']


def start(name, params=None):
    global node
    params = dict(params or {})
    current[0] = name
    p = dict(dict(map_frame='map', close_distance_m=0.25, keepout_margin_m=0.2, keepout_resolution_m=0.05,
                  clear_on_start=False), **params)

    def apply():
        global node
        if node is not None:
            node.destroy_node()
        overrides[:] = [Parameter(k, value=v) for k, v in dict(p, zones_file=os.path.join(args.work, 'py', name)).items()]
        node = I.IgnoreZones()
    ev = {'ev': 'start', 'params': dict(p, zones_file=os.path.join(args.work, 'rs', name))}
    emit(ev, apply)


def write_file(name, content):
    for side in ('py', 'rs'):
        path = os.path.join(args.work, side, name)
        if content is None:
            if os.path.exists(path):
                os.remove(path)
        else:
            with open(path, 'w') as f:
                f.write(content)


def tf(t, is_static):
    s, q, tr = t.header.stamp, t.transform.rotation, t.transform.translation

    def apply():
        (node.tf_buffer.set_transform_static if is_static else node.tf_buffer.set_transform)(t, 'default_authority')
    emit({'ev': 'tf', 'static': is_static, 'parent': t.header.frame_id, 'child': t.child_frame_id,
          'stamp_ns': s.sec * S + s.nanosec, 't': [tr.x, tr.y, tr.z], 'q': [q.x, q.y, q.z, q.w]}, apply)


def click(frame, x, y, z=0.0):
    msg = PointStamped()
    msg.header.frame_id = frame
    msg.point.x, msg.point.y, msg.point.z = float(x), float(y), float(z)
    emit({'ev': 'click', 'frame': frame, 'p': [msg.point.x, msg.point.y, msg.point.z]}, lambda: node._on_click(msg))


def cmd(text):
    emit({'ev': 'cmd', 'text': text, 'now': now[0]}, lambda: node._on_command(String(data=text)))


def jcmd(**kw):
    cmd(json.dumps(kw))


def tick(dt=1.0):
    now[0] += dt
    emit({'ev': 'markers'}, node._publish_markers)
    if rng.random() < 0.5:
        emit({'ev': 'expire', 'now': now[0]}, node._expire)


# ---- TF from the recording ----
r = SequentialReader()
r.open(StorageOptions(uri=args.bag, storage_id='mcap'), ConverterOptions('', ''))
r.set_filter(StorageFilter(topics=['/tf', '/tf_static']))
statics, dynamic = [], []
while r.has_next() and len(dynamic) < 30000:
    topic, data, _ = r.read_next()
    for t in deserialize_message(data, TFMessage).transforms:
        (statics if topic == '/tf_static' else dynamic).append(t)
dynamic = [t for t in dynamic if t.header.stamp.sec > 1000]   # skip RTAB-Map's pre-clock stamp
tf_pos = [0]


def feed_tf(n):
    for t in dynamic[tf_pos[0]:tf_pos[0] + n]:
        tf(t, False)
    tf_pos[0] = (tf_pos[0] + n) % max(1, len(dynamic) - n)


FRAMES = ['map', 'odom', 'base_link', 'camera_link', 'camera_left_optical_frame', 'camera_imu_frame', 'nowhere', '']


def rand_click():
    click(rng.choice(FRAMES), rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-0.5, 0.5))


def bump_mark(k):
    x, y, a = rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-math.pi, math.pi)
    pts = [[round(x + u * math.cos(a) - v * math.sin(a), 3), round(y + u * math.sin(a) + v * math.cos(a), 3)]
           for u, v in ((0.0, 0.25), (0.12, 0.25), (0.12, -0.25), (0.0, -0.25))]
    jcmd(op='add', kind='keepout', name=f'bump_123456_{k}', points=pts, margin=0.15, ttl_s=600.0)


def cliff_mark():
    x, y = rng.uniform(-4, 4), rng.uniform(-4, 4)
    jcmd(op='add', kind='keepout', name=f'cliff_{rng.randint(100000, 235959)}',
         points=[[round(x + dx, 3), round(y + dy, 3)] for dx, dy in ((0.175, 0.175), (-0.175, 0.175), (-0.175, -0.175), (0.175, -0.175))])


def rand_poly(n=None):
    n = n or rng.randint(3, 8)
    cx, cy, rad = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.2, 2.0)
    return [[cx + rad * math.cos(2 * math.pi * k / n + rng.uniform(-0.3, 0.3)) * rng.uniform(0.6, 1.0),
             cy + rad * math.sin(2 * math.pi * k / n) * rng.uniform(0.6, 1.0)] for k in range(n)]


ODD_COMMANDS = [
    '{bad json', '{"op": "add"', '{}', '{"op": 5}', '{"op": null}', '{"op": "explode"}', 'close', ' Close ', 'UNDO',
    'cancel', 'nonsense', '', '   ', '{"op": "add", "points": [[0,0],[1,0]]}', '{"op": "add", "points": "abc"}',
    '{"op": "add", "points": 5}', '{"op": "add", "points": null}', '{"op": "add", "points": [[0,0],[1,0],[1]]}',
    '{"op": "add", "points": [[0,0],[1,0],{"x":1}]}', '{"op": "add", "points": [[0,0],[1,0],5]}',
    '{"op": "add", "points": ["12","34","56"]}', '{"op": "add", "points": [[0,0],[1,0],["1.5","2_0"]]}',
    '{"op": "add", "points": [[0,0],[1,0],["x","1"]]}', '{"op": "add", "points": [[0,0],[1e3,0],[1,1]]}',
    '{"op": "add", "points": [[0,0],[1,0],["nan",1]]}', '{"op": "add", "points": [[0,0],[1,0],[true,null]]}',
    '{"op": "add", "kind": "bogus", "points": [[0,0],[1,0],[1,1]]}', '{"op": "add", "kind": 5, "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "kind": "", "points": [[0,0],[1,0],[1,1]]}', '{"op": "add", "kind": null, "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "kind": "keepout", "margin": 1.5, "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "kind": "keepout", "margin": "0.3", "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "kind": "keepout", "margin": [1], "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "kind": "keepout", "margin": null, "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "kind": "keepout", "ttl_s": 0, "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "kind": "keepout", "ttl_s": 86401, "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "kind": "keepout", "ttl_s": "30", "points": [[0.1,0],[1,0],[1,1]]}',
    '{"op": "add", "name": "  ", "points": [[0,0],[1,0],[1,1]]}', '{"op": "add", "name": 0, "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "name": 12.5, "points": [[0,0],[1,0],[1,1]]}', '{"op": "add", "name": true, "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "name": ["a b", 1], "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "name": "K\\u00fcche / TV!! \\u2603 \\u00e9t\\u00e9 _-x", "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "add", "name": "' + 'n' * 60 + '", "points": [[0,0],[1,0],[1,1]]}',
    '{"op": "delete", "name": "nope"}', '{"op": "delete"}', '{"op": "delete", "name": 5}',
    '{"op": "rename", "name": "nope", "new_name": "x"}', '{"op": "rename", "name": "zone_1", "new_name": "!!!"}',
    '{"op": "rename", "name": "zone_1", "new_name": "renamed one"}', '{"op": "rename", "new_name": "zz"}',
    '{"op": "add", "points": [[0,0],[1,0],[1,1]], "name": "dup"}', '{"op": "add", "points": [[2,0],[3,0],[3,1]], "name": "dup"}',
    '{"op": "add", "points": [[0,0],[1,0],[1,1]], "extra": {"a": [1, 2]}}',
    '{"op": "add", "points": ' + json.dumps([[k * 0.01, (k % 2) * 0.5] for k in range(65)]) + '}',
    '{"op": "add", "points": ' + json.dumps([[k * 0.01, (k % 2) * 0.5] for k in range(64)]) + '}',
]


def random_event(k):
    r = rng.random()
    if r < 0.18:
        feed_tf(rng.randint(1, 30))
    elif r < 0.38:
        rand_click()
    elif r < 0.45:
        bump_mark(k)
    elif r < 0.48:
        cliff_mark()
    elif r < 0.6:
        kind = rng.choice([None, 'exclusion', 'keepout', 'keepout'])
        kw = dict(op='add', points=[[round(x, rng.choice([1, 3, 6])), round(y, 4)] for x, y in rand_poly()])
        if kind:
            kw['kind'] = kind
        if rng.random() < 0.5:
            kw['name'] = rng.choice(['tv', 'sofa', 'zone_1', 'keepout_2', 'Hall way', None, 'dup'])
        if rng.random() < 0.3:
            kw['margin'] = rng.choice([0.0, 0.05, 0.1234567, 0.5, 1.0, 1])
        if rng.random() < 0.2:
            kw['ttl_s'] = rng.choice([1.0, 5.5, 60, 600.0])
        jcmd(**kw)
    elif r < 0.66:
        names = [z['name'] for z in node.zones] or ['none']
        jcmd(op='delete', name=rng.choice(names))
    elif r < 0.69:
        names = [z['name'] for z in node.zones] or ['none']
        jcmd(op='rename', name=rng.choice(names), new_name=rng.choice(['tv', 'renamed', 'a-b_c 1', '#', 'zone_9']))
    elif r < 0.78:
        cmd(rng.choice(ODD_COMMANDS))
    elif r < 0.81:
        cmd(rng.choice(['undo', 'close', 'cancel']))
    elif r < 0.82 and len(node.zones) > 20:
        cmd(rng.choice(['clear', '{"op": "clear"}']))
    else:
        tick(rng.choice([0.5, 1.0, 2.0, 2.0, 30.0, 300.0, 601.0]))


# ---- scenarios ----
start('s1.zones.json')                       # no file yet
for t in statics:
    tf(t, True)
feed_tf(200)
for frame in ('map', 'odom', 'base_link', 'camera_left_optical_frame'):
    for x, y in ((0, 0), (1.0, 0.1), (1.2, 1.1), (0.1, 1.0)):
        click(frame, x, y, 0.3)
    click(frame, 0.05, 0.05)                  # near the first corner: closes (or a 5th corner off the map frame)
    feed_tf(20)
    tick()
click('nowhere', 1, 1)
click('', 1, 1)
click('map', 5, 5)
click('map', 6, 5)
cmd('close')                                 # too few corners
cmd('undo')
cmd('undo')
cmd('undo')                                  # removes the last zone
click('map', 5, 5)
click('map', 6, 5)
click('map', 6, 6)
cmd(' Close ')
cmd('cancel')
for k in range(5):
    bump_mark(k)
    tick(2.0)
cliff_mark()
for text in ODD_COMMANDS:
    cmd(text)
tick(601.0)                                  # the bump marks expire
tick(2.0)
start('s1.zones.json')                       # restart: reload the saved zones
tick()
cmd('clear')
tick()

hand_edited = json.dumps({'frame': 'map', 'note': 'edited by hand', 'zones': [
    {'name': 'tv', 'points': [[0, 0], [2, 0], [2, 1], [0, 1]], 'kind': 'keepout', 'margin': 0, 'color': 'red'},
    {'name': 'K\u00fcche \u2603', 'points': [[1.5, -2.25], [3.125, -2], [2, -1]], 'kind': 'exclusion'},
    {'points': [[4, 4], [5, 4], [5, 5.5]], 'name': 'reordered', 'kind': 'keepout', 'margin': None},
    {'name': 'temp', 'points': [[-1, -1], [-2, -1], [-2, -2]], 'kind': 'keepout', 'expires': now[0] + 5},
    {'name': 'nokind', 'points': [[7, 7], [8, 7], [8, 8], [7.5, 9]]},
]}, indent=2)
write_file('s2.zones.json', hand_edited)
start('s2.zones.json')
tick(2.0)
tick(6.0)                                    # 'temp' expires
jcmd(op='rename', name='tv', new_name='television')
jcmd(op='add', kind='keepout', points=[[0.5, 0.5], [1, 0.5], [1, 1]])
tick()

write_file('s3.zones.json', '{"frame": "map", "zones": [ broken')
start('s3.zones.json')                       # corrupt file: error, no zones
cmd('{"op": "add", "points": [[0,0],[1,0],[1,1]]}')

write_file('s4.zones.json', hand_edited)
start('s4.zones.json', {'clear_on_start': True})   # new mapping session: file moved to .bak
tick()
write_file('s5.zones.json', None)
start('s5.zones.json', {'clear_on_start': True})   # nothing to move
tick()

start('s6.zones.json', {'map_frame': 'odom', 'keepout_margin_m': 0.35, 'keepout_resolution_m': 0.1,
                        'close_distance_m': 0.5})
for t in statics:
    tf(t, True)
feed_tf(100)
for x, y in ((0, 0), (1.0, 0.1), (1.2, 1.1), (0.3, 0.2)):
    click('base_link', x, y)
for k in range(3):
    bump_mark(k)
tick(2.0)

start('s7.zones.json')
for t in statics:
    tf(t, True)
for k in range(args.fuzz):
    random_event(k)

# ---- run Rust and compare ----
path = os.path.join(args.work, 'events.jsonl')
with open(path, 'w') as f:
    for ev in events:
        f.write(json.dumps(ev) + '\n')
lines = subprocess.run([args.rust, path], capture_output=True, text=True, check=True).stdout.splitlines()
assert len(lines) == len(expected), (len(lines), len(expected))

py_dir, rs_dir = os.path.join(args.work, 'py'), os.path.join(args.work, 'rs')
counts = {'events': len(events), 'zones_msgs': 0, 'masks': 0, 'mask_cells': 0, 'marker_arrays': 0, 'markers': 0,
          'logs': 0, 'file_states': 0}
diffs = {'zones': 0, 'mask': 0, 'markers': 0, 'filter_info': 0, 'logs': 0, 'files': 0}
log_diffs, shown = [], 0
for k, (ev, exp, line) in enumerate(zip(events, expected, lines)):
    got = json.loads(line)
    counts['zones_msgs'] += len(exp['zones'])
    counts['masks'] += len(exp['mask'])
    counts['mask_cells'] += sum(len(m['data']) for m in exp['mask'])
    counts['marker_arrays'] += len(exp['markers'])
    counts['markers'] += sum(len(a) for a in exp['markers'])
    counts['logs'] += len(exp['logs'])
    for key in ('zones', 'mask', 'markers', 'filter_info'):
        if exp[key] != got[key]:
            diffs[key] += 1
            if shown < 6:
                shown += 1
                print(f'DIFF {key} at event {k} {json.dumps(ev)[:200]}\n  py {str(exp[key])[:400]}\n  rs {str(got[key])[:400]}')
    py_logs = [[lvl, text.replace(py_dir, '<dir>')] for lvl, text in exp['logs']]
    rs_logs = [[lvl, text.replace(rs_dir, '<dir>')] for lvl, text in got['logs']]
    if py_logs != rs_logs:
        diffs['logs'] += 1
        log_diffs.append((ev, py_logs, rs_logs))
    counts['file_states'] += 1
    if files[k] != got['file']:
        diffs['files'] += 1
        if shown < 6:
            shown += 1
            print(f'DIFF file at event {k}: py {str(files[k])[:300]}\n  rs {str(got["file"])[:300]}')
print('compared:', ', '.join(f'{k} {v}' for k, v in counts.items()))
print('differences:', ', '.join(f'{k} {v}' for k, v in diffs.items()))
kinds = {}
for ev, p, r in log_diffs:
    key = 'json syntax' if ev.get('text', '').startswith('{') and any('Expecting' in t or 'Unterminated' in t for _, t in p) \
        else 'corrupt file' if any('Could not read' in t for _, t in p) else 'other'
    kinds[key] = kinds.get(key, 0) + 1
    if key == 'other' and kinds[key] <= 5:
        print('LOG DIFF', json.dumps(ev)[:200], '\n  py', p, '\n  rs', r)
print('log differences by cause:', kinds)
node.destroy_node()
rclpy.shutdown()
shutil.rmtree(args.work, ignore_errors=True)
