#!/usr/bin/env python3
"""Load for tools/ab_web_cpu.sh. Publishes what RTAB-Map, the gap filler and Nav2 publish on the
robot (map and filled map at 1 Hz with a growing map, map path 1 Hz, local costmap 2 Hz, plan 1 Hz,
map -> odom TF 20 Hz) and plays the web page against each server: one websocket (state at 10 Hz),
one MJPEG stream and the fetches app.js makes whenever a version changes (map.json + map.png, path,
plan, obstacles, zones). Prints per-server traffic counts at the end.
  ab_web_load.py --ports 18081 18082 --duration S      (password in $AB_PASSWORD)
"""
import argparse
import asyncio
import json
import math
import os
import re
import threading
import time

import aiohttp
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from tf2_msgs.msg import TFMessage

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Load(Node):
    def __init__(self):
        super().__init__('ab_web_load')
        self.pub_map = self.create_publisher(OccupancyGrid, '/rtabmap/map', LATCHED)
        self.pub_filled = self.create_publisher(OccupancyGrid, '/rtabmap/map_filled', LATCHED)
        self.pub_path = self.create_publisher(Path, '/rtabmap/mapPath', 1)
        self.pub_plan = self.create_publisher(Path, '/plan', 1)
        self.pub_costmap = self.create_publisher(OccupancyGrid, '/local_costmap/costmap', 1)
        self.pub_tf = self.create_publisher(TFMessage, '/tf', 100)
        self.rng = np.random.default_rng(1)
        self.grid = np.full((520, 640), -1, np.int8)
        self.t = 0
        self.create_timer(1.0, self.slow)
        self.create_timer(0.5, self.costmap)
        self.create_timer(0.05, self.tf)

    def occupancy(self, frame, data, res, origin):
        g = OccupancyGrid()
        g.header.frame_id = frame
        g.header.stamp = self.get_clock().now().to_msg()
        g.info.resolution = res
        g.info.height, g.info.width = data.shape
        g.info.origin.position.x, g.info.origin.position.y = origin
        g.info.origin.orientation.w = 1.0
        g.data = data.ravel().tolist()
        return g

    def path(self, n, scale):
        p = Path()
        p.header.frame_id = 'map'
        for i in range(n):
            ps = PoseStamped()
            ps.pose.position.x = math.cos(i / 50) * scale * (1 + i / 1000)
            ps.pose.position.y = math.sin(i / 50) * scale * (1 + i / 1000)
            p.poses.append(ps)
        return p

    def slow(self):
        # the map grows: a new explored room-sized patch every second
        self.t += 1
        h, w = self.grid.shape
        cx, cy = int(w / 2 + math.cos(self.t / 9) * w * 0.35), int(h / 2 + math.sin(self.t / 7) * h * 0.35)
        x0, y0 = max(0, cx - 40), max(0, cy - 30)
        patch = self.grid[y0:cy + 30, x0:cx + 40]
        patch[patch < 0] = 0
        walls = self.rng.random(patch.shape) < 0.03
        patch[walls] = 100
        raw = self.occupancy('map', self.grid, 0.05, (-16.0, -13.0))
        self.pub_map.publish(raw)
        self.pub_filled.publish(raw)
        self.pub_path.publish(self.path(800 + self.t, 3.0))
        self.pub_plan.publish(self.path(150, 1.0))

    def costmap(self):
        c = np.zeros((60, 60), np.int8)
        c[self.rng.random((60, 60)) < 0.08] = 100
        self.pub_costmap.publish(self.occupancy('odom', c, 0.05, (-1.5, -1.5)))

    def tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = 'map', 'odom'
        t.transform.translation.x, t.transform.rotation.w = 0.1, 1.0
        self.pub_tf.publish(TFMessage(transforms=[t]))


async def page(port, password, stop, stats):
    async with aiohttp.ClientSession() as s:
        async with s.post(f'http://127.0.0.1:{port}/login', data={'password': password}, allow_redirects=False) as r:
            token = re.search(r'roomba_session=([^;]+)', r.headers['Set-Cookie']).group(1)
        hdr = {'Cookie': f'roomba_session={token}', 'X-Requested-With': 'roomba-web'}

        async def stream():
            async with s.get(f'http://127.0.0.1:{port}/stream.mjpg', headers=hdr) as resp:
                while not stop.is_set():
                    chunk = await resp.content.readany()
                    if not chunk:
                        break
                    stats['stream_bytes'] += len(chunk)
                    stats['frames'] += chunk.count(b'--frame\r\n')

        async def fetch(path):
            async with s.get(f'http://127.0.0.1:{port}{path}', headers=hdr) as r:
                await r.read()
                stats['fetches'] += 1

        async def ws():
            seen = {}
            async with s.ws_connect(f'http://127.0.0.1:{port}/ws', headers=hdr) as w:
                async for msg in w:
                    if stop.is_set():
                        break
                    d = json.loads(msg.data)
                    stats['ws'] += 1
                    for key, paths in (('map_version', ('/api/map.json', f'/api/map.png?v={d["map_version"]}')),
                                       ('path_version', ('/api/path',)), ('plan_version', ('/api/plan',)),
                                       ('obstacles_version', ('/api/obstacles',)), ('zones_version', ('/api/zones',))):
                        if seen.get(key) != d[key]:
                            seen[key] = d[key]
                            for p in paths:
                                await fetch(p)
        await asyncio.gather(stream(), ws(), return_exceptions=True)


async def main_async(args):
    stop = asyncio.Event()
    stats = {p: {'ws': 0, 'frames': 0, 'stream_bytes': 0, 'fetches': 0} for p in args.ports}
    tasks = [asyncio.create_task(page(p, os.environ['AB_PASSWORD'], stop, stats[p])) for p in args.ports]
    await asyncio.sleep(args.duration)
    stop.set()
    await asyncio.sleep(1.0)
    for t in tasks:
        t.cancel()
    print(json.dumps(stats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ports', type=int, nargs='+', required=True)
    ap.add_argument('--duration', type=float, required=True)
    args = ap.parse_args()
    rclpy.init()
    node = Load()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    asyncio.run(main_async(args))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
