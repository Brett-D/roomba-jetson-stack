#!/usr/bin/env python3
"""The Roomba's Open Interface on a pty, backed by the Isaac Sim robot: lets the robot's real roomba_driver
(Rust, robot config, only the port changed) drive the simulated Roomba.

  isaac_roomba.py --link /root/isaacloop/run/x/ttyROOMBA [--log serial.jsonl]

Like roomba-serial-ros2's tools/fake_roomba.py (whose OI tables and framing it reuses) the OI modes, the
15 ms sensor stream (opcode 148, whatever layout the driver asks for), pause/resume and queries behave as
on the robot, but:
- drive commands (137 drive, 145 drive direct, 146 drive PWM; Safe/Full mode only) become wheel speeds on
  /isaac/roomba/wheel_cmd (std_msgs/Float64MultiArray [left, right] m/s), re-sent every 100 ms;
- the sensor values come from Isaac Sim's /isaac/roomba/state (Float64MultiArray):
  [t, left wheel travel m, right wheel travel m, bump left, bump right, wheel drop, cliff L, FL, FR, R,
   light bumper L, FL, CL, CR, FR, R (0-4095), wall signal, home base, update count]
  encoders = travel / (pi * 0.072 / 508.8) counts (16 bit, wrapping), cliff signals 2500 / 40, battery a
  steady 19.8 V pack at -0.6 A.
- Safe mode drops to Passive (motors off) on a cliff while driving forward or a wheel drop, as the OI does.
- The stream is paced by SIM time (--pace sim, default): one frame per Isaac state message (one per app update,
  ~16.7 ms of sim time, close to the OI's 15 ms). The driver stamps each frame with the ROS (sim) clock and turns
  encoder deltas into wheel velocities, so a 15 ms WALL-clock stream would alias when the sim runs slower than real
  time (most frames repeat the counts -> velocity 0, the next one jumps); the EKF then trusts the many zero samples
  (tight covariance at low speed) and under-estimates the distance by ~35 %. Without state messages (Isaac paused
  or not yet up) it falls back to the wall-clock stream so the driver's link stays alive.
"""
import argparse
import json
import math
import os
import select
import signal
import struct
import sys
import threading
import time
import tty

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                                'src', 'roomba-serial-ros2', 'tools'))
from fake_roomba import BOOT_TEXT, FMT, M_PER_COUNT, TICK, command_length, ids_of  # noqa: E402

import rclpy  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import Float64MultiArray  # noqa: E402

WHEEL_BASE = 0.235


class IsaacRoomba:
    def __init__(self, a):
        self.link = a.link
        self.master, slave = os.openpty()
        tty.setraw(slave)
        self.slave = slave
        os.set_blocking(self.master, False)
        tmp = self.link + '.new'
        if os.path.lexists(tmp):
            os.unlink(tmp)
        os.symlink(os.ttyname(slave), tmp)
        os.replace(tmp, self.link)
        self.log = open(a.log, 'w') if a.log else None
        self.mode = 0
        self.layout = None
        self.streaming = False
        self.requested = (0, 0, 0, 0)
        self.wheels = (0.0, 0.0)
        self.buf = bytearray()
        self.state = [0.0] * 19
        self.state_t = 0.0
        self.state_sim_t = -1.0
        self.pace = a.pace
        self.fresh = False
        self.wake_r, self.wake_w = os.pipe()
        os.set_blocking(self.wake_r, False)
        os.set_blocking(self.wake_w, False)
        self.lock = threading.Lock()
        self.stats = {'frames': 0, 'commands': 0, 'safety_stops': 0, 'state_msgs': 0}
        rclpy.init()
        self.node = rclpy.create_node('isaac_roomba')
        q = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.node.create_publisher(Float64MultiArray, '/isaac/roomba/wheel_cmd', q)
        self.node.create_subscription(Float64MultiArray, '/isaac/roomba/state', self.on_state, q)
        self.spin = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin.start()
        self._event('start', pty=os.ttyname(slave), link=self.link)

    def _event(self, ev, **kw):
        if self.log:
            kw.update(ev=ev, t=time.monotonic())
            self.log.write(json.dumps(kw) + '\n')
            self.log.flush()

    def on_state(self, m):
        with self.lock:
            if len(m.data) >= 18:
                self.state = list(m.data)
                self.state_t = time.monotonic()
                self.stats['state_msgs'] += 1
                if m.data[0] > self.state_sim_t:
                    self.state_sim_t = m.data[0]
                    self.fresh = True
                    try:
                        os.write(self.wake_w, b'.')
                    except BlockingIOError:
                        pass

    # ---------------------------------------------------------------- wheels
    def set_wheels(self, left, right, why):
        if (left, right) != self.wheels:
            self._event('wheels', left=round(left, 4), right=round(right, 4), why=why)
        self.wheels = (left, right)
        self.publish_wheels()

    def publish_wheels(self):
        m = Float64MultiArray()
        m.data = [float(self.wheels[0]), float(self.wheels[1])]
        self.pub.publish(m)

    def set_mode(self, mode):
        if mode != self.mode:
            self._event('mode', mode=mode)
            self.mode = mode
        if mode not in (2, 3):
            self.set_wheels(0.0, 0.0, f'mode {mode}')

    # ---------------------------------------------------------------- commands
    def on_bytes(self, data):
        self.buf += data
        while self.buf:
            n = command_length(self.buf)
            if n is None:
                break
            if n == 0:
                del self.buf[:1]
                continue
            cmd = bytes(self.buf[:n])
            del self.buf[:n]
            self.stats['commands'] += 1
            self.execute(cmd)

    def execute(self, cmd):
        op = cmd[0]
        if op == 128:
            self.set_mode(1)
            return
        if self.mode == 0:
            return
        if op in (130, 131):
            self.set_mode(2)
        elif op == 132:
            self.set_mode(3)
        elif op in (133, 134, 135, 136, 143):
            self.set_mode(1)
        elif op == 173:
            self.set_mode(0)
            self.streaming = False
        elif op == 7:
            self.set_mode(0)
            self.streaming = False
            self.layout = None
            os.write(self.master, BOOT_TEXT)
        elif op == 148:
            self.layout = list(cmd[2:])
            self.streaming = True
            self._event('stream', ids=self.layout)
        elif op == 150:
            self.streaming = bool(cmd[1]) and self.layout is not None
        elif op in (142, 149):
            ids = [cmd[1]] if op == 142 else list(cmd[2:])
            v = self.values()
            os.write(self.master, b''.join(self.packet(i, v) for i in ids))
        elif op in (137, 145, 146) and self.mode in (2, 3):
            a, b = struct.unpack('>hh', cmd[1:])
            if op == 137:            # velocity mm/s, radius mm
                self.requested = (a, b, 0, 0)
                v = a / 1000.0
                if b in (32767, -32768):
                    left = right = v
                elif b == 1:
                    left, right = -v, v
                elif b == -1:
                    left, right = v, -v
                else:
                    r = b / 1000.0
                    left, right = v * (r - WHEEL_BASE / 2) / r, v * (r + WHEEL_BASE / 2) / r
            elif op == 145:          # right, left mm/s
                self.requested = (0, 0, a, b)
                right, left = a / 1000.0, b / 1000.0
            else:                    # right, left PWM -255..255 (~0.5 m/s at full)
                right, left = a / 255.0 * 0.5, b / 255.0 * 0.5
            self.set_wheels(left, right, f'op {op}')

    # ---------------------------------------------------------------- sensors
    def values(self):
        with self.lock:
            s = list(self.state)
        _, dl, dr, bl, br, drop, cl, cfl, cfr, cr = s[:10]
        light = [int(x) for x in s[10:16]]
        wall = int(s[16])
        home = int(s[17])
        moving = self.wheels != (0.0, 0.0)
        v = {7: int(br) | int(bl) << 1 | int(drop) << 2 | int(drop) << 3, 8: int(wall > 60), 9: int(cl), 10: int(cfl),
             11: int(cfr), 12: int(cr), 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0,
             22: 19800, 23: -600 if moving else -300, 24: 27, 25: 2000, 26: 2696, 27: wall,
             28: 40 if cl else 2500, 29: 40 if cfl else 2550, 30: 40 if cfr else 2450, 31: 40 if cr else 2480,
             32: 0, 33: 0, 34: home << 1, 35: self.mode, 36: 0, 37: 0, 38: len(self.layout or []),
             39: self.requested[0], 40: self.requested[1], 41: self.requested[2], 42: self.requested[3],
             43: int(round(dl / M_PER_COUNT)) % 65536, 44: int(round(dr / M_PER_COUNT)) % 65536,
             45: sum(1 << i for i, x in enumerate(light) if x >= 200), 46: light[0], 47: light[1], 48: light[2],
             49: light[3], 50: light[4], 51: light[5], 52: 0, 53: 0, 54: 120 if moving else 0,
             55: 118 if moving else 0, 56: 0, 57: 0, 58: 1 if moving and sum(self.wheels) > 0 else 0}
        return v

    RANGE = {'B': (0, 255), 'b': (-128, 127), 'H': (0, 65535), 'h': (-32768, 32767)}

    def packet(self, pid, v):
        out = b''
        for i in ids_of(pid):
            lo, hi = self.RANGE[FMT[i]]
            out += struct.pack('>' + FMT[i], max(lo, min(hi, int(v[i]))))
        return out

    def safety(self, v):
        """Safe mode: cliff while driving forward, or a wheel drop -> Passive, motors off."""
        fwd = sum(self.wheels) > 0
        cliff = any(v[i] for i in (9, 10, 11, 12))
        if self.mode == 2 and ((cliff and fwd) or (v[7] & 0b1100)):
            self.stats['safety_stops'] += 1
            self._event('safety', cliff=cliff, drop=bool(v[7] & 0b1100))
            self.set_mode(1)

    def tick(self):
        v = self.values()
        self.safety(v)
        if not self.streaming or not self.layout:
            return
        v = self.values()
        payload = b''.join(bytes([pid]) + self.packet(pid, v) for pid in self.layout)
        body = bytes([19, len(payload)]) + payload
        try:
            os.write(self.master, body + bytes([-sum(body) & 0xFF]))
            self.stats['frames'] += 1
        except BlockingIOError:
            self._event('overflow')

    def run(self):
        stop = []
        signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
        signal.signal(signal.SIGINT, lambda *_: stop.append(1))
        nxt = time.monotonic() + TICK
        last_pub = 0.0
        last_report = time.monotonic()
        while not stop:
            now = time.monotonic()
            try:
                ready, _, _ = select.select([self.master, self.wake_r], [], [], max(0.0, min(nxt - now, 0.05)))
            except InterruptedError:
                continue
            if self.master in ready:
                try:
                    data = os.read(self.master, 4096)
                except (BlockingIOError, InterruptedError, OSError):
                    data = b''
                if data:
                    self.on_bytes(data)
            if self.wake_r in ready:
                try:
                    os.read(self.wake_r, 4096)
                except BlockingIOError:
                    pass
            now = time.monotonic()
            sim_paced = self.pace == 'sim' and self.state_t and now - self.state_t < 0.25
            if sim_paced:
                with self.lock:
                    fresh, self.fresh = self.fresh, False
                if fresh:
                    self.tick()
                nxt = now + TICK
            while now >= nxt:
                self.tick()
                nxt += TICK
                if now - nxt > 0.5:          # fell far behind (suspended): resync
                    nxt = now + TICK
            if now - last_pub > 0.1:
                last_pub = now
                self.publish_wheels()
            if now - last_report > 10.0:
                last_report = now
                age = now - self.state_t if self.state_t else None
                print(json.dumps({'isaac_roomba': self.stats, 'mode': self.mode, 'wheels': self.wheels,
                                  'state_age_s': round(age, 2) if age is not None else None}), flush=True)
        self._event('stop', **self.stats)
        if os.path.lexists(self.link):
            os.unlink(self.link)
        self.node.destroy_node()
        rclpy.try_shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--link', required=True)
    ap.add_argument('--log', default='')
    ap.add_argument('--pace', choices=('sim', 'wall'), default='sim',
                    help='sensor stream: one frame per Isaac state message (sim) or every 15 ms wall clock (wall)')
    IsaacRoomba(ap.parse_args()).run()


if __name__ == '__main__':
    sys.exit(main())
