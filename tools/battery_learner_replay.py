#!/usr/bin/env python3
"""Replay test of battery_learner (roomba_rs) through ROS: a fake /roomba/sensors publisher in a ROS
domain of its own drives a simulated 18 V pack (5S Li-ion, 2500 mAh, 0.2 ohm, its open-circuit voltage
the stock curve 1.03 V down, i.e. the seeded curve) through whole discharges and swaps, and checks the
node's /battery_state, its ~/pack_swapped service, log lines, exit codes, state file and CSV logs.

  tools/battery_learner_replay.py [--bin PATH] [--work DIR] [--speed 100] [--domain 79]

Frames are 1 s apart in stamp time and published --speed times faster than real time (the learner
times everything by the frame stamps). The run:
  1. first run, no state file (seed 18.29 V = 62 %): pack 1, truly at 55 %, starts from the voltage
  2. discharge (0.2 A idle / 0.6 A driving) to 35 %; the node is restarted (SIGINT): it continues
  3. discharge to 1 bar (15 %), pack out 90 s (no data: no /battery_state), a full pack in: 100 %
  4. full discharge to 1 bar; the node is stopped (SIGTERM), the pack swapped while it is off and the
     node started again: swap detected at start-up, 100 %, capacity learned (true 2500 mAh)
  5. a third pack down to 50 %: the published charge tracks the truth; a 30 s sensor gap without a
     jump (a driver restart) is the same pack
  6. ~/pack_swapped with no swap detected: a new pack at 100 %, nothing learned from the old one
Needs rclpy, roomba_msgs and the built node (default: roomba_rs's install, `colcon build`).
Writes only under --work (default ./battery_learner_replay).
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# The stock Li-ion curve (roomba_driver_rs battery.rs LI_ION_CURVE): cell volts -> charge.
CURVE = [(3.00, 0.00), (3.45, 0.05), (3.60, 0.10), (3.67, 0.20), (3.71, 0.30), (3.75, 0.40),
         (3.79, 0.50), (3.85, 0.60), (3.92, 0.70), (4.00, 0.80), (4.10, 0.90), (4.20, 1.00)]
OFFSET_V = 1.03      # the Roomba reads the pack this much below the stock curve (the seed's shift)
R_OHM = 0.2
CAPACITY_MAH = 2500.0
STEP_MV = 27.7       # the Roomba's voltage readings come in ~28 mV steps
START_STAMP = 1_800_000_000.0


def cell_voltage(soc):
    soc = min(max(soc, 0.0), 1.0)
    for (v0, f0), (v1, f1) in zip(CURVE, CURVE[1:]):
        if soc <= f1:
            return v0 + (v1 - v0) * (soc - f0) / (f1 - f0)
    return CURVE[-1][0]


class Pack:
    def __init__(self, soc):
        self.soc = soc

    def draw(self, current_ma, dt):
        self.soc += current_ma * dt / 3600.0 / CAPACITY_MAH

    def voltage_mv(self, current_ma):
        v = 5 * cell_voltage(self.soc) - OFFSET_V + current_ma / 1000.0 * R_OHM
        return int(round(v * 1000.0 / STEP_MV) * STEP_MV)


def profile(t):
    """40 s cycles: 26 s driving at 0.6 A, 14 s at rest at 0.2 A (about 0.46 A on average)."""
    return -600.0 if t % 40 < 26 else -200.0


class Failure(Exception):
    pass


def check(cond, what):
    print(('  ok    ' if cond else '  FAIL  ') + what, flush=True)
    if not cond:
        raise Failure(what)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', help='battery_learner executable (default: from the roomba_rs install)')
    ap.add_argument('--work', default='battery_learner_replay', help='state dir, params and logs go here')
    ap.add_argument('--speed', type=float, default=100.0, help='stamp seconds per real second')
    ap.add_argument('--domain', default='79', help='ROS_DOMAIN_ID (keep it off the robot)')
    args = ap.parse_args()
    os.environ['ROS_DOMAIN_ID'] = args.domain
    os.environ.setdefault('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')

    import rclpy
    from builtin_interfaces.msg import Time
    from roomba_msgs.msg import SensorState
    from sensor_msgs.msg import BatteryState
    from std_srvs.srv import Trigger

    binary = args.bin
    if not binary:
        from ament_index_python.packages import get_package_prefix
        binary = os.path.join(get_package_prefix('roomba_rs'), 'lib', 'roomba_rs', 'battery_learner')
    work = Path(args.work).resolve()
    state_dir = work / 'state'
    shutil.rmtree(state_dir, ignore_errors=True)
    state_dir.mkdir(parents=True)
    params = work / 'params.yaml'
    params.write_text(
        'battery_learner:\n  ros__parameters:\n'
        f'    state_dir: {state_dir}\n'
        '    cells: 5\n    internal_resistance_ohm: 0.2\n    one_bar_fraction: 0.15\n'
        '    capacity_mah: 2000.0\n    seed_voltage: 18.29\n    seed_fraction: 0.62\n')
    log_path = work / 'learner.log'
    log_path.write_text('')

    rclpy.init()
    node = rclpy.create_node('battery_replay')
    pub = node.create_publisher(SensorState, '/roomba/sensors', 10)
    client = node.create_client(Trigger, '/battery_learner/pack_swapped')
    battery = []   # (wall time, BatteryState)
    node.create_subscription(BatteryState, '/battery_state', lambda m: battery.append((time.monotonic(), m)), 10)

    sim = {'t': START_STAMP, 'pack': Pack(0.55), 'wall0': time.monotonic(), 'n': 0}
    proc = {'p': None}

    def spin(seconds=0.0):
        end = time.monotonic() + seconds
        while True:
            rclpy.spin_once(node, timeout_sec=max(0.0, min(0.01, end - time.monotonic())))
            if time.monotonic() >= end:
                return

    def frame(current_ma):
        pack = sim['pack']
        pack.draw(current_ma, 1.0)
        sim['t'] += 1.0
        m = SensorState()
        sec = int(sim['t'])
        m.header.stamp = Time(sec=sec, nanosec=int((sim['t'] - sec) * 1e9))
        m.voltage_mv = pack.voltage_mv(current_ma)
        m.current_ma = int(current_ma)
        m.oi_mode = SensorState.MODE_SAFE
        pub.publish(m)
        # pace: one frame per 1/speed s of real time
        sim['n'] += 1
        spin(max(0.0, sim['wall0'] + sim['n'] / args.speed - time.monotonic()))

    def repace():
        sim['wall0'], sim['n'] = time.monotonic(), 0

    def run(seconds, current=profile):
        for _ in range(int(seconds)):
            frame(current(sim['t']))

    def run_to(soc, current=profile):
        while sim['pack'].soc > soc:
            frame(current(sim['t']))

    def idle_for(wall_s):
        """Rest frames for this much real time (keeps /battery_state coming)."""
        end = time.monotonic() + wall_s
        while time.monotonic() < end:
            frame(-200.0)

    def latest(max_age=2.5):
        if battery and time.monotonic() - battery[-1][0] <= max_age:
            return battery[-1][1]
        return None

    def wait_battery(pred, wall_s, what):
        end = time.monotonic() + wall_s
        while time.monotonic() < end:
            frame(-200.0)
            m = latest()
            if m is not None and pred(m):
                return m
        m = latest()
        check(False, f"{what} (last: {m.percentage if m else None})")

    def log_text():
        return log_path.read_text(errors='replace')

    def start():
        offset = len(log_text())
        env = dict(os.environ, RCUTILS_LOGGING_BUFFERED_STREAM='0')
        proc['p'] = subprocess.Popen([binary, '--ros-args', '--params-file', str(params)],
                                     stdout=open(log_path, 'a'), stderr=subprocess.STDOUT, env=env)
        end = time.monotonic() + 30
        while 'battery_learner running' not in log_text()[offset:] or pub.get_subscription_count() < 1:
            if proc['p'].poll() is not None or time.monotonic() > end:
                check(False, 'battery_learner started')
            spin(0.1)
        check(True, 'battery_learner started')
        spin(1.0)  # discovery of the new node (a stopped one can linger in the graph for a while)
        repace()

    def stop(sig):
        p = proc['p']
        p.send_signal(sig)
        try:
            rc = p.wait(15)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = p.wait()
        check(rc == 0, f'battery_learner exits 0 on {signal.Signals(sig).name} (got {rc})')
        battery.clear()

    def truth():
        return sim['pack'].soc

    def state():
        return json.loads((state_dir / 'battery_learner.json').read_text())

    worst = {}

    def track(label, seconds=None, to=None):
        """Discharge and record the worst |published - true| charge."""
        err = 0.0
        count = 0
        while (seconds is not None and count < seconds) or (to is not None and sim['pack'].soc > to):
            frame(profile(sim['t']))
            count += 1
            if count % 50 == 0:
                m = latest()
                if m is not None:
                    err = max(err, abs(m.percentage - truth()))
        worst[label] = err
        return err

    try:
        print(f'battery_learner replay: {binary}, domain {args.domain}, {args.speed:g}x, work {work}')
        print('1. first run, pack 1 truly at 55 %')
        start()
        m = wait_battery(lambda m: True, 10, 'first /battery_state')
        check(abs(m.percentage - 0.55) < 0.04, f'starts from the seeded curve: {m.percentage:.3f} (truth 0.55)')
        check(m.power_supply_status == BatteryState.POWER_SUPPLY_STATUS_DISCHARGING, 'status discharging')
        check(m.serial_number == 'pack 1' and abs(m.capacity - 2.0) < 1e-6, f'pack 1, capacity prior 2 Ah ({m.serial_number}, {m.capacity})')
        check('no saved charge: pack 1 starts at' in log_text(), 'logged the seeded start')

        print('2. discharge to 35 %, restart the node')
        err = track('pack 1 (voltage-anchored)', to=0.35)
        check(err < 0.06, f'tracks the truth within 6 points (worst {err:.3f})')
        before = latest().percentage
        stop(signal.SIGINT)
        check(state()['pack']['id'] == 1, 'state saved at shutdown')
        start()
        m = wait_battery(lambda m: True, 10, '/battery_state after the restart')
        check(abs(m.percentage - before) < 0.02, f'continues after a restart: {before:.3f} -> {m.percentage:.3f}')
        check('continuing pack 1' in log_text(), 'logged "continuing pack 1"')

        print('3. discharge to 1 bar, swap with the node running')
        track('pack 1 to 1 bar', to=0.15)
        spin(5.0)  # pack out: no frames (the node stops publishing gap_s + 1 s after the last one)
        check(latest(max_age=1.5) is None, 'no /battery_state while the pack is out')
        sim['t'] += 90.0
        sim['pack'] = Pack(1.0)
        repace()
        m = wait_battery(lambda m: m.serial_number == 'pack 2', 10, 'pack 2 published')
        check(m.percentage > 0.99, f'new pack at 100 % ({m.percentage:.3f})')
        text = log_text()
        check('new pack (voltage jump after a sensor gap' in text, 'logged the detected swap')
        check('pack 1 out at 1 bar' in text, 'learned from pack 1 (started from its voltage)')

        print('4. full discharge, swap while the node is off')
        err = track('pack 2 (full, prior capacity)', to=0.15)
        check(err < 0.08, f'tracks the truth within 8 points with the capacity not learned yet (worst {err:.3f})')
        stop(signal.SIGTERM)
        sim['pack'] = Pack(1.0)
        sim['t'] += 600.0
        start()
        m = wait_battery(lambda m: m.serial_number == 'pack 3', 10, 'pack 3 published')
        check(m.percentage > 0.99, f'swap detected at start-up: 100 % ({m.percentage:.3f})')
        text = log_text()
        check('voltage jump since the last run' in text, 'logged the start-up swap')
        s = state()
        learned = s['learned_capacity_mah']
        check(2300 < learned < 2600, f'learned capacity {learned:.0f} mAh (true {CAPACITY_MAH:.0f}, prior 2000)')
        est = [r['capacity_mah'] for r in s['history']]
        check(len(est) == 2 and all(e and abs(e - CAPACITY_MAH) < 0.1 * CAPACITY_MAH for e in est),
              f'per-pack estimates {", ".join(f"{e:.0f}" for e in est)}')

        print('5. third pack to 50 %, a sensor gap without a jump')
        err = track('pack 3 (learned)', to=0.5)
        check(err < 0.04, f'tracks the truth within 4 points (worst {err:.3f})')
        sim['t'] += 30.0
        spin(0.5)
        repace()
        idle_for(3.0)
        m = latest()
        check(m is not None and m.serial_number == 'pack 3', 'same pack after the gap')
        check('sensor data back after 31 s' in log_text() and 'same pack' in log_text(), 'logged the gap')

        print('6. ~/pack_swapped (no swap detected: re-anchors, teaches nothing)')
        future = client.call_async(Trigger.Request())
        while not future.done():
            frame(-200.0)
        r = future.result()
        check(r.success and r.message.startswith('new pack 4 at 100 %'), f'service reply: {r.message}')
        m = wait_battery(lambda m: m.serial_number == 'pack 4', 10, 'pack 4 published')
        check(m.percentage > 0.99, f'manual reset to 100 % ({m.percentage:.3f})')
        check('pack 3 out after' in log_text() and 'not learned from (no swap was detected)' in log_text(),
              'logged the manual reset')
        stop(signal.SIGINT)

        s = state()
        check([r['how'] for r in s['history']] == ['gap', 'startup', 'manual'], 'history: gap, startup, manual')
        check(s['learned_capacity_mah'] == learned, 'learned capacity unchanged by the manual reset')
        rows = {p: len((state_dir / f'battery_pack_{p:04d}.csv').read_text().splitlines()) - 1 for p in (1, 2, 3)}
        check(all(n > 100 for n in rows.values()), f'discharge CSV rows per pack {rows}')
        print('\nPacks:')
        for r in s['history']:
            cap = f"{r['capacity_mah']:.0f} mAh" if r['capacity_mah'] else '-'
            print(f"  pack {r['id']}: out by {r['how']}, {r['anchor']} start {r['start_soc']:.2f}, "
                  f"used {r['used_mah']:.0f} mAh, estimate {cap}, weight {r['weight']}")
        print(f"  learned capacity {s['learned_capacity_mah']:.0f} mAh, curve offset {s['prior_offset_v']:+.3f} V "
              f"(true {OFFSET_V:+.2f})")
        print('Worst |published - true| charge: ' + ', '.join(f'{k} {v:.3f}' for k, v in worst.items()))
        print('PASS')
        return 0
    except Failure:
        print(f'FAILED; node log: {log_path}')
        return 1
    finally:
        if proc['p'] is not None and proc['p'].poll() is None:
            proc['p'].kill()
            proc['p'].wait()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
