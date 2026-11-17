#!/usr/bin/env python3
"""Offline check of sound_events (src/roomba_rs) against a fake Roomba driver: no robot needed.

Starts the sound_events node in a private ROS domain (localhost only) with HOME set to a scratch
directory, so its default mute file is <scratch>/.config/roomba/sound_muted. Then plays a scripted
scenario (about 42 s) on fake /roomba/sensors and /explorer/status, serves /roomba/define_song,
/roomba/play_song and /roomba/set_mode like the driver (song_playing while a song plays; play needs
Safe/Full mode) and checks:
  - which songs are asked for and when (hello, uh-oh, done jingle), with the full notes and play = true;
  - nothing during start-up (Passive mode, the latched status of an earlier run; those records are kept),
    a bump in Passive mode, the light bumpers alone,
    a second bump within 4 s, a bump while muted or while a song plays, a finished run in Passive mode,
    a second hello;
  - HELO on the display and cleared ~3 s later, only in Safe/Full mode (with --auto-reenter-mode, where the
    driver would re-enter Safe mode for a display command in Passive: no display command at all);
  - the OI mode never changed;
  - a slow define_song answer (2.5 s) doesn't hold anything up;
  - the node exits with status 0 on SIGINT.

  source /opt/ros/jazzy/setup.bash && source <workspace>/install/setup.bash
  ROS_DOMAIN_ID=77 python3 tools/sound_events_offline.py [--node <sound_events binary>] [--dir D] [--keep]
                                                        [--auto-reenter-mode]

Without --node it runs the roomba_rs sound_events installed in the sourced workspace (directly, not through
`ros2 run`, which doesn't pass a SIGINT sent to it on to the node). The scratch directory is made in --dir (default:
the current directory) and removed afterwards unless --keep. Exit status 0 when every check passed.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'

import rclpy  # noqa: E402
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from ament_index_python.packages import get_package_prefix  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from roomba_msgs.msg import SensorState  # noqa: E402
from roomba_msgs.srv import DefineSong, PlaySong, SetMode  # noqa: E402
from std_msgs.msg import String  # noqa: E402

PASSIVE, SAFE = 1, 2
SONGS = {0: ('hello', [67, 72, 76], [10, 10, 24]),
         1: ('uh-oh', [76, 76, 69], [12, 6, 28]),
         2: ('done', [72, 76, 79, 84, 0, 79, 84], [7, 7, 7, 14, 4, 8, 36])}


class FakeDriver(Node):
    def __init__(self, auto_reenter_mode):
        # The driver's node name: sound_events reads its auto_reenter_mode parameter.
        super().__init__('roomba_driver')
        self.declare_parameter('auto_reenter_mode', auto_reenter_mode)
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.mode = PASSIVE
        self.bump = [False, False]
        self.light = False
        self.song_until = 0.0
        self.delay = 0.0
        self.explorer = ('done', 'Exploration finished: no reachable unmapped areas left - back at the start point')
        # Everything from the start, start-up included; t is time.monotonic().
        self.calls = []        # (t, service, number, notes, durations, play, success)
        self.displays = []     # (t, text, mode)
        self.mode_calls = []   # (t, mode)
        g = ReentrantCallbackGroup()
        self.create_service(DefineSong, 'roomba/define_song', self.define_song, callback_group=g)
        self.create_service(PlaySong, 'roomba/play_song', self.play_song, callback_group=g)
        self.create_service(SetMode, 'roomba/set_mode', self.set_mode, callback_group=g)
        self.create_subscription(String, 'roomba/display', self.on_display, 10, callback_group=g)
        self.pub_sensors = self.create_publisher(SensorState, 'roomba/sensors', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.pub_status = self.create_publisher(String, 'explorer/status', latched)
        self.create_timer(0.02, self.publish_sensors, callback_group=MutuallyExclusiveCallbackGroup())
        # The explorer repeats its status every second.
        self.create_timer(1.0, self.publish_status, callback_group=MutuallyExclusiveCallbackGroup())
        self.publish_status()

    def define_song(self, req, res):
        t = time.monotonic()
        with self.lock:
            mode, delay = self.mode, self.delay
        res.success = len(req.notes) == len(req.durations) and 1 <= len(req.notes) <= 16 and req.song_number <= 4
        res.message = f'song {req.song_number}'
        if res.success and req.play:
            if mode in (2, 3):
                with self.lock:
                    self.song_until = max(self.song_until, time.monotonic() + sum(req.durations) / 64)
                res.message = f'song {req.song_number} defined and played sent'
            else:
                res.success = False
                res.message = f'song {req.song_number} defined and played needs Safe or Full mode'
        self.calls.append((t, 'define_song', req.song_number, list(req.notes), list(req.durations), req.play,
                           res.success))
        if delay:
            time.sleep(delay)  # a slow driver
        return res

    def play_song(self, req, res):
        self.calls.append((time.monotonic(), 'play_song', req.song_number, None, None, True, True))
        res.success, res.message = True, f'play song {req.song_number} sent'
        return res

    def set_mode(self, req, res):
        self.mode_calls.append((time.monotonic(), req.mode))
        res.success, res.mode = True, req.mode
        return res

    def on_display(self, m):
        with self.lock:
            mode = self.mode
        self.displays.append((time.monotonic(), m.data, mode))

    def publish_sensors(self):
        m = SensorState()
        m.header.stamp = self.get_clock().now().to_msg()
        with self.lock:
            m.oi_mode = self.mode
            m.bump_left, m.bump_right = self.bump
            m.light_bump_center_left = m.light_bump_center_right = self.light
            m.light_bump_center_left_signal = m.light_bump_center_right_signal = 900 if self.light else 20
            m.song_playing = time.monotonic() < self.song_until
        self.pub_sensors.publish(m)

    def publish_status(self):
        with self.lock:
            state, message = self.explorer
        status = {'state': state, 'message': message, 'reached': 4, 'failed': 1, 'frontiers': 0,
                  'home': [0.0, 0.0], 'goal': None}
        self.pub_status.publish(String(data=json.dumps(status)))

    # ---- scenario controls
    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def set_explorer(self, state, message=''):
        self.set(explorer=(state, message))
        self.publish_status()

    def press(self, left=True, right=False, hold=0.3):
        self.set(bump=[left, right])
        threading.Timer(hold, lambda: self.set(bump=[False, False])).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--node', help='sound_events binary (default: the installed roomba_rs one)')
    ap.add_argument('--dir', default='.', help='where to make the scratch directory (default: here)')
    ap.add_argument('--keep', action='store_true', help='keep the scratch directory (node log)')
    ap.add_argument('--auto-reenter-mode', action='store_true',
                    help='the fake driver reports auto_reenter_mode: true (expect no display commands)')
    a = ap.parse_args()
    if os.environ.get('ROS_DOMAIN_ID', '0') in ('', '0'):
        sys.exit('Set ROS_DOMAIN_ID to a private domain (e.g. 77) so this cannot reach a running robot.')

    scratch = os.path.abspath(tempfile.mkdtemp(prefix='sound_events_offline_', dir=a.dir))
    mute = os.path.join(scratch, '.config', 'roomba', 'sound_muted')
    os.makedirs(os.path.dirname(mute))
    env = dict(os.environ, HOME=scratch, RCUTILS_COLORIZED_OUTPUT='0', ROS_LOG_DIR=os.path.join(scratch, 'ros_log'))
    node = a.node or os.path.join(get_package_prefix('roomba_rs'), 'lib', 'roomba_rs', 'sound_events')
    cmd = [node, '--ros-args', '--log-level', 'sound_events:=debug']
    log_path = os.path.join(scratch, 'sound_events.log')
    log = open(log_path, 'w')
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

    rclpy.init()
    fake = FakeDriver(a.auto_reenter_mode)
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(fake)
    threading.Thread(target=ex.spin, daemon=True).start()

    # Wait for the node to subscribe, then start the scenario clock.
    deadline = time.monotonic() + 30
    while (fake.pub_sensors.get_subscription_count() == 0 or fake.pub_status.get_subscription_count() == 0) \
            and time.monotonic() < deadline:
        time.sleep(0.1)
    if fake.pub_sensors.get_subscription_count() == 0:
        proc.kill()
        sys.exit(f'sound_events never subscribed; its log: {log_path}')
    time.sleep(1.0)
    # The records of the start-up (Passive mode, the latched "done" of an earlier run) are kept: the
    # checks below see them too, and they must be empty.
    fake.t0 = time.monotonic()

    def at(t, what, fn):
        time.sleep(max(0.0, fake.t0 + t - time.monotonic()))
        print(f'  {t:5.1f} s  {what}', flush=True)
        fn()

    print('Scenario:')
    at(0.0, 'Passive mode; latched explorer status "done" of an earlier run', lambda: None)
    at(1.0, 'bump (Passive mode)', lambda: fake.press())
    at(2.0, 'Safe mode', lambda: fake.set(mode=SAFE))
    at(6.0, 'bump left', lambda: fake.press())
    at(7.0, 'bump right (1 s after the uh-oh)', lambda: fake.press(False, True))
    at(10.5, 'bump left', lambda: fake.press())
    at(14.0, 'mute file created', lambda: open(mute, 'w').close())
    at(15.0, 'bump left (muted)', lambda: fake.press())
    at(16.0, 'mute file removed', lambda: os.remove(mute))
    at(17.0, 'bump left', lambda: fake.press())
    at(19.0, 'explorer: exploring', lambda: fake.set_explorer('exploring', 'Looking for unmapped areas…'))
    at(22.0, 'explorer: returning', lambda: fake.set_explorer(
        'returning', 'Exploration finished: no reachable unmapped areas left'))
    at(24.0, 'explorer: done, back at the start point (repeated every second)', lambda: fake.set_explorer(
        'done', 'Exploration finished: no reachable unmapped areas left - back at the start point'))
    at(26.0, 'light bumpers only', lambda: fake.set(light=True))
    at(26.5, 'light bumpers clear', lambda: fake.set(light=False))
    at(29.0, 'another node plays a song (1.5 s)', lambda: fake.set(song_until=time.monotonic() + 1.5))
    at(29.5, 'bump left (song playing)', lambda: fake.press())
    at(31.0, 'explorer: exploring', lambda: fake.set_explorer('exploring', 'Looking for unmapped areas…'))
    at(32.0, 'Passive mode', lambda: fake.set(mode=PASSIVE))
    at(33.0, 'explorer: done, finished (Passive mode)', lambda: fake.set_explorer('done', 'Exploration finished'))
    at(34.0, 'Safe mode again', lambda: fake.set(mode=SAFE))
    at(36.0, 'driver gets slow: define_song answers after 2.5 s', lambda: fake.set(delay=2.5))
    at(36.5, 'bump left', lambda: fake.press())
    at(37.0, 'driver fast again; explorer: exploring', lambda: (
        fake.set(delay=0.0), fake.set_explorer('exploring', 'Looking for unmapped areas…')))
    at(37.5, 'explorer: done, back home on request (uh-oh call still unanswered)', lambda: fake.set_explorer(
        'done', 'Returning to the start point - back at the start point'))
    at(42.0, 'SIGINT to sound_events', lambda: proc.send_signal(signal.SIGINT))
    try:
        code = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        code = 'hung'
    log.close()
    ex.shutdown()
    fake.destroy_node()
    rclpy.shutdown()

    calls = [(t - fake.t0, *rest) for t, *rest in fake.calls]
    displays = [(t - fake.t0, text, mode) for t, text, mode in fake.displays]
    mode_calls = [(round(t - fake.t0, 2), mode) for t, mode in fake.mode_calls]
    print('\nService calls (t from scenario start; negative: during start-up):')
    for t, srv, n, notes, durations, play, ok in calls:
        print(f'  {t:6.2f} s  {srv} song {n} ({SONGS.get(n, ("?",))[0]}) notes={notes} durations={durations} '
              f'play={play} -> success={ok}')
    print('Display:')
    for t, text, mode in displays:
        print(f'  {t:6.2f} s  "{text}" (OI mode {mode})')

    checks = []

    def check(name, ok, detail=''):
        checks.append(ok)
        print(f'  [{"PASS" if ok else "FAIL"}] {name}{": " + detail if detail else ""}')

    print('\nChecks:')
    early = ([f'{srv} song {n} at {t:.2f} s' for t, srv, n, *_ in calls if t < 0]
             + [f'display "{text}" at {t:.2f} s' for t, text, _ in displays if t < 0])
    check('nothing played or shown during start-up (Passive mode, latched "done" of an earlier run)',
          not early, ', '.join(early))
    expected = [(2.0, 0), (6.0, 1), (10.5, 1), (17.0, 1), (24.0, 2), (36.5, 1), (38.5, 2)]
    got = [(t, n) for t, srv, n, _, _, play, _ in calls if srv == 'define_song' and play]
    check('songs asked for, in order', [n for _, n in got] == [n for _, n in expected],
          ', '.join(f'{SONGS[n][0]} {t:.2f} s' for t, n in got))
    for (te, n), (tg, ng) in zip(expected, got):
        check(f'{SONGS[n][0]} at {te:.1f} s', ng == n and abs(tg - te) <= 0.35,
              f'{tg:.2f} s ({(tg - te) * 1000:+.0f} ms)')
    check('every call is define_song with the full song and play = true',
          all(srv == 'define_song' and play and n in SONGS and (notes, durations) == SONGS[n][1:]
              for _, srv, n, notes, durations, play, _ in calls),
          f'{len(calls)} calls')
    disp = [(round(t, 2), text) for t, text, _ in displays]
    if a.auto_reenter_mode:
        check('no display command (driver has auto_reenter_mode: true)', not disp, str(disp))
    else:
        check('HELO at hello, cleared ~3 s later', [text for _, text in disp] == ['HELO', '    ']
              and abs(disp[0][0] - 2.0) <= 0.35 and abs(disp[1][0] - 5.0) <= 0.35, str(disp))
    check('display only written in Safe/Full mode', all(mode in (2, 3) for *_, mode in displays))
    check('OI mode never changed (no set_mode call)', not mode_calls, str(mode_calls))
    check('exits with status 0 on SIGINT', code == 0, str(code))
    with open(log_path) as f:
        node_log = f.read()
    print('\nsound_events log:')
    for line in node_log.splitlines():
        print('  ' + line)
    if a.keep:
        print(f'\nScratch directory kept: {scratch}')
    else:
        shutil.rmtree(scratch, ignore_errors=True)
    ok = all(checks)
    print(f'\n{"ALL PASSED" if ok else "FAILED"} ({sum(checks)}/{len(checks)} checks)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
