#!/usr/bin/env python3
"""Runs 2D digital-twin scenarios headless, one after the other, and summarises the outcomes.

  tools/twin2d_run.py src/roomba_bringup/config/twin2d/*.yaml
  tools/twin2d_run.py --seeds 1,2,3 --out /data/twin/twin2d/runs/batch1 explore_from_dock.yaml door_closes.yaml

For each scenario and seed: starts scripts/run_twin2d.sh (twin2d.launch.py: the twin plus the real
stack) in its own session with out_dir <out>/<scenario>_s<seed>, samples the CPU of every process of
the run every few seconds, waits for the launch to end by itself (the twin writes result.json and
the launch shuts down when the explorer is done or the scenario's time is up), stops it if it
overruns, and kills anything left in its session. Then writes <out>/summary.json and
<out>/summary.md (one row per run) and prints the table. Bare scenario names are looked up in
src/roomba_bringup/config/twin2d/. Extra launch arguments: --launch-args "web:=false foxglove:=true".
ROS_DOMAIN_ID: the environment's, default 68.

Parallel runs (the whole stack uses about 1.2 of the Jetson's 6 cores): --jobs 3 runs three at a
time, each in its own ROS domain (--domains, default 71, 72, ...), with its own web and Foxglove
ports (18100 / 18800 + domain), pty and out_dir:
  tools/twin2d_run.py --jobs 3 --seeds 1,2 --out /data/twin/twin2d/runs/after1 explore_from_dock stairs
--watch gives the first slot the usual ports (web 18097, Foxglove 18765, Foxglove started) so a run
can be watched live, and --now-file says which run that slot is on; --web-auth gives every run's web
app the same password (an auth.json made with set_password under a temporary HOME); --ws runs
another workspace's build (e.g. one of the previous version, for a before/after with these scenarios).
"""
import argparse
import glob
import json
import os
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
SCENARIOS = os.path.join(WS, 'src', 'roomba_bringup', 'config', 'twin2d')
CLK = os.sysconf('SC_CLK_TCK')
ACTIVE = set()   # sessions (launch pids) of the runs in progress


def session_pids(sid):
    pids = []
    for d in glob.glob('/proc/[0-9]*'):
        try:
            with open(d + '/stat') as f:
                fields = f.read().rsplit(')', 1)[1].split()
            if int(fields[3]) == sid:  # field 6 (session) after pid, comm, state, ppid, pgrp
                pids.append(int(d[6:]))
        except (OSError, IndexError, ValueError):
            pass
    return pids


def proc_name(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            args = f.read().split(b'\0')
    except OSError:
        return None
    args = [a.decode(errors='replace') for a in args if a]
    for a in args:
        if a.startswith('__node:='):
            return a.split('=', 1)[1]
    for k, a in enumerate(args):
        if a == '-r' and k + 1 < len(args) and args[k + 1].startswith('__node:='):
            return args[k + 1].split('=', 1)[1]
    if not args:
        return None
    name = os.path.basename(args[0])
    if name.startswith('python') and len(args) > 1:
        name = os.path.basename(args[1])
        if 'launch' in args:
            name = 'ros2 launch'
    return name


def cpu_ticks(pid):
    try:
        with open(f'/proc/{pid}/stat') as f:
            fields = f.read().rsplit(')', 1)[1].split()
        return int(fields[11]) + int(fields[12])  # utime + stime
    except (OSError, IndexError, ValueError):
        return None


def total_ticks():
    with open('/proc/stat') as f:
        parts = f.readline().split()[1:]
    vals = [int(v) for v in parts]
    return sum(vals), vals[3] + vals[4]  # total, idle + iowait


class CpuMeter:
    """Per-process CPU (% of one core) of a session, accumulated over the run."""

    def __init__(self, sid):
        self.sid = sid
        self.last = {}
        self.used = {}
        self.names = {}
        self.t0 = time.monotonic()
        self.sys0 = total_ticks()

    def sample(self):
        for pid in session_pids(self.sid):
            t = cpu_ticks(pid)
            if t is None:
                continue
            if pid not in self.names:
                self.names[pid] = proc_name(pid) or str(pid)
            prev = self.last.get(pid, 0)
            self.used[pid] = self.used.get(pid, 0) + max(0, t - prev)
            self.last[pid] = t

    def report(self):
        secs = max(time.monotonic() - self.t0, 1e-3)
        by_name = {}
        for pid, ticks in self.used.items():
            name = self.names.get(pid, str(pid))
            by_name[name] = by_name.get(name, 0.0) + ticks / CLK / secs * 100.0
        tot, idle = total_ticks()
        busy = 1.0 - (idle - self.sys0[1]) / max(tot - self.sys0[0], 1)
        rows = sorted(by_name.items(), key=lambda kv: -kv[1])
        return {'seconds': round(secs, 1), 'stack_pct_of_one_core': round(sum(by_name.values()), 1),
                'machine_busy_pct': round(busy * 100.0, 1), 'cores': os.cpu_count(),
                'per_process_pct': {k: round(v, 1) for k, v in rows}}


def kill_session(sid, grace=25.0):
    try:
        os.kill(sid, signal.SIGTERM)  # the launch (graceful.py): SIGINT to every node
    except ProcessLookupError:
        pass
    end = time.monotonic() + grace
    while time.monotonic() < end and session_pids(sid):
        time.sleep(0.5)
    for pid in session_pids(sid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(0.5)
    return session_pids(sid)


def scenario_path(name):
    if os.path.isfile(name):
        return os.path.abspath(name)
    for cand in (os.path.join(SCENARIOS, name), os.path.join(SCENARIOS, name + '.yaml')):
        if os.path.isfile(cand):
            return cand
    sys.exit(f'scenario not found: {name}')


def duration_of(path):
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return float(data.get('duration_s', 1200.0))


def run_one(scenario, seed, out_root, launch_args, sample_s, slot=None, opts=None):
    """slot: None (the environment's ROS domain, default ports) or {'domain', 'watch'}; opts: the
    runner's options (web_auth, now_file)."""
    stem = os.path.splitext(os.path.basename(scenario))[0]
    out = os.path.join(out_root, stem if seed < 0 else f'{stem}_s{seed}')
    os.makedirs(out, exist_ok=True)
    for stale in ('result.json', 'trajectory.png', 'timeline.csv', 'events.jsonl', 'zones.json'):
        if os.path.exists(os.path.join(out, stale)):
            os.remove(os.path.join(out, stale))
    ws = opts.ws if opts is not None and opts.ws else WS
    cmd = [os.path.join(ws, 'scripts', 'run_twin2d.sh'), f'scenario:={scenario}', f'out_dir:={out}',
           f'seed:={seed}'] + launch_args
    env = dict(os.environ)
    env.setdefault('ROS_DOMAIN_ID', '68')
    if slot is not None:
        # Parallel runs: each in its own ROS domain, with its own web and Foxglove ports. The watch
        # slot keeps the usual ones (18097, 18765) and runs Foxglove, for watching it live.
        d = slot['domain']
        env['ROS_DOMAIN_ID'] = str(d)
        cmd += (['web_port:=18097', 'foxglove_port:=18765', 'foxglove:=true'] if slot.get('watch')
                else [f'web_port:={18100 + d}', f'foxglove_port:={18800 + d}'])
    if opts is not None and opts.web_auth:
        # A fixed twin-only web password (auth.json from set_password) instead of a new one per run.
        cfg = os.path.join(out, 'home', '.config', 'roomba_web')
        os.makedirs(cfg, exist_ok=True)
        shutil.copyfile(opts.web_auth, os.path.join(cfg, 'auth.json'))
        os.chmod(os.path.join(cfg, 'auth.json'), 0o600)
    if opts is not None and opts.now_file and slot is not None and slot.get('watch'):
        with open(opts.now_file, 'w') as f:
            f.write(f'Watch slot (http://localhost:18097, Foxglove ws://localhost:18765): {stem} seed {seed}\n'
                    f'since {time.strftime("%Y-%m-%d %H:%M:%S")}, out_dir {out}, ROS_DOMAIN_ID {slot["domain"]}\n')
    log = open(os.path.join(out, 'launch.log'), 'w')
    print(f'== {stem} seed {seed}: {out}' + ('' if slot is None else f' (ROS_DOMAIN_ID {slot["domain"]})'),
          flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, env=env)
    ACTIVE.add(proc.pid)
    meter = CpuMeter(proc.pid)
    limit = duration_of(scenario) * 1.25 + 180.0
    started = time.monotonic()
    overran = False
    try:
        while proc.poll() is None:
            time.sleep(sample_s)
            meter.sample()
            if time.monotonic() - started > limit:
                overran = True
                print(f'   overran {limit:.0f} s: stopping it', flush=True)
                break
    except KeyboardInterrupt:
        print('   interrupted: stopping the run', flush=True)
        kill_session(proc.pid)
        raise
    finally:
        cpu = meter.report()
        left = kill_session(proc.pid)
        ACTIVE.discard(proc.pid)
        log.close()
    with open(os.path.join(out, 'cpu.json'), 'w') as f:
        json.dump(cpu, f, indent=1)
    result_file = os.path.join(out, 'result.json')
    result = json.load(open(result_file)) if os.path.isfile(result_file) else {'outcome': 'no result.json'}
    result['run'] = {'out_dir': out, 'wall_s': round(time.monotonic() - started, 1), 'overran': overran,
                     'left_running': left, 'launch_args': launch_args, 'cpu': cpu,
                     'ros_domain_id': int(env['ROS_DOMAIN_ID'])}
    with open(os.path.join(out, 'run.json'), 'w') as f:
        json.dump(result['run'], f, indent=1)
    return stem, seed, result


def row(stem, seed, r):
    cov = r.get('coverage', {})
    goals = r.get('nav2_goals', {})
    counts = r.get('counts', {})
    stuck = r.get('stuck', {})
    bat = r.get('battery', {})
    cpu = r.get('run', {}).get('cpu', {})
    pct = lambda v: '' if v is None else f'{100 * v:.0f}'  # noqa: E731
    return {
        'scenario': stem, 'seed': seed, 'outcome': r.get('outcome', '?'),
        'coverage_pct': pct(cov.get('final')), 'reachable_pct': pct(cov.get('reachable_final')),
        'sim_s': r.get('sim_time_s', ''), 'distance_m': r.get('distance_m', ''),
        'bumps': len(r.get('bumps', [])), 'contacts': len(r.get('contacts', [])),
        'stuck': f"{stuck.get('count', '')}/{stuck.get('total_s', '')}s",
        'idle_wait_s': stuck.get('idle_total_s', ''),
        'goals_ok_abort_cancel': f"{goals.get('succeeded', '')}/{goals.get('aborted', '')}/{goals.get('canceled', '')}",
        'false_success': len(goals.get('false_successes', [])),
        'recoveries': sum(v for k, v in counts.items() if k in ('nav2_spin', 'nav2_backup', 'nav2_wait')),
        'driver_backups': counts.get('driver_bump_backups', 0),
        'interventions': r.get('operator_interventions', ''),
        'battery_end_pct': bat.get('end_pct', ''),
        'stack_cpu_pct': cpu.get('stack_pct_of_one_core', ''),
        'explorer_final': (r.get('explorer', {}).get('final_message') or '')[:70],
    }


def markdown(rows):
    if not rows:
        return ''
    keys = list(rows[0].keys())
    out = ['| ' + ' | '.join(keys) + ' |', '|' + '|'.join('---' for _ in keys) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(r[k]) for k in keys) + ' |')
    return '\n'.join(out) + '\n'


def run_job(sc, seed, out_root, launch_args, sample_s, slot, opts=None):
    for attempt in range(2):
        stem, seed_used, result = run_one(sc, seed, out_root, launch_args, sample_s, slot, opts)
        # Nav2's lifecycle bring-up occasionally hangs (a lost change_state response), or the
        # driver's mode never reaches the twin: the explorer then never starts. That says nothing
        # about the scenario: run it again.
        detail = result.get('outcome_detail', '')
        infra = (result.get('outcome') == 'explorer_failed_to_start'
                 and ('Navigation is not running' in detail or 'never reported safe mode' in detail))
        if not infra or attempt == 1:
            break
        print(f'   {stem} seed {seed}: the explorer never started ({detail}): running it again', flush=True)
        result['run']['retried_after'] = detail
        failed = os.path.join(out_root, 'nav2_bringup_failures')
        os.makedirs(failed, exist_ok=True)
        with open(os.path.join(failed, f'{stem}_{int(time.time())}.json'), 'w') as f:
            json.dump(result, f, indent=1)
    return stem, result.get('seed', seed_used), result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scenarios', nargs='+')
    ap.add_argument('--seeds', default='', help='comma-separated seeds (default: each scenario\'s own)')
    ap.add_argument('--out', default='', help='default /data/twin/twin2d/runs/batch_<time>')
    ap.add_argument('--launch-args', default='', help='extra twin2d.launch.py arguments')
    ap.add_argument('--sample-s', type=float, default=5.0)
    ap.add_argument('--jobs', type=int, default=1, help='runs at a time (each in its own ROS domain)')
    ap.add_argument('--domains', default='', help='comma-separated ROS_DOMAIN_IDs for parallel runs '
                                                  '(default 71, 72, ... when --jobs > 1)')
    ap.add_argument('--watch', action='store_true',
                    help='parallel runs: the first slot uses web port 18097 and Foxglove on 18765')
    ap.add_argument('--web-auth', default='', help='auth.json (from set_password) to give every run\'s web app')
    ap.add_argument('--now-file', default='', help='--watch: write what the watched slot runs (and since when) here')
    ap.add_argument('--ws', default='', help='workspace whose scripts/run_twin2d.sh (and install) to run '
                                             '(default: this one; e.g. a build of an older version for a before/after)')
    a = ap.parse_args()

    def on_term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    out_root = a.out or time.strftime('/data/twin/twin2d/runs/batch_%Y%m%d_%H%M%S')
    os.makedirs(out_root, exist_ok=True)
    scenarios = [scenario_path(s) for s in a.scenarios]
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()] or [-1]
    domains = [int(d) for d in a.domains.split(',') if d.strip()]
    if a.jobs > 1 and not domains:
        domains = list(range(71, 71 + a.jobs))
    if domains and len(domains) < a.jobs:
        sys.exit(f'--jobs {a.jobs} needs {a.jobs} --domains')
    slots = queue.Queue()
    for k, d in enumerate(domains[:max(a.jobs, 1)]):
        slots.put({'domain': d, 'watch': a.watch and k == 0})
    if not domains:
        slots.put(None)
    jobs = [(sc, seed) for sc in scenarios for seed in seeds]
    done = [None] * len(jobs)
    lock = threading.Lock()
    launch_args = shlex.split(a.launch_args)

    def write_summary():
        finished = [d for d in done if d is not None]
        rows = [row(stem, seed, result) for stem, seed, result in finished]
        with open(os.path.join(out_root, 'summary.json'), 'w') as f:
            json.dump({'rows': rows, 'runs': [{'scenario': s, 'seed': sd, 'result': r} for s, sd, r in finished]},
                      f, indent=1)
        with open(os.path.join(out_root, 'summary.md'), 'w') as f:
            f.write(markdown(rows))
        return rows

    def worker(i):
        sc, seed = jobs[i]
        slot = slots.get()
        try:
            stem, seed_used, result = run_job(sc, seed, out_root, launch_args, a.sample_s, slot, a)
        finally:
            slots.put(slot)
        with lock:
            done[i] = (stem, seed_used, result)
            print('   ' + json.dumps(row(stem, seed_used, result)), flush=True)
            write_summary()

    threads = []
    try:
        for i in range(len(jobs)):
            while sum(t.is_alive() for t in threads) >= max(a.jobs, 1):
                time.sleep(0.5)
            t = threading.Thread(target=worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(2.0)   # stagger the launches
        for t in threads:
            while t.is_alive():
                t.join(0.5)
    except KeyboardInterrupt:
        print('   interrupted: stopping the runs', flush=True)
        for sid in list(ACTIVE):
            kill_session(sid)
        raise
    rows = write_summary()
    print()
    print(markdown(rows))


if __name__ == '__main__':
    main()
