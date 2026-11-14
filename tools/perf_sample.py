#!/usr/bin/env python3
"""CPU / GPU / RAM usage of the robot stack over a time window (Jetson).

  python3 perf_sample.py [seconds] [--json out.json] [--interval s]

Per-process CPU comes from /proc/<pid>/stat (100% = one core), total CPU from /proc/stat
(100% = all 6 cores), GPU load (GR3D_FREQ) and RAM from tegrastats. Processes are grouped by
a short name (node executable or ROS node name) and summed when there are several.
"""
import json
import os
import re
import subprocess
import sys
import time

HZ = os.sysconf('SC_CLK_TCK')
NCPU = os.cpu_count()
# Only report processes that belong to the stack (and the bag player/recorder).
KEEP = re.compile(r'ros2|/opt/ros/|roomba_ws|component_container|python3 .*(ros|roomba)')


def cpu_total():
    with open('/proc/stat') as f:
        vals = [int(v) for v in f.readline().split()[1:]]
    idle = vals[3] + vals[4]
    return sum(vals), idle


def proc_name(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            args = [a.decode(errors='replace') for a in f.read().split(b'\0') if a]
    except OSError:
        return None
    if not args:
        return None
    cmd = ' '.join(args)
    if not KEEP.search(cmd) or 'perf_sample' in cmd:
        return None
    m = re.search(r'__node:=(\S+)', cmd)
    if m:
        return m.group(1)
    for a in args:
        if '/lib/' in a or '/bin/' in a:
            base = os.path.basename(a)
            if base not in ('python3', 'bash', 'sh', 'nice'):
                if base == 'ros2' and len(args) > 2:
                    return 'ros2 ' + ' '.join(args[2:4]) if 'bag' in args else 'ros2 ' + args[2]
                return base
    return os.path.basename(args[0])


def proc_times():
    out = {}
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/stat') as f:
                st = f.read()
        except OSError:
            continue
        fields = st[st.rfind(')') + 2:].split()
        out[int(pid)] = int(fields[11]) + int(fields[12])
    return out


def main():
    args = sys.argv[1:]
    dur, out_json, interval = 30.0, None, 5.0
    while args:
        a = args.pop(0)
        if a == '--json':
            out_json = args.pop(0)
        elif a == '--interval':
            interval = float(args.pop(0))
        else:
            dur = float(a)
    teg = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=subprocess.PIPE, text=True)
    names = {}
    acc = {}
    cpu_busy = cpu_all = 0
    t_end = time.monotonic() + dur
    t0, i0 = cpu_total()
    p0 = proc_times()
    while time.monotonic() < t_end:
        time.sleep(min(interval, max(0.1, t_end - time.monotonic())))
        t1, i1 = cpu_total()
        p1 = proc_times()
        cpu_all += t1 - t0
        cpu_busy += (t1 - t0) - (i1 - i0)
        for pid, ticks in p1.items():
            if pid not in names:
                names[pid] = proc_name(pid)
            if names[pid] and pid in p0:
                acc[names[pid]] = acc.get(names[pid], 0) + ticks - p0[pid]
        t0, i0, p0 = t1, i1, p1
    teg.terminate()
    lines = teg.stdout.read().splitlines()
    gpu = [int(m.group(1)) for m in (re.search(r'GR3D_FREQ (\d+)%', ln) for ln in lines) if m]
    ram = [int(m.group(1)) for m in (re.search(r'RAM (\d+)/', ln) for ln in lines) if m]
    wall = cpu_all / NCPU / HZ  # seconds actually sampled
    procs = {k: round(100.0 * v / HZ / wall, 1) for k, v in acc.items() if v > 0}
    res = {
        'seconds': round(wall, 1),
        'cpu_total_pct': round(100.0 * cpu_busy / cpu_all, 1) if cpu_all else None,
        'cpu_cores_used': round(NCPU * cpu_busy / cpu_all, 2) if cpu_all else None,
        'stack_cpu_pct_of_core': round(sum(procs.values()), 1),
        'gpu_pct_mean': round(sum(gpu) / len(gpu), 1) if gpu else None,
        'gpu_pct_max': max(gpu) if gpu else None,
        'ram_mb_mean': round(sum(ram) / len(ram)) if ram else None,
        'processes': dict(sorted(procs.items(), key=lambda kv: -kv[1])),
    }
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(res, f, indent=1)
    print(f"{res['seconds']} s: CPU {res['cpu_total_pct']}% of {NCPU} cores "
          f"({res['cpu_cores_used']} cores), GPU mean {res['gpu_pct_mean']}% max {res['gpu_pct_max']}%, "
          f"RAM {res['ram_mb_mean']} MB")
    for k, v in res['processes'].items():
        if v >= 0.5:
            print(f'  {v:6.1f}%  {k}')


if __name__ == '__main__':
    main()
