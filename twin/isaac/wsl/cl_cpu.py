#!/usr/bin/env python3
"""CPU use of this closed-loop run's processes only (other stacks may share the WSL VM).

  cl_cpu.py RUN_DIR [--every 10]

Every N s: CPU time of all descendants of the pids in RUN_DIR/pids (from /proc, so exact, not top's snapshot),
grouped by process name, as % of one core; plus the whole VM's busy %. Appends JSON lines to RUN_DIR/logs/cpu.jsonl;
stops when RUN_DIR/pids disappears or on SIGTERM.
"""
import json
import os
import sys
import time

HZ = os.sysconf('SC_CLK_TCK')


def procs():
    out = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            s = open(f'/proc/{d}/stat').read()
        except OSError:
            continue
        name = s[s.index('(') + 1:s.rindex(')')]
        f = s[s.rindex(')') + 2:].split()
        out[int(d)] = (int(f[1]), name, int(f[11]) + int(f[12]))
    return out


def vm_busy():
    f = [int(x) for x in open('/proc/stat').readline().split()[1:]]
    return sum(f) - f[3] - f[4], sum(f)


def main():
    run = sys.argv[1]
    every = float(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 10.0
    os.makedirs(os.path.join(run, 'logs'), exist_ok=True)
    log = open(os.path.join(run, 'logs', 'cpu.jsonl'), 'a')
    prev, prev_vm, prev_t = {}, vm_busy(), time.monotonic()
    ncpu = os.cpu_count()
    first = True
    while os.path.exists(os.path.join(run, 'pids')):
        time.sleep(every)
        try:
            roots = {int(x) for x in open(os.path.join(run, 'pids')).read().split() if x.strip().isdigit()}
        except OSError:
            break
        P = procs()
        kids = {}
        for pid, (ppid, _, _) in P.items():
            kids.setdefault(ppid, []).append(pid)
        mine, todo = set(), [r for r in roots if r in P]
        while todo:
            p = todo.pop()
            if p not in mine:
                mine.add(p)
                todo.extend(kids.get(p, []))
        now = time.monotonic()
        dt = now - prev_t
        by = {}
        for p in mine:
            ticks = P[p][2]
            d = ticks - prev[p] if p in prev else (0 if first else ticks)   # new process: all its time is recent
            by[P[p][1]] = by.get(P[p][1], 0.0) + d / HZ / dt * 100
        prev = {p: P[p][2] for p in mine}
        b, t = vm_busy()
        vm = (b - prev_vm[0]) / max(t - prev_vm[1], 1) * 100
        prev_vm, prev_t = (b, t), now
        if first:
            first = False
            continue
        log.write(json.dumps({'wall': time.time(), 'n_procs': len(mine), 'stack_pct_one_core': round(sum(by.values()), 1),
                              'vm_busy_pct': round(vm, 1), 'ncpu': ncpu,
                              'by_name': {k: round(v, 1) for k, v in sorted(by.items(), key=lambda kv: -kv[1]) if v >= 1}}) + '\n')
        log.flush()


if __name__ == '__main__':
    main()
