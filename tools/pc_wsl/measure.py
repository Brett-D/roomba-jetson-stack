"""CPU and memory of process groups (one per smoke stack): measure.py <secs> <pgid> [<pgid> ...]"""
import os
import sys
import time

TICK = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')


def sample(groups):
    cpu = {g: 0 for g in groups}
    rss = {g: 0 for g in groups}
    procs = {g: 0 for g in groups}
    top = {g: {} for g in groups}
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/stat') as f:
                s = f.read()
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read().split(b'\0')
        except OSError:
            continue
        rest = s[s.rfind(')') + 2:].split()
        pgrp = int(rest[2])
        if pgrp in cpu:
            t = int(rest[11]) + int(rest[12])
            cpu[pgrp] += t
            rss[pgrp] += int(rest[21]) * PAGE
            procs[pgrp] += 1
            name = os.path.basename(cmd[0].decode(errors='replace')) if cmd and cmd[0] else rest[0]
            if name.startswith('python') and len(cmd) > 1:
                name = os.path.basename(cmd[1].decode(errors='replace'))
            top[pgrp][f'{pid}:{name}'] = t
    return cpu, rss, procs, top


def main():
    secs = float(sys.argv[1])
    groups = [int(g) for g in sys.argv[2:]]
    c0, _, _, t0 = sample(groups)
    w0 = time.time()
    time.sleep(secs)
    c1, rss, procs, t1 = sample(groups)
    dt = time.time() - w0
    total_cpu = total_rss = 0
    for g in groups:
        cpu = 100.0 * (c1[g] - c0[g]) / TICK / dt
        total_cpu += cpu
        total_rss += rss[g]
        per = sorted(((100.0 * (v - t0[g].get(k, v)) / TICK / dt, k.split(':', 1)[1]) for k, v in t1[g].items()),
                     reverse=True)[:6]
        print(f'pgid {g}: {procs[g]} procs, CPU {cpu:.0f}% (of one core), RSS {rss[g] / 2**20:.0f} MiB; top: '
              + ', '.join(f'{n} {c:.0f}%' for c, n in per), flush=True)
    load = os.getloadavg()
    with open('/proc/meminfo') as f:
        mem = {l.split(':')[0]: int(l.split()[1]) for l in f}
    print(f'TOTAL {len(groups)} stacks: CPU {total_cpu:.0f}% of one core = {total_cpu / os.cpu_count():.1f}% of '
          f'{os.cpu_count()} threads; RSS {total_rss / 2**30:.2f} GiB; loadavg {load[0]:.1f}; '
          f'MemAvailable {mem["MemAvailable"] / 2**20:.1f} GiB', flush=True)


if __name__ == '__main__':
    main()
