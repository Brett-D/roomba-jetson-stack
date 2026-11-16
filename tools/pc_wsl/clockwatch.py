"""Watches CLOCK_REALTIME against CLOCK_MONOTONIC: reports steps and the slew rate."""
import sys
import time

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60
r0, m0 = time.clock_gettime_ns(time.CLOCK_REALTIME), time.clock_gettime_ns(time.CLOCK_MONOTONIC)
pr, pm = r0, m0
steps = []
behind = 0
n = 0
end = m0 + int(secs * 1e9)
while True:
    time.sleep(0.005)
    r, m = time.clock_gettime_ns(time.CLOCK_REALTIME), time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    d = (r - pr) - (m - pm)
    n += 1
    if abs(d) > 200_000:
        steps.append((round((m - m0) / 1e9, 2), round(d / 1e6, 3), round(r / 1e9, 2)))
        if '--stream' in sys.argv:
            print(f'step {d / 1e6:+.1f} ms at realtime {r / 1e9:.2f}', flush=True)
    if d < 0:
        behind += 1
    pr, pm = r, m
    if m > end:
        break
drift = ((r - r0) - (m - m0)) / 1e6
print(f'{secs:.0f}s: realtime-monotonic drift {drift:+.3f} ms ({drift / secs * 1000:+.1f} ppm); '
      f'{behind}/{n} samples where realtime advanced less than monotonic; steps >0.2 ms: {steps[:10]}')
