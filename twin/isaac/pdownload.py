"""Resumable, parallel ranged download (for multi-GB wheels over a flaky link).

  python pdownload.py URL OUTFILE [--threads 8] [--chunk-mb 128]

Keeps whatever OUTFILE already holds, fetches the rest in chunks (each chunk resumes on error),
then appends the chunks in order.
"""
import argparse
import os
import sys
import threading
import time
import urllib.request


def total_size(url):
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length'])


def fetch(url, start, end, part, stop):
    want = end - start + 1
    for attempt in range(200):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= want:
            return True
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start + have}-{end}'})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, 'ab') as f:
                while not stop.is_set():
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
        except Exception as e:  # noqa: BLE001
            time.sleep(min(2 + attempt, 20))
    return os.path.getsize(part) >= want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('out')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--chunk-mb', type=int, default=128)
    a = ap.parse_args()
    total = total_size(a.url)
    have = os.path.getsize(a.out) if os.path.exists(a.out) else 0
    print(f'total {total / 1e9:.3f} GB, have {have / 1e9:.3f} GB', flush=True)
    chunk = a.chunk_mb << 20
    jobs = [(s, min(s + chunk, total) - 1, f'{a.out}.part{i:04d}') for i, s in enumerate(range(have, total, chunk))]
    stop = threading.Event()
    sem = threading.Semaphore(a.threads)
    results = {}

    def run(job):
        with sem:
            results[job[2]] = fetch(a.url, *job, stop)

    threads = [threading.Thread(target=run, args=(j,), daemon=True) for j in jobs]
    for t in threads:
        t.start()
    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        got = sum(os.path.getsize(p) for _, _, p in jobs if os.path.exists(p))
        print(f'{(have + got) / 1e9:.3f} / {total / 1e9:.3f} GB  {got / 1e6 / (time.time() - t0):.1f} MB/s', flush=True)
    if not all(results.get(p) for _, _, p in jobs):
        print('some chunks failed; rerun to resume', flush=True)
        sys.exit(1)
    with open(a.out, 'ab') as f:
        for _, _, p in jobs:
            with open(p, 'rb') as src:
                while True:
                    b = src.read(1 << 24)
                    if not b:
                        break
                    f.write(b)
            os.remove(p)
    print('done', os.path.getsize(a.out), flush=True)


if __name__ == '__main__':
    main()
