"""Training curves and evaluation plots for docs/learned-skills.md.

  python plot_curves.py --unstuck E:/twin/learned/runs/unstuck/main --cem E:/twin/learned/runs/explore_cem/main/cem.csv
                        --explore_eval E:/twin/learned/runs/explore_eval/explore_eval.json --out ../../docs/learned-skills
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def tb_scalars(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ev = EventAccumulator(glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))[0], size_guidance={"scalars": 0})
    ev.Reload()
    return {t: np.array([(s.step, s.value) for s in ev.Scalars(t)]) for t in ev.Tags()["scalars"]}


def smooth(y, k=9):
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def plot_unstuck(run_dir, out):
    s = tb_scalars(run_dir)
    keys = [("Train/mean_reward", "mean episode reward"), ("Episode/success", "success rate (train tiles)"),
            ("Episode/bumps", "bumper hits per episode"), ("Episode/time_s", "episode time (s)")]
    keys = [(k, t) for k, t in keys if k in s]
    fig, axs = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.2))
    for ax, (k, title) in zip(np.atleast_1d(axs), keys):
        d = s[k]
        ax.plot(d[:, 0], d[:, 1], color="#90caf9", lw=0.8)
        ax.plot(d[:, 0], smooth(d[:, 1]), color="#1565c0", lw=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PPO iteration (4096 envs x 24 steps)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out, "unstuck_training.png")
    fig.savefig(p, dpi=110)
    print(p, sorted(s))


def plot_cem(path, out):
    rows = list(csv.DictReader(open(path)))
    g = np.array([int(r["gen"]) for r in rows])
    fr = np.array([float(r["fit_rule"]) for r in rows])
    fm = np.array([float(r["fit_mu"]) for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    axs[0].plot(g, fr, "o-", color="#9e9e9e", ms=3, label="explorer rule")
    axs[0].plot(g, fm, "o-", color="#2e7d32", ms=3, label="CEM mean")
    axs[0].set_title("fitness on the generation's houses", fontsize=10)
    axs[0].set_xlabel("generation")
    axs[0].legend(fontsize=8)
    axs[1].plot(g, fm - fr, "o-", color="#2e7d32", ms=3)
    axs[1].plot(g, smooth(fm - fr, 5), color="#1b5e20", lw=1.6)
    axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_title("CEM mean - rule (same houses)", fontsize=10)
    axs[1].set_xlabel("generation")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out, "explore_training.png")
    fig.savefig(p, dpi=110)
    print(p)


def plot_explore_eval(path, out):
    d = json.load(open(path))
    sets = [k for k in d]
    fig, axs = plt.subplots(2, (len(sets) + 1) // 2, figsize=(4.2 * ((len(sets) + 1) // 2), 6.4))
    t = np.arange(0, 1801, 30) / 60
    colors = {"rule": "#757575", "learned_linear": "#2e7d32", "learned_net": "#1565c0"}
    for ax, k in zip(axs.ravel(), sets):
        for name, r in d[k].items():
            c = np.mean([e["curve"] for e in r["episodes"]], 0)
            ax.plot(t, c, color=colors.get(name, None), lw=1.6, label=name)
        ax.set_title(k.replace("_", " "), fontsize=10)
        ax.set_xlabel("minutes")
        ax.set_ylabel("coverage")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out, "explore_coverage.png")
    fig.savefig(p, dpi=110)
    print(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unstuck", default="")
    ap.add_argument("--cem", default="")
    ap.add_argument("--explore_eval", default="")
    ap.add_argument("--out", default="docs/learned-skills")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.unstuck:
        plot_unstuck(a.unstuck, a.out)
    if a.cem:
        plot_cem(a.cem, a.out)
    if a.explore_eval:
        plot_explore_eval(a.explore_eval, a.out)
