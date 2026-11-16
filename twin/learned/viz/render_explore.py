"""Render the exploration comparison video (explorer rule vs learned frontier ranker) from runs
recorded by record_explore.py: the map revealing over time, frontiers, the candidate goals with
their scores, the chosen goal and its path, the drive home, and a coverage-vs-time inset.
1920 x 1080, H.264 (yuv420p, faststart).

  python render_explore.py --runs E:/twin/learned/runs/video/explore_up3.pkl E:/twin/learned/runs/video/explore_h56.pkl \
      --out E:/twin/videos/learned/explore_rule_vs_learned.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Wedge  # noqa: E402

FPS = 30
W_PX, H_PX = 1920, 1080
BG = "#14181d"
SIDE_COL = ["#ffcc80", "#80d8ff"]
SIDE_NAME = ["RULE \u2014 the explorer today", "LEARNED \u2014 re-ranks the same candidates"]
SIDE_SUB = ["picks the lowest score = path distance \u2212 0.2 \u00d7 frontier size", "CEM + PPO ranker over the map crop and candidate features"]
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

C_UNK = np.array([0.56, 0.60, 0.65])
C_UNK_WALL = np.array([0.45, 0.49, 0.54])
C_FREE = np.array([0.98, 0.98, 0.97])
C_OCC = np.array([0.15, 0.20, 0.22])
C_FRONT = np.array([1.0, 0.70, 0.0])


def fmt_t(t):
    return f"{int(t // 60):2d}:{int(t % 60):02d}"


class Side:
    """Replays one controller's run: belief, pose, current decision."""

    def __init__(self, run, e):
        self.run, self.e = run, e
        self.tl = run["timeline"][e]  # (t, x, y, yaw, n_diffs, decision)
        self.diffs = run["diffs"][e]
        self.dec = run["decisions"][e]
        self.dec_t = np.array([d["t"] for d in self.dec])
        self.H = run["H"]
        self.reach = run["reach0"].reshape(-1)
        self.area = max(self.reach.sum(), 1)
        self.belief = run["belief0"][e].reshape(-1).astype(np.int8).copy()
        self.applied = 0
        # coverage / distance at every timeline entry
        b = self.belief.copy()
        cov, k = [], 0
        for row in self.tl:
            n = int(row[4])
            while k < n:
                idx, val = self.diffs[k]
                b[idx] = val
                k += 1
            cov.append(((b == 0) & self.reach).sum() / self.area)
        self.cov = np.minimum(np.array(cov), 1.0)
        xy = self.tl[:, 1:3]
        self.cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        self.t_end = float(self.tl[-1, 0])
        self.final = run["final"][e]
        t90 = np.nonzero(self.cov >= 0.9)[0]
        self.t90 = float(self.tl[t90[0], 0]) if len(t90) else None

    def at(self, t):
        tl = self.tl
        i = int(np.searchsorted(tl[:, 0], t, side="right")) - 1
        i = max(0, min(i, len(tl) - 1))
        n = int(tl[i, 4])
        while self.applied < n:
            idx, val = self.diffs[self.applied]
            self.belief[idx] = val
            self.applied += 1
        j = min(i + 1, len(tl) - 1)
        w = 0.0 if j == i or tl[j, 0] <= tl[i, 0] else float(np.clip((t - tl[i, 0]) / (tl[j, 0] - tl[i, 0]), 0, 1))
        x = tl[i, 1] + w * (tl[j, 1] - tl[i, 1])
        y = tl[i, 2] + w * (tl[j, 2] - tl[i, 2])
        dy = math.atan2(math.sin(tl[j, 3] - tl[i, 3]), math.cos(tl[j, 3] - tl[i, 3]))
        yaw = tl[i, 3] + w * dy
        k = int(np.searchsorted(self.dec_t, t, side="right")) - 1
        dist = self.cum[i] + w * (self.cum[j] - self.cum[i])
        cov = self.cov[i]
        return x, y, yaw, (self.dec[k] if k >= 0 else None), k, i, dist, cov


def frontier_mask(b, H):
    g = b.reshape(H, H)
    unk = np.pad(g < 0, 1, constant_values=True)
    n4 = unk[:-2, 1:-1] | unk[2:, 1:-1] | unk[1:-1, :-2] | unk[1:-1, 2:]
    return (g == 0) & n4


class Panel:
    def __init__(self, fig, rect, run, e, bbox):
        self.s = Side(run, e)
        self.run, self.e = run, e
        self.ax = ax = fig.add_axes(rect)
        H, res = run["H"], run["res"]
        self.res = res
        self.occ = run["occ"].reshape(-1)
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(SIDE_COL[e])
            sp.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
        x0, y0, x1, y1 = bbox
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        self.im = ax.imshow(np.zeros((H, H, 3)), extent=(0, H * res, 0, H * res), origin="lower", interpolation="nearest", zorder=1)
        hx, hy = run["home"]
        ax.plot([hx], [hy], marker="s", ms=13, color="#2e7d32", mec="white", mew=1.5, zorder=6)
        ax.text(hx, hy - 0.55, "home", color="#1b5e20", fontsize=9, ha="center", va="top", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        self.trail, = ax.plot([], [], color="#1565c0", lw=1.6, alpha=0.75, zorder=3)
        self.plan, = ax.plot([], [], color="#d81b60", lw=2.2, ls=(0, (4, 2)), zorder=4)
        self.home_path, = ax.plot([], [], color="#2e7d32", lw=2.2, ls=(0, (4, 2)), zorder=4)
        self.cands = ax.scatter([], [], s=[], zorder=5)
        self.choice, = ax.plot([], [], marker="*", ms=22, color="#d81b60", mec="white", mew=1.5, ls="none", zorder=7)
        self.labels = [ax.text(0, 0, "", fontsize=9, fontweight="bold", color="#212121", ha="left", va="bottom", zorder=8,
                               bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8)) for _ in range(5)]
        self.fov = Wedge((0, 0), 3.0, 0, 1, facecolor="#b3e5fc", alpha=0.35, lw=0, zorder=2)
        ax.add_patch(self.fov)
        self.robot = Circle((0, 0), 0.25, facecolor="#1e88e5", edgecolor="white", lw=1.5, zorder=9)
        ax.add_patch(self.robot)
        self.head, = ax.plot([], [], color="white", lw=2, zorder=10)
        self.note = ax.text(0.015, 0.985, "", transform=ax.transAxes, fontsize=11, color="white", ha="left", va="top", zorder=12,
                            bbox=dict(boxstyle="round,pad=0.3", fc="#263238", ec="none", alpha=0.88))
        self.done = ax.text(0.5, 0.5, "", transform=ax.transAxes, fontsize=22, color="white", ha="center", va="center", fontweight="bold", zorder=13,
                            bbox=dict(boxstyle="round,pad=0.5", fc="#2e7d32", ec="none", alpha=0.9))
        self.trail_pts = []

    def update(self, t):
        s = self.s
        x, y, yaw, dec, k, i, dist, cov = s.at(t)
        H = s.H
        b = s.belief
        img = np.empty((H * H, 3))
        unk = b < 0
        img[:] = C_FREE
        img[unk] = C_UNK
        img[unk & self.occ] = C_UNK_WALL
        img[b == 1] = C_OCC
        fm = frontier_mask(b, H).reshape(-1)
        img[fm] = C_FRONT
        self.im.set_data(img.reshape(H, H, 3))
        tl = s.tl[: i + 1]
        self.trail.set_data(np.r_[tl[:, 1], x], np.r_[tl[:, 2], y])
        self.robot.center = (x, y)
        self.head.set_data([x, x + 0.4 * math.cos(yaw)], [y, y + 0.4 * math.sin(yaw)])
        self.fov.set_center((x, y))
        self.fov.set_theta1(math.degrees(yaw) - 42.5)
        self.fov.set_theta2(math.degrees(yaw) + 42.5)
        finished = t >= s.t_end
        for lb in self.labels:
            lb.set_visible(False)
        self.home_path.set_data([], [])
        if dec is not None and dec["choice"] >= 0 and not finished:
            m = dec["mask"]
            g = dec["goal"][m]
            if self.e == 0:
                order = np.argsort(dec["score"][m])
                vals = [f"{v:.1f}" for v in dec["score"][m]]
            else:
                order = np.argsort(-dec["prob"][m])
                vals = [f"{100 * v:.0f}%" for v in dec["prob"][m]]
            sizes = np.full(len(g), 60.0)
            cols = np.tile(np.array([[1, 1, 1, 0.95]]), (len(g), 1))
            self.cands.set_offsets(g)
            self.cands.set_sizes(sizes)
            self.cands.set_facecolors(cols)
            self.cands.set_edgecolors("#37474f")
            self.cands.set_linewidths(1.5)
            for r, j in enumerate(order[:5]):
                lb = self.labels[r]
                lb.set_position((g[j, 0] + 0.22, g[j, 1] + 0.12))
                lb.set_text(vals[j])
                lb.set_color("#ad1457" if r == 0 else "#212121")
                lb.set_visible(True)
            ch = dec["goal"][dec["choice"]]
            self.choice.set_data([ch[0]], [ch[1]])
            plan = dec["plan"]
            self.plan.set_data(plan[:, 1] * self.res + self.res / 2, plan[:, 0] * self.res + self.res / 2)
            n_c = int(m.sum())
            if self.e == 0:
                txt = f"decision {k + 1}: {n_c} candidates, picks the lowest score"
            else:
                rule_rank = int(np.argsort(dec["score"][m]).tolist().index(int(np.nonzero(np.nonzero(m)[0] == dec["choice"])[0][0]))) + 1
                txt = f"decision {k + 1}: {n_c} candidates, picks the rule's #{rule_rank}" + ("" if rule_rank == 1 else "  \u2190 differs")
            self.note.set_text(txt)
        else:
            self.cands.set_offsets(np.zeros((0, 2)))
            self.choice.set_data([], [])
            self.plan.set_data([], [])
            if dec is not None and "home_path" in dec:
                hp = dec["home_path"]
                self.home_path.set_data(hp[:, 0], hp[:, 1])
            self.note.set_text("no frontiers left: explorer finishes and drives home" if dec is not None else "")
        if finished:
            f = s.final
            self.done.set_text(f"DONE and home at {fmt_t(s.t_end)}\ncoverage {100 * f['cov']:.0f}%  \u00b7  {f['dist']:.0f} m driven")
            self.done.set_visible(True)
        else:
            self.done.set_visible(False)
        return dist, cov


def bbox_of(run, margin=1.0):
    H, res = run["H"], run["res"]
    m = run["reach0"]
    rr, cc = np.nonzero(m)
    x0, x1 = cc.min() * res - margin, (cc.max() + 1) * res + margin
    y0, y1 = rr.min() * res - margin, (rr.max() + 1) * res + margin
    return x0, y0, x1, y1


def build(run, title, sub, speed):
    fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100, facecolor=BG)
    fig.text(0.015, 0.966, title, color="white", fontsize=23, fontweight="bold", va="center")
    fig.text(0.015, 0.93, sub, color="#b0bec5", fontsize=13, va="center")
    clock = fig.text(0.985, 0.962, "", color="white", fontsize=28, fontweight="bold", ha="right", va="center", family="Consolas")
    fig.text(0.985, 0.928, f"simulated time, played {speed:.0f}\u00d7 faster", color="#90a4ae", fontsize=11.5, ha="right", va="center")
    bb = bbox_of(run)
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    avail_w, avail_h = 905, 700
    sc = min(avail_w / bw, avail_h / bh)
    pw, ph = bw * sc, bh * sc
    top = 1080 - 150
    panels, stats = [], []
    for e in range(2):
        cx = 480 + 960 * e
        rect = [(cx - pw / 2) / W_PX, (top - ph) / H_PX, pw / W_PX, ph / H_PX]
        fig.text((cx - 452) / W_PX, (top + 26) / H_PX, SIDE_NAME[e], color=SIDE_COL[e], fontsize=14.5, fontweight="bold", va="center")
        fig.text((cx + 452) / W_PX, (top + 26) / H_PX, SIDE_SUB[e], color="#90a4ae", fontsize=10.5, ha="right", va="center")
        panels.append(Panel(fig, rect, run, e, bb))
        stats.append(fig.text(cx / W_PX, (top - ph - 22) / H_PX, "", color="white", fontsize=13, ha="center", va="center"))
    # coverage vs time inset
    iax = fig.add_axes([0.06, 0.035, 0.6, 0.13])
    iax.set_facecolor("#1d232a")
    for sp in iax.spines.values():
        sp.set_color("#455a64")
    tmax = max(p.s.t_end for p in panels) * 1.02
    iax.set_xlim(0, tmax / 60)
    iax.set_ylim(0, 1.0)
    iax.set_ylabel("coverage", color="#b0bec5", fontsize=10)
    iax.set_xlabel("simulated minutes", color="#b0bec5", fontsize=10, labelpad=1)
    iax.tick_params(colors="#90a4ae", labelsize=9)
    iax.axhline(0.9, color="#607d8b", lw=0.8, ls=":")
    iax.text(tmax / 60 * 0.005, 0.92, "90 %", color="#90a4ae", fontsize=8.5, va="bottom")
    curves = [iax.plot([], [], color=SIDE_COL[e], lw=2.2, label=["rule", "learned"][e])[0] for e in range(2)]
    iax.legend(loc="lower right", fontsize=9.5, facecolor="#263238", edgecolor="none", labelcolor="white")
    vline = iax.axvline(0, color="white", lw=1, alpha=0.6)
    side_txt = fig.text(0.69, 0.10, "", color="white", fontsize=12.5, va="center", linespacing=1.5, family="Consolas")

    def update(t):
        clock.set_text(fmt_t(t))
        rows = []
        for e, p in enumerate(panels):
            dist, cov = p.update(t)
            s = p.s
            if t >= s.t_end:  # the env's own totals once home (the drive home is only drawn approximately)
                dist, cov = s.final["dist"], s.final["cov"]
            n = int(np.searchsorted(s.tl[:, 0], t, side="right"))
            curves[e].set_data(np.r_[s.tl[:n, 0], min(t, s.t_end)] / 60, np.r_[s.cov[:n], cov])
            done = t >= s.t_end
            stats[e].set_text(f"coverage {100 * cov:4.1f}%   \u00b7   driven {dist:5.1f} m   \u00b7   goals {sum(1 for d in s.dec if d['t'] <= t and d['choice'] >= 0)}"
                              + ("   \u00b7   home" if done else ""))
            t90 = f"{fmt_t(s.t90)}" if s.t90 is not None and s.t90 <= t else "  -  "
            rows.append((["rule   ", "learned"][e], 100 * cov, dist, t90, fmt_t(s.t_end) if done else "  -  "))
        vline.set_xdata([t / 60, t / 60])
        side_txt.set_text("           coverage  driven  90% at  home at\n" + "\n".join(
            f"{r[0]}    {r[1]:5.1f}%  {r[2]:5.0f} m   {r[3]}   {r[4]}" for r in rows))

    return fig, update, max(p.s.t_end for p in panels)


def card(lines, extra=None):
    fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100, facecolor=BG)
    y = 0.66 if extra else 0.56
    for i, (txt, size, col, bold) in enumerate(lines):
        fig.text(0.5, y - i * 0.075, txt, color=col, fontsize=size, ha="center", va="center", fontweight="bold" if bold else "normal")
    if extra:
        extra(fig)
    return fig


def summary(evaljson):
    d = json.load(open(evaljson))

    def stats(key):
        r, l = d[key]["rule"]["summary"], d[key]["learned_net"]["summary"]
        return r, l

    def table(fig):
        sets = [("200 generated houses", "heldout_static"), ("real upstairs plan (16 starts)", "plan_upstairs_static"), ("real downstairs plan (16 starts)", "plan_downstairs_static")]
        xs = [0.18, 0.43, 0.51, 0.63, 0.71, 0.83, 0.91]
        y0 = 0.47
        hdr = ["", "rule", "learned", "rule", "learned", "rule", "learned"]
        for j, (nm, _) in enumerate(sets):
            fig.text((xs[1 + 2 * j] + xs[2 + 2 * j]) / 2, y0 + 0.05, nm, color="#b0bec5", fontsize=14, ha="center")
        rows = [("time incl. drive home", "time_s", lambda v: f"{v / 60:.1f} min"), ("distance driven", "dist_m", lambda v: f"{v:.0f} m"),
                ("final coverage", "coverage", lambda v: f"{100 * v:.1f}%"), ("time to 90 % (median)", "t90_s_median", lambda v: f"{v / 60:.1f} min" if v else "-"),
                ("re-driven share", "revisit_frac", lambda v: f"{100 * v:.0f}%")]
        for j, h in enumerate(hdr):
            if h:
                fig.text(xs[j], y0, h, color=SIDE_COL[(j + 1) % 2], fontsize=14, ha="center", fontweight="bold")
        for i, (nm, key, f) in enumerate(rows):
            y = y0 - 0.06 * (i + 1)
            fig.text(xs[0], y, nm, color="white", fontsize=15, ha="left", va="center")
            for j, (_, sk) in enumerate(sets):
                r, l = stats(sk)
                fig.text(xs[1 + 2 * j], y, f(r[key]), color="white", fontsize=15, ha="center", va="center")
                fig.text(xs[2 + 2 * j], y, f(l[key]), color="white", fontsize=15, ha="center", va="center", fontweight="bold")
        fig.text(0.5, 0.06, "Static doors, no people, full battery; same start pose for both. With doors closing and people walking, failed goals dropped 29 %.\n"
                 "Simulation only: the learned ranker only re-orders the explorer's own candidates, so all of its safety rules stay in place.",
                 color="#90a4ae", fontsize=12.5, ha="center", linespacing=1.6)

    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=["the real upstairs floor plan", "a generated house (held out)"])
    ap.add_argument("--seconds", nargs="+", type=float, default=[22.0, 17.0])
    ap.add_argument("--eval", default="E:/twin/learned/runs/explore_eval_net/explore_eval.json")
    ap.add_argument("--out", default="E:/twin/videos/learned/explore_rule_vs_learned.mp4")
    ap.add_argument("--preview", type=float, default=-1.0, help="fraction of run 0 to write as PNG, then exit")
    a = ap.parse_args()
    runs = [pickle.load(open(p, "rb")) for p in a.runs]
    title = "Smarter exploration: which frontier next?"
    if a.preview >= 0:
        r = runs[0]
        tend = max(r["final"][0]["t"], r["final"][1]["t"])
        fig, upd, tend = build(r, title, f"{a.names[0]} \u00b7 preview", 30)
        upd(a.preview * tend)
        fig.savefig(a.out.replace(".mp4", f"_preview_{a.preview:.2f}.png"), facecolor=BG)
        return
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W_PX}x{H_PX}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    n = 0

    def emit(fig, frames=1):
        nonlocal n
        fig.canvas.draw()
        buf = bytes(fig.canvas.buffer_rgba())
        for _ in range(frames):
            ff.stdin.write(buf)
            n += 1

    intro = card([
        ("Smarter exploration: which frontier should the Roomba map next?", 30, "white", True),
        ("Nav2 still does all the driving. Only the choice among the explorer's own candidate goals changes.", 18, "#b0bec5", False),
        ("Left: the explorer's rule today (nearest / biggest frontier).   Right: the learned ranker.", 18, "#b0bec5", False),
        ("Same house, same start, same sensing (85\u00b0 depth view, 3 m range, 0.3 m blind zone). 2D GPU simulation.", 15, "#90a4ae", False),
        ("grey = not yet seen \u00b7 white = mapped floor \u00b7 dark = walls \u00b7 amber = frontier \u00b7 circles = candidates (label: score / learned %) \u00b7 pink star = chosen goal", 13.5, "#90a4ae", False),
    ])
    emit(intro, int(4.5 * FPS))
    plt.close(intro)
    for r, name, secs in zip(runs, a.names, a.seconds):
        tend = max(r["final"][0]["t"], r["final"][1]["t"])
        speed = tend / secs
        speed = float(np.ceil(speed / 5) * 5)
        c = card([(name[0].upper() + name[1:], 34, "white", True)])
        emit(c, int(1.5 * FPS))
        plt.close(c)
        fig, upd, tend = build(r, title, f"{name[0].upper() + name[1:]}  \u00b7  rule vs learned, same start  \u00b7  static doors, no people", speed)
        nfr = int(math.ceil(tend / speed * FPS)) + 1
        for f in range(nfr):
            upd(min(f * speed / FPS, tend))
            emit(fig)
        emit(fig, int(3.0 * FPS))
        plt.close(fig)
        print("run done", name, n, flush=True)
    end = card([("Held-out evaluation (rule vs learned, paired)", 30, "white", True)], extra=summary(a.eval))
    emit(end, int(6.5 * FPS))
    plt.close(end)
    ff.stdin.close()
    ff.wait()
    print(f"{a.out}: {n} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
