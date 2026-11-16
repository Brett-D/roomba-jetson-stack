"""Render the unstuck comparison video (classical recovery vs learned policy) from the rollouts
recorded by record_unstuck.py: two 3 x 3 grids side by side, the same held-out stuck situations,
top view with obstacles, the robot footprint / heading, bumper hits, light-bumper and depth rays,
the costmap's lethal cells, outcome labels and a clock. 1920 x 1080, H.264 (yuv420p, faststart).

  python render_unstuck.py --dir E:/twin/learned/runs/video --out E:/twin/videos/learned/unstuck_classical_vs_learned.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "unstuck"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge  # noqa: E402

import scenarios as S  # noqa: E402

FPS = 30
W_PX, H_PX = 1920, 1080
BG = "#14181d"
FLOOR = "#ece8dc"
VIEW = 1.55
HFOV = math.radians(85.0)
FAN_COLS = 24
LIGHT_ANGLES = [1.2, 0.65, 0.2, -0.2, -0.65, -1.2]
CAM_X = 0.153
MODE_NAMES = {0: "following the path", 1: "recovery: spin +90\u00b0", 2: "recovery: spin \u2212180\u00b0", 3: "recovery: back up 10 cm", 4: "recovery: clear costmap + wait"}
KIND_LABEL = {
    "house_wall": "House \u00b7 nose to a wall",
    "house_furniture": "House \u00b7 nose to furniture",
    "house_corner": "House \u00b7 wedged in a corner",
    "house_railing": "House \u00b7 railing + stair drop",
    "house_door": "House \u00b7 at a door jamb",
    "overhang": "Wedged under furniture",
    "bump_ring": "Inside a ring of bump marks",
    "chair_legs": "Between chair legs",
    "narrow_gap": "In a narrow gap",
    "threshold": "Threshold / rug edges",
    "door_leaf": "Behind a door leaf",
    "clutter": "Clutter",
    "nose_wall": "Nose to a wall",
    "corner": "In a corner",
    "railing": "Invisible railing",
    "cliff_edge": "At a cliff edge",
}
ROUNDS = [
    ("Round 1 of 2: stuck spots cut from the real floor plans (held out)", [0, 9, 17, 14, 42, 47, 29, 53, 58]),
    ("Round 2 of 2: generated stuck situations (held out, never trained on)", [120, 181, 117, 158, 125, 134, 122, 118, 153]),
]
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


class Rollouts:
    def __init__(self, path):
        z = np.load(path)
        self.d = {k: z[k] for k in z.files}
        meta = json.load(open(path.replace(".npz", ".json")))
        self.eps = {r["tile"]: r for r in meta["episodes"]}
        self.env_of = {int(t): e for e, t in enumerate(self.d["tile"][0])}

    def get(self, tile):
        e = self.env_of[tile]
        r = self.eps[tile]
        k = r["end_step"]
        sl = {key: v[: k + 1, e] for key, v in self.d.items()}
        sl["lethal"] = np.unpackbits(sl["lethal"], axis=-1)[:, : S.GRID_N * S.GRID_N].reshape(-1, S.GRID_N, S.GRID_N)
        return sl, r


def prim_style(p):
    if p.z0 >= S.BLOCK_Z:
        return dict(facecolor="#90a4ae", alpha=0.22, zorder=9)  # high: the robot drives under it
    if p.z0 > S.BODY_TOP:
        return dict(facecolor="#ff9800", alpha=0.55, zorder=9, hatch="xx")  # overhang at camera height
    if p.z1 < S.STEP_BLOCK:
        return dict(facecolor="#a5d6c8", alpha=0.9, zorder=2)  # rug the robot drives over
    if p.z1 < 0.06:
        return dict(facecolor="#a1887f", alpha=1.0, zorder=2)  # threshold / step below the camera's obstacle band
    if p.depth == "none":
        return dict(facecolor="#ab47bc", alpha=1.0, zorder=4)  # invisible to the depth camera
    if p.depth == "sparse":
        return dict(facecolor="#ce93d8", alpha=1.0, zorder=4)
    return dict(facecolor="#37474f", alpha=1.0, zorder=4)


class Cell:
    def __init__(self, fig, rect, tile: S.Tile, data, ep, controller):
        self.ax = ax = fig.add_axes(rect)
        self.tile, self.d, self.ep, self.ctl = tile, data, ep, controller
        sx, sy, _ = tile.start
        self.c = (sx, sy)
        ax.set_xlim(sx - VIEW, sx + VIEW)
        ax.set_ylim(sy - VIEW, sy + VIEW)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(FLOOR)
        for sp in ax.spines.values():
            sp.set_linewidth(3)
            sp.set_color("#3a424c")
        for x0, y0, x1, y1 in tile.holes:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#ef9a9a", edgecolor="#c62828", hatch="///", lw=0.8, zorder=1))
        for p in tile.prims:
            st = prim_style(p)
            hatch = st.pop("hatch", None)
            ax.add_patch(Polygon(p.polygon(), closed=True, lw=0.6 if hatch else 0, edgecolor="#e65100" if hatch else "none", hatch=hatch, **st))
        self.lethal = ax.imshow(np.zeros((S.GRID_N, S.GRID_N, 4)), extent=(-S.TILE / 2, S.TILE / 2, -S.TILE / 2, S.TILE / 2), origin="lower",
                                interpolation="nearest", zorder=3)
        gx, gy = tile.goal
        self.goal_in = abs(gx - sx) < VIEW - 0.1 and abs(gy - sy) < VIEW - 0.1
        if self.goal_in:
            ax.plot([gx], [gy], marker="*", ms=16, color="#2e7d32", mec="white", mew=1.2, zorder=10)
        else:
            a = math.atan2(gy - sy, gx - sx)
            r = VIEW - 0.18
            ex, ey = sx + r * math.cos(a), sy + r * math.sin(a)
            ax.annotate("", xy=(ex + 0.12 * math.cos(a), ey + 0.12 * math.sin(a)), xytext=(ex - 0.12 * math.cos(a), ey - 0.12 * math.sin(a)),
                        arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=2.5, mutation_scale=16), zorder=10)
            ax.text(ex - 0.28 * math.cos(a), ey - 0.28 * math.sin(a), "goal", color="#2e7d32", fontsize=8.5, ha="center", va="center",
                    fontweight="bold", zorder=10)
        self.trail, = ax.plot([], [], color="#1565c0", lw=1.6, alpha=0.65, zorder=5)
        self.fan = LineCollection([], linewidths=1.3, zorder=6)
        ax.add_collection(self.fan)
        self.blindzone = Wedge((0, 0), 0.3, 0, 1, facecolor="#455a64", alpha=0.18, lw=0, zorder=6)
        ax.add_patch(self.blindzone)
        self.body = Circle((0, 0), S.ROBOT_R, facecolor="#1e88e5", edgecolor="white", lw=1.5, alpha=0.95, zorder=7)
        ax.add_patch(self.body)
        self.head, = ax.plot([], [], color="white", lw=2.2, zorder=8)
        self.cam = Circle((0, 0), 0.03, facecolor="#0d1b2a", lw=0, zorder=8)
        ax.add_patch(self.cam)
        self.bumper = Wedge((0, 0), S.ROBOT_R + 0.035, 0, 1, width=0.035, facecolor="#ff1744", lw=0, zorder=8)
        ax.add_patch(self.bumper)
        self.topring = Circle((0, 0), S.ROBOT_R + 0.06, fill=False, edgecolor="#d500f9", lw=3, zorder=8)
        ax.add_patch(self.topring)
        self.light = LineCollection([], linewidths=2.6, zorder=8)
        ax.add_collection(self.light)
        self.ring = Circle((0, 0), S.ROBOT_R + S.SUCCESS_CLEARANCE, fill=False, edgecolor="#00c853", lw=2, ls="--", zorder=10)
        ax.add_patch(self.ring)
        kw = dict(transform=ax.transAxes, zorder=12)
        ax.text(0.03, 0.97, KIND_LABEL.get(tile.kind, tile.kind), fontsize=10, color="white", ha="left", va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#263238", ec="none", alpha=0.85), **kw)
        self.badge = ax.text(0.97, 0.035, "", fontsize=11.5, color="white", ha="right", va="bottom", fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.3", fc="#455a64", ec="none"), **kw)
        self.status = ax.text(0.03, 0.035, "", fontsize=9, color="white", ha="left", va="bottom",
                              bbox=dict(boxstyle="round,pad=0.25", fc="#37474f", ec="none", alpha=0.8), **kw)
        self.warn = ax.text(0.03, 0.885, "", fontsize=9, color="#212121", ha="left", va="top", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.25", fc="#ffd54f", ec="none"), **kw)

    def update(self, t):
        d, ep = self.d, self.ep
        k = ep["end_step"]
        s = min(t / 0.1, k)
        i0 = int(math.floor(s))
        i1 = min(i0 + 1, k)
        w = s - i0
        x = (1 - w) * d["xy"][i0] + w * d["xy"][i1]
        dy = math.atan2(math.sin(d["yaw"][i1] - d["yaw"][i0]), math.cos(d["yaw"][i1] - d["yaw"][i0]))
        yaw = float(d["yaw"][i0] + w * dy)
        ended = t / 0.1 >= k and k < len(d["t"])
        c, sn = math.cos(yaw), math.sin(yaw)
        self.body.center = (x[0], x[1])
        self.head.set_data([x[0], x[0] + 0.17 * c], [x[1], x[1] + 0.17 * sn])
        cx, cy = x[0] + CAM_X * c, x[1] + CAM_X * sn
        self.cam.center = (cx, cy)
        tr = d["xy"][: i0 + 1]
        self.trail.set_data(np.r_[tr[:, 0], x[0]], np.r_[tr[:, 1], x[1]])
        lo = max(0, i0 - 2)
        hit = d["base_hit"][lo: i0 + 1].any() or d["bumped"][lo: i0 + 1].any()
        brg = float(d["hit_bearing"][i0]) if d["base_hit"][i0] else 0.0
        self.bumper.set_center((x[0], x[1]))
        self.bumper.set_theta1(math.degrees(yaw + brg) - 55)
        self.bumper.set_theta2(math.degrees(yaw + brg) + 55)
        self.bumper.set_visible(bool(hit))
        self.topring.center = (x[0], x[1])
        self.topring.set_visible(bool(d["top"][lo: i0 + 1].any()))
        # depth fan (per column: nearest obstacle return, else farthest floor return, else nothing)
        az = np.linspace(HFOV / 2, -HFOV / 2, FAN_COLS) + yaw
        ob, fr = d["obst_hd"][i0], d["free_hd"][i0]
        segs, cols = [], []
        for j in range(FAN_COLS):
            if ob[j] < 50:
                L = ob[j]
                col = (1.0, 0.70, 0.0, 0.95 if L < 0.7 else (0.55 if L < 1.3 else 0.28))
            elif fr[j] > 0:
                L, col = min(fr[j], 2.5), (0.30, 0.82, 0.88, 0.4)
            else:
                continue
            segs.append([(cx, cy), (cx + L * math.cos(az[j]), cy + L * math.sin(az[j]))])
            cols.append(col)
        self.fan.set_segments(segs)
        self.fan.set_colors(cols)
        self.blindzone.set_center((cx, cy))
        self.blindzone.set_theta1(math.degrees(yaw) - 42.5)
        self.blindzone.set_theta2(math.degrees(yaw) + 42.5)
        # light bumpers: 6 short IR rays coloured by proximity
        ir = d["ir"][i0]
        lsegs, lcols = [], []
        for j, la in enumerate(LIGHT_ANGLES):
            a = yaw + la
            ox, oy = x[0] + 0.155 * math.cos(a), x[1] + 0.155 * math.sin(a)
            lsegs.append([(ox, oy), (ox + 0.12 * math.cos(a), oy + 0.12 * math.sin(a))])
            v = float(np.clip(ir[j], 0, 1))
            lcols.append((0.4 + 0.6 * v, 0.73 * (1 - v) + 0.09 * v, 0.42 * (1 - v) + 0.27 * v, 0.95))
        self.light.set_segments(lsegs)
        self.light.set_colors(lcols)
        # costmap lethal cells (marks from the depth camera, bump keep-out strips, marks the spot started with)
        rgba = np.zeros((S.GRID_N, S.GRID_N, 4))
        rgba[d["lethal"][i0] > 0] = (0.90, 0.22, 0.21, 0.38)
        self.lethal.set_data(rgba)
        # labels
        outcome = ep["outcome"]
        warn = []
        if d["blind"][i0] and not ended:
            warn.append("camera blind (< 0.3 m): no forward")
        if d["reflex"][i0] and not ended:
            warn.append("driver bump reflex")
        self.warn.set_text("\n".join(warn))
        self.warn.set_visible(bool(warn))
        if ended:
            if outcome == "success":
                txt, fc = f"FREE in {ep['time_s']:.1f} s", "#2e7d32"
            elif outcome == "cliff":
                txt, fc = "FELL DOWN THE DROP", "#c62828"
            elif outcome == "fall":
                txt, fc = "TIPPED / FELL", "#c62828"
            else:
                txt, fc = "STILL STUCK after 15 s", "#c62828"
            self.badge.set_text(txt)
            self.badge.set_visible(True)
            self.badge.get_bbox_patch().set_facecolor(fc)
            for sp in self.ax.spines.values():
                sp.set_color(fc)
            self.ring.center = (x[0], x[1])
            self.ring.set_visible(outcome == "success")
            self.status.set_visible(False)
        else:
            self.badge.set_text("")
            self.badge.set_visible(False)
            for sp in self.ax.spines.values():
                sp.set_color("#3a424c")
            self.ring.set_visible(False)
            v, wz = d["cmd_out"][i0]
            if self.ctl == "classical":
                self.status.set_text(MODE_NAMES.get(int(d["mode"][i0]), ""))
            else:
                self.status.set_text(f"policy: v {v:+.2f} m/s  \u03c9 {wz:+.2f} rad/s")
            self.status.set_visible(True)
        bumps = int(d["bumps"][i0])
        return outcome == "success" and ended, ended, bumps


def legend(fig):
    rows = [
        [("patch", "#37474f", "wall / furniture"),
         ("patch", "#ab47bc", "invisible to the depth camera (railing posts, legs)"),
         ("hatch", "#ff9800", "overhang at camera height"),
         ("patch", "#a1887f", "threshold / step"),
         ("drop", "#ef9a9a", "drop (stairs)"),
         ("patch", "#e57373", "costmap lethal cells / bump marks"),
         ("star", "#2e7d32", "goal")],
        [("robot", "#1e88e5", "Roomba (0.34 m), white = heading, dot = camera"),
         ("line", "#ffb300", "depth ray hits obstacle"),
         ("line", "#4dd0e1", "depth ray sees floor"),
         ("line", "#ff1744", "light bumpers (green far, red close)"),
         ("arc", "#ff1744", "bumper hit"),
         ("ring", "#d500f9", "camera / top hit"),
         ("dash", "#00c853", "0.3 m clearance reached = free")],
    ]
    ax = fig.add_axes([0, 0, 1, 78 / H_PX])
    ax.set_xlim(0, W_PX)
    ax.set_ylim(0, 78)
    ax.axis("off")
    r = fig.canvas.get_renderer()
    for ri, items in enumerate(rows):
        yc = 56 - ri * 32
        x = 36
        for kind, col, label in items:
            if kind == "line":
                ax.plot([x, x + 26], [yc, yc], color=col, lw=3)
            elif kind == "arc":
                ax.add_patch(Wedge((x + 13, yc - 8), 15, 40, 140, width=4, color=col))
            elif kind == "ring":
                ax.add_patch(Circle((x + 13, yc), 10, fill=False, edgecolor=col, lw=2.5))
            elif kind == "dash":
                ax.add_patch(Circle((x + 13, yc), 10, fill=False, edgecolor=col, lw=2, ls="--"))
            elif kind == "star":
                ax.plot([x + 13], [yc], marker="*", ms=15, color=col, mec="white", mew=1)
            elif kind == "robot":
                ax.add_patch(Circle((x + 13, yc), 11, facecolor=col, edgecolor="white", lw=1.2))
                ax.plot([x + 13, x + 24], [yc, yc], color="white", lw=2)
            else:
                ax.add_patch(Rectangle((x, yc - 8), 26, 16, facecolor=col, edgecolor="#c62828" if kind == "drop" else ("#e65100" if kind == "hatch" else "none"),
                                       hatch="///" if kind == "drop" else ("xx" if kind == "hatch" else None), lw=0.8))
            t = ax.text(x + 34, yc, label, color="#cfd8dc", fontsize=10.5, va="center")
            x += 34 + t.get_window_extent(renderer=r).width + 30


def build_round(title, tiles, idx, cls, pol):
    fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100, facecolor=BG)
    fig.text(0.015, 0.968, "Getting unstuck: today's classical recovery vs the learned skill", color="white", fontsize=24, fontweight="bold", va="center")
    fig.text(0.015, 0.932, title + "  \u00b7  Isaac Lab (PhysX) rollouts, same start, same physics, same sensors", color="#b0bec5", fontsize=13.5, va="center")
    clock = fig.text(0.985, 0.955, "", color="white", fontsize=30, fontweight="bold", ha="right", va="center", family="Consolas")
    cell = 280
    gap = 8
    top = 1080 - 142
    sides = []
    for side, (x0, ctl, rol, head) in enumerate(((52, "classical", cls, "CLASSICAL \u2014 Nav2 spin / back up / clear + wait, driver bump reflex"),
                                                   (1012, "policy", pol, "LEARNED \u2014 PPO policy at 10 Hz, same sensors, same safety layers"))):
        fig.text(x0 / W_PX, (top + 24) / H_PX, head, color="#ffcc80" if side == 0 else "#80d8ff", fontsize=13.5, fontweight="bold", va="center")
        tally = fig.text((x0 + 3 * cell + 2 * gap) / W_PX, (top + 24) / H_PX, "", color="white", fontsize=13.5, ha="right", va="center")
        cells = []
        for i, tid in enumerate(idx):
            r, c = divmod(i, 3)
            rect = [(x0 + c * (cell + gap)) / W_PX, (top - (r + 1) * cell - r * gap) / H_PX, cell / W_PX, cell / H_PX]
            data, ep = rol.get(tid)
            cells.append(Cell(fig, rect, tiles[tid], data, ep, ctl))
        sides.append((cells, tally))
    legend(fig)

    def update(t):
        clock.set_text(f"t = {min(t, 15.0):4.1f} s")
        for cells, tally in sides:
            free = bumps = 0
            for cl in cells:
                ok, ended, b = cl.update(t)
                free += ok
                bumps += b
            tally.set_text(f"freed {free}/9   \u00b7   bumps {bumps}")

    return fig, update


def card(lines, sub=None):
    fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100, facecolor=BG)
    y = 0.62 if sub else 0.55
    for i, (txt, size, col, bold) in enumerate(lines):
        fig.text(0.5, y - i * 0.075, txt, color=col, fontsize=size, ha="center", va="center", fontweight="bold" if bold else "normal")
    if sub:
        sub(fig)
    return fig


def summary_table(fig):
    rows = [
        ("", "classical", "learned", "classical", "learned"),
        ("got free within 15 s", "28 %", "86 %", "30 %", "85 %"),
        ("median time to get free", "6.9 s", "1.5 s", "7.7 s", "1.1 s"),
        ("bumps per attempt", "0.29", "0.17", "0.15", "0.01"),
        ("fell down a drop / tipped", "1 %", "0 %", "3 %", "0 %"),
    ]
    xs = [0.27, 0.47, 0.57, 0.71, 0.81]
    fig.text(0.52, 0.51, "440 generated situations", color="#b0bec5", fontsize=15, ha="center")
    fig.text(0.76, 0.51, "117 spots from the real floor plans", color="#b0bec5", fontsize=15, ha="center")
    for i, r in enumerate(rows):
        y = 0.45 - i * 0.065
        for j, v in enumerate(r):
            col = "#ffcc80" if (i == 0 and j in (1, 3)) else ("#80d8ff" if (i == 0 and j in (2, 4)) else "white")
            fig.text(xs[j], y, v, color=col, fontsize=17, ha="left" if j == 0 else "center", va="center", fontweight="bold" if (i == 0 or j in (2, 4)) else "normal")
    fig.text(0.5, 0.08, "Simulation only (Isaac Lab, PhysX). Not tried on the real robot yet: rugs, wheel slip and the light bumpers still need real-world checks.",
             color="#90a4ae", fontsize=12.5, ha="center")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="E:/twin/learned/runs/video")
    ap.add_argument("--out", default="E:/twin/videos/learned/unstuck_classical_vs_learned.mp4")
    ap.add_argument("--preview", type=float, default=-1.0, help="write one PNG of --round at this time and exit")
    ap.add_argument("--round", type=int, default=0)
    a = ap.parse_args()
    tiles = S.load(os.path.join(a.dir, "pool_tiles.json"))
    cls = Rollouts(os.path.join(a.dir, "classical.npz"))
    pol = Rollouts(os.path.join(a.dir, "policy.npz"))
    if a.preview >= 0:
        fig, upd = build_round(ROUNDS[a.round][0], tiles, ROUNDS[a.round][1], cls, pol)
        upd(a.preview)
        fig.savefig(a.out.replace(".mp4", f"_preview_r{a.round}_{a.preview:.1f}.png"), dpi=100, facecolor=BG)
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
        ("The Roomba is stuck. Who gets it out?", 34, "white", True),
        ("Left: what the robot does today (Nav2 recovery behaviours + the driver's bump reflex)", 18, "#ffcc80", False),
        ("Right: the learned unstuck skill (PPO in Isaac Lab), driving through the same smoother, collision monitor and reflexes", 18, "#80d8ff", False),
        ("Each pair starts from exactly the same stuck pose; the policy never saw these situations in training. 15 s per attempt.", 15, "#b0bec5", False),
        ("Success = 0.3 m of clear space around the robot (then Nav2 takes over again).", 15, "#b0bec5", False),
    ])
    emit(intro, int(4.5 * FPS))
    plt.close(intro)
    for title, idx in ROUNDS:
        c = card([(title.split(":")[0], 34, "white", True), (title.split(":", 1)[1].strip(), 20, "#b0bec5", False)])
        emit(c, int(1.6 * FPS))
        plt.close(c)
        fig, upd = build_round(title, tiles, idx, cls, pol)
        T = 15.0
        for f in range(int(T * FPS) + 1):
            upd(f / FPS)
            emit(fig)
        emit(fig, int(2.0 * FPS))
        plt.close(fig)
        print("round done", title, n, flush=True)
    end = card([("Held-out evaluation (one attempt per situation)", 30, "white", True)], sub=summary_table)
    emit(end, int(6 * FPS))
    plt.close(end)
    ff.stdin.close()
    ff.wait()
    print(f"{a.out}: {n} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
