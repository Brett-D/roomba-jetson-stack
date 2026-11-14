"""Renders the current platform parts on a mock Roomba 550 (assembled, exploded, top and side views).

Run after roomba_platform.py; writes assembly_on_roomba.png next to this file.
Roomba features (bumper, handle, IR receiver, serial port, wheels) are approximate and for orientation only.
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roomba_platform as rp  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assembly_on_roomba.png")
R, BODY_H = 170.0, 88.0
SERIAL = rp.SERIAL_PORT
LIGHT = np.array([0.4, -0.55, 0.75]) / np.linalg.norm([0.4, -0.55, 0.75])

COLORS = {
    "frame left": (0.25, 0.55, 0.9), "frame right": (0.35, 0.65, 0.95), "battery cradle": (0.95, 0.6, 0.2),
    "jetson holder": (0.3, 0.78, 0.45), "camera backplate": (0.5, 0.5, 0.56), "camera shroud": (0.22, 0.22, 0.26),
    "leg": (0.85, 0.3, 0.75),
}
ENV_COLORS = {"battery": (0.15, 0.15, 0.15), "jetson": (0.1, 0.35, 0.1), "camera case": (0.05, 0.05, 0.05)}
EXPLODE = {"leg": 0, "frame left": 45, "frame right": 45, "battery cradle": 95, "jetson holder": 95,
           "camera backplate": 95, "camera shroud": 95, "battery": 95, "jetson": 95, "camera case": 95}


def group_of(name):
  return "leg" if name.startswith("leg") else name


def color_of(name):
  return COLORS.get(group_of(name), (0.6, 0.6, 0.6))


def roomba_mesh():
  """Body cylinder, bumper band and a few top features as (faces, rgb) lists."""
  faces, cols = [], []
  th = np.linspace(0, 2 * np.pi, 73)
  for a0, a1 in zip(th[:-1], th[1:]):
    p = lambda a, r, z: (r * np.cos(a), r * np.sin(a), z)
    front = 0.12 * np.pi < (a0 + a1) / 2 < 0.88 * np.pi
    faces.append([p(a0, R, -BODY_H), p(a1, R, -BODY_H), p(a1, R, -4), p(a0, R, -4)])
    cols.append((0.55, 0.55, 0.58) if front else (0.42, 0.42, 0.45))
    faces.append([(0, 0, -4), p(a0, R, -4), p(a1, R, -4)])
    cols.append((0.78, 0.78, 0.8))
  return faces, cols


def shade(m, rgb, alpha=1.0):
  s = 0.35 + 0.65 * np.abs(m.face_normals @ LIGHT)
  c = np.outer(s, rgb)
  return np.c_[c, np.full(len(c), alpha)]


def draw3d(ax, meshes, envs, explode=False, hide=()):
  f, c = roomba_mesh()
  ax.add_collection3d(Poly3DCollection(f, facecolors=c, edgecolor="none"))
  for name, m in meshes.items():
    if name in hide:
      continue
    dz = EXPLODE.get(group_of(name), 45) if explode else 0
    v = m.vertices + [0, 0, dz]
    ax.add_collection3d(Poly3DCollection(v[m.faces], facecolors=shade(m, color_of(name)), edgecolor="none"))
  for name, m in envs.items():
    dz = EXPLODE[name] if explode else 0
    v = m.vertices + [0, 0, dz]
    ax.add_collection3d(Poly3DCollection(v[m.faces], facecolors=shade(m, ENV_COLORS[name], 0.55), edgecolor="none"))
  ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_zlim(-100, 300 if explode else 260)
  ax.set_box_aspect((1, 1, 1.1 if explode else 1.0)); ax.set_axis_off()


def main():
  parts, envs = rp.build()
  meshes = {k: rp.to_tm(v) for k, v in parts.items()}
  emeshes = {k: rp.to_tm(v) for k, v in envs.items() if k in ENV_COLORS}

  fig = plt.figure(figsize=(22, 16))

  ax = fig.add_subplot(2, 2, 1, projection="3d")
  draw3d(ax, meshes, emeshes, hide=())
  ax.view_init(24, -62)
  ax.set_title("1. Assembled on the Roomba (front = camera; dark = battery, Jetson, camera)", fontsize=15)

  ax = fig.add_subplot(2, 2, 2, projection="3d")
  draw3d(ax, meshes, emeshes, explode=True)
  ax.view_init(18, -62)
  ax.set_title("2. Exploded stack: Roomba -> 4 legs -> frame (2 halves) -> cradle / holder / camera", fontsize=15)
  for txt, z, col in (("battery / Jetson / camera", EXPLODE["battery"] + 70, (0.2, 0.2, 0.2)),
                      ("frame", EXPLODE["frame left"] + 50, COLORS["frame left"]), ("legs", 25, COLORS["leg"])):
    ax.text(-175, -175, z, txt, color=col, fontsize=12, weight="bold")

  ax = fig.add_subplot(2, 2, 3)
  ax.add_patch(Circle((0, 0), R, fc=(0.9, 0.9, 0.92), ec="0.35", lw=1.5))
  ax.add_patch(Wedge((0, 0), R + 3, 22, 158, width=10, fc=(0.6, 0.6, 0.65), ec="none"))
  ax.text(0, R + 8, "BUMPER (front)", ha="center", fontsize=11, color="0.3")
  ax.add_patch(Circle((0, 150), 9, fc=(0.3, 0.3, 0.35))); ax.text(14, 150, "IR receiver", fontsize=9, va="center")
  ax.add_patch(Rectangle((-118, -30), 10, 60, fc="0.25")); ax.add_patch(Rectangle((108, -30), 10, 60, fc="0.25"))
  ax.text(-113, -40, "wheel", ha="center", fontsize=9); ax.text(113, -40, "wheel", ha="center", fontsize=9)
  ax.add_patch(Circle(rp.BUTTON, 12, fc=(0.85, 0.2, 0.2), ec="k"))
  ax.text(rp.BUTTON[0], rp.BUTTON[1] - 22, "CLEAN\n(open above)", ha="center", fontsize=9, weight="bold")
  ax.add_patch(Circle(SERIAL, 7, fc="gold", ec="k"))
  ax.text(SERIAL[0], SERIAL[1] - 20, "serial port\n(open above)", ha="center", fontsize=9, weight="bold")
  for name, m in meshes.items():
    z = m.bounds[1][2] - 0.6 if name.startswith("frame") else m.bounds[0][2] + 0.6
    s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if s is not None:
      for e in s.discrete:
        ax.plot(e[:, 0], e[:, 1], color=color_of(name), lw=0.8)
  ax.axvline(0, ymin=0.08, ymax=0.92, color=COLORS["frame left"], lw=1, ls="--")
  ax.text(0, rp.FRAME_Y0 - 10, "seam: right half drops onto 3 dovetails; everything locates on printed pegs",
          ha="center", fontsize=9, color=COLORS["frame left"])
  for name, m in emeshes.items():
    b = m.bounds
    ax.add_patch(Rectangle(b[0][:2], *(b[1][:2] - b[0][:2]), fc=ENV_COLORS[name], alpha=0.25))
    ax.text(*((b[0][:2] + b[1][:2]) / 2), name.upper(), ha="center", va="center", fontsize=12, weight="bold")
  for gx, gy, ang in rp.leg_layout():
    a = np.radians(ang)
    ax.plot([gx, gx + 26 * np.cos(a)], [gy, gy + 26 * np.sin(a)], color=COLORS["leg"], lw=4)
    ax.add_patch(Circle((gx, gy), 26, fill=False, ls=":", color=COLORS["leg"]))
  used = {s for _, s in rp.LEG_SCREWS}
  for s, (x, y) in rp.ROOMBA_SCREWS.items():
    ax.plot(x, y, marker="X" if s in used else "x", ms=11 if s in used else 7,
            color=COLORS["leg"] if s in used else "0.35", mew=2)
    ax.text(x + 4, y + 4, s, fontsize=9, color=COLORS["leg"] if s in used else "0.35", weight="bold")
  ax.text(0, -R - 14, "magenta = legs; X = Roomba screw each foot uses (x = other screws seen in your photo)",
          ha="center", fontsize=10, color=COLORS["leg"])
  ax.set_aspect("equal"); ax.set_xlim(-190, 190); ax.set_ylim(-195, 190); ax.axis("off")
  ax.set_title("3. Top view, front = up", fontsize=15)

  ax = fig.add_subplot(2, 2, 4)
  ax.add_patch(Rectangle((-R, -BODY_H), 2 * R, BODY_H - 4, fc=(0.85, 0.85, 0.88), ec="0.4"))
  ax.add_patch(Rectangle((R - 30, -BODY_H + 10), 33, BODY_H - 18, fc=(0.6, 0.6, 0.65), ec="none"))
  ax.text(R - 14, -BODY_H / 2, "bumper", rotation=90, ha="center", va="center", fontsize=9)
  ax.axhline(0, color="0.4", lw=0.8, ls="--"); ax.text(-R, 2, "top frame (faceplate removed)", fontsize=9, color="0.3")
  for name, m in meshes.items():
    s = m.section(plane_origin=[0 if not name.startswith("battery") else -70, 0, 0],
                  plane_normal=[1, 0, 0]) if not name.startswith("leg") else None
    if s is not None:
      for e in s.discrete:
        ax.plot(e[:, 1], e[:, 2], color=color_of(name), lw=1.0)
  for name, m in emeshes.items():
    b = m.bounds
    ax.add_patch(Rectangle((b[0][1], b[0][2]), b[1][1] - b[0][1], b[1][2] - b[0][2], fc=ENV_COLORS[name], alpha=0.3))
    ax.text((b[0][1] + b[1][1]) / 2, b[1][2] + 3, name, ha="center", fontsize=10)
  dz = rp.deck_z()
  ax.annotate("", xy=(-150, 0), xytext=(-150, dz), arrowprops=dict(arrowstyle="<->"))
  ax.text(-147, dz / 2, f"{dz:.0f} mm legs", va="center", fontsize=10)
  cam = envs["camera case"].bounding_box()
  ax.annotate("", xy=(cam[4], -BODY_H - 6), xytext=(R, -BODY_H - 6), arrowprops=dict(arrowstyle="<->"))
  ax.text((cam[4] + R) / 2, -BODY_H - 16, f"{R - cam[4]:.0f} mm inside bumper", ha="center", fontsize=9)
  ax.set_aspect("equal"); ax.set_xlim(-200, 200); ax.set_ylim(-110, 230); ax.axis("off")
  ax.set_title("4. Side section through the centre (front = right)", fontsize=15)

  plt.tight_layout()
  plt.savefig(OUT, dpi=70)
  print("saved", OUT)


if __name__ == "__main__":
  main()
