"""Printable ArUco markers for RTAB-Map landmarks (vslam.launch.py markers:=true).

  python tools/markers/make_markers.py [--out markers.pdf] [--count 10] [--size_mm 150]

One marker per Letter page: ID 0 is the robot's home / start spot, the others go one or two per
room. DICT_4X4_50 (RTAB-Map Marker/Dictionary 0); --size_mm must match the launch argument
marker_length (metres). Needs opencv-contrib (cv2.aruco) and matplotlib.
"""
import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

W_IN, H_IN = 8.5, 11.0


def mm(v):
    return v / 25.4


def page(pdf, dictionary, marker_id, size_mm):
    fig = plt.figure(figsize=(W_IN, H_IN))
    s = mm(size_mm)
    ax = fig.add_axes([(W_IN - s) / 2 / W_IN, 3.0 / H_IN, s / W_IN, s / H_IN])
    ax.imshow(cv2.aruco.generateImageMarker(dictionary, marker_id, 600, borderBits=1),
              cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.axis("off")
    name = "HOME / START SPOT" if marker_id == 0 else f"ROOM MARKER {marker_id}"
    fig.text(0.5, 0.93, f"ID {marker_id}", ha="center", fontsize=40, weight="bold")
    fig.text(0.5, 0.895, name, ha="center", fontsize=16)
    fig.text(0.5, 0.86, f"Roomba navigation marker  -  ArUco DICT_4X4_50  -  {size_mm:g} mm",
             ha="center", fontsize=10, color="0.35")
    bx = (W_IN - mm(100)) / 2
    bar = fig.add_axes([bx / W_IN, 1.9 / H_IN, mm(100) / W_IN, 0.04 / H_IN])
    bar.set_facecolor("black")
    bar.set_xticks([])
    bar.set_yticks([])
    fig.text(0.5, 1.65 / H_IN, f"Print at 100 % / Actual size. This bar must measure 100 mm; "
             f"the black square {size_mm:g} mm.", ha="center", fontsize=9)
    fig.text(0.5, 1.4 / H_IN, "Mount flat on a wall, centre about 20 cm above the floor, facing open "
             "floor. Matte paper, no glare. Never move it after mapping.", ha="center", fontsize=8, color="0.3")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="markers.pdf")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--size_mm", type=float, default=150.0)
    a = ap.parse_args()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    with PdfPages(a.out) as pdf:
        for i in range(a.count):
            page(pdf, dictionary, i, a.size_mm)
    print(f"wrote {a.out}: IDs 0-{a.count - 1}, {a.size_mm:g} mm")


if __name__ == "__main__":
    main()