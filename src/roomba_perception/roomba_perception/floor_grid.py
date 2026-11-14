"""Floor grid overlay for camera images (a "guardian"-style check of the floor calibration).

The grid lies on the floor plane (z = 0) of a ground-aligned frame: the map (world-fixed, so
it should stay put while the robot moves) or base_link (moves with the robot). It is
projected through the camera intrinsics and the TF pose of the camera, so it lines up with
the real floor only if the camera height, tilt and intrinsics are right, and it stays still
on the floor only if the odometry is right.
"""
import math

import cv2
import numpy as np


def quat_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def grid_lines(center_xy, spacing, extent, step):
    """Sampled floor-grid lines around center_xy: (lines, samples, 3) array and major-line flags."""
    cx, cy = center_xy
    n = int(extent / spacing)
    x0 = math.floor(cx / spacing) * spacing
    y0 = math.floor(cy / spacing) * spacing
    offs = np.arange(-n, n + 1) * spacing
    s = np.arange(-extent, extent + step / 2, step)
    lines, major = [], []
    for o in offs:
        x = x0 + o
        lines.append(np.stack([np.full_like(s, x), y0 + s, np.zeros_like(s)], axis=1))
        major.append(abs(x - round(x)) < 1e-3)
        y = y0 + o
        lines.append(np.stack([x0 + s, np.full_like(s, y), np.zeros_like(s)], axis=1))
        major.append(abs(y - round(y)) < 1e-3)
    return np.array(lines), major


def draw_floor_grid(img, rot, trans, fx, fy, cx, cy, spacing=0.25, max_range=2.5, min_depth=0.15,
                    color=(255, 220, 60), alpha=0.55):
    """Draw the z = 0 plane of a frame onto img.

    rot, trans: the frame's points in the camera optical frame are rot @ p + trans.
    Lines every `spacing` m, bolder every metre, out to `max_range` m from the camera
    (beyond that, a camera this close to the floor squeezes them onto the horizon).
    Returns the number of drawn segments.
    """
    cam = -rot.T @ trans                      # camera position in the grid frame
    # Floor lines project to straight lines; samples only clip them at the range/near plane.
    lines, major = grid_lines(cam[:2], spacing, max_range, 0.1)
    pts = lines @ rot.T + trans               # (L, S, 3) in the optical frame
    z = pts[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = fx * pts[..., 0] / z + cx
        v = fy * pts[..., 1] / z + cy
    h, w = img.shape[:2]
    dist = np.linalg.norm(lines[..., :2] - cam[:2], axis=-1)
    ok = (z > min_depth) & (dist < max_range) & (np.abs(u) < 4 * w) & (np.abs(v) < 4 * h)
    overlay = img.copy()
    drawn = 0
    for li in range(lines.shape[0]):
        valid = ok[li]
        if not valid.any():
            continue
        # Split the line into runs of consecutive valid samples and draw each as a polyline.
        idx = np.flatnonzero(valid)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        thick = 2 if major[li] else 1
        for run in runs:
            if len(run) < 2:
                continue
            poly = np.stack([u[li, run], v[li, run]], axis=1).round().astype(np.int32)
            cv2.polylines(overlay, [poly], False, color, thick, cv2.LINE_AA)
            drawn += len(run) - 1
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)
    return drawn
