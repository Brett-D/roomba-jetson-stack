"""Parametric Roomba 500-series (550) development platform - open-frame version.

A light rail frame (left/right halves joined by vertical dovetails, dropped onto pegs on the leg tops) carries a
MYNT EYE D1000-IR-120 stereo camera (front: a flat plate screwed to two posts on the frame, plus a shroud), an Anker 737 power bank (left) and a Jetson Orin Nano Developer Kit
(rear-right, screwless port-locked holder, ports facing forward). The middle stays open so the CLEAN button can be
pressed by hand and the Mini-DIN (Open Interface) plug comes straight up. Four adjustable legs replace four of
the eleven top-frame screws, so nothing is drilled.

Assembly axes: origin = Roomba centre on the top frame (faceplate removed), +y = forward, +x = robot's right,
z = up. Requires: pip install manifold3d trimesh numpy
"""
import math
import os
import sys

import manifold3d as mf
import trimesh

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- Measure / verify these ------------------------------------------------------------------------------
LEG_HEIGHTS = [16.0, 16.0, 16.0, 16.0]  # foot bottom -> frame underside; set per leg if the screw bosses differ
ROOMBA_BUMP_H = 5.0        # tallest feature on the top frame under the rails (measured)
CAM_BOTTOM = 29.0          # camera case underside above the top frame (6 mm over the frame top)
BUTTON = (0.0, 15.0)       # CLEAN button centre: 155 mm back from the bumper front
SERIAL_PORT = (88.0, 15.0)  # Mini-DIN: ~88 mm right of the CLEAN button, ~70 mm in from the right edge

CAM_W, CAM_D, CAM_H = 165.0, 45.0, 60.0          # camera case (lens face = W x H); width re-measured 165
CAM_BOLT_X = (73.0, 83.0)  # outer case-back screw columns 156 mm apart (confirmed), horizontal slots give +-5 mm of slide
CAM_BOLT_Z = (24.25, 24.25)  # both rows above/below the case centre: ~48.5 apart, ~6 mm up from the case bottom (confirmed)
CAM_SCREW_LEN = 12.0       # the camera's own case screws (thread length)
LENS_X, LENS_DZ = 60.0, 8.0                      # 120 mm baseline; lens centres above case centre (from photo)
LENS_BACK_D, LENS_FRONT_D = 30.0, 30.0           # straight lens holes: the shroud prints face-down, so a flared
                                                 # bevel becomes an unsupported overhang; 30 mm clears the 121 deg FOV
IR_X, IR_W, IR_H = 30.0, 20.0, 30.0              # IR emitter cut-outs (from photo)
CABLE_X = (47.0,)                                # USB cable exit slot(s) in the shroud floor (from photo)

BAT_L, BAT_W, BAT_H = 155.7, 54.6, 49.5          # Anker 737, display face up, ports facing the rear
TRAY_W, TRAY_D, TRAY_H = 103.0, 90.5, 34.0      # Jetson in its tray: along the port edge, depth, heatsink top
JET_X0, JET_FRONT_Y = 4.0, -10.0                 # holder's left edge and front (port wall) face; ports face forward
DC_SPAN, DC_TOP = (1.0, 17.0), 18.0              # barrel jack span from the left end of the port edge (looking at it)
USB_X0, USB_W, USB_TOP = 37.0, 30.0, 23.0        # USB stack: start from that same left end, width, top height
SILL_H = 4.0                                     # must stay below every port and the PCB edge - VERIFY
LEDGE_D = 4.0                                    # how far the ledges reach over the port tops
CLAMP_Z = 4.0                                    # rear clamp screw height above the tray bottom
RIB_Y = (20.0, 70.0)                             # crush ribs on each holder side wall, from the port edge
RIB_BASE, RIB_INTERFERENCE = 2.0, 0.3            # rib width on the wall; how far the apex squeezes into the tray

# ---- Frame -----------------------------------------------------------------------------------------------
FRAME_X, FRAME_Y0, FRAME_Y1 = 115.0, -120.0, 95.0   # outer half-width, rear and front edges
RAIL_W, FRONT_RAIL_W, FRAME_T, FRAME_R = 15.0, 30.0, 7.0, 20.0
CROSS_Y = (-25.0, -10.0)                         # middle cross rail (supports the Jetson and battery)
# Frame halves meet at x = 0 with vertical dovetails (left half's tongue drops into the right half's socket).
DOVE = {"front": (10.0, 14.0, 8.0), "rail": (6.0, 9.0, 6.0)}   # neck width, flare width, length
DOVE_CLR = 0.35                                  # per side; Ender prints dovetails ~0.3 mm fat per side
DOVE_EF = 0.6                                    # bottom chamfer (elephant's foot) + top lead-in height
BUTT_GAP = 0.3                                   # gap between the two halves' flat end faces

# ---- Tool-less pegs: printed pointing up on the frame (accessories) and on the leg tops (frame) -------------
PEG_D, PEG_HOLE = 6.0, 6.3                       # peg core, mating hole (same holes as before)
PEG_RIB_R, PEG_RIB_W = 3.3, 1.0                  # 3 crush ribs: 0.15 mm interference each, for a snug press fit
PEG_CHAMFER = 0.8
PEG_H_ACC = 4.5                                  # frame pegs into the battery cradle / Jetson holder bases (5 / 5.5)
PEG_H_LEG = FRAME_T - 1.0                        # leg pegs stop 1 mm below the frame top

M3 = 3.4
NUT_R, NUT_DEPTH = 5.9 / math.sqrt(3), 2.6       # hex pocket for M3 nuts (5.5 AF + clearance)
CLR = 0.6

# ---- Legs ------------------------------------------------------------------------------------------------
COL_D, FOOT_T, ARM_W, ARM_L = 16.0, 5.0, 12.0, 26.0
SLOT_W, SLOT_R0 = 3.8, 9.0                       # slot for the Roomba's own screw
SCREW_HEAD_D, SCREW_SINK = 8.4, 2.0              # head recess along the slot: clears the fillet, leaves 3 mm clamped
SLOT_MIN = COL_D / 2 + 0.3 + SCREW_HEAD_D / 2    # closest screw position whose head clears the column

# ---- Accessories -----------------------------------------------------------------------------------------
BASE_T = 5.0
BAT_WALL, BAT_SIDE_H, BAT_END_H = 3.0, 25.0, 8.0
BAT_X0, BAT_Y0 = -113.0, -118.0                  # cradle sits on the left, rear and cross rails
STRAP_W, STRAP_H = 27.0, 4.5
HOLDER_BASE, HOLDER_WALL, HOLDER_BACK_T, SIDE_H, BEAM_H = 5.5, 3.0, 6.0, 8.0, 4.0
# Camera posts: part of each frame half, behind the camera's outer screw columns. The camera's own screws go in
# from behind, through POST_CLAMP mm of post, straight into the case (no separate back plate).
CAM_SCREW_CLR, CAM_HEAD_CB, POST_CLAMP = 4.4, 7.3, 3.0   # camera M4 screws (6.75 head): clearance, counterbore, plastic under the head
UP_WEB, UP_D, UP_WEB_H = 2.5, 8.0, 45.0          # edge webs: thickness; post depth (front rail edge); web height
UP_W = (CAM_BOLT_X[1] - CAM_BOLT_X[0]) + CAM_HEAD_CB + 2 * (UP_WEB + 0.5)   # post width keeps hex-key access open
SH_WALL, SH_FRONT, SH_CLR = 2.5, 2.5, 0.4                        # camera shroud
# Truss lightening (triangle lattice). Each switch is kept only if it measurably saves filament.
TRUSS_BATTERY, TRUSS_LEGS = True, False                          # measured: only the battery truss saves filament
ISO_RIB, ISO_CORNER = 2.4, 1.0
LEG_POST_D, LEG_POST_R, LEG_STRUT_D, LEG_CAP_H, LEG_COLLAR = 5.0, 5.5, 3.5, 12.0, 13.0


def box(x0, y0, z0, x1, y1, z1):
  return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def rbox(x0, y0, z0, x1, y1, z1, r):
  cs = mf.CrossSection.square([x1 - x0 - 2 * r, y1 - y0 - 2 * r]).offset(r, mf.JoinType.Round, 2.0, 32)
  return mf.Manifold.extrude(cs, z1 - z0).translate([x0 + r, y0 + r, z0])


def cyl(d, z0, z1, x=0.0, y=0.0, seg=40):
  return mf.Manifold.cylinder(z1 - z0, d / 2, d / 2, seg).translate([x, y, z0])


def cyl_y(d0, d1, y0, y1, x, z, seg=48):
  return mf.Manifold.cylinder(y1 - y0, d0 / 2, d1 / 2, seg).rotate([-90, 0, 0]).translate([x, y0, z])


def rbox_y(x0, z0, x1, z1, y0, y1, r):
  """Rounded box whose rounded profile lies in the x-z plane, extruded along +y."""
  cs = mf.CrossSection.square([x1 - x0 - 2 * r, z1 - z0 - 2 * r]).offset(r, mf.JoinType.Round, 2.0, 32)
  return mf.Manifold.extrude(cs.translate([x0 + r, -z1 + r]), y1 - y0).rotate([-90, 0, 0]).translate([0, y0, 0])


def hexp(z0, z1, x, y):
  return mf.Manifold.cylinder(z1 - z0, NUT_R, NUT_R, 6).translate([x, y, z0])


def union(parts):
  return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def deck_z():
  """Frame underside height (leg tops)."""
  return min(LEG_HEIGHTS)


def deck_top():
  """Frame top surface, where every accessory sits."""
  return deck_z() + FRAME_T


def rail_lines():
  front = FRAME_Y1 - FRONT_RAIL_W / 2
  rear = FRAME_Y0 + RAIL_W / 2
  side = FRAME_X - RAIL_W / 2
  cross = (CROSS_Y[0] + CROSS_Y[1]) / 2
  return front, rear, side, cross


def frame_holes():
  """Spare M3 accessory grid (nut pockets underneath), kept clear of the seam and of the pin holes."""
  front, rear, side, cross = rail_lines()
  holes = []
  for x in range(-100, 101, 20):
    if x:  # no hole on the x = 0 seam
      holes += [(float(x), front), (float(x), rear)]
  for x in range(-80, 81, 20):
    if x:
      holes.append((float(x), cross))
  for y in range(-100, 61, 20):
    holes += [(-side, float(y)), (side, float(y))]
  pegs = all_pegs()
  return [h for h in holes if abs(h[0]) > 14 and all(math.dist(h, p) > 8.5 for p in pegs)]


def make_peg(h):
  """Peg standing on z = 0, h tall: chamfered core plus 3 crush ribs that taper out before the chamfer."""
  top = h - PEG_CHAMFER
  core = union([cyl(PEG_D, 0, top + 0.01, seg=40),
                mf.Manifold.cylinder(PEG_CHAMFER, PEG_D / 2, PEG_D / 2 - PEG_CHAMFER, 40).translate([0, 0, top])])
  r_in, w = PEG_D / 2 - 0.3, PEG_RIB_W / 2
  rib = mf.Manifold.batch_hull([box(r_in, -w, 0, PEG_RIB_R, w, top - 1.0),
                                box(r_in, -w, top - 0.01, r_in + 0.01, w, top)])
  return union([core] + [rib.rotate([0, 0, a]) for a in (0, 120, 240)])


def peg_hole(x, y, base_top):
  """Through hole in an accessory base for a frame peg."""
  return cyl(PEG_HOLE, deck_top() - 1, base_top + 1, x, y, 40)


# ---- Truss (isogrid) lightening ---------------------------------------------------------------------------
def cs_poly(pts):
  p = list(pts)
  area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(p, p[1:] + p[:1]))
  return mf.CrossSection([p if area > 0 else p[::-1]])


def cs_rect(u0, v0, u1, v1):
  return mf.CrossSection.square([u1 - u0, v1 - v0]).translate([u0, v0])


def cs_circle(u, v, r):
  return mf.CrossSection.circle(r, 32).translate([u, v])


def iso_holes(u0, u1, v0, v1, side, keepouts=(), rib=ISO_RIB, rows=None):
  """Triangle (isogrid / Warren truss) openings in the (u, v) rectangle; +v is 'up' when the panel prints standing.

  rows=N sizes the triangles so N rows exactly fill the band (big openings save the most filament).
  Only whole triangles are kept, so every edge and keep-out stays solid. Inverted triangles bridge a short
  flat top; upright ones need no support. Returns (CrossSection, count).
  """
  if rows:
    side = (v1 - v0) / rows / (math.sqrt(3) / 2)
  h = side * math.sqrt(3) / 2
  inner = cs_rect(u0, v0, u1, v1)
  keep = mf.CrossSection.compose(list(keepouts)) if keepouts else None
  rows = int((v1 - v0 + 1e-6) // h)
  if rows < 1:
    return None, 0
  vs = v0 + ((v1 - v0) - rows * h) / 2
  n_tri = max(int(2 * (u1 - u0) / side) - 1, 1)              # centre the run of triangles along u
  us = u0 + ((u1 - u0) - (n_tri + 1) * side / 2) / 2
  holes = []
  for j in range(rows):
    vb, vt = vs + j * h, vs + (j + 1) * h
    shift = (side / 2) if j % 2 else 0.0
    for k in range(-2, int((u1 - u0) // side) + 3):
      ub = us + k * side + shift
      for tri in ([(ub, vb), (ub + side, vb), (ub + side / 2, vt)],
                  [(ub + side / 2, vt), (ub + 1.5 * side, vt), (ub + side, vb)]):
        hole = cs_poly(tri).offset(-(rib / 2 + ISO_CORNER), mf.JoinType.Miter, 4.0, 16)
        if hole.is_empty():
          continue
        hole = hole.offset(ISO_CORNER, mf.JoinType.Round, 2.0, 16)
        a = hole.area()
        if (hole ^ inner).area() < a - 1e-3:
          continue
        if keep is not None and (hole ^ keep).area() > 1e-3:
          continue
        holes.append(hole)
  if not holes:
    return None, 0
  return mf.CrossSection.compose(holes), len(holes)


def extrude_xy(cs, z0, z1):
  return mf.Manifold.extrude(cs, z1 - z0).translate([0, 0, z0])


def extrude_yx(cs, z0, z1):
  """(u, v) = (y, x), extruded along z (lets a horizontal panel's truss run along y)."""
  return mf.Manifold.extrude(cs, z1 - z0).transform([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, z1]])


def extrude_yz(cs, x0, x1):
  """(u, v) = (y, z), extruded along +x."""
  return mf.Manifold.extrude(cs, x1 - x0).transform([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]])


def extrude_xz(cs, y0, y1):
  """(u, v) = (x, z), extruded along y."""
  return mf.Manifold.extrude(cs, y1 - y0).transform([[1, 0, 0, 0], [0, 0, -1, y1], [0, 1, 0, 0]])


# ---- Frame -----------------------------------------------------------------------------------------------
def cam_screw_slots():
  """(x_min, x_max, z_min, z_max) of each outer camera-screw position range: 2 per post (top and bottom row)."""
  zc = CAM_BOTTOM + CAM_H / 2
  out = []
  for s in (-1, 1):
    xa, xb = sorted((s * CAM_BOLT_X[0], s * CAM_BOLT_X[1]))
    for row in (-1, 1):
      za, zb = sorted((zc + row * CAM_BOLT_Z[0], zc + row * CAM_BOLT_Z[1]))
      out.append((xa, xb, za, zb))
  return out


def slot_y(d, y0, y1, xa, xb, za, zb):
  """Rounded slot of width d along y, covering the screw position range [xa..xb] x [za..zb]."""
  return mf.Manifold.batch_hull([cyl_y(d, d, y0, y1, x, z, 24) for x in (xa, xb) for z in (za, zb)])


def make_frame():
  z0, zt = deck_z(), deck_top()
  outer = rbox(-FRAME_X, FRAME_Y0, z0, FRAME_X, FRAME_Y1, zt, FRAME_R)
  ix = FRAME_X - RAIL_W
  openings = [rbox(-ix, CROSS_Y[1], z0 - 1, ix, FRAME_Y1 - FRONT_RAIL_W, zt + 1, 6.0),   # front: button, port
              rbox(-ix, FRAME_Y0 + RAIL_W, z0 - 1, ix, CROSS_Y[0], zt + 1, 6.0)]        # rear
  posts = []
  zc = CAM_BOTTOM + CAM_H / 2
  up_top = zc + CAM_BOLT_Z[1] + CAM_HEAD_CB / 2 + 2.5
  xc, hw = (CAM_BOLT_X[0] + CAM_BOLT_X[1]) / 2, UP_W / 2
  for s in (-1, 1):  # posts behind the camera's outer bolt columns, braced by webs along each post edge
    xa, xb = sorted((s * (xc - hw), s * (xc + hw)))
    posts.append(box(xa, FRAME_Y1 - UP_D, zt - 0.01, xb, FRAME_Y1, up_top))
    for ex in (xa, xb - UP_WEB):
      posts.append(prism_x([(FRAME_Y1, zt - 0.01), (FRAME_Y1, zt + UP_WEB_H), (FRAME_Y1 - FRONT_RAIL_W, zt - 0.01)],
                           ex, ex + UP_WEB))
  cuts = list(openings)
  for x, y in frame_holes():
    cuts.append(cyl(M3, z0 - 1, zt + 1, x, y, 24))
    cuts.append(hexp(z0 - 0.01, z0 + NUT_DEPTH, x, y))                                # nut pockets underneath
  cuts += [cyl(PEG_HOLE, z0 - 1, zt + 1, x, y, 40) for x, y in LEG_PEGS()]   # leg pegs come up into these
  for xa, xb, za, zb in cam_screw_slots():  # camera's screws go in from behind, through POST_CLAMP of plastic
    cuts.append(slot_y(CAM_SCREW_CLR, FRAME_Y1 - UP_D - 1, FRAME_Y1 + 1, xa, xb, za, zb))
    cuts.append(slot_y(CAM_HEAD_CB, FRAME_Y1 - UP_D - 1, FRAME_Y1 - POST_CLAMP, xa, xb, za, zb))
  pegs = [make_peg(PEG_H_ACC).translate([x, y, zt - 0.01]) for x, y in BAT_PEGS + JET_PEGS]
  return union([outer] + posts) - union(cuts) + union(pegs)


def dovetails(clr=0.0):
  """Tongues on the left half, crossing x = 0 into the right half, one per rail. clr = 0: the tongues, with
  stepped chamfers on the bottom (elephant's foot) and top (lead-in). clr > 0: the sockets, flared at the bottom."""
  z0, zt = deck_z(), deck_top()
  front, rear, side, cross = rail_lines()
  steps = 4
  out = []
  for y, (wn, wf, ln) in ((front, DOVE["front"]), (cross, DOVE["rail"]), (rear, DOVE["rail"])):
    pts = [(-1.0, y - wn / 2 - clr), (0.0, y - wn / 2 - clr), (ln + clr, y - wf / 2 - clr),
           (ln + clr, y + wf / 2 + clr), (0.0, y + wn / 2 + clr), (-1.0, y + wn / 2 + clr)]
    prof = cs_poly(pts)
    dz = DOVE_EF / steps
    if clr:
      out.append(extrude_xy(prof, z0 - 1, zt + 1))
      for i in range(steps):  # bottom flare: wider towards the bed
        out.append(extrude_xy(prof.offset(DOVE_EF * (steps - i) / steps, mf.JoinType.Miter, 4.0),
                              z0 - (1 if i == 0 else 0) + i * dz, z0 + (i + 1) * dz))
    else:
      out.append(extrude_xy(prof, z0 + DOVE_EF - 0.01, zt - DOVE_EF + 0.01))
      for i in range(steps):  # narrower towards the bed and towards the top
        inset = prof.offset(-DOVE_EF * (steps - i) / steps, mf.JoinType.Miter, 4.0)
        out.append(extrude_xy(inset, z0 + i * dz, z0 + (i + 1) * dz + 0.01))
        out.append(extrude_xy(inset, zt - (i + 1) * dz - 0.01, zt - i * dz))
  return union(out)


def split_frame(frame):
  """Left/right halves with vertical dovetails at x = 0: lower the right half onto the left half's tongues."""
  big = 400
  left = (frame ^ box(-big, -big, -big, -BUTT_GAP / 2, big, big)) + (frame ^ dovetails())
  right = (frame ^ box(BUTT_GAP / 2, -big, -big, big, big, big)) - dovetails(DOVE_CLR)
  return left, right


def make_leg(h):
  """Foot (slot over a Roomba screw) + column + a peg on top that drops into the frame.
  Column is a 3-post truss when TRUSS_LEGS, otherwise a solid cylinder."""
  zc = h - LEG_CAP_H
  arm = mf.Manifold.batch_hull([cyl(ARM_W, 0, FOOT_T), cyl(ARM_W, 0, FOOT_T, ARM_L, 0)])
  pad = cyl(COL_D + 8, 0, FOOT_T)
  fillet = mf.Manifold.cylinder(6.0, (COL_D + 8) / 2, COL_D / 2, 48).translate([0, 0, FOOT_T])
  if not TRUSS_LEGS:
    leg = union([cyl(COL_D, 0, h), arm, pad, fillet])
  else:
    leg = union([arm, pad, fillet, cyl(COL_D, 0, LEG_COLLAR), cyl(COL_D, zc, h)] + leg_truss(zc))
  leg = leg + make_peg(PEG_H_LEG).translate([0, 0, h - 0.01])
  slot = mf.Manifold.batch_hull([cyl(SLOT_W, -1, FOOT_T + 1, SLOT_R0, 0, 24),
                                 cyl(SLOT_W, -1, FOOT_T + 1, ARM_L, 0, 24)])
  head = mf.Manifold.batch_hull([cyl(SCREW_HEAD_D, FOOT_T - SCREW_SINK, h, x, 0, 32) for x in (SLOT_R0, ARM_L + 3)])
  head = head ^ box(COL_D / 2 + 0.3, -20, -1, 60, 20, h + 1)
  return leg - union([slot, head])


def make_peg_fit_test():
  """7 mm coupon with holes at PEG_HOLE -0.2 / +0 / +0.2 (1, 2, 3 notches), plus a tab carrying one peg."""
  z = FRAME_T
  coupon = rbox(-22, -8, 0, 22, 8, z, 3)
  cuts = []
  for i, (x, d) in enumerate(((-13, -0.2), (0, 0.0), (13, 0.2))):
    cuts.append(cyl(PEG_HOLE + d, -1, z + 1, x, 0, 40))
    for k in range(i + 1):
      mx = x + (k - i / 2) * 2.0
      cuts.append(box(mx - 0.5, 5.0, z - 0.6, mx + 0.5, 7.0, z + 1))
  tab = rbox(-8, 12, 0, 8, 28, 3, 3) + make_peg(PEG_H_LEG).translate([0, 20, 2.99])
  return (coupon - union(cuts)) + tab


def leg_truss(zc):
  posts, struts = [], []
  pts = [(LEG_POST_R * math.cos(math.radians(a)), LEG_POST_R * math.sin(math.radians(a))) for a in (90, 210, 330)]
  for i, (px, py) in enumerate(pts):
    posts.append(cyl(LEG_POST_D, LEG_COLLAR - 0.5, zc + 0.5, px, py, 24))
    qx, qy = pts[(i + 1) % 3]
    for (ax, ay), (bx, by) in (((px, py), (qx, qy)), ((qx, qy), (px, py))):  # X-braced bay between posts
      struts.append(mf.Manifold.batch_hull([cyl(LEG_STRUT_D, LEG_COLLAR - 1, LEG_COLLAR - 0.99, ax, ay, 16),
                                            cyl(LEG_STRUT_D, zc + 1, zc + 1.01, bx, by, 16)]))
  return posts + struts


# ---- Battery cradle --------------------------------------------------------------------------------------
_FRONT, _REAR, _SIDE, _CROSS = rail_lines()
BAT_PEGS = [(-_SIDE, -100.0), (-_SIDE, 20.0), (-80.0, _CROSS)]       # frame pegs; spread so the cradle can't tip
JET_PEGS = [(20.0, _CROSS), (80.0, _CROSS), (_SIDE, -100.0), (_SIDE, -60.0)]
# Each leg: frame hole its top peg goes into -> Roomba screw its foot slot goes over.
LEG_SCREWS = [((55.0, 77.0), "h"), ((-60.0, 74.0), "i"), ((_SIDE, -40.0), "d"), ((-_SIDE, -86.0), "e")]


def LEG_PEGS():
  return [p for p, _ in LEG_SCREWS]


def all_pegs():
  return BAT_PEGS + JET_PEGS + LEG_PEGS()


def bat_geometry():
  ox0, oy0 = BAT_X0, BAT_Y0
  ix0, iy0 = ox0 + BAT_WALL, oy0 + BAT_WALL
  ix1, iy1 = ix0 + BAT_W + 2 * CLR, iy0 + BAT_L + 2 * CLR
  return ox0, oy0, ix0, iy0, ix1, iy1, ix1 + BAT_WALL, iy1 + BAT_WALL


def make_battery_cradle():
  z0 = deck_top()
  ox0, oy0, ix0, iy0, ix1, iy1, ox1, oy1 = bat_geometry()
  base = rbox(ox0, oy0, z0, ox1, oy1, z0 + BASE_T, 4.0)
  sides = box(ox0, oy0, z0, ox1, oy1, z0 + BASE_T + BAT_SIDE_H) - box(ix0, oy0 - 1, z0 + BASE_T, ix1, oy1 + 1, z0 + 99)
  ends = box(ox0, oy0, z0, ox1, oy1, z0 + BASE_T + BAT_END_H) - box(ix0, iy0, z0 + BASE_T, ix1, iy1, z0 + 99)
  part = union([base, sides, ends])
  ymid = (iy0 + iy1) / 2
  cuts = [box(ox0 - 1, ymid + s * 50 - STRAP_W / 2, z0 + BASE_T, ox1 + 1, ymid + s * 50 + STRAP_W / 2,
              z0 + BASE_T + STRAP_H) for s in (-1, 1)]
  if TRUSS_BATTERY:
    # Base: one row of large triangles along the length (u = y, v = x), clear of the pin seats and leg pockets.
    keep = [cs_circle(y, x, PEG_HOLE / 2 + 4.0) for x, y in BAT_PEGS]
    base_iso, _ = iso_holes(oy0 + 5, oy1 - 5, ox0 + 5, ox1 - 5, 0, keep, rows=1)
    if base_iso is not None:
      cuts.append(extrude_yx(base_iso, z0 - 1, z0 + BASE_T + 1))
    wall_iso, _ = iso_holes(oy0 + 6, oy1 - 6, z0 + BASE_T + STRAP_H + 2.0, z0 + BASE_T + BAT_SIDE_H - 2.5, 0, rows=1)
    if wall_iso is not None:
      cuts.append(extrude_yz(wall_iso, ox0 - 1, ox1 + 1))
  else:
    cuts.append(rbox(ix0 + 7, iy0 + 22, z0 - 1, ix1 - 7, iy1 - 22, z0 + BASE_T + 1, 5.0))
  cuts += [peg_hole(x, y, z0 + BASE_T) for x, y in BAT_PEGS]
  env = box(ix0 + CLR, iy0 + CLR, z0 + BASE_T, ix1 - CLR, iy1 - CLR, z0 + BASE_T + BAT_H)
  return part - union(cuts), env


# ---- Jetson holder (screwless: port ledges + sill + rear clamp screws) ----------------------------------
def jet_local():
  """Holder in its own frame: tray's port edge at y = 0 facing -y, left end of that edge at x = 0."""
  zt = deck_top() + HOLDER_BASE                 # tray bottom
  ys = -0.3                                     # inner face of the port wall / sill
  yb = TRAY_D + LEDGE_D + 1.0                   # inner face of the back wall (room to drop in, then slide)
  return 0.0, 0.0, zt, ys, yb


def jet_transform():
  """Rotate the local holder 180 deg (ports face forward) and place it at JET_X0 / JET_FRONT_Y."""
  _, _, _, ys, _ = jet_local()
  ox1 = TRAY_W + CLR + HOLDER_WALL
  tx, ty = JET_X0 + ox1, JET_FRONT_Y + (ys - HOLDER_WALL)
  place = lambda m: m.rotate([0, 0, 180]).translate([tx, ty, 0])
  to_local = lambda x, y: (tx - x, ty - y)
  return place, to_local


def make_jetson_holder():
  place, to_local = jet_transform()
  x0, y0, zt, ys, yb = jet_local()
  zd = deck_top()
  c, w = CLR, HOLDER_WALL
  ox0, ox1 = x0 - c - w, x0 + TRAY_W + c + w
  oy0, oy1 = ys - w, yb + HOLDER_BACK_T
  beam_z0, beam_z1 = zt + USB_TOP + 0.4, zt + USB_TOP + 0.4 + BEAM_H

  base = rbox(ox0, oy0, zd, ox1, oy1, zt, 4.0)
  sides = [box(ox0, oy0, zd, x0 - c, oy1, zt + SIDE_H), box(x0 + TRAY_W + c, oy0, zd, ox1, oy1, zt + SIDE_H)]
  ribs = []
  for wall_x, sgn in ((x0 - c, 1), (x0 + TRAY_W + c, -1)):
    reach = c + RIB_INTERFERENCE                # apex pokes RIB_INTERFERENCE into the tray footprint
    for ry in RIB_Y:
      yc = y0 + ry
      ribs.append(mf.Manifold.batch_hull([
          box(wall_x - 0.01, yc - RIB_BASE / 2, zt, wall_x + 0.01, yc + RIB_BASE / 2, zt + SIDE_H),
          box(wall_x + sgn * reach - 0.005, yc - 0.005, zt, wall_x + sgn * reach + 0.005, yc + 0.005,
              zt + SIDE_H - reach - 1.0)]))      # 45 deg lead-in at the top so the tray drops in
  back = box(ox0, yb, zd, ox1, oy1, zt + SIDE_H + 4)
  sill = box(ox0, oy0, zd, ox1, ys, zt + SILL_H)
  posts = [box(ox0, oy0, zd, x0 - c, ys, beam_z1), box(x0 + TRAY_W + c, oy0, zd, ox1, ys, beam_z1)]
  beam = box(ox0, oy0, beam_z0, ox1, ys, beam_z1)
  usb_ledge = box(x0 + USB_X0 + 1, ys - 0.01, beam_z0, x0 + USB_X0 + USB_W - 1, ys + LEDGE_D, beam_z1)
  dc_tab = box(x0 + DC_SPAN[0], ys - 0.01, zt + DC_TOP + 0.4, x0 + DC_SPAN[1], ys + LEDGE_D, beam_z1)
  dc_web = box(x0 + DC_SPAN[0], oy0, zt + DC_TOP + 0.4, x0 + DC_SPAN[1], ys, beam_z1)
  part = union([base, back, sill, beam, usb_ledge, dc_tab, dc_web] + sides + posts + ribs)

  cuts = [rbox(x0 + 12, y0 + 12, zd - 1, x0 + TRAY_W - 12, y0 + TRAY_D - 12, zt + 1, 6.0)]  # airflow / weight
  for cx in (x0 + TRAY_W * 0.25, x0 + TRAY_W * 0.75):                                        # rear clamp screws
    zc = zt + CLAMP_Z
    cuts.append(cyl_y(M3, M3, yb - 1, oy1 + 1, cx, zc, 24))
    ny = (yb + oy1) / 2
    cuts.append(box(cx - 5.9 / 2, ny - NUT_DEPTH / 2 - 0.1, zc - NUT_R, cx + 5.9 / 2, ny + NUT_DEPTH / 2 + 0.1,
                    zt + SIDE_H + 5))                                                        # drop-in nut slot
  cuts += [peg_hole(*to_local(x, y), zt) for x, y in JET_PEGS]
  local = part - union(cuts)
  # Envelope excludes the intended crush-rib interference so the collision check only flags real clashes.
  ri = RIB_INTERFERENCE + 0.05
  env = box(x0 + ri, y0, zt, x0 + TRAY_W - ri, y0 + TRAY_D, zt + TRAY_H)
  return place(local), place(env), local


# ---- Camera: screws straight to the frame posts; shroud hides the case -----------------------------------
def cam_geometry():
  y_bp = FRAME_Y1                      # posts' front faces
  y_case = FRAME_Y1 + 0.1              # case back, flat against the posts
  zc = CAM_BOTTOM + CAM_H / 2
  sx = CAM_W / 2 + SH_CLR + SH_WALL    # shroud outer half-width
  sz0 = CAM_BOTTOM - SH_CLR - SH_WALL
  sz1 = CAM_BOTTOM + CAM_H + SH_CLR + SH_WALL
  return y_bp, y_case, zc, sx, sz0, sz1


def prism_x(pts_yz, x0, x1):
  """Extrude a (y, z) polygon along +x between x0 and x1."""
  p = list(pts_yz)
  area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(p, p[1:] + p[:1]))
  cs = mf.CrossSection([p if area > 0 else p[::-1]])
  return mf.Manifold.extrude(cs, x1 - x0).transform([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]])


def camera_envelope():
  """The camera case, its back flat against the posts' front faces."""
  _, y_case = cam_geometry()[:2]
  return rbox_y(-CAM_W / 2, CAM_BOTTOM, CAM_W / 2, CAM_BOTTOM + CAM_H, y_case, y_case + CAM_D, 4.0)


def make_camera_shroud():
  y_bp, y_case, zc, sx, sz0, sz1 = cam_geometry()
  ix = CAM_W / 2 + SH_CLR
  iz0, iz1 = CAM_BOTTOM - SH_CLR, CAM_BOTTOM + CAM_H + SH_CLR
  y0 = y_case + 0.2                    # rear rim sits just in front of the posts
  yi = y_case + CAM_D + SH_CLR         # inside of the front face
  y1 = yi + SH_FRONT
  part = rbox_y(-sx, sz0, sx, sz1, y0, y1, 5.0) - rbox_y(-ix, iz0, ix, iz1, y0 - 1, yi, 3.0)
  lz = zc + LENS_DZ
  cuts = [cyl_y(LENS_BACK_D, LENS_FRONT_D, yi - 0.01, y1 + 0.01, s * LENS_X, lz) for s in (-1, 1)]
  cuts += [rbox_y(s * IR_X - IR_W / 2, lz - IR_H / 2, s * IR_X + IR_W / 2, lz + IR_H / 2, yi - 1, y1 + 1, 3.0)
           for s in (-1, 1)]
  for x in CABLE_X:
    cuts.append(box(x - 6, y0 - 1, sz0 - 1, x + 6, yi - 3, iz0 + 0.5))
  return part - union(cuts)


# ---- Layout / export -------------------------------------------------------------------------------------
# Roomba top-cover screws, located from the ruler photo (about +-5 mm): label -> (x, y) in platform coordinates.
ROOMBA_SCREWS = {"d": (111.7, -56.2), "e": (-108.2, -66.6), "f": (95.4, 49.0), "g": (-99.9, 41.0),
                 "h": (70.3, 66.0), "i": (-77.0, 60.3), "j": (101.2, 104.1), "k": (-111.4, 95.8),
                 "l": (22.7, 139.6), "m": (-35.5, 137.4)}


def leg_layout():
  """(x, y, arm angle) per leg; the arm points from the leg column to its Roomba screw."""
  out = []
  for (gx, gy), s in LEG_SCREWS:
    sx, sy = ROOMBA_SCREWS[s]
    out.append((gx, gy, math.degrees(math.atan2(sy - gy, sx - gx))))
  return out


def leg_reach():
  return [(s, math.dist((gx, gy), ROOMBA_SCREWS[s])) for (gx, gy), s in LEG_SCREWS]


def to_tm(m):
  mesh = m.to_mesh()
  return trimesh.Trimesh(vertices=mesh.vert_properties[:, :3], faces=mesh.tri_verts, process=True)


def export(m, name, rot=None):
  for r in ([rot] if rot and not isinstance(rot[0], (list, tuple)) else (rot or [])):
    m = m.rotate(r)
  b = m.bounding_box()
  m = m.translate([-(b[0] + b[3]) / 2, -(b[1] + b[4]) / 2, -b[2]])
  t = to_tm(m)
  t.export(os.path.join(OUT, name))
  e = t.extents
  fits = max(e[0], e[1]) <= 219 and min(e[0], e[1]) <= 198 and e[2] <= 240
  print(f"  {name:28} {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f} mm  watertight {t.is_watertight}  "
        f"bodies {len(t.split(only_watertight=False))}  ~{t.volume / 1000 * 1.27:.0f} g if solid  fits bed: {fits}")
  return t


def build():
  bat, bat_env = make_battery_cradle()
  jet, jet_env, _ = make_jetson_holder()
  left, right = split_frame(make_frame())
  parts = {"frame left": left, "frame right": right, "battery cradle": bat, "jetson holder": jet,
           "camera shroud": make_camera_shroud()}
  dz = deck_z()
  envs = {"battery": bat_env, "jetson": jet_env, "camera case": camera_envelope(),
          "CLEAN button access": cyl(30.0, 0.0, 300.0, *BUTTON),
          "serial plug": cyl(16.0, 0.0, 60.0, *SERIAL_PORT)}
  for i, (xa, xb, za, zb) in enumerate(cam_screw_slots()):  # screw head in each counterbore
    y0 = FRAME_Y1 - POST_CLAMP - 4.8
    envs[f"camera screw head {i + 1}"] = mf.Manifold.batch_hull(
        [cyl_y(CAM_HEAD_CB - 0.4, CAM_HEAD_CB - 0.4, y0, FRAME_Y1 - POST_CLAMP - 0.05, x, z, 24) for x, z in
         (((xa + xb) / 2, (za + zb) / 2),)])
  for i, ((gx, gy, ang), h) in enumerate(zip(leg_layout(), LEG_HEIGHTS)):
    parts[f"leg {i + 1}"] = make_leg(h).rotate([0, 0, ang]).translate([gx, gy, dz - h])
  return parts, envs


def main():
  parts, envs = build()
  print("Clearance checks:")
  ok = True
  items = list(parts.items()) + list(envs.items())
  # Crush ribs are meant to overlap their holes; ignore only the thin rib annulus, so a missing hole still shows up.
  rib_zone = union([cyl(2 * PEG_RIB_R + 0.1, -10, 300, x, y, 40) - cyl(PEG_HOLE - 0.02, -11, 301, x, y, 40)
                    for x, y in all_pegs()])
  for i, (a, ma) in enumerate(items):
    for b, mb in items[i + 1:]:
      if a.startswith(b) or b.startswith(a):
        continue  # holder vs its own payload (port-lock ledges are meant to overlap)
      v = ((ma ^ mb) - rib_zone).volume()
      if v > 0.5:
        ok = False
        print(f"  COLLISION {a} vs {b}: {v:.1f} mm3")
  cam = envs["camera case"].bounding_box()
  print(f"  camera case: front face {cam[4]:.0f} mm from centre (bumper 170), underside {cam[2]:.0f} mm up, "
        f"lens about {CAM_BOTTOM + CAM_H / 2 + LENS_DZ:.0f} mm above the top frame")
  print(f"  camera screws x {CAM_SCREW_LEN:g} mm: {POST_CLAMP:g} mm through the post, "
        f"{CAM_SCREW_LEN - POST_CLAMP:g} mm left for the case threads (minus the back cover if refitted)")
  for h in LEG_HEIGHTS:  # frame underside must clear the bumps on the Roomba's top
    gap = deck_z() - ROOMBA_BUMP_H
    good = gap >= 3.0
    ok &= good
  print(f"  legs {deck_z():g} mm: frame underside {gap:.1f} mm above the {ROOMBA_BUMP_H:g} mm bumps "
        f"({'OK' if good else 'TOO SHORT'})")
  for s, d in leg_reach():
    good = SLOT_MIN <= d <= ARM_L - 2
    ok &= good
    print(f"  leg foot -> Roomba screw {s}: {d:4.1f} mm ({'OK' if good else 'OUT OF SLOT RANGE'}, "
          f"usable {SLOT_MIN:.1f}-{ARM_L - 2:g})")
  print("  CLEAN button and serial plug are clear from above" if ok else "  !! fix collisions")

  print("Exports (print orientation):")
  export(parts["frame left"], "frame_left.stl", rot=[0, 0, 90])     # bottom down, camera post up
  export(parts["frame right"], "frame_right.stl", rot=[0, 0, 90])
  export(make_peg_fit_test(), "peg_fit_test.stl")
  _, _, _, cross = rail_lines()   # one rail joint, both halves, laid side by side
  zt = deck_top()
  tl = parts["frame left"] ^ box(-14, cross - 10, -1, 0.5, cross + 10, zt + 1)
  tr = parts["frame right"] ^ box(-0.5, cross - 10, -1, 14, cross + 10, zt + 1)
  export(tl + tr.translate([6, 25, 0]), "dovetail_fit_test.stl")
  export(parts["battery cradle"], "battery_cradle.stl")
  export(parts["jetson holder"], "jetson_holder.stl")
  _, _, local = make_jetson_holder()
  _, _, _, ys, _ = jet_local()
  export(local ^ box(-200, -200, -50, 200, ys + LEDGE_D + 3, 300), "jetson_port_wall_fit_test.stl")
  shroud = parts["camera shroud"]
  export(shroud, "camera_shroud.stl", rot=[-90, 0, 0])
  sb = shroud.bounding_box()
  export(shroud ^ box(-200, sb[4] - 8, -50, 200, sb[4] + 1, 500), "camera_shroud_fit_test.stl", rot=[-90, 0, 0])
  legs = {f"leg_h{h:g}.stl" for h in LEG_HEIGHTS}
  for f in os.listdir(OUT):
    if f.startswith("leg_h") and f.endswith(".stl") and f not in legs:
      os.remove(os.path.join(OUT, f))
  for h in sorted(set(LEG_HEIGHTS)):
    export(make_leg(h), f"leg_h{h:g}.stl")

  mass = {"battery": 630, "jetson": 180, "camera case": 250}
  tot = sum(mass.values())
  cx = sum(mass[k] * (envs[k].bounding_box()[0] + envs[k].bounding_box()[3]) / 2 for k in mass) / tot
  cy = sum(mass[k] * (envs[k].bounding_box()[1] + envs[k].bounding_box()[4]) / 2 for k in mass) / tot
  print(f"Payload CG (battery, Jetson, camera): x {cx:+.0f} mm, y {cy:+.0f} mm from the Roomba centre")
  for old in ("camera_cradle.stl", "battery_tunnel.stl", "pi_shelf.stl", "jetson_plate.stl", "deck.stl",
              "canopy.stl", "camera_backplate.stl", "splice_front.stl", "splice_rail.stl", "push_pin.stl",
              "pin_fit_test.stl"):
    p = os.path.join(OUT, old)
    if os.path.exists(p):
      os.remove(p)
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
