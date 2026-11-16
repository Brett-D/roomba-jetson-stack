"""Roomba 550 + D1000 as a small USD articulation for the unstuck RL environment.

Same dimensions as twin/isaac/robot/roomba550_d1000.urdf, written directly with pxr (UsdPhysics
only) so it loads kit-less (OVPhysX) without the URDF importer:

  base_link   body shell (cylinder r 0.17, z 0.025-0.092: the bumper; the real shell is slanted at
              the bottom so it climbs ~1.5 cm), front caster sphere (r 0.03, zero friction; no rear
              skid: with rigid wheels it high-centres the robot on rug edges, the real wheels are
              sprung); frictions combine by averaging, so the floor is 0.1 and the tyres 1.9 ->
              tyre/floor 1.0, caster/floor 0.05
  top_link    D1000 housing and the payload on the lid (Jetson, power bank, pack): a separate body
              on a fixed joint, so hitting an overhang with them does not press the bumper, as on
              the real robot
  left/right_wheel_link  cylinders r 0.036 on velocity-driven revolute joints, y = +-0.1175
"""

from __future__ import annotations

import argparse
import os

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

WHEEL_R = 0.036
WHEEL_Y = 0.1175
BODY_R = 0.17


def _material(stage, path, mu):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _body(stage, path, pos, mass, inertia, com=(0.0, 0.0, 0.0)):
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    m = UsdPhysics.MassAPI.Apply(x.GetPrim())
    m.CreateMassAttr(mass)
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    return x.GetPrim()


def _collider(prim):
    UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def build(path: str) -> str:
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Roomba")
    stage.SetDefaultPrim(root.GetPrim())

    tyre = _material(stage, "/Roomba/Materials/tyre", 1.9)
    slick = _material(stage, "/Roomba/Materials/slick", 0.0)
    shell = _material(stage, "/Roomba/Materials/shell", 0.3)

    base = _body(stage, "/Roomba/base_link", (0, 0, 0), 3.6, (0.028, 0.028, 0.052), (0.025, 0.0, 0.045))
    UsdPhysics.ArticulationRootAPI.Apply(base)
    body = UsdGeom.Cylinder.Define(stage, "/Roomba/base_link/body")
    body.CreateRadiusAttr(BODY_R)
    body.CreateHeightAttr(0.067)
    body.CreateAxisAttr("Z")
    body.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.0585))
    _bind(_collider(body.GetPrim()), shell)
    casters = _body(stage, "/Roomba/caster_link", (0, 0, 0), 0.3, (2e-3, 2e-3, 3e-3))
    cj = UsdPhysics.FixedJoint.Define(stage, "/Roomba/base_link/caster_joint")
    cj.CreateBody0Rel().SetTargets([base.GetPath()])
    cj.CreateBody1Rel().SetTargets([casters.GetPath()])
    for name, x, z, r in (("caster_front", 0.12, 0.03, 0.03),):
        s = UsdGeom.Sphere.Define(stage, f"/Roomba/caster_link/{name}")
        s.CreateRadiusAttr(r)
        s.AddTranslateOp().Set(Gf.Vec3d(x, 0, z))
        _bind(_collider(s.GetPrim()), slick)

    top = _body(stage, "/Roomba/top_link", (0, 0, 0), 0.8, (0.004, 0.004, 0.004), (-0.03, 0, 0.125))
    housing = UsdGeom.Cube.Define(stage, "/Roomba/top_link/camera_housing")
    housing.CreateSizeAttr(1.0)
    housing.AddTranslateOp().Set(Gf.Vec3d(0.1375, 0, 0.13))
    housing.AddScaleOp().Set(Gf.Vec3f(0.030, 0.165, 0.0296))
    _bind(_collider(housing.GetPrim()), shell)
    payload = UsdGeom.Cube.Define(stage, "/Roomba/top_link/payload")
    payload.CreateSizeAttr(1.0)
    payload.AddTranslateOp().Set(Gf.Vec3d(-0.06, 0, 0.122))
    payload.AddScaleOp().Set(Gf.Vec3f(0.14, 0.12, 0.06))
    _bind(_collider(payload.GetPrim()), shell)
    fj = UsdPhysics.FixedJoint.Define(stage, "/Roomba/base_link/top_joint")
    fj.CreateBody0Rel().SetTargets([base.GetPath()])
    fj.CreateBody1Rel().SetTargets([top.GetPath()])
    fj.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))

    for side, y in (("left", WHEEL_Y), ("right", -WHEEL_Y)):
        w = _body(stage, f"/Roomba/{side}_wheel_link", (0, y, WHEEL_R), 0.15, (5.4e-5, 9.7e-5, 5.4e-5))
        s = UsdGeom.Cylinder.Define(stage, f"/Roomba/{side}_wheel_link/tyre")
        s.CreateRadiusAttr(WHEEL_R)
        s.CreateHeightAttr(0.02)
        s.CreateAxisAttr("Y")
        _bind(_collider(s.GetPrim()), tyre)
        j = UsdPhysics.RevoluteJoint.Define(stage, f"/Roomba/base_link/{side}_wheel_joint")
        j.CreateAxisAttr("Y")
        j.CreateBody0Rel().SetTargets([base.GetPath()])
        j.CreateBody1Rel().SetTargets([w.GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(0, y, WHEEL_R))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(0.0)
        d.CreateDampingAttr(1.0)
        d.CreateMaxForceAttr(0.3)
    stage.GetRootLayer().Save()
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "assets", "roomba_rl.usda"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print(build(a.out))
