"""Import the Roomba URDF and finish it into a simulation-ready robot USD.

  python build_robot.py [--out E:\\twin\\assets\\usd\\roomba550_d1000.usd]

1. URDF -> USD with Isaac Sim's URDF importer (urdf-usd-converter based, 6.x API).
2. Post-process in a wrapper layer:
   - wheel joints: velocity drives (stiffness 0, damping 1e3 N*m*s/rad... scaled), high-friction tyres
   - caster spheres: frictionless material (front caster + rear skid)
   - D1000 cameras (USD pinhole, real K incl. principal point offset) on the left/right optical
     frames; depth comes from the left camera's distance_to_image_plane
   - IMU (IsaacImuSensor) at camera_imu_frame
   - semantic label "robot"
Prints the prim tree and writes <out>_info.json with the prim paths other scripts use.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument('--out', default=None)
ap.add_argument('--tree', action='store_true', help='print the imported prim tree')
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

import twin_common as tc  # noqa: E402
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402

out = args.out or tc.ROBOT_USD
out_dir = os.path.dirname(out)
imp_dir = os.path.join(out_dir, 'roomba_import')
if os.path.isdir(imp_dir):
    shutil.rmtree(imp_dir)
os.makedirs(imp_dir, exist_ok=True)

cfg = URDFImporterConfig(
    urdf_path=tc.URDF,
    usd_path=imp_dir,
    merge_fixed_joints=False,
    collision_from_visuals=False,
    fix_base=False,
    joint_drive_type='force',
    joint_target_type={'.*wheel_joint': 'velocity'},
    override_joint_stiffness={'.*wheel_joint': 0.0},
    override_joint_damping={'.*wheel_joint': 2.0},
    run_asset_transformer=False,
)
imported = URDFImporter(cfg).import_urdf()
print('IMPORTED', imported)

src = Usd.Stage.Open(imported)
if args.tree:
    for p in src.Traverse():
        print('  ', p.GetPath(), p.GetTypeName(), [s for s in p.GetAppliedSchemas()])

root_prim = src.GetDefaultPrim()
root_path = str(root_prim.GetPath())


def find(name):
    for p in src.Traverse():
        if p.GetName() == name:
            return p
    return None


paths = {}
for n in ['base_link', 'left_wheel_link', 'right_wheel_link', 'camera_link', 'camera_left_optical_frame',
          'camera_right_optical_frame', 'camera_imu_frame', 'left_wheel_joint', 'right_wheel_joint']:
    p = find(n)
    paths[n] = str(p.GetPath()) if p else None
art_roots = [str(p.GetPath()) for p in src.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
casters = [str(p.GetPath()) for p in src.Traverse() if 'caster' in p.GetName()]
wheel_cols = [str(p.GetPath()) for p in src.Traverse()
              if p.HasAPI(UsdPhysics.CollisionAPI) and 'wheel' in str(p.GetPath())]
print('PATHS', json.dumps(paths, indent=1))
print('ARTICULATION_ROOTS', art_roots, 'CASTERS', casters, 'WHEEL_COLLIDERS', wheel_cols)

# ---------------- wrapper stage ----------------
stage = Usd.Stage.CreateNew(out)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.GetRootLayer().subLayerPaths.append(os.path.relpath(imported, out_dir).replace('\\', '/'))
stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))
root = stage.GetPrimAtPath(root_path)

looks = UsdGeom.Scope.Define(stage, root_path + '/TwinLooks')


def phys_material(name, sf, df, rest=0.0, combine='min'):
    m = UsdShade.Material.Define(stage, f'{root_path}/TwinLooks/{name}')
    api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    api.CreateStaticFrictionAttr(sf)
    api.CreateDynamicFrictionAttr(df)
    api.CreateRestitutionAttr(rest)
    m.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set(combine)
    return m


slick = phys_material('caster_slick', 0.0, 0.0, combine='min')
tyre = phys_material('tyre', 1.2, 1.0, combine='max')
for c in casters:
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(c)).Bind(slick, UsdShade.Tokens.strongerThanDescendants,
                                                                  'physics')
for c in wheel_cols:
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(c)).Bind(tyre, UsdShade.Tokens.strongerThanDescendants,
                                                                  'physics')

# wheel drives: velocity control (target set by the diff-drive controller)
for j in ('left_wheel_joint', 'right_wheel_joint'):
    jp = stage.GetPrimAtPath(paths[j])
    drive = UsdPhysics.DriveAPI.Apply(jp, 'angular')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(2.0)        # N*m*s/deg in PhysX angular drive units
    drive.CreateMaxForceAttr(5.0)
    drive.CreateTargetVelocityAttr(0.0)

# cameras: USD cameras look along -Z with +Y up; ROS optical frames look along +Z with +Y down
# -> rotate 180 deg about X under each optical frame.
cams = {}
for side, frame in (('left', 'camera_left_optical_frame'), ('right', 'camera_right_optical_frame')):
    cp = paths[frame] + f'/{side}_camera'
    cam = UsdGeom.Camera.Define(stage, cp)
    UsdGeom.Xformable(cam).ClearXformOpOrder()
    UsdGeom.Xformable(cam).AddOrientOp().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))
    tc.set_pinhole(cam.GetPrim())
    cams[side] = cp

# IMU at the driver's camera_imu_frame
imu_path = paths['camera_imu_frame'] + '/imu_sensor'
try:
    import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
    imu = IsaacSensorSchema.IsaacImuSensor.Define(stage, imu_path)
    imu.CreateLinearAccelerationFilterWidthAttr().Set(1)
    imu.CreateAngularVelocityFilterWidthAttr().Set(1)
    imu.CreateOrientationFilterWidthAttr().Set(1)
    imu.CreateEnabledAttr().Set(True)
except Exception as e:  # noqa: BLE001
    print('IMU schema not available:', e)
    imu_path = None

tc.add_semantic_label(root, 'robot')
stage.GetRootLayer().Save()

info = {'usd': out, 'imported': imported, 'root': root_path, 'paths': paths, 'articulation_roots': art_roots,
        'casters': casters, 'wheel_colliders': wheel_cols, 'cameras': cams, 'imu': imu_path,
        'camera_aperture': tc.camera_aperture()}
tc.save_json(info, os.path.splitext(out)[0] + '_info.json')
print('ROBOT_INFO ' + json.dumps(info))
app.close()
