"""Shared scene assembly for the Isaac Sim twin (import after SimulationApp is created).

  house   /World/House        reference to house.usd (variant look = scan | clean)
  lights  /World/Lights       dome + ceiling sphere lights over the drivable area
  robot   /World/Roomba       reference to roomba550_d1000.usd (physics articulation)
  people  /World/People       Isaac People characters animated by the Replicator Agent (IRA)
                              behaviour runtime (wander on a NavMesh baked from the house)
"""
import json
import math
import os
import random

import carb
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

import house_lib
import twin_common as tc

PEOPLE_ASSETS = [
    'F_Business_02/F_Business_02.usd', 'F_Medical_01/F_Medical_01.usd', 'M_Medical_01/M_Medical_01.usd',
    'female_adult_police_01_new/female_adult_police_01_new.usd',
    'female_adult_police_02/female_adult_police_02.usd',
    'female_adult_police_03_new/female_adult_police_03_new.usd',
    'male_adult_construction_01_new/male_adult_construction_01_new.usd',
    'male_adult_construction_03/male_adult_construction_03.usd',
    'male_adult_construction_05_new/male_adult_construction_05_new.usd',
    'male_adult_police_04/male_adult_police_04.usd',
]


def assets_root():
    from isaacsim.storage.native import get_assets_root_path
    return get_assets_root_path()


def app_update(n=1):
    for _ in range(n):
        omni.kit.app.get_app().update()


class TwinScene:
    def __init__(self, look='scan', seed=0):
        self.stage = omni.usd.get_context().get_stage()
        self.rng = random.Random(seed)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.DefinePrim('/World', 'Xform')
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath('/World'))
        self.grid, self.meta = house_lib.load_grid(tc.GRID_YAML)
        self.path = np.array(tc.load_path_poses())
        self.free_xy = house_lib.free_space_samples(self.grid, self.meta, self.path[:, :2], 0.20, 1.0)
        self.house = self.stage.DefinePrim('/World/House', 'Xform')
        self.house.GetReferences().AddReference(tc.HOUSE_USD)
        self.set_look(look)
        self.robot_info = json.load(open(os.path.splitext(tc.ROBOT_USD)[0] + '_info.json'))
        self.people = []

    # ---------------- house ----------------
    def set_look(self, look):
        self.house.GetVariantSets().GetVariantSet('look').SetVariantSelection(look)
        self.look = look

    # ---------------- lights ----------------
    def add_lights(self, spacing=3.0, intensity=60000.0, dome=300.0, temperature=3500.0):
        UsdGeom.Scope.Define(self.stage, '/World/Lights')
        d = UsdLux.DomeLight.Define(self.stage, '/World/Lights/Dome')
        d.CreateIntensityAttr(dome)
        self.dome = d
        # one ceiling light per spacing x spacing cell that contains drivable free space
        cells = {}
        for x, y in self.free_xy:
            cells.setdefault((int(math.floor(x / spacing)), int(math.floor(y / spacing))), []).append((x, y))
        self.ceiling_lights = []
        for i, pts in enumerate(cells.values()):
            if len(pts) < 20:
                continue
            x, y = np.mean(pts, axis=0)
            light = UsdLux.SphereLight.Define(self.stage, f'/World/Lights/Ceiling_{i:02d}')
            light.CreateRadiusAttr(0.08)
            light.CreateIntensityAttr(intensity)
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(temperature)
            light.CreateTreatAsPointAttr(False)
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), 2.30))
            self.ceiling_lights.append(light)
        return self.ceiling_lights

    # ---------------- robot ----------------
    def add_robot(self, x=0.0, y=0.0, yaw=0.0, path='/World/Roomba'):
        prim = self.stage.DefinePrim(path, 'Xform')
        prim.GetReferences().AddReference(tc.ROBOT_USD)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        self.robot_t = xf.AddTranslateOp()
        self.robot_r = xf.AddRotateZOp()
        self.robot_t.Set(Gf.Vec3d(x, y, 0.0))
        self.robot_r.Set(math.degrees(yaw))
        self.robot_path = path
        r = self.robot_info['root']
        self.robot_paths = {k: (v.replace(r, path, 1) if v else v) for k, v in self.robot_info['paths'].items()}
        self.cam_paths = {k: v.replace(r, path, 1) for k, v in self.robot_info['cameras'].items()}
        self.imu_path = self.robot_info['imu'].replace(r, path, 1) if self.robot_info.get('imu') else None
        return prim

    def place_robot(self, x, y, yaw):
        self.robot_t.Set(Gf.Vec3d(x, y, 0.0))
        self.robot_r.Set(math.degrees(yaw))

    # ---------------- people (IRA behaviour runtime) ----------------
    def enable_people_runtime(self):
        mgr = omni.kit.app.get_app().get_extension_manager()
        for ext in ('omni.anim.navigation.bundle', 'omni.anim.behavior.bundle', 'omni.anim.graph.bundle',
                    'isaacsim.replicator.agent.core'):
            mgr.set_extension_enabled_immediate(ext, True)
        app_update(2)

    def bake_navmesh(self, agent_radius=0.25):
        import omni.anim.navigation.core as nav
        res = self.meta['resolution']
        h, w = self.grid.shape
        ox, oy = self.meta['origin'][:2]
        omni.kit.commands.execute('CreateNavMeshVolumeCommand', parent_prim_path=Sdf.Path('/World'),
                                  volume_type=nav.NAVMESH_VOLUME_INCLUDE, position=Gf.Vec3d(0, 0, 0))
        app_update(2)
        vol = [p for p in self.stage.GetPrimAtPath('/World').GetChildren() if 'NavMesh' in p.GetName()]
        if vol:
            xf = UsdGeom.Xformable(vol[0])
            cx, cy = ox + w * res / 2, oy + h * res / 2
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(cx, cy, 1.0))
                elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    op.Set(Gf.Vec3d(w * res, h * res, 2.4))
        settings = carb.settings.get_settings()
        settings.set('/exts/omni.anim.navigation.core/navMesh/config/agentRadius', agent_radius * 100.0)
        inav = nav.acquire_interface()
        inav.start_navmesh_baking_and_wait()
        ok = inav.get_navmesh() is not None
        print('NAVMESH', 'ok' if ok else 'FAILED', flush=True)
        return ok

    def spawn_people(self, n, positions=None, root='/World/People', walk_range=(2.0, 6.0)):
        from isaacsim.replicator.agent.schema import (IRA_CHARACTER_API, WANDER_IDLE_ANIMATIONS,
                                                      WANDER_IDLE_TIME_RANGES, WANDER_IDLE_WEIGHTS,
                                                      WANDER_PRIM_TYPE, WANDER_WALK_DISTANCE_RANGE)
        from omni.metropolis.schema import METRO_AGENT_ROUTINES
        aroot = assets_root()
        UsdGeom.Scope.Define(self.stage, root)
        lib = f'{root}/HumanMotionLibrary'
        omni.kit.commands.execute('CreatePayload', usd_context=omni.usd.get_context(), path_to=lib,
                                  asset_path=f'{aroot}/Isaac/People/MotionLibrary/HumanMotionLibrary.usd')
        app_update(2)
        if positions is None:
            idx = self.rng.sample(range(len(self.free_xy)), n)
            positions = [tuple(self.free_xy[i]) for i in idx]
        for i in range(n):
            asset = f'{aroot}/Isaac/People/Characters/{PEOPLE_ASSETS[i % len(PEOPLE_ASSETS)]}'
            p = f'{root}/P{i:02d}'
            omni.kit.commands.execute('CreatePayload', usd_context=omni.usd.get_context(), path_to=p,
                                      asset_path=asset)
            app_update(1)
            prim = self.stage.GetPrimAtPath(p)
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            x, y = positions[i]
            xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), 0.0))
            xf.AddRotateZOp().Set(self.rng.uniform(0, 360))
            skelroot = [q for q in Usd.PrimRange(prim) if q.GetTypeName() == 'SkelRoot']
            if not skelroot:
                carb.log_error(f'no SkelRoot in {asset}')
                continue
            sr = skelroot[0]
            omni.kit.commands.execute('ApplyBehaviorAgentAPICommand', skelroot_prim_paths=[sr.GetPath()],
                                      motion_library_prim_path=lib, motion_library_skeleton_rig='Human')
            app_update(1)
            sr.ApplyAPI(IRA_CHARACTER_API)
            beh = self.stage.DefinePrim(f'{sr.GetPath()}/wander', WANDER_PRIM_TYPE)
            beh.GetAttribute(WANDER_WALK_DISTANCE_RANGE).Set(Gf.Vec2d(*walk_range))
            beh.GetAttribute(WANDER_IDLE_ANIMATIONS).Set(['idle'])
            beh.GetAttribute(WANDER_IDLE_WEIGHTS).Set([1.0])
            beh.GetAttribute(WANDER_IDLE_TIME_RANGES).Set([[1.0, 4.0]])
            sr.GetRelationship(METRO_AGENT_ROUTINES).SetTargets([beh.GetPath()])
            tc.add_semantic_label(prim, 'person')
            self.people.append({'path': p, 'skelroot': str(sr.GetPath()), 'asset': asset})
        app_update(2)
        return self.people

    def person_positions(self):
        """World (x, y) of each person's SkelRoot (reads the animated transform)."""
        out = []
        cache = UsdGeom.XformCache()
        for pp in self.people:
            m = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(pp['skelroot']))
            t = m.ExtractTranslation()
            out.append((t[0], t[1]))
        return out
