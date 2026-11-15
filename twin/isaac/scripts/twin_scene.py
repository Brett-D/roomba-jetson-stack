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
    def __init__(self, look='remodel', seed=0):
        """look: 'remodel' (floor-plan house, default), 'scan' or 'clean' (RTAB-Map mesh / grid blocks)."""
        self.stage = omni.usd.get_context().get_stage()
        self.rng = random.Random(seed)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.DefinePrim('/World', 'Xform')
        self.stage.SetDefaultPrim(self.stage.GetPrimAtPath('/World'))
        self.grid, self.meta = house_lib.load_grid(tc.GRID_YAML)
        self.path = np.array(tc.load_path_poses())
        self.house = self.stage.DefinePrim('/World/House', 'Xform')
        self.look = look
        if look == 'remodel':
            self.house.GetReferences().AddReference(tc.REMODEL_USD)
            self.plan = json.load(open(tc.FLOORPLAN_JSON))
            self.obstacles = self._plan_obstacles()
            self.free_xy = self._plan_free_xy()
        else:
            self.house.GetReferences().AddReference(tc.HOUSE_USD)
            self.plan = None
            self.obstacles = self.grid == house_lib.OCCUPIED
            self.free_xy = house_lib.free_space_samples(self.grid, self.meta, self.path[:, :2], 0.20, 1.0)
            self.set_look(look)
        self.robot_info = json.load(open(os.path.splitext(tc.ROBOT_USD)[0] + '_info.json'))
        self.people = []

    # ---------------- house ----------------
    def set_look(self, look):
        if self.look == 'remodel':
            return
        self.house.GetVariantSets().GetVariantSet('look').SetVariantSelection(look)
        self.look = look

    def _raster(self, polys, filled=True, width=1):
        import cv2
        H, W = self.grid.shape
        res, (ox, oy) = self.meta['resolution'], self.meta['origin'][:2]
        m = np.zeros((H, W), np.uint8)
        for p in polys:
            q = np.round(((np.asarray(p) - [ox, oy]) / res) * 8).astype(np.int32)
            if filled:
                cv2.fillPoly(m, [q], 1, shift=3)
            else:
                cv2.polylines(m, [q], False, 1, width, shift=3)
        return m.astype(bool)

    def _plan_obstacles(self):
        p = self.plan
        return (self._raster(p['walls']) | self._raster([r['polyline'] for r in p['railings']], False, 2) |
                self._raster([f['footprint'] for f in p['furniture']]) |
                self._raster([f['footprint'] for f in p.get('fixtures', [])]) |
                self._raster([p['stair_void']['polygon']]))

    def _plan_free_xy(self, clearance=0.22):
        from scipy import ndimage
        indoor = self._raster([r['polygon'] for r in self.plan['rooms'] if not r['outdoor']])
        d = ndimage.distance_transform_edt(~self.obstacles) * self.meta['resolution']
        ok = indoor & (d >= clearance)
        rows, cols = np.nonzero(ok)
        res, (ox, oy) = self.meta['resolution'], self.meta['origin'][:2]
        return np.stack([ox + (cols + 0.5) * res, oy + (rows + 0.5) * res], 1)

    def randomize_remodel(self, rng):
        """Domain-randomisation hook for the remodelled house: floor textures/tints per room, wall paint,
        doors open/closed. Returns a dict for the frame metadata."""
        from pxr import Sdf, UsdShade
        if self.look != 'remodel':
            return {}
        tex = tc.TEXTURE_DIR.replace('\\', '/')
        choices = {'bedroom': ['carpet_beige', 'carpet_gray', 'carpet_taupe', 'carpet_blue', 'wood_oak', 'wood_maple'],
                   'hallway': ['carpet_taupe', 'carpet_beige', 'carpet_gray', 'wood_oak', 'wood_walnut', 'wood_maple'],
                   'bathroom': ['tile_white', 'tile_gray', 'tile_beige'], 'closet': ['carpet_gray', 'carpet_beige'],
                   'balcony': ['stone_gray']}
        info = {}
        for i, r in enumerate(self.plan['rooms']):
            sh = UsdShade.Shader(self.stage.GetPrimAtPath(f'/World/House/Looks/floor_{i}/Tex'))
            if not sh:
                continue
            kind = 'bedroom' if r['type'].startswith('bedroom') else r['type']
            t = rng.choice(choices.get(kind, ['carpet_gray']))
            sh.GetInput('file').Set(Sdf.AssetPath(f'{tex}/{t}.png'))
            g = rng.uniform(0.75, 1.2)
            sh.GetInput('scale').Set(Gf.Vec4f(g * rng.uniform(0.92, 1.08), g, g * rng.uniform(0.9, 1.08), 1.0))
            info[f'floor_{i}'] = t
        wall = UsdShade.Shader(self.stage.GetPrimAtPath('/World/House/Looks/wall/Tex'))
        if wall:
            base = rng.uniform(0.6, 1.0)
            tint = (base * rng.uniform(0.9, 1.05), base * rng.uniform(0.9, 1.03), base * rng.uniform(0.8, 1.0))
            wall.GetInput('scale').Set(Gf.Vec4f(*tint, 1.0))
            info['wall_tint'] = [round(v, 3) for v in tint]
        closed = 0
        doors = self.stage.GetPrimAtPath('/World/House/Doors')
        if doors:
            for leaf in Usd.PrimRange(doors):
                a_open = leaf.GetAttribute('twin:open_yaw_deg')
                if not a_open:
                    continue
                shut = rng.random() < 0.15
                closed += shut
                yaw = leaf.GetAttribute('twin:closed_yaw_deg').Get() if shut else a_open.Get()
                for op in UsdGeom.Xformable(leaf).GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                        op.Set(float(yaw))
        info['door_leaves_closed'] = closed
        return info

    # ---------------- lights ----------------
    def add_lights(self, spacing=3.0, intensity=60000.0, dome=300.0, temperature=3500.0):
        UsdGeom.Scope.Define(self.stage, '/World/Lights')
        d = UsdLux.DomeLight.Define(self.stage, '/World/Lights/Dome')
        d.CreateIntensityAttr(dome)
        self.dome = d
        if self.look == 'remodel':
            # room lights come with the house; add daylight through the windows
            sun = UsdLux.DistantLight.Define(self.stage, '/World/Lights/Sun')
            sun.CreateIntensityAttr(2500.0)
            sun.CreateAngleAttr(0.8)
            sx = UsdGeom.Xformable(sun)
            self.sun_rot = sx.AddRotateXYZOp()
            self.sun_rot.Set(Gf.Vec3f(-50.0, 0.0, 30.0))
            self.sun = sun
            self.ceiling_lights = [UsdLux.SphereLight(p) for p in Usd.PrimRange(self.house)
                                   if p.GetTypeName() == 'SphereLight']
            self.light_base = {str(L.GetPath()): L.GetIntensityAttr().Get() for L in self.ceiling_lights}
            return self.ceiling_lights
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
