"""Isaac Lab (3.0, kit-less OVPhysX) environment for the learned unstuck skill.

Many Roombas, each starting stuck in a tile of world.py / scenarios.py. Per 0.1 s step the env
emulates what the real stack does around the skill's cmd_vel:

  policy (v, w) -> velocity_smoother limits -> collision_monitor (StopZone / SlowdownZone on the
  depth obstacle points + floor_obstacles' blind wall) -> roomba_driver (bump stop + backup reflex,
  max wheel speed) -> wheel velocity targets (randomised gains, torque limit, latency)

and builds the policy's observation from what the robot has at runtime:

  local costmap patch (24 x 24 x 0.1 m, robot frame, lethal cells): spatio-temporal marks from a
    depth ray fan (0.3-2.0 m, 5 s decay, frustum clearing) + explorer bump keep-out strips (+
    marks the tile starts with)
  light bumpers (6 IR rays, proximity 0..1), bumper switches, wheel-encoder v / w, IMU yaw rate,
  previous action, motion over the last 1 s (VSLAM pose), goal direction / distance, time used

The observation layout is mirrored by src/roomba_rs/src/learned/unstuck.rs.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import scenarios as S
import world as W

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_USD = os.path.join(HERE, "assets", "roomba_rl.usda")

# ---- robot / stack constants (real values) ----
WHEEL_R = 0.036
WHEEL_BASE = 0.235
V_MAX, V_MIN, W_MAX = 0.25, -0.15, 0.8  # velocity_smoother max/min
ACC_V, DEC_V, ACC_W, DEC_W = 0.35, 0.5, 1.5, 2.0
WHEEL_MAX = 0.4  # driver max_wheel_speed_m_s
CAM_X, CAM_Z = 0.153, 0.13
DEPTH_MIN, DEPTH_MAX, OBST_RANGE = 0.3, 2.5, 2.0
OBST_ZMIN, OBST_ZMAX = 0.06, 1.0
DECAY_S = 5.0
HFOV = math.radians(85.0)
FAN_COLS = 24
FAN_ELEV = [math.radians(e) for e in (-25.0, -15.0, -8.0, -3.0, 3.0, 12.0, 22.0)]
LIGHT_ANGLES = [1.2, 0.65, 0.2, -0.2, -0.65, -1.2]
CLIFF_ANGLES = [1.05, 0.35, -0.35, -1.05]
IR_RANGE = 0.12
LIGHT_STOP = 0.85  # front-left proximity that the driver treats as a bump (signal 200 of ~235)
PATCH = 24
PATCH_RES = 0.1
WINDOW = 1.5  # local costmap rolling window half size
OBS_EXTRA = 20
OBS_DIM = PATCH * PATCH + OBS_EXTRA


class TilesImporter(TerrainImporter):
    """Imports the stuck tiles' collision mesh instead of a ground plane."""

    def import_ground_plane(self, name: str, size=None):
        self.import_mesh(name, W.load_world(self.cfg.tiles_path).collision)


@configclass
class TilesImporterCfg(TerrainImporterCfg):
    class_type: type = TilesImporter
    tiles_path: str = ""


ROBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            PhysxRigidBodyCfg(max_depenetration_velocity=1.0, max_linear_velocity=100.0, max_angular_velocity=3000.0),
        ],
        articulation_props=[
            PhysxArticulationCfg(enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"], stiffness=0.0, damping=0.5, effort_limit_sim=0.8, velocity_limit_sim=12.0
        )
    },
)


@configclass
class UnstuckSceneCfg(InteractiveSceneCfg):
    terrain = TilesImporterCfg(
        prim_path="/World/tiles",
        terrain_type="plane",
        collision_group=-1,
        physics_material=PhysxRigidBodyMaterialCfg(
            static_friction=0.1, dynamic_friction=0.1, restitution=0.0
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG
    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/(base_link|top_link)", history_length=0)


@configclass
class UnstuckEnvCfg(DirectRLEnvCfg):
    decimation = 10
    episode_length_s = 15.0
    action_space = 2
    observation_space = OBS_DIM
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=10, physics=OvPhysxCfg())
    scene: UnstuckSceneCfg = UnstuckSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)
    tiles_path: str = os.path.join(HERE, "data", "tiles_train.json")
    # evaluation: env i always replays tile i % T from its fixed start; no randomised extras
    eval_mode: bool = False
    # controller under test in eval: "policy" (actions from outside) or "classical" (baseline.py)
    # (the env itself only needs to know for logging)
    # domain randomisation
    wheel_gain_range = (0.85, 1.05)
    effort_range = (0.5, 1.0)
    latency_prob = 0.5
    ir_gain_range = (0.7, 1.3)
    depth_dropout = 0.05
    patch_flip = 0.01
    ignore_right_bumper_prob = 0.5
    # rewards
    rew_success = 10.0
    rew_progress = 2.0
    rew_clear = 3.0
    rew_bump = -1.0
    rew_contact = -0.1
    rew_top = -0.5
    rew_cliff = -10.0
    rew_time = -0.05
    rew_jerk = -0.02


def _yaw_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)


def patch_offsets(device) -> torch.Tensor:
    """[P*P*4, 2] sample points (forward u, left v) of the robot-frame patch: row 0 = farthest
    ahead, col 0 = leftmost, 2 x 2 sub-samples per cell (max of the four)."""
    idx = torch.arange(PATCH, device=device, dtype=torch.float32)
    u = (PATCH / 2 - 0.5 - idx) * PATCH_RES
    v = (PATCH / 2 - 0.5 - idx) * PATCH_RES
    U, V = torch.meshgrid(u, v, indexing="ij")
    q = PATCH_RES / 4
    subs = torch.tensor([[q, q], [q, -q], [-q, q], [-q, -q]], device=device)
    pts = torch.stack([U, V], -1).reshape(-1, 1, 2) + subs.reshape(1, 4, 2)
    return pts.reshape(-1, 2)


class UnstuckEnv(DirectRLEnv):
    cfg: UnstuckEnvCfg

    def __init__(self, cfg: UnstuckEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.scene.terrain.tiles_path = cfg.tiles_path
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        N = self.num_envs
        self.robot = self.scene["robot"]
        self.contact = self.scene["contact"]
        self.wheel_ids, _ = self.robot.find_joints([".*_wheel_joint"])
        names = self.robot.joint_names
        self.left_first = names[self.wheel_ids[0]].startswith("left")
        self._wheel_ids_t = torch.tensor(self.wheel_ids, device=dev, dtype=torch.int32)
        cnames = self.contact.body_names
        self.base_ci = [i for i, n in enumerate(cnames) if n == "base_link"][0]
        self.top_ci = [i for i, n in enumerate(cnames) if n == "top_link"][0]

        w = W.load_world(self.cfg.tiles_path)
        self.world = w
        self.T = len(w.tiles)
        self.tile_center = torch.tensor(w.centers, device=dev)
        self.tile_clear = torch.tensor(w.clearance, device=dev)
        self.tile_marks = torch.tensor(w.marks, device=dev)
        self.tile_holes = torch.tensor(np.nan_to_num(w.holes, nan=1e6), device=dev)
        self.tile_start = torch.tensor(w.starts, device=dev)
        self.tile_goal = torch.tensor(w.goals, device=dev)
        kinds = sorted(set(w.kinds))
        self.kind_names = kinds
        self.tile_kind = torch.tensor([kinds.index(k) for k in w.kinds], device=dev)
        self.depth_mesh = convert_to_warp_mesh(np.asarray(w.depth.vertices, np.float32), np.asarray(w.depth.faces, np.int32), dev)
        self.ir_mesh = convert_to_warp_mesh(np.asarray(w.ir.vertices, np.float32), np.asarray(w.ir.faces, np.int32), dev)

        # ray patterns (robot frame)
        az = torch.linspace(HFOV / 2, -HFOV / 2, FAN_COLS, device=dev)  # left to right
        el = torch.tensor(FAN_ELEV, device=dev)
        A, E = torch.meshgrid(az, el, indexing="ij")  # [cols, rows]
        self.fan_dir = torch.stack([torch.cos(E) * torch.cos(A), torch.cos(E) * torch.sin(A), torch.sin(E)], -1).reshape(-1, 3)
        self.fan_col = torch.arange(FAN_COLS, device=dev).repeat_interleave(len(FAN_ELEV))
        la = torch.tensor(LIGHT_ANGLES, device=dev)
        self.ir_origin = torch.stack([0.155 * torch.cos(la), 0.155 * torch.sin(la), torch.full_like(la, 0.035)], -1)
        self.ir_dir = torch.stack([torch.cos(la), torch.sin(la), torch.zeros_like(la)], -1)
        ca = torch.tensor(CLIFF_ANGLES, device=dev)
        self.cliff_pts = torch.stack([0.15 * torch.cos(ca), 0.15 * torch.sin(ca)], -1)
        self.patch_pts = patch_offsets(dev)
        g = (torch.arange(S.GRID_N, device=dev, dtype=torch.float32) + 0.5) * S.GRID_RES - S.TILE / 2
        GY, GX = torch.meshgrid(g, g, indexing="ij")
        self.grid_xy = torch.stack([GX.reshape(-1), GY.reshape(-1)], -1)  # [G, 2] tile frame (row = y)

        # per-env state
        f = lambda *s: torch.zeros(*s, device=dev)
        self.tile = torch.zeros(N, dtype=torch.long, device=dev)
        self.goal_w = f(N, 2)
        self.mark_t = torch.full((N, S.GRID_N * S.GRID_N), -1e9, device=dev)
        self.keepout = torch.zeros(N, S.GRID_N * S.GRID_N, dtype=torch.bool, device=dev)
        self.t = f(N)
        self.act = f(N, 2)
        self.prev_act = f(N, 2)
        self.delayed_act = f(N, 2)
        self.sm = f(N, 2)  # smoother output (v, w)
        self.cm_stop = torch.zeros(N, dtype=torch.bool, device=dev)
        self.cm_slow = torch.zeros(N, dtype=torch.bool, device=dev)
        self.reflex_steps = torch.zeros(N, dtype=torch.long, device=dev)  # remaining
        self.reflex_side = f(N)
        self.reflex_times = torch.full((N, 4), -1e9, device=dev)
        self.bumped = torch.zeros(N, dtype=torch.bool, device=dev)
        self.top_hit = torch.zeros(N, dtype=torch.bool, device=dev)
        self.bump_new = torch.zeros(N, dtype=torch.bool, device=dev)
        self.bump_bearing = f(N)
        self.bump_lr = f(N, 2)
        self.ir = f(N, 6)
        self.wheel_targets = f(N, 2)
        self.cmd_out = f(N, 2)
        self.ep_goal0 = f(N)
        self.clear_now = f(N)
        self.hist = f(N, 11, 3)  # pose history (x, y, yaw) at 10 Hz
        self.prev_goal_d = f(N)
        self.prev_clear = f(N)
        # domain randomisation
        self.wheel_gain = torch.ones(N, 2, device=dev)
        self.delay = torch.zeros(N, dtype=torch.bool, device=dev)
        self.ir_gain = torch.ones(N, 6, device=dev)
        self.ignore_right = torch.zeros(N, dtype=torch.bool, device=dev)
        self.gyro_bias = f(N)
        # episode stats
        self.ep_bumps = f(N)
        self.ep_top = f(N)
        self.ep_ret = f(N)
        self.outcome = torch.zeros(N, dtype=torch.long, device=dev)  # 1 success 2 cliff 3 fall 4 timeout
        self.eval_episodes = 0
        self.eval_tile_cursor = 0
        self.eval_log: list = []
        self._randomise_static()

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        pass  # everything comes from UnstuckSceneCfg

    def _randomise_static(self):
        N, dev = self.num_envs, self.device
        c = self.cfg
        if c.eval_mode:
            return
        eff = torch.empty(N, 2, device=dev).uniform_(*c.effort_range)
        try:
            self.robot.write_joint_effort_limit_to_sim_index(limits=eff, joint_ids=self._wheel_ids_t)
        except Exception as e:  # noqa: BLE001
            print(f"[unstuck] effort-limit randomisation unavailable: {e}")

    # ------------------------------------------------------------------ pose helpers
    def _pose(self):
        p = self.robot.data.root_pos_w.torch
        yaw = self.robot.data.heading_w.torch
        return p, yaw

    def _local(self, p):
        return p[:, :2] - self.tile_center[self.tile]

    def _clearance(self, local):
        c = ((local + S.TILE / 2) / S.GRID_RES).long().clamp(0, S.GRID_N - 1)
        return self.tile_clear[self.tile, c[:, 1], c[:, 0]]

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor):
        a = torch.nan_to_num(actions).clamp(-1.0, 1.0)
        self.prev_act = self.act.clone()
        self.act = a
        use = torch.where(self.delay.unsqueeze(-1), self.delayed_act, a)
        self.delayed_act = a
        self._drive(use)

    def command_from_action(self, a):
        v = torch.where(a[:, 0] > 0, a[:, 0] * V_MAX, -a[:, 0] * V_MIN)
        return v, a[:, 1] * W_MAX

    def _drive(self, a):
        dt = self.step_dt
        v_cmd, w_cmd = self.command_from_action(a)
        # velocity_smoother (open loop)
        v0, w0 = self.sm[:, 0], self.sm[:, 1]
        dv_max = torch.where(v_cmd.abs() > v0.abs(), ACC_V, DEC_V) * dt
        dw_max = torch.where(w_cmd.abs() > w0.abs(), ACC_W, DEC_W) * dt
        v = v0 + torch.clamp(v_cmd - v0, -dv_max, dv_max)
        w = w0 + torch.clamp(w_cmd - w0, -dw_max, dw_max)
        self.sm = torch.stack([v, w], -1)
        # collision_monitor: velocity polygons act only while driving forward
        fwd = v >= 0.01
        stop = fwd & self.cm_stop
        slow = fwd & self.cm_slow & ~stop
        v = torch.where(stop, 0.0, torch.where(slow, v * 0.7, v))
        w = torch.where(stop, 0.0, torch.where(slow, w * 0.7, w))
        # driver: wheel speeds, bump reflex overrides
        vl = v - w * WHEEL_BASE / 2
        vr = v + w * WHEEL_BASE / 2
        rs = self.reflex_steps
        backing = rs > 5
        turning = (rs > 0) & ~backing
        side = self.reflex_side  # +1 hit on the left -> turn right (clockwise)
        vl = torch.where(backing, -0.1, torch.where(turning, 0.1 * side, vl))
        vr = torch.where(backing, -0.1, torch.where(turning, -0.1 * side, vr))
        self.reflex_steps = (rs - 1).clamp(min=0)
        vl = vl.clamp(-WHEEL_MAX, WHEEL_MAX) * self.wheel_gain[:, 0]
        vr = vr.clamp(-WHEEL_MAX, WHEEL_MAX) * self.wheel_gain[:, 1]
        wl, wr = vl / WHEEL_R, vr / WHEEL_R
        self.wheel_targets = torch.stack([wl, wr] if self.left_first else [wr, wl], -1)
        self.cmd_out = torch.stack([(vl + vr) / 2, (vr - vl) / WHEEL_BASE], -1)

    def _apply_action(self):
        self.robot.set_joint_velocity_target_index(target=self.wheel_targets, joint_ids=self._wheel_ids_t)

    # ------------------------------------------------------------------ dones / rewards
    def _get_dones(self):
        dev = self.device
        self.t += self.step_dt
        p, yaw = self._pose()
        local = self._local(p)
        # contacts
        F = self.contact.data.net_forces_w.torch  # [N, B, 3]
        fb = F[:, self.base_ci]
        c, s = torch.cos(yaw), torch.sin(yaw)
        fx = c * fb[:, 0] + s * fb[:, 1]
        fy = -s * fb[:, 0] + c * fb[:, 1]
        fh = torch.hypot(fx, fy)
        bearing = torch.atan2(-fy, -fx)
        switch = (fh > 1.5) & (bearing.abs() < 1.75)
        left = switch & (bearing > -0.25)
        right = switch & (bearing < 0.25) & ~self.ignore_right
        light = self.ir[:, 1] >= LIGHT_STOP
        bumped = left | right | light
        self.bump_new = bumped & ~self.bumped
        self.bumped = bumped
        self.bump_lr = torch.stack([left | light, right], -1).float()
        self.bump_bearing = torch.where(switch, bearing, torch.full_like(bearing, 0.65))
        top = F[:, self.top_ci].norm(dim=-1) > 1.5
        top_new = top & ~self.top_hit
        self.top_hit = top
        self.contact_any = (fh > 1.5) | top
        # driver reflex on a new bump while driving forward (rate limited)
        fwd = self.cmd_out[:, 0] > 0.01
        trig = self.bump_new & fwd & (self.reflex_steps == 0)
        recent = (self.t.unsqueeze(-1) - self.reflex_times < 30.0).sum(-1)
        last = self.reflex_times.max(-1).values
        allowed = trig & (recent < 4) & (self.t - last >= 3.0)
        side = torch.where(left & ~right, 1.0, torch.where(right & ~left, -1.0, 1.0))
        self.reflex_side = torch.where(allowed, side, self.reflex_side)
        self.reflex_steps = torch.where(allowed, torch.full_like(self.reflex_steps, 13), self.reflex_steps)
        slot = self.reflex_times.argmin(-1)
        upd = self.reflex_times.clone()
        upd[torch.arange(self.num_envs, device=dev), slot] = self.t
        self.reflex_times = torch.where(allowed.unsqueeze(-1), upd, self.reflex_times)
        # stop on a bump the reflex doesn't take (smoother state reset = immediate stop)
        self.sm = torch.where((trig & ~allowed).unsqueeze(-1), torch.zeros_like(self.sm), self.sm)
        self.ep_bumps += self.bump_new.float()
        self.ep_top += top_new.float()
        self._top_new = top_new
        # outcome
        clear = self._clearance(local)
        self.clear_now = clear
        cp = local.unsqueeze(1) + torch.stack(
            [c.unsqueeze(-1) * self.cliff_pts[:, 0] - s.unsqueeze(-1) * self.cliff_pts[:, 1],
             s.unsqueeze(-1) * self.cliff_pts[:, 0] + c.unsqueeze(-1) * self.cliff_pts[:, 1]], -1)  # [N, 4, 2]
        h = self.tile_holes[self.tile]  # [N, H, 4]
        inside = (cp[:, :, None, 0] >= h[:, None, :, 0]) & (cp[:, :, None, 0] <= h[:, None, :, 2]) & \
                 (cp[:, :, None, 1] >= h[:, None, :, 1]) & (cp[:, :, None, 1] <= h[:, None, :, 3])
        cliff = inside.any(-1).any(-1)
        grav = self.robot.data.projected_gravity_b.torch
        fall = (p[:, 2] < -0.08) | (grav[:, 2] > -0.8) | (local.abs().max(-1).values > S.TILE / 2)
        success = clear >= S.SUCCESS_CLEARANCE
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        self.success, self.cliff, self.fall = success, cliff, fall
        terminated = success | cliff | fall
        return terminated, timeout & ~terminated

    def _get_rewards(self):
        c = self.cfg
        p, _ = self._pose()
        gd = torch.norm(p[:, :2] - self.goal_w, dim=-1)
        clear = self.clear_now
        r = c.rew_progress * (self.prev_goal_d - gd).clamp(-0.1, 0.1)
        r += c.rew_clear * (clear.clamp(max=S.SUCCESS_CLEARANCE) - self.prev_clear.clamp(max=S.SUCCESS_CLEARANCE))
        r += c.rew_bump * self.bump_new.float() + c.rew_contact * self.contact_any.float() + c.rew_top * self._top_new.float()
        r += c.rew_time
        da = self.act - self.prev_act
        r += c.rew_jerk * (da ** 2).sum(-1)
        r += c.rew_success * self.success.float() + c.rew_cliff * (self.cliff | self.fall).float()
        self.prev_goal_d = gd
        self.prev_clear = clear
        self.ep_ret += r
        # outcome bookkeeping for the envs that end now
        done = self.reset_terminated | self.reset_time_outs
        out = torch.where(self.success, 1, torch.where(self.cliff, 2, torch.where(self.fall, 3, 4)))
        self.outcome = torch.where(done, out, torch.zeros_like(out))
        if done.any():
            ids = done.nonzero().squeeze(-1)
            log = self.extras.setdefault("log", {})
            o = self.outcome[ids]
            log["Episode/success"] = (o == 1).float().mean()
            log["Episode/cliff_or_fall"] = ((o == 2) | (o == 3)).float().mean()
            log["Episode/timeout"] = (o == 4).float().mean()
            log["Episode/bumps"] = self.ep_bumps[ids].mean()
            log["Episode/top_contacts"] = self.ep_top[ids].mean()
            log["Episode/time_s"] = self.t[ids].mean()
            if self.cfg.eval_mode:
                for i in ids.tolist():
                    self.eval_log.append({
                        "tile": int(self.tile[i]), "kind": self.kind_names[int(self.tile_kind[self.tile[i]])],
                        "outcome": ["", "success", "cliff", "fall", "timeout"][int(self.outcome[i])],
                        "time_s": float(self.t[i]), "bumps": int(self.ep_bumps[i]), "top": int(self.ep_top[i]),
                        "goal_progress_m": float(self.ep_goal0[i] - gd[i]), "return": float(self.ep_ret[i]),
                    })
        return r

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(ids)
        dev = self.device
        if self.cfg.eval_mode:
            tiles = (torch.arange(n, device=dev) + self.eval_tile_cursor) % self.T
            self.eval_tile_cursor += n
        else:
            tiles = torch.randint(0, self.T, (n,), device=dev)
        self.tile[ids] = tiles
        st = self.tile_start[tiles]
        jitter = 0.0 if self.cfg.eval_mode else 1.0
        yaw = st[:, 2] + jitter * torch.empty(n, device=dev).uniform_(-0.08, 0.08)
        pos = torch.zeros(n, 3, device=dev)
        pos[:, :2] = self.tile_center[tiles] + st[:, :2]
        pos[:, 2] = 0.003
        pose = torch.cat([pos, _yaw_quat_xyzw(yaw)], -1)
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
        jp = self.robot.data.default_joint_pos.torch[ids].clone()
        self.robot.write_joint_position_to_sim_index(position=jp, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp), env_ids=ids)
        self.goal_w[ids] = self.tile_center[tiles] + self.tile_goal[tiles]
        self.mark_t[ids] = -1e9
        self.keepout[ids] = self.tile_marks[tiles].reshape(n, -1)
        self.t[ids] = 0.0
        for b in (self.act, self.prev_act, self.delayed_act, self.sm, self.bump_lr, self.ir, self.cmd_out):
            b[ids] = 0.0
        self.cm_stop[ids] = False
        self.cm_slow[ids] = False
        self.reflex_steps[ids] = 0
        self.reflex_times[ids] = -1e9
        self.bumped[ids] = False
        self.top_hit[ids] = False
        self.hist[ids] = torch.cat([pos[:, :2], yaw.unsqueeze(-1)], -1).unsqueeze(1)
        gd = torch.norm(pos[:, :2] - self.goal_w[ids], dim=-1)
        self.prev_goal_d[ids] = gd
        self.ep_goal0[ids] = gd
        local = st[:, :2]
        c = ((local + S.TILE / 2) / S.GRID_RES).long().clamp(0, S.GRID_N - 1)
        self.prev_clear[ids] = self.tile_clear[tiles, c[:, 1], c[:, 0]]
        self.ep_bumps[ids] = 0
        self.ep_top[ids] = 0
        self.ep_ret[ids] = 0
        if self.cfg.eval_mode:
            self.wheel_gain[ids] = 0.95
            self.delay[ids] = False
            self.ir_gain[ids] = 1.0
            self.ignore_right[ids] = True  # as on the real robot
            self.gyro_bias[ids] = 0.0
        else:
            self.wheel_gain[ids] = torch.empty(n, 2, device=dev).uniform_(*self.cfg.wheel_gain_range)
            self.delay[ids] = torch.rand(n, device=dev) < self.cfg.latency_prob
            self.ir_gain[ids] = torch.empty(n, 6, device=dev).uniform_(*self.cfg.ir_gain_range)
            self.ignore_right[ids] = torch.rand(n, device=dev) < self.cfg.ignore_right_bumper_prob
            self.gyro_bias[ids] = torch.randn(n, device=dev) * 0.01

    # ------------------------------------------------------------------ sensing + observations
    def _sense(self):
        dev = self.device
        N = self.num_envs
        p, yaw = self._pose()
        c, s = torch.cos(yaw), torch.sin(yaw)

        def rot(v):  # robot frame [*, R, 3] -> world
            x = c[:, None] * v[..., 0] - s[:, None] * v[..., 1]
            y = s[:, None] * v[..., 0] + c[:, None] * v[..., 1]
            return torch.stack([x, y, v[..., 2]], -1)

        # depth fan
        R = self.fan_dir.shape[0]
        cam = p.clone()
        cam[:, 0] += c * CAM_X
        cam[:, 1] += s * CAM_X
        cam[:, 2] = p[:, 2] + CAM_Z
        dirs = rot(self.fan_dir.unsqueeze(0).expand(N, R, 3))
        starts = cam.unsqueeze(1).expand(N, R, 3)
        _, dist, _, _ = raycast_mesh(starts.reshape(-1, 3), dirs.reshape(-1, 3).contiguous(), self.depth_mesh, max_dist=DEPTH_MAX + 0.5, return_distance=True)
        dist = dist.reshape(N, R)
        if not self.cfg.eval_mode and self.cfg.depth_dropout > 0:
            dist = torch.where(torch.rand_like(dist) < self.cfg.depth_dropout, torch.full_like(dist, float("inf")), dist)
        valid = (dist >= DEPTH_MIN) & (dist <= DEPTH_MAX)
        hit = starts + dirs * dist.clamp(max=10.0).unsqueeze(-1)
        hz = hit[..., 2] - p[:, 2:3]
        obst = valid & (hz > OBST_ZMIN) & (hz < OBST_ZMAX)
        blind = valid.sum(-1) < 2
        # robot-frame obstacle points for the collision monitor
        rel = hit[..., :2] - p[:, None, :2]
        rx = c[:, None] * rel[..., 0] + s[:, None] * rel[..., 1]
        ry = -s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]
        in_stop = obst & (rx >= 0.17) & (rx <= 0.35) & (ry.abs() <= 0.19)
        in_slow = obst & (rx >= 0.17) & (rx <= 0.62) & (ry.abs() <= 0.22)
        self.cm_stop = in_stop.any(-1) | blind  # blind wall: 0.15 m ahead of the camera
        self.cm_slow = in_slow.any(-1)
        # costmap marking (tile frame grid)
        tl = self.tile_center[self.tile]
        mark = obst & (dist <= OBST_RANGE)
        loc = hit[..., :2] - tl[:, None, :]
        ci = ((loc + S.TILE / 2) / S.GRID_RES).long()
        ok = mark & (ci >= 0).all(-1) & (ci < S.GRID_N).all(-1)
        flat = (ci[..., 1] * S.GRID_N + ci[..., 0]).clamp(0, S.GRID_N * S.GRID_N - 1)
        # frustum clearing first: cells in view nearer than the nearest obstacle in their column
        hdist = torch.where(obst, dist * torch.cos(torch.tensor(FAN_ELEV, device=dev)).repeat(FAN_COLS).unsqueeze(0), torch.full_like(dist, 9.0))
        colmin = torch.full((N, FAN_COLS), 9.0, device=dev).scatter_reduce(1, self.fan_col.unsqueeze(0).expand(N, R), hdist, "amin")
        camloc = cam[:, :2] - tl
        d = self.grid_xy.unsqueeze(0) - camloc.unsqueeze(1)  # [N, G, 2]
        rng_ = d.norm(dim=-1)
        b = torch.atan2(d[..., 1], d[..., 0]) - yaw.unsqueeze(-1)
        b = torch.atan2(torch.sin(b), torch.cos(b))
        col = ((HFOV / 2 - b) / HFOV * FAN_COLS).long().clamp(0, FAN_COLS - 1)
        inview = (b.abs() < 0.7) & (rng_ >= DEPTH_MIN) & (rng_ <= 3.0)
        clear_ = inview & (rng_ < torch.gather(colmin, 1, col) - 0.07)
        self.mark_t = torch.where(clear_, torch.full_like(self.mark_t, -1e9), self.mark_t)
        tt = self.t.unsqueeze(-1).expand(N, R)
        self.mark_t.scatter_reduce_(1, torch.where(ok, flat, torch.zeros_like(flat)), torch.where(ok, tt, torch.full_like(tt, -1e9)), "amax")
        # explorer bump keep-out strips on new bumps
        if self.bump_new.any():
            ids = self.bump_new.nonzero().squeeze(-1)
            a = yaw[ids] + self.bump_bearing[ids]
            ca_, sa_ = torch.cos(a), torch.sin(a)
            uv = torch.tensor([[0.18, -0.25], [0.24, -0.25], [0.24, 0.25], [0.18, 0.25]], device=dev)
            px = (p[ids, 0] - tl[ids, 0]).unsqueeze(-1) + uv[:, 0] * ca_.unsqueeze(-1) - uv[:, 1] * sa_.unsqueeze(-1)
            py = (p[ids, 1] - tl[ids, 1]).unsqueeze(-1) + uv[:, 0] * sa_.unsqueeze(-1) + uv[:, 1] * ca_.unsqueeze(-1)
            quad = torch.stack([px, py], -1)  # [E, 4, 2]
            g = self.grid_xy.unsqueeze(0)
            inside = torch.ones(len(ids), g.shape[1], dtype=torch.bool, device=dev)
            for k in range(4):
                a0, a1 = quad[:, k], quad[:, (k + 1) % 4]
                cross = (a1[:, 0] - a0[:, 0]).unsqueeze(-1) * (g[..., 1] - a0[:, 1].unsqueeze(-1)) - (a1[:, 1] - a0[:, 1]).unsqueeze(-1) * (g[..., 0] - a0[:, 0].unsqueeze(-1))
                inside &= cross >= -1e-6
            self.keepout[ids] |= inside
        # light bumpers
        io = rot(self.ir_origin.unsqueeze(0).expand(N, 6, 3)) + p.unsqueeze(1)
        io[..., 2] = p[:, 2:3] + 0.035
        idir = rot(self.ir_dir.unsqueeze(0).expand(N, 6, 3))
        _, idist, _, _ = raycast_mesh(io.reshape(-1, 3).contiguous(), idir.reshape(-1, 3).contiguous(), self.ir_mesh, max_dist=IR_RANGE, return_distance=True)
        idist = idist.reshape(N, 6)
        prox = (1.0 - idist / IR_RANGE).clamp(0, 1)
        prox = torch.nan_to_num(prox, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.cfg.eval_mode:
            prox = (prox * self.ir_gain + 0.02 * torch.randn_like(prox)).clamp(0, 1)
        self.ir = prox

    def costmap_patch(self) -> torch.Tensor:
        """[N, P*P] lethal patch sampled like roomba_rs::learned::patch (window +-1.5 m)."""
        N = self.num_envs
        p, yaw = self._pose()
        c, s = torch.cos(yaw), torch.sin(yaw)
        uv = self.patch_pts
        wx = c[:, None] * uv[:, 0] - s[:, None] * uv[:, 1]
        wy = s[:, None] * uv[:, 0] + c[:, None] * uv[:, 1]
        inwin = (wx.abs() <= WINDOW) & (wy.abs() <= WINDOW)
        loc = p[:, :2] - self.tile_center[self.tile]
        gx = ((loc[:, 0:1] + wx + S.TILE / 2) / S.GRID_RES).floor().long()
        gy = ((loc[:, 1:2] + wy + S.TILE / 2) / S.GRID_RES).floor().long()
        ok = inwin & (gx >= 0) & (gx < S.GRID_N) & (gy >= 0) & (gy < S.GRID_N)
        flat = (gy.clamp(0, S.GRID_N - 1) * S.GRID_N + gx.clamp(0, S.GRID_N - 1))
        lethal = ((self.t.unsqueeze(-1) - self.mark_t) < DECAY_S) | self.keepout
        v = torch.gather(lethal, 1, flat) & ok
        return v.reshape(N, PATCH * PATCH, 4).any(-1).float()

    def _get_observations(self) -> dict:
        self._sense()
        N = self.num_envs
        p, yaw = self._pose()
        patch = self.costmap_patch()
        if not self.cfg.eval_mode and self.cfg.patch_flip > 0:
            flip = torch.rand_like(patch) < self.cfg.patch_flip
            patch = torch.where(flip, 1.0 - patch, patch)
        jv = self.robot.data.joint_vel.torch[:, self.wheel_ids]
        wl, wr = (jv[:, 0], jv[:, 1]) if self.left_first else (jv[:, 1], jv[:, 0])
        v_enc = (wl + wr) * WHEEL_R / 2
        w_enc = (wr - wl) * WHEEL_R / WHEEL_BASE
        yaw_rate = self.robot.data.root_link_ang_vel_w.torch[:, 2] + self.gyro_bias
        if not self.cfg.eval_mode:
            yaw_rate = yaw_rate + 0.01 * torch.randn_like(yaw_rate)
        # pose history (10 Hz): motion over the last 1 s in the current robot frame
        cur = torch.cat([p[:, :2], yaw.unsqueeze(-1)], -1)
        self.hist = torch.cat([self.hist[:, 1:], cur.unsqueeze(1)], 1)
        old = self.hist[:, 0]
        dxy = cur[:, :2] - old[:, :2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        mdx = c * dxy[:, 0] + s * dxy[:, 1]
        mdy = -s * dxy[:, 0] + c * dxy[:, 1]
        mdyaw = torch.atan2(torch.sin(cur[:, 2] - old[:, 2]), torch.cos(cur[:, 2] - old[:, 2]))
        g = self.goal_w - p[:, :2]
        gx = c * g[:, 0] + s * g[:, 1]
        gy = -s * g[:, 0] + c * g[:, 1]
        gd = torch.hypot(gx, gy)
        gb = torch.atan2(gy, gx)
        extra = torch.stack(
            [
                *self.ir.unbind(-1),
                self.bump_lr[:, 0], torch.where(self.ignore_right, 0.0, self.bump_lr[:, 1]),
                v_enc / V_MAX, w_enc / W_MAX, yaw_rate / W_MAX,
                self.act[:, 0], self.act[:, 1],
                mdx / 0.25, mdy / 0.25, mdyaw / W_MAX,
                (gd / 3.0).clamp(max=1.0), torch.cos(gb), torch.sin(gb),
                self.t / self.cfg.episode_length_s,
            ],
            -1,
        )
        obs = torch.cat([patch, extra], -1)
        self.last_obs = obs
        return {"policy": obs}


# ---------------------------------------------------------------- gym registration
import gymnasium as gym  # noqa: E402

gym.register(
    id="Roomba-Unstuck-v0",
    entry_point=f"{__name__}:UnstuckEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:UnstuckEnvCfg"},
)
