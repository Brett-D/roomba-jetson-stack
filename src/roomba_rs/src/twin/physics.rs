//! The simulated Roomba 550: differential drive from the wheel speeds the Open Interface was told,
//! collisions with the house (it stops at what it hits, pushing against it with the bumper
//! pressed), bumper switches, light bumpers, wall sensor, cliff sensors, falling down stairs, and
//! the Ridgid pack's voltage under load.

use crate::battery::LI_ION_CURVE;
use crate::explorer::{CLIFF_ANGLES, LIGHT_ANGLES};

use super::rng::Rng;
use super::scenario::{BatteryCfg, RobotCfg};
use super::world::{Contact, What, World, P2};

/// Contact closer than this to the robot's edge counts as touching (bumper pressed).
const TOUCH_M: f64 = 0.003;
/// Light bumpers and the wall sensor look this far.
const IR_RANGE_M: f64 = 0.25;
/// Height band the light bumpers see.
const IR_BAND: (f64, f64) = (0.01, 0.08);
/// Cliff sensors sit this far from the centre, at CLIFF_ANGLES.
const CLIFF_RADIUS_M: f64 = 0.155;
/// Wall sensor direction (right side, a little forward).
const WALL_ANGLE: f64 = -1.2;

/// Signal (0-4095) of an IR proximity sensor with this gain at distance d (m): ~250 at 1.5 cm for
/// gain 1, ~10 at 15 cm, as the front-left light bumper reads on this unit.
pub fn ir_signal(d: f64, gain: f64) -> f64 {
    if !d.is_finite() || d >= IR_RANGE_M {
        return 0.0;
    }
    (gain * 0.285 / (d.max(0.0) + 0.019).powi(2)).min(4095.0)
}

/// Resting voltage of one Li-ion cell at this state of charge (LI_ION_CURVE inverted).
pub fn cell_voltage(soc: f64) -> f64 {
    let soc = soc.clamp(0.0, 1.0);
    for w in LI_ION_CURVE.windows(2) {
        let ((v0, f0), (v1, f1)) = (w[0], w[1]);
        if soc <= f1 {
            return v0 + (v1 - v0) * (soc - f0) / (f1 - f0);
        }
    }
    LI_ION_CURVE[LI_ION_CURVE.len() - 1].0
}

#[derive(Clone, Debug)]
pub struct Pack {
    pub cfg: BatteryCfg,
    pub soc: f64,
    /// Discharge current (A, positive).
    pub current: f64,
}

impl Pack {
    pub fn new(cfg: &BatteryCfg) -> Self {
        Pack { cfg: cfg.clone(), soc: cfg.soc.clamp(0.0, 1.0), current: cfg.idle_a }
    }

    pub fn step(&mut self, dt: f64, speed: f64, stalled: bool) {
        let c = &self.cfg;
        self.current = c.idle_a + c.drive_a * (speed.abs() / 0.25).min(2.0) + if stalled { c.stall_a } else { 0.0 };
        self.soc = (self.soc - self.current * dt / 3600.0 / c.capacity_ah.max(0.01) * c.drain_scale).max(0.0);
    }

    /// Terminal voltage under the present load (V).
    pub fn voltage(&self) -> f64 {
        self.cfg.cells as f64 * cell_voltage(self.soc) - self.current * self.cfg.resistance_ohm
    }
}

/// Something that happened during a physics step.
#[derive(Clone, Debug, PartialEq)]
pub enum StepEvent {
    /// The robot ran into something (new contact): (point, what, bearing rad relative to heading).
    Collision { point: P2, what: What, bearing: f64 },
    /// It drove over a drop-off and fell (wheel drops, can't move any more).
    Fell { cliff: usize },
}

#[derive(Clone, Debug)]
pub struct Robot {
    pub cfg: RobotCfg,
    /// True pose (map frame).
    pub x: f64,
    pub y: f64,
    pub th: f64,
    /// Commanded wheel speeds (m/s), from the Open Interface.
    pub target: (f64, f64),
    /// Actual wheel speeds (acceleration-limited).
    pub vl: f64,
    pub vr: f64,
    /// Body velocity achieved in the last step (m/s, rad/s) and forward acceleration (m/s^2).
    pub v: f64,
    pub w: f64,
    pub accel: f64,
    /// Encoder counts (unwrapped).
    pub enc: (f64, f64),
    /// Distance (mm) and angle (deg) since the last OI packets 19/20.
    pub odo_mm: f64,
    pub odo_deg: f64,
    pub contact: Option<Contact>,
    pub bump: (bool, bool),
    /// Pushing against something (wheels stalled).
    pub stalled: bool,
    pub cliff: [bool; 4],
    pub fallen: bool,
    pub light: [u16; 6],
    pub wall_signal: u16,
    pub distance_total: f64,
    pub turned_total: f64,
}

impl Robot {
    pub fn new(cfg: &RobotCfg, pose: [f64; 3]) -> Self {
        Robot {
            cfg: cfg.clone(),
            x: pose[0],
            y: pose[1],
            th: pose[2],
            target: (0.0, 0.0),
            vl: 0.0,
            vr: 0.0,
            v: 0.0,
            w: 0.0,
            accel: 0.0,
            enc: (0.0, 0.0),
            odo_mm: 0.0,
            odo_deg: 0.0,
            contact: None,
            bump: (false, false),
            stalled: false,
            cliff: [false; 4],
            fallen: false,
            light: [0; 6],
            wall_signal: 0,
            distance_total: 0.0,
            turned_total: 0.0,
        }
    }

    pub fn pose(&self) -> [f64; 3] {
        [self.x, self.y, self.th]
    }

    fn band(&self) -> (f64, f64) {
        (self.cfg.climb_m, self.cfg.height_m)
    }

    /// Clearance between the robot's edge and the nearest obstacle at p (negative = overlapping).
    fn clearance(&self, world: &World, p: P2) -> f64 {
        let r = self.cfg.radius_m;
        let (zlo, zhi) = self.band();
        world.nearest(p, r + 0.05, zlo, zhi).map_or(1.0, |c| c.dist - r)
    }

    pub fn teleport(&mut self, pose: [f64; 3]) {
        self.x = pose[0];
        self.y = pose[1];
        self.th = pose[2];
        self.vl = 0.0;
        self.vr = 0.0;
    }

    /// Advance by dt seconds.
    pub fn step(&mut self, world: &World, dt: f64) -> Vec<StepEvent> {
        let mut events = Vec::new();
        let a = self.cfg.max_accel_mps2 * dt;
        let (tl, tr) = if self.fallen { (0.0, 0.0) } else { self.target };
        self.vl += (tl - self.vl).clamp(-a, a);
        self.vr += (tr - self.vr).clamp(-a, a);
        let wb = self.cfg.wheel_base_m;
        let v_cmd = (self.vl + self.vr) / 2.0;
        let w_wheels = (self.vr - self.vl) / wb;
        let w = w_wheels * self.cfg.turn_efficiency;
        let th_mid = self.th + w * dt / 2.0;
        let (dx, dy) = (v_cmd * th_mid.cos() * dt, v_cmd * th_mid.sin() * dt);

        // Largest fraction of the translation that keeps the robot clear (rotation never collides:
        // the robot is round). Already overlapping (a door closed on it, a person walked into it):
        // any move that doesn't make it worse.
        let now_clear = self.clearance(world, [self.x, self.y]);
        let ok = |f: f64| {
            let c = self.clearance(world, [self.x + dx * f, self.y + dy * f]);
            c >= 0.0 || c >= now_clear - 1e-9
        };
        let mut f = 1.0;
        if (dx != 0.0 || dy != 0.0) && !ok(1.0) {
            let (mut lo, mut hi) = (0.0, 1.0);
            for _ in 0..12 {
                let mid = (lo + hi) / 2.0;
                if ok(mid) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            f = if ok(lo) { lo } else { 0.0 };
        }
        let blocked = f < 1.0 - 1e-9;
        self.x += dx * f;
        self.y += dy * f;
        self.th = (self.th + w * dt + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        let v = v_cmd * f;
        self.accel = if dt > 0.0 { (v - self.v) / dt } else { 0.0 };
        self.v = v;
        self.w = w;
        self.distance_total += (dx * f).hypot(dy * f);
        self.turned_total += (w * dt).abs();
        self.odo_mm += v * dt * 1000.0;
        self.odo_deg += (w * dt).to_degrees();
        // Encoders: the wheels turn as commanded while turning (the body scrubs); a blocked robot's
        // wheels stall on the forward part.
        let m_per_count = std::f64::consts::PI * self.cfg.wheel_diameter_m / self.cfg.encoder_counts_per_rev;
        let (wl, wr) = (v - w_wheels * wb / 2.0, v + w_wheels * wb / 2.0);
        self.enc.0 += wl * dt / m_per_count;
        self.enc.1 += wr * dt / m_per_count;
        self.stalled = blocked && v_cmd.abs() > 0.01;

        let r = self.cfg.radius_m;
        let (zlo, zhi) = self.band();
        let was_touching = self.contact.is_some();
        self.contact = world.nearest([self.x, self.y], r + TOUCH_M, zlo, zhi);
        let mut bump = (false, false);
        if let Some(c) = self.contact {
            let bearing = wrap((c.point[1] - self.y).atan2(c.point[0] - self.x) - self.th);
            if bearing.abs() <= std::f64::consts::FRAC_PI_2 {
                bump = if bearing.abs() < 0.35 { (true, true) } else if bearing > 0.0 { (true, false) } else { (false, true) };
            }
            if !was_touching && (blocked || now_clear > 0.0) {
                events.push(StepEvent::Collision { point: c.point, what: c.what, bearing });
            }
        }
        self.bump = bump;

        for (k, &(_, ang)) in CLIFF_ANGLES.iter().enumerate() {
            let a = self.th + ang;
            self.cliff[k] = world.cliff_at([self.x + CLIFF_RADIUS_M * a.cos(), self.y + CLIFF_RADIUS_M * a.sin()]).is_some();
        }
        if !self.fallen {
            if let Some(k) = world.cliff_at([self.x, self.y]) {
                self.fallen = true;
                self.vl = 0.0;
                self.vr = 0.0;
                events.push(StepEvent::Fell { cliff: k });
            }
        }
        events
    }

    /// Light bumpers and the wall sensor (call at the sensor rate).
    pub fn sense_ir(&mut self, world: &World, rng: &mut Rng) {
        let r = self.cfg.radius_m * 0.98;
        let moving = self.v.abs() > 0.01 || self.w.abs() > 0.05;
        let floor = if moving { self.cfg.light_floor_signal } else { 5.0 };
        for (k, &ang) in LIGHT_ANGLES.iter().enumerate() {
            let a = self.th + ang;
            let d = [a.cos(), a.sin()];
            let (s, _) = world.ir_ray([self.x + r * d[0], self.y + r * d[1]], d, IR_RANGE_M, IR_BAND.0, IR_BAND.1);
            let sig = ir_signal(s, self.cfg.light_gain[k]) + floor * self.cfg.light_gain[k] + rng.gauss(3.0);
            self.light[k] = sig.round().clamp(0.0, 4095.0) as u16;
        }
        let a = self.th + WALL_ANGLE;
        let d = [a.cos(), a.sin()];
        let (s, _) = world.ir_ray([self.x + r * d[0], self.y + r * d[1]], d, IR_RANGE_M, IR_BAND.0, IR_BAND.1);
        self.wall_signal = (ir_signal(s, 0.3) + rng.gauss(1.0)).round().clamp(0.0, 4095.0) as u16;
    }
}

pub fn wrap(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::twin::world::{Body, DepthVis, Grid, Kind, Shape};

    fn corridor() -> World {
        // 4 m x 2 m room, walls one cell thick.
        let (w, h) = (80, 40);
        let mut g = Grid::new(w, h, 0.05, [0.0, 0.0]);
        for i in 0..w {
            g.solid[i] = true;
            g.solid[(h - 1) * w + i] = true;
        }
        for j in 0..h {
            g.solid[j * w] = true;
            g.solid[j * w + w - 1] = true;
        }
        World::new(g)
    }

    fn run(robot: &mut Robot, world: &World, secs: f64) -> Vec<StepEvent> {
        let mut ev = Vec::new();
        for _ in 0..(secs / 0.005).round() as usize {
            ev.extend(robot.step(world, 0.005));
        }
        ev
    }

    #[test]
    fn drives_straight_and_turns_in_place_with_encoders() {
        let world = corridor();
        let mut r = Robot::new(&RobotCfg::default(), [1.0, 1.0, 0.0]);
        r.target = (0.2, 0.2);
        run(&mut r, &world, 2.0);
        // 2 s at 0.2 m/s minus the ramp (0.2 / 1.5 s at half speed).
        assert!((r.x - (1.0 + 0.4 - 0.2 * 0.2 / 1.5 / 2.0)).abs() < 0.005, "{}", r.x);
        assert!((r.y - 1.0).abs() < 1e-9 && r.th.abs() < 1e-9);
        let m_per_count = std::f64::consts::PI * 0.072 / 508.8;
        assert!((r.enc.0 * m_per_count - (r.x - 1.0)).abs() < 1e-6);
        r.target = (-0.1175, 0.1175); // 1 rad/s
        run(&mut r, &world, 1.0);
        assert!(r.th > 0.85 && r.th < 1.0, "{}", r.th);
        assert!(r.bump == (false, false) && r.contact.is_none());
    }

    #[test]
    fn wall_stops_the_robot_with_the_bumper_pressed() {
        let world = corridor();
        let mut r = Robot::new(&RobotCfg::default(), [3.0, 1.0, 0.0]);
        r.target = (0.25, 0.25);
        let ev = run(&mut r, &world, 4.0);
        // Wall cells start at x = 3.95: the robot's edge stops there.
        assert!((r.x - (3.95 - 0.17)).abs() < 0.002, "{}", r.x);
        assert_eq!(r.bump, (true, true));
        assert!(r.stalled);
        assert_eq!(ev.iter().filter(|e| matches!(e, StepEvent::Collision { what: What::Wall, .. })).count(), 1);
        // Backing off releases it.
        r.target = (-0.1, -0.1);
        run(&mut r, &world, 0.5);
        assert_eq!(r.bump, (false, false));
        // A glancing hit on the left reads as a left bump only.
        let mut r = Robot::new(&RobotCfg::default(), [2.0, 1.6, 1.0]);
        r.target = (0.2, 0.2);
        run(&mut r, &world, 3.0);
        assert_eq!(r.bump, (true, false));
    }

    #[test]
    fn turn_efficiency_scrubs_the_body_turn_not_the_encoders() {
        let world = corridor();
        let cfg = RobotCfg { turn_efficiency: 0.5, ..Default::default() };
        let mut r = Robot::new(&cfg, [2.0, 1.0, 0.0]);
        r.target = (-0.1175, 0.1175);
        run(&mut r, &world, 1.0);
        let m_per_count = std::f64::consts::PI * 0.072 / 508.8;
        let wheel_turn = (r.enc.1 - r.enc.0) * m_per_count / 0.235;
        assert!((r.th - wheel_turn / 2.0).abs() < 1e-6, "{} {}", r.th, wheel_turn);
    }

    #[test]
    fn light_bumpers_see_a_post_the_robot_is_about_to_hit() {
        let mut world = corridor();
        world.bodies.push(Body {
            name: "post".into(),
            kind: Kind::Obstacle,
            shape: Shape::Circle { c: [2.0 + 0.2, 1.0 + 0.04], r: 0.02 },
            zmin: 0.0,
            zmax: 0.9,
            depth: DepthVis::Invisible,
            ir: true,
            blocks: true,
            active: true,
        });
        let mut r = Robot::new(&RobotCfg::default(), [2.0, 1.0, 0.0]);
        let mut rng = Rng::new(1);
        r.sense_ir(&world, &mut rng);
        // Centre-left (0.2 rad) points at it ~2 cm away; front-left sees it too, the right side doesn't.
        assert!(r.light[2] > 80, "{:?}", r.light);
        assert!(r.light[5] < 20 && r.light[4] < 20, "{:?}", r.light);
        assert!(ir_signal(0.015, 1.0) > 240.0 && ir_signal(0.015, 1.0) < 260.0);
        assert!(ir_signal(0.15, 1.0) < 12.0 && ir_signal(0.3, 1.0) == 0.0);
    }

    #[test]
    fn cliffs_fire_the_sensors_then_the_robot_falls() {
        let mut world = corridor();
        world.cliffs.push(("stairs".into(), Shape::Polygon(vec![[2.5, 0.0], [4.0, 0.0], [4.0, 2.0], [2.5, 2.0]])));
        let mut r = Robot::new(&RobotCfg::default(), [2.2, 1.0, 0.0]);
        r.step(&world, 0.005);
        assert_eq!(r.cliff, [false; 4]);
        r.target = (0.2, 0.2);
        let mut seen_front = false;
        let mut fell = false;
        for _ in 0..1000 {
            for e in r.step(&world, 0.005) {
                fell |= matches!(e, StepEvent::Fell { .. });
            }
            if r.cliff[1] && r.cliff[2] && !fell {
                seen_front = true;
            }
            if fell {
                break;
            }
        }
        assert!(seen_front && fell && r.fallen);
        let x = r.x;
        run(&mut r, &world, 1.0);
        assert_eq!(r.x, x);
    }

    #[test]
    fn pack_voltage_follows_the_cell_curve_and_drains() {
        assert!((cell_voltage(0.5) - 3.79).abs() < 1e-9 && (cell_voltage(1.0) - 4.2).abs() < 1e-9);
        assert!((crate::battery::li_ion_fraction(cell_voltage(0.37)) - 0.37).abs() < 1e-9);
        let mut p = Pack::new(&BatteryCfg { soc: 0.5, capacity_ah: 1.0, drain_scale: 1.0, ..Default::default() });
        let v_idle = p.voltage();
        p.step(1.0, 0.25, false);
        assert!(p.voltage() < v_idle);
        let soc = p.soc;
        for _ in 0..3600 {
            p.step(1.0, 0.0, false);
        }
        assert!((soc - p.soc - 0.22).abs() < 0.001, "{}", soc - p.soc);
    }
}
