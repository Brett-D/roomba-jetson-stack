//! battery_learner: the robot's /battery_state from a self-learning gauge of the 18 V tool pack wired
//! to the Roomba's battery contacts (coulomb counting from 100 % at each pack swap, the capacity and
//! the voltage curve learned from the swaps at 1 bar; see roomba_rs::battery_learner).
//!
//! Subscribes: roomba/sensors (roomba_msgs/SensorState: voltage_mv, current_ma, header stamp)
//! Publishes:  battery_state (sensor_msgs/BatteryState) at publish_rate_hz while sensor data comes in
//!             (robot.launch.py battery_learner:=true moves the driver's own to roomba/battery_raw)
//! Services:   ~/pack_swapped (std_srvs/Trigger): a full pack was just put in
//! Files:      <state_dir>/battery_learner.json (written atomically every save_period_s, at swaps
//!             and at shutdown), <state_dir>/battery_pack_NNNN.csv (a discharge sample per
//!             sample_period_s)
//! Parameters: state_dir (~/.local/state/roomba), publish_rate_hz (1), save_period_s (30), and those
//!             of roomba_rs::battery_learner::Params

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use rclrs::*;
use ros_env::{builtin_interfaces, roomba_msgs, sensor_msgs, std_msgs, std_srvs};

use roomba_rs::battery::{STATUS_DISCHARGING, TECHNOLOGY_LION};
use roomba_rs::battery_learner::{append_sample, load_state, save_state, Learner, Params, STATE_FILE};
use roomba_rs::node_params::{block_signals, mono, on_signal, spin, NodeParams};
use roomba_rs::watchdog::ParamSource;

const STATUS_UNKNOWN: u8 = 0;
const HEALTH_UNKNOWN: u8 = 0;

struct Core {
    learner: Learner,
    node: Node,
    dir: PathBuf,
    path: PathBuf,
    save_period: f64,
    /// Stamp of the latest frame, for the published message.
    stamp: builtin_interfaces::msg::Time,
    /// Steady time of the latest new frame.
    last_fresh: f64,
    last_save: f64,
    dirty: bool,
    csv_failing: bool,
}

impl Core {
    fn save(&mut self) {
        match save_state(&self.path, &self.learner.state) {
            Ok(()) => {
                self.dirty = false;
                self.last_save = mono();
            }
            Err(e) => log!(self.node.warn().throttle(Duration::from_secs(60)), "cannot write {}: {e}", self.path.display()),
        }
    }

    fn info(&self, lines: Vec<String>) {
        for line in lines {
            log!(self.node.info(), "{line}");
        }
    }

    fn on_sensors(&mut self, m: &roomba_msgs::msg::SensorState) {
        let s = &m.header.stamp;
        let unix = unix_now();
        let stamp = if s.sec == 0 && s.nanosec == 0 { unix } else { s.sec as f64 + s.nanosec as f64 * 1e-9 };
        let out = self.learner.on_frame(stamp, unix, m.voltage_mv as i32, m.current_ma as i32);
        if !out.fresh {
            return;
        }
        self.stamp = s.clone();
        self.last_fresh = mono();
        self.dirty = true;
        self.info(out.logs);
        if let Some(sample) = out.sample {
            match append_sample(&self.dir, &sample) {
                Ok(()) => self.csv_failing = false,
                Err(e) if !self.csv_failing => {
                    self.csv_failing = true;
                    log!(self.node.warn(), "cannot append to the discharge log in {}: {e}", self.dir.display());
                }
                Err(_) => {}
            }
        }
        if out.save || mono() - self.last_save >= self.save_period {
            self.save();
        }
    }

    fn battery_state(&self) -> Option<sensor_msgs::msg::BatteryState> {
        // Like the driver: nothing while the Roomba sends no data (asleep, pack out).
        if mono() - self.last_fresh > self.learner.p.gap_s + 1.0 {
            return None;
        }
        let e = self.learner.estimate()?;
        let capacity_ah = e.capacity_mah / 1000.0;
        Some(sensor_msgs::msg::BatteryState {
            header: std_msgs::msg::Header { stamp: self.stamp.clone(), frame_id: String::new() },
            voltage: e.voltage as f32,
            temperature: f32::NAN,
            current: e.current_a as f32,
            charge: (e.soc * capacity_ah) as f32,
            capacity: capacity_ah as f32,
            design_capacity: (self.learner.p.capacity_mah / 1000.0) as f32,
            percentage: e.soc as f32,
            power_supply_status: if e.current_a < 0.0 { STATUS_DISCHARGING } else { STATUS_UNKNOWN },
            power_supply_health: HEALTH_UNKNOWN,
            power_supply_technology: TECHNOLOGY_LION,
            present: true,
            serial_number: format!("pack {}", e.pack),
            ..Default::default()
        })
    }
}

fn lock(core: &Mutex<Core>) -> MutexGuard<'_, Core> {
    core.lock().unwrap_or_else(|e| e.into_inner())
}

fn unix_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

fn expand_home(path: &str) -> PathBuf {
    let home = || PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    match path.strip_prefix("~/") {
        Some(rest) => home().join(rest),
        None if path == "~" => home(),
        None => PathBuf::from(path),
    }
}

/// The saved state, or None to start over (a corrupt file is kept as .bad for a look).
fn load(node: &Node, path: &Path) -> Option<roomba_rs::battery_learner::State> {
    match load_state(path) {
        Ok(state) => state,
        Err(e) => {
            let bad = path.with_extension(format!("json.bad-{}", unix_now() as u64));
            let _ = std::fs::rename(path, &bad);
            log!(node.warn(), "cannot read {} ({e}); moved it to {} and starting over", path.display(), bad.display());
            None
        }
    }
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let signals = block_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("battery_learner")?;
    let mut params = NodeParams::new(&node);
    let p = Params::load(&mut params)?;
    let dir = expand_home(&params.string("state_dir", "~/.local/state/roomba")?);
    let publish_rate = params.float("publish_rate_hz", 1.0)?.max(0.1);
    let save_period = params.float("save_period_s", 30.0)?.max(1.0);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(STATE_FILE);

    let saved = load(&node, &path);
    match &saved {
        Some(s) => log!(
            node.info(),
            "state {}: pack {} at {:.0} %, capacity {}, {} pack(s) learned from",
            path.display(),
            s.pack.id,
            s.pack.soc * 100.0,
            s.learned_capacity_mah.map_or(format!("{:.0} mAh (prior)", p.capacity_mah), |c| format!("{c:.0} mAh (learned)")),
            s.history.iter().filter(|r| r.capacity_mah.is_some() && !r.outlier).count()
        ),
        None => log!(node.info(), "no state file yet ({}): the first reading starts pack 1 from its voltage", path.display()),
    }
    let core = Arc::new(Mutex::new(Core {
        learner: Learner::new(p, saved, unix_now()),
        node: node.clone(),
        dir,
        path,
        save_period,
        stamp: builtin_interfaces::msg::Time::default(),
        last_fresh: f64::NEG_INFINITY,
        last_save: mono(),
        dirty: false,
        csv_failing: false,
    }));

    let publisher = node.create_publisher::<sensor_msgs::msg::BatteryState>("battery_state".keep_last(10))?;
    let c = Arc::clone(&core);
    let _sensors = node.create_subscription("roomba/sensors".keep_last(10), move |m: roomba_msgs::msg::SensorState| {
        lock(&c).on_sensors(&m);
    })?;
    let c = Arc::clone(&core);
    let _service = node.create_service::<std_srvs::srv::Trigger, _>("~/pack_swapped", move |_: std_srvs::srv::Trigger_Request| {
        let mut core = lock(&c);
        let (message, lines) = core.learner.manual_swap(unix_now());
        core.info(lines);
        core.save();
        std_srvs::srv::Trigger_Response { success: true, message }
    })?;
    // Steady-clock timer (a ROS-time timer can stop for good when the system clock steps).
    let c = Arc::clone(&core);
    let _publish = node.create_timer_repeating(Duration::from_secs_f64(1.0 / publish_rate).steady_time(), move || {
        let msg = lock(&c).battery_state();
        if let Some(msg) = msg {
            publisher.publish(msg).ok();
        }
    })?;
    let c = Arc::clone(&core);
    on_signal(signals, move || {
        let mut core = lock(&c);
        if core.dirty {
            core.save();
        }
    });

    log!(node.info(), "battery_learner running: publishing battery_state, service {}/pack_swapped", node.fully_qualified_name());
    spin(&mut executor)?;
    drop(params);
    Ok(())
}
