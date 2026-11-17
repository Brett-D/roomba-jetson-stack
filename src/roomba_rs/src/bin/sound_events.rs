//! sound_events: event sounds on the Roomba's beeper (logic in roomba_rs::sound_events).
//!
//! - "hello" (song 0) and HELO on the display for display_s, once, when the driver first reports
//!   Safe or Full mode after the node started (HELO only if the driver's auto_reenter_mode is false:
//!   with true, a display command that crosses a drop to Passive makes it re-enter Safe mode);
//! - "uh-oh" (song 1) on a rising edge of bump_left / bump_right (not the light bumpers), at most one
//!   per bump_interval_s and never while a song plays;
//! - a short "done" jingle (song 2) when an exploration run finishes or gets back to its start point,
//!   once per run.
//!
//! Nothing plays outside Safe/Full mode and the node never changes the mode. The web control page
//! defines and plays songs in the same slots, so every sound is one /roomba/define_song call with the
//! full notes and play = true (no song is defined ahead); songs 3 and 4 are not used here. Muted
//! while mute_file exists (checked at each event: `touch` / `rm` it any time); enabled: false turns
//! the node off. Started by robot.launch.py (sounds:=true, the default) with either driver.
//!
//! Bumps come from the /roomba/sensors snapshots (the driver has no bump event topic; at its
//! sensor_rate_hz, 5 Hz on the robot, a press shorter than ~0.2 s can be missed).
//!
//! Parameters: enabled (true), mute_file (~/.config/roomba/sound_muted), bump_interval_s (4.0),
//!             display_s (3.0)
//! Subscribes: /roomba/sensors, /explorer/status (latched)
//! Publishes:  /roomba/display
//! Uses:       /roomba/define_song (roomba_msgs/srv/DefineSong), /roomba_driver/get_parameters
//!             (auto_reenter_mode, asked whenever the driver appears)

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::Result;
use rclrs::*;
use ros_env::{rcl_interfaces, roomba_msgs, std_msgs};

use rcl_interfaces::srv::{GetParameters, GetParameters_Request, GetParameters_Response};
use roomba_msgs::srv::{DefineSong, DefineSong_Request, DefineSong_Response};
use roomba_rs::node_params::{mono, NodeParams};
use roomba_rs::sound_events::{Action, Event, Params, Sensors, Skip, SoundEvents};
use roomba_rs::watchdog::ParamSource;

const DEFINE_SONG: &str = "/roomba/define_song";
const DRIVER_PARAMS: &str = "/roomba_driver/get_parameters";
const AUTO_REENTER: &str = "auto_reenter_mode";
/// rcl_interfaces/ParameterType PARAMETER_BOOL.
const PARAMETER_BOOL: u8 = 1;

struct State {
    core: SoundEvents,
    define: Client<DefineSong>,
    get_params: Client<GetParameters>,
    display: Publisher<std_msgs::msg::String>,
    mute_file: PathBuf,
    /// The define_song service was available at the last tick.
    driver: bool,
    /// Driver appearances so far (an answer from an earlier one is ignored), and whether this one
    /// still has to be asked for its auto_reenter_mode.
    driver_gen: u64,
    ask_params: bool,
    node: Node,
    worker: Arc<OnceLock<Worker<State>>>,
}

fn expand(path: &str) -> PathBuf {
    match path.strip_prefix("~/") {
        Some(rest) => Path::new(&std::env::var("HOME").unwrap_or_default()).join(rest),
        None => path.into(),
    }
}

fn skip_reason(why: Skip, mute_file: &Path) -> String {
    match why {
        Skip::Mode(Some(m)) => format!("OI mode {m} is not Safe (2) or Full (3)"),
        Skip::Mode(None) => "no sensor data from the driver".into(),
        Skip::NoDriver => format!("{DEFINE_SONG} is not available"),
        Skip::Muted => format!("muted ({} exists)", mute_file.display()),
        Skip::Busy => "a song is playing".into(),
        Skip::RateLimited => "less than bump_interval_s after the last uh-oh".into(),
    }
}

impl State {
    fn run(&mut self, actions: Vec<Action>) {
        for a in actions {
            match a {
                Action::Play(e) => self.play(e),
                Action::Display(text) => {
                    self.display.publish(std_msgs::msg::String { data: text.into() }).ok();
                }
                Action::Skipped(e, why) => {
                    let msg = format!("Not playing \"{}\" ({}): {}", e.song().name, e.what(), skip_reason(why, &self.mute_file));
                    // Bumps can come often; hello and the jingle are once each.
                    if e == Event::Bump {
                        log!(self.node.debug(), "{msg}");
                    } else {
                        log!(self.node.info(), "{msg}");
                    }
                }
            }
        }
    }

    /// One define_song call with the full song and play = true; the answer comes back on the worker.
    fn play(&mut self, e: Event) {
        let song = e.song();
        let play = self.core.plays();
        log!(self.node.info(), "Playing song {} \"{}\" ({})", song.number, song.name, e.what());
        let request = DefineSong_Request {
            song_number: song.number,
            notes: song.notes.to_vec(),
            durations: song.durations.to_vec(),
            play: true,
        };
        let worker = Arc::clone(&self.worker);
        let sent = self.define.call_then(request, move |r: DefineSong_Response| {
            if let Some(w) = worker.get() {
                let _ = w.run(move |s: &mut State| s.answered(e, play, r.success, &r.message));
            }
        });
        if let Err(err) = sent {
            self.answered(e, play, false, &err.to_string());
        }
    }

    fn answered(&mut self, e: Event, play: u64, ok: bool, message: &str) {
        self.core.on_played(play, mono(), ok);
        if !ok {
            log!(self.node.warn(), "Song {} \"{}\" did not play: {message}", e.song().number, e.song().name);
        }
    }

    fn on_sensors(&mut self, m: roomba_msgs::msg::SensorState) {
        let s = Sensors {
            oi_mode: m.oi_mode,
            bump_left: m.bump_left,
            bump_right: m.bump_right,
            light_bump: m.light_bump_left
                || m.light_bump_front_left
                || m.light_bump_center_left
                || m.light_bump_center_right
                || m.light_bump_front_right
                || m.light_bump_right,
            song_playing: m.song_playing,
        };
        let file = &self.mute_file;
        let actions = self.core.on_sensors(s, mono(), &|| file.exists());
        self.run(actions);
    }

    fn on_status(&mut self, json: &str) {
        let file = &self.mute_file;
        let actions = self.core.on_explorer_status(json, mono(), &|| file.exists());
        self.run(actions);
    }

    fn tick(&mut self) {
        let driver = self.define.service_is_ready().unwrap_or(false);
        if driver != self.driver && self.core.enabled() {
            if driver {
                log!(self.node.info(), "Driver found ({DEFINE_SONG})");
                self.driver_gen += 1;
                self.ask_params = true;
            } else {
                log!(self.node.warn(), "{DEFINE_SONG} went away; waiting for the driver");
            }
            self.driver = driver;
        }
        if driver && self.ask_params && self.get_params.service_is_ready().unwrap_or(false) {
            self.ask_params = false;
            self.ask_auto_reenter();
        }
        let file = &self.mute_file;
        let actions = self.core.tick(mono(), driver, &|| file.exists());
        self.run(actions);
    }

    fn ask_auto_reenter(&mut self) {
        let gen = self.driver_gen;
        let worker = Arc::clone(&self.worker);
        let request = GetParameters_Request { names: vec![AUTO_REENTER.to_string()] };
        let sent = self.get_params.call_then(request, move |r: GetParameters_Response| {
            let value = r.values.first().filter(|v| v.type_ == PARAMETER_BOOL).map(|v| v.bool_value);
            if let Some(w) = worker.get() {
                let _ = w.run(move |s: &mut State| s.on_auto_reenter(gen, value));
            }
        });
        if sent.is_err() {
            self.on_auto_reenter(gen, None);
        }
    }

    fn on_auto_reenter(&mut self, gen: u64, value: Option<bool>) {
        if gen != self.driver_gen {
            return;
        }
        self.core.on_driver_params(value);
        match value {
            Some(false) => log!(self.node.info(), "Driver's {AUTO_REENTER}: false: HELO goes on the display"),
            Some(true) => log!(
                self.node.info(),
                "Driver's {AUTO_REENTER}: true: no HELO on the display (a display command in Passive mode would make it re-enter Safe mode)"
            ),
            None => log!(self.node.warn(), "Could not read the driver's {AUTO_REENTER}: no HELO on the display"),
        }
    }
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    // Steady clock throughout: ROS-time timers stop an rclrs 0.8 worker for good on a clock step.
    let node = executor.create_node("sound_events".clock_type(ClockType::SteadyTime))?;
    let mut declared = NodeParams::new(&node);
    let p = Params {
        enabled: declared.boolean("enabled", true)?,
        bump_interval: declared.float("bump_interval_s", 4.0)?,
        display_time: declared.float("display_s", 3.0)?,
        ..Params::default()
    };
    let mute_file = expand(&declared.string("mute_file", "~/.config/roomba/sound_muted")?);
    if p.enabled {
        log!(
            node.info(),
            "Event sounds on: hello, uh-oh on bumps, a jingle when exploring finishes (mute: touch {}); waiting for {DEFINE_SONG}",
            mute_file.display()
        );
    } else {
        log!(node.info(), "Event sounds off (enabled: false)");
    }
    let latched = || QoSProfile::topics_default().keep_last(1).reliable().transient_local();
    let worker_cell = Arc::new(OnceLock::new());
    let state = State {
        core: SoundEvents::new(p),
        define: node.create_client::<DefineSong>(DEFINE_SONG)?,
        get_params: node.create_client::<GetParameters>(DRIVER_PARAMS)?,
        display: node.create_publisher("/roomba/display".keep_last(10))?,
        mute_file,
        driver: false,
        driver_gen: 0,
        ask_params: false,
        node: node.clone(),
        worker: Arc::clone(&worker_cell),
    };
    let worker = node.create_worker::<State>(state);
    let _ = worker_cell.set(Arc::clone(&worker));
    let _sensors = worker.create_subscription::<roomba_msgs::msg::SensorState, _>(
        "/roomba/sensors".keep_last(10),
        |s: &mut State, m: roomba_msgs::msg::SensorState| s.on_sensors(m),
    )?;
    let _status = worker.create_subscription::<std_msgs::msg::String, _>(
        "/explorer/status".qos(latched()),
        |s: &mut State, m: std_msgs::msg::String| s.on_status(&m.data),
    )?;
    let _tick = worker.create_timer_repeating(Duration::from_millis(100).steady_time(), |s: &mut State| s.tick())?;
    roomba_rs::node_params::spin(&mut executor)?;
    drop(declared);
    Ok(())
}
