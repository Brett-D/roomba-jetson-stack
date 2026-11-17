//! The page's ROS side, through the ros2 CLI (this binary has no ROS libraries): the SOUND panel's
//! tone buttons (roomba/define_song with play), VACUUM ARMED (roomba/cleaning_motors), the driver's
//! OI mode (roomba/mode) and hazards (roomba/sensors) those need, and the BATTERY panel
//! (battery_state, and battery_learner's pack_swapped service for NEW PACK).
//!
//! The driver reports no motor state, so VACUUM ARMED shows the commanded state. One thread sends
//! every motor command, one at a time, so the last command sent is always the one wanted now: an
//! OFF asked for while an ON is on its way follows it. It disarms by itself when the robot stack
//! stops, the OI mode leaves Safe / Full (the Roomba stops its motors then) or a cliff or wheel drop
//! is reported (then it also sends all-off, for Full mode). After a failed command the state is
//! unknown: hazards stay watched and OFF is retried until it goes out. At the page's start the
//! state is unknown too (a restart leaves the robot running), so it begins by sending OFF, and a
//! driver that went away (it may have crashed with the motors on) is owed an OFF by the next one.

use std::io::{BufRead, BufReader};
use std::process::Child;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::cmd::{self, CmdResult};
use crate::robot::{self, Op, Power};
use crate::App;

pub struct Tone {
    pub key: &'static str,
    pub song: u8,
    pub notes: &'static [u8],
    /// 1/64 s per note
    pub durations: &'static [u8],
}

/// The sounds as tested on the robot. sound_events uses songs 0 (hello), 1 (uh-oh) and 2 (done);
/// each press defines its song before playing it, so sharing the slots is fine. The 500-series OI
/// takes song numbers 0-4 (Song, opcode 140: "Song Number (0 - 4)"; roomba_driver_rs checks 0-4)
/// and CHATTER in slot 4 was played on this Roomba 550.
pub const TONES: &[Tone] = &[
    Tone { key: "ow", song: 2, notes: &[96, 91, 86, 81, 76], durations: &[3, 3, 3, 4, 12] },
    // 30 is a rest
    Tone { key: "armed", song: 3, notes: &[88, 30, 88, 30, 95], durations: &[6, 3, 6, 3, 18] },
    Tone { key: "chatter", song: 4, notes: &[96, 84, 100, 79, 103, 88, 108, 91, 98], durations: &[4, 3, 5, 3, 4, 3, 6, 3, 10] },
    Tone { key: "hello", song: 0, notes: &[67, 72, 76], durations: &[10, 10, 24] },
    Tone { key: "uhoh", song: 1, notes: &[76, 76, 69], durations: &[12, 6, 28] },
];

const DEBOUNCE: Duration = Duration::from_secs(1);
const CLI_TIMEOUT_S: u64 = 15;

#[derive(Default)]
pub struct RosState {
    /// roomba/mode while the driver runs (the monitor's latest message).
    pub mode: Option<String>,
    /// VACUUM ARMED: whether the page wants the vacuum, main brush and side brush on, the last
    /// state it sent successfully (None: unknown), the command on its way, an OFF to send even if
    /// the state looks off already, and when to retry after a failure.
    pub vac_want: bool,
    pub vac_sent: Option<bool>,
    pub vac_sending: Option<bool>,
    vac_force_off: bool,
    vac_retry_at: Option<Instant>,
    vac_failures: u32,
    pub vac_note: Option<(bool, String, Instant)>,
    pub tone_pending: Option<&'static str>,
    pub tone_last: Option<(&'static str, bool, String, Instant)>,
    last_press: Option<Instant>,
    /// The latest battery_state while the driver runs, and when it came.
    pub battery: Option<(Battery, Instant)>,
    /// Whether /battery_learner/pack_swapped exists (None: not checked, or no battery_learner).
    pub learner_service: Option<bool>,
    learner_checked: Option<Instant>,
    learner_checking: bool,
    pub pack_pending: bool,
    pub pack_last: Option<(bool, String, Instant)>,
}

impl RosState {
    /// Armed: ON went out and is still wanted.
    pub fn armed(&self) -> bool {
        self.vac_want && self.vac_sent == Some(true) && self.vac_sending.is_none()
    }

    /// What VACUUM ARMED is switching to: the command on its way, or the next one (while a driver
    /// runs to take it).
    pub fn vac_pending(&self, driver: bool) -> Option<bool> {
        self.vac_sending.or_else(|| {
            let due = self.vac_retry_at.is_none_or(|t| Instant::now() >= t);
            next_motor_command(self.vac_want, self.vac_sent, self.vac_force_off).filter(|_| driver && due)
        })
    }

    /// The last command failed (or the page just started) and OFF hasn't gone out since.
    pub fn vac_unknown(&self) -> bool {
        self.vac_sent.is_none() && self.vac_sending.is_none()
    }

    /// Hazards are watched while the motors are (or may be) on.
    fn vac_maybe_on(&self) -> bool {
        self.vac_want || self.vac_sending == Some(true) || self.vac_sent != Some(false)
    }
}

/// The motor command that brings the motors to what the page wants (None: nothing to send).
pub fn next_motor_command(want: bool, sent: Option<bool>, force_off: bool) -> Option<bool> {
    if force_off {
        Some(false)
    } else if sent != Some(want) {
        Some(want)
    } else {
        None
    }
}

/// What the BATTERY panel shows from a battery_state message. serial_number and capacity come
/// from battery_learner (the pack, "pack N", and its learned Ah): without it they are empty / NaN.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Battery {
    /// 0-100
    pub percentage: Option<f64>,
    pub voltage: Option<f64>,
    pub pack: Option<String>,
    pub capacity_ah: Option<f64>,
}

/// One `ros2 topic echo` BatteryState message (top-level `key: value` lines).
pub fn parse_battery(lines: &[String]) -> Battery {
    let value = |key: &str| {
        lines.iter().find_map(|l| l.strip_prefix(key)?.strip_prefix(':').map(|v| v.trim().trim_matches(|c| c == '\'' || c == '"').to_string()))
    };
    let number = |key: &str| value(key).and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite());
    // float32 fields: round off the float32 noise (0.5600000023841858)
    let round = |v: f64, places: i32| (v * 10f64.powi(places)).round() / 10f64.powi(places);
    Battery {
        // BatteryState.percentage is 0-1
        percentage: number("percentage").map(|p| round(if p <= 1.0 { p * 100.0 } else { p }.clamp(0.0, 100.0), 1)),
        voltage: number("voltage").filter(|v| *v > 0.0).map(|v| round(v, 2)),
        pack: value("serial_number").map(|s| s.strip_prefix("pack ").unwrap_or(&s).trim().to_string()).filter(|s| !s.is_empty()),
        capacity_ah: number("capacity").filter(|c| *c > 0.0).map(|c| round(c, 2)),
    }
}

pub const PACK_SERVICE: &str = "/battery_learner/pack_swapped";

/// battery_learner processes (not the ros2 CLI calling its service).
pub fn learner_running(app: &App) -> bool {
    cmd::processes().iter().any(|p| p.cmdline.contains(&app.cfg.learner_match) && !p.cmdline.contains("bin/ros2 "))
}

/// Why NEW PACK is unavailable right now (None: available).
pub fn pack_unavailable(app: &App) -> Option<String> {
    let stack = robot::launches(app).any() || !robot::drivers(app).is_empty();
    let learner = learner_running(app);
    let ctl = robot::lock(app);
    if ctl.power != Power::On {
        return Some("The Jetson is shutting down.".into());
    }
    if !stack {
        return Some("Robot stack not running.".into());
    }
    if !learner {
        return Some("battery_learner isn't running (not installed yet?).".into());
    }
    match ctl.ros.learner_service {
        None => Some(format!("Looking for {PACK_SERVICE}…")),
        Some(false) => Some(format!("No {PACK_SERVICE} service.")),
        Some(true) => None,
    }
}

pub fn request_pack(app: &Arc<App>) -> Result<String, String> {
    if let Some(why) = pack_unavailable(app) {
        return Err(why);
    }
    let mut ctl = robot::lock(app);
    if ctl.ros.pack_pending {
        return Err("Still telling battery_learner…".into());
    }
    ctl.ros.pack_pending = true;
    drop(ctl);
    let app2 = Arc::clone(app);
    let spawned = std::thread::Builder::new().name("new-pack".into()).spawn(move || {
        let call = format!("{} && exec timeout {CLI_TIMEOUT_S} ros2 service call {PACK_SERVICE} std_srvs/srv/Trigger", app2.cfg.ros_setup);
        let r = cmd::run(&call, Duration::from_secs(CLI_TIMEOUT_S + 10));
        let (ok, message) = service_response(&r.output).unwrap_or_else(|| (false, format!("No answer from {PACK_SERVICE}.")));
        eprintln!("new pack: {} {message}", if ok { "ok" } else { "FAILED" });
        let mut ctl = robot::lock(&app2);
        ctl.ros.pack_pending = false;
        ctl.ros.pack_last = Some((ok, message, Instant::now()));
    });
    if spawned.is_err() {
        robot::lock(app).ros.pack_pending = false;
        return Err("Cannot start a thread.".into());
    }
    Ok("Telling battery_learner about the new pack…".into())
}

/// Looks for the pack_swapped service (ros2 service list): when battery_learner appears, then every
/// minute (every 10 s while it is missing).
fn check_learner(app: &Arc<App>, learner: bool) {
    let mut ctl = robot::lock(app);
    if !learner {
        ctl.ros.learner_service = None;
        ctl.ros.learner_checked = None;
        return;
    }
    let every = if ctl.ros.learner_service == Some(true) { Duration::from_secs(60) } else { Duration::from_secs(10) };
    if ctl.ros.learner_checking || ctl.ros.learner_checked.is_some_and(|t| t.elapsed() < every) {
        return;
    }
    ctl.ros.learner_checking = true;
    drop(ctl);
    let app2 = Arc::clone(app);
    let spawned = std::thread::Builder::new().name("learner-check".into()).spawn(move || {
        let r = cmd::run(&format!("{} && exec timeout {CLI_TIMEOUT_S} ros2 service list", app2.cfg.ros_setup), Duration::from_secs(CLI_TIMEOUT_S + 10));
        let found = r.ok().then(|| r.output.lines().any(|l| l.trim() == PACK_SERVICE));
        let mut ctl = robot::lock(&app2);
        ctl.ros.learner_checking = false;
        ctl.ros.learner_checked = Some(Instant::now());
        if found.is_some() {
            ctl.ros.learner_service = found;
        }
    });
    if spawned.is_err() {
        robot::lock(app).ros.learner_checking = false;
    }
}

fn join(v: &[u8]) -> String {
    v.iter().map(u8::to_string).collect::<Vec<_>>().join(", ")
}

pub fn song_cmd(setup: &str, t: &Tone) -> String {
    format!(
        "{setup} && exec timeout {CLI_TIMEOUT_S} ros2 service call /roomba/define_song roomba_msgs/srv/DefineSong \
         '{{song_number: {}, notes: [{}], durations: [{}], play: true}}'",
        t.song,
        join(t.notes),
        join(t.durations)
    )
}

/// All three cleaning motors on or off at full speed (opcode 138), normal directions. Sent three
/// times (it is a state, so repeats are harmless) and kept alive a second: a single message from a
/// CLI that exits right away can be lost while the driver is still matching the new publisher.
/// Run with `cmd::run_killing`: nothing of a command that timed out can publish after the next one.
pub fn motors_cmd(setup: &str, on: bool) -> String {
    format!(
        "{setup} && exec timeout -k 3 {CLI_TIMEOUT_S} ros2 topic pub -t 3 -r 5 -w 1 --max-wait-time-secs 8 --keep-alive 1 /roomba/cleaning_motors \
         roomba_msgs/msg/CleaningMotors '{{use_pwm: false, main_brush: {on}, side_brush: {on}, vacuum: {on}}}'"
    )
}

/// `ros2 service call` output: roomba_msgs.srv.DefineSong_Response(success=True, message='...')
pub fn service_response(output: &str) -> Option<(bool, String)> {
    let line = output.lines().find(|l| l.contains("_Response("))?;
    let ok = line.contains("success=True");
    let message = line
        .split_once("message=")
        .map(|(_, m)| m.trim_end().trim_end_matches(')').trim_matches(|c| c == '\'' || c == '"').to_string())
        .unwrap_or_default();
    Some((ok, message))
}

/// Whether `ros2 topic pub -w 1` found the driver and published.
pub fn published(r: &CmdResult) -> Result<(), String> {
    if r.ok() && r.output.contains("publishing #1") {
        Ok(())
    } else if r.output.contains("publishing #1") {
        Err(r.last_line())
    } else {
        Err("roomba_driver isn't listening on roomba/cleaning_motors".into())
    }
}

/// Why the tone buttons and arming are unavailable right now (None: available).
pub fn unavailable(app: &App) -> Option<String> {
    let (stack, driver) = stack_and_driver(app);
    unavailable_locked(&robot::lock(app), stack, driver)
}

/// Whether a robot stack and a driver run (process scans, done before taking the control lock).
fn stack_and_driver(app: &App) -> (bool, bool) {
    let driver = !robot::drivers(app).is_empty();
    (driver || robot::launches(app).any(), driver)
}

/// `unavailable` with the control lock held, so a request can check and commit under one lock.
fn unavailable_locked(ctl: &robot::Ctl, stack: bool, driver: bool) -> Option<String> {
    if ctl.power != Power::On {
        return Some("The Jetson is shutting down.".into());
    }
    match ctl.busy.as_ref().map(|b| b.op) {
        Some(Op::Stop | Op::Shutdown) => return Some("The robot is stopping.".into()),
        Some(Op::Start) => return Some("The robot is starting.".into()),
        None => {}
    }
    if !stack {
        return Some("Robot stack not running.".into());
    }
    if !driver {
        return Some("Waiting for roomba_driver…".into());
    }
    match ctl.ros.mode.as_deref() {
        None => Some("Waiting for the driver's OI mode…".into()),
        Some("safe" | "full") => None,
        Some(m) => Some(format!("OI mode {m}: needs Safe or Full (Enable driving on the robot page).")),
    }
}

pub fn request_tone(app: &Arc<App>, key: &str) -> Result<String, String> {
    let tone = TONES.iter().find(|t| t.key == key).ok_or("No such sound.")?;
    let (stack, driver) = stack_and_driver(app);
    let mut ctl = robot::lock(app);
    if let Some(why) = unavailable_locked(&ctl, stack, driver) {
        return Err(why);
    }
    if ctl.ros.last_press.is_some_and(|t| t.elapsed() < DEBOUNCE) {
        return Err("One press a second.".into());
    }
    if ctl.ros.tone_pending.is_some() {
        return Err("Still sending the last sound…".into());
    }
    ctl.ros.last_press = Some(Instant::now());
    ctl.ros.tone_pending = Some(tone.key);
    drop(ctl);
    let app2 = Arc::clone(app);
    if std::thread::Builder::new().name("tone".into()).spawn(move || play(&app2, tone)).is_err() {
        robot::lock(app).ros.tone_pending = None;
        return Err("Cannot start a thread.".into());
    }
    Ok(format!("Playing {}…", key.to_uppercase()))
}

fn play(app: &App, tone: &'static Tone) {
    let r = cmd::run(&song_cmd(&app.cfg.ros_setup, tone), Duration::from_secs(CLI_TIMEOUT_S + 10));
    let (ok, message) = service_response(&r.output).unwrap_or_else(|| (false, "No answer from roomba_driver (roomba/define_song).".into()));
    eprintln!("sound {} (song {}): {} {message}", tone.key, tone.song, if ok { "ok" } else { "FAILED" });
    let mut ctl = robot::lock(app);
    if ctl.ros.tone_pending == Some(tone.key) {
        ctl.ros.tone_pending = None;
    }
    ctl.ros.tone_last = Some((tone.key, ok, message, Instant::now()));
}

pub fn request_vacuum(app: &Arc<App>, arm: bool) -> Result<String, String> {
    let (stack, driver) = stack_and_driver(app);
    let mut ctl = robot::lock(app);
    // Checked under the lock that commits the arming: a stop or shutdown that has begun (and sends
    // the final OFF) or a mode change can't slip in between.
    if arm {
        if let Some(why) = unavailable_locked(&ctl, stack, driver) {
            return Err(why);
        }
    }
    let r = &mut ctl.ros;
    if arm {
        if r.armed() {
            return Ok("Already armed.".into());
        }
        if r.vac_sending.is_some() || r.vac_force_off || r.vac_sent != Some(false) {
            return Err("VACUUM ARMED is still switching (or switching off after a failure): try again in a moment.".into());
        }
        r.vac_want = true;
        r.vac_note = None;
    } else {
        // An explicit OFF goes out even if the motors look off already.
        r.vac_want = false;
        r.vac_force_off = true;
        r.vac_retry_at = None;
    }
    drop(ctl);
    app.ctl_cv.notify_all();
    Ok(if arm { "Arming: vacuum and brushes on…".into() } else { "Vacuum and brushes off…".into() })
}

/// The process groups of motor commands on their way (see `cmd::run_killing`).
pub fn motor_registry(cfg: &crate::Config) -> std::path::PathBuf {
    cfg.state_dir.join("motor-commands")
}

/// What every motor command's command line contains (`motors_cmd`).
pub const MOTOR_MARKER: &str = "/roomba/cleaning_motors";

/// Sends every motor command, one at a time, until the motors are as the page wants them.
pub fn motor_worker(app: Arc<App>) {
    loop {
        let driver = !robot::drivers(&app).is_empty();
        let target = {
            let mut ctl = robot::lock(&app);
            let r = &mut ctl.ros;
            let due = r.vac_retry_at.is_none_or(|t| Instant::now() >= t);
            match next_motor_command(r.vac_want, r.vac_sent, r.vac_force_off) {
                Some(target) if driver && due => {
                    r.vac_sending = Some(target);
                    r.vac_force_off = false;
                    Some(target)
                }
                _ => {
                    // Requests wake this up; the driver and retries are checked twice a second.
                    let _ = app.ctl_cv.wait_timeout(ctl, Duration::from_millis(500));
                    None
                }
            }
        };
        let Some(target) = target else { continue };
        let r = cmd::run_killing(&motors_cmd(&app.cfg.ros_setup, target), Duration::from_secs(CLI_TIMEOUT_S + 10), &motor_registry(&app.cfg));
        let result = published(&r);
        eprintln!("vacuum {}: {result:?}", if target { "on" } else { "off" });
        let mut play_armed = false;
        {
            let mut ctl = robot::lock(&app);
            let safe = matches!(ctl.ros.mode.as_deref(), Some("safe" | "full"));
            let tone_free = ctl.ros.tone_pending.is_none();
            let r = &mut ctl.ros;
            let now = Instant::now();
            r.vac_sending = None;
            let reason_shown = r.vac_note.as_ref().is_some_and(|(ok, _, at)| !ok && now.duration_since(*at) < Duration::from_secs(30));
            match result {
                Ok(()) => {
                    let was_unknown = r.vac_sent.is_none();
                    r.vac_sent = Some(target);
                    r.vac_retry_at = None;
                    r.vac_failures = 0;
                    if target && r.vac_want && !safe {
                        r.vac_want = false;
                        r.vac_note = Some((false, "The OI mode left Safe / Full while arming: switching off.".into(), now));
                    } else if target && r.vac_want {
                        r.vac_note = Some((true, "Armed: vacuum, main brush and side brush on.".into(), now));
                        if tone_free {
                            r.tone_pending = Some("armed");
                            play_armed = true;
                        }
                    } else if !target && !reason_shown {
                        let text = if was_unknown { "Vacuum and brushes off (their state was unknown)." } else { "Vacuum and brushes off." };
                        r.vac_note = Some((true, text.into(), now));
                    }
                }
                Err(e) => {
                    // Unknown now: give up arming, keep watching hazards, retry OFF (3 s, doubling up to a minute).
                    r.vac_sent = None;
                    r.vac_want = false;
                    r.vac_failures += 1;
                    r.vac_retry_at = Some(now + Duration::from_secs(3 << r.vac_failures.min(5).saturating_sub(1)).min(Duration::from_secs(60)));
                    let what = if target { "on" } else { "off" };
                    r.vac_note = Some((false, format!("Couldn't switch the vacuum {what}: {e}. Its state is unknown: retrying OFF."), now));
                }
            }
        }
        app.ctl_cv.notify_all();
        // On its own thread: a slow sound service must not hold up the next motor command.
        if play_armed {
            let app2 = Arc::clone(&app);
            if std::thread::Builder::new().name("tone".into()).spawn(move || play(&app2, &TONES[1])).is_err() {
                let mut ctl = robot::lock(&app);
                if ctl.ros.tone_pending == Some("armed") {
                    ctl.ros.tone_pending = None;
                }
            }
        }
    }
}

/// All-off before the robot is stopped (stop and shut down), whether or not the page armed it:
/// waits until OFF has gone out after anything already on its way (at most 40 s, or until the
/// driver is gone: then the OFF waits for the next driver).
pub fn vacuum_off_before_stop(app: &App) {
    let mut ctl = robot::lock(app);
    ctl.ros.vac_want = false;
    ctl.ros.vac_force_off = true;
    ctl.ros.vac_retry_at = None;
    app.ctl_cv.notify_all();
    let deadline = Instant::now() + Duration::from_secs(40);
    let result = loop {
        let r = &ctl.ros;
        if r.vac_sending.is_none() && !r.vac_force_off && r.vac_sent == Some(false) {
            break Ok(());
        }
        if r.vac_sending.is_none() && !r.vac_force_off && r.vac_sent.is_none() {
            break Err("OFF failed");
        }
        if Instant::now() > deadline {
            break Err("no answer in 40 s");
        }
        ctl = app.ctl_cv.wait_timeout(ctl, Duration::from_millis(500)).unwrap_or_else(|e| e.into_inner()).0;
        drop(ctl);
        let gone = robot::drivers(app).is_empty();
        ctl = robot::lock(app);
        if gone {
            break Err("the driver has stopped (OFF goes to the next one)");
        }
    };
    eprintln!("vacuum off before the stop: {result:?}");
    let note = match result {
        Ok(()) => (true, "Vacuum and brushes off (robot stopping).".to_string()),
        Err(e) => (false, format!("Vacuum off before the stop: {e}.")),
    };
    ctl.ros.vac_note = Some((note.0, note.1, Instant::now()));
}

/// A hazard, a mode change or the stack stopping: off, with the reason shown. `publish`: send OFF
/// (a hazard in Full mode; the driver gone: to the next driver); else the Roomba has stopped its
/// motors itself (it leaves Safe / Full).
fn auto_disarm(app: &Arc<App>, reason: &str, publish: bool) {
    let mut ctl = robot::lock(app);
    let r = &mut ctl.ros;
    // Already disarmed if nothing wants the motors on and an OFF follows any ON (each hazard
    // message repeats the hazard).
    let on_unanswered = !r.vac_force_off && (r.vac_sending == Some(true) || (r.vac_sending.is_none() && r.vac_sent == Some(true)));
    if !r.vac_want && !on_unanswered {
        return;
    }
    r.vac_want = false;
    if publish {
        r.vac_force_off = true;
        r.vac_retry_at = None;
    } else if r.vac_sending.is_none() {
        r.vac_sent = Some(false);
    }
    r.vac_note = Some((false, format!("Disarmed: {reason}."), Instant::now()));
    drop(ctl);
    app.ctl_cv.notify_all();
    eprintln!("vacuum disarmed: {reason}");
}

const HAZARDS: [&str; 6] = ["wheel_drop_left", "wheel_drop_right", "cliff_left", "cliff_front_left", "cliff_front_right", "cliff_right"];

/// The hazards set in one `ros2 topic echo` SensorState message (top-level `key: value` lines).
pub fn hazards(lines: &[String]) -> Vec<&'static str> {
    HAZARDS
        .into_iter()
        .filter(|h| lines.iter().any(|l| l.strip_prefix(h).and_then(|r| r.strip_prefix(':')).is_some_and(|v| v.trim() == "true")))
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Mode,
    Sensors,
    Battery,
}

fn spawn_monitor(app: &Arc<App>, kind: Kind) -> Option<Child> {
    let echo = match kind {
        // Latched: say so, or an echo started before the driver's publisher exists picks volatile
        // and misses the mode published at the driver's start.
        Kind::Mode => "ros2 topic echo --qos-durability transient_local --qos-reliability reliable --field data /roomba/mode",
        Kind::Sensors => "ros2 topic echo --qos-reliability best_effort /roomba/sensors",
        Kind::Battery => "ros2 topic echo --qos-reliability best_effort --no-arr /battery_state",
    };
    let mut child = cmd::spawn_piped(&format!("{} && exec {echo}", app.cfg.ros_setup)).ok()?;
    let out = child.stdout.take()?;
    let app = Arc::clone(app);
    std::thread::spawn(move || {
        let mut msg: Vec<String> = Vec::new();
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            let line = line.trim_end().to_string();
            match kind {
                Kind::Mode if line.trim() == "---" || line.trim().is_empty() || line.starts_with("WARNING") => {}
                Kind::Mode => {
                    let mode = line.trim().trim_matches('\'').to_lowercase();
                    robot::lock(&app).ros.mode = Some(mode.clone());
                    if mode != "safe" && mode != "full" {
                        auto_disarm(&app, &format!("OI mode {mode} (the Roomba stops its motors)"), false);
                    }
                }
                Kind::Sensors if line == "---" => {
                    let found = hazards(&msg);
                    if !found.is_empty() {
                        auto_disarm(&app, &found.join(", ").replace('_', " "), true);
                    }
                    msg.clear();
                }
                Kind::Sensors => msg.push(line),
                Kind::Battery if line == "---" => {
                    robot::lock(&app).ros.battery = Some((parse_battery(&msg), Instant::now()));
                    msg.clear();
                }
                Kind::Battery => msg.push(line),
            }
        }
    });
    Some(child)
}

struct Monitor {
    child: Option<Child>,
    retry_at: Instant,
}

impl Monitor {
    fn new() -> Self {
        Monitor { child: None, retry_at: Instant::now() }
    }

    fn keep(&mut self, want: bool, app: &Arc<App>, kind: Kind) {
        if let Some(c) = self.child.as_mut() {
            let exited = !matches!(c.try_wait(), Ok(None));
            if exited || !want {
                let _ = c.kill();
                let _ = c.wait();
                self.child = None;
                if exited {
                    self.retry_at = Instant::now() + Duration::from_secs(5);
                }
            }
        }
        if want && self.child.is_none() && Instant::now() >= self.retry_at {
            self.child = spawn_monitor(app, kind);
            self.retry_at = Instant::now() + Duration::from_secs(5);
        }
    }
}

/// Runs for the life of the page: the mode and battery monitors while a driver runs, the hazard
/// monitor while armed, disarming when the stack stops, and the battery_learner service check.
pub fn supervise(app: Arc<App>) {
    let (mut mode, mut sensors, mut battery) = (Monitor::new(), Monitor::new(), Monitor::new());
    loop {
        let driver = !robot::drivers(&app).is_empty();
        let (power_on, watch_hazards) = {
            let ctl = robot::lock(&app);
            (ctl.power == Power::On, ctl.ros.vac_maybe_on())
        };
        mode.keep(driver && power_on, &app, Kind::Mode);
        sensors.keep(driver && power_on && watch_hazards, &app, Kind::Sensors);
        battery.keep(driver && power_on, &app, Kind::Battery);
        if !driver {
            // The driver stops the motors when it ends, but not if it crashed or was killed: if they
            // may have been on, an OFF is owed to the next driver (sent as soon as one runs).
            auto_disarm(&app, "robot stack stopped", true);
            let mut ctl = robot::lock(&app);
            let r = &mut ctl.ros;
            r.mode = None;
            r.battery = None;
            r.vac_retry_at = None;
            r.vac_failures = 0;
        }
        check_learner(&app, power_on && learner_running(&app));
        std::thread::sleep(Duration::from_secs(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tones_are_valid_songs() {
        assert_eq!(TONES.iter().map(|t| t.key).collect::<Vec<_>>(), ["ow", "armed", "chatter", "hello", "uhoh"]);
        for t in TONES {
            assert!(t.song <= 4 && !t.notes.is_empty() && t.notes.len() <= 16 && t.notes.len() == t.durations.len(), "{}", t.key);
        }
        let armed_s: u32 = TONES[1].durations.iter().map(|&d| d as u32).sum();
        assert_eq!(armed_s, 36, "about half a second");
    }

    #[test]
    fn builds_the_cli_commands() {
        assert_eq!(
            song_cmd("S", &TONES[0]),
            "S && exec timeout 15 ros2 service call /roomba/define_song roomba_msgs/srv/DefineSong \
             '{song_number: 2, notes: [96, 91, 86, 81, 76], durations: [3, 3, 3, 4, 12], play: true}'"
        );
        assert!(motors_cmd("S", false).starts_with("S && exec timeout -k 3 15 ros2 topic pub "));
        assert!(motors_cmd("S", false).ends_with("'{use_pwm: false, main_brush: false, side_brush: false, vacuum: false}'"));
    }

    #[test]
    fn reads_cli_output() {
        let out = "requester: making request: ...\n\nresponse:\nroomba_msgs.srv.DefineSong_Response(success=False, message='song 2 defined and played needs Safe or Full mode; robot is in passive mode')\n\n";
        assert_eq!(service_response(out), Some((false, "song 2 defined and played needs Safe or Full mode; robot is in passive mode".into())));
        assert_eq!(service_response("waiting for service to become available...\n"), None);
        let lines: Vec<String> = ["oi_mode: 2", "cliff_front_left: true", "wheel_drop_left: false", "  cliff_left: true"].map(String::from).into();
        assert_eq!(hazards(&lines), ["cliff_front_left"], "only top-level fields");
    }

    #[test]
    fn the_last_motor_command_is_the_one_wanted() {
        assert_eq!(next_motor_command(false, None, false), Some(false), "unknown (start, failure): OFF");
        assert_eq!(next_motor_command(true, Some(false), false), Some(true));
        assert_eq!(next_motor_command(true, Some(true), false), None);
        assert_eq!(next_motor_command(false, Some(true), false), Some(false), "a hazard after ON");
        assert_eq!(next_motor_command(false, Some(false), true), Some(false), "an explicit OFF always goes out");
        let mut r = RosState { vac_want: true, vac_sent: Some(false), vac_sending: Some(true), ..Default::default() };
        assert!(!r.armed() && r.vac_maybe_on(), "arming: hazards watched");
        r.vac_sending = None;
        r.vac_sent = Some(true);
        assert!(r.armed());
        r.vac_sent = None;
        assert!(r.vac_unknown() && r.vac_maybe_on(), "after a failure: unknown, still watched");
    }

    #[test]
    fn reads_battery_state() {
        let msg = |s: &str| s.lines().map(String::from).collect::<Vec<_>>();
        let learned = msg("header:\n  frame_id: ''\nvoltage: 18.43000030517578\ncapacity: 4.119999885559082\npercentage: 0.5600000023841858\nserial_number: 'pack 5'\ncell_voltage: '<array type: float[0]>'");
        assert_eq!(parse_battery(&learned), Battery { percentage: Some(56.0), voltage: Some(18.43), pack: Some("5".into()), capacity_ah: Some(4.12) });
        // Without battery_learner: no pack number, capacity NaN
        let plain = msg("voltage: 18.4\ncapacity: .nan\npercentage: .nan\nserial_number: ''");
        assert_eq!(parse_battery(&plain), Battery { percentage: None, voltage: Some(18.4), pack: None, capacity_ah: None });
    }
}
