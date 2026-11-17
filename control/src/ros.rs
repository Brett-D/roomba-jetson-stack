//! The page's ROS side, through the ros2 CLI (this binary has no ROS libraries): the SOUND panel's
//! tone buttons (roomba/define_song with play), VACUUM ARMED (roomba/cleaning_motors), and the
//! driver's OI mode (roomba/mode) and hazards (roomba/sensors) those need.
//!
//! The driver reports no motor state, so VACUUM ARMED shows the commanded state. It disarms by
//! itself when the robot stack stops, the OI mode leaves Safe / Full (the Roomba stops its motors
//! then) or a cliff or wheel drop is reported (then it also sends all-off, for Full mode).

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
/// each press defines its song before playing it, so sharing the slots is fine.
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
    /// VACUUM ARMED as commanded (vacuum, main brush and side brush on).
    pub armed: bool,
    /// Switching on (true) or off (false).
    pub vac_pending: Option<bool>,
    pub vac_note: Option<(bool, String, Instant)>,
    pub tone_pending: Option<&'static str>,
    pub tone_last: Option<(&'static str, bool, String, Instant)>,
    last_press: Option<Instant>,
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
pub fn motors_cmd(setup: &str, on: bool) -> String {
    format!(
        "{setup} && exec timeout {CLI_TIMEOUT_S} ros2 topic pub -t 3 -r 5 -w 1 --max-wait-time-secs 8 --keep-alive 1 /roomba/cleaning_motors \
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
    let driver = !robot::drivers(app).is_empty();
    let stack = driver || robot::launches(app).any();
    let ctl = robot::lock(app);
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
    if let Some(why) = unavailable(app) {
        return Err(why);
    }
    let mut ctl = robot::lock(app);
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
    let driver = !robot::drivers(app).is_empty();
    if arm {
        if let Some(why) = unavailable(app) {
            return Err(why);
        }
    }
    let mut ctl = robot::lock(app);
    if ctl.ros.vac_pending.is_some() {
        return Err("VACUUM ARMED is still switching.".into());
    }
    if arm && ctl.ros.armed {
        return Ok("Already armed.".into());
    }
    if !arm && !driver {
        ctl.ros.armed = false;
        return Ok("Vacuum off (the robot stack isn't running).".into());
    }
    ctl.ros.vac_pending = Some(arm);
    drop(ctl);
    let app2 = Arc::clone(app);
    let spawned = std::thread::Builder::new().name("vacuum".into()).spawn(move || switch_vacuum(&app2, arm));
    if spawned.is_err() {
        robot::lock(app).ros.vac_pending = None;
        return Err("Cannot start a thread.".into());
    }
    Ok(if arm { "Arming: vacuum and brushes on…".into() } else { "Vacuum and brushes off…".into() })
}

fn switch_vacuum(app: &App, arm: bool) {
    let r = cmd::run(&motors_cmd(&app.cfg.ros_setup, arm), Duration::from_secs(CLI_TIMEOUT_S + 10));
    let result = published(&r);
    eprintln!("vacuum {}: {:?}", if arm { "on" } else { "off" }, result);
    let mut ctl = robot::lock(app);
    ctl.ros.vac_pending = None;
    let now = Instant::now();
    match result {
        Err(e) => ctl.ros.vac_note = Some((false, format!("Couldn't switch the vacuum {}: {e}", if arm { "on" } else { "off" }), now)),
        Ok(()) if !arm => {
            ctl.ros.armed = false;
            ctl.ros.vac_note = Some((true, "Vacuum and brushes off.".into(), now));
        }
        Ok(()) if !matches!(ctl.ros.mode.as_deref(), Some("safe" | "full")) => {
            ctl.ros.vac_note = Some((false, "The OI mode left Safe / Full while arming: not armed.".into(), now));
        }
        Ok(()) => {
            ctl.ros.armed = true;
            ctl.ros.vac_note = Some((true, "Armed: vacuum, main brush and side brush on.".into(), now));
            if ctl.ros.tone_pending.is_none() {
                ctl.ros.tone_pending = Some("armed");
                drop(ctl);
                play(app, &TONES[1]);
            }
        }
    }
}

/// All-off before the robot is stopped (stop and shut down), whether or not the page armed it.
pub fn vacuum_off_before_stop(app: &App) {
    let r = cmd::run(&motors_cmd(&app.cfg.ros_setup, false), Duration::from_secs(CLI_TIMEOUT_S + 10));
    let result = published(&r);
    eprintln!("vacuum off before the stop: {result:?}");
    let mut ctl = robot::lock(app);
    ctl.ros.armed = false;
    let note = match result {
        Ok(()) => (true, "Vacuum and brushes off (robot stopping).".to_string()),
        Err(e) => (false, format!("Vacuum off before the stop failed: {e}")),
    };
    ctl.ros.vac_note = Some((note.0, note.1, Instant::now()));
}

fn auto_disarm(app: &Arc<App>, reason: &str, send_off: bool) {
    let mut ctl = robot::lock(app);
    if !ctl.ros.armed {
        return;
    }
    ctl.ros.armed = false;
    ctl.ros.vac_note = Some((false, format!("Disarmed: {reason}."), Instant::now()));
    drop(ctl);
    eprintln!("vacuum disarmed: {reason}");
    if send_off {
        let app2 = Arc::clone(app);
        let _ = std::thread::Builder::new().name("vacuum-off".into()).spawn(move || {
            let r = cmd::run(&motors_cmd(&app2.cfg.ros_setup, false), Duration::from_secs(CLI_TIMEOUT_S + 10));
            eprintln!("vacuum off after a hazard: {:?}", published(&r));
        });
    }
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
}

fn spawn_monitor(app: &Arc<App>, kind: Kind) -> Option<Child> {
    let echo = match kind {
        // Latched: say so, or an echo started before the driver's publisher exists picks volatile
        // and misses the mode published at the driver's start.
        Kind::Mode => "ros2 topic echo --qos-durability transient_local --qos-reliability reliable --field data /roomba/mode",
        Kind::Sensors => "ros2 topic echo --qos-reliability best_effort /roomba/sensors",
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

/// Runs for the life of the page: the mode monitor while a driver runs, the hazard monitor while
/// armed, and disarming when the stack stops.
pub fn supervise(app: Arc<App>) {
    let (mut mode, mut sensors) = (Monitor::new(), Monitor::new());
    loop {
        let driver = !robot::drivers(&app).is_empty();
        let (power_on, armed) = {
            let ctl = robot::lock(&app);
            (ctl.power == Power::On, ctl.ros.armed || ctl.ros.vac_pending == Some(true))
        };
        mode.keep(driver && power_on, &app, Kind::Mode);
        sensors.keep(driver && power_on && armed, &app, Kind::Sensors);
        if !driver {
            robot::lock(&app).ros.mode = None;
            auto_disarm(&app, "robot stack stopped", false);
        }
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
}
