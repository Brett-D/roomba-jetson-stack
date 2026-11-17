//! The robot stack and the Jetson: start, stop, shut down, COMMS CHECK. One action at a time; each
//! runs in its own thread and reports its progress through `Ctl`, which /api/status shows.

use std::path::Path;
use std::sync::{Arc, MutexGuard};
use std::time::{Duration, Instant};

use crate::cmd;
use crate::comms::{self, Link};
use crate::App;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Start,
    Stop,
    Shutdown,
}

impl Op {
    pub fn as_str(self) -> &'static str {
        match self {
            Op::Start => "start",
            Op::Stop => "stop",
            Op::Shutdown => "shutdown",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Power {
    On,
    PoweringOff,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckVia {
    Serial,
    Ros,
}

pub struct Busy {
    pub op: Op,
    pub message: String,
    pub since: Instant,
    /// The robot is being stopped (SIGTERM sent, or stop_slam.sh running).
    pub stopping: bool,
    /// Shut down was pressed while a stop was running: power off when it ends.
    pub then_poweroff: bool,
}

pub struct Done {
    pub op: Op,
    pub ok: bool,
    pub message: String,
    pub at: Instant,
}

pub struct LinkResult {
    pub link: Link,
    pub message: String,
    pub via: CheckVia,
    pub at: Instant,
}

pub struct Ctl {
    pub busy: Option<Busy>,
    pub last: Option<Done>,
    pub power: Power,
    /// A browser has been told "powering off" (status served after the power state changed).
    pub power_seen: bool,
    pub checking: Option<(CheckVia, Instant)>,
    pub link: Option<LinkResult>,
    pub ros: crate::ros::RosState,
}

impl Default for Ctl {
    fn default() -> Self {
        Ctl { busy: None, last: None, power: Power::On, power_seen: false, checking: None, link: None, ros: Default::default() }
    }
}

pub fn lock(app: &App) -> MutexGuard<'_, Ctl> {
    app.ctl.lock().unwrap_or_else(|e| e.into_inner())
}

/// Robot launches: the one whose PID run_robot.sh wrote to the PID file, and any other (started
/// some other way). A PID only counts while that process's command line still contains the launch
/// pattern ("ros2 launch roomba_bringup"), so a stale PID file (after a reboot, or a reused PID)
/// never points at an unrelated process.
pub struct Launches {
    pub ours: Option<i32>,
    pub others: Vec<i32>,
}

impl Launches {
    pub fn any(&self) -> bool {
        self.ours.is_some() || !self.others.is_empty()
    }
}

pub fn read_pid(pid_file: &Path) -> Option<i32> {
    std::fs::read_to_string(pid_file).ok()?.trim().parse().ok().filter(|p| *p > 1)
}

fn is_launch(pid: i32, pattern: &str) -> bool {
    cmd::cmdline(pid).is_some_and(|c| c.contains(pattern))
}

pub fn launches(app: &App) -> Launches {
    let cfg = &app.cfg;
    let me = unsafe { libc::getuid() };
    let file_pid = read_pid(&cfg.pid_file);
    let mut out = Launches { ours: None, others: Vec::new() };
    for p in cmd::processes() {
        if p.uid != me || !p.cmdline.contains(&cfg.launch_match) {
            continue;
        }
        if Some(p.pid) == file_pid {
            out.ours = Some(p.pid);
        } else {
            out.others.push(p.pid);
        }
    }
    out
}

/// Roomba driver processes (any user): while one runs it owns the serial port.
pub fn drivers(app: &App) -> Vec<i32> {
    cmd::processes().into_iter().filter(|p| p.cmdline.contains(&app.cfg.driver_match)).map(|p| p.pid).collect()
}

fn set_phase(app: &App, message: &str, stopping: bool) {
    eprintln!("{message}");
    if let Some(b) = lock(app).busy.as_mut() {
        b.message = message.to_string();
        b.stopping = stopping;
    }
}

fn finish(app: &App, ok: bool, message: String) {
    eprintln!("{} {message}", if ok { "done:" } else { "FAILED:" });
    let mut ctl = lock(app);
    let op = ctl.busy.take().map_or(Op::Stop, |b| b.op);
    ctl.last = Some(Done { op, ok, message, at: Instant::now() });
}

fn spawn(name: &str, f: impl FnOnce() + Send + 'static) -> Result<(), String> {
    std::thread::Builder::new().name(name.into()).spawn(f).map(|_| ()).map_err(|e| format!("cannot start a thread: {e}"))
}

fn begin(ctl: &mut Ctl, op: Op, message: &str, stopping: bool) {
    ctl.busy = Some(Busy { op, message: message.into(), since: Instant::now(), stopping, then_poweroff: false });
}

fn refuse_when_busy(ctl: &Ctl) -> Result<(), String> {
    if ctl.power != Power::On {
        return Err("The Jetson is shutting down.".into());
    }
    match &ctl.busy {
        Some(b) => Err(format!("Busy: {}", b.message)),
        None => Ok(()),
    }
}

pub fn request_start(app: &Arc<App>) -> Result<String, String> {
    let l = launches(app);
    let mut ctl = lock(app);
    refuse_when_busy(&ctl)?;
    if matches!(ctl.checking, Some((CheckVia::Serial, _))) {
        return Err("COMMS CHECK has the serial port: try again in a few seconds.".into());
    }
    if l.any() {
        return Err("The robot is already running.".into());
    }
    begin(&mut ctl, Op::Start, "Starting: clearing leftovers, then launching (saved map)…", false);
    drop(ctl);
    let app2 = Arc::clone(app);
    spawn("start", move || start(&app2)).inspect_err(|_| {
        lock(app).busy = None;
    })?;
    Ok("Starting the robot (saved map).".into())
}

fn start(app: &Arc<App>) {
    eprintln!("start: {}", app.cfg.start_cmd);
    let r = cmd::run(&app.cfg.start_cmd, app.cfg.start_timeout);
    for line in r.output.lines() {
        eprintln!("  start | {line}");
    }
    // run_robot.sh writes the PID file just before it exits.
    let deadline = Instant::now() + Duration::from_secs(5);
    while launches(app).ours.is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(250));
    }
    if launches(app).ours.is_some() {
        finish(app, true, "Robot started: wait for \u{201c}Tracking OK\u{201d} in the log.".into());
    } else if r.timed_out {
        finish(app, false, format!("The start command is still running after {} s.", app.cfg.start_timeout.as_secs()));
    } else {
        let code = r.code.map_or("a signal".to_string(), |c| c.to_string());
        finish(app, false, format!("Start failed (exit {code}): {}", r.last_line()));
    }
}

pub fn request_stop(app: &Arc<App>) -> Result<String, String> {
    if !launches(app).any() {
        let ctl = lock(app);
        refuse_when_busy(&ctl)?;
        return Ok("The robot is not running.".into());
    }
    let mut ctl = lock(app);
    refuse_when_busy(&ctl)?;
    begin(&mut ctl, Op::Stop, "Stopping, saving map…", true);
    drop(ctl);
    let app2 = Arc::clone(app);
    spawn("stop", move || {
        let (ok, msg) = stop_robot(&app2);
        let then_poweroff = lock(&app2).busy.as_ref().is_some_and(|b| b.then_poweroff);
        if then_poweroff {
            if let Some(b) = lock(&app2).busy.as_mut() {
                b.op = Op::Shutdown;
            }
            power_off(&app2, &msg);
        } else {
            finish(&app2, ok, msg);
        }
    })
    .inspect_err(|_| {
        lock(app).busy = None;
    })?;
    Ok("Stopping the robot.".into())
}

/// SIGTERM to the launch (graceful.py: every node stops like Ctrl-C and RTAB-Map saves the map),
/// wait for it to end, and only if it's still running the stop timeout after SIGTERM (or a launch
/// without a PID file is running) fall back to stop_slam.sh. `t0` only times the whole stop.
fn stop_robot(app: &Arc<App>) -> (bool, String) {
    let cfg = &app.cfg;
    let t0 = Instant::now();
    if !drivers(app).is_empty() {
        set_phase(app, "Vacuum and brushes off, then stopping…", true);
        crate::ros::vacuum_off_before_stop(app);
    }
    let first = launches(app);
    let mut clean = false;
    if let Some(pid) = first.ours {
        set_phase(app, "Stopping, saving map…", true);
        eprintln!("stop: SIGTERM to the launch, PID {pid}");
        // SAFETY: kill(2) on a PID whose command line we just checked.
        unsafe { libc::kill(pid, libc::SIGTERM) };
        let sigterm = Instant::now();
        while is_launch(pid, &cfg.launch_match) && sigterm.elapsed() < cfg.stop_timeout {
            std::thread::sleep(Duration::from_millis(500));
        }
        if !is_launch(pid, &cfg.launch_match) {
            clean = true;
            // Like stop_slam.sh: the PID file goes when the launch has ended.
            if read_pid(&cfg.pid_file) == Some(pid) {
                let _ = std::fs::remove_file(&cfg.pid_file);
            }
        }
    }
    let rest = launches(app);
    if !rest.any() {
        return match first.ours {
            Some(_) => (true, format!("Stopped after {} s.", t0.elapsed().as_secs())),
            None => (true, "The robot was not running.".into()),
        };
    }
    let why = if first.ours.is_some() && !clean {
        format!("Still running {} s after SIGTERM: forcing stop (stop_slam.sh)…", cfg.stop_timeout.as_secs())
    } else {
        "A robot launch not started from here is running: stopping it (stop_slam.sh)…".to_string()
    };
    set_phase(app, &why, true);
    let r = cmd::run(&cfg.force_stop_cmd, Duration::from_secs(300));
    for line in r.output.lines() {
        eprintln!("  stop_slam | {line}");
    }
    if launches(app).any() {
        (false, format!("Still running after stop_slam.sh: {}", r.last_line()))
    } else {
        (true, format!("Stopped with stop_slam.sh after {} s.", t0.elapsed().as_secs()))
    }
}

pub fn request_shutdown(app: &Arc<App>) -> Result<String, String> {
    let running = launches(app).any();
    let mut ctl = lock(app);
    if ctl.power != Power::On {
        return Err("The Jetson is already shutting down.".into());
    }
    match ctl.busy.as_mut() {
        Some(b) if b.op == Op::Stop => {
            b.then_poweroff = true;
            return Ok("Shutting down when the robot has stopped.".into());
        }
        Some(b) => return Err(format!("Busy: {}", b.message)),
        None => {}
    }
    begin(&mut ctl, Op::Shutdown, "Shutting down: stopping the robot first…", running);
    drop(ctl);
    let app2 = Arc::clone(app);
    spawn("shutdown", move || {
        let (_, msg) = stop_robot(&app2);
        power_off(&app2, &msg);
    })
    .inspect_err(|_| {
        lock(app).busy = None;
    })?;
    Ok("Shutting down: the robot stops first (saving the map), then the Jetson powers off.".into())
}

/// After the robot has stopped (or failed to: power-off goes ahead, systemd stops what's left).
/// The power-off can end this service before its command returns, so the browser is first told
/// "powering off" (up to 5 s for a status request to carry it) and knows the shutdown began.
fn power_off(app: &Arc<App>, stop_message: &str) {
    {
        let mut ctl = lock(app);
        ctl.power = Power::PoweringOff;
        ctl.power_seen = false;
    }
    set_phase(app, "Powering off…", false);
    let mut ctl = lock(app);
    let deadline = Instant::now() + Duration::from_secs(5);
    while !ctl.power_seen && Instant::now() < deadline {
        ctl = app.ctl_cv.wait_timeout(ctl, Duration::from_millis(250)).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(ctl);
    let cfg = &app.cfg;
    if cfg.dry_run_poweroff {
        eprintln!("power-off (dry run): not running `{}`", cfg.poweroff_cmd);
        lock(app).power = Power::Off;
        finish(app, true, format!("{stop_message} Dry run: would now run `{}`; nothing was powered off.", cfg.poweroff_cmd));
        return;
    }
    eprintln!("power-off: {}", cfg.poweroff_cmd);
    let r = cmd::run(&cfg.poweroff_cmd, Duration::from_secs(60));
    if r.ok() {
        lock(app).power = Power::Off;
        finish(app, true, format!("{stop_message} Powering off."));
    } else {
        lock(app).power = Power::On;
        finish(app, false, format!("Power-off failed: {}", r.last_line()));
    }
}

/// SOUND switch: the sound_events node checks for the mute file live, so creating or removing it
/// is all there is to do.
pub fn set_sound(app: &App, muted: bool) -> Result<String, String> {
    let file = &app.cfg.sound_mute_file;
    if muted {
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("Cannot create {}: {e}", dir.display()))?;
        }
        std::fs::OpenOptions::new().create(true).append(true).open(file).map_err(|e| format!("Cannot create {}: {e}", file.display()))?;
        eprintln!("sound muted ({})", file.display());
        Ok("Sound muted.".into())
    } else {
        match std::fs::remove_file(file) {
            Ok(()) => eprintln!("sound on (removed {})", file.display()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("Cannot remove {}: {e}", file.display())),
        }
        Ok("Sound on.".into())
    }
}

pub fn request_check(app: &Arc<App>) -> Result<String, String> {
    let via = if launches(app).any() || !drivers(app).is_empty() { CheckVia::Ros } else { CheckVia::Serial };
    let mut ctl = lock(app);
    refuse_when_busy(&ctl)?;
    if ctl.checking.is_some() {
        return Err("COMMS CHECK is already running.".into());
    }
    ctl.checking = Some((via, Instant::now()));
    drop(ctl);
    let app2 = Arc::clone(app);
    spawn("comms", move || {
        let out = check_comms(&app2, via);
        eprintln!("comms check ({via:?}): {:?}: {}", out.link, out.message);
        let mut ctl = lock(&app2);
        ctl.checking = None;
        ctl.link = Some(LinkResult { link: out.link, message: out.message, via, at: Instant::now() });
    })
    .inspect_err(|_| {
        lock(app).checking = None;
    })?;
    Ok(match via {
        CheckVia::Ros => "Robot stack running: asking the driver over ROS…".into(),
        CheckVia::Serial => "Checking the Roomba on the serial port…".into(),
    })
}

pub fn check_comms(app: &App, via: CheckVia) -> comms::Outcome {
    let cfg = &app.cfg;
    match via {
        CheckVia::Ros => comms::ros_check(&cfg.ros_setup, 8),
        CheckVia::Serial => {
            // Check again right before touching the port.
            if let Some(pid) = drivers(app).first() {
                return comms::Outcome { link: Link::Busy, message: format!("roomba_driver is running (PID {pid}): press COMMS CHECK again to ask it over ROS.") };
            }
            if launches(app).any() {
                return comms::Outcome { link: Link::Busy, message: "The robot stack is starting: press COMMS CHECK again.".into() };
            }
            let port = app.port_config();
            comms::serial_check(&port.port, port.baud, Duration::from_secs(1), 3)
        }
    }
}
