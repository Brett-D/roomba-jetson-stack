//! roomba-control: the always-on control page of the robot's Jetson (port 80, started at boot by
//! systemd; see scripts/install_control_page.sh). Log in with the robot web app's password, then
//! start / stop the robot stack (scripts/run_robot.sh auto, SIGTERM to its launch), check the
//! Roomba's serial link, arm the vacuum, play sounds on the Roomba, see the battery and tell
//! battery_learner about a new pack, watch the Jetson's health, the robot log and the robot page's
//! camera / map / depth views (port 8080), and shut the Jetson down.
//! No ROS libraries: it runs while the robot stack is off and reaches ROS through the ros2 CLI.

mod auth;
mod cmd;
mod comms;
mod robot;
mod ros;
mod status;
mod web;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const MAX_ACTIVE_REQUESTS: usize = 64;

pub struct Config {
    pub listen: String,
    pub start_cmd: String,
    pub force_stop_cmd: String,
    pub poweroff_cmd: String,
    pub dry_run_poweroff: bool,
    pub pid_file: PathBuf,
    pub log_file: PathBuf,
    pub auth_file: PathBuf,
    pub stop_timeout: Duration,
    pub start_timeout: Duration,
    pub session_ttl: Duration,
    pub robot_web_port: u16,
    pub driver_config: PathBuf,
    pub serial_dev: Option<PathBuf>,
    pub camera_usb: (String, String),
    pub launch_match: String,
    pub driver_match: String,
    pub learner_match: String,
    pub ros_setup: String,
    pub disk_path: PathBuf,
    /// The sound_events node stays silent while this file exists.
    pub sound_mute_file: PathBuf,
}

const HELP: &str = "\
roomba-control [options]      the control page
roomba-control --check-comms  run COMMS CHECK once and print the result

  --listen ADDR            address:port to serve on ([::]:80: IPv6 and IPv4; mDNS gives
                           phones the IPv6 address first)
  --robot-ws DIR           robot workspace ($HOME/roomba_ws): its scripts/, install/ and
                           src/roomba_bringup/config/roomba_driver_robot.yaml
  --start-cmd CMD          bash command for Start (<ws>/scripts/run_robot.sh auto)
  --force-stop-cmd CMD     bash command when SIGTERM didn't stop the launch (<ws>/scripts/stop_slam.sh)
  --poweroff-cmd CMD       bash command for the power-off (sudo -n /usr/bin/systemctl poweroff)
  --dry-run-poweroff       shut-down stops the robot but doesn't run the power-off command
  --pid-file FILE          the launch PID run_robot.sh writes ($HOME/robot.pid)
  --log-file FILE          the robot log ($HOME/robot.log)
  --auth-file FILE         the robot web app's password file ($HOME/.config/roomba_web/auth.json)
  --stop-timeout-s N       wait this long after SIGTERM before stop_slam.sh (100: the launch
                           escalates to SIGTERM after 5 s and SIGKILL 85 s later)
  --start-timeout-s N      give up waiting for the start command (300)
  --session-ttl-s N        login lifetime (604800 = 7 days)
  --robot-web-port N       the robot web app's port (8080)
  --driver-config FILE     the driver YAML with the serial port (see --robot-ws)
  --serial-dev DEV         serial port to use instead of the driver YAML's
  --camera-usb VID:PID     the camera's USB id (1e4e:0120, MYNT EYE)
  --launch-match TEXT      command-line text of a robot launch (ros2 launch roomba_bringup)
  --driver-match TEXT      command-line text of a Roomba driver process (lib/roomba_driver:
                           the installed Rust or Python driver, not a build of it)
  --learner-match TEXT     command-line text of the battery_learner process (battery_learner)
  --sound-mute-file FILE   SOUND switch: this file exists while the Roomba is muted
                           ($HOME/.config/roomba/sound_muted, read live by sound_events)
  --ros-setup CMD          bash to set up ROS for the comms check over ROS
                           (source /opt/ros/jazzy/setup.bash && source <ws>/install/setup.bash)
";

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<(Config, bool), String> {
    let home = cmd::home();
    let mut v: std::collections::HashMap<String, String> = Default::default();
    let (mut dry_run, mut check) = (false, false);
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => return Err(String::new()),
            "--dry-run-poweroff" => dry_run = true,
            "--check-comms" => check = true,
            "--listen" | "--robot-ws" | "--start-cmd" | "--force-stop-cmd" | "--poweroff-cmd" | "--pid-file" | "--log-file"
            | "--auth-file" | "--stop-timeout-s" | "--start-timeout-s" | "--session-ttl-s" | "--robot-web-port"
            | "--driver-config" | "--serial-dev" | "--camera-usb" | "--launch-match" | "--driver-match" | "--learner-match" | "--ros-setup" | "--sound-mute-file" => {
                let val = it.next().ok_or_else(|| format!("{a} needs a value"))?;
                v.insert(a[2..].to_string(), val);
            }
            _ => return Err(format!("unknown argument {a}")),
        }
    }
    let ws = v.remove("robot-ws").map(PathBuf::from).unwrap_or_else(|| home.join("roomba_ws"));
    let path = |v: &mut std::collections::HashMap<String, String>, k: &str, default: PathBuf| v.remove(k).map(PathBuf::from).unwrap_or(default);
    let secs = |v: &mut std::collections::HashMap<String, String>, k: &str, default: u64| -> Result<Duration, String> {
        match v.remove(k) {
            Some(s) => s.parse().map(Duration::from_secs).map_err(|_| format!("--{k}: not a number of seconds")),
            None => Ok(Duration::from_secs(default)),
        }
    };
    let q = |p: PathBuf| cmd::shell_quote(&p.display().to_string());
    let camera = v.remove("camera-usb").unwrap_or_else(|| "1e4e:0120".into()).to_ascii_lowercase();
    let camera_usb = camera.split_once(':').map(|(a, b)| (a.to_string(), b.to_string())).ok_or("--camera-usb: VID:PID")?;
    let cfg = Config {
        listen: v.remove("listen").unwrap_or_else(|| "[::]:80".into()),
        start_cmd: v.remove("start-cmd").unwrap_or_else(|| format!("{} auto", q(ws.join("scripts/run_robot.sh")))),
        force_stop_cmd: v.remove("force-stop-cmd").unwrap_or_else(|| q(ws.join("scripts/stop_slam.sh"))),
        poweroff_cmd: v.remove("poweroff-cmd").unwrap_or_else(|| "sudo -n /usr/bin/systemctl poweroff".into()),
        dry_run_poweroff: dry_run,
        pid_file: path(&mut v, "pid-file", home.join("robot.pid")),
        log_file: path(&mut v, "log-file", home.join("robot.log")),
        auth_file: path(&mut v, "auth-file", home.join(".config/roomba_web/auth.json")),
        stop_timeout: secs(&mut v, "stop-timeout-s", 100)?,
        start_timeout: secs(&mut v, "start-timeout-s", 300)?,
        session_ttl: secs(&mut v, "session-ttl-s", 7 * 24 * 3600)?,
        robot_web_port: v.remove("robot-web-port").map_or(Ok(8080), |s| s.parse().map_err(|_| "--robot-web-port: not a port"))?,
        driver_config: path(&mut v, "driver-config", ws.join("src/roomba_bringup/config/roomba_driver_robot.yaml")),
        serial_dev: v.remove("serial-dev").map(PathBuf::from),
        camera_usb,
        launch_match: v.remove("launch-match").unwrap_or_else(|| "ros2 launch roomba_bringup".into()),
        driver_match: v.remove("driver-match").unwrap_or_else(|| "lib/roomba_driver".into()),
        learner_match: v.remove("learner-match").unwrap_or_else(|| "battery_learner".into()),
        ros_setup: v
            .remove("ros-setup")
            .unwrap_or_else(|| format!("source /opt/ros/jazzy/setup.bash && source {}", q(ws.join("install/setup.bash")))),
        sound_mute_file: path(&mut v, "sound-mute-file", home.join(".config/roomba/sound_muted")),
        disk_path: home,
    };
    Ok((cfg, check))
}

pub struct App {
    pub cfg: Config,
    pub ctl: Mutex<robot::Ctl>,
    /// Signalled when `ctl` changes in a way a waiting thread cares about (motor commands, power-off).
    pub ctl_cv: Condvar,
    pub sessions: Mutex<auth::Sessions>,
    pub limiter: Mutex<auth::Limiter>,
    /// One password check at a time (scrypt is deliberately slow, and so is a wrong answer).
    pub login_gate: Mutex<()>,
}

impl App {
    pub fn new(cfg: Config) -> App {
        let ttl = cfg.session_ttl;
        App {
            cfg,
            ctl: Mutex::new(robot::Ctl::default()),
            ctl_cv: Condvar::new(),
            sessions: Mutex::new(auth::Sessions::new(ttl)),
            limiter: Mutex::new(auth::Limiter::default()),
            login_gate: Mutex::new(()),
        }
    }

    pub fn port_config(&self) -> comms::PortConfig {
        comms::port_config(&self.cfg.driver_config, self.cfg.serial_dev.as_deref())
    }
}

struct Active(Arc<AtomicUsize>);

impl Drop for Active {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn main() {
    let (cfg, check) = match parse_args(std::env::args().skip(1)) {
        Ok(c) => c,
        Err(e) if e.is_empty() => {
            print!("{HELP}");
            return;
        }
        Err(e) => {
            eprintln!("{e}\n\n{HELP}");
            std::process::exit(2);
        }
    };
    let app = Arc::new(App::new(cfg));
    if check {
        let via = if robot::launches(&app).any() || !robot::drivers(&app).is_empty() { robot::CheckVia::Ros } else { robot::CheckVia::Serial };
        let out = robot::check_comms(&app, via);
        println!("{} ({:?}, via {:?}): {}", out.link.as_str(), out.link, via, out.message);
        std::process::exit(if out.link == comms::Link::Responding { 0 } else { 1 });
    }
    if !Path::new(&app.cfg.auth_file).exists() {
        eprintln!("warning: {} doesn't exist: nobody can log in (set the password with set_password)", app.cfg.auth_file.display());
    }
    // [::] also takes IPv4 (bindv6only=0); without IPv6 on the machine, fall back to IPv4 only.
    let fallback = app.cfg.listen.strip_prefix("[::]:").map(|port| format!("0.0.0.0:{port}"));
    let server = tiny_http::Server::http(&app.cfg.listen)
        .map(|s| (s, app.cfg.listen.clone()))
        .or_else(|e| match &fallback {
            Some(v4) => tiny_http::Server::http(v4).map(|s| (s, v4.clone())),
            None => Err(e),
        });
    let server = match server {
        Ok((s, addr)) => {
            eprintln!("roomba-control listening on {addr}");
            s
        }
        Err(e) => {
            eprintln!("cannot listen on {}: {e}", app.cfg.listen);
            std::process::exit(1);
        }
    };
    {
        let app = Arc::clone(&app);
        std::thread::Builder::new().name("ros-monitor".into()).spawn(move || ros::supervise(app)).expect("monitor thread");
    }
    {
        let app = Arc::clone(&app);
        std::thread::Builder::new().name("motors".into()).spawn(move || ros::motor_worker(app)).expect("motor thread");
    }
    let active = Arc::new(AtomicUsize::new(0));
    for rq in server.incoming_requests() {
        if active.load(Ordering::SeqCst) >= MAX_ACTIVE_REQUESTS {
            let _ = rq.respond(tiny_http::Response::from_string("busy").with_status_code(503));
            continue;
        }
        active.fetch_add(1, Ordering::SeqCst);
        let guard = Active(Arc::clone(&active));
        let app = Arc::clone(&app);
        // A request that can't get a thread is dropped (the connection closes).
        let _ = std::thread::Builder::new().spawn(move || {
            let _guard = guard;
            web::handle(&app, rq);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_come_from_the_robot_workspace() {
        let (c, check) = parse_args(["--robot-ws", "/w s", "--dry-run-poweroff"].map(String::from)).unwrap();
        assert!(!check && c.dry_run_poweroff);
        assert_eq!(c.start_cmd, "'/w s/scripts/run_robot.sh' auto");
        assert_eq!(c.force_stop_cmd, "'/w s/scripts/stop_slam.sh'");
        assert_eq!(c.driver_config, Path::new("/w s/src/roomba_bringup/config/roomba_driver_robot.yaml"));
        assert_eq!(c.stop_timeout, Duration::from_secs(100));
        assert!(parse_args(["--bogus"].map(String::from)).is_err());
    }
}
