//! COMMS CHECK: is the Roomba answering on its serial link?
//!
//! - Robot stack running (roomba_driver owns the port): never touch the port; ask the driver over
//!   ROS instead (`ros2 topic echo --once` of roomba/mode and battery_state, with a timeout).
//! - Stack stopped: open the port from roomba_driver_robot.yaml at its baud rate, 8N1, and send
//!   only Start (128, which leaves the Roomba in Passive mode) and one sensor query (142, packet 3:
//!   the 10-byte power group). Never Safe / Full (131 / 132) or any drive or motor command.

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::cmd;

pub const OI_START: u8 = 128;
pub const OI_SENSORS: u8 = 142;
pub const PACKET_POWER: u8 = 3;
const POWER_LEN: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Link {
    Responding,
    NotResponding,
    PortMissing,
    /// The port is in use (or the check could not run): nothing learned about the Roomba.
    Busy,
}

impl Link {
    pub fn as_str(self) -> &'static str {
        match self {
            Link::Responding => "responding",
            Link::NotResponding => "not_responding",
            Link::PortMissing => "port_missing",
            Link::Busy => "busy",
        }
    }
}

#[derive(Debug)]
pub struct Outcome {
    pub link: Link,
    pub message: String,
}

fn outcome(link: Link, message: impl Into<String>) -> Outcome {
    Outcome { link, message: message.into() }
}

pub const ASLEEP: &str = "No reply: Roomba asleep, press Clean until its lights come on.";

/// The driver's port and baud rate from roomba_driver_robot.yaml (`port:` and `baud:` under
/// roomba_driver.ros__parameters; the file has no other keys of those names).
pub fn parse_driver_yaml(text: &str) -> Option<(String, Option<u32>)> {
    let value = |key: &str| {
        text.lines().find_map(|line| {
            let line = match line.find(" #") {
                Some(i) => &line[..i],
                None => line,
            };
            let v = line.trim().strip_prefix(key)?.strip_prefix(':')?.trim();
            let v = v.trim_matches(|c| c == '"' || c == '\'');
            (!v.is_empty()).then(|| v.to_string())
        })
    };
    let port = value("port")?;
    Some((port, value("baud").and_then(|b| b.parse().ok())))
}

#[derive(Clone, Debug)]
pub struct PortConfig {
    pub port: PathBuf,
    pub baud: u32,
    /// Where it came from, for messages.
    pub source: String,
}

pub fn port_config(driver_yaml: &Path, override_port: Option<&Path>) -> PortConfig {
    if let Some(p) = override_port {
        return PortConfig { port: p.to_path_buf(), baud: 115200, source: "--serial-dev".into() };
    }
    match std::fs::read_to_string(driver_yaml).ok().as_deref().and_then(parse_driver_yaml) {
        Some((port, baud)) => PortConfig { port: port.into(), baud: baud.unwrap_or(115200), source: driver_yaml.display().to_string() },
        None => PortConfig { port: "/dev/ttyUSB0".into(), baud: 115200, source: format!("default ({} unreadable)", driver_yaml.display()) },
    }
}

#[derive(Debug, PartialEq)]
pub struct PowerGroup {
    pub charging_state: u8,
    pub voltage_mv: u16,
    pub current_ma: i16,
    pub temperature_c: i8,
    pub charge_mah: u16,
    pub capacity_mah: u16,
}

/// Sensor packet 3 (packets 21-26): charging state, voltage, current, temperature, charge, capacity.
pub fn parse_power_group(b: &[u8; POWER_LEN]) -> PowerGroup {
    PowerGroup {
        charging_state: b[0],
        voltage_mv: u16::from_be_bytes([b[1], b[2]]),
        current_ma: i16::from_be_bytes([b[3], b[4]]),
        temperature_c: b[5] as i8,
        charge_mah: u16::from_be_bytes([b[6], b[7]]),
        capacity_mah: u16::from_be_bytes([b[8], b[9]]),
    }
}

fn charging_name(state: u8) -> &'static str {
    match state {
        0 => "not charging",
        1 => "reconditioning",
        2 => "full charging",
        3 => "trickle charging",
        4 => "waiting",
        5 => "charging fault",
        _ => "unknown charging state",
    }
}

pub fn describe(p: &PowerGroup) -> Option<String> {
    // A reply at the wrong baud rate or from something else is noise: a Roomba reads 6-25 V.
    if !(6000..=25000).contains(&p.voltage_mv) || p.charging_state > 5 {
        return None;
    }
    Some(format!(
        "OK: voltage {:.2} V, OI mode passive (after Start); current {} mA, battery {} \u{b0}C, {}.",
        p.voltage_mv as f64 / 1000.0,
        p.current_ma,
        p.temperature_c,
        charging_name(p.charging_state)
    ))
}

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// The serial port, raw 8N1 like the driver opens it (DTR / RTS low), held exclusively (flock and
/// TIOCEXCL) while the check runs.
struct Tty {
    fd: libc::c_int,
}

impl Drop for Tty {
    fn drop(&mut self) {
        // SAFETY: the fd belongs to this struct and is closed once.
        unsafe {
            libc::ioctl(self.fd, libc::TIOCNXCL);
            libc::close(self.fd);
        }
    }
}

fn speed(baud: u32) -> Option<libc::speed_t> {
    Some(match baud {
        19200 => libc::B19200,
        38400 => libc::B38400,
        57600 => libc::B57600,
        115200 => libc::B115200,
        _ => return None,
    })
}

impl Tty {
    fn open(path: &Path, baud: u32) -> Result<Tty, Outcome> {
        let speed = speed(baud).ok_or_else(|| outcome(Link::Busy, format!("Unsupported baud rate {baud}.")))?;
        let c = CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| outcome(Link::Busy, "Bad port path."))?;
        // SAFETY: open(2) of a NUL-terminated path.
        let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDWR | libc::O_NOCTTY | libc::O_NONBLOCK | libc::O_CLOEXEC) };
        if fd < 0 {
            let e = errno();
            let err = std::io::Error::from_raw_os_error(e);
            return Err(match e {
                libc::ENOENT | libc::ENODEV | libc::ENXIO => {
                    outcome(Link::PortMissing, format!("Port missing: check the USB-serial cable ({}).", path.display()))
                }
                libc::EBUSY => outcome(Link::Busy, format!("Port busy: another program has it open exclusively ({err}).")),
                _ => outcome(Link::Busy, format!("Cannot open {}: {err}.", path.display())),
            });
        }
        let tty = Tty { fd };
        // SAFETY: plain ioctl / flock / termios calls on the fd we own; termios is plain data.
        unsafe {
            if libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) != 0 {
                return Err(outcome(Link::Busy, "Port busy: another program holds a lock on it."));
            }
            libc::ioctl(fd, libc::TIOCEXCL);
            let mut t: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(fd, &mut t) != 0 {
                return Err(outcome(Link::Busy, format!("Cannot configure the port: {}.", std::io::Error::last_os_error())));
            }
            libc::cfmakeraw(&mut t);
            t.c_cflag |= libc::CLOCAL | libc::CREAD | libc::CS8;
            t.c_cflag &= !(libc::PARENB | libc::CSTOPB | libc::CRTSCTS);
            t.c_iflag &= !(libc::IXON | libc::IXOFF | libc::IXANY);
            t.c_cc[libc::VMIN] = 0;
            t.c_cc[libc::VTIME] = 0;
            libc::cfsetispeed(&mut t, speed);
            libc::cfsetospeed(&mut t, speed);
            if libc::tcsetattr(fd, libc::TCSANOW, &t) != 0 {
                return Err(outcome(Link::Busy, format!("Cannot configure the port: {}.", std::io::Error::last_os_error())));
            }
            // DTR / RTS low like the driver (a pty has no modem lines: ignore the error).
            let lines: libc::c_int = libc::TIOCM_DTR | libc::TIOCM_RTS;
            libc::ioctl(fd, libc::TIOCMBIC, &lines);
            libc::tcflush(fd, libc::TCIOFLUSH);
        }
        Ok(tty)
    }

    fn write_all(&self, mut data: &[u8]) -> Result<(), String> {
        let deadline = Instant::now() + Duration::from_millis(500);
        while !data.is_empty() {
            // SAFETY: writes from a valid slice.
            let n = unsafe { libc::write(self.fd, data.as_ptr().cast(), data.len()) };
            if n > 0 {
                data = &data[n as usize..];
            } else if n < 0 && errno() != libc::EAGAIN && errno() != libc::EINTR {
                return Err(std::io::Error::last_os_error().to_string());
            } else if Instant::now() > deadline {
                return Err("write timed out".into());
            } else {
                self.poll(libc::POLLOUT, Duration::from_millis(50));
            }
        }
        Ok(())
    }

    fn poll(&self, events: libc::c_short, timeout: Duration) -> bool {
        let mut p = libc::pollfd { fd: self.fd, events, revents: 0 };
        // SAFETY: one valid pollfd.
        unsafe { libc::poll(&mut p, 1, timeout.as_millis().min(i32::MAX as u128) as i32) > 0 }
    }

    /// Up to `n` bytes, waiting at most `timeout` in total.
    fn read_upto(&self, n: usize, timeout: Duration) -> Vec<u8> {
        let deadline = Instant::now() + timeout;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || !self.poll(libc::POLLIN, left) {
                break;
            }
            let mut buf = [0u8; 64];
            let want = (n - out.len()).min(buf.len());
            // SAFETY: reads into a valid buffer of at least `want` bytes.
            let r = unsafe { libc::read(self.fd, buf.as_mut_ptr().cast(), want) };
            if r > 0 {
                out.extend_from_slice(&buf[..r as usize]);
            } else if r == 0 || (errno() != libc::EAGAIN && errno() != libc::EINTR) {
                break;
            }
        }
        out
    }

    fn flush_input(&self) {
        // SAFETY: tcflush on the fd we own.
        unsafe { libc::tcflush(self.fd, libc::TCIFLUSH) };
    }
}

/// Start + power-group query, `attempts` times with `timeout` for each reply. The caller has made
/// sure the robot stack (roomba_driver) isn't running.
pub fn serial_check(port: &Path, baud: u32, timeout: Duration, attempts: u32) -> Outcome {
    if !port.exists() {
        return outcome(Link::PortMissing, format!("Port missing: check the USB-serial cable ({}).", port.display()));
    }
    if let Some((pid, name)) = cmd::fd_holders(port).into_iter().next() {
        return outcome(Link::Busy, format!("Port busy: open in PID {pid} ({name}); not touching it."));
    }
    let tty = match Tty::open(port, baud) {
        Ok(t) => t,
        Err(o) => return o,
    };
    let mut garbled = None;
    for attempt in 0..attempts {
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(200));
        }
        tty.flush_input();
        if let Err(e) = tty.write_all(&[OI_START]) {
            return outcome(Link::Busy, format!("Cannot write to the port: {e}."));
        }
        std::thread::sleep(Duration::from_millis(30)); // the OI wants 20 ms after a mode change
        if let Err(e) = tty.write_all(&[OI_SENSORS, PACKET_POWER]) {
            return outcome(Link::Busy, format!("Cannot write to the port: {e}."));
        }
        let reply = tty.read_upto(POWER_LEN, timeout);
        if let Ok(b) = <[u8; POWER_LEN]>::try_from(reply.as_slice()) {
            match describe(&parse_power_group(&b)) {
                Some(msg) => return outcome(Link::Responding, msg),
                None => garbled = Some(b.len()),
            }
        } else if !reply.is_empty() {
            garbled = Some(reply.len());
        }
    }
    match garbled {
        Some(n) => outcome(
            Link::NotResponding,
            format!("Garbled reply ({n} bytes): check the baud rate ({baud}) and the cable, or wake the Roomba (press Clean)."),
        ),
        None => outcome(Link::NotResponding, ASLEEP),
    }
}

/// The value `ros2 topic echo --once --field <f>` printed: the line before its first "---".
pub fn echo_value(output: &str) -> Option<String> {
    let lines: Vec<&str> = output.lines().map(str::trim).collect();
    let end = lines.iter().position(|l| *l == "---")?;
    lines[..end].iter().rev().find(|l| !l.is_empty()).map(|l| l.trim_matches('\'').to_string())
}

/// The robot stack is running: ask the driver over ROS (both topics at once, `timeout_s` each).
pub fn ros_check(ros_setup: &str, timeout_s: u64) -> Outcome {
    let echo = |field: &str, topic: &str, qos: &str| {
        format!("{ros_setup} && exec timeout {timeout_s} ros2 topic echo --once {qos}--field {field} {topic}")
    };
    let wait = Duration::from_secs(timeout_s + 20);
    let mode_cmd = echo("data", "/roomba/mode", "--qos-durability transient_local --qos-reliability reliable ");
    let mode = std::thread::spawn(move || cmd::run(&mode_cmd, wait));
    let battery = cmd::run(&echo("voltage", "/battery_state", ""), wait);
    let mode = mode.join().ok().and_then(|r| echo_value(&r.output));
    ros_outcome(mode.as_deref(), echo_value(&battery.output).and_then(|v| v.parse::<f64>().ok()), timeout_s)
}

pub fn ros_outcome(mode: Option<&str>, voltage: Option<f64>, timeout_s: u64) -> Outcome {
    let mode_text = mode.unwrap_or("unknown");
    match voltage {
        Some(v) => outcome(Link::Responding, format!("OK (via the running driver): voltage {v:.2} V, OI mode {mode_text}.")),
        None if mode.is_some() => outcome(
            Link::NotResponding,
            format!("The driver is running but sent no battery data for {timeout_s} s (OI mode {mode_text}). {ASLEEP}"),
        ),
        None => outcome(
            Link::NotResponding,
            format!("No data from the driver in {timeout_s} s: is roomba_driver running? See the robot log."),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::FromRawFd;

    #[test]
    fn reads_the_port_from_the_driver_yaml() {
        let yaml = "roomba_driver:\n  ros__parameters:\n    port: /dev/serial/by-id/usb-FTDI-port0   # the cable\n    baud: 115200\n    initial_mode: safe\n";
        assert_eq!(parse_driver_yaml(yaml), Some(("/dev/serial/by-id/usb-FTDI-port0".into(), Some(115200))));
        assert_eq!(parse_driver_yaml("    port: '/dev/ttyUSB1'\n"), Some(("/dev/ttyUSB1".into(), None)));
        assert_eq!(parse_driver_yaml("    baud: 57600\n"), None);
    }

    #[test]
    fn decodes_the_power_group() {
        // not charging, 18.432 V, -350 mA, 24 C, 1200 / 2600 mAh
        let b = [0, 0x48, 0x00, 0xfe, 0xa2, 24, 0x04, 0xb0, 0x0a, 0x28];
        let p = parse_power_group(&b);
        assert_eq!(p, PowerGroup { charging_state: 0, voltage_mv: 18432, current_ma: -350, temperature_c: 24, charge_mah: 1200, capacity_mah: 2600 });
        assert!(describe(&p).unwrap().starts_with("OK: voltage 18.43 V, OI mode passive"));
        assert_eq!(describe(&parse_power_group(&[0xff; 10])), None, "noise is not a reply");
    }

    #[test]
    fn reads_ros2_topic_echo_output() {
        assert_eq!(echo_value("safe\n---\n").as_deref(), Some("safe"));
        assert_eq!(echo_value("WARNING: topic [/battery_state] does not appear to be published yet\n18.43\n---\n").as_deref(), Some("18.43"));
        assert_eq!(echo_value("WARNING: topic [/x] does not appear to be published yet\n"), None);
        assert_eq!(ros_outcome(Some("safe"), Some(18.4), 8).link, Link::Responding);
        assert_eq!(ros_outcome(Some("passive"), None, 8).link, Link::NotResponding);
        assert!(ros_outcome(None, None, 8).message.starts_with("No data from the driver"));
    }

    /// A pty stands in for the Roomba: the check must send exactly Start + one power-group query
    /// per attempt, and decode the reply.
    fn fake_roomba(reply: Option<[u8; 10]>, attempts: u32) -> (Outcome, Vec<u8>) {
        let (mut master, mut slave) = (0, 0);
        let mut name = [0 as libc::c_char; 128];
        // SAFETY: openpty fills the two fds and the name buffer.
        let r = unsafe { libc::openpty(&mut master, &mut slave, name.as_mut_ptr(), std::ptr::null(), std::ptr::null()) };
        assert_eq!(r, 0);
        // Not inherited by processes other tests start meanwhile: the check would see them holding
        // the port.
        for fd in [master, slave] {
            // SAFETY: fcntl on the fds openpty just returned.
            unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
        }
        // The slave fd stays open (the master side would see a hangup without it); the check
        // opens the device again by name and ignores this process's own fds.
        // SAFETY: openpty wrote a NUL-terminated name.
        let path = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
        // SAFETY: master is a fresh fd owned by this File.
        let mut m = unsafe { std::fs::File::from_raw_fd(master) };
        let roomba = std::thread::spawn(move || {
            use std::io::{Read, Write};
            let mut seen = Vec::new();
            let mut buf = [0u8; 16];
            let deadline = Instant::now() + Duration::from_secs(3);
            while Instant::now() < deadline {
                let mut p = libc::pollfd { fd: master, events: libc::POLLIN, revents: 0 };
                // SAFETY: one valid pollfd.
                if unsafe { libc::poll(&mut p, 1, 50) } <= 0 {
                    continue;
                }
                let Ok(n) = m.read(&mut buf) else { break };
                seen.extend_from_slice(&buf[..n]);
                if seen.ends_with(&[OI_SENSORS, PACKET_POWER]) {
                    if let Some(r) = reply {
                        m.write_all(&r).unwrap();
                        std::thread::sleep(Duration::from_millis(200));
                        break;
                    }
                }
            }
            seen
        });
        let out = serial_check(Path::new(&path), 115200, Duration::from_millis(300), attempts);
        let seen = roomba.join().unwrap();
        // SAFETY: closes the slave fd openpty gave us.
        unsafe { libc::close(slave) };
        (out, seen)
    }

    #[test]
    fn serial_check_sends_only_start_and_the_sensor_query() {
        let (out, seen) = fake_roomba(Some([2, 0x3f, 0x5c, 0x03, 0xe8, 30, 0, 0, 0, 0]), 3);
        assert_eq!(out.link, Link::Responding, "{}", out.message);
        assert!(out.message.starts_with("OK: voltage 16.22 V, OI mode passive"), "{}", out.message);
        assert_eq!(seen, [OI_START, OI_SENSORS, PACKET_POWER]);
    }

    #[test]
    fn serial_check_reports_a_sleeping_roomba() {
        let (out, seen) = fake_roomba(None, 3);
        assert_eq!((out.link, out.message.as_str()), (Link::NotResponding, ASLEEP));
        assert_eq!(seen, [128, 142, 3, 128, 142, 3, 128, 142, 3], "3 attempts, nothing else sent");
    }

    #[test]
    fn serial_check_reports_a_missing_port() {
        let out = serial_check(Path::new("/dev/serial/by-id/no-such-adapter"), 115200, Duration::from_millis(100), 1);
        assert_eq!(out.link, Link::PortMissing);
        assert!(out.message.starts_with("Port missing: check the USB-serial cable"));
    }
}
