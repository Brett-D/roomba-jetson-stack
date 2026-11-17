//! Running the configured shell commands (start, forced stop, power-off, ROS queries) and looking
//! at other processes through /proc.

use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

const KEEP_OUTPUT: usize = 8192;

pub struct CmdResult {
    /// Exit code; None when it timed out (still running, left alone) or died from a signal.
    pub code: Option<i32>,
    pub timed_out: bool,
    /// The end of stdout + stderr.
    pub output: String,
}

impl CmdResult {
    pub fn ok(&self) -> bool {
        self.code == Some(0)
    }

    pub fn last_line(&self) -> String {
        self.output.lines().rev().map(str::trim).find(|l| !l.is_empty()).unwrap_or("").to_string()
    }
}

/// `bash -lc <cmd>` in $HOME with stdout and stderr captured: a login shell, so commands see the
/// same environment as in an SSH login (~/.profile). The ambient capability the service gets for
/// port 80 is dropped first, so the robot runs exactly as when started from a shell.
pub fn run(cmd: &str, timeout: Duration) -> CmdResult {
    run_inner(cmd, timeout, None)
}

/// `run`, but in its own process group, all of which is killed (SIGKILL) and the shell reaped
/// before this returns when the timeout expires: nothing it started can still act afterwards (the
/// motor commands, which must arrive in order). While it runs, its group ID is listed in
/// `registry`, so that if this service ends meanwhile (a restart, a crash; systemd's
/// KillMode=process leaves children running) the next start can kill it: `kill_leftover_groups`.
pub fn run_killing(cmd: &str, timeout: Duration, registry: &Path) -> CmdResult {
    run_inner(cmd, timeout, Some(registry))
}

fn run_inner(cmd: &str, timeout: Duration, registry: Option<&Path>) -> CmdResult {
    let kill_on_timeout = registry.is_some();
    let mut command = Command::new("bash");
    if kill_on_timeout {
        command.process_group(0);
    }
    command.arg("-lc").arg(format!("exec 2>&1\n{cmd}")).stdin(Stdio::null()).stdout(Stdio::piped());
    if let Some(home) = std::env::var_os("HOME") {
        command.current_dir(home);
    }
    // SAFETY: prctl is async-signal-safe; it only changes this (child) process's capabilities.
    unsafe {
        command.pre_exec(|| {
            libc::prctl(libc::PR_CAP_AMBIENT, libc::PR_CAP_AMBIENT_CLEAR_ALL as libc::c_ulong, 0, 0, 0);
            Ok(())
        });
    }
    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => return CmdResult { code: None, timed_out: false, output: format!("cannot run bash: {e}") },
    };
    let pgid = child.id() as i32;
    if let Some(r) = registry {
        update_registry(r, pgid, true);
    }
    let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let (done_tx, done_rx) = mpsc::channel();
    if let Some(mut out) = child.stdout.take() {
        let buf = Arc::clone(&buf);
        std::thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            while let Ok(n) = out.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                let mut b = buf.lock().unwrap_or_else(|e| e.into_inner());
                b.extend_from_slice(&chunk[..n]);
                if b.len() > KEEP_OUTPUT {
                    let cut = b.len() - KEEP_OUTPUT;
                    b.drain(..cut);
                }
            }
            let _ = done_tx.send(());
        });
    }
    let start = Instant::now();
    let (code, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break (status.code(), false),
            Ok(None) if start.elapsed() >= timeout && kill_on_timeout => {
                // SAFETY: kill(2) on the process group this child leads (process_group(0)).
                unsafe { libc::kill(-pgid, libc::SIGKILL) };
                let _ = child.wait();
                break (None, true);
            }
            Ok(None) if start.elapsed() >= timeout => {
                // Leave it running, but reap it when it ends.
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                break (None, true);
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(e) => {
                return CmdResult { code: None, timed_out: false, output: format!("wait failed: {e}") };
            }
        }
    };
    if let Some(r) = registry {
        update_registry(r, pgid, false);
    }
    // Something the command left in the background may still hold the pipe: don't wait for it.
    let _ = done_rx.recv_timeout(Duration::from_secs(1));
    let output = String::from_utf8_lossy(&buf.lock().unwrap_or_else(|e| e.into_inner())).into_owned();
    CmdResult { code, timed_out, output }
}

static REGISTRY: Mutex<()> = Mutex::new(());

fn read_registry(registry: &Path) -> Vec<i32> {
    std::fs::read_to_string(registry).unwrap_or_default().lines().filter_map(|l| l.trim().parse().ok()).filter(|p| *p > 1).collect()
}

/// Adds or removes a process group ID in the registry file (one per line).
fn update_registry(registry: &Path, pgid: i32, add: bool) {
    let _g = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let mut ids = read_registry(registry);
    ids.retain(|&p| p != pgid);
    if add {
        ids.push(pgid);
    }
    if let Some(dir) = registry.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let text: String = ids.iter().map(|p| format!("{p}\n")).collect();
    if let Err(e) = std::fs::write(registry, text) {
        eprintln!("cannot write {}: {e}", registry.display());
    }
}

/// The process group of a process (/proc/PID/stat, field 5).
fn pgrp(pid: i32) -> Option<i32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    stat.rsplit_once(") ")?.1.split(' ').nth(2)?.parse().ok()
}

/// For the start of the service, before its first `run_killing` command: kills (SIGKILL) the
/// process groups an earlier run left in `registry` that are still alive and still run a command
/// containing `marker` (so a group ID reused by something else is left alone), waits up to 3 s
/// for each to go, and empties the registry. Returns the groups it killed.
pub fn kill_leftover_groups(registry: &Path, marker: &str) -> Vec<i32> {
    let _g = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    let members = |pgid: i32| processes().into_iter().filter(|p| pgrp(p.pid) == Some(pgid)).collect::<Vec<_>>();
    let mut killed = Vec::new();
    for pgid in read_registry(registry) {
        if !members(pgid).iter().any(|p| p.cmdline.contains(marker)) {
            continue;
        }
        // SAFETY: kill(2) on a process group that runs our command.
        unsafe { libc::kill(-pgid, libc::SIGKILL) };
        killed.push(pgid);
        let deadline = Instant::now() + Duration::from_secs(3);
        while !members(pgid).is_empty() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    let _ = std::fs::remove_file(registry);
    killed
}

/// Single-quoted for bash.
pub fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// `bash -lc <cmd>` left running with its stdout piped (stderr dropped), like `run` otherwise.
pub fn spawn_piped(cmd: &str) -> std::io::Result<std::process::Child> {
    let mut command = Command::new("bash");
    command.arg("-lc").arg(cmd).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    if let Some(home) = std::env::var_os("HOME") {
        command.current_dir(home);
    }
    // SAFETY: as in `run`; PR_SET_PDEATHSIG only concerns this (child) process.
    unsafe {
        command.pre_exec(|| {
            libc::prctl(libc::PR_CAP_AMBIENT, libc::PR_CAP_AMBIENT_CLEAR_ALL as libc::c_ulong, 0, 0, 0);
            // A monitor must not outlive the page (systemd's KillMode=process leaves children).
            // SIGKILL: rclpy turns SIGTERM into a shutdown that ros2 topic echo can sit through.
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong, 0, 0, 0);
            Ok(())
        });
    }
    command.spawn()
}

pub struct Proc {
    pub pid: i32,
    pub uid: u32,
    pub cmdline: String,
}

/// Every process with a command line (not kernel threads or zombies), except this one and
/// pgrep / pkill / grep, whose own command lines contain the patterns they look for.
pub fn processes() -> Vec<Proc> {
    let me = std::process::id() as i32;
    let Ok(dir) = std::fs::read_dir("/proc") else { return Vec::new() };
    let mut out = Vec::new();
    for entry in dir.flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|s| s.parse::<i32>().ok()) else { continue };
        if pid == me {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        let Some(cmdline) = cmdline(pid) else { continue };
        let argv0 = cmdline.split(' ').next().unwrap_or("");
        let name = argv0.rsplit('/').next().unwrap_or(argv0);
        if matches!(name, "pgrep" | "pkill" | "grep") {
            continue;
        }
        out.push(Proc { pid, uid: meta.uid(), cmdline });
    }
    out
}

/// /proc/<pid>/cmdline with the arguments joined by spaces; None for kernel threads, zombies and
/// processes that are gone.
pub fn cmdline(pid: i32) -> Option<String> {
    let raw = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    if raw.is_empty() {
        return None;
    }
    let s = String::from_utf8_lossy(&raw);
    Some(s.trim_end_matches('\0').replace('\0', " "))
}

pub fn comm(pid: i32) -> String {
    std::fs::read_to_string(format!("/proc/{pid}/comm")).map(|s| s.trim().to_string()).unwrap_or_else(|_| "?".into())
}

/// Processes (readable to us: our own user's) that have `dev` open.
pub fn fd_holders(dev: &Path) -> Vec<(i32, String)> {
    let Ok(target) = std::fs::canonicalize(dev) else { return Vec::new() };
    let me = std::process::id() as i32;
    let mut out = Vec::new();
    let Ok(dir) = std::fs::read_dir("/proc") else { return out };
    for entry in dir.flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|s| s.parse::<i32>().ok()) else { continue };
        if pid == me {
            continue;
        }
        let Ok(fds) = std::fs::read_dir(entry.path().join("fd")) else { continue };
        if fds.flatten().any(|fd| std::fs::read_link(fd.path()).ok().as_deref() == Some(target.as_path())) {
            out.push((pid, comm(pid)));
        }
    }
    out
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gone(pid: i32) -> bool {
        // Killed and reparented: gone, or a zombie until init reaps it.
        std::fs::read_to_string(format!("/proc/{pid}/stat")).map_or(true, |s| s.rsplit_once(") ").is_some_and(|(_, r)| r.starts_with('Z')))
    }

    #[test]
    fn run_killing_ends_everything_it_started() {
        let dir = std::env::temp_dir().join(format!("roomba-control-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let pid_file = dir.join("pid");
        // A shell that ignores SIGTERM and a child of it: both must be gone when run_killing returns.
        let cmd = format!("trap '' TERM; sleep 30 & echo $! > {}; wait", shell_quote(pid_file.to_str().unwrap()));
        let registry = dir.join("groups");
        let t0 = Instant::now();
        let r = run_killing(&cmd, Duration::from_millis(800), &registry);
        assert!(r.timed_out && r.code.is_none());
        assert!(read_registry(&registry).is_empty(), "unregistered when it ends");
        assert!(t0.elapsed() < Duration::from_secs(5));
        let pid: i32 = std::fs::read_to_string(&pid_file).unwrap().trim().parse().unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while !gone(pid) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(gone(pid), "the background sleep {pid} survived");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_restart_kills_the_motor_commands_left_running() {
        let dir = std::env::temp_dir().join(format!("roomba-control-test-kill-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let registry = dir.join("groups");
        // As a service that died mid-command leaves them: running, each in its own group, listed.
        let group = |marker: &str| {
            let mut c = Command::new("bash");
            c.arg("-c").arg("sleep 30; true").arg(marker).process_group(0);
            c.spawn().unwrap()
        };
        let mut ours = group("cleaning_motors-test");
        let mut other = group("something-else");
        for c in [&ours, &other] {
            update_registry(&registry, c.id() as i32, true);
        }
        std::thread::sleep(Duration::from_millis(200));
        let killed = kill_leftover_groups(&registry, "cleaning_motors-test");
        assert_eq!(killed, [ours.id() as i32]);
        assert!(ours.try_wait().unwrap().is_some(), "killed");
        assert!(other.try_wait().unwrap().is_none(), "a group without the marker is left alone");
        assert!(!registry.exists());
        let _ = other.kill();
        let _ = other.wait();
        let _ = ours.wait();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
