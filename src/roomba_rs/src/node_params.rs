//! Parameter declaration for the rclrs nodes (a [`ParamSource`] over a node). The handles are kept
//! for the node's lifetime: rclrs undeclares a parameter when its handle is dropped, and
//! `ros2 param get/list` should show them as for the Python nodes (which read them once at startup).

use std::any::Any;
use std::sync::Arc;

use rclrs::*;

use crate::watchdog::ParamSource;

pub struct NodeParams<'a> {
    node: &'a Node,
    handles: Vec<Box<dyn Any>>,
}

impl<'a> NodeParams<'a> {
    pub fn new(node: &'a Node) -> Self {
        NodeParams { node, handles: Vec::new() }
    }

    fn keep<T: ParameterVariant + 'static>(&mut self, p: MandatoryParameter<T>) -> T {
        let value = p.get();
        self.handles.push(Box::new(p));
        value
    }
}

impl ParamSource for NodeParams<'_> {
    fn float(&mut self, name: &str, default: f64) -> anyhow::Result<f64> {
        let p = self.node.declare_parameter(name).default(default).mandatory()?;
        Ok(self.keep(p))
    }
    fn int(&mut self, name: &str, default: i64) -> anyhow::Result<i64> {
        let p = self.node.declare_parameter(name).default(default).mandatory()?;
        Ok(self.keep(p))
    }
    fn boolean(&mut self, name: &str, default: bool) -> anyhow::Result<bool> {
        let p = self.node.declare_parameter(name).default(default).mandatory()?;
        Ok(self.keep(p))
    }
    fn string(&mut self, name: &str, default: &str) -> anyhow::Result<String> {
        let p = self.node.declare_parameter::<Arc<str>>(name).default(default.into()).mandatory()?;
        Ok(self.keep(p).to_string())
    }
}

/// time.monotonic(): CLOCK_MONOTONIC in seconds.
pub fn mono() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: clock_gettime only writes the timespec passed to it.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// Blocks SIGINT and SIGTERM in this thread and in every thread started after it; [`on_signal`]
/// then takes them in a thread of its own. Call first thing in main (after
/// [`crate::default_signals`], so that a signal ignored by the parent shell is caught again).
pub fn block_signals() -> libc::sigset_t {
    // SAFETY: plain signal-mask calls on a local sigset_t.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
        set
    }
}

/// On SIGINT / SIGTERM: runs `f`, then exits with status 0 - like an rclpy node on Ctrl-C, so
/// launch reports a clean finish rather than a process killed by a signal.
pub fn on_signal(set: libc::sigset_t, f: impl FnOnce() + Send + 'static) {
    std::thread::spawn(move || {
        let mut sig = 0;
        // SAFETY: sigwait only reads the set and writes `sig`.
        unsafe { libc::sigwait(&set, &mut sig) };
        f();
        std::process::exit(0);
    });
}

/// Spins the executor for the node's lifetime. Jazzy's rcl_wait returns RCL_RET_TIMEOUT when a wait
/// that ended for a ROS-time timer finds the timer not due yet because the system clock was stepped
/// back meanwhile (WSL2's time sync steps it every ~30 s; NTP can too). rclrs 0.8 reports that as
/// an error and stops spinning, which ended the node: spin again when only timeouts were reported.
pub fn spin(executor: &mut Executor) -> Result<(), RclrsError> {
    spin_until(executor, || false).first_error()
}

/// [`spin`] that also returns once `done()` holds after a spin ends (set the flag `done` reads
/// before calling `halt_spinning`). Returns the errors other than those timeouts.
pub fn spin_until(executor: &mut Executor, done: impl Fn() -> bool) -> Vec<RclrsError> {
    loop {
        let errors = executor.spin(SpinOptions::default());
        if errors.is_empty() || done() || errors.iter().any(|e| !e.is_timeout()) {
            return errors.into_iter().filter(|e| !e.is_timeout()).collect();
        }
    }
}
