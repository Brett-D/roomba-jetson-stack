//! Processing used by the Rust nodes, kept free of ROS types so it can be unit tested - except
//! `roomba` (which owns the Roomba's serial port) and
//! `tf_listener`, the /tf subscriptions feeding `tf_buffer` (tf2's BufferCore for rclrs nodes).
pub mod battery;
pub mod cv;
pub mod driver;
pub mod ekf;
pub mod explorer;
pub mod floor;
pub mod gap_filler;
pub mod geometry;
pub mod kinematics;
pub mod nav_coach;
pub mod node_params;
pub mod ocv;
pub mod oi;
pub mod person;
pub mod roomba;
pub mod scene_filter;
pub mod static_tf;
pub mod tf_buffer;
pub mod tf_listener;
pub mod trt;
pub mod twin;
pub mod watchdog;
pub mod web;
pub mod zones;

/// Makes SIGINT and SIGTERM end the process at once with exit status 0, as an rclpy node exits on
/// Ctrl-C (killed by the signal, launch reported every Rust node as "process has died" with an
/// ERROR line at each shutdown). A process started in the background by a non-interactive shell
/// (run_robot.sh, record scripts) inherits SIGINT as ignored, and rclrs installs no handler, so
/// without this Ctrl-C / launch shutdown would not stop the node. Call first thing in main().
/// Nodes that need to act on shutdown block the signals afterwards and take them with
/// [`node_params::on_signal`].
pub fn default_signals() {
    extern "C" fn exit_now(_: libc::c_int) {
        // SAFETY: _exit is async-signal-safe.
        unsafe { libc::_exit(0) }
    }
    // SAFETY: signal() only changes this process's signal dispositions; the handler only calls _exit.
    unsafe {
        libc::signal(libc::SIGINT, exit_now as extern "C" fn(libc::c_int) as libc::sighandler_t);
        libc::signal(libc::SIGTERM, exit_now as extern "C" fn(libc::c_int) as libc::sighandler_t);
    }
}
