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
pub mod oi;
pub mod roomba;
pub mod scene_filter;
pub mod static_tf;
pub mod tf_buffer;
pub mod tf_listener;
pub mod watchdog;
pub mod web;
pub mod zones;

/// Restores the default action (terminate) for SIGINT and SIGTERM. A process started in the
/// background by a non-interactive shell (run_robot.sh, record scripts) inherits SIGINT as
/// ignored, and rclrs installs no handler, so without this Ctrl-C / launch shutdown would not
/// stop the node. Call first thing in main().
pub fn default_signals() {
    // SAFETY: signal() with SIG_DFL only changes this process's signal dispositions.
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_DFL);
        libc::signal(libc::SIGTERM, libc::SIG_DFL);
    }
}
