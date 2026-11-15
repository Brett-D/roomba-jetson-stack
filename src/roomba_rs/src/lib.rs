//! Pure processing used by the Rust nodes, kept free of ROS types so it can be unit tested.
pub mod floor;
pub mod geometry;
pub mod static_tf;

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
