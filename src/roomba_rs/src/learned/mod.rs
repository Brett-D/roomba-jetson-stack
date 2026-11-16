//! Learned navigation skills (phase 1, opt-in cargo feature `learned`; docs/learned-skills.md).
//!
//! Two small ONNX policies trained in simulation, run on the CPU with tract (pure Rust):
//! - [`unstuck`]: drives the robot out of a stuck situation from the local costmap patch, the
//!   light bumpers, bumpers, wheel odometry, IMU and the goal it was heading to.
//! - [`explore`]: re-ranks the explorer's own frontier candidates (it never adds a candidate, so
//!   every existing explorer safety rule still applies).
//!
//! [`patch`] and the observation / feature builders are plain Rust and reproduce the training
//! environments' layouts exactly (twin/learned/unstuck/unstuck_env.py, twin/learned/explore/env2d.py).
pub mod explore;
pub mod onnx;
pub mod patch;
pub mod unstuck;
