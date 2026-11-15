//! The 2D digital twin of the house and the Roomba (src/bin/twin2d.rs): a simulated robot that
//! the real navigation stack (roomba_driver over a pty, floor_obstacles, Nav2, explorer, ...) drives
//! around the house, plus scenario events and the outcome measurements. ROS-free:
//! - [`world`]: floor plan (solid grid from RTAB-Map's saved map) plus doors, railings, obstacles,
//!   people and cliffs, with the geometry queries (clearance, 2D rays);
//! - [`scenario`]: the scenario file (YAML);
//! - [`physics`]: differential drive with collisions, bumpers, light bumpers, cliffs, battery;
//! - [`oi_sim`]: the Roomba's Open Interface as the driver sees it on the serial port;
//! - [`depth`]: the depth camera (2.5D ray casting) and the IMU;
//! - [`mapper`]: RTAB-Map's 2D occupancy grid built from what the simulated camera saw;
//! - [`metrics`]: outcome measurements and the trajectory / coverage plot;
//! - [`rng`]: a small seeded random generator (runs are deterministic given the seed, apart from
//!   the timing of the real nodes).
pub mod depth;
pub mod mapper;
pub mod metrics;
pub mod oi_sim;
pub mod physics;
pub mod rng;
pub mod scenario;
pub mod world;
