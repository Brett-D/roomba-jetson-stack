# roomba-jetson-stack

ROS 2 Jazzy stack that turns an old **iRobot Roomba 550** into a mapping, self-navigating robot.
A **Jetson Orin Nano Super** rides on a 3D-printed platform with a **MYNT EYE D1000** stereo camera
and drives the Roomba through its Open Interface serial port.

The serial driver lives in its own repository and is included here as a submodule:
[Brett-D/roomba-serial-ros2](https://github.com/Brett-D/roomba-serial-ros2).

## What's in here

| Path | Contents |
|---|---|
| `src/roomba_bringup` | Launch files (`robot.launch.py`, `vslam.launch.py`, `nav.launch.py`, `handheld_slam.launch.py`), Nav2 configs, robot driver config |
| `src/roomba_perception` | VSLAM watchdog/odometry filter with wheel bridging, EKF input conditioning (`ekf_inputs`), floor obstacle detection, explorer, ignore zones, nav coach, person detector, IMU calibration |
| `src/roomba_rs` | Rust ports of hot-path nodes (rclrs): `floor_obstacles`, `ekf_inputs`, `vslam_watchdog`, `map_gap_filler` and `ignore_zones` (same topics, parameters, files and output as the Python nodes at a fraction of the CPU; `vendor/` holds generated Rust bindings for the Isaac ROS VSLAM interfaces). `tf_buffer`/`tf_listener` give rclrs nodes a tf2-equivalent TF buffer (bit-identical lookups). `rust_nodes:=false` switches back to the Python nodes |
| `src/roomba_web` | Web app on port 8080: live map, robot status, drive pad, tap-to-go navigation, login |
| `src/mynteye_d1000_driver` | ROS 2 node for the MYNT EYE D1000 stereo/depth/IMU camera |
| `src/basalt_vio_ros` | ROS 2 wrapper for Basalt visual-inertial odometry |
| `src/roomba-serial-ros2` | Roomba Open Interface driver (submodule) |
| `scripts/` | `run_robot.sh`, `stop_slam.sh`, `record_run.sh` / `compare_pipelines.sh` (A vs B on a recorded drive), mapping/replay helpers, setup scripts |
| `tools/` | Probes and diagnostics (motion step test, costmap/obstacle probes, map render, TF/drift/CPU-GPU checks, comparison monitor/renderer, Python vs Rust A/B checks); `tools/oi_tests/` holds early serial/OI bench scripts |
| `docs/` | Wiring and power diagrams |
| `hardware/platform` | 3D-printed platform: STLs, fit tests and the Python generator |

## Data flow

```
MYNT D1000 ──> Basalt VIO ──> vslam_watchdog ──> odom ──> RTAB-Map (map) ──> Nav2 ──> collision monitor ──> /cmd_vel
                                  ▲                                                                         │
Roomba wheel odometry + gyro ─────┘                                              roomba_driver (serial) <───┘
```

TF: `map → odom → base_link → camera_link` (camera mount: x 0.135 m, z 0.12 m).

## Hardware

- Roomba 550 (R3 board), Open Interface at 115200 baud
- Jetson Orin Nano Super 8 GB, ROS 2 Jazzy
- MYNT EYE D1000 stereo camera (USB)
- FT232 USB-TTL adapter set to **5 V** on the Roomba mini-DIN (the Jetson header UART with a BSS138 shifter was unreliable)
- Power: Ridgid 18 V Li-ion pack → 7.5 A fuse → 15 V / 12 A buck regulator → Roomba battery contacts and Jetson.
  See [`docs/battery-voltage-monitor.png`](docs/battery-voltage-monitor.png) for the ADS1115 pack-voltage monitor.
  **Do not dock or plug in the Roomba charger while running from the Ridgid pack.**
- 3D-printed platform: STLs and generator in [`hardware/platform/`](hardware/platform/)

System wiring (power, camera and serial), [`docs/system-wiring.png`](docs/system-wiring.png):

![System wiring](docs/system-wiring.png)

## Build

```bash
git clone --recurse-submodules https://github.com/Brett-D/roomba-jetson-stack.git ~/roomba_ws
cd ~/roomba_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -y
# Rust toolchain for src/roomba_rs (cargo fetches rclrs; message crates come from /opt/ros)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
sudo apt install -y libclang-dev mold && source ~/.cargo/env   # mold: GNU ld takes 10+ min to link
colcon build --symlink-install
```

Python vs Rust checks on a recorded drive: `tools/ab_floor_obstacles.sh` (side by side, frame by
frame, plus CPU), `tools/ab_ekf_offline.py` (same message sequence into both, every output compared)
and `tools/ab_ekf_inputs.sh` (CPU). For the watchdog, `tools/record_vio_replay.sh` first replays a
raw recording through the VIO and records its output; `tools/ab_watchdog_offline.py` then feeds that
to both watchdogs (TF, odometry, status, service calls and logs compared, optionally with injected
faults) and `tools/ab_watchdog.sh` measures CPU. `tools/ab_tf_buffer.py` compares the Rust TF buffer
with tf2 on recorded /tf. `tools/record_map_replay.sh` replays a raw recording through RTAB-Map and
records its grid and path; `tools/ab_map_gap_filler_offline.py` compares the filled grids cell for
cell and `tools/ab_ignore_zones_offline.py` the zone outputs (JSON, masks, markers, zones file) on
scripted and random commands, clicks and bump marks; `tools/ab_scene_filter.py` checks the zone/line
of sight helpers and `tools/ab_zones_gap_filler_cpu.sh` measures CPU and memory of both pairs.

## Run

```bash
~/roomba_ws/scripts/run_robot.sh            # reuse the saved map, or start one if none exists
~/roomba_ws/scripts/run_robot.sh mapping    # start a new map (old one backed up to ~/maps/backup)
~/roomba_ws/scripts/stop_slam.sh            # stop everything
```

Then open `http://<jetson-ip>:8080` and wait for **Tracking OK**. Wake the Roomba first (press Clean) if its lights are off.
The web login password is generated on first run and stored in `~/.config/roomba_web/initial_password`;
change it with `ros2 run roomba_web set_password`.

### Pipelines A (default) and B (GPU)

Extra arguments to `run_robot.sh` / `robot.launch.py` pick the odometry, the fusion and the costmap obstacles:

| Argument | Default (A) | GPU pipeline (B) |
|---|---|---|
| `odometry:=` | `basalt` (CPU VIO) | `cuvslam` (Isaac ROS Visual SLAM) |
| `fusion:=` | `watchdog` (vslam_watchdog publishes `odom → base_link`) | `ekf` (robot_localization: wheels + calibrated gyro + VIO, `config/ekf_robot.yaml`) |
| `obstacles:=` | `floor` (floor_obstacles + voxel layer, `nav2_robot.yaml`) | `nvblox` (GPU TSDF/ESDF, `config/nvblox_robot.yaml`, `nav2_robot_nvblox.yaml`) |

```bash
~/roomba_ws/scripts/run_robot.sh mapping                                                   # A
~/roomba_ws/scripts/run_robot.sh mapping odometry:=cuvslam fusion:=ekf obstacles:=nvblox    # B
```

The three choices are independent. With `fusion:=ekf` the watchdog still checks the VIO, resets it and
publishes `/vslam_filter/status`. The collision monitor always uses floor_obstacles.

### Recording a drive and comparing the pipelines

```bash
~/roomba_ws/scripts/record_run.sh [--duration S] [--zstd] [--name X]   # with the robot running; Ctrl-C stops
~/roomba_ws/scripts/compare_pipelines.sh ~/bags/robot_<stamp> [--background] [--rate 0.5] [--only A|B]
```

`record_run.sh` records only the raw camera, IMU and Roomba topics plus `/tf_static` (no odometry/map TF) to
`~/bags/robot_<stamp>` (mcap, about 1.2 GB/min uncompressed). `compare_pipelines.sh` replays it through A and B
(sim time, camera and Roomba drivers off), each into its own RTAB-Map database, and writes metrics (tracking losses,
loop closures, map area, end-vs-start pose, CPU/GPU), a side-by-side map image and an overlay to `/data/maps/compare/`.

## Known quirks

- Sending Start (128) drops the Roomba from Safe to Passive; the driver only re-sends it after a repeated stream stall.
- This robot's right bumper switch is stuck; `roomba_driver_robot.yaml` ignores `bump_right` and stops on the light bumper instead.
- Nav2 Jazzy velocity polygons must cover every velocity range, so the collision monitor has small `turn` and `reverse` sub-polygons.
- Wheel odometry over-reports in-place turns by about 40% on carpet; the watchdog trusts gyro and VIO for rotation.

## License

Apache-2.0, except `src/basalt_vio_ros`, which is BSD-3-Clause like Basalt itself.
