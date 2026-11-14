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
| `src/roomba_perception` | VSLAM watchdog/odometry filter with wheel bridging, floor obstacle detection, explorer, ignore zones, nav coach, person detector, IMU calibration |
| `src/roomba_web` | Web app on port 8080: live map, robot status, drive pad, tap-to-go navigation, login |
| `src/mynteye_d1000_driver` | ROS 2 node for the MYNT EYE D1000 stereo/depth/IMU camera |
| `src/basalt_vio_ros` | ROS 2 wrapper for Basalt visual-inertial odometry |
| `src/roomba-serial-ros2` | Roomba Open Interface driver (submodule) |
| `scripts/` | `run_robot.sh`, `stop_slam.sh`, mapping/replay helpers, setup scripts |
| `tools/` | Probes and diagnostics (motion step test, costmap/obstacle probes, map render); `tools/oi_tests/` holds early serial/OI bench scripts |
| `docs/` | Wiring and power diagrams |

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

## Build

```bash
git clone --recurse-submodules https://github.com/Brett-D/roomba-jetson-stack.git ~/roomba_ws
cd ~/roomba_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
```

## Run

```bash
~/roomba_ws/scripts/run_robot.sh            # reuse the saved map, or start one if none exists
~/roomba_ws/scripts/run_robot.sh mapping    # start a new map (old one backed up to ~/maps/backup)
~/roomba_ws/scripts/stop_slam.sh            # stop everything
```

Then open `http://<jetson-ip>:8080` and wait for **Tracking OK**. Wake the Roomba first (press Clean) if its lights are off.
The web login password is generated on first run and stored in `~/.config/roomba_web/initial_password`;
change it with `ros2 run roomba_web set_password`.

## Known quirks

- Sending Start (128) drops the Roomba from Safe to Passive; the driver only re-sends it after a repeated stream stall.
- This robot's right bumper switch is stuck; `roomba_driver_robot.yaml` ignores `bump_right` and stops on the light bumper instead.
- Nav2 Jazzy velocity polygons must cover every velocity range, so the collision monitor has small `turn` and `reverse` sub-polygons.
- Wheel odometry over-reports in-place turns by about 40% on carpet; the watchdog trusts gyro and VIO for rotation.

## License

Apache-2.0, except `src/basalt_vio_ros`, which is BSD-3-Clause like Basalt itself.
