# Isaac Sim training twin (house + Roomba 550 + MYNT EYE D1000)

Phase 1 of the "both in parallel" plan: **Isaac Sim on the Windows PC for training** (synthetic data,
later learned behaviours), next to the 2D navigation twin on the Jetson. Everything here runs on the
PC; nothing is deployed to the robot.

```
twin/isaac/
  env.ps1, ros_env.ps1        cache/temp locations on E:\twin, bundled ROS 2 Jazzy for rclpy
  install_isaacsim.ps1        uv + Python 3.12 + CUDA PyTorch + isaacsim 6.2.0 (pip) + eval env
  pdownload.py                resumable parallel download (the 5.5 GB extscache wheel kept dropping)
  jetson/                     run on the Jetson: bag extraction, house mesh/grid export, IMU axis check
  robot/roomba550_d1000.urdf  Roomba 550 + D1000 (frames copied from the real /tf_static)
  remodel/                    floor-plan house: digitised floor plans, registration, USD build, validation
  scripts/                    Isaac Sim scripts (asset build, scene, ROS bridge, data generation)
  eval/                       detector baseline, sim-vs-real image statistics, fine-tune proof
  run_closed_loop.ps1         closed loop: Isaac Sim (GUI) drives the robot's real stack in WSL (section 7)
  wsl/                        WSL side of the closed loop: workspace setup, TensorRT engines, OI emulator, monitor
  foxglove/                   Foxglove/Lichtblick layout for the closed loop (ws://localhost:18766)
```
Big files stay out of git, under `E:\twin` (install, caches, assets, datasets, runs, screenshots).

## 1. Install (Windows 11, RTX 5090, driver 621.14)

| | |
|---|---|
| Isaac Sim | **6.2.0** (Oct 2026, Kit 110.4.0), pip packages `isaacsim[all,extscache]==6.2.0.0` |
| Python | 3.12.16 (uv-managed, `E:\twin\python`); Isaac Sim 6.x requires 3.12 (3.11 was 5.x) |
| PyTorch | 2.11.0+cu128 (Blackwell needs CUDA 12.8+) |
| Renderer | RTX Real-Time 2.0 (path traced) on Vulkan; the Intel iGPU is skipped |
| Location | venv `E:\twin\isaacsim-env`, eval/train venv `E:\twin\eval-env`, caches `E:\twin\cache` |

Why pip and not the standalone zip: same 6.2.0 bits, but a normal venv (scripts run with plain
`python`), works with uv, and is the documented route for Isaac Lab 3.0 (GA, supports Isaac Sim 6.2,
pip install on Windows). Requirements met: Windows 11, RTX ≥ 4080 / 16 GB (5090, 32 GB),
driver ≥ 595.97, 32 GB+ RAM, SSD. WSL and the container are not supported for running the simulator
on Windows (the container is Linux-only).

```powershell
powershell -ExecutionPolicy Bypass -File twin\isaac\install_isaacsim.ps1
```
Installing means accepting the NVIDIA Omniverse License Agreement (`OMNI_KIT_ACCEPT_EULA=YES` in
`env.ps1`). Notes:
- Always dot-source `env.ps1` first: uv/pip/torch/warp/ultralytics caches and `%TEMP%` go to
  `E:\twin` (nothing large on C:, nothing on F:). Kit itself still keeps its small shader/log cache in
  `%LOCALAPPDATA%\ov`.
- `isaacsim-extscache-kit` is a single 5.5 GB wheel; over the VPN its download failed repeatedly
  ("peer closed connection"). `pdownload.py` fetches it in resumable parallel chunks; the install script checks its sha256
  (from `https://pypi.nvidia.com/isaacsim-extscache-kit/`) and installs it from the local file.
- Windows long paths were already enabled.
- Smoke test: `scripts\smoke_render.py` renders a frame headless (first start ~55 s, first frame
  ~100 s while shaders compile; later starts are faster).
- Render settings: use DLAA (or DLSS). With FXAA/TAA the real-time path tracer is left un-denoised
  (Immerkaer noise sigma 60 vs 1.4 with DLAA, `scripts\render_settings_test.py`).

## 2. Data from the robot (`jetson/`)

Work happens only in `/data/twin/isaac` on the Jetson (the source DB is read-only; `~/maps`,
`~/models` untouched).

```bash
python3 extract_bag.py /data/bags/robot_20261114_145937_fresh_map8 /data/twin/isaac/bag_extract
cp /data/twin/source/vslam.db /data/twin/isaac/vslam.db && bash export_house.sh   # grid, poses, cloud, mesh
python3 imu_axes.py
```
Measured from the fresh_map8 recording (2076 s, no MCAP index → one full scan, ~2.5 min):

| | real D1000 / Roomba |
|---|---|
| images | 640x360, **15.0 Hz** (not 30: output_scale 0.5 halves the rate too), left `bgr8`, mono `mono8`, depth `16UC1` mm registered to left |
| K (all three) | fx = fy = 349.2, cx = 318.035, cy = 174.522, no distortion (rectified) |
| right P[3] | -41.3376 → baseline **0.118378 m** (D1000-120) |
| FOV | 85.0° x 54.5° |
| IMU | 199.7 Hz, frame `camera_imu_frame` = left optical rotated 180° about x (x right, y up, z back); at rest accel (-0.14, 9.76, 0.09), noise σ ≈ 0.018 m/s², gyro σ ≈ 0.001 rad/s; no orientation |
| TF | base_link→camera_link (0.153, 0, 0.13); camera_link→left/right optical (0, ±0.05919, 0), rpy (-90°, 0, -90°) |
| drive | wheel base 0.235 m, wheel Ø 0.072 m, max wheel speed 0.4 m/s, wheel odom 30 Hz |

The camera lens is **0.13 m** above the floor (tape; depth reads ~0.11 m), so the data pipeline uses
0.13 m (jittered 0.11-0.16 m), not 0.24 m. The bag has no `/perception/person_detector/*`, no `/tf`
and no `/rtabmap/mapPath`; the robot path comes from the database's optimized graph instead
(`house_poses.txt`, 620 base_link poses in the map frame).

## 3. Assets

```powershell
. twin\isaac\env.ps1; cd twin\isaac\scripts
python build_house.py      # E:\twin\assets\usd\house.usd (+ house_textures\, house_stats.json)
python build_robot.py      # E:\twin\assets\usd\roomba550_d1000.usd (+ roomba_import\, _info.json)
python test_robot.py       # projection / depth / drive checks
python render_views.py     # top-down, oblique and sim-vs-real views -> E:\twin\screens
```
**House** (`build_house.py`), all in RTAB-Map's `map` frame so it lines up with the Jetson's 2D twin:
- `/House/Scan/*`: textured mesh from `rtabmap-export --mesh --texture` (Poisson, 6 × 4096² textures,
  97.7k faces), split by a geometric heuristic into wall / furniture labels; the bumpy scan floor
  and stretched gap-bridging triangles are dropped.
- `/House/Blocks/{wall,furniture}`: occupancy-grid cells extruded to the height of the scan points
  above them (≥ 1.3 m → full-height wall). These are the colliders.
- `/House/Floor` (collider) and `/House/Ceiling` at 2.5 m, variant set `look = scan | clean`
  (clean shows the blocks instead of the scan; used for domain randomisation).

**Robot** (`robot/roomba550_d1000.urdf` → Isaac Sim 6.2 URDF importer → `build_robot.py`):
articulation rooted at `base_link`, two velocity-driven wheel joints, frictionless front caster and
rear skid, tyre friction, D1000 housing, USD pinhole cameras on both optical frames reproducing the
real K (principal-point offset included), `IsaacImuSensor` at `camera_imu_frame`.
`test_robot.py`: a target projected 0.09 px from the real-K prediction, depth correct, 0.2 m/s for
3 s → 0.593 m (yaw drift 0.3°), 0.8 rad/s for 3 s → 123° (89 %, wheel scrub).

**People**: Isaac Sim People characters (10 outfits) animated by the Replicator Agent (IRA)
behaviour runtime on a NavMesh baked from the house (`twin_scene.py`). A bare UsdSkel binding does
not work: the characters use a Reallusion rig and the clips the Biped rig; IRA's motion library
retargets them.

### Remodelled house from the floor plans (default look, `remodel/`)
The scan mesh was too poor, so the house is rebuilt from the floor plans
(`E:\twin\assets\floorplan_upstairs.png`, `floorplan_downstairs.png`), registered into the RTAB-Map
`map` frame. Scenes, the ROS twin and the data generator use it by default (`--look scan|clean` still work).

```powershell
. twin\isaac\env.ps1; cd twin\isaac\remodel
python asset_catalog.py; python textures.py             # ArchVis furniture sizes, 12 tileable textures
python plan_register.py                                 # upstairs plan -> map frame (registration.json + overlay)
python build_remodel.py --with-floor1                   # upstairs  -> house_remodel.usd, floorplan.json/.svg
python build_remodel.py --floor1                        # downstairs -> house_remodel_f1.usd, floorplan_floor1.json/.svg
python validate_remodel.py                              # metrics + E:\twin\screens\remodel_overlay.png
python ..\scripts\render_views.py --floor1 --prefix remodel_   # overviews, stair views, sim-vs-real
```
- **Digitising** (`floorplan_upstairs_px.json`, `floorplan_floor1_px.json`, image pixels): rooms (polygon,
  name, type, stated dimensions), doors (single / double / glass / garage / cased opening, hinge side
  and swing), windows, stair, railings, bathroom / kitchen / laundry fixtures. Walls are the plan's
  thick black pixels (`plan_extract.wall_mask`) extruded as drawn (curved bays and diagonals included);
  downstairs, exterior gaps that are not doors become windows automatically.
- **Scale** from the stated room sizes: upstairs **36.80 px/m, spread 0.39 %** over 8 rooms; downstairs
  fitted on its own 36.75 px/m (-0.15 %, 10 rooms; spread 2.5 % from the small rooms) → same drawing
  scale, 36.80 used for both.
- **Registration** (upstairs ↔ occupancy grid, `plan_register.py`): exhaustive rotation (+ mirror) and
  FFT translation search on plan wall points vs occupied cells, then ICP with scale. Result: rotation
  100.33°, translation (2.512, 2.111) m, scale 0.976, no mirror. 58 % of the plan walls in the mapped
  area lie within 15 cm of occupied cells (median 9.8 cm), 96.9 % of the recorded path is on plan floor
  and no pose is in the stair void; the runner-up (121.6°) scored clearly lower. The map is the
  upstairs; the robot never mapped the downstairs.
- **Stairwell** (checked against the downstairs plan): the curved stair rises clockwise from the entry;
  its top is the north end of the crescent next to the arrow. Upstairs, the well circle, the stair and
  the two-storey entry south of it (up to the front wall with the tall windows) are open to below,
  bounded by the corridor railing (west, down to the small closet) and the well railing (north/east, to
  the east wall). The stair top itself has no railing: that is the drop-off. 16 risers × 0.181 m.
- **Downstairs stacking**: same scale and orientation as upstairs; offset (13, 207) px found with the
  staircase as anchor (storage under the upper flight, stair lines of the overlapping flight match
  84 % within 2 px) refined by a wall chamfer search (37 % of upstairs wall pixels within 3 px of a
  downstairs wall: the outer walls stack). Assumed heights: 2.44 m upstairs walls, 2.65 m downstairs
  ceilings + 0.25 m slab → **2.90 m floor to floor**. The downstairs USD sits at z = -2.90 m under
  `/World/HouseF1`; its ceiling is open under the upstairs void (two-storey entry).
- **Contents**: upstairs 15 rooms, 13 hinged doors (open by default; `twin:closed_yaw_deg` /
  `twin:open_yaw_deg` on `/World/House/Doors/<id>/Leaf<k>` to close them; the balcony glass doors are
  closed), 6 windows, 3 bathrooms' fixtures, closets, railings, 17 room lights + chandelier;
  furniture: 5 pieces from interior obstacle blobs of the grid (where the robot saw them) + 14 room
  templates (procedural beds, ArchVis dressers / nightstands / shelves / chairs / plants). Downstairs
  17 rooms (kitchen with counters, island and stools, pantry, dining, living, office, family room,
  bathroom, laundry, garage, entry with the storage under the stair, two balconies, veranda), 9 hinged
  doors (front door closed), closed garage door, 7 cased openings, 19 windows, 28 lights,
  32 furniture pieces. Floors are textured per room type; materials and lights are randomised by
  `TwinScene.randomize_remodel` and `gen_dataset.py`.
- **Physics**: the walkable floor collider is one welded grid mesh of the floor tops; per-room floor
  boxes left internal faces at their seams that stopped the caster dead (PhysX ghost contacts).
- **Validation** (`validation.json`, `remodel_overlay.png`): wall coverage 52.1 % (10 cm) / 58.0 % (15 cm),
  wall IoU 0.32, 48 % of occupied grid cells explained by plan walls/furniture/fixtures within 15 cm.
  Recorded path: 87 of 620 poses have the robot disc touching a plan element (81 walls — doorways
  where plan and map disagree by 0.1–0.3 m — and 15 furniture/fixtures), 0 in the void.
- **2D plan for the Jetson twin**: `floorplan.json/.svg` (upstairs) and `floorplan_floor1.json/.svg`
  in the map frame (rooms, wall polygons, doors with type/swing, windows, stair void + top step,
  railings, fixtures, furniture footprints), copied to `/data/twin/isaac/remodel/` on the Jetson.

## 4. Run the twin with ROS 2 (PC-local)

```powershell
. twin\isaac\ros_env.ps1; cd twin\isaac\scripts
python run_twin_ros.py --seconds 120 --people 3 --drive path --record E:\twin\runs\drive1
# path survey of the remodelled house (logs stuck spots, hops past them)
python run_twin_ros.py --people 0 --speed 0.3 --seconds 900 --drive path --unstick --drive-log E:\twin\eval\drive_remodel\drive.csv
# second terminal
. twin\isaac\ros_env.ps1; python ros_check.py --seconds 25 [--drive]
```
Same topics / types / frame ids / encodings / rates as the real D1000 driver and Roomba driver:
`/camera/left/image_rect_color` (bgr8), `/camera/left/image_rect_mono`, `/camera/right/image_rect_mono`,
`/camera/depth/image_rect_raw` (16UC1 mm, 0 outside 0.3-7 m), the three `camera_info`, `/camera/imu/data_raw`
(200 Hz, real noise level), `/tf_static`, `/wheel/odom` (30 Hz), plus `/ground_truth/odom` (map →
base_link) and `/clock`. `/cmd_vel` (Twist) drives the wheels with the driver's kinematics (0.5 s
timeout, 0.4 m/s wheel cap); `--drive path` follows the recorded RTAB-Map path with pure pursuit.
ROS 2 is Isaac Sim's bundled Jazzy (`isaacsim.ros2.core`) used through rclpy in the sim process;
`ROS_DOMAIN_ID=69`, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, Fast DDS.
Verified (`E:\twin\eval\ros_check*.json`): images 15.0 Hz, IMU 200.0 Hz, odom 30.0 Hz, clock 60 Hz in
sim time; real-time factor 0.43 with 3 walking people; `/cmd_vel` moved the robot.

## 5. Synthetic data (`gen_dataset.py`)

```powershell
. twin\isaac\env.ps1; cd twin\isaac\scripts
python gen_dataset.py --frames 1600 --look mix --cameras 1 --camera-model 0 --furniture 5 --out E:\twin\datasets\v1   # v1
# v2: remodelled house, 3 processes x 4 batched camera rigs, D1000 camera model applied
foreach ($w in 0,1,2) { Start-Process python -ArgumentList "gen_dataset.py --frames 4000 --cameras 4 --worker $w --seed 2 --out E:\twin\datasets\v2" }
python ..\eval\camera_model.py --in E:\twin\datasets\v2\images_raw --out E:\twin\datasets\v2ae\images   # re-apply the camera model (auto-exposure)
python ..\eval\dataset_stats.py E:\twin\datasets\v2
```
`--cameras N` renders N camera rigs per simulation step (one Replicator orchestrator step, N render
products), `--worker k` names files `w<k>_<index>` and offsets the seed so several processes can
write into one dataset (~11 GB VRAM and ~7 GB RAM each; three saturate the RTX 5090 at ~80 %).
`images/` holds the camera-model output, `images_raw/` the render.
Robot-eye camera (real K, 640x360, 0.13 m ± jitter, pitch ±4°, roll ±2°) on drivable free space near
the recorded path, 75 % of the time aimed near a walking person. Randomised: lights (on/off,
intensity 0.25-2.5x, 2700-6500 K, dome), look (scan / clean) with floor / wall / furniture colours,
procedural railings and baluster rows (0.75-1.05 m, 9-16 cm spacing, 1.5-4.5 cm posts, metal / wood /
white), ArchVis furniture placement, people (8 characters walking / idling).
Output: `images/` RGB, `labels/` YOLO (0 person, 1 railing; boxes ≥ 30 % visible, ≥ 6 px tall),
`depth/` uint16 mm (D1000 validity), `semantic/` class ids (`classes.json`), `meta/` per-frame JSON,
`data.yaml`, `stats.json`, `README.md`.

## 6. Evaluate (`eval/`, use `E:\twin\eval-env`)

```powershell
python eval_yolo.py --synthetic E:\twin\datasets\v1 --real E:\twin\real_frames\left --out E:\twin\eval
python sim_vs_real.py --real E:\twin\real_frames\left --sim E:\twin\datasets\v1\images --out E:\twin\eval
python contact_sheet.py E:\twin\datasets\v1 E:\twin\screens\dataset_v1_sheet.jpg
python train_yolo.py --data E:\twin\datasets\v1 --epochs 5        # optional proof of the loop
# v2: full fine-tunes (30 epochs), then a short hard-negative round on the auto-exposure images
python train_yolo.py --data E:\twin\datasets\v2 --model yolo11n.pt --name yolo11n_synth_v2 --epochs 30 --batch 64
python train_yolo.py --data E:\twin\datasets\v2 --model yolo11s.pt --name yolo11s_synth_v2 --epochs 30 --batch 64
python train_yolo.py --data E:\twin\datasets\v2ae --model E:\twin\runs\train\yolo11s_synth_v2\weights\best.pt `
    --name yolo11s_synth_v2ae_neg --negatives E:\twin\real_frames\left --epochs 10 --lr0 0.002 --warmup-epochs 0.5
python eval_yolo.py --model <onnx> --synthetic E:\twin\datasets\v2 --list E:\twin\datasets\v2\splits\val.txt `
    --real E:\twin\real_frames\left --real-list E:\twin\datasets\v2ae\splits\real_heldout.txt --tag <name>_ --out E:\twin\eval\v2
# the robot's detector profiles (person_model:= stock / finetuned / finetuned_n, each with its own classes) side by
# side on synthetic val and on two people-free robot recordings (fresh_map8, fresh_map7 frames in E:\twin\real_frames_fm7)
python compare_robot_detectors.py --out E:\twin\eval\hookup
```
`--negatives` adds the first 70 % (by time) of the real frames — the recording has no people, so
they are true negatives — as background images, and keeps the last 30 % as a held-out false-positive
test. Do not run two trainings at once on this PC: Windows dataloader workers each copy the
dataset state and two runs exhausted the 150 GB commit limit.
`yolo_onnx.py` reproduces the Jetson node exactly (letterbox 384x640 grey 114, BGR→RGB, class 0,
conf 0.45, OpenCV NMS 0.5).

## 7. Closed loop on the PC: Isaac Sim drives the robot's real stack in WSL

```powershell
# WSL Ubuntu-24.04, as root: own workspace from this checkout's main (+ the roomba-serial-ros2 submodule with
# roomba_driver_rs at the pinned commit; re-run after pulling) and TensorRT engines from the ONNX copies (once)
wsl -d Ubuntu-24.04 -u root -- bash /mnt/e/twin/roomba-jetson-stack/twin/isaac/wsl/setup_ws.sh   # BRANCH=... to pick another
wsl -d Ubuntu-24.04 -u root -- bash /mnt/e/twin/roomba-jetson-stack/twin/isaac/wsl/build_engines.sh
# start (WSL stack first, then the Isaac Sim window); again while it runs = new run from the dock
powershell -ExecutionPolicy Bypass -File twin\isaac\run_closed_loop.ps1 -Name my_run [-Record] [-People 2]
    [-Engine yolo11s_synth_384x640_fp16.engine] [-ViewGain 2] [-LookAround] [-StairKeepout] [-Assist] [-Headless]
# stop (graceful: RTAB-Map saves its database; results copied to E:\twin\runs\closedloop\<run>)
powershell -ExecutionPolicy Bypass -File twin\isaac\run_closed_loop.ps1 -Stop
# score a run against ground truth and the floor plan
E:\twin\isaacsim-env\Scripts\python.exe twin\isaac\scripts\analyze_closed_loop.py E:\twin\runs\closedloop\<run>
```
**Watch:** the Isaac Sim window (follow camera, docked "Robot camera (D1000 left)" and "Robot depth
(turbo)" panels); the robot's web app at http://localhost:18098 (password in
`E:\twin\isaac_watch_password.txt`); Foxglove at `ws://localhost:18766` with
`src/roomba_bringup/config/foxglove/explorer_debug.json` (the explorer's frontiers, candidate goals and
chosen goal, costmaps, keep-out mask, plan, footprint, depth) or `twin/isaac/foxglove/closed_loop.json` (map +
costmaps + nvblox, left camera with detections, colourised depth, status). Use
Lichtblick (open-source Foxglove Studio fork, no account) or Foxglove desktop (free sign-in).
`E:\twin\isaac_NOW.txt` always has a one-line status.

How it is wired:
- **WSL** (`wsl/closed_loop.sh`, `ROS_DOMAIN_ID=60`, `rmw_zenoh_cpp`, own Zenoh router on
  tcp/127.0.0.1:7447): `robot.launch.py` as on the robot via `wsl/isaac_robot.launch.py` (D1000 driver
  off, odometry cuvslam + EKF fusion, nvblox pipeline B, RTAB-Map mapping into a test database under
  `/root/isaacloop/runs/<run>`, Nav2, Rust nodes, explorer, person detector with a TensorRT engine
  built in WSL, web app on :18098 with the run's own HOME), foxglove_bridge on :18766, the run
  monitor (`wsl/cl_monitor.py`) and a CPU sampler for this run's processes (`wsl/cl_cpu.py`).
- **The real roomba_driver** (Rust `roomba_driver_rs` from the roomba-serial-ros2 submodule, robot config, only the port changed) talks the Roomba Open
  Interface to `wsl/isaac_roomba.py` on a pty, which turns drive commands into wheel speeds for Isaac
  and Isaac's state into OI sensor packets (encoders, bumpers, light bumpers, cliffs, wall, modes,
  Safe→Passive on a cliff). So `/wheel/odom`, `/roomba/sensors`, `battery_state`, `roomba/mode`, the
  bump reflex and the cmd_vel timeout are the driver's own. The stream is paced by sim time (one packet
  per Isaac step), otherwise wheel velocities alias at < 1x real time.
- **Isaac Sim** (`scripts/run_closed_loop.py`, bundled rclpy over Zenoh): `/clock`, the D1000 topics
  (left colour + mono, right mono, depth 16UC1 mm 0.3–7 m registered to the left camera, three
  camera_info, IMU 200 Hz, `/tf_static` as the real driver), ground truth `/ground_truth/odom`, people
  positions, status. Only the cameras the stack needs are rendered, every 4th step (15 Hz sim time).
  Lights x2.5 (`--light-scale`, the robot-camera setting: raw renders are ~3x darker than the real auto-exposed
  frames; the dataset generator is unchanged). The GUI viewport and the recording get another x2 exposure
  (`-ViewGain`, RTX film ISO); RTX tonemapping is global (a camera's exposure attribute and per-render-product
  overrides were ignored in a test), so the robot camera images are mapped back with a per-channel LUT calibrated
  at start-up (mean error 0.7 of 255 levels): what the stack sees is unchanged. The walking people's body colliders
  are switched off (kinematic bodies shoved the robot 0.6-1 m, a kidnapping the stack can't know about); the bumper
  model still feels their legs.
  The Kit GUI needs the RTSS (MSI Afterburner / RivaTuner) Vulkan layer disabled for this process
  only (`DISABLE_RTSS_LAYER=1` and the two Steam layers): it crashed the NVIDIA Vulkan driver (integer
  divide by zero in nvoglv64.dll) while creating the swapchain.
- **Sim-side workarounds from before the navigation fixes**, all off by default now (logged in
  `monitor/events.jsonl` when used): `-LookAround` turns 360° with Nav2's Spin before exploring;
  `-StairKeepout` adds the stairwell as a keep-out zone through the zone editor, as the web app does;
  `-Assist` reverses a robot that has not moved for 40 s of sim time by 0.3 m through Isaac's teleop override
  and counts it as an intervention. `-WallTimeoutScale k` multiplies the explorer's timeouts; it timed them on
  the wall clock until main cd218c6 (k ≈ 1 / real-time factor was needed), now on the node clock (k = 1).

Outputs per run: `monitor/poses.csv` (2 Hz: ground truth, RTAB-Map map pose, EKF odom, cuVSLAM,
wheel odometry, cmd_vel), `events.jsonl`, `detections.csv`, `people.csv`, the last RTAB-Map grid and
global costmap, `logs/` (launch, driver bridge, OI serial log, CPU), `analysis/` (summary.json,
trajectory.png, map_vs_plan.png).

## Results (2026-11-15, see `E:\twin\eval`, `E:\twin\screens`)

| | |
|---|---|
| dataset v1 | 1600 frames in 607 s; 3136 person boxes in 1258 frames, 591 railing boxes in 455 frames; person distance median 2.4 m (10-90 %: 1.0-5.2 m) |
| baseline yolo11n (Jetson ONNX), all 1600 synthetic | precision 0.91, recall 0.22, AP50 0.32 at conf 0.45; recall 1 % for boxes < 40 px, 37 % for > 160 px; 33 % at < 1.5 m, 7 % beyond 5 m |
| same on the 160-frame val split | P 0.90, R 0.22, AP50 0.32 |
| fine-tuned 8 epochs on 1440 synthetic frames (2 classes) | val P 0.95, R 0.40, AP50 0.62 for person (ultralytics val: person mAP50 0.59, railing 0.22); exported 1x3x384x640 ONNX; not deployed |
| real fresh_map8 frames (1039, every 2 s) | no people were in this recording; baseline: 1 detection, a false positive on the stair balusters; fine-tuned: 0 |
| sim vs real (median) | brightness 50 vs 61, contrast 33 vs 24, Laplacian variance 175 vs 12 (real is much softer), crushed blacks 7 % vs 0 %, colour B:G:R 32:47:61 vs 37:68:58 (real is green); `eval/camera_model.py` brings them to 61 / 25 / 21 / 0 % / 38:62:65 |

### Remodelled house, dataset v2 and training (2026-11-15, `E:\twin\eval\v2`, `E:\twin\runs\train`)

| | |
|---|---|
| dataset v2 | 11,958 frames (3 workers x 4 rigs, ~25 min, 8.7 frames/s, 3.3x v1); 26,344 person boxes in 78 % of frames, 37,492 railing boxes; person distance median 2.3 m; 4.7 GB (+ `v2ae` re-processed images 3.3 GB) |
| GPU use | generation: 81 % median utilisation, 31.5 GB VRAM, ~265 W; training: 72 % median (yolo11s ~79 %, yolo11n alone 45-88 %: the dataloader's mosaic augmentation is CPU-bound), up to 31 GB VRAM, ~260 W (`E:\twin\eval\gpu\*.csv`) |
| physics drive of the recorded path | 106 m at 0.3 m/s with pure pursuit: 76 % driven, 24 stuck spots (hopped past): east bedroom door, north closet/bath doors, primary bath door — places where plan and map disagree by 0.1-0.3 m — and 1 template bed; none at railings or the stair void (`E:\twin\eval\drive_remodel\summary.json`, `remodel_drive_survey.png`) |
| sim vs real (median, 1000 frames) | v2 with the first camera model: brightness 32 vs 62 (the remodel renders at ~20). Camera model with auto-exposure (`v2ae`): brightness 64 vs 62, contrast 28 vs 24, saturation 108 vs 111, noise σ 0.68 vs 0.55, Laplacian variance 13.0 vs 11.5, B:G:R 37:68:64 vs 37:68:59 |

Person detection at the deployed settings (conf 0.45, IoU 0.5, boxes ≥ 10 px; the val splits were
never trained on; "real FP" = frames with a person detection among the 312 held-out real frames of
fresh_map8, which has no people in it):

| model (all exported 1x3x384x640) | v2 val P / R / AP50 | v2-AE val P / R / AP50 | v1 val AP50 | real FP frames |
|---|---|---|---|---|
| baseline yolo11n (Jetson ONNX, COCO) | 0.93 / 0.30 / 0.53 | 0.93 / 0.28 / 0.52 | 0.32 | 1 |
| yolo11n fine-tuned on v1 (8 epochs) | 0.91 / 0.13 / 0.35 | 0.86 / 0.02 / 0.14 | 0.62 | 0 |
| yolo11n fine-tuned on v2 (30 epochs, 20 min) | 0.96 / 0.74 / 0.86 | 0.96 / 0.66 / 0.81 | 0.28 | 19 |
| yolo11s fine-tuned on v2 (30 epochs, 27 min) | 0.95 / 0.77 / 0.88 | 0.94 / 0.70 / 0.83 | 0.29 | 11 |
| yolo11n v2 → v2-AE + 727 real negatives (+10 epochs) | 0.94 / 0.73 / 0.85 | 0.95 / 0.73 / 0.85 | 0.29 | **0** |
| **yolo11s v2 → v2-AE + 727 real negatives (+10 epochs)** | 0.96 / 0.74 / 0.87 | 0.96 / 0.75 / 0.86 | 0.33 | **0** |

- Recall by box height for the best model (v2-AE val): 0 % < 40 px, 22 % 40-80, 59 % 80-160, 88 % > 160
  (baseline 0 / 1 / 16 / 36 %); by distance 80 % < 1.5 m, 77 % 1.5-3 m, 69 % 3-5 m, 68 % > 5 m.
- The v2-only models fire on real balusters, newel posts and dark door edges (vertical dark bars look
  like legs at 0.13 m); a short round with real people-free frames as background images removes all
  of those on the held-out frames without losing synthetic recall. Precision on v1 (scan-mesh cave
  look, no camera model) drops for every v2 model: that look is no longer representative.
- yolo11s vs yolo11n: +1-2 points recall/AP50 at equal precision.
- **Jetson budget** (`jetson/trt_bench.sh`, TensorRT 10.16 FP16 engines built in `/data/twin/isaac/trt`,
  idle Orin Nano Super): yolo11n 384x640 = 3.9 GFLOPs, **3.48 ms** GPU compute (3.79 ms latency);
  yolo11s = 12.9 GFLOPs, **5.35 ms** (5.68 ms). In production yolo11n takes ~9 ms under load, so
  yolo11s would be ~14 ms — well inside the 66 ms frame period at 15 Hz. Nothing was deployed.

### Closed loop (2026-11-15/16, `E:\twin\runs\closedloop\<run>`, scored by `analyze_closed_loop.py`)

All runs: upstairs floor-plan house, start at the dock (map origin), 2 walking people, yolo11n engine.
"Coverage" = RTAB-Map free cells on the plan's floor (99 m² excluding the stair void and furniture).
Runs 1-7 used main 4650dca (before the navigation fixes); runs 8-9 main 40bc7ff (explorer look-around,
reachability, retries and finish-at-home, keep-out marks off the robot, drop-off detection in
floor_obstacles, nav2_guard, Rust driver `roomba_driver_rs`), **unaided**: no keep-out zone, no assists,
no sim-side look-around. Real-time factor 0.23-0.33 with a free GPU (runs 1-2, 8-9; 0.14-0.2 while
other jobs loaded the CPU), 0.05-0.15 in runs 3-7 while an Isaac Lab training used ~95 % of the GPU.

| run | stack / sim side | explored (sim s) | goals ok / failed | driven (GT) | EKF / map distance | ATE map pose | coverage | what happened |
|---|---|---|---|---|---|---|---|---|
| 1 `cl_test2` | 4650dca; OI stream paced by wall clock | 140 | 4 / 1 | 11.4 m | 59 % | 1.45 m | 11 % | frontier goal "timed out" after 11 s of sim (60 s wall); "no reachable unmapped areas" at 140 s; wedged nose-to-door-frame on the way home until the 300 s (wall) return timeout |
| 2 `cl_run2` | same | 102 | 4 / 0 | 10.5 m | 64 % (wheels 98 %, cuVSLAM ~100 %) | 1.91 m | 12 % | map smeared by the short EKF, ghost obstacles around the robot, wedged at a bed/shelf (spin and back-up "Collision Ahead"); explorer still said "Paused: visual tracking is recovering" |
| 3 `cl_run3` | OI stream paced by sim time | 207 | 8 / 2 | 25.5 m | 95 % | **0.17 m** until 328 s, 0.64 m overall | 25 % | "finished" at 20 % coverage (frontier behind the start never faced), found the west hallway only after driving home; a simulated person then shoved the robot 1 m / 135° → wedged at the railing (3219 "collision ahead") |
| 4 `cl_run4` | + 360° look-around | 31 | 0 / 0 | 0.2 m | - | 0.10 m | 9 % | drove north into the balusters 3 s after the first goal (a frontier inside the stair void, as in run 5), bumped, wedged (1021 "collision ahead", 18 behaviours aborted) |
| 5 `cl_run5` | + assist | 21 | 0 / 1 | 1.1 m | - | 0.52 m | 9 % | same railing crash; a person shoved it at 103 s |
| 6, 7 | + stairwell keep-out | - | - | - | - | - | - | 6: shoved by a person before exploring; 7: stopped at 30 s sim (GPU busy) |
| 8 `cl_run8` | **40bc7ff, unaided**; pushes by people undone each step | 300 | 3 / 13 | 19.2 m | 93 % | 1.00 m | 36 % | people kept walking into the robot; undoing 437 pushes jolted the IMU and camera, 19 VIO recoveries, map pose 1.8 m off, wedged facing a wall corner: stopped (sim problem, fixed for run 9) |
| **9 `cl_run9`** | **40bc7ff, unaided**; people's body colliders off | 973 | 6 / 18 | 76.0 m | 95 % | **0.38 m** (final 0.10 m, max 2.2 m) | **44 %** (56 % of the 2D twin's grid area) | looked around and explored the west hallway, the south-west bedroom, the corridor and the east bedroom; repeatedly chose frontiers inside the stair void; never entered it (no cliff event) but ended at the top of the stairs, enclosed by its own bump keep-out marks: "Could not reach the start point", trapped ~290 s until stopped |

Run 9 in detail (`E:\twin\runs\closedloop\cl_run9\analysis`, `closedloop_run9_*.png`):
- **Stairwell**: the robot never entered the void and no cliff sensor fired: drop-off detection works at
  the stair head. But RTAB-Map still maps the void as floor (14 m² of "free" outside the plan floor), the
  explorer kept choosing frontiers in it (goal (3.71, 1.6), 0.65 m inside the void, then "The map opened
  up around an area that failed earlier (3.7, 1.6): trying it again"), the robot's centre came within
  0.19 m of the railing line (body touching) and 11 of its 16 bumper events were within 0.3 m of the
  railing; 392 s within 0.3 m of it.
- **Stuck**: 3 episodes of no motion for ≥ 30 s (36 s, 30 s, ~290 s). The last: at the stair head the
  bump marks (each a 50 min keep-out patch; 10 of the 16 bumps had a person within 0.6 m) formed a
  closed ring around the robot. The marks keep off the robot's own cells, but the planner found no way
  out ("GridBased plugin failed to plan"), the return home failed and the explorer kept sending goals
  (`closedloop_run9_trap_costmap.png`, `closedloop_run9_stairhead_robotcam.png`).
- **Tracking**: 44 VIO recoveries in 973 s (run 3: 3 in 207 s); 23 of 45 recovery starts came during an
  in-place turn, and the robot now spends 32 % of the time turning in place (look-around, facing
  frontiers; run 3: 8 %); 15 with a person within 1.5 m. cuVSLAM drift 20 % of distance between resets
  (run 3: 4.6 %). The map pose still ended 0.10 m from ground truth (EKF odometry 1.1 m).
- **Contacts**: 29 bumper contacts in Isaac, 16 driver bump events, 0 cliffs, 0 pushes by people.
- **Finish**: not finished. The explorer ran its finish check once ("Exploration finished: no
  reachable unmapped areas left"), could not reach the start, resumed exploring from where it was and
  stayed trapped.
- Person detector (frame level, person within ±40° and 0.3-7 m with a line of sight through the plan
  walls): recall 0.71 / precision 0.73 (0.87 below 1.5 m, 0.80 at 1.5-3 m, 0.55 at 3-7 m); run 8 0.83 / 0.86;
  run 3 0.34 / 0.65 (few encounters).
- Load: the stack ~0.9 CPU core in WSL; Isaac ~50-100 ms per step (CPU-bound Kit update); GPU 27-30 %.
- Its explorer timeouts were 5x too long (goal 300 s of sim instead of 60 s): run 9 still used the
  `-WallTimeoutScale 5` default from the wall-clock era, while the explorer now times goals on the node
  clock. The default is 1 now.

Compared with the 2D twin on the same floor plan (`/root/twin2d_data/runs/plan_after6/explore_from_dock_s1`, `_s2` in WSL):
83-90 % coverage in 934-1105 s, 126-143 m, 29-34 of 50-59 goals, 22-26 bumps; on the RTAB-Map grid
world (`after7`): 64-68 % in 335-347 s, no bumps. Isaac adds what the 2D twin does not model: seeing
through balusters, the depth blind zone below 0.3 m, VIO failures during turns, people's legs on the
bumper and walking past at the stair head.

**Stack findings** (evidence in the run folders' `logs/robot_launch.log`, `monitor/events.jsonl`,
`analysis/`; screenshots `E:\twin\screens\closedloop_*.png`). Status after main 40bc7ff (run 9):
1. *The stairwell reads as open floor.* Runs 4-5: the depth camera at 0.13 m looks through the
   balusters, the void below never shows a floor, RTAB-Map and the costmap ray-traced it as free, the
   explorer picked a frontier inside it and the robot hit the balusters within seconds.
   **Partly fixed**: drop-off detection keeps the robot out of the void and off the stair head (run 9,
   no cliff event), but the void is still mapped as floor, frontiers inside it are still chosen and
   retried, and the robot rubs along the railing. Next: never take frontiers inside drop-off cells, and
   mark them in the RTAB-Map grid too.
2. *A wedged robot can't move.* Runs 1-5: once the footprint touches lethal cost RPP refuses every
   command ("detected collision ahead!") and Spin/BackUp abort with "Collision Ahead". **Changed**: no
   such deadlock against a wall in run 9 (3 behaviour aborts), but a new trap: a ring of bump keep-out
   marks around the robot (many from people's legs) leaves no path out. Next: expire or drop marks that
   enclose the robot, don't mark bumps when a person is next to the robot, or allow a short reverse.
3. *The final heading was ignored, so the explorer never looked around* (run 3: "finished" at 20 %).
   **Fixed**: it looks around and faces frontiers; coverage 44 % vs 25 % before. Side effect: in-place
   turns are where half of the VIO recoveries start (slower turns or pauses between steps may help).
4. *Stale status* "Paused: visual tracking is recovering" (run 2). **Fixed**: the message returns to the
   goal when tracking is back.
5. *Wall-clock timeouts* (run 1). **Fixed for the explorer** (node clock since cd218c6).
6. Minor: with velocity-proportional wheel covariances the EKF favours low speed samples; noisy or
   aliased wheel velocities bias it short (how the sim-time pacing bug showed up as -35 % distance).
   cuVSLAM warns when two frames are > 80 ms apart (≈ 1 % of frames in runs 8-9; cause not found).

**Sim-side fixes** (in `run_closed_loop.py` / `wsl/`): OI stream paced by sim time (fixes the EKF);
people's body colliders off (they walk through the robot instead of kidnapping it; the bumper model
still feels legs); lights x2.5 (robot-camera setting) and x2 view exposure for the GUI and recordings with
the robot cameras mapped back. Workarounds from before the fixes are opt-in flags now (`-LookAround`,
`-StairKeepout`, `-Assist`, `-WallTimeoutScale`).

**Videos** (`E:\twin\videos\web`, H.264 yuv420p +faststart + VP9 WebM + poster JPG; rendered from Isaac
per sim step, so playback is smooth although the sim runs slower than real time; run 9, brighter view
exposure, robot camera as the stack sees it):

| file | length | MP4 / WebM | shows |
|---|---|---|---|
| `autonomous_hero_loop` | 12 s | 6.9 / 4.5 MB | run 9: the stack exploring on its own past a walking person at the railing, follow camera + robot camera inset, cross-faded loop |
| `autonomous_highlights` | 76 s | 33.6 / 30.4 MB | run 9, first 7 minutes: look-arounds, hallways, bedrooms, people; long stretches at 2-4x (labelled) |
| `depth_heatmap_autonomous` | 25 s | 9.0 / 5.4 MB | run 9: follow camera + robot camera + depth heat map at the stairwell (turbo, black outside 0.3-7 m, legend) |

## Known limits / next steps
- The scan mesh is noisy (IR depth from 0.13 m, Poisson surfaces with holes and shards); it is kept as
  `--look scan|clean`. The default is the floor-plan house; a dedicated capture (video walk-through,
  NuRec / 3D Gaussian splat) could add real textures and clutter later.
- Floor-plan house: the floor plans are approximate (doorways 0.1-0.3 m off the map, see the drive
  survey); furniture outside the mapped blobs is generic; door positions and wall thickness come from
  the drawing; heights are assumptions (2.44 m / 2.65 m / 2.90 m floor to floor).
- Dataset v2 was rendered before the stairwell correction (placeholder below the well); render v3 with
  the downstairs and the auto-exposure camera model built in (`gen_dataset.py` now applies it).
- Real-time factor 0.4 with two cameras at 15 Hz, depth and IMU at 200 Hz (ROS twin, headless); the
  closed loop with the GUI window, three robot cameras and walking people reaches 0.23-0.33 on a free
  GPU (the Kit update is CPU-bound, ~40-65 ms per step), 0.14-0.2 while other jobs load the CPU and
  0.05-0.15 while a training shares the GPU.
- Closed loop on the PC: the unaided run (run 9) explored 44 % of the plan in 16 min of sim before it was
  trapped at the stair head; re-run after the remaining stairwell / bump-mark findings are fixed. The
  stairwell keep-out via the zone editor (`-StairKeepout`) is untested in a run.
- Closed loop with the Jetson needs the network phase (Ethernet, firewall rule for DDS on the PC,
  same RMW/domain, or a Zenoh bridge).
