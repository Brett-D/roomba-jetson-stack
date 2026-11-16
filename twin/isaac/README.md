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
```
`--negatives` adds the first 70 % (by time) of the real frames — the recording has no people, so
they are true negatives — as background images, and keeps the last 30 % as a held-out false-positive
test. Do not run two trainings at once on this PC: Windows dataloader workers each copy the
dataset state and two runs exhausted the 150 GB commit limit.
`yolo_onnx.py` reproduces the Jetson node exactly (letterbox 384x640 grey 114, BGR→RGB, class 0,
conf 0.45, OpenCV NMS 0.5).

## 7. Closed loop on the PC: Isaac Sim drives the robot's real stack in WSL

```powershell
# one-time (WSL Ubuntu-24.04, as root): own workspace from main + TensorRT engines from the ONNX copies
wsl -d Ubuntu-24.04 -u root -- bash /mnt/e/twin/roomba-jetson-stack/twin/isaac/wsl/setup_ws.sh
wsl -d Ubuntu-24.04 -u root -- bash /mnt/e/twin/roomba-jetson-stack/twin/isaac/wsl/build_engines.sh
# start (WSL stack first, then the Isaac Sim window); again while it runs = new run from the dock
powershell -ExecutionPolicy Bypass -File twin\isaac\run_closed_loop.ps1 -Name my_run [-Record] [-People 2]
    [-Engine yolo11s_synth_384x640_fp16.engine] [-NoStairKeepout] [-NoAssist] [-NoLookAround] [-Headless]
# stop (graceful: RTAB-Map saves its database; results copied to E:\twin\runs\closedloop\<run>)
powershell -ExecutionPolicy Bypass -File twin\isaac\run_closed_loop.ps1 -Stop
# score a run against ground truth and the floor plan
E:\twin\isaacsim-env\Scripts\python.exe twin\isaac\scripts\analyze_closed_loop.py E:\twin\runs\closedloop\<run>
```
**Watch:** the Isaac Sim window (follow camera, docked "Robot camera (D1000 left)" and "Robot depth
(turbo)" panels); the robot's web app at http://localhost:18098 (password in
`E:\twin\isaac_watch_password.txt`); Foxglove at `ws://localhost:18766` with
`twin/isaac/foxglove/closed_loop.json` (map + costmaps + nvblox, `/plan`, `/explorer/markers` once the
explorer publishes them, obstacle points, left camera with detections, colourised depth, status). Use
Lichtblick (open-source Foxglove Studio fork, no account) or Foxglove desktop (free sign-in).
`E:\twin\isaac_NOW.txt` always has a one-line status.

How it is wired:
- **WSL** (`wsl/closed_loop.sh`, `ROS_DOMAIN_ID=60`, `rmw_zenoh_cpp`, own Zenoh router on
  tcp/127.0.0.1:7447): `robot.launch.py` as on the robot via `wsl/isaac_robot.launch.py` (D1000 driver
  off, odometry cuvslam + EKF fusion, nvblox pipeline B, RTAB-Map mapping into a test database under
  `/root/isaacloop/runs/<run>`, Nav2, Rust nodes, explorer, person detector with a TensorRT engine
  built in WSL, web app on :18098 with the run's own HOME), foxglove_bridge on :18766, the run
  monitor (`wsl/cl_monitor.py`) and a CPU sampler for this run's processes (`wsl/cl_cpu.py`).
- **The real roomba_driver** (Rust, robot config, only the port changed) talks the Roomba Open
  Interface to `wsl/isaac_roomba.py` on a pty, which turns drive commands into wheel speeds for Isaac
  and Isaac's state into OI sensor packets (encoders, bumpers, light bumpers, cliffs, wall, modes,
  Safe→Passive on a cliff). So `/wheel/odom`, `/roomba/sensors`, `battery_state`, `roomba/mode`, the
  bump reflex and the cmd_vel timeout are the driver's own. The stream is paced by sim time (one packet
  per Isaac step), otherwise wheel velocities alias at < 1x real time.
- **Isaac Sim** (`scripts/run_closed_loop.py`, bundled rclpy over Zenoh): `/clock`, the D1000 topics
  (left colour + mono, right mono, depth 16UC1 mm 0.3–7 m registered to the left camera, three
  camera_info, IMU 200 Hz, `/tf_static` as the real driver), ground truth `/ground_truth/odom`, people
  positions, status. Only the cameras the stack needs are rendered, every 4th step (15 Hz sim time).
  Lights are 2.5x brighter for the GUI/videos (`--light-scale`; the dataset generator is unchanged).
  The Kit GUI needs the RTSS (MSI Afterburner / RivaTuner) Vulkan layer disabled for this process
  only (`DISABLE_RTSS_LAYER=1` and the two Steam layers): it crashed the NVIDIA Vulkan driver (integer
  divide by zero in nvoglv64.dll) while creating the swapchain.
- **Run options** (all logged in `monitor/events.jsonl`): `--look-around` (default) turns 360° with
  Nav2's Spin before exploring; the stairwell keep-out zone (default) is added through the zone editor
  exactly as the web app does; `--assist` (default) reverses a robot that has not moved for 40 s of
  sim time by 0.3 m through Isaac's teleop override and counts it as an intervention. `-NoStairKeepout
  -NoAssist -NoLookAround` gives the stack's unaided behaviour. The explorer's wall-clock timeouts are
  scaled by `-WallTimeoutScale` (default 5 ≈ 1 / real-time factor).

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

### Closed loop (2026-11-15, `E:\twin\runs\closedloop\<run>`, scored by `analyze_closed_loop.py`)

All runs: upstairs floor-plan house, start at the dock (map origin), 2 walking people, yolo11n engine.
Real-time factor 0.23-0.28 while the GPU was free; 0.05-0.15 from run 3 on, while an Isaac Lab training
used ~95 % of the GPU (so no full exploration was possible; to be re-run after the navigation fixes).
"Coverage" = RTAB-Map free cells on the plan's floor (99 m² excluding the stair void and furniture).

| run | sim-side state | explored (sim s) | goals ok / failed | driven (GT) | EKF / map distance | ATE map pose | coverage | what happened |
|---|---|---|---|---|---|---|---|---|
| 1 `cl_test2` | OI stream paced by wall clock | 140 | 4 / 1 | 11.4 m | 59 % | 1.45 m | 11 % | frontier goal "timed out" after 11 s of sim (60 s wall); "no reachable unmapped areas" at 140 s; wedged nose-to-door-frame on the way home until the 300 s (wall) return timeout |
| 2 `cl_run2` | same | 102 | 4 / 0 | 10.5 m | 64 % (wheels 98 %, cuVSLAM ~100 %) | 1.91 m | 12 % | map smeared by the short EKF, ghost obstacles around the robot, wedged at a bed/shelf (spin and back-up "Collision Ahead"); explorer still said "Paused: visual tracking is recovering" |
| 3 `cl_run3` | OI stream paced by sim time | 207 | 8 / 2 | 25.5 m | 95 % | **0.17 m** until 328 s, 0.64 m overall | 25 % | "finished" at 20 % coverage (frontier behind the start never faced), found the west hallway only after driving home; a simulated person then shoved the robot 1 m / 135° → wedged at the railing (3219 "collision ahead") |
| 4 `cl_run4` | + 360° look-around | 31 | 0 / 0 | 0.2 m | - | 0.10 m | 9 % | drove north into the balusters 3 s after the first goal (a frontier inside the stair void, as in run 5), bumped, wedged (1021 "collision ahead", 18 behaviours aborted) |
| 5 `cl_run5` | + assist | 21 | 0 / 1 | 1.1 m | - | 0.52 m | 9 % | same railing crash; a person shoved it at 103 s |
| 6, 7 | + stairwell keep-out | - | - | - | - | - | - | 6: shoved by a person before exploring (and the keep-out was sent to the wrong topic; fixed and checked against the zone node on its own since, not yet in a run); 7: stopped at 30 s sim (GPU busy) |

Other measurements (run 3 unless noted): cuVSLAM drift 4.6 % of distance between resets (7 odometry resets,
3 watchdog recoveries); wheel odometry ATE 0.15 m in run 2; bumper contacts 3 (1 driver bump event), no
cliffs; person detector (yolo11n, frame level, person within ±40° and 0.3-7 m with a clear line of sight
through the plan walls) recall 0.34 / precision 0.65 — 1.00 below 1.5 m, 0.19 at 1.5-3 m, 0.31 at
3-7 m; the stack used ~1 CPU core in WSL (bt_navigator, controller, RTAB-Map, nvblox, person detector
5-10 % of a core each), Isaac ~2-3 cores, GPU 80-94 % (shared). The 2D twin's `explore_from_dock` on the
grid world reached 64-68 % coverage in 250-370 s with 14-20 goals and no bumps (run 3 here: 30 % of the same area); it does not model
seeing through balusters, the 0.3 m depth blind zone, VIO failure or people. The optional run with the
retrained yolo11s (person + railing) engine was not done (`build_engines.sh` builds it; the node needs a
2-class `classes` override).

**Stack findings** (evidence in the run folders' `logs/robot_launch.log`, `monitor/events.jsonl`,
`analysis/`; screenshots `E:\twin\screens\closedloop_*.png`):
1. *The stairwell reads as open floor.* The depth camera at 0.13 m looks through the balusters; the
   void below never shows a floor, so RTAB-Map and the costmap ray-trace it as free and the thin
   balusters are only partly marked. The explorer picked a frontier inside the void and Nav2 planned
   through a gap in the railing; the robot hit the balusters within seconds (runs 4 and 5, like the
   real crash; `closedloop_railing_run5.png`). Fix: treat "floor expected but not seen" as a drop-off
   (negative obstacle) in floor_obstacles, or at least keep a keep-out zone around the stairwell.
2. *A wedged robot can't move.* Once the footprint touches lethal cost (after a bump, at a door frame,
   or because of a smeared map), RPP refuses every command ("detected collision ahead!" every
   cycle) and the Spin/BackUp behaviours abort with "Collision Ahead"; only the driver's bump reflex
   moves it (~1 cm). Runs 1, 2, 3, 4, 5. Fix: a recovery that reverses a short way without the
   collision check when the rear is clear, or a nav_coach unstick manoeuvre.
3. *The final heading is ignored, so the explorer never looks around.* It sends frontier goals that
   face the unknown, but `yaw_goal_tolerance: 6.28` accepts any heading and the goal is then marked
   visited; a frontier right behind the start was never seen and exploration "finished" at 20 %
   (run 3). Fix: turn to the requested heading for explorer goals (own goal checker or a spin on
   arrival), or only mark a frontier visited once it was in the camera's field of view.
4. *Stale status.* "Paused: visual tracking is recovering" stays after tracking recovers while Nav2 is
   driving again (run 2, from 46 s to the end); the web app shows the wrong state.
5. *Wall-clock timeouts.* The explorer (goal 60 s, response 10 s, return 300 s, finish confirm 20 s),
   nav_coach, vslam_watchdog and the driver's 0.5 s cmd_vel timeout use the monotonic clock, so they
   fire early when the sim runs slower than real time (run 1). `isaac_robot.launch.py` scales the
   explorer's timeouts as a workaround; using the node clock would fix it.
6. Minor: with velocity-proportional wheel covariances the EKF favours low speed samples; noisy or
   aliased wheel velocities bias it short (how the sim-time pacing bug showed up as -35 % distance).
   cuVSLAM warns when two frames are > 80 ms apart (15 Hz camera with a dropped frame).

**Workarounds and sim-side fixes** (flags of `run_closed_loop.ps1`): look-around before exploring
(`-NoLookAround` to disable), stairwell keep-out zone (`-NoStairKeepout`), logged assists for a robot
stuck 40 s (`-NoAssist`), explorer wall-clock timeouts x5 (`-WallTimeoutScale`), OI stream paced by sim
time (fixes the EKF), pushes by simulated people undone (they walk through the robot instead of
kidnapping it), lights x2.5 for viewing.

**Videos** (`E:\twin\videos\web`, H.264 yuv420p +faststart + VP9 WebM + poster JPG; rendered from Isaac
per sim step, so playback is smooth although the sim runs slower than real time):

| file | length | MP4 / WebM | shows |
|---|---|---|---|
| `autonomous_hero_loop` | 12 s | 5.5 / 3.1 MB | run 3: the stack exploring on its own, follow camera + robot camera inset, cross-faded loop |
| `autonomous_highlights` | 51 s | 24.8 / 9.9 MB | run 3 highlights, long stretches at 2-4x (labelled) |
| `path_replay_hero_loop` | 12 s | 6.3 / 3.1 MB | recorded path replayed through the house with people (not autonomous) |
| `path_replay_highlights` | 72 s | 34.5 / 13.7 MB | same, highlights |
| `depth_heatmap_autonomous` | 25 s | 5.6 / 7.0 MB | follow camera + robot camera + depth heat map (turbo, black outside 0.3-7 m, legend) |

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
  closed loop with the GUI window, three robot cameras and walking people reaches 0.23-0.28 on a free
  GPU (the Kit update is CPU-bound, ~50-65 ms per step) and 0.05-0.15 while a training shares the GPU.
- Closed loop on the PC: a full unaided exploration still has to be run once the navigation findings
  above are fixed and the GPU is free; the stairwell keep-out via the zone editor is untested in a run.
- Closed loop with the Jetson needs the network phase (Ethernet, firewall rule for DDS on the PC,
  same RMW/domain, or a Zenoh bridge).
