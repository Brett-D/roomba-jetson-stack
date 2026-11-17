"""Visual-inertial SLAM: MYNT D1000 stereo + IMU -> visual-inertial odometry -> RTAB-Map mapping.

Odometry (odom -> base frame: camera_link when hand-held, base_link on the robot) comes from
Basalt (default; stereo-inertial, CPU) or NVIDIA cuVSLAM (GPU), filtered by vslam_watchdog.
On the robot, fusion:=ekf hands odom -> base_link to a robot_localization EKF instead (wheels +
gyro + the VIO poses the watchdog accepted); the watchdog then only supervises the VIO.
RTAB-Map consumes that odometry plus RGB-D for loop closure, the persistent map (map -> odom)
and the 2D occupancy grid used later by Nav2. On the robot, use robot.launch.py, which includes
this file with platform:=robot.

Usage:
  ros2 launch roomba_bringup vslam.launch.py                   # auto: reuse saved map if one exists
  ros2 launch roomba_bringup vslam.launch.py mode:=mapping     # start a new map (old one is replaced)
  ros2 launch roomba_bringup vslam.launch.py mode:=localization
  ros2 launch roomba_bringup vslam.launch.py odometry:=cuvslam # previous odometry backend
"""
import os

import yaml
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            SetLaunchConfiguration)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node, SetParameter
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# Basalt calibration (rectified pinhole intrinsics, T_imu_cam, IMU noise). The package ships the
# one generated from this camera's factory calibration; a file here overrides it.
BASALT_CALIB_FILE = os.path.expanduser('~/.config/roomba/basalt_calib.json')

# cuVSLAM's IMU noise model, replaced by measured values when ~/.config/roomba/imu_noise.yaml
# exists (written by `ros2 run roomba_perception imu_calibrate`). cuVSLAM's own defaults (tuned
# for a RealSense IMU) trust this IMU far too much: replaying a recorded walk, they let
# tracking explode (544 jumps > 0.5 m and 260 km of drift in 4 min); 10x the noise gave
# 16 jumps and 31 gyro disagreements instead of 157.
IMU_NOISE_DEFAULTS = {
    'gyro_noise_density': 0.00244,
    'gyro_random_walk': 0.00019393,
    'accel_noise_density': 0.01862,
    'accel_random_walk': 0.03,
    'calibration_frequency': 200.0,
}
IMU_NOISE_FILE = os.path.expanduser('~/.config/roomba/imu_noise.yaml')


def _imu_noise():
    params = dict(IMU_NOISE_DEFAULTS)
    if os.path.isfile(IMU_NOISE_FILE):
        with open(IMU_NOISE_FILE) as f:
            for line in f:
                key, sep, value = line.partition(':')
                if sep and key.strip() in params:
                    params[key.strip()] = float(value)
    return params


def _resolve_mode(context):
    """Sets the 'localization' launch configuration from 'mode' (auto/mapping/localization)."""
    mode = LaunchConfiguration('mode').perform(context)
    db = os.path.expanduser(LaunchConfiguration('database_path').perform(context))
    if mode == 'auto':
        loc = os.path.isfile(db) and os.path.getsize(db) > 0
    elif mode in ('mapping', 'localization'):
        loc = mode == 'localization'
    else:
        raise RuntimeError(f"mode must be auto, mapping or localization (got '{mode}')")
    if loc and not os.path.isfile(db):
        raise RuntimeError(f'mode:=localization but no map database at {db}')
    desc = f'localization in saved map {db}' if loc else f'mapping a NEW map (replaces {db})'
    return [SetLaunchConfiguration('localization', 'true' if loc else 'false'),
            LogInfo(msg=f'[vslam] mode={mode}: {desc}')]


def _watchdog(context):
    """Motion filter, with robot limits and wheel odometry on the Roomba. With fusion:=ekf it
    only supervises the VIO, and robot_localization's EKF publishes odom -> base_link."""
    platform = LaunchConfiguration('platform').perform(context)
    if platform not in ('handheld', 'robot'):
        raise RuntimeError(f"platform must be handheld or robot (got '{platform}')")
    fusion = LaunchConfiguration('fusion').perform(context)
    if fusion not in ('watchdog', 'ekf'):
        raise RuntimeError(f"fusion must be watchdog or ekf (got '{fusion}')")
    if fusion == 'ekf' and platform != 'robot':
        raise RuntimeError('fusion:=ekf needs wheel odometry: robot only (robot.launch.py)')
    params = {'base_frame': LaunchConfiguration('base_frame').perform(context)}
    # Rust ports (roomba_rs) of the watchdog and ekf_inputs: identical output on recorded drives
    # (tools/ab_watchdog_offline.py, tools/ab_ekf_offline.py) at a fraction of the CPU.
    pkg = 'roomba_rs' if LaunchConfiguration('rust_nodes').perform(context) == 'true' else 'roomba_perception'
    if platform == 'robot':
        params.update({
            'max_speed_mps': 0.6,
            'max_accel_mps2': 1.5,
            'max_vertical_speed_mps': 0.3,
            'max_height_m': 1.0,
            'wheel_odom_topic': '/wheel/odom',
            'planar': True,
        })
    nodes = []
    if fusion == 'ekf':
        params['publish_tf'] = False
        bringup = FindPackageShare('roomba_bringup').perform(context)
        nodes += [
            # Calibrated wheel twist and bias-free gyro with honest covariances for the EKF.
            Node(package=pkg, executable='ekf_inputs', name='ekf_inputs', output='screen'),
            Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node',
                 output='screen', parameters=[os.path.join(bringup, 'config', 'ekf_robot.yaml')],
                 remappings=[('odometry/filtered', '/odom')]),
        ]
    return [Node(package=pkg, executable='vslam_watchdog', name='vslam_watchdog',
                 output='screen', parameters=[params])] + nodes


def _people(context, localization, is_robot):
    """person_detector with the person_model profile of config/person_models.yaml."""
    profile = LaunchConfiguration('person_model').perform(context)
    path = os.path.join(FindPackageShare('roomba_bringup').perform(context), 'config', 'person_models.yaml')
    with open(path) as f:
        profiles = yaml.safe_load(f)
    if profile not in profiles:
        raise RuntimeError(f"person_model must be one of {', '.join(profiles)} (got '{profile}')")
    params = {k: os.path.expanduser(v) if isinstance(v, str) else v for k, v in profiles[profile].items()}
    engine = params.get('engine_path', '')
    log = f'[vslam] person_model={profile}: {engine}'
    if not os.path.isfile(engine):
        log += ' (MISSING: the detector will exit)'
    pkg = 'roomba_rs' if LaunchConfiguration('rust_nodes').perform(context) == 'true' else 'roomba_perception'
    return [LogInfo(msg=log), Node(
        package=pkg,
        executable='person_detector',
        name='person_detector',
        namespace='perception',
        output='screen',
        # Lower CPU priority so tracking always wins under load.
        prefix='nice -n 10',
        # The floor grid overlay needs a frame whose z = 0 is the floor, which only exists on the
        # robot (base_link / planar map).
        parameters=[{**params, 'require_localization': localization, 'floor_grid': is_robot}],
    )]


def generate_launch_description():
    localization = LaunchConfiguration('localization')
    odometry = LaunchConfiguration('odometry')
    use_basalt = PythonExpression(["'", odometry, "' == 'basalt'"])
    use_cuvslam = PythonExpression(["'", odometry, "' == 'cuvslam'"])
    on_robot = ParameterValue(
        PythonExpression(["'", LaunchConfiguration('platform'), "' == 'robot' and '",
                          LaunchConfiguration('ground_constraint'), "' == 'true'"]), value_type=bool)
    is_robot = ParameterValue(
        PythonExpression(["'", LaunchConfiguration('platform'), "' == 'robot'"]), value_type=bool)
    handheld = ParameterValue(
        PythonExpression(["'", LaunchConfiguration('platform'), "' != 'robot'"]), value_type=bool)

    args = [
        DeclareLaunchArgument('rust_nodes', default_value='true',
                              description='Use the Rust ports (package roomba_rs) of nodes that have one'),
        DeclareLaunchArgument('mode', default_value='auto',
                              description='auto (reuse saved map if present), mapping (new map), localization'),
        DeclareLaunchArgument('markers', default_value='true',
                              description='Use printed ArUco markers (DICT_4X4_50, tools/markers) as RTAB-Map '
                                          'landmarks: seeing one fixes the robot\'s position in the map '
                                          'whatever the lighting'),
        DeclareLaunchArgument('marker_length', default_value='0.15',
                              description='Printed marker side in metres (black square)'),
        DeclareLaunchArgument('odometry', default_value='basalt',
                              description='basalt (CPU, most robust on recorded walks) or cuvslam (GPU)'),
        DeclareLaunchArgument('ground_constraint', default_value='true',
                              description='cuVSLAM on the robot: keep the odometry in the floor plane'),
        DeclareLaunchArgument('database_path',
                              default_value=os.path.expanduser('~/maps/vslam.db')),
        DeclareLaunchArgument('foxglove', default_value='true'),
        DeclareLaunchArgument('people', default_value='true',
                              description='Run TensorRT person/pet detection with distance'),
        DeclareLaunchArgument('person_model', default_value='stock',
                              description='person_detector model: a profile of config/person_models.yaml '
                                          '(stock = COCO YOLO11n; finetuned = YOLO11s person + railing; '
                                          'finetuned_n = YOLO11n person + railing)'),
        DeclareLaunchArgument('web', default_value='true', description='Run the web app'),
        DeclareLaunchArgument('web_port', default_value='8080'),
        DeclareLaunchArgument('base_frame', default_value='camera_link',
                              description='Body frame tracked by the odometry (base_link on the robot)'),
        DeclareLaunchArgument('platform', default_value='handheld',
                              description='handheld (camera carried by a person) or robot (camera on the '
                                          'Roomba: robot motion limits, wheel odometry, planar map)'),
        DeclareLaunchArgument('ir_depth_only', default_value='true',
                              description='true: IR projector on depth frames only (15 Hz, clean images); '
                                          'false: every frame (30 Hz, depth synced to color)'),
        DeclareLaunchArgument('ir_intensity', default_value='4',
                              description='IR projector power 0-10 (0 = off, passive stereo depth only)'),
        DeclareLaunchArgument('camera', default_value='true',
                              description='false: no camera driver (replaying a recorded walk)'),
        DeclareLaunchArgument('fusion', default_value='watchdog',
                              description='watchdog: vslam_watchdog publishes odom -> base (filtered VIO, '
                                          'wheel-bridged); ekf (robot only): robot_localization fuses '
                                          'wheels + gyro + VIO (config/ekf_robot.yaml)'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true when replaying a bag with --clock'),
        # Stopping (Ctrl-C, or SIGTERM / SIGINT through scripts/graceful.py): launch sends every
        # process SIGINT, SIGTERM 5 s later (sigterm_timeout) and SIGKILL this much later still.
        # RTAB-Map writes its map database while it shuts down, which takes a while for a big map:
        # launch's default 5 s could kill it mid-write. Applies to every process of the launch.
        DeclareLaunchArgument('sigkill_timeout', default_value='85',
                              description='s from SIGTERM to SIGKILL when stopping (RTAB-Map saves its map)'),
        SetParameter(name='use_sim_time',
                     value=ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)),
        OpaqueFunction(function=_resolve_mode),
    ]

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare('mynteye_d1000_driver'), '/launch/d1000.launch.py']),
        launch_arguments={
            'ir_depth_only': LaunchConfiguration('ir_depth_only'),
            'ir_intensity': LaunchConfiguration('ir_intensity'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('camera')),
    )

    visual_slam = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'rectified_images': True,
            'tracking_mode': 1,  # visual-inertial
            # cuVSLAM's SLAM backend runs only to expose its 3D landmarks for visualization; its
            # map frame is aliased to its odometry frame and it never publishes map TF, so
            # RTAB-Map owns map -> odom. The robot skips it (and the views) to save CPU.
            'enable_localization_n_mapping': handheld,
            # On the Roomba the camera moves in the floor plane of base_link (z up).
            'enable_ground_constraint_in_odometry': on_robot,
            'map_frame': 'vslam_odom',
            'publish_map_to_odom_tf': False,
            # vslam_watchdog (or, with fusion:=ekf, robot_localization) is the only publisher
            # of odom -> base_frame.
            'publish_odom_to_base_tf': False,
            'base_frame': LaunchConfiguration('base_frame'),
            # cuVSLAM odometry lives in vslam_odom; vslam_watchdog anchors it into odom.
            'odom_frame': 'vslam_odom',
            'imu_frame': 'camera_imu_frame',
            'camera_optical_frames': ['camera_left_optical_frame', 'camera_right_optical_frame'],
            # IMU noise model: measured values if available, otherwise cuVSLAM defaults.
            **_imu_noise(),
            # Camera runs 15 Hz in IR-depth-only mode (66.7 ms between frames).
            'image_jitter_threshold_ms': 80.0,
            'imu_jitter_threshold_ms': 10.0,
            'enable_slam_visualization': handheld,
            'enable_observations_view': handheld,
            'enable_landmarks_view': handheld,
            # cuVSLAM defaults to best-effort (SENSOR_DATA); under CPU load whole frames were lost
            # and IMU-only gaps made tracking fail. Reliable delivery resends lost fragments.
            'image_qos': 'DEFAULT',
            'imu_qos': 'DEFAULT',
        }],
        remappings=[
            ('visual_slam/image_0', '/camera/left/image_rect_mono'),
            ('visual_slam/camera_info_0', '/camera/left/camera_info'),
            ('visual_slam/image_1', '/camera/right/image_rect_mono'),
            ('visual_slam/camera_info_1', '/camera/right/camera_info'),
            ('visual_slam/imu', '/camera/imu/data_raw'),
        ],
    )

    vslam_container = ComposableNodeContainer(
        name='visual_slam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[visual_slam],
        output='screen',
        condition=IfCondition(use_cuvslam),
    )

    # Basalt stereo-inertial odometry on the same topics/service as cuVSLAM. On a recorded
    # handheld walk it had no jumps and ~10x less height drift than cuVSLAM.
    basalt_share = FindPackageShare('basalt_vio_ros')
    basalt_vio = Node(
        package='basalt_vio_ros',
        executable='basalt_vio_node',
        name='basalt_vio',
        output='screen',
        condition=IfCondition(use_basalt),
        parameters=[{
            'calib_path': (BASALT_CALIB_FILE if os.path.isfile(BASALT_CALIB_FILE)
                           else PathJoinSubstitution([basalt_share, 'config', 'basalt_calib.json'])),
            'config_path': PathJoinSubstitution([basalt_share, 'config', 'basalt_config.json']),
            'odom_frame': 'vslam_odom',
            'base_frame': LaunchConfiguration('base_frame'),
            'imu_frame': 'camera_imu_frame',
        }],
        remappings=[
            ('left/image', '/camera/left/image_rect_mono'),
            ('right/image', '/camera/right/image_rect_mono'),
            ('imu', '/camera/imu/data_raw'),
        ],
    )

    # Motion sanity filter: only physically plausible odometry reaches odom -> base_frame;
    # resets the odometry if tracking is lost or glitches repeat.
    vslam_watchdog = OpaqueFunction(function=_watchdog)

    # Orientation from IMU for RTAB-Map's gravity constraints.
    imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        namespace='camera',
        parameters=[{'use_mag': False, 'publish_tf': False, 'world_frame': 'enu'}],
        remappings=[('imu/data_raw', '/camera/imu/data_raw'), ('imu/data', '/camera/imu/data')],
    )

    rtabmap_args = PythonExpression([
        # Parenthesized: without them the conditional swallowed every argument below in
        # localization mode.
        "('' if '", localization, "' == 'true' else '--delete_db_on_start')",
        # RayTracing clears cells seen as free again, so people who walked through
        # don't leave phantom obstacles that would later block line of sight.
        " + ' --Grid/RangeMax 2.5 --Grid/MaxObstacleHeight 1.0 --Grid/RayTracing true'",
        # Loop closures must be verified by enough inliers. cuVSLAM's drift needed a lenient 15;
        # with Basalt's accurate odometry 15 let similar-looking doors match the wrong place
        # and warp the map, 25 gave a clean map on the same recorded walk.
        " + ' --Vis/MinInliers ' + ('25' if '", odometry, "' == 'basalt' else '15')",
        # Free space comes mostly from ray tracing (a carried camera rarely sees the floor). With
        # the default ProbMiss 0.4 two passing rays erase a wall seen once, so interior walls
        # vanished; 0.45 needs ~4. Wherever the camera was carried is free (footprint).
        " + ' --GridGlobal/ProbMiss 0.45 --GridGlobal/FootprintRadius 0.25'",
        # The robot moves in the floor plane: optimize x, y and yaw only. Its camera sees the floor,
        # so ground is "below 6 cm in base_link" (as floor_obstacles), not surface normals,
        # which were noisy on carpet; isolated specks are filtered so they don't become
        # robot-sized blobs after costmap inflation.
        " + (' --Reg/Force3DoF true --Grid/NormalsSegmentation false --Grid/MaxGroundHeight 0.06'"
        "     ' --Grid/NoiseFilteringRadius 0.1 --Grid/NoiseFilteringMinNeighbors 5'"
        "     ' --GridGlobal/FootprintRadius 0.2'"
        "    if '", LaunchConfiguration('platform'), "' == 'robot' else '')",
        # Printed markers as landmarks (tools/markers/make_markers.py): the first sighting places a
        # marker in the map, every later one relocalizes the robot even when the room looks different
        # (a map made in daylight matched 0 features at night). Beyond 3 m the 640x360 image gives
        # too few pixels per marker cell for a reliable pose.
        " + (' --RGBD/MarkerDetection true --Marker/Dictionary 0 --Marker/MaxRange 3.0 --Marker/Length '",
        "     + '", LaunchConfiguration('marker_length'), "' if '", LaunchConfiguration('markers'), "' == 'true' else '')",
    ])

    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare('rtabmap_launch'), '/launch/rtabmap.launch.py']),
        launch_arguments={
            'namespace': 'rtabmap',
            'rgb_topic': '/camera/left/image_rect_color',
            'depth_topic': '/camera/depth/image_rect_raw',
            'camera_info_topic': '/camera/left/camera_info',
            'frame_id': LaunchConfiguration('base_frame'),
            'visual_odometry': 'false',
            # Read the filtered odometry from TF. Variance per link (~1 s of motion) should match
            # the odometry's real drift: too small and RTAB-Map rejects loop closures that must
            # absorb drift (cuVSLAM: a 7 deg correction rejected at 0.001, so 0.005); Basalt
            # drifts far less and 0.001 lets RTAB-Map reject wrong closures instead.
            'odom_frame_id': 'odom',
            'odom_tf_linear_variance': PythonExpression(["'0.001' if '", odometry, "' == 'basalt' else '0.005'"]),
            'odom_tf_angular_variance': PythonExpression(["'0.001' if '", odometry, "' == 'basalt' else '0.005'"]),
            'imu_topic': '/camera/imu/data',
            'wait_imu_to_init': 'true',
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.05',
            'rgbd_sync': 'true',
            'approx_rgbd_sync': 'false',
            'topic_queue_size': '3',
            'rtabmap_args': rtabmap_args,
            'localization': localization,
            'database_path': LaunchConfiguration('database_path'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'rtabmap_viz': 'false',
            'rviz': 'false',
        }.items(),
    )

    republish = Node(
        package='image_transport',
        executable='republish',
        name='left_republish',
        namespace='camera',
        parameters=[{'in_transport': 'raw', 'out_transport': 'compressed'}],
        remappings=[('in', '/camera/left/image_rect_color'), ('out/compressed', '/camera/left/view/compressed')],
        condition=IfCondition(LaunchConfiguration('foxglove')),
    )

    # Rust port (roomba_rs) by default: identical detections, distances, ignore decisions and annotated
    # images (tools/ab_person_detector_offline.py) at a fraction of the CPU. The model (engine and
    # classes) is a profile of config/person_models.yaml, chosen with person_model:=.
    people = OpaqueFunction(function=_people, kwargs={'localization': localization, 'is_robot': is_robot},
                            condition=IfCondition(LaunchConfiguration('people')))

    # Ignore zones are stored next to the map database they were drawn on. Rust port (roomba_rs) by
    # default: identical outputs and zones file (tools/ab_ignore_zones_offline.py).
    ignore_zones = Node(
        package=PythonExpression(["'roomba_rs' if '", LaunchConfiguration('rust_nodes'),
                                  "' == 'true' else 'roomba_perception'"]),
        executable='ignore_zones',
        name='ignore_zones',
        namespace='perception',
        output='screen',
        parameters=[{
            'zones_file': PythonExpression(
                ["'", LaunchConfiguration('database_path'), "'.rsplit('.db', 1)[0] + '.zones.json'"]),
            # Mapping mode deletes the database, so zones drawn on the old map are set aside.
            'clear_on_start': PythonExpression(["'", localization, "' != 'true'"]),
            # A keep-out zone added over the robot leaves its footprint (0.18 m) + 0.1 m free, so it
            # can still turn and back away (a bump mark over the robot once blocked every move).
            'robot_clear_radius_m': 0.28,
            'base_frame': LaunchConfiguration('base_frame'),
        }],
        # Always on: besides the person detector's exclusion zones it serves the keep-out mask the
        # costmaps use, including the explorer's bump and cliff marks (it once ran only with
        # people:=true, so with the detector off every bump mark was silently dropped).
    )

    # Rust port (roomba_rs) by default: same routes, login, websocket state, images and commands
    # (tools/ab_web.sh); it serves roomba_web's page. rust_nodes:=false runs the Python server.
    web_app = Node(
        package=PythonExpression(["'roomba_rs' if '", LaunchConfiguration('rust_nodes'),
                                  "' == 'true' else 'roomba_web'"]),
        executable='web_server',
        name='web_server',
        output='screen',
        parameters=[{'port': LaunchConfiguration('web_port'), 'localization': localization,
                     'base_frame': LaunchConfiguration('base_frame'), 'floor_grid': is_robot}],
        condition=IfCondition(LaunchConfiguration('web')),
    )

    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': 8765,
            'topic_whitelist': [
                '/tf', '/tf_static', '/camera/left/view/compressed', '/camera/left/camera_info',
                '/camera/imu/data', '/rtabmap/map', '/rtabmap/mapPath', '/rtabmap/cloud_map',
                '/rtabmap/info', '/visual_slam/tracking/odometry', '/visual_slam/tracking/vo_path',
                '/visual_slam/status', '/visual_slam/vis/landmarks_cloud',
                '/visual_slam/vis/observations_cloud', '/rosout', '/diagnostics',
                '/perception/person_detector/annotated/compressed',
                '/perception/person_detector/markers',
                '/perception/person_detector/detections_3d',
                '/perception/ignore_zones/markers',
                # for src/roomba_bringup/config/foxglove/explorer_debug.json (the bridge only sends
                # what a panel subscribes to; the raw depth image is ~10 MB/s while shown)
                '/global_costmap/costmap', '/local_costmap/costmap', '/keepout_filter_mask',
                '/plan', '/local_plan', '/local_costmap/published_footprint', '/explorer/markers',
                '/explorer/status', '/vslam_filter/status', '/perception/obstacles',
                '/perception/floor_points', '/camera/depth/image_rect_raw',
            ],
        }],
        condition=IfCondition(LaunchConfiguration('foxglove')),
    )

    return LaunchDescription(args + [camera, vslam_container, basalt_vio, vslam_watchdog, imu_filter, rtabmap,
                                     republish, people, ignore_zones, web_app, foxglove])
