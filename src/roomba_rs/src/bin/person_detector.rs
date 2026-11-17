//! person_detector (Rust): people (and pets) with distance for the Roomba robot. Drop-in replacement
//! for roomba_perception/person_detector.py: same node name, parameters, topics, QoS, exact-time sync,
//! outputs and log lines (see that file's docstring). The processing is roomba_rs::person (YOLO
//! through TensorRT, depth / StereoBM distance, ignore zones and line of sight, annotated JPEG);
//! TF lookups use roomba_rs::tf_buffer.
//!
//! Subscribes: /camera/left/image_rect_color, /camera/depth/image_rect_raw, /camera/right/image_rect_mono
//!             (exact-time synchronized), /camera/{left,right}/camera_info, /rtabmap/map,
//!             /perception/ignore_zones, /rtabmap/localization_pose, ~/floor_grid, /tf, /tf_static
//! Publishes:  ~/annotated/compressed, ~/detections_2d, ~/detections_3d, ~/ignored_3d, ~/markers,
//!             ~/objects_3d (classes that are not people, e.g. railings; see `person_classes`)
//!
//! The executor thread only receives (and synchronizes) messages; a worker thread runs the frames in
//! arrival order, so TF keeps arriving while a frame waits for its transform (as with tf2_ros'
//! listener thread in Python). At most two synchronized sets wait for the worker; older ones are
//! dropped, so a slow moment never makes the detections lag (Python works off its 5-deep queues).

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use rclrs::*;
use ros_env::{builtin_interfaces, geometry_msgs, nav_msgs, sensor_msgs, std_msgs, vision_msgs, visualization_msgs};

use roomba_rs::person::{Det3d, Detector, Frame, FrameOutput, ImageView, Log, MarkerOut, Params, TfLookup, TimeSync3, MARKER_LIFETIME_NS};
use roomba_rs::tf_buffer::{TfBuffer, TfError, Transform};
use roomba_rs::tf_listener::{stamp_ns, TfListener};
use roomba_rs::zones::Level;

type Image = sensor_msgs::msg::Image;

/// Synchronized sets allowed to wait for the worker.
const MAX_PENDING_FRAMES: usize = 2;

enum Event {
    LeftInfo([f64; 12]),
    RightInfo([f64; 12]),
    Frame(Box<[Image; 3]>),
    Map(nav_msgs::msg::OccupancyGrid),
    Zones(String),
    Localized,
    Grid(bool),
    Stats,
}

/// Events for the worker thread, in arrival order.
#[derive(Default)]
struct Queue {
    events: Mutex<VecDeque<Event>>,
    ready: Condvar,
}

impl Queue {
    fn push(&self, e: Event) {
        let mut q = self.events.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(e, Event::Frame(_)) {
            let frames = q.iter().filter(|e| matches!(e, Event::Frame(_))).count();
            if frames >= MAX_PENDING_FRAMES {
                if let Some(i) = q.iter().position(|e| matches!(e, Event::Frame(_))) {
                    q.remove(i);
                }
            }
        }
        q.push_back(e);
        self.ready.notify_one();
    }

    fn pop(&self) -> Event {
        let mut q = self.events.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(e) = q.pop_front() {
                return e;
            }
            q = self.ready.wait(q).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// tf2_ros Buffer.lookup_transform with a timeout: retries until the transform arrives (the executor
/// thread keeps filling the buffer) or the time is up.
struct WaitingTf {
    buffer: Arc<Mutex<TfBuffer>>,
}

impl TfLookup for WaitingTf {
    fn lookup(&self, target: &str, source: &str, time_ns: i64, timeout_s: f64) -> Result<Transform, TfError> {
        let deadline = Instant::now() + Duration::from_secs_f64(timeout_s.max(0.0));
        loop {
            let r = self.buffer.lock().unwrap_or_else(|e| e.into_inner()).lookup_transform(target, source, time_ns);
            if r.is_ok() || Instant::now() >= deadline {
                return r.map(|t| t.transform);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Declares parameters and keeps their handles (rclrs undeclares a parameter when its handle drops).
struct Declared<'a> {
    node: &'a Node,
    handles: Vec<Box<dyn std::any::Any>>,
}

impl Declared<'_> {
    fn get<T: ParameterVariant + 'static>(&mut self, name: &str, default: T) -> Result<T> {
        let p = self.node.declare_parameter(name).default(default).mandatory()?;
        let value = p.get();
        self.handles.push(Box::new(p));
        Ok(value)
    }
}

struct Pubs {
    img: Publisher<sensor_msgs::msg::CompressedImage>,
    d2: Publisher<vision_msgs::msg::Detection2DArray>,
    d3: Publisher<vision_msgs::msg::Detection3DArray>,
    ignored: Publisher<vision_msgs::msg::Detection3DArray>,
    /// Only when some configured class is not a person.
    objects: Option<Publisher<vision_msgs::msg::Detection3DArray>>,
    markers: Publisher<visualization_msgs::msg::MarkerArray>,
}

fn point(p: [f64; 3]) -> geometry_msgs::msg::Point {
    geometry_msgs::msg::Point { x: p[0], y: p[1], z: p[2] }
}

fn identity() -> geometry_msgs::msg::Quaternion {
    geometry_msgs::msg::Quaternion { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
}

fn hypothesis(class_id: &str, score: f64, position: geometry_msgs::msg::Point) -> vision_msgs::msg::ObjectHypothesisWithPose {
    let mut h = vision_msgs::msg::ObjectHypothesisWithPose::default();
    h.hypothesis.class_id = class_id.into();
    h.hypothesis.score = score;
    h.pose.pose.position = position;
    h.pose.pose.orientation = identity();
    h
}

fn det3d_array(header: &std_msgs::msg::Header, dets: &[Det3d]) -> vision_msgs::msg::Detection3DArray {
    let detections = dets
        .iter()
        .map(|d| {
            let mut m = vision_msgs::msg::Detection3D { header: header.clone(), id: d.id.clone(), ..Default::default() };
            m.bbox.center.position = point(d.position);
            m.bbox.center.orientation = identity();
            m.bbox.size = geometry_msgs::msg::Vector3 { x: d.size[0], y: d.size[1], z: d.size[2] };
            m.results.push(hypothesis(&d.class_id, d.score, point(d.position)));
            m
        })
        .collect();
    vision_msgs::msg::Detection3DArray { header: header.clone(), detections }
}

fn marker(header: &std_msgs::msg::Header, m: &MarkerOut) -> visualization_msgs::msg::Marker {
    let mut out = visualization_msgs::msg::Marker {
        header: header.clone(),
        ns: m.ns.into(),
        id: m.id,
        type_: m.kind,
        action: 0,
        text: m.text.clone(),
        lifetime: builtin_interfaces::msg::Duration { sec: 0, nanosec: MARKER_LIFETIME_NS },
        ..Default::default()
    };
    out.pose.position = point(m.position);
    out.pose.orientation = identity();
    out.scale = geometry_msgs::msg::Vector3 { x: m.scale[0], y: m.scale[1], z: m.scale[2] };
    out.color = std_msgs::msg::ColorRGBA { r: m.color[0], g: m.color[1], b: m.color[2], a: m.color[3] };
    out
}

fn publish(pubs: &Pubs, header: &std_msgs::msg::Header, out: FrameOutput) {
    let detections = out
        .det2d
        .iter()
        .map(|d| {
            let mut m = vision_msgs::msg::Detection2D { header: header.clone(), ..Default::default() };
            m.bbox.center.position = vision_msgs::msg::Point2D { x: d.center[0], y: d.center[1] };
            m.bbox.size_x = d.size[0];
            m.bbox.size_y = d.size[1];
            m.results.push(hypothesis(&d.class_id, d.score, geometry_msgs::msg::Point::default()));
            m
        })
        .collect();
    pubs.d2.publish(vision_msgs::msg::Detection2DArray { header: header.clone(), detections }).ok();
    pubs.d3.publish(det3d_array(header, &out.det3d)).ok();
    pubs.ignored.publish(det3d_array(header, &out.ignored)).ok();
    if let Some(objects) = &pubs.objects {
        objects.publish(det3d_array(header, &out.objects)).ok();
    }
    if !out.markers.is_empty() {
        let markers = out.markers.iter().map(|m| marker(header, m)).collect();
        pubs.markers.publish(visualization_msgs::msg::MarkerArray { markers }).ok();
    }
    if let Some(data) = out.jpeg {
        pubs.img.publish(sensor_msgs::msg::CompressedImage { header: header.clone(), format: "jpeg".into(), data }).ok();
    }
}

fn log_all(node: &Node, logs: impl IntoIterator<Item = Log>) {
    for (level, text) in logs {
        match level {
            Level::Info => log!(node.info(), "{text}"),
            Level::Warn => log!(node.warn(), "{text}"),
            Level::Error => log!(node.error(), "{text}"),
        }
    }
}

fn view(m: &Image) -> ImageView<'_> {
    ImageView { data: &m.data, width: m.width as usize, height: m.height as usize, step: m.step as usize }
}

/// The worker thread: applies events in order and runs the frames.
fn run_worker(mut det: Detector, queue: Arc<Queue>, pubs: Pubs, tf: WaitingTf, node: Node) {
    let start = Instant::now();
    let clock = move || start.elapsed().as_secs_f64();
    loop {
        match queue.pop() {
            Event::LeftInfo(p) => det.on_left_info(p),
            Event::RightInfo(p) => det.on_right_info(p),
            Event::Zones(text) => log_all(&node, [det.on_zones(&text)]),
            Event::Localized => log_all(&node, det.on_localized()),
            Event::Grid(on) => log_all(&node, [det.on_grid_toggle(on)]),
            Event::Stats => log_all(&node, det.stats()),
            Event::Map(m) => {
                let (i, o, q) = (&m.info, &m.info.origin.position, &m.info.origin.orientation);
                det.on_map(i.width as usize, i.height as usize, i.resolution, [o.x, o.y], [q.x, q.y, q.z, q.w], &m.header.frame_id, &m.data);
            }
            Event::Frame(msgs) => {
                let [color, depth, right] = &*msgs;
                let frame = Frame {
                    stamp_ns: stamp_ns(&color.header.stamp),
                    frame_id: &color.header.frame_id,
                    color: view(color),
                    depth: view(depth),
                    right: view(right),
                };
                let annotate = pubs.img.get_subscription_count().unwrap_or(0) > 0;
                let mut logs = Vec::new();
                let result = det.process(&frame, annotate, &tf, &clock, &mut logs);
                log_all(&node, logs);
                match result {
                    Ok(Some(out)) => publish(&pubs, &color.header, out),
                    Ok(None) => {}
                    Err(e) => {
                        // Python's node dies with the RuntimeError; so does this one.
                        log!(node.fatal(), "{e}");
                        std::process::exit(1);
                    }
                }
            }
        }
    }
}

struct SyncState {
    sync: TimeSync3<Image>,
    queue: Arc<Queue>,
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("person_detector")?;
    let mut d = Declared { node: &node, handles: Vec::new() };
    let defaults = Params::default();
    let engine_path: Arc<str> = d.get("engine_path", Arc::from(defaults.engine_path.as_str()))?;
    let conf_threshold = d.get("conf_threshold", defaults.conf_threshold)?;
    let nms_threshold = d.get("nms_threshold", defaults.nms_threshold)?;
    let classes: Arc<[i64]> = d.get("classes", Arc::from(defaults.classes.as_slice()))?;
    let strings = |v: &[String]| -> Arc<[Arc<str>]> { v.iter().map(|s| Arc::from(s.as_str())).collect() };
    let class_names: Arc<[Arc<str>]> = d.get("class_names", strings(&defaults.class_names))?;
    let person_classes: Arc<[Arc<str>]> = d.get("person_classes", strings(&defaults.person_classes))?;
    let jpeg_quality = d.get("jpeg_quality", defaults.jpeg_quality)?;
    let min_depth_pixels = d.get("min_depth_pixels", defaults.min_depth_pixels)?;
    let map_frame: Arc<str> = d.get("map_frame", Arc::from(defaults.map_frame.as_str()))?;
    let params = Params {
        engine_path: engine_path.to_string(),
        conf_threshold,
        nms_threshold,
        classes: classes.to_vec(),
        class_names: class_names.iter().map(|s| s.to_string()).collect(),
        person_classes: person_classes.iter().map(|s| s.to_string()).collect(),
        jpeg_quality,
        min_depth_pixels,
        map_frame: map_frame.to_string(),
        ignore_zones: d.get("ignore_zones", defaults.ignore_zones)?,
        line_of_sight: d.get("line_of_sight", defaults.line_of_sight)?,
        occupied_threshold: d.get("occupied_threshold", defaults.occupied_threshold)?,
        min_blocking_cells: d.get("min_blocking_cells", defaults.min_blocking_cells)?,
        target_margin_m: d.get("target_margin_m", defaults.target_margin_m)?,
        map_topic: d.get::<Arc<str>>("map_topic", Arc::from(defaults.map_topic.as_str()))?.to_string(),
        require_localization: d.get("require_localization", defaults.require_localization)?,
        floor_grid: d.get("floor_grid", defaults.floor_grid)?,
        floor_grid_fallback_frame: d.get::<Arc<str>>("floor_grid_fallback_frame", Arc::from(defaults.floor_grid_fallback_frame.as_str()))?.to_string(),
    };
    let map_topic = params.map_topic.clone();
    let det = match Detector::new(params) {
        Ok((det, loaded)) => {
            log_all(&node, [loaded]);
            det
        }
        Err(e) => {
            log!(node.fatal(), "{e}");
            std::process::exit(1);
        }
    };

    let has_objects = det.has_objects();
    let queue = Arc::new(Queue::default());
    let push = |q: &Arc<Queue>| {
        let q = Arc::clone(q);
        move |e: Event| q.push(e)
    };

    let tf = TfListener::new(&node)?;
    let pubs = Pubs {
        img: node.create_publisher("~/annotated/compressed".keep_last(5))?,
        d2: node.create_publisher("~/detections_2d".keep_last(5))?,
        d3: node.create_publisher("~/detections_3d".keep_last(5))?,
        ignored: node.create_publisher("~/ignored_3d".keep_last(5))?,
        objects: if has_objects { Some(node.create_publisher("~/objects_3d".keep_last(5))?) } else { None },
        markers: node.create_publisher("~/markers".keep_last(5))?,
    };

    let info = |p: fn([f64; 12]) -> Event| {
        let send = push(&queue);
        move |m: sensor_msgs::msg::CameraInfo| send(p(m.p))
    };
    let _left_info = node.create_subscription::<sensor_msgs::msg::CameraInfo, _>("/camera/left/camera_info".keep_last(5), info(Event::LeftInfo))?;
    let _right_info = node.create_subscription::<sensor_msgs::msg::CameraInfo, _>("/camera/right/camera_info".keep_last(5), info(Event::RightInfo))?;

    let sync = node.create_worker::<SyncState>(SyncState { sync: TimeSync3::new(10), queue: Arc::clone(&queue) });
    let image_sub = |topic: &str, input: usize| {
        sync.create_subscription::<Image, _>(topic.keep_last(5), move |s: &mut SyncState, m: Image| {
            if let Some(set) = s.sync.add(input, stamp_ns(&m.header.stamp), m) {
                s.queue.push(Event::Frame(Box::new(set)));
            }
        })
    };
    let _color = image_sub("/camera/left/image_rect_color", 0)?;
    let _depth = image_sub("/camera/depth/image_rect_raw", 1)?;
    let _right = image_sub("/camera/right/image_rect_mono", 2)?;

    // RTAB-Map publishes the grid latched (transient local) and only on change.
    let latched = || QoSProfile::topics_default().keep_last(1).reliable().transient_local();
    let send = push(&queue);
    let _map = node.create_subscription::<nav_msgs::msg::OccupancyGrid, _>(map_topic.as_str().qos(latched()), move |m: nav_msgs::msg::OccupancyGrid| send(Event::Map(m)))?;
    let send = push(&queue);
    let _zones = node.create_subscription::<std_msgs::msg::String, _>("/perception/ignore_zones".qos(latched()), move |m: std_msgs::msg::String| send(Event::Zones(m.data)))?;
    let send = push(&queue);
    let _localized = node.create_subscription::<geometry_msgs::msg::PoseWithCovarianceStamped, _>(
        "/rtabmap/localization_pose".keep_last(1),
        move |_m: geometry_msgs::msg::PoseWithCovarianceStamped| send(Event::Localized),
    )?;
    let send = push(&queue);
    let _grid = node.create_subscription::<std_msgs::msg::Bool, _>("~/floor_grid".keep_last(2), move |m: std_msgs::msg::Bool| send(Event::Grid(m.data)))?;
    let send = push(&queue);
    // Steady clock: a ROS-time timer can stop this wait set for good on a clock step (see explorer).
    let _stats = node.create_timer_repeating(Duration::from_secs(30).steady_time(), move || send(Event::Stats))?;

    let worker_tf = WaitingTf { buffer: Arc::clone(tf.buffer()) };
    let (worker_queue, worker_node) = (Arc::clone(&queue), node.clone());
    std::thread::Builder::new().name("person_detector".into()).spawn(move || run_worker(det, worker_queue, pubs, worker_tf, worker_node))?;
    roomba_rs::node_params::spin(&mut executor)?;
    Ok(())
}
