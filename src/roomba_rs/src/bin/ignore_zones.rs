//! ignore_zones (Rust): zone editor for exclusion zones (people inside are ignored) and navigation
//! keep-out zones, including the explorer's temporary bump and cliff marks. Drop-in replacement for
//! roomba_perception/ignore_zones.py: same node name, topics, QoS, parameters, zones file (format and
//! path) and outputs (see that file's docstring); the logic is roomba_rs::zones, TF lookups use
//! roomba_rs::tf_buffer.
//!
//! Subscribes: /clicked_point (geometry_msgs/PointStamped, any frame), ~/command (std_msgs/String),
//!             /tf, /tf_static
//! Publishes:  /perception/ignore_zones (String JSON, transient local), /perception/ignore_zones/markers,
//!             /keepout_filter_mask (OccupancyGrid, transient local), /costmap_filter_info (transient local)

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use rclrs::*;
use ros_env::{builtin_interfaces, geometry_msgs, nav2_msgs, nav_msgs, std_msgs, visualization_msgs};

use roomba_rs::tf_buffer::TfBuffer;
use roomba_rs::tf_listener::{HasTfBuffer, WorkerTfListener};
use roomba_rs::zones::{expand_user, IgnoreZones, Level, MarkerSpec, Output, Params, DELETEALL};

/// time.time()
fn wall() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
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
    zones: Publisher<std_msgs::msg::String>,
    markers: Publisher<visualization_msgs::msg::MarkerArray>,
    mask: Publisher<nav_msgs::msg::OccupancyGrid>,
}

struct State {
    core: IgnoreZones,
    tf: TfBuffer,
    pubs: Pubs,
    node: Node,
}

impl HasTfBuffer for State {
    fn tf_buffer(&mut self) -> &mut TfBuffer {
        &mut self.tf
    }
}

fn header(node: &Node, frame: &str) -> std_msgs::msg::Header {
    let ns = node.get_clock().now().nsec;
    std_msgs::msg::Header {
        stamp: builtin_interfaces::msg::Time { sec: ns.div_euclid(1_000_000_000) as i32, nanosec: ns.rem_euclid(1_000_000_000) as u32 },
        frame_id: frame.into(),
    }
}

fn marker(m: &MarkerSpec, header: &std_msgs::msg::Header) -> visualization_msgs::msg::Marker {
    let mut out = visualization_msgs::msg::Marker {
        header: header.clone(),
        ns: m.ns.clone(),
        id: m.id,
        type_: m.kind,
        action: m.action,
        text: m.text.clone(),
        ..Default::default()
    };
    out.pose.position = geometry_msgs::msg::Point { x: m.position[0], y: m.position[1], z: m.position[2] };
    out.pose.orientation = geometry_msgs::msg::Quaternion { x: 0.0, y: 0.0, z: 0.0, w: m.orientation_w };
    out.scale = geometry_msgs::msg::Vector3 { x: m.scale[0], y: m.scale[1], z: m.scale[2] };
    out.color = std_msgs::msg::ColorRGBA { r: m.color[0], g: m.color[1], b: m.color[2], a: m.color[3] };
    out.points = m.points.iter().map(|p| geometry_msgs::msg::Point { x: p[0], y: p[1], z: p[2] }).collect();
    out
}

impl State {
    /// The robot's position in the map frame (latest TF), if known.
    fn robot(&self) -> Option<[f64; 2]> {
        let p = &self.core.params;
        let t = self.tf.lookup_transform(&p.map_frame, &p.base_frame, 0).ok()?;
        Some([t.transform.translation[0], t.transform.translation[1]])
    }

    fn emit(&self, outputs: Vec<Output>) {
        let frame = &self.core.params.map_frame;
        for o in outputs {
            match o {
                Output::Zones(data) => {
                    self.pubs.zones.publish(std_msgs::msg::String { data }).ok();
                }
                Output::Mask(m) => {
                    let mut grid = nav_msgs::msg::OccupancyGrid { header: header(&self.node, frame), data: m.data, ..Default::default() };
                    grid.info.resolution = self.core.params.keepout_resolution_m as f32;
                    grid.info.width = m.width as u32;
                    grid.info.height = m.height as u32;
                    grid.info.origin.position.x = m.origin_x;
                    grid.info.origin.position.y = m.origin_y;
                    grid.info.origin.orientation = geometry_msgs::msg::Quaternion { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
                    self.pubs.mask.publish(grid).ok();
                }
                Output::Markers(ms) => {
                    let h = header(&self.node, frame);
                    let clear = MarkerSpec {
                        ns: String::new(),
                        id: 0,
                        kind: 0,
                        action: DELETEALL,
                        position: [0.0; 3],
                        orientation_w: 1.0,
                        scale: [0.0; 3],
                        color: [0.0; 4],
                        points: Vec::new(),
                        text: String::new(),
                    };
                    let markers = std::iter::once(&clear).chain(ms.iter()).map(|m| marker(m, &h)).collect();
                    self.pubs.markers.publish(visualization_msgs::msg::MarkerArray { markers }).ok();
                }
                Output::Log(Level::Info, text) => log!(self.node.info(), "{text}"),
                Output::Log(Level::Warn, text) => log!(self.node.warn(), "{text}"),
                Output::Log(Level::Error, text) => log!(self.node.error(), "{text}"),
            }
        }
    }
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("ignore_zones")?;
    let mut d = Declared { node: &node, handles: Vec::new() };
    let zones_file: Arc<str> = d.get("zones_file", Arc::from("~/maps/vslam.zones.json"))?;
    let map_frame: Arc<str> = d.get("map_frame", Arc::from("map"))?;
    let params = Params {
        zones_file: expand_user(&zones_file),
        map_frame: map_frame.to_string(),
        close_distance_m: d.get("close_distance_m", 0.25)?,
        keepout_margin_m: d.get("keepout_margin_m", 0.2)?,
        keepout_resolution_m: d.get("keepout_resolution_m", 0.05)?,
        clear_on_start: d.get("clear_on_start", false)?,
        robot_clear_radius_m: d.get("robot_clear_radius_m", 0.0)?,
        base_frame: d.get::<Arc<str>>("base_frame", Arc::from("base_link"))?.to_string(),
        keepout_soft_band_m: d.get("keepout_soft_band_m", 0.0)?,
    };

    let latched = || QoSProfile::topics_default().keep_last(1).reliable().transient_local();
    let pubs = Pubs {
        zones: node.create_publisher("/perception/ignore_zones".qos(latched()))?,
        markers: node.create_publisher("/perception/ignore_zones/markers".keep_last(5))?,
        mask: node.create_publisher("/keepout_filter_mask".qos(latched()))?,
    };
    let filter_info: Publisher<nav2_msgs::msg::CostmapFilterInfo> = node.create_publisher("/costmap_filter_info".qos(latched()))?;
    filter_info
        .publish(nav2_msgs::msg::CostmapFilterInfo {
            header: header(&node, &params.map_frame),
            type_: 0,
            filter_mask_topic: "/keepout_filter_mask".into(),
            base: 0.0,
            multiplier: 1.0,
        })
        .ok();

    let (core, startup) = IgnoreZones::new(params);
    let state = State { core, tf: TfBuffer::new(), pubs, node: node.clone() };
    state.emit(startup);
    let worker = node.create_worker::<State>(state);
    let _tf = WorkerTfListener::new(&worker, &node)?;
    let _click = worker.create_subscription::<geometry_msgs::msg::PointStamped, _>(
        "/clicked_point".keep_last(10),
        |s: &mut State, m: geometry_msgs::msg::PointStamped| {
            let out = s.core.on_click(&m.header.frame_id, [m.point.x, m.point.y, m.point.z], &s.tf);
            s.emit(out);
        },
    )?;
    let _command = worker.create_subscription::<std_msgs::msg::String, _>(
        "~/command".keep_last(10),
        |s: &mut State, m: std_msgs::msg::String| {
            let robot = s.robot();
            let out = s.core.on_command(&m.data, wall(), robot);
            s.emit(out);
        },
    )?;
    // Steady-clock timers (the core gets the time it uses): a ROS-time timer whose wait ends just as
    // the system clock is stepped back makes rcl_wait report a timeout, and rclrs 0.8 then stops this
    // worker's wait set for good while the node's own one keeps spinning (the worker froze).
    let _markers = worker.create_timer_repeating(Duration::from_secs(1).steady_time(), |s: &mut State| {
        let out = vec![s.core.markers()];
        s.emit(out);
    })?;
    let _expire = worker.create_timer_repeating(Duration::from_secs(2).steady_time(), |s: &mut State| {
        let out = s.core.expire(wall());
        s.emit(out);
    })?;
    roomba_rs::node_params::spin(&mut executor)?;
    Ok(())
}
