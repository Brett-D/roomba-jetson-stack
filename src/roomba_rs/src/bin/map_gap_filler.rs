//! map_gap_filler (Rust): fills small gaps in RTAB-Map's occupancy grid so dotted obstacles (a
//! railing's posts) become solid walls, and joins >= 3 lined-up small dots into fences - never across
//! the robot's own path. Drop-in replacement for roomba_perception/map_gap_filler.py: same node
//! name, topics, QoS, parameters, logs and output grid (cell for cell); the processing is
//! roomba_rs::gap_filler.
//!
//! Subscribes: /rtabmap/map (nav_msgs/OccupancyGrid, transient local), /rtabmap/mapPath (nav_msgs/Path)
//! Publishes:  /rtabmap/map_filled (nav_msgs/OccupancyGrid, transient local) - for the costmaps

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use rclrs::*;
use ros_env::nav_msgs;

use roomba_rs::gap_filler::{GapFiller, GridInfo, Params};

/// Declares parameters and keeps their handles: rclrs undeclares a parameter when its handle is
/// dropped, and `ros2 param get/list` should show them as for the Python node.
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

struct State {
    filler: GapFiller,
    publisher: Publisher<nav_msgs::msg::OccupancyGrid>,
    node: Node,
    last_throttled: Option<Instant>,
}

impl State {
    fn on_map(&mut self, m: nav_msgs::msg::OccupancyGrid) {
        let i = &m.info;
        let info = GridInfo {
            width: i.width as usize,
            height: i.height as usize,
            resolution: i.resolution,
            origin_x: i.origin.position.x,
            origin_y: i.origin.position.y,
        };
        if info.width * info.height != m.data.len() && info.width > 0 && info.height > 0 {
            log!(self.node.error(), "Map data has {} cells, expected {} x {}: ignored", m.data.len(), info.width, info.height);
            return;
        }
        let Some(filled) = self.filler.on_map(&info, &m.data) else {
            self.publisher.publish(m).ok();
            return;
        };
        let out = nav_msgs::msg::OccupancyGrid { header: m.header, info: m.info, data: filled.data };
        self.publisher.publish(out).ok();
        for (throttled, text) in filled.logs {
            if throttled {
                // rclpy's throttle_duration_sec=30.0
                if self.last_throttled.is_some_and(|t| t.elapsed() < Duration::from_secs(30)) {
                    continue;
                }
                self.last_throttled = Some(Instant::now());
            }
            log!(self.node.info(), "{text}");
        }
    }
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("map_gap_filler")?;
    let mut d = Declared { node: &node, handles: Vec::new() };
    let gap_m = d.get("gap_m", 0.10)?;
    let occupied = d.get("occupied_threshold", 65i64)?;
    let params = Params {
        gap_m,
        occupied,
        join_fences: d.get("join_fences", true)?,
        spacing_m: d.get("fence_spacing_m", 0.45)?,
        tolerance_m: d.get("fence_tolerance_m", 0.06)?,
        min_posts: d.get("fence_min_posts", 3i64)?,
        max_dot_cells: d.get("fence_max_dot_cells", 6i64)?,
        path_clear_radius_m: d.get("path_clear_radius_m", 0.17)?,
    };
    let latched = || QoSProfile::topics_default().keep_last(1).reliable().transient_local();
    let output_topic: Arc<str> = d.get("output_topic", Arc::from("/rtabmap/map_filled"))?;
    let publisher = node.create_publisher::<nav_msgs::msg::OccupancyGrid>((&*output_topic).qos(latched()))?;
    let input_topic: Arc<str> = d.get("input_topic", Arc::from("/rtabmap/map"))?;

    let state = State { filler: GapFiller::new(params), publisher, node: node.clone(), last_throttled: None };
    let worker = node.create_worker::<State>(state);
    let _map = worker.create_subscription::<nav_msgs::msg::OccupancyGrid, _>(
        (&*input_topic).qos(latched()),
        |s: &mut State, m: nav_msgs::msg::OccupancyGrid| s.on_map(m),
    )?;
    let _path = worker.create_subscription::<nav_msgs::msg::Path, _>(
        "/rtabmap/mapPath".qos(QoSProfile::topics_default().keep_last(1).reliable()),
        |s: &mut State, m: nav_msgs::msg::Path| {
            s.filler.on_path(m.poses.iter().map(|p| (p.pose.position.x, p.pose.position.y)).collect());
        },
    )?;
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}

