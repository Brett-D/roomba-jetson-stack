//! explorer (Rust): frontier exploration - drives the robot to the edge of the known map until
//! nothing is left, then back to where it started. Drop-in replacement for
//! roomba_perception/explorer.py: same node name, parameters, topics and QoS, status JSON, keep-out
//! commands, log lines, NavigateToPose goals and controller parameter requests (see that file's
//! docstring). The logic is roomba_rs::explorer; this file only connects it to ROS.
//!
//! Subscribes: /global_costmap/costmap, /roomba/mode (latched), /battery_state, /roomba/sensors,
//!             /vslam_filter/status, ~/command ("start" | "stop" | "home"), /tf, /tf_static
//! Publishes:  ~/status (JSON, transient local), /perception/ignore_zones/command, ~/markers (how it
//!             decides where to go; transient local, only while someone listens)
//! Uses:       navigate_to_pose action, /controller_server/get_parameters|set_parameters

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::Result;
use rclrs::*;
use ros_env::{builtin_interfaces, geometry_msgs, nav2_msgs, nav_msgs, rcl_interfaces, roomba_msgs, sensor_msgs, std_msgs, visualization_msgs};

use nav2_msgs::action::{NavigateToPose, Spin};
use rcl_interfaces::msg::{Parameter, ParameterValue};
use rcl_interfaces::srv::{GetParameters, GetParameters_Request, GetParameters_Response, SetParameters, SetParameters_Request, SetParameters_Response};
use roomba_rs::explorer::{Explorer, GoalRequest, Grid, Io, Params, Sensors, PARAMETER_DOUBLE, ROTATE_PARAM};
use roomba_rs::node_params::{block_signals, on_signal, NodeParams};
use roomba_rs::tf_buffer::TfBuffer;
use roomba_rs::tf_listener::{ros_time, HasTfBuffer, WorkerTfListener};
use roomba_rs::watchdog::Level;
use roomba_rs::zones::{MarkerSpec, DELETEALL};

struct RosIo {
    node: Node,
    /// The worker owning [`State`], to hand responses back to the explorer.
    worker: Arc<OnceLock<Worker<State>>>,
    tf: TfBuffer,
    nav: ActionClient<NavigateToPose>,
    /// Accepted goals that have no result yet, by sequence number.
    goals: HashMap<u64, CancellationClient<NavigateToPose>>,
    spinner: ActionClient<Spin>,
    /// Accepted look-around turns that have no result yet, by sequence number.
    spins: HashMap<u64, CancellationClient<Spin>>,
    get_params: Client<GetParameters>,
    set_params: Client<SetParameters>,
    pub_status: Publisher<std_msgs::msg::String>,
    pub_zone: Publisher<std_msgs::msg::String>,
    pub_markers: Publisher<visualization_msgs::msg::MarkerArray>,
    cliff_timers: Vec<WorkerTimer<State>>,
}

impl RosIo {
    fn on_worker(&self, f: impl FnOnce(&mut State) + Send + 'static) {
        if let Some(w) = self.worker.get() {
            let _ = w.run(f);
        }
    }
}

impl Io for RosIo {
    fn now(&mut self) -> f64 {
        self.node.get_clock().now().nsec as f64 * 1e-9
    }

    fn lookup(&mut self, target: &str, source: &str) -> Option<([f64; 3], [f64; 4])> {
        let t = self.tf.lookup_transform(target, source, 0).ok()?;
        Some((t.transform.translation, t.transform.rotation))
    }

    fn nav_ready(&mut self) -> bool {
        self.nav.server_is_available().unwrap_or(false)
    }

    fn send_goal(&mut self, g: GoalRequest) {
        let mut goal = nav2_msgs::action::NavigateToPose_Goal::default();
        goal.pose.header.frame_id = g.frame;
        goal.pose.header.stamp = ros_time(self.node.get_clock().now().nsec);
        goal.pose.pose.position.x = g.x;
        goal.pose.pose.position.y = g.y;
        goal.pose.pose.orientation.z = g.qz;
        goal.pose.pose.orientation.w = g.qw;
        let seq = g.seq;
        let requested = match self.nav.try_request_goal(goal) {
            Ok(r) => r,
            Err(e) => {
                // rclpy would raise here; the explorer gives up on it like on any request Nav2
                // never answers (goal_response_timeout_s).
                log!(self.node.error(), "Could not send the navigation goal: {e}");
                return;
            }
        };
        let worker = self.worker.get().cloned();
        let _ = self.node.commands().run(async move {
            let Some(worker) = worker else { return };
            match requested.await {
                None => {
                    let _ = worker.run(move |s: &mut State| s.core.on_goal_response(seq, false, &mut s.io));
                }
                Some(client) => {
                    let GoalClient { result, cancellation, .. } = client;
                    let _ = worker.run(move |s: &mut State| {
                        s.io.goals.insert(seq, cancellation);
                        s.core.on_goal_response(seq, true, &mut s.io);
                    });
                    let (status, _) = result.await;
                    let _ = worker.run(move |s: &mut State| {
                        s.io.goals.remove(&seq);
                        s.core.on_result(seq, status as i8, &mut s.io);
                    });
                }
            }
        });
    }

    fn cancel_goal(&mut self, seq: u64) {
        if let Some(c) = self.goals.get(&seq) {
            if let Err(e) = c.try_cancel() {
                log!(self.node.warn(), "Could not cancel the navigation goal: {e}");
            }
        }
    }

    fn get_params_ready(&mut self) -> bool {
        self.get_params.service_is_ready().unwrap_or(false)
    }

    fn set_params_ready(&mut self) -> bool {
        self.set_params.service_is_ready().unwrap_or(false)
    }

    fn get_rotate(&mut self) {
        let worker = Arc::clone(&self.worker);
        let request = GetParameters_Request { names: vec![ROTATE_PARAM.to_string()] };
        let sent = self.get_params.call_then(request, move |r: GetParameters_Response| {
            if let Some(w) = worker.get() {
                let value = r.values.first().map(|v| (v.type_, v.double_value));
                let _ = w.run(move |s: &mut State| s.core.on_saved_rotate(Ok(value), &mut s.io));
            }
        });
        if let Err(e) = sent {
            let msg = e.to_string();
            self.on_worker(move |s: &mut State| s.core.on_saved_rotate(Err(msg), &mut s.io));
        }
    }

    fn set_rotate(&mut self, value: f64) {
        let param = Parameter {
            name: ROTATE_PARAM.to_string(),
            value: ParameterValue { type_: PARAMETER_DOUBLE, double_value: value, ..Default::default() },
        };
        let request = SetParameters_Request { parameters: vec![param] };
        if let Err(e) = self.set_params.call_then(request, |_: SetParameters_Response| {}) {
            log!(self.node.warn(), "Could not set {ROTATE_PARAM}: {e}");
        }
    }

    fn status(&mut self, data: String) {
        self.pub_status.publish(std_msgs::msg::String { data }).ok();
    }

    fn zone_command(&mut self, data: String) {
        self.pub_zone.publish(std_msgs::msg::String { data }).ok();
    }

    fn log(&mut self, level: Level, msg: String) {
        match level {
            Level::Info => log!(self.node.info(), "{msg}"),
            Level::Warn => log!(self.node.warn(), "{msg}"),
            Level::Error => log!(self.node.error(), "{msg}"),
        }
    }

    fn local_hms(&mut self) -> String {
        // SAFETY: time() with a null pointer only returns the time; localtime_r writes `tm`.
        let tm = unsafe {
            let t = libc::time(std::ptr::null_mut());
            let mut tm: libc::tm = std::mem::zeroed();
            libc::localtime_r(&t, &mut tm);
            tm
        };
        format!("{:02}{:02}{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
    }

    fn start_cliff_timer(&mut self) {
        let Some(worker) = self.worker.get() else { return };
        // One-shot timers fire 0.6 s after the mode change: only the latest few can be pending.
        if self.cliff_timers.len() >= 4 {
            self.cliff_timers.remove(0);
        }
        match worker.create_timer_oneshot(Duration::from_millis(600).steady_time(), |s: &mut State| s.core.cliff_check(&mut s.io)) {
            Ok(t) => self.cliff_timers.push(t),
            Err(e) => log!(self.node.error(), "Could not start the cliff check timer: {e}"),
        }
    }

    fn spin_ready(&mut self) -> bool {
        self.spinner.server_is_available().unwrap_or(false)
    }

    fn send_spin(&mut self, seq: u64, target_yaw: f32, sec: i32, nanosec: u32) {
        let mut goal = nav2_msgs::action::Spin_Goal::default();
        goal.target_yaw = target_yaw;
        goal.time_allowance.sec = sec;
        goal.time_allowance.nanosec = nanosec;
        let requested = match self.spinner.try_request_goal(goal) {
            Ok(r) => r,
            Err(e) => {
                // Given up on like a turn request that is never answered (goal_response_timeout_s).
                log!(self.node.error(), "Could not send the look-around turn: {e}");
                return;
            }
        };
        let worker = self.worker.get().cloned();
        let _ = self.node.commands().run(async move {
            let Some(worker) = worker else { return };
            match requested.await {
                None => {
                    let _ = worker.run(move |s: &mut State| s.core.on_spin_response(seq, false, &mut s.io));
                }
                Some(client) => {
                    let GoalClient { result, cancellation, .. } = client;
                    let _ = worker.run(move |s: &mut State| {
                        s.io.spins.insert(seq, cancellation);
                        s.core.on_spin_response(seq, true, &mut s.io);
                    });
                    let (status, _) = result.await;
                    let _ = worker.run(move |s: &mut State| {
                        s.io.spins.remove(&seq);
                        s.core.on_spin_result(seq, status as i8, &mut s.io);
                    });
                }
            }
        });
    }

    fn cancel_spin(&mut self, seq: u64) {
        if let Some(c) = self.spins.get(&seq) {
            if let Err(e) = c.try_cancel() {
                log!(self.node.warn(), "Could not cancel the look-around turn: {e}");
            }
        }
    }

    fn markers_wanted(&mut self) -> bool {
        self.pub_markers.get_subscription_count().unwrap_or(0) > 0
    }

    fn markers(&mut self, frame: &str, specs: &[MarkerSpec]) {
        let ns = self.node.get_clock().now().nsec;
        let header = std_msgs::msg::Header {
            stamp: builtin_interfaces::msg::Time { sec: ns.div_euclid(1_000_000_000) as i32, nanosec: ns.rem_euclid(1_000_000_000) as u32 },
            frame_id: frame.into(),
        };
        let mut clear = visualization_msgs::msg::Marker { header: header.clone(), action: DELETEALL, ..Default::default() };
        clear.pose.orientation.w = 1.0;
        let markers = std::iter::once(clear).chain(specs.iter().map(|m| marker(m, &header))).collect();
        self.pub_markers.publish(visualization_msgs::msg::MarkerArray { markers }).ok();
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

struct State {
    core: Explorer,
    io: RosIo,
}

impl HasTfBuffer for State {
    fn tf_buffer(&mut self) -> &mut TfBuffer {
        &mut self.io.tf
    }
}

fn sensors(m: &roomba_msgs::msg::SensorState) -> Sensors {
    Sensors {
        bump_left: m.bump_left,
        bump_right: m.bump_right,
        wheel_drop_left: m.wheel_drop_left,
        wheel_drop_right: m.wheel_drop_right,
        cliff: [m.cliff_left, m.cliff_front_left, m.cliff_front_right, m.cliff_right],
        light: [
            m.light_bump_left_signal,
            m.light_bump_front_left_signal,
            m.light_bump_center_left_signal,
            m.light_bump_center_right_signal,
            m.light_bump_front_right_signal,
            m.light_bump_right_signal,
        ],
    }
}

/// SIGINT / SIGTERM: cancel the goal being driven to (as the Python node does on Ctrl-C), then exit.
fn cancel_on_signal(set: libc::sigset_t, worker: Worker<State>) {
    on_signal(set, move || {
        let mut done = worker.run(|s: &mut State| s.core.shutdown(&mut s.io));
        for _ in 0..50 {
            if !matches!(done.try_recv(), Ok(None)) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        std::thread::sleep(Duration::from_millis(200)); // let a cancel request go out
    });
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let signals = block_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("explorer")?;
    let mut declared = NodeParams::new(&node);
    let params = Params::load(&mut declared)?;

    let latched = || QoSProfile::topics_default().keep_last(1).reliable().transient_local();
    let worker_cell = Arc::new(OnceLock::new());
    let mut io = RosIo {
        node: node.clone(),
        worker: Arc::clone(&worker_cell),
        tf: TfBuffer::new(),
        nav: node.create_action_client::<NavigateToPose>("navigate_to_pose")?,
        goals: HashMap::new(),
        spinner: node.create_action_client::<Spin>("spin")?,
        spins: HashMap::new(),
        get_params: node.create_client::<GetParameters>("/controller_server/get_parameters")?,
        set_params: node.create_client::<SetParameters>("/controller_server/set_parameters")?,
        pub_status: node.create_publisher("~/status".qos(latched()))?,
        pub_zone: node.create_publisher("/perception/ignore_zones/command".keep_last(10))?,
        pub_markers: node.create_publisher("~/markers".qos(latched()))?,
        cliff_timers: Vec::new(),
    };
    let core = Explorer::new(params, &mut io);
    let wants_sensors = core.wants_sensors();
    let worker = node.create_worker::<State>(State { core, io });
    let _ = worker_cell.set(Arc::clone(&worker));
    cancel_on_signal(signals, Arc::clone(&worker));

    let _tf = WorkerTfListener::new(&worker, &node)?;
    let _grid = worker.create_subscription::<nav_msgs::msg::OccupancyGrid, _>(
        "/global_costmap/costmap".qos(latched()),
        |s: &mut State, m: nav_msgs::msg::OccupancyGrid| {
            s.core.on_grid(Grid {
                width: m.info.width,
                height: m.info.height,
                resolution: m.info.resolution,
                origin_x: m.info.origin.position.x,
                origin_y: m.info.origin.position.y,
                data: m.data,
            })
        },
    )?;
    let _mode = worker.create_subscription::<std_msgs::msg::String, _>(
        "/roomba/mode".qos(latched()),
        |s: &mut State, m: std_msgs::msg::String| s.core.on_mode(&m.data, &mut s.io),
    )?;
    let _battery = worker.create_subscription::<sensor_msgs::msg::BatteryState, _>(
        "/battery_state".keep_last(2),
        |s: &mut State, m: sensor_msgs::msg::BatteryState| s.core.on_battery(m.percentage),
    )?;
    let _filter = worker.create_subscription::<std_msgs::msg::String, _>(
        "/vslam_filter/status".keep_last(2),
        |s: &mut State, m: std_msgs::msg::String| s.core.on_filter(&m.data),
    )?;
    let _command = worker.create_subscription::<std_msgs::msg::String, _>(
        "~/command".keep_last(10),
        |s: &mut State, m: std_msgs::msg::String| s.core.on_command(&m.data, &mut s.io),
    )?;
    let _sensors = if wants_sensors {
        Some(worker.create_subscription::<roomba_msgs::msg::SensorState, _>(
            "/roomba/sensors".keep_last(10),
            |s: &mut State, m: roomba_msgs::msg::SensorState| s.core.on_sensors(&sensors(&m), &mut s.io),
        )?)
    } else {
        None
    };
    // Steady-clock timers: with a ROS-time (system clock) timer, a wait that ends for it just as the
    // clock is stepped back (WSL's time sync every ~30 s, an NTP step) makes rcl_wait report a
    // timeout; rclrs 0.8 then stops this worker's wait set for good while the node's other wait set
    // keeps the executor spinning, so the explorer froze (no ticks, no Nav2 results). Its timeouts
    // still run on the node clock (Io::now), so sim time is respected.
    let _tick = worker.create_timer_repeating(Duration::from_secs(1).steady_time(), |s: &mut State| s.core.tick(&mut s.io))?;

    log!(node.info(), "explorer (Rust) running");
    roomba_rs::node_params::spin(&mut executor)?;
    Ok(())
}
