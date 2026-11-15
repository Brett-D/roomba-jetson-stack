//! vslam_watchdog (Rust): motion sanity filter and recovery supervisor for the visual-inertial
//! odometry (Basalt or cuVSLAM). Drop-in replacement for roomba_perception/vslam_watchdog.py: same
//! topics, parameters, TF, status JSON, service calls and recovery sequence (see that file's
//! docstring). The logic is roomba_rs::watchdog; this file only connects it to ROS.
//!
//! Subscribes: /visual_slam/tracking/odometry (vslam_odom), /visual_slam/status, imu_topic (gyro),
//!             optional wheel odometry (wheel_odom_topic), /tf_static (base_frame <- IMU frame)
//! Publishes:  TF odom -> base_frame and odom -> vslam_odom, /odom, /vslam_filter/status;
//!             with publish_tf false only fused_vio_topic and the status
//! Calls:      /visual_slam/reset, <rtabmap_ns>/pause|resume|trigger_new_map

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::Result;
use isaac_ros_visual_slam_interfaces::{msg::VisualSlamStatus, srv as isaac_srv};
use rclrs::*;
use ros_env::{builtin_interfaces, geometry_msgs, nav_msgs, sensor_msgs, std_msgs, std_srvs, tf2_msgs};

use roomba_rs::static_tf::StaticTree;
use roomba_rs::watchdog::{Io, Level, Odom, ParamSource, Params, Pose, Stamp, Svc, Twist, Watchdog};

/// time.monotonic(): CLOCK_MONOTONIC in seconds.
fn mono() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: clock_gettime only writes the timespec passed to it.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

fn secs(h: &std_msgs::msg::Header) -> f64 {
    h.stamp.sec as f64 + h.stamp.nanosec as f64 * 1e-9
}

fn header(stamp: Stamp, frame: &str) -> std_msgs::msg::Header {
    std_msgs::msg::Header {
        stamp: builtin_interfaces::msg::Time { sec: stamp.sec, nanosec: stamp.nanosec },
        frame_id: frame.into(),
    }
}

fn transform(stamp: Stamp, parent: &str, child: &str, (t, q): Pose) -> geometry_msgs::msg::TransformStamped {
    geometry_msgs::msg::TransformStamped {
        header: header(stamp, parent),
        child_frame_id: child.into(),
        transform: geometry_msgs::msg::Transform {
            translation: geometry_msgs::msg::Vector3 { x: t[0], y: t[1], z: t[2] },
            rotation: geometry_msgs::msg::Quaternion { x: q[0], y: q[1], z: q[2], w: q[3] },
        },
    }
}

/// Declares the node's parameters. The handles are kept for the node's lifetime: rclrs undeclares a
/// parameter when its handle is dropped, and `ros2 param get/list` should show them as for the
/// Python node (which also reads them once at startup).
struct NodeParams<'a> {
    node: &'a Node,
    handles: Vec<Box<dyn std::any::Any>>,
}

impl NodeParams<'_> {
    fn keep<T: ParameterVariant + 'static>(&mut self, p: MandatoryParameter<T>) -> T {
        let value = p.get();
        self.handles.push(Box::new(p));
        value
    }
}

impl ParamSource for NodeParams<'_> {
    fn float(&mut self, name: &str, default: f64) -> Result<f64> {
        let p = self.node.declare_parameter(name).default(default).mandatory()?;
        Ok(self.keep(p))
    }
    fn int(&mut self, name: &str, default: i64) -> Result<i64> {
        let p = self.node.declare_parameter(name).default(default).mandatory()?;
        Ok(self.keep(p))
    }
    fn boolean(&mut self, name: &str, default: bool) -> Result<bool> {
        let p = self.node.declare_parameter(name).default(default).mandatory()?;
        Ok(self.keep(p))
    }
    fn string(&mut self, name: &str, default: &str) -> Result<String> {
        let p = self.node.declare_parameter::<Arc<str>>(name).default(default.into()).mandatory()?;
        Ok(self.keep(p).to_string())
    }
}

/// The watchdog's outputs as ROS publishers, service clients and log lines.
struct RosIo {
    node: Node,
    /// The worker owning [`State`], to hand service responses back to the watchdog.
    worker: Arc<OnceLock<Worker<State>>>,
    reset: Client<isaac_srv::Reset>,
    pause: Client<std_srvs::srv::Empty>,
    resume: Client<std_srvs::srv::Empty>,
    new_map: Client<std_srvs::srv::Empty>,
    pub_tf: Publisher<tf2_msgs::msg::TFMessage>,
    pub_odom: Publisher<nav_msgs::msg::Odometry>,
    pub_status: Publisher<std_msgs::msg::String>,
    odom_frame: String,
    vslam_frame: String,
    base_frame: String,
}

impl RosIo {
    fn empty_client(&self, svc: Svc) -> &Client<std_srvs::srv::Empty> {
        match svc {
            Svc::Pause => &self.pause,
            Svc::Resume => &self.resume,
            _ => &self.new_map,
        }
    }
}

impl Io for RosIo {
    fn service_ready(&self, svc: Svc) -> bool {
        match svc {
            Svc::Reset => self.reset.service_is_ready(),
            _ => self.empty_client(svc).service_is_ready(),
        }
        .unwrap_or(false)
    }

    fn tf(&mut self, stamp: Stamp, base: Pose, anchor: Pose) {
        let transforms = vec![
            transform(stamp, &self.odom_frame, &self.base_frame, base),
            transform(stamp, &self.odom_frame, &self.vslam_frame, anchor),
        ];
        self.pub_tf.publish(tf2_msgs::msg::TFMessage { transforms }).ok();
    }

    fn odom(&mut self, o: Odom) {
        let mut m = nav_msgs::msg::Odometry::default();
        m.header = header(o.stamp, &self.odom_frame);
        m.child_frame_id = self.base_frame.clone();
        m.pose.pose.position = geometry_msgs::msg::Point { x: o.pos[0], y: o.pos[1], z: o.pos[2] };
        m.pose.pose.orientation = geometry_msgs::msg::Quaternion { x: o.quat[0], y: o.quat[1], z: o.quat[2], w: o.quat[3] };
        m.pose.covariance = o.pose_covariance;
        let v = |a: [f64; 3]| geometry_msgs::msg::Vector3 { x: a[0], y: a[1], z: a[2] };
        m.twist.twist.linear = v(o.twist.linear);
        m.twist.twist.angular = v(o.twist.angular);
        m.twist.covariance = o.twist.covariance;
        self.pub_odom.publish(m).ok();
    }

    fn call(&mut self, svc: Svc) {
        let worker = self.worker.get().cloned();
        // The response comes back on the executor; hand it to the worker that owns the watchdog.
        let report = move |error: Option<String>| {
            if let Some(w) = &worker {
                let _ = w.run(move |s: &mut State| s.core.service_done(svc, error, mono(), &mut s.io));
            }
        };
        let done = report.clone();
        let sent = match svc {
            Svc::Reset => self
                .reset
                .call_then(isaac_srv::Reset_Request::default(), move |_: isaac_srv::Reset_Response| done(None)),
            _ => self
                .empty_client(svc)
                .call_then(std_srvs::srv::Empty_Request::default(), move |_: std_srvs::srv::Empty_Response| done(None)),
        };
        if let Err(e) = sent {
            report(Some(e.to_string()));
        }
    }

    fn log(&mut self, level: Level, msg: String) {
        match level {
            Level::Info => log!(self.node.info(), "{msg}"),
            Level::Warn => log!(self.node.warn(), "{msg}"),
            Level::Error => log!(self.node.error(), "{msg}"),
        }
    }
}

struct State {
    core: Watchdog,
    io: RosIo,
    tree: StaticTree,
    base_frame: String,
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("vslam_watchdog")?;
    let mut declared = NodeParams { node: &node, handles: Vec::new() };
    let p = Params::load(&mut declared)?;

    let odom_topic = if p.publish_tf { "/odom".to_string() } else { p.fused_topic.clone() };
    let worker_cell = Arc::new(OnceLock::new());
    let io = RosIo {
        node: node.clone(),
        worker: Arc::clone(&worker_cell),
        reset: node.create_client::<isaac_srv::Reset>(p.service_name(Svc::Reset).as_str())?,
        pause: node.create_client::<std_srvs::srv::Empty>(p.service_name(Svc::Pause).as_str())?,
        resume: node.create_client::<std_srvs::srv::Empty>(p.service_name(Svc::Resume).as_str())?,
        new_map: node.create_client::<std_srvs::srv::Empty>(p.service_name(Svc::NewMap).as_str())?,
        // tf2_ros TransformBroadcaster QoS: reliable, volatile, depth 100.
        pub_tf: node.create_publisher("/tf".qos(QoSProfile::topics_default().keep_last(100).reliable()))?,
        pub_odom: node.create_publisher(odom_topic.as_str().keep_last(20))?,
        pub_status: node.create_publisher("/vslam_filter/status".keep_last(2))?,
        odom_frame: p.odom_frame.clone(),
        vslam_frame: p.vslam_frame.clone(),
        base_frame: p.base_frame.clone(),
    };
    let (imu_topic, wheel_topic) = (p.imu_topic.clone(), p.wheel_topic.clone());
    let state = State { base_frame: p.base_frame.clone(), core: Watchdog::new(p, mono()), io, tree: StaticTree::default() };
    let worker = node.create_worker::<State>(state);
    let _ = worker_cell.set(Arc::clone(&worker));

    let _wheel = if wheel_topic.is_empty() {
        None
    } else {
        Some(worker.create_subscription::<nav_msgs::msg::Odometry, _>(
            wheel_topic.as_str().keep_last(50),
            |s: &mut State, m: nav_msgs::msg::Odometry| {
                let (p, q) = (&m.pose.pose.position, &m.pose.pose.orientation);
                let tw = &m.twist.twist;
                s.core.on_wheel(secs(&m.header), p.x, p.y, [q.x, q.y, q.z, q.w], tw.linear.x, tw.angular.z, mono());
            },
        )?)
    };
    let _status = worker.create_subscription::<VisualSlamStatus, _>(
        "/visual_slam/status".keep_last(10),
        |s: &mut State, m: VisualSlamStatus| s.core.on_status(m.vo_state, mono(), &mut s.io),
    )?;
    let _odom = worker.create_subscription::<nav_msgs::msg::Odometry, _>(
        "/visual_slam/tracking/odometry".keep_last(20),
        |s: &mut State, m: nav_msgs::msg::Odometry| {
            let (p, q) = (&m.pose.pose.position, &m.pose.pose.orientation);
            let tw = &m.twist.twist;
            let odom = Odom {
                stamp: Stamp { sec: m.header.stamp.sec, nanosec: m.header.stamp.nanosec },
                pos: [p.x, p.y, p.z],
                quat: [q.x, q.y, q.z, q.w],
                pose_covariance: m.pose.covariance,
                twist: Twist {
                    linear: [tw.linear.x, tw.linear.y, tw.linear.z],
                    angular: [tw.angular.x, tw.angular.y, tw.angular.z],
                    covariance: m.twist.covariance,
                },
            };
            s.core.on_odom(&odom, mono(), &mut s.io);
        },
    )?;
    let _imu = worker.create_subscription::<sensor_msgs::msg::Imu, _>(
        imu_topic.as_str().sensor_data_qos(),
        |s: &mut State, m: sensor_msgs::msg::Imu| {
            let (w, a) = (&m.angular_velocity, &m.linear_acceleration);
            let State { core, io, tree, base_frame } = s;
            let frame = &m.header.frame_id;
            core.on_imu(secs(&m.header), [w.x, w.y, w.z], [a.x, a.y, a.z], mono(), || tree.rotation(base_frame, frame), io);
        },
    )?;
    // The IMU extrinsic is static (camera driver and the robot's camera mount publish it on
    // /tf_static), so a full tf2 buffer isn't needed.
    let _tf_static = worker.create_subscription::<tf2_msgs::msg::TFMessage, _>(
        "/tf_static".qos(QoSProfile::topics_default().keep_last(100).reliable().transient_local()),
        |s: &mut State, m: tf2_msgs::msg::TFMessage| {
            for t in m.transforms {
                let r = t.transform.rotation;
                s.tree.insert(&t.header.frame_id, &t.child_frame_id, [r.x, r.y, r.z, r.w]);
            }
        },
    )?;
    // Node clock, like rclpy's create_timer (sim time when use_sim_time is set).
    let _tick = worker.create_timer_repeating(Duration::from_millis(100).node_time(), |s: &mut State| {
        s.core.tick(mono(), &mut s.io)
    })?;
    let _status_timer = worker.create_timer_repeating(Duration::from_secs(1).node_time(), |s: &mut State| {
        let data = s.core.status(mono());
        s.io.pub_status.publish(std_msgs::msg::String { data }).ok();
    })?;

    log!(node.info(), "vslam_watchdog (Rust) running");
    executor.spin(SpinOptions::default()).first_error()?;
    Ok(())
}
