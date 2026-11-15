//! ROS side of [`TfBuffer`]: /tf and /tf_static subscriptions with the QoS of tf2_ros'
//! TransformListener (reliable, keep last 100; /tf_static transient local so late joiners get the
//! static tree), plus message conversions. Two ways to use it:
//!
//! - [`TfListener`] (tf2_ros Buffer + TransformListener): the buffer is shared behind a mutex and
//!   filled by node-level subscriptions; look up from any callback.
//!   ```ignore
//!   let tf = TfListener::new(&node)?;
//!   let t = tf.lookup_transform("map", "base_link", 0)?; // 0 = latest
//!   ```
//! - [`WorkerTfListener`]: the buffer lives in a worker's payload (implement [`HasTfBuffer`]), so the
//!   worker's own callbacks look up without locking and see every transform received before them.
//!   ```ignore
//!   struct State { tf: TfBuffer, .. }
//!   impl HasTfBuffer for State { fn tf_buffer(&mut self) -> &mut TfBuffer { &mut self.tf } }
//!   let _tf = WorkerTfListener::new(&worker, &node)?;   // keep it alive
//!   ```

use std::sync::{Arc, Mutex, MutexGuard};

use rclrs::*;
use ros_env::{builtin_interfaces, geometry_msgs, std_msgs, tf2_msgs};

use crate::tf_buffer::{StampedTransform, TfBuffer, TfError, Transform};

/// The authority tf2_ros' TransformListener records for every transform.
pub const AUTHORITY: &str = "default_authority";

/// /tf: reliable, volatile, keep last 100 (tf2_ros TransformListener / TransformBroadcaster).
pub fn tf_qos() -> QoSProfile {
    QoSProfile::topics_default().keep_last(100).reliable().volatile()
}

/// /tf_static: reliable, transient local, keep last 100 (tf2_ros StaticTransformBroadcaster).
pub fn tf_static_qos() -> QoSProfile {
    QoSProfile::topics_default().keep_last(100).reliable().transient_local()
}

/// Nanoseconds since the epoch of a header stamp (tf2's TimePoint).
pub fn stamp_ns(t: &builtin_interfaces::msg::Time) -> i64 {
    t.sec as i64 * 1_000_000_000 + t.nanosec as i64
}

pub fn ros_time(ns: i64) -> builtin_interfaces::msg::Time {
    builtin_interfaces::msg::Time {
        sec: ns.div_euclid(1_000_000_000) as i32,
        nanosec: ns.rem_euclid(1_000_000_000) as u32,
    }
}

pub fn from_msg(t: &geometry_msgs::msg::Transform) -> Transform {
    let (p, q) = (&t.translation, &t.rotation);
    Transform { translation: [p.x, p.y, p.z], rotation: [q.x, q.y, q.z, q.w] }
}

pub fn to_msg(t: &StampedTransform) -> geometry_msgs::msg::TransformStamped {
    let (p, q) = (t.transform.translation, t.transform.rotation);
    geometry_msgs::msg::TransformStamped {
        header: std_msgs::msg::Header { stamp: ros_time(t.stamp_ns), frame_id: t.frame_id.clone() },
        child_frame_id: t.child_frame_id.clone(),
        transform: geometry_msgs::msg::Transform {
            translation: geometry_msgs::msg::Vector3 { x: p[0], y: p[1], z: p[2] },
            rotation: geometry_msgs::msg::Quaternion { x: q[0], y: q[1], z: q[2], w: q[3] },
        },
    }
}

/// Adds one received transform (what tf2_ros' listener does for each transform of a message).
pub fn add_transform(buffer: &mut TfBuffer, t: &geometry_msgs::msg::TransformStamped, is_static: bool) -> Result<(), String> {
    buffer.set_transform(&t.header.frame_id, &t.child_frame_id, stamp_ns(&t.header.stamp), &from_msg(&t.transform), AUTHORITY, is_static)
}

/// Adds all transforms of a /tf or /tf_static message; returns the reasons for any it ignored.
pub fn add_message(buffer: &mut TfBuffer, msg: &tf2_msgs::msg::TFMessage, is_static: bool) -> Vec<String> {
    msg.transforms.iter().filter_map(|t| add_transform(buffer, t, is_static).err()).collect()
}

/// tf2 prints ignored transforms to the console: old data as a warning, the rest as errors.
fn report(node: &Node, errors: Vec<String>) {
    for e in errors {
        if e.starts_with("TF_OLD_DATA") {
            log!(node.warn(), "{e}");
        } else {
            log!(node.error(), "{e}");
        }
    }
}

/// tf2_ros Buffer + TransformListener: subscriptions on the node's executor filling a shared buffer.
pub struct TfListener {
    buffer: Arc<Mutex<TfBuffer>>,
    _tf: Subscription<tf2_msgs::msg::TFMessage>,
    _tf_static: Subscription<tf2_msgs::msg::TFMessage>,
}

impl TfListener {
    pub fn new(node: &Node) -> Result<Self, RclrsError> {
        Self::with_buffer(node, Arc::new(Mutex::new(TfBuffer::new())))
    }

    pub fn with_buffer(node: &Node, buffer: Arc<Mutex<TfBuffer>>) -> Result<Self, RclrsError> {
        let subscribe = |topic: &str, qos: QoSProfile, is_static: bool| {
            let (buffer, logger) = (Arc::clone(&buffer), node.clone());
            node.create_subscription::<tf2_msgs::msg::TFMessage, _>(topic.qos(qos), move |m: tf2_msgs::msg::TFMessage| {
                let errors = add_message(&mut lock(&buffer), &m, is_static);
                report(&logger, errors);
            })
        };
        let _tf = subscribe("/tf", tf_qos(), false)?;
        let _tf_static = subscribe("/tf_static", tf_static_qos(), true)?;
        Ok(TfListener { buffer, _tf, _tf_static })
    }

    pub fn buffer(&self) -> &Arc<Mutex<TfBuffer>> {
        &self.buffer
    }

    pub fn lock(&self) -> MutexGuard<'_, TfBuffer> {
        lock(&self.buffer)
    }

    /// [`TfBuffer::lookup_transform`]: `source` -> `target` at `time_ns` (0 = latest).
    pub fn lookup_transform(&self, target: &str, source: &str, time_ns: i64) -> Result<StampedTransform, TfError> {
        self.lock().lookup_transform(target, source, time_ns)
    }

    pub fn can_transform(&self, target: &str, source: &str, time_ns: i64) -> Result<(), String> {
        self.lock().can_transform(target, source, time_ns)
    }
}

fn lock(buffer: &Mutex<TfBuffer>) -> MutexGuard<'_, TfBuffer> {
    buffer.lock().unwrap_or_else(|e| e.into_inner())
}

/// A worker payload holding a [`TfBuffer`].
pub trait HasTfBuffer {
    fn tf_buffer(&mut self) -> &mut TfBuffer;
}

impl HasTfBuffer for TfBuffer {
    fn tf_buffer(&mut self) -> &mut TfBuffer {
        self
    }
}

/// /tf and /tf_static subscriptions on a worker, filling the buffer in its payload. Keep it alive.
pub struct WorkerTfListener<S: Send + Sync + 'static> {
    _tf: WorkerSubscription<tf2_msgs::msg::TFMessage, S>,
    _tf_static: WorkerSubscription<tf2_msgs::msg::TFMessage, S>,
}

impl<S: HasTfBuffer + Send + Sync + 'static> WorkerTfListener<S> {
    pub fn new(worker: &Worker<S>, node: &Node) -> Result<Self, RclrsError> {
        let subscribe = |topic: &str, qos: QoSProfile, is_static: bool| {
            let logger = node.clone();
            worker.create_subscription::<tf2_msgs::msg::TFMessage, _>(
                topic.qos(qos),
                move |s: &mut S, m: tf2_msgs::msg::TFMessage| {
                    let errors = add_message(s.tf_buffer(), &m, is_static);
                    report(&logger, errors);
                },
            )
        };
        Ok(WorkerTfListener { _tf: subscribe("/tf", tf_qos(), false)?, _tf_static: subscribe("/tf_static", tf_static_qos(), true)? })
    }
}
