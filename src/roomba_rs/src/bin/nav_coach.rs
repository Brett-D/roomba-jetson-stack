//! nav_coach (Rust): turns Nav2's velocity commands into instructions a person can follow, or with
//! robot_mode into a description of what the robot is doing. Drop-in replacement for
//! roomba_perception/nav_coach.py: same node name, parameters, topics and instruction JSON (see that
//! file's docstring). The logic is roomba_rs::nav_coach; this file only connects it to ROS.
//!
//! Subscribes: /cmd_vel_nav, /cmd_vel, /collision_monitor_state, /perception/obstacles (with TF to
//!             base_frame), /navigate_to_pose/_action/status|feedback
//! Publishes:  /nav_coach/instruction (std_msgs/String JSON, 5 Hz)

use std::time::Duration;

use anyhow::Result;
use rclrs::*;
use ros_env::{action_msgs, geometry_msgs, nav2_msgs, sensor_msgs, std_msgs};

use roomba_rs::nav_coach::{NavCoach, Params};
use roomba_rs::node_params::{block_signals, mono, on_signal, NodeParams};
use roomba_rs::tf_buffer::TfBuffer;
use roomba_rs::tf_listener::{HasTfBuffer, WorkerTfListener};

struct State {
    core: NavCoach,
    tf: TfBuffer,
    publisher: Publisher<std_msgs::msg::String>,
}

impl HasTfBuffer for State {
    fn tf_buffer(&mut self) -> &mut TfBuffer {
        &mut self.tf
    }
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    // Ctrl-C / launch shutdown: exit cleanly like the Python node.
    on_signal(block_signals(), || {});
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("nav_coach")?;
    let mut declared = NodeParams::new(&node);
    let params = Params::load(&mut declared)?;

    let publisher = node.create_publisher("/nav_coach/instruction".keep_last(10))?;
    let worker = node.create_worker::<State>(State { core: NavCoach::new(params), tf: TfBuffer::new(), publisher });
    let _tf = WorkerTfListener::new(&worker, &node)?;
    let _cmd_nav = worker.create_subscription::<geometry_msgs::msg::Twist, _>(
        "/cmd_vel_nav".keep_last(10),
        |s: &mut State, m: geometry_msgs::msg::Twist| s.core.on_cmd_nav(m.linear.x, m.angular.z, mono()),
    )?;
    let _cmd_out = worker.create_subscription::<geometry_msgs::msg::Twist, _>(
        "/cmd_vel".keep_last(10),
        |s: &mut State, m: geometry_msgs::msg::Twist| s.core.on_cmd_out(m.linear.x, m.angular.z, mono()),
    )?;
    let _cm = worker.create_subscription::<nav2_msgs::msg::CollisionMonitorState, _>(
        "/collision_monitor_state".keep_last(10),
        |s: &mut State, m: nav2_msgs::msg::CollisionMonitorState| s.core.on_collision_monitor(m.action_type, mono()),
    )?;
    // Obstacles in the stop zone directly ahead (the controller may simply stop without
    // commanding anything, so the collision monitor alone would stay silent).
    let _obstacles = worker.create_subscription::<sensor_msgs::msg::PointCloud2, _>(
        "/perception/obstacles".keep_last(2),
        |s: &mut State, m: sensor_msgs::msg::PointCloud2| {
            let frame = s.core.params().base_frame.clone();
            let tf = s.tf.lookup_transform(&frame, &m.header.frame_id, 0).ok();
            let tf = tf.map(|t| (t.transform.translation, t.transform.rotation));
            s.core.on_obstacles(&m.data, m.width, tf, mono());
        },
    )?;
    let _status = worker.create_subscription::<action_msgs::msg::GoalStatusArray, _>(
        "/navigate_to_pose/_action/status".keep_last(10),
        |s: &mut State, m: action_msgs::msg::GoalStatusArray| {
            let list: Vec<(i32, u32, i8)> =
                m.status_list.iter().map(|g| (g.goal_info.stamp.sec, g.goal_info.stamp.nanosec, g.status)).collect();
            s.core.on_status(&list, mono());
        },
    )?;
    let _feedback = worker.create_subscription::<nav2_msgs::action::NavigateToPose_FeedbackMessage, _>(
        "/navigate_to_pose/_action/feedback".keep_last(10),
        |s: &mut State, m: nav2_msgs::action::NavigateToPose_FeedbackMessage| s.core.on_feedback(m.feedback.distance_remaining),
    )?;
    // Steady-clock timers (the core gets the time it uses): a ROS-time timer whose wait ends just as
    // the system clock is stepped back makes rcl_wait report a timeout, and rclrs 0.8 then stops this
    // worker's wait set for good while the node's own one keeps spinning (the worker froze).
    let _tick = worker.create_timer_repeating(Duration::from_millis(200).steady_time(), |s: &mut State| {
        let data = s.core.tick(mono());
        s.publisher.publish(std_msgs::msg::String { data }).ok();
    })?;

    log!(node.info(), "nav_coach (Rust) running");
    roomba_rs::node_params::spin(&mut executor)?;
    Ok(())
}
