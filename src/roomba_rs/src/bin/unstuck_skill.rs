//! unstuck_skill: learned recovery as an action server (docs/learned-skills.md; phase 1, opt-in:
//! roomba_rs built with `--features learned`, not started by any launch file by default).
//!
//! Action /unstuck_skill (roomba_skills_msgs/action/Unstuck): the explorer or nav_coach calls it
//! when Nav2 gives up on a goal (aborted after its recoveries, no progress). At 10 Hz the node
//! builds the policy observation (roomba_rs::learned::unstuck) from the local costmap, the light
//! bumpers / bumpers, wheel odometry, the fused yaw rate and the VSLAM pose, runs the ONNX policy and
//! publishes the command on `cmd_vel_nav`: the same input Nav2's controller and behaviours use, so
//! the velocity smoother, the collision monitor and the driver's bump reflex and cliff stops all
//! stay in the loop. It never publishes outside an active goal.
//!
//! It ends with SUCCEEDED once the local costmap shows >= 0.3 m around the robot for 0.5 s with
//! the bumper and the light bumpers free and the depth camera not blind (nose to a wall the costmap
//! looks empty),
//! TIMED_OUT after min(time_limit_s, max_time_s), FAILED on a cliff / wheel drop /
//! Roomba leaving Safe mode / missing data / model error, CANCELED on request; the robot is stopped
//! in every case. On anything but SUCCEEDED the caller falls back to the classical recovery (Nav2's
//! recoveries, the explorer's blacklist) exactly as without the skill.
//!
//! Subscribes: /local_costmap/costmap, /roomba/sensors, /roomba/mode, /wheel/odom, /odom,
//!             /perception/floor_status (blind depth), /tf(_static)
//! Publishes:  cmd_vel_nav (geometry_msgs/Twist) while a goal runs, ~/status (JSON)

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use futures::channel::oneshot;
use rclrs::*;
use ros_env::{geometry_msgs, nav_msgs, roomba_msgs, roomba_skills_msgs, std_msgs};

use roomba_rs::learned::patch::{lethal_patch, GridView};
use roomba_rs::learned::unstuck::{clearance, light_proximity, Outcome, Session, Tick, UnstuckPolicy};
use roomba_rs::node_params::{mono, NodeParams};
use roomba_rs::tf_buffer::TfBuffer;
use roomba_rs::tf_listener::{HasTfBuffer, WorkerTfListener};
use roomba_rs::watchdog::ParamSource;
use roomba_skills_msgs::action::{Unstuck, Unstuck_Feedback, Unstuck_Result};

const SUCCEEDED: u8 = 0;
const TIMED_OUT: u8 = 1;
const FAILED: u8 = 2;
const CANCELED: u8 = 3;

struct Params {
    model: String,
    max_time_s: f64,
    map_frame: String,
    odom_frame: String,
    base_frame: String,
    light_ref: [f32; 6],
    stale_s: f64,
}

#[derive(Default)]
struct Costmap {
    width: usize,
    height: usize,
    resolution: f64,
    origin: (f64, f64),
    data: Vec<i8>,
    stamp: f64,
}

struct Active {
    session: Session,
    goal: Option<(f64, f64)>,
    done: Option<oneshot::Sender<Unstuck_Result>>,
    feedback: FeedbackPublisher<Unstuck>,
}

struct State {
    p: Params,
    tf: TfBuffer,
    policy: Option<UnstuckPolicy>,
    costmap: Costmap,
    sensors: Option<(roomba_msgs::msg::SensorState, f64)>,
    mode: String,
    /// Last time floor_obstacles reported a blind depth frame.
    blind_at: f64,
    wheel: (f64, f64),
    yaw_rate: f64,
    cmd: Publisher<geometry_msgs::msg::Twist>,
    status: Publisher<std_msgs::msg::String>,
    active: Option<Active>,
    node: Node,
}

impl HasTfBuffer for State {
    fn tf_buffer(&mut self) -> &mut TfBuffer {
        &mut self.tf
    }
}

fn yaw_of(q: [f64; 4]) -> f64 {
    (2.0 * (q[3] * q[2] + q[0] * q[1])).atan2(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
}

impl State {
    fn pose(&self, target: &str) -> Option<(f64, f64, f64)> {
        let t = self.tf.lookup_transform(target, &self.p.base_frame, 0).ok()?;
        Some((t.transform.translation[0], t.transform.translation[1], yaw_of(t.transform.rotation)))
    }

    fn stop(&self) {
        self.cmd.publish(geometry_msgs::msg::Twist::default()).ok();
    }

    fn finish(&mut self, outcome: u8, message: String, clearance_m: f64) {
        let Some(mut a) = self.active.take() else { return };
        self.stop();
        let now = mono();
        let result = Unstuck_Result {
            outcome,
            message: message.clone(),
            elapsed_s: a.session.elapsed(now) as f32,
            clearance_m: clearance_m as f32,
            bumps: a.session.bumps,
        };
        log!(self.node.info(), "unstuck: {message} after {:.1} s, {} bumps", result.elapsed_s, result.bumps);
        self.status
            .publish(std_msgs::msg::String { data: format!("{{\"state\": \"idle\", \"last\": \"{message}\"}}") })
            .ok();
        if let Some(tx) = a.done.take() {
            let _ = tx.send(result);
        }
    }

    /// The 10 Hz control step.
    fn step(&mut self) {
        if self.active.is_none() {
            return;
        }
        let now = mono();
        let Some(policy) = self.policy.as_ref() else {
            self.finish(FAILED, "no policy model loaded".into(), 0.0);
            return;
        };
        let (Some(odom_pose), Some(map_pose)) = (self.pose(&self.p.odom_frame), self.pose(&self.p.map_frame)) else {
            self.finish(FAILED, "no robot pose (TF)".into(), 0.0);
            return;
        };
        let fresh_sensors = self.sensors.as_ref().filter(|s| now - s.1 < self.p.stale_s);
        if self.costmap.data.is_empty() || now - self.costmap.stamp > 2.0 || fresh_sensors.is_none() {
            self.finish(FAILED, "local costmap or Roomba sensors missing / stale".into(), 0.0);
            return;
        }
        let s = &fresh_sensors.unwrap().0;
        let g = GridView {
            width: self.costmap.width,
            height: self.costmap.height,
            resolution: self.costmap.resolution,
            origin_x: self.costmap.origin.0,
            origin_y: self.costmap.origin.1,
            data: &self.costmap.data,
        };
        let patch = lethal_patch(&g, odom_pose.0, odom_pose.1, odom_pose.2);
        let clear = clearance(&g, odom_pose.0, odom_pose.1, 0.6);
        let signals = [
            s.light_bump_left_signal,
            s.light_bump_front_left_signal,
            s.light_bump_center_left_signal,
            s.light_bump_center_right_signal,
            s.light_bump_front_right_signal,
            s.light_bump_right_signal,
        ];
        let mut light = [0f32; 6];
        for k in 0..6 {
            light[k] = light_proximity(signals[k], self.p.light_ref[k]);
        }
        let cliff = s.cliff_left || s.cliff_front_left || s.cliff_front_right || s.cliff_right || s.wheel_drop_left || s.wheel_drop_right;
        let passive = !matches!(self.mode.as_str(), "safe" | "full");
        let a = self.active.as_mut().unwrap();
        let tick = Tick {
            now,
            patch: &patch,
            clearance_m: clear,
            light,
            bump_left: s.bump_left,
            bump_right: s.bump_right,
            v_enc: self.wheel.0,
            w_enc: self.wheel.1,
            yaw_rate: self.yaw_rate,
            pose: map_pose,
            goal: a.goal,
            cliff_or_passive: cliff || passive,
            camera_blind: now - self.blind_at < 0.5,
        };
        match a.session.tick(policy, &tick) {
            Ok((v, w)) => {
                let mut t = geometry_msgs::msg::Twist::default();
                t.linear.x = v;
                t.angular.z = w;
                self.cmd.publish(t).ok();
                let fb = Unstuck_Feedback {
                    elapsed_s: a.session.elapsed(now) as f32,
                    clearance_m: clear as f32,
                    linear_x: v as f32,
                    angular_z: w as f32,
                };
                let _ = a.feedback.publish(fb);
            }
            Err(Outcome::Succeeded) => self.finish(SUCCEEDED, "free".into(), clear),
            Err(Outcome::TimedOut) => self.finish(TIMED_OUT, "time limit reached".into(), clear),
            Err(Outcome::Failed(why)) => self.finish(FAILED, why, clear),
        }
    }
}

fn load_params(node: &Node) -> Result<Params> {
    let mut p = NodeParams::new(node);
    let light: Vec<f32> = p
        .string("light_reference_signals", "60,235,140,140,140,60")?
        .split(',')
        .map(|v| v.trim().parse::<f32>().unwrap_or(0.0))
        .collect();
    let mut light_ref = [0f32; 6];
    for (k, v) in light.iter().take(6).enumerate() {
        light_ref[k] = *v;
    }
    Ok(Params {
        model: p.string("model", "")?,
        max_time_s: p.float("max_time_s", 15.0)?,
        map_frame: p.string("map_frame", "map")?,
        odom_frame: p.string("odom_frame", "odom")?,
        base_frame: p.string("base_frame", "base_link")?,
        light_ref,
        stale_s: p.float("sensor_timeout_s", 1.0)?,
    })
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("unstuck_skill")?;
    let p = load_params(&node)?;
    let policy = match UnstuckPolicy::load(&p.model) {
        Ok(m) => Some(m),
        Err(e) => {
            log!(node.error(), "unstuck policy {} not loaded ({e:#}): every goal will FAIL (classical fallback)", p.model);
            None
        }
    };
    let latched = || QoSProfile::topics_default().keep_last(1).reliable().transient_local();
    let state = State {
        p,
        tf: TfBuffer::new(),
        policy,
        costmap: Costmap::default(),
        sensors: None,
        mode: String::new(),
        blind_at: -1e9,
        wheel: (0.0, 0.0),
        yaw_rate: 0.0,
        cmd: node.create_publisher("cmd_vel_nav".keep_last(1))?,
        status: node.create_publisher("~/status".qos(latched()))?,
        active: None,
        node: node.clone(),
    };
    let worker = node.create_worker::<State>(state);
    let _tf = WorkerTfListener::new(&worker, &node)?;
    let _costmap = worker.create_subscription::<nav_msgs::msg::OccupancyGrid, _>(
        "/local_costmap/costmap".qos(latched()),
        |s: &mut State, m: nav_msgs::msg::OccupancyGrid| {
            s.costmap = Costmap {
                width: m.info.width as usize,
                height: m.info.height as usize,
                resolution: m.info.resolution as f64,
                origin: (m.info.origin.position.x, m.info.origin.position.y),
                data: m.data,
                stamp: mono(),
            };
        },
    )?;
    let _sensors = worker.create_subscription::<roomba_msgs::msg::SensorState, _>(
        "/roomba/sensors".keep_last(10),
        |s: &mut State, m: roomba_msgs::msg::SensorState| s.sensors = Some((m, mono())),
    )?;
    let _mode = worker.create_subscription::<std_msgs::msg::String, _>(
        "/roomba/mode".qos(latched()),
        |s: &mut State, m: std_msgs::msg::String| s.mode = m.data,
    )?;
    let _floor = worker.create_subscription::<std_msgs::msg::String, _>(
        "/perception/floor_status".keep_last(5),
        |s: &mut State, m: std_msgs::msg::String| {
            if m.data.contains("\"blind\": true") {
                s.blind_at = mono();
            }
        },
    )?;
    let _wheel = worker.create_subscription::<nav_msgs::msg::Odometry, _>(
        "/wheel/odom".keep_last(5),
        |s: &mut State, m: nav_msgs::msg::Odometry| s.wheel = (m.twist.twist.linear.x, m.twist.twist.angular.z),
    )?;
    let _odom = worker.create_subscription::<nav_msgs::msg::Odometry, _>(
        "/odom".keep_last(5),
        |s: &mut State, m: nav_msgs::msg::Odometry| s.yaw_rate = m.twist.twist.angular.z,
    )?;
    let _timer = worker.create_timer_repeating(Duration::from_millis(100).node_time(), |s: &mut State| s.step())?;

    let w = Arc::clone(&worker);
    let busy = Arc::new(Mutex::new(false));
    let _server = node.create_action_server::<Unstuck, _>("unstuck_skill", move |requested: RequestedGoal<Unstuck>| {
        let w = Arc::clone(&w);
        let busy = Arc::clone(&busy);
        async move {
            if std::mem::replace(&mut *busy.lock().unwrap(), true) {
                return requested.reject(); // one goal at a time
            }
            let goal = requested.goal().clone();
            let executing = requested.accept().execute();
            let (tx, rx) = oneshot::channel();
            let feedback = executing.feedback_publisher();
            let target = (!goal.goal.header.frame_id.is_empty()).then(|| (goal.goal.pose.position.x, goal.goal.pose.position.y));
            let _ = w.run(move |s: &mut State| {
                let limit = (goal.time_limit_s as f64).clamp(0.5, s.p.max_time_s);
                log!(s.node.info(), "unstuck: started ({limit:.0} s limit)");
                s.status.publish(std_msgs::msg::String { data: "{\"state\": \"running\"}".into() }).ok();
                s.active = Some(Active { session: Session::new(mono(), limit), goal: target, done: Some(tx), feedback });
            });
            let terminated = match executing.unless_cancel_requested(rx).await {
                Ok(Ok(result)) => {
                    if result.outcome == SUCCEEDED {
                        executing.succeeded_with(result)
                    } else {
                        executing.aborted_with(result)
                    }
                }
                Ok(Err(_)) => executing.aborted_with(Unstuck_Result { outcome: FAILED, message: "skill stopped".into(), ..Default::default() }),
                Err(()) => {
                    let (ctx, crx) = oneshot::channel();
                    let _ = w.run(move |s: &mut State| {
                        if let Some(a) = s.active.as_mut() {
                            a.done = Some(ctx);
                        }
                        s.finish(CANCELED, "canceled".into(), 0.0);
                    });
                    let result = crx.await.unwrap_or_default();
                    executing.begin_cancelling().cancelled_with(result)
                }
            };
            *busy.lock().unwrap() = false;
            terminated
        }
    })?;

    log!(node.info(), "unstuck_skill running (action /unstuck_skill)");
    roomba_rs::node_params::spin(&mut executor)?;
    Ok(())
}
