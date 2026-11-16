//! nav2_guard: makes sure Nav2's servers come up even when the lifecycle manager's start-up hangs on
//! a lost change_state answer (see roomba_rs::nav2_guard). Started by nav.launch.py next to
//! lifecycle_manager_navigation with the same node_names.
//!
//! Parameters: node_names (string array), startup_timeout_s (30), call_timeout_s (5),
//!             max_changes (30)
//! Uses:       <node>/get_state, <node>/change_state (lifecycle_msgs)

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::Result;
use rclrs::*;
use ros_env::lifecycle_msgs;

use lifecycle_msgs::srv::{ChangeState, ChangeState_Request, ChangeState_Response, GetState, GetState_Request, GetState_Response};
use roomba_rs::nav2_guard::{state_name, Call, Event, Guard, GuardParams};
use roomba_rs::node_params::mono;

struct State {
    guard: Guard,
    /// startup_timeout_s, for the log.
    timeout: f64,
    get: Vec<Client<GetState>>,
    change: Vec<Client<ChangeState>>,
    node: Node,
    worker: Arc<OnceLock<Worker<State>>>,
}

fn list(v: &[(String, u8)]) -> String {
    v.iter().map(|(n, s)| format!("{n}: {}", state_name(*s))).collect::<Vec<_>>().join(", ")
}

impl State {
    fn tick(&mut self) {
        let (call, events) = self.guard.tick(mono());
        for e in events {
            match e {
                Event::AlreadyActive => log!(self.node.info(), "Nav2 is active"),
                Event::Intervening(v) => log!(
                    self.node.warn(),
                    "Nav2 is not active {:.0} s after start-up ({}): bringing it up (the lifecycle manager's start-up must have hung)",
                    self.timeout,
                    list(&v)
                ),
                Event::BroughtUp => log!(self.node.warn(), "Nav2 is active now (brought up by nav2_guard)"),
                Event::GaveUp(v) => log!(self.node.error(), "Could not bring Nav2 up: {}", list(&v)),
                Event::NoAnswer(n) => log!(self.node.warn(), "{n} did not answer within the call timeout"),
            }
        }
        let worker = Arc::clone(&self.worker);
        match call {
            Some(Call::GetState(i)) => {
                let sent = self.get[i].call_then(GetState_Request::default(), move |r: GetState_Response| {
                    if let Some(w) = worker.get() {
                        let _ = w.run(move |s: &mut State| s.guard.on_state(i, Some(r.current_state.id)));
                    }
                });
                if sent.is_err() {
                    self.guard.on_state(i, None);
                }
            }
            Some(Call::ChangeState(i, transition)) => {
                log!(self.node.info(), "{} {}", if transition == 1 { "Configuring" } else { "Activating" }, self.guard.name(i));
                let mut request = ChangeState_Request::default();
                request.transition.id = transition;
                let sent = self.change[i].call_then(request, move |_: ChangeState_Response| {
                    if let Some(w) = worker.get() {
                        let _ = w.run(move |s: &mut State| s.guard.on_changed(i));
                    }
                });
                if sent.is_err() {
                    self.guard.on_changed(i);
                }
            }
            None => {}
        }
    }
}

fn main() -> Result<()> {
    roomba_rs::default_signals();
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("nav2_guard")?;
    let names: Arc<[Arc<str>]> = node.declare_parameter::<Arc<[Arc<str>]>>("node_names").default(Arc::from([])).mandatory()?.get();
    let startup = node.declare_parameter("startup_timeout_s").default(30.0).mandatory()?;
    let call = node.declare_parameter("call_timeout_s").default(5.0).mandatory()?;
    let max_changes = node.declare_parameter("max_changes").default(30i64).mandatory()?;
    let p = GuardParams { startup_timeout: startup.get(), call_timeout: call.get(), max_changes: max_changes.get().max(0) as u32 };
    let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    let mut get = Vec::new();
    let mut change = Vec::new();
    for n in &names {
        get.push(node.create_client::<GetState>(&format!("/{n}/get_state"))?);
        change.push(node.create_client::<ChangeState>(&format!("/{n}/change_state"))?);
    }
    let worker_cell = Arc::new(OnceLock::new());
    let timeout = p.startup_timeout;
    let state = State { guard: Guard::new(names, p, mono()), timeout, get, change, node: node.clone(), worker: Arc::clone(&worker_cell) };
    let worker = node.create_worker::<State>(state);
    let _ = worker_cell.set(Arc::clone(&worker));
    // One call per tick: at 1 Hz a round of seven get_state calls plus a transition took ~8 s per
    // server (a full bring-up ~2 min, seen in the twin); 10 Hz does it in about one.
    let _tick = worker.create_timer_repeating(Duration::from_millis(100), |s: &mut State| s.tick())?;
    roomba_rs::node_params::spin(&mut executor)?;
    Ok(())
}
