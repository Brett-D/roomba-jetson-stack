//! nav2_guard's logic: makes sure Nav2 comes up. The lifecycle manager configures and activates
//! the Nav2 servers one by one at start-up and waits for every change_state answer without a
//! timeout; when one answer is lost (seen once in ~25 twin launches: controller_server configured,
//! the manager still waiting) it hangs forever, the servers stay inactive and the explorer only
//! says "Navigation is not running". After `startup_timeout` the guard asks every server for its
//! lifecycle state; if any is not active it brings them up itself, in the manager's order (first
//! configure what is unconfigured, then activate what is inactive), one transition at a time,
//! re-reading the states after each and giving up on answers that take longer than `call_timeout`.
//! Once all are active it is done (bonds are not formed then: a server that dies later isn't
//! noticed by the hung manager).
//!
//! ROS-free: [`Guard::tick`] returns the service calls to make; their answers come back through
//! [`Guard::on_state`] / [`Guard::on_changed`] (the node in src/bin/nav2_guard.rs).

/// lifecycle_msgs/State ids.
pub const UNCONFIGURED: u8 = 1;
pub const INACTIVE: u8 = 2;
pub const ACTIVE: u8 = 3;
/// lifecycle_msgs/Transition ids.
pub const CONFIGURE: u8 = 1;
pub const ACTIVATE: u8 = 3;

#[derive(Clone, Debug, PartialEq)]
pub enum Call {
    /// `<node>/get_state` of node `i`.
    GetState(usize),
    /// `<node>/change_state` of node `i` with this transition.
    ChangeState(usize, u8),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Everything was active at the first check: nothing done.
    AlreadyActive,
    /// Not all active after the start-up timeout: (node, state) of those that weren't.
    Intervening(Vec<(String, u8)>),
    /// Brought up by the guard.
    BroughtUp,
    /// Still not active after `max_changes` transitions.
    GaveUp(Vec<(String, u8)>),
    /// A call's answer didn't come within `call_timeout`.
    NoAnswer(String),
}

#[derive(Clone, Debug)]
pub struct GuardParams {
    pub startup_timeout: f64,
    pub call_timeout: f64,
    pub max_changes: u32,
}

pub struct Guard {
    names: Vec<String>,
    p: GuardParams,
    start: f64,
    states: Vec<Option<u8>>,
    /// The call waiting for an answer and when it was made.
    pending: Option<(Call, f64)>,
    /// Next node to ask for its state in this round (== names.len(): round complete).
    query: usize,
    changes: u32,
    intervened: bool,
    done: bool,
}

pub fn state_name(s: u8) -> &'static str {
    match s {
        UNCONFIGURED => "unconfigured",
        INACTIVE => "inactive",
        ACTIVE => "active",
        4 => "finalized",
        10 => "configuring",
        11 => "cleaning up",
        12 => "shutting down",
        13 => "activating",
        14 => "deactivating",
        15 => "error processing",
        _ => "unknown",
    }
}

impl Guard {
    pub fn new(names: Vec<String>, p: GuardParams, now: f64) -> Self {
        let n = names.len();
        Guard { names, p, start: now, states: vec![None; n], pending: None, query: 0, changes: 0, intervened: false, done: n == 0 }
    }

    pub fn done(&self) -> bool {
        self.done
    }

    pub fn name(&self, i: usize) -> &str {
        &self.names[i]
    }

    /// The periodic timer (10 Hz): the next call to make, if any, and what happened.
    pub fn tick(&mut self, now: f64) -> (Option<Call>, Vec<Event>) {
        let mut events = Vec::new();
        if self.done || now - self.start < self.p.startup_timeout {
            return (None, events);
        }
        if let Some((call, sent)) = &self.pending {
            if now - sent <= self.p.call_timeout {
                return (None, events);
            }
            let i = match call {
                Call::GetState(i) | Call::ChangeState(i, _) => *i,
            };
            events.push(Event::NoAnswer(self.names[i].clone()));
            self.pending = None;
            self.query = 0; // read the states again
        }
        if self.query < self.names.len() {
            let call = Call::GetState(self.query);
            self.pending = Some((call.clone(), now));
            return (Some(call), events);
        }
        // A full round of states: all active, or the next transition.
        let not_active: Vec<(String, u8)> = self
            .states
            .iter()
            .enumerate()
            .filter(|(_, s)| **s != Some(ACTIVE))
            .map(|(i, s)| (self.names[i].clone(), s.unwrap_or(0)))
            .collect();
        if not_active.is_empty() {
            self.done = true;
            events.push(if self.intervened { Event::BroughtUp } else { Event::AlreadyActive });
            return (None, events);
        }
        if !self.intervened {
            self.intervened = true;
            events.push(Event::Intervening(not_active.clone()));
        }
        if self.changes >= self.p.max_changes {
            self.done = true;
            events.push(Event::GaveUp(not_active));
            return (None, events);
        }
        let first = |state: u8| self.states.iter().position(|s| *s == Some(state));
        let call = match (first(UNCONFIGURED), first(INACTIVE)) {
            (Some(i), _) => Call::ChangeState(i, CONFIGURE),
            (None, Some(i)) => Call::ChangeState(i, ACTIVATE),
            // Only transitional or unknown states: wait for them to settle and read them again.
            (None, None) => {
                self.query = 0;
                self.changes += 1;
                return (None, events);
            }
        };
        self.changes += 1;
        self.pending = Some((call.clone(), now));
        (Some(call), events)
    }

    /// Answer to GetState(i) (None: the call failed).
    pub fn on_state(&mut self, i: usize, state: Option<u8>) {
        if !matches!(self.pending, Some((Call::GetState(j), _)) if j == i) {
            return; // given up on already
        }
        self.pending = None;
        self.states[i] = state;
        self.query = i + 1;
    }

    /// Answer to ChangeState(i, _) (success or not): read all states again.
    pub fn on_changed(&mut self, i: usize) {
        if !matches!(self.pending, Some((Call::ChangeState(j, _), _)) if j == i) {
            return;
        }
        self.pending = None;
        self.query = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> Guard {
        let names = ["controller_server", "planner_server", "bt_navigator"].iter().map(|s| s.to_string()).collect();
        Guard::new(names, GuardParams { startup_timeout: 30.0, call_timeout: 5.0, max_changes: 20 }, 0.0)
    }

    /// Ticks once a second from `t`, answering with the simulated servers' states; returns the
    /// transitions made and the events.
    fn run(g: &mut Guard, states: &mut [u8], t: &mut f64, lose: &mut Vec<usize>) -> (Vec<(usize, u8)>, Vec<Event>) {
        let (mut changes, mut events) = (Vec::new(), Vec::new());
        for _ in 0..100 {
            let (call, ev) = g.tick(*t);
            events.extend(ev);
            match call {
                Some(Call::GetState(i)) => g.on_state(i, Some(states[i])),
                Some(Call::ChangeState(i, tr)) => {
                    changes.push((i, tr));
                    states[i] = if tr == CONFIGURE { INACTIVE } else { ACTIVE };
                    if lose.contains(&i) {
                        lose.retain(|&j| j != i); // done, but the answer is lost
                    } else {
                        g.on_changed(i);
                    }
                }
                None => {}
            }
            if g.done() {
                break;
            }
            *t += 1.0;
        }
        (changes, events)
    }

    #[test]
    fn leaves_a_working_start_up_alone() {
        let mut g = guard();
        assert_eq!(g.tick(29.0), (None, vec![]), "waits startup_timeout");
        let (mut states, mut t) = ([ACTIVE; 3], 30.0);
        let (changes, events) = run(&mut g, &mut states, &mut t, &mut vec![]);
        assert!(changes.is_empty());
        assert_eq!(events, vec![Event::AlreadyActive]);
    }

    #[test]
    fn brings_up_what_a_hung_manager_left() {
        // The manager configured controller_server but never got the answer: the rest untouched.
        let mut g = guard();
        let (mut states, mut t) = ([INACTIVE, UNCONFIGURED, UNCONFIGURED], 30.0);
        let (changes, events) = run(&mut g, &mut states, &mut t, &mut vec![2]);
        assert_eq!(changes, vec![(1, CONFIGURE), (2, CONFIGURE), (0, ACTIVATE), (1, ACTIVATE), (2, ACTIVATE)]);
        assert_eq!(states, [ACTIVE; 3]);
        assert!(matches!(&events[0], Event::Intervening(v) if v.len() == 3));
        assert!(events.contains(&Event::NoAnswer("bt_navigator".into())), "its own lost answer is given up on: {events:?}");
        assert_eq!(events.last(), Some(&Event::BroughtUp));
    }

    #[test]
    fn gives_up_on_a_server_stuck_in_a_transition() {
        let mut g = guard();
        let (mut states, mut t) = ([ACTIVE, 10, ACTIVE], 30.0);
        let (changes, events) = run(&mut g, &mut states, &mut t, &mut vec![]);
        assert!(changes.is_empty());
        assert!(matches!(events.last(), Some(Event::GaveUp(v)) if v == &vec![("planner_server".to_string(), 10)]), "{events:?}");
    }
}
