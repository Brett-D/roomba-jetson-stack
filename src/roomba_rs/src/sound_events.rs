//! sound_events' logic: event sounds on the Roomba's beeper.
//!
//! - **hello** (song 0) and HELO on the display, once, when the driver first reports Safe or Full
//!   mode after the node started;
//! - **uh-oh** (song 1) on a rising edge of the physical bumper (left or right; the light bumpers
//!   alone never), at most one per `bump_interval` and never over a song that is playing;
//! - **done** (song 2), a short jingle, when an exploration run ends with "finished" or "back at the
//!   start point": once per run (the explorer repeats its status every second).
//!
//! Nothing plays outside Safe/Full mode, and the node never changes the mode (the driver re-enters
//! Safe mode for display commands outside it, so the display is only written in Safe/Full too).
//! Muted while the mute file exists: checked at each event, so it can be toggled live.
//! Other nodes (the web control page) define and play songs in the same five slots, so nothing
//! here relies on a song defined earlier: every sound is one define_song call with the full notes
//! and `play = true`.
//!
//! ROS-free: the node (src/bin/sound_events.rs) feeds in /roomba/sensors, /explorer/status and a
//! steady-clock tick, and carries out the returned [`Action`]s.

/// OI modes (roomba_msgs/SensorState.oi_mode).
pub const MODE_SAFE: u8 = 2;
pub const MODE_FULL: u8 = 3;

/// The display text shown with hello, and the text that clears it.
pub const HELLO_TEXT: &str = "HELO";
pub const CLEAR_TEXT: &str = "    ";

/// A Roomba song (Open Interface opcode 140): MIDI notes 31-127 (anything else is a rest) and
/// durations in 1/64 s, at most 16 notes.
#[derive(Debug, PartialEq)]
pub struct Song {
    pub number: u8,
    pub name: &'static str,
    pub notes: &'static [u8],
    pub durations: &'static [u8],
}

impl Song {
    /// Length in seconds.
    pub fn seconds(&self) -> f64 {
        self.durations.iter().map(|&d| d as f64).sum::<f64>() / 64.0
    }
}

/// "hel-lo": G4 C5 E5 (0.69 s).
pub const HELLO: Song = Song { number: 0, name: "hello", notes: &[67, 72, 76], durations: &[10, 10, 24] };
/// "uh-oh": E5 E5 A4 (0.72 s).
pub const UH_OH: Song = Song { number: 1, name: "uh-oh", notes: &[76, 76, 69], durations: &[12, 6, 28] };
/// "ta-da": C5 E5 G5 C6 up the major chord, a breath, then G5 and a long C6 (1.30 s).
pub const DONE: Song = Song { number: 2, name: "done", notes: &[72, 76, 79, 84, 0, 79, 84], durations: &[7, 7, 7, 14, 4, 8, 36] };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Hello,
    Bump,
    Done,
}

impl Event {
    pub fn song(self) -> &'static Song {
        match self {
            Event::Hello => &HELLO,
            Event::Bump => &UH_OH,
            Event::Done => &DONE,
        }
    }

    pub fn what(self) -> &'static str {
        match self {
            Event::Hello => "driver in Safe/Full mode",
            Event::Bump => "bumper hit",
            Event::Done => "exploration finished",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// The OI mode isn't Safe or Full (None: no recent sensor data).
    Mode(Option<u8>),
    /// The driver's define_song service isn't available.
    NoDriver,
    /// The mute file exists.
    Muted,
    /// A song is playing, or one was just requested.
    Busy,
    /// Less than `bump_interval` after the last uh-oh.
    RateLimited,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Play the event's song: one define_song call with its notes and `play = true`.
    Play(Event),
    /// Show this text on the 4-digit display (/roomba/display).
    Display(&'static str),
    /// The event's song was not played, and why.
    Skipped(Event, Skip),
}

#[derive(Clone, Debug)]
pub struct Params {
    pub enabled: bool,
    /// Minimum time between two uh-ohs (s).
    pub bump_interval: f64,
    /// How long HELO stays on the display (s).
    pub display_time: f64,
    /// Without a sensor message for this long the mode counts as unknown (s).
    pub sensors_timeout: f64,
    /// A define_song call without an answer counts as busy for this long (s).
    pub call_timeout: f64,
    /// How long hello and the done jingle wait for a playing song to end before they are dropped (s).
    pub queue_time: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params { enabled: true, bump_interval: 4.0, display_time: 3.0, sensors_timeout: 1.0, call_timeout: 2.0, queue_time: 5.0 }
    }
}

/// The SensorState fields it uses.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sensors {
    pub oi_mode: u8,
    pub bump_left: bool,
    pub bump_right: bool,
    /// Any light bumper: never an uh-oh on its own (they see walls all the time).
    pub light_bump: bool,
    pub song_playing: bool,
}

/// The explorer's "done" message says exploration finished, or that it got back to the start
/// point (a trip home asked for or on low battery). No jingle after a trip home that failed.
pub fn worth_a_jingle(message: &str) -> bool {
    message.contains("finished") || message.contains("back at the start point")
}

pub struct SoundEvents {
    p: Params,
    /// The driver's define_song service is available.
    driver: bool,
    mode: Option<u8>,
    sensors_at: f64,
    song_playing: bool,
    /// Bumper states of the previous message (None: none yet, or after a gap).
    bumps: Option<(bool, bool)>,
    hello_pending: bool,
    last_uh_oh: f64,
    /// The song it asked for plays until then.
    playing_until: f64,
    /// Play requests so far, and the length of the last one's song (s).
    plays: u64,
    play_len: f64,
    /// The play request waiting for its answer (its number) and when it was sent.
    in_flight: Option<(u64, f64)>,
    /// Hello / done waiting for the playing song to end, with their deadlines.
    queued: Vec<(Event, f64)>,
    /// The explorer was seen exploring or returning since the last "done".
    run_armed: bool,
    /// When to clear HELO from the display.
    clear_at: Option<f64>,
}

impl SoundEvents {
    pub fn new(p: Params) -> Self {
        SoundEvents {
            hello_pending: p.enabled,
            p,
            driver: false,
            mode: None,
            sensors_at: f64::NEG_INFINITY,
            song_playing: false,
            bumps: None,
            last_uh_oh: f64::NEG_INFINITY,
            playing_until: f64::NEG_INFINITY,
            plays: 0,
            play_len: 0.0,
            in_flight: None,
            queued: Vec::new(),
            run_armed: false,
            clear_at: None,
        }
    }

    pub fn enabled(&self) -> bool {
        self.p.enabled
    }

    fn control(&self) -> bool {
        matches!(self.mode, Some(MODE_SAFE | MODE_FULL))
    }

    fn busy(&self, now: f64) -> bool {
        self.song_playing || now < self.playing_until || self.in_flight.is_some_and(|(_, t)| now - t < self.p.call_timeout)
    }

    /// Why no song can play now, if so. `muted` is only asked when everything else allows it.
    fn gate(&self, now: f64, muted: &dyn Fn() -> bool) -> Option<Skip> {
        if !self.control() {
            Some(Skip::Mode(self.mode))
        } else if !self.driver {
            Some(Skip::NoDriver)
        } else if muted() {
            Some(Skip::Muted)
        } else if self.busy(now) {
            Some(Skip::Busy)
        } else {
            None
        }
    }

    fn play(&mut self, e: Event, now: f64, out: &mut Vec<Action>) {
        out.push(Action::Play(e));
        self.plays += 1;
        self.play_len = e.song().seconds();
        self.in_flight = Some((self.plays, now));
        self.playing_until = now + self.play_len;
        if e == Event::Bump {
            self.last_uh_oh = now;
        }
    }

    /// Plays `e` or says why not. Hello and done wait (up to `queue_time`) for a playing song to
    /// end; an uh-oh over a song is dropped.
    fn attempt(&mut self, e: Event, now: f64, muted: &dyn Fn() -> bool, out: &mut Vec<Action>) {
        match self.gate(now, muted) {
            None => self.play(e, now, out),
            Some(Skip::Busy) if e != Event::Bump => {
                if !self.queued.iter().any(|q| q.0 == e) {
                    self.queued.push((e, now + self.p.queue_time));
                }
            }
            Some(why) => out.push(Action::Skipped(e, why)),
        }
    }

    fn check_hello(&mut self, now: f64, muted: &dyn Fn() -> bool, out: &mut Vec<Action>) {
        if self.hello_pending && self.control() && self.driver {
            self.hello_pending = false;
            out.push(Action::Display(HELLO_TEXT));
            self.clear_at = Some(now + self.p.display_time);
            self.attempt(Event::Hello, now, muted, out);
        }
    }

    /// A /roomba/sensors message.
    pub fn on_sensors(&mut self, s: Sensors, now: f64, muted: &dyn Fn() -> bool) -> Vec<Action> {
        let mut out = Vec::new();
        if !self.p.enabled {
            return out;
        }
        self.mode = Some(s.oi_mode);
        self.sensors_at = now;
        self.song_playing = s.song_playing;
        let rising = self.bumps.is_some_and(|(l, r)| (s.bump_left && !l) || (s.bump_right && !r));
        self.bumps = Some((s.bump_left, s.bump_right));
        self.check_hello(now, muted, &mut out);
        if rising {
            if now - self.last_uh_oh < self.p.bump_interval {
                out.push(Action::Skipped(Event::Bump, Skip::RateLimited));
            } else {
                self.attempt(Event::Bump, now, muted, &mut out);
            }
        }
        out
    }

    /// An /explorer/status message (JSON with "state" and "message").
    pub fn on_explorer_status(&mut self, json: &str, now: f64, muted: &dyn Fn() -> bool) -> Vec<Action> {
        let mut out = Vec::new();
        if !self.p.enabled {
            return out;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else { return out };
        match v.get("state").and_then(|s| s.as_str()) {
            Some("exploring" | "returning") => self.run_armed = true,
            Some("done") if self.run_armed => {
                self.run_armed = false;
                if worth_a_jingle(v.get("message").and_then(|m| m.as_str()).unwrap_or("")) {
                    self.attempt(Event::Done, now, muted, &mut out);
                }
            }
            _ => {}
        }
        out
    }

    /// The steady-clock timer: `driver` is whether the define_song service is available.
    pub fn tick(&mut self, now: f64, driver: bool, muted: &dyn Fn() -> bool) -> Vec<Action> {
        let mut out = Vec::new();
        if !self.p.enabled {
            return out;
        }
        self.driver = driver;
        if now - self.sensors_at > self.p.sensors_timeout {
            self.mode = None;
            self.bumps = None;
            self.song_playing = false;
        }
        if self.in_flight.is_some_and(|(_, t)| now - t >= self.p.call_timeout) {
            self.in_flight = None;
        }
        self.check_hello(now, muted, &mut out);
        let mut queued = std::mem::take(&mut self.queued);
        queued.retain(|&(e, deadline)| {
            if now > deadline {
                out.push(Action::Skipped(e, Skip::Busy));
                return false;
            }
            match self.gate(now, muted) {
                None => {
                    self.play(e, now, &mut out);
                    false
                }
                Some(Skip::Busy) => true,
                Some(why) => {
                    out.push(Action::Skipped(e, why));
                    false
                }
            }
        });
        self.queued = queued;
        if self.clear_at.is_some_and(|t| now >= t) && self.control() {
            self.clear_at = None;
            out.push(Action::Display(CLEAR_TEXT));
        }
        out
    }

    /// Number of the last [`Action::Play`] (for [`SoundEvents::on_played`]).
    pub fn plays(&self) -> u64 {
        self.plays
    }

    /// The driver's answer to play request `play` (`ok`: success), or its call failed. A late answer
    /// to an earlier request changes nothing. On success the song may only have started now (the
    /// driver can be slow to answer), so it counts as playing for its full length from now.
    pub fn on_played(&mut self, play: u64, now: f64, ok: bool) {
        if play != self.plays {
            return;
        }
        self.in_flight = None;
        self.playing_until = if ok { self.playing_until.max(now + self.play_len) } else { self.playing_until.min(now) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const PASSIVE: u8 = 1;

    fn safe() -> Sensors {
        Sensors { oi_mode: MODE_SAFE, ..Default::default() }
    }

    fn bump(left: bool, right: bool) -> Sensors {
        Sensors { bump_left: left, bump_right: right, ..safe() }
    }

    fn unmuted() -> bool {
        false
    }

    fn status(state: &str, message: &str) -> String {
        format!(r#"{{"state": "{state}", "message": "{message}", "reached": 3, "failed": 0, "frontiers": 0, "home": [0.0, 0.0], "goal": null}}"#)
    }

    /// Driver up, Safe mode, hello played (its song is over by t + 1) and HELO cleared.
    fn ready(t: f64) -> SoundEvents {
        let mut s = SoundEvents::new(Params { display_time: 0.0, ..Params::default() });
        s.tick(t, true, &unmuted);
        s.on_sensors(safe(), t, &unmuted);
        s.on_played(s.plays(), t, true);
        assert_eq!(s.tick(t, true, &unmuted), [Action::Display(CLEAR_TEXT)]);
        s
    }

    fn plays(actions: &[Action]) -> Vec<Event> {
        actions.iter().filter_map(|a| if let Action::Play(e) = a { Some(*e) } else { None }).collect()
    }

    #[test]
    fn songs_fit_the_open_interface() {
        for (song, number) in [(&HELLO, 0), (&UH_OH, 1), (&DONE, 2)] {
            assert_eq!(song.number, number);
            assert_eq!(song.notes.len(), song.durations.len(), "{}", song.name);
            assert!((1..=16).contains(&song.notes.len()), "{}", song.name);
            assert!(song.seconds() <= 3.0, "{}", song.name);
        }
        assert_eq!(HELLO.notes, [67, 72, 76]);
        assert_eq!(HELLO.durations, [10, 10, 24]);
        assert_eq!(UH_OH.notes, [76, 76, 69]);
        assert_eq!(UH_OH.durations, [12, 6, 28]);
    }

    #[test]
    fn hello_once_when_the_driver_first_reports_safe_or_full() {
        let mut s = SoundEvents::new(Params::default());
        assert!(s.tick(0.0, true, &unmuted).is_empty());
        let passive = Sensors { oi_mode: PASSIVE, ..Default::default() };
        assert!(s.on_sensors(passive, 0.1, &unmuted).is_empty());
        let full = Sensors { oi_mode: MODE_FULL, ..Default::default() };
        assert_eq!(s.on_sensors(full, 0.2, &unmuted), [Action::Display(HELLO_TEXT), Action::Play(Event::Hello)]);
        s.on_played(s.plays(), 0.3, true);
        // Leaving Safe mode (a cliff, the charger) and coming back: no second hello.
        s.on_sensors(passive, 5.0, &unmuted);
        assert!(s.on_sensors(safe(), 6.0, &unmuted).is_empty());
        assert!(plays(&s.tick(6.1, true, &unmuted)).is_empty());
    }

    #[test]
    fn hello_waits_for_the_driver_service() {
        let mut s = SoundEvents::new(Params::default());
        assert!(s.on_sensors(safe(), 0.0, &unmuted).is_empty(), "service not seen yet");
        assert!(s.tick(0.1, false, &unmuted).is_empty());
        s.on_sensors(safe(), 0.15, &unmuted);
        assert_eq!(s.tick(0.2, true, &unmuted), [Action::Display(HELLO_TEXT), Action::Play(Event::Hello)]);
    }

    #[test]
    fn helo_is_cleared_after_display_time_in_safe_mode_only() {
        let mut s = SoundEvents::new(Params::default());
        s.tick(0.0, true, &unmuted);
        s.on_sensors(safe(), 0.0, &unmuted);
        s.on_played(s.plays(), 0.1, true);
        s.on_sensors(safe(), 2.9, &unmuted);
        assert!(s.tick(2.9, true, &unmuted).is_empty());
        // Passive at clear time: writing the display would make the driver re-enter Safe mode.
        let passive = Sensors { oi_mode: PASSIVE, ..Default::default() };
        s.on_sensors(passive, 3.1, &unmuted);
        assert!(s.tick(3.1, true, &unmuted).is_empty());
        s.on_sensors(safe(), 4.0, &unmuted);
        assert_eq!(s.tick(4.0, true, &unmuted), [Action::Display(CLEAR_TEXT)]);
        assert!(s.tick(5.0, true, &unmuted).is_empty());
    }

    #[test]
    fn uh_oh_on_rising_bumper_edges_only() {
        let mut s = ready(0.0);
        assert_eq!(s.on_sensors(bump(true, false), 1.0, &unmuted), [Action::Play(Event::Bump)]);
        s.on_played(s.plays(), 1.05, true);
        // Held down: no new edge.
        assert!(s.on_sensors(bump(true, false), 1.1, &unmuted).is_empty());
        assert!(s.on_sensors(bump(true, false), 6.0, &unmuted).is_empty());
        // The right bumper joining the held left one is a new edge.
        assert_eq!(s.on_sensors(bump(true, true), 6.1, &unmuted), [Action::Play(Event::Bump)]);
        s.on_played(s.plays(), 6.15, true);
        // Releasing is not an edge.
        assert!(s.on_sensors(bump(false, false), 11.0, &unmuted).is_empty());
        // Light bumpers alone: nothing.
        let light = Sensors { light_bump: true, ..safe() };
        assert!(s.on_sensors(light, 11.1, &unmuted).is_empty());
        assert_eq!(s.on_sensors(bump(false, true), 11.2, &unmuted), [Action::Play(Event::Bump)]);
    }

    #[test]
    fn bumper_held_when_the_node_starts_is_no_edge() {
        let mut s = SoundEvents::new(Params::default());
        s.tick(0.0, true, &unmuted);
        let actions = s.on_sensors(bump(true, false), 0.0, &unmuted);
        assert_eq!(plays(&actions), [Event::Hello]);
        // A gap in the sensor data forgets the bumper state too.
        s.on_played(s.plays(), 0.1, true);
        s.tick(5.0, true, &unmuted);
        assert!(s.on_sensors(bump(true, false), 5.1, &unmuted).is_empty());
    }

    #[test]
    fn at_most_one_uh_oh_per_bump_interval() {
        let mut s = ready(0.0);
        assert_eq!(plays(&s.on_sensors(bump(true, false), 10.0, &unmuted)), [Event::Bump]);
        s.on_played(s.plays(), 10.05, true);
        s.on_sensors(bump(false, false), 11.0, &unmuted);
        assert_eq!(s.on_sensors(bump(true, false), 13.9, &unmuted), [Action::Skipped(Event::Bump, Skip::RateLimited)]);
        s.on_sensors(bump(false, false), 13.95, &unmuted);
        assert_eq!(plays(&s.on_sensors(bump(true, false), 14.0, &unmuted)), [Event::Bump]);
    }

    #[test]
    fn no_uh_oh_while_a_song_plays() {
        let mut s = ready(0.0);
        let playing = Sensors { song_playing: true, ..safe() };
        s.on_sensors(playing, 10.0, &unmuted);
        let hit = Sensors { bump_left: true, ..playing };
        assert_eq!(s.on_sensors(hit, 10.1, &unmuted), [Action::Skipped(Event::Bump, Skip::Busy)]);
        // Right after it asked for a song, before the driver reports it playing.
        let mut s = ready(0.0);
        s.on_explorer_status(&status("exploring", ""), 20.0, &unmuted);
        assert_eq!(plays(&s.on_explorer_status(&status("done", "Exploration finished"), 20.0, &unmuted)), [Event::Done]);
        s.on_played(s.plays(), 20.05, true);
        assert_eq!(s.on_sensors(bump(true, false), 20.5, &unmuted), [Action::Skipped(Event::Bump, Skip::Busy)]);
        s.on_sensors(bump(false, false), 21.0, &unmuted);
        assert_eq!(plays(&s.on_sensors(bump(true, false), 21.4, &unmuted)), [Event::Bump], "the jingle is over");
    }

    #[test]
    fn a_call_without_an_answer_is_busy_only_until_the_call_timeout() {
        let mut s = ready(0.0);
        assert_eq!(plays(&s.on_sensors(bump(true, false), 10.0, &unmuted)), [Event::Bump]);
        // No answer: the driver is slow. The jingle waits meanwhile...
        s.on_explorer_status(&status("exploring", ""), 11.0, &unmuted);
        assert!(s.on_explorer_status(&status("done", "Exploration finished"), 11.0, &unmuted).is_empty());
        s.on_sensors(bump(true, false), 11.5, &unmuted);
        assert!(s.tick(11.5, true, &unmuted).is_empty());
        // ...and plays once the call timed out.
        s.on_sensors(bump(true, false), 12.0, &unmuted);
        assert_eq!(s.tick(12.0, true, &unmuted), [Action::Play(Event::Done)]);
        // The uh-oh's late answer doesn't end the wait for the jingle's.
        let jingle = s.plays();
        s.on_played(jingle - 1, 12.5, true);
        assert!(s.busy(13.5), "jingle call still waiting");
        // Its own answer: playing for the jingle's length from the answer on (1.30 s).
        s.on_played(jingle, 13.6, true);
        assert!(s.busy(14.85));
        assert!(!s.busy(14.95));
    }

    #[test]
    fn jingle_once_per_exploration_run() {
        let mut s = ready(0.0);
        // The latched status of an earlier run: no jingle.
        assert!(s.on_explorer_status(&status("done", "Exploration finished"), 1.0, &unmuted).is_empty());
        assert!(s.on_explorer_status(&status("idle", ""), 1.5, &unmuted).is_empty());
        s.on_explorer_status(&status("exploring", "Looking for unmapped areas…"), 2.0, &unmuted);
        s.on_explorer_status(&status("returning", "Exploration finished: no reachable unmapped areas left"), 3.0, &unmuted);
        // Back home to look again, then exploring once more: still the same run.
        s.on_explorer_status(&status("exploring", "Back at the start point; checking for unmapped areas again…"), 4.0, &unmuted);
        let done = status("done", "Exploration finished: no reachable unmapped areas left - back at the start point");
        assert_eq!(s.on_explorer_status(&done, 5.0, &unmuted), [Action::Play(Event::Done)]);
        s.on_played(s.plays(), 5.05, true);
        // The explorer repeats its status every second.
        for t in 6..12 {
            assert!(s.on_explorer_status(&done, t as f64, &unmuted).is_empty());
            assert!(s.tick(t as f64, true, &unmuted).iter().all(|a| !matches!(a, Action::Play(_))));
        }
        // The next run gets its own jingle; going home on request counts too.
        s.on_explorer_status(&status("exploring", ""), 20.0, &unmuted);
        s.on_explorer_status(&status("returning", "Returning to the start point"), 25.0, &unmuted);
        let home = status("done", "Returning to the start point - back at the start point");
        s.on_sensors(safe(), 30.0, &unmuted);
        assert_eq!(plays(&s.on_explorer_status(&home, 30.0, &unmuted)), [Event::Done]);
    }

    #[test]
    fn no_jingle_after_a_failed_trip_home() {
        let mut s = ready(0.0);
        s.on_explorer_status(&status("exploring", ""), 1.0, &unmuted);
        let failed = status("done", "Battery at 12%: returning to the start point - could not drive back to the start point");
        assert!(s.on_explorer_status(&failed, 2.0, &unmuted).is_empty());
        assert!(worth_a_jingle("Exploration finished: no reachable unmapped areas left - could not drive back to the start point"));
        assert!(worth_a_jingle("Exploration finished"));
        assert!(!worth_a_jingle("Stopped"));
        assert!(s.on_explorer_status("not json", 3.0, &unmuted).is_empty());
    }

    #[test]
    fn mute_file_is_checked_at_each_event() {
        let muted = Cell::new(true);
        let asked = Cell::new(0);
        let is_muted = || {
            asked.set(asked.get() + 1);
            muted.get()
        };
        let mut s = SoundEvents::new(Params::default());
        s.tick(0.0, true, &is_muted);
        // HELO still shows: the mute is for sounds.
        assert_eq!(s.on_sensors(safe(), 0.0, &is_muted), [Action::Display(HELLO_TEXT), Action::Skipped(Event::Hello, Skip::Muted)]);
        assert_eq!(asked.get(), 1);
        // Sensor messages and ticks without an event don't look at the file.
        for k in 1..50 {
            s.on_sensors(safe(), k as f64 * 0.015, &is_muted);
        }
        s.tick(1.0, true, &is_muted);
        assert_eq!(asked.get(), 1);
        assert_eq!(s.on_sensors(bump(true, false), 1.1, &is_muted), [Action::Skipped(Event::Bump, Skip::Muted)]);
        // Unmuted live: the next bump plays.
        muted.set(false);
        s.on_sensors(bump(false, false), 1.2, &is_muted);
        assert_eq!(s.on_sensors(bump(true, false), 1.3, &is_muted), [Action::Play(Event::Bump)]);
        assert_eq!(asked.get(), 3);
    }

    #[test]
    fn nothing_plays_outside_safe_or_full_mode() {
        let mut s = ready(0.0);
        let passive = Sensors { oi_mode: PASSIVE, ..Default::default() };
        s.on_sensors(passive, 10.0, &unmuted);
        assert_eq!(s.on_sensors(Sensors { bump_left: true, ..passive }, 10.1, &unmuted), [Action::Skipped(Event::Bump, Skip::Mode(Some(PASSIVE)))]);
        s.on_explorer_status(&status("exploring", ""), 11.0, &unmuted);
        assert_eq!(
            s.on_explorer_status(&status("done", "Exploration finished"), 12.0, &unmuted),
            [Action::Skipped(Event::Done, Skip::Mode(Some(PASSIVE)))]
        );
        // No sensor data for a while: the mode is unknown.
        s.on_sensors(safe(), 20.0, &unmuted);
        s.tick(21.5, true, &unmuted);
        s.on_explorer_status(&status("exploring", ""), 21.6, &unmuted);
        assert_eq!(s.on_explorer_status(&status("done", "Exploration finished"), 21.7, &unmuted), [Action::Skipped(Event::Done, Skip::Mode(None))]);
        // The driver's service gone: nothing is sent.
        let mut s = ready(0.0);
        s.tick(1.0, false, &unmuted);
        s.on_sensors(safe(), 1.0, &unmuted);
        assert_eq!(s.on_sensors(bump(true, false), 1.1, &unmuted), [Action::Skipped(Event::Bump, Skip::NoDriver)]);
    }

    #[test]
    fn jingle_waits_for_a_playing_song_but_not_forever() {
        let mut s = ready(0.0);
        assert_eq!(plays(&s.on_sensors(bump(true, false), 10.0, &unmuted)), [Event::Bump]);
        s.on_played(s.plays(), 10.05, true);
        s.on_explorer_status(&status("exploring", ""), 10.1, &unmuted);
        assert!(s.on_explorer_status(&status("done", "Exploration finished"), 10.2, &unmuted).is_empty());
        // The uh-oh (0.72 s) ends, but the driver still reports a song playing for a moment.
        let playing = Sensors { song_playing: true, ..bump(true, false) };
        s.on_sensors(playing, 10.8, &unmuted);
        assert!(s.tick(10.8, true, &unmuted).is_empty());
        s.on_sensors(bump(true, false), 11.0, &unmuted);
        assert_eq!(s.tick(11.0, true, &unmuted), [Action::Play(Event::Done)]);
        // Something else keeps playing past queue_time: dropped.
        let mut s = ready(0.0);
        s.on_explorer_status(&status("exploring", ""), 1.0, &unmuted);
        let playing = Sensors { song_playing: true, ..safe() };
        s.on_sensors(playing, 1.0, &unmuted);
        assert!(s.on_explorer_status(&status("done", "Exploration finished"), 1.0, &unmuted).is_empty());
        for k in 1..50 {
            let t = 1.0 + k as f64 * 0.1;
            s.on_sensors(playing, t, &unmuted);
            assert!(s.tick(t, true, &unmuted).is_empty());
        }
        s.on_sensors(playing, 6.1, &unmuted);
        assert_eq!(s.tick(6.1, true, &unmuted), [Action::Skipped(Event::Done, Skip::Busy)]);
    }

    #[test]
    fn a_late_acknowledgement_keeps_the_song_playing_for_its_length() {
        let mut s = SoundEvents::new(Params { display_time: 0.0, ..Params::default() });
        s.tick(10.0, true, &unmuted);
        assert_eq!(plays(&s.on_sensors(safe(), 10.0, &unmuted)), [Event::Hello]);
        s.on_explorer_status(&status("exploring", ""), 10.1, &unmuted);
        assert!(s.on_explorer_status(&status("done", "Exploration finished"), 10.2, &unmuted).is_empty());
        // The driver answers late, as it starts hello (0.69 s): hello plays until ~11.39.
        s.on_sensors(safe(), 10.7, &unmuted);
        s.on_played(s.plays(), 10.7, true);
        s.on_sensors(safe(), 11.3, &unmuted);
        assert!(plays(&s.tick(11.3, true, &unmuted)).is_empty(), "hello still playing");
        s.on_sensors(safe(), 11.4, &unmuted);
        assert_eq!(plays(&s.tick(11.4, true, &unmuted)), [Event::Done]);
    }

    #[test]
    fn a_failed_play_does_not_count_as_playing() {
        let mut s = ready(0.0);
        s.on_explorer_status(&status("exploring", ""), 1.0, &unmuted);
        assert_eq!(plays(&s.on_explorer_status(&status("done", "Exploration finished"), 1.0, &unmuted)), [Event::Done]);
        s.on_played(s.plays(), 1.05, false);
        assert_eq!(plays(&s.on_sensors(bump(true, false), 1.1, &unmuted)), [Event::Bump]);
    }

    #[test]
    fn disabled_does_nothing() {
        let mut s = SoundEvents::new(Params { enabled: false, ..Params::default() });
        assert!(s.tick(0.0, true, &unmuted).is_empty());
        assert!(s.on_sensors(safe(), 0.0, &unmuted).is_empty());
        assert!(s.on_sensors(bump(true, false), 0.1, &unmuted).is_empty());
        s.on_explorer_status(&status("exploring", ""), 1.0, &unmuted);
        assert!(s.on_explorer_status(&status("done", "Exploration finished"), 2.0, &unmuted).is_empty());
        assert!(s.tick(5.0, true, &unmuted).is_empty());
    }
}
