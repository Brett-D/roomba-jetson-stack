//! The Roomba's Open Interface as the driver sees it on the serial port (the twin's side of the
//! pty). Behaves like tools/fake_roomba.py and the robot: Start/Safe/Full/Stop/Reset and the
//! cleaning behaviours change the OI mode (Safe drops to Passive on a cliff or wheel drop, motors
//! off), opcode 148 streams frames [19][n][id][data]...[checksum] every 15 ms with whatever layout
//! was asked for last (the driver alternates its primary and secondary layouts), 150 pauses and
//! resumes, 142/149 answer queries, and the drive commands (137, 145, 146) set the wheel speeds,
//! in Safe/Full mode only. The sensor values come from the caller (`raw(packet id)`).

use crate::oi;

/// Command opcode -> data bytes (None: variable length). As fake_roomba.py.
fn args(op: u8) -> Option<Option<usize>> {
    Some(match op {
        7 | 128 | 130..=136 | 143 | 173 => Some(0),
        129 | 138 | 141 | 142 | 150 => Some(1),
        162 => Some(2),
        139 | 144 | 168 => Some(3),
        137 | 145 | 146 | 163 | 164 => Some(4),
        165 => Some(1),
        167 => Some(15),
        140 | 148 | 149 => None,
        _ => return None,
    })
}

/// Bytes in the command at the start of buf: Some(0) for an unknown opcode, None if incomplete.
fn command_length(buf: &[u8]) -> Option<usize> {
    let op = buf[0];
    let Some(n) = args(op) else { return Some(0) };
    if let Some(n) = n {
        return (buf.len() > n).then_some(1 + n);
    }
    if buf.len() < 2 {
        return None;
    }
    let total = if op == oi::SONG {
        if buf.len() < 3 {
            return None;
        }
        3 + 2 * buf[2] as usize
    } else {
        2 + buf[1] as usize
    };
    (buf.len() >= total).then_some(total)
}

/// What the robot printed when it booted (after opcode 7), as a Roomba 550 does.
pub const BOOT_TEXT: &[u8] = b"bl-start\r\nSTR730\r\nbootloader id: #x47186549 82ECCFFF\r\nbootloader info rev: #xF000\r\n\
bootloader rev: #x0001\r\n2007-05-14-1715-L   \r\nRoomba by iRobot!\r\nstr730\r\n2010-03-12-1430-L   \r\n\
battery-current-zero 258\r\n";

#[derive(Clone, Debug)]
pub struct OiSim {
    pub mode: u8,
    pub streaming: bool,
    pub layout: Vec<u8>,
    buf: Vec<u8>,
    /// Packets 39-42: requested velocity, radius, right velocity, left velocity.
    pub requested: [i32; 4],
    /// Wheel speeds the robot drives at (left, right m/s).
    pub target: (f64, f64),
    pub wheel_base_m: f64,
    t0: Option<f64>,
    k: u64,
    pub frames: u64,
    pub commands: u64,
    /// Things worth logging (mode changes, unknown bytes), drained by the caller.
    pub log: Vec<String>,
}

pub const TICK_S: f64 = oi::STREAM_PERIOD_S;

impl OiSim {
    pub fn new(wheel_base_m: f64) -> Self {
        OiSim {
            mode: oi::MODE_OFF,
            streaming: false,
            layout: Vec::new(),
            buf: Vec::new(),
            requested: [0; 4],
            target: (0.0, 0.0),
            wheel_base_m,
            t0: None,
            k: 0,
            frames: 0,
            commands: 0,
            log: Vec::new(),
        }
    }

    fn set_mode(&mut self, mode: u8, why: &str) {
        if mode != self.mode {
            self.log.push(format!("OI mode {} -> {}{why}", oi::mode_str(Some(self.mode as i64)), oi::mode_str(Some(mode as i64))));
            self.mode = mode;
        }
        if !matches!(mode, oi::MODE_SAFE | oi::MODE_FULL) {
            self.target = (0.0, 0.0);
            self.requested = [0; 4];
        }
    }

    /// Safe mode stops and drops to Passive on a cliff, wheel drop or the charger.
    pub fn check_safety(&mut self, unsafe_now: bool) {
        if self.mode == oi::MODE_SAFE && unsafe_now {
            self.set_mode(oi::MODE_PASSIVE, " (safety: cliff or wheel drop)");
        }
    }

    /// Bytes written by the driver. Returns what the robot answers (query replies, boot text).
    pub fn on_bytes(&mut self, data: &[u8], now: f64, unsafe_now: bool, raw: &impl Fn(u8) -> i32) -> Vec<u8> {
        let mut out = Vec::new();
        self.buf.extend_from_slice(data);
        while !self.buf.is_empty() {
            let Some(n) = command_length(&self.buf) else { break };
            if n == 0 {
                self.log.push(format!("ignored unknown byte {:#04x}", self.buf[0]));
                self.buf.remove(0);
                continue;
            }
            let cmd: Vec<u8> = self.buf.drain(..n).collect();
            self.commands += 1;
            self.execute(&cmd, now, unsafe_now, raw, &mut out);
        }
        out
    }

    fn execute(&mut self, cmd: &[u8], now: f64, unsafe_now: bool, raw: &impl Fn(u8) -> i32, out: &mut Vec<u8>) {
        let op = cmd[0];
        if op == oi::START {
            self.set_mode(oi::MODE_PASSIVE, "");
            return;
        }
        if self.mode == oi::MODE_OFF {
            return; // OI off: only Start is heard
        }
        let i16_at = |k: usize| i16::from_be_bytes([cmd[k], cmd[k + 1]]) as i32;
        match op {
            oi::CONTROL | oi::SAFE => {
                if unsafe_now {
                    self.set_mode(oi::MODE_PASSIVE, " (Safe refused: cliff or wheel drop)");
                } else {
                    self.set_mode(oi::MODE_SAFE, "");
                }
            }
            oi::FULL => self.set_mode(oi::MODE_FULL, ""),
            oi::POWER | oi::SPOT | oi::CLEAN | oi::MAX | oi::SEEK_DOCK => {
                self.set_mode(oi::MODE_PASSIVE, " (cleaning behaviour: the twin just stands still)")
            }
            oi::STOP => {
                self.set_mode(oi::MODE_OFF, "");
                self.streaming = false;
            }
            oi::RESET => {
                self.set_mode(oi::MODE_OFF, " (reset)");
                self.streaming = false;
                self.layout.clear();
                out.extend_from_slice(BOOT_TEXT);
            }
            oi::STREAM => {
                if self.t0.is_none() {
                    self.t0 = Some(now);
                    self.k = 0;
                }
                self.layout = cmd[2..].to_vec();
                self.streaming = true;
            }
            oi::PAUSE_RESUME_STREAM => self.streaming = cmd[1] != 0 && !self.layout.is_empty(),
            oi::SENSORS | oi::QUERY_LIST => {
                let ids = if op == oi::SENSORS { &cmd[1..2] } else { &cmd[2..] };
                for &pid in ids {
                    match oi::encode(pid, raw) {
                        Ok(bytes) => out.extend(bytes),
                        Err(e) => self.log.push(format!("query: {e}")),
                    }
                }
            }
            oi::DRIVE | oi::DRIVE_DIRECT | oi::DRIVE_PWM if matches!(self.mode, oi::MODE_SAFE | oi::MODE_FULL) => {
                let (a, b) = (i16_at(1), i16_at(3));
                match op {
                    oi::DRIVE => {
                        self.requested = [a, b, 0, 0];
                        self.target = drive_wheels(a as f64 / 1000.0, b, self.wheel_base_m);
                    }
                    oi::DRIVE_DIRECT => {
                        self.requested = [0, 0, a, b];
                        self.target = (b as f64 / 1000.0, a as f64 / 1000.0);
                    }
                    _ => {
                        self.requested = [0; 4];
                        self.target = (b as f64 / 255.0 * 0.5, a as f64 / 255.0 * 0.5);
                    }
                }
            }
            _ => {}
        }
    }

    /// Whether a stream frame is due at `now` (frame k at t0 + k * 15 ms; a late caller gets one
    /// frame and skips the missed ones).
    pub fn frame_due(&mut self, now: f64) -> bool {
        let Some(t0) = self.t0 else { return false };
        if now < t0 + (self.k + 1) as f64 * TICK_S {
            return false;
        }
        self.k = ((now - t0) / TICK_S).floor() as u64;
        self.streaming && !self.layout.is_empty()
    }

    pub fn frame(&mut self, raw: &impl Fn(u8) -> i32) -> Option<Vec<u8>> {
        match oi::encode_stream_frame(&self.layout, raw) {
            Ok(f) => {
                self.frames += 1;
                Some(f)
            }
            Err(e) => {
                self.log.push(format!("stream: {e}"));
                self.streaming = false;
                None
            }
        }
    }
}

/// Opcode 137's (velocity m/s, radius mm) as (left, right) wheel speeds.
pub fn drive_wheels(v: f64, radius_mm: i32, wheel_base: f64) -> (f64, f64) {
    match radius_mm {
        32767 | -32768 => (v, v),
        -1 => (v, -v),
        1 => (-v, v),
        r => {
            let r = r as f64 / 1000.0;
            (v * (r - wheel_base / 2.0) / r, v * (r + wheel_base / 2.0) / r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oi::{Field as F, StreamParser, STREAM_PRIMARY, STREAM_SECONDARY};

    fn raw(pid: u8) -> i32 {
        match pid {
            35 => 2,
            43 => 1000,
            44 => 2000,
            _ => 0,
        }
    }

    #[test]
    fn modes_follow_the_commands_and_safety() {
        let mut s = OiSim::new(0.235);
        assert!(s.on_bytes(&[oi::SAFE], 0.0, false, &raw).is_empty());
        assert_eq!(s.mode, oi::MODE_OFF); // only Start is heard
        s.on_bytes(&[oi::START], 0.0, false, &raw);
        assert_eq!(s.mode, oi::MODE_PASSIVE);
        s.on_bytes(&oi::drive_direct(100.0, 100.0), 0.0, false, &raw);
        assert_eq!(s.target, (0.0, 0.0)); // Passive: no driving
        s.on_bytes(&[oi::SAFE], 0.0, true, &raw);
        assert_eq!(s.mode, oi::MODE_PASSIVE); // cliff: Safe refused
        s.on_bytes(&[oi::SAFE], 0.0, false, &raw);
        assert_eq!(s.mode, oi::MODE_SAFE);
        // Split across two writes, right wheel first on the wire.
        let cmd = oi::drive_direct(-100.0, 200.0);
        s.on_bytes(&cmd[..2], 0.0, false, &raw);
        s.on_bytes(&cmd[2..], 0.0, false, &raw);
        assert_eq!(s.target, (-0.1, 0.2));
        assert_eq!(s.requested, [0, 0, 200, -100]);
        s.check_safety(true);
        assert_eq!((s.mode, s.target), (oi::MODE_PASSIVE, (0.0, 0.0)));
        s.on_bytes(&[oi::FULL], 0.0, true, &raw);
        s.on_bytes(&oi::drive(100.0, oi::RADIUS_TURN_CW), 0.0, true, &raw);
        assert_eq!(s.target, (0.1, -0.1));
        s.check_safety(true);
        assert_eq!(s.mode, oi::MODE_FULL); // Full mode ignores cliffs
        assert_eq!(s.on_bytes(&[oi::RESET], 0.0, false, &raw), BOOT_TEXT);
        assert_eq!(s.mode, oi::MODE_OFF);
    }

    #[test]
    fn streams_the_requested_layouts_every_15_ms() {
        let mut s = OiSim::new(0.235);
        s.on_bytes(&[oi::START], 0.0, false, &raw);
        assert!(!s.frame_due(1.0));
        s.on_bytes(&oi::stream(&STREAM_PRIMARY).unwrap(), 1.0, false, &raw);
        assert!(!s.frame_due(1.010));
        assert!(s.frame_due(1.0151));
        assert!(!s.frame_due(1.020));
        let mut parser = StreamParser::new(&[&STREAM_PRIMARY, &STREAM_SECONDARY]).unwrap();
        let v = parser.feed(&s.frame(&raw).unwrap()).pop().unwrap();
        assert_eq!((v.get(F::OiMode), v.get(F::LeftEncoderCounts)), (Some(2), Some(1000)));
        s.on_bytes(&oi::stream(&STREAM_SECONDARY).unwrap(), 1.02, false, &raw);
        assert!(s.frame_due(1.031));
        let frames = parser.feed_frames(&s.frame(&raw).unwrap());
        assert_eq!(frames[0].0, oi::stream_payload_length(&STREAM_SECONDARY).unwrap());
        s.on_bytes(&oi::pause_resume_stream(false), 1.04, false, &raw);
        assert!(!s.frame_due(1.05));
        // A late caller gets one frame, not a burst.
        s.on_bytes(&oi::pause_resume_stream(true), 1.06, false, &raw);
        assert!(s.frame_due(1.2));
        assert!(!s.frame_due(1.2));
    }

    #[test]
    fn queries_and_garbage() {
        let mut s = OiSim::new(0.235);
        s.on_bytes(&[oi::START], 0.0, false, &raw);
        assert_eq!(s.on_bytes(&[0xFF, oi::SENSORS, 35], 0.0, false, &raw), [2]);
        assert_eq!(s.on_bytes(&oi::query_list(&[43, 44]).unwrap(), 0.0, false, &raw), [0x03, 0xE8, 0x07, 0xD0]);
        assert_eq!(s.log, ["OI mode off -> passive", "ignored unknown byte 0xff"]);
    }

    #[test]
    fn drive_radius_to_wheels() {
        assert_eq!(drive_wheels(0.2, 32767, 0.235), (0.2, 0.2));
        let (l, r) = drive_wheels(0.2, 500, 0.235);
        assert!((l - 0.2 * (0.5 - 0.1175) / 0.5).abs() < 1e-12 && (r - 0.2 * (0.5 + 0.1175) / 0.5).abs() < 1e-12);
        assert_eq!(drive_wheels(0.1, 1, 0.235), (-0.1, 0.1));
    }
}
