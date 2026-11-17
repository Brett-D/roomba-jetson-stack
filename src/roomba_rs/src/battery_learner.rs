//! battery_learner: a self-learning state-of-charge gauge for the 18 V power-tool pack (5 Li-ion
//! cells) wired to the Roomba's battery contacts, from the Roomba's own voltage and current readings
//! (/roomba/sensors). ROS-free so it can be unit tested; the node is src/bin/battery_learner.rs.
//!
//! - **Coulomb counting** from current_ma (negative = discharging): a pack starts at 100 % when it
//!   goes in, and `SoC = start - used_mAh / capacity`.
//! - **Swaps**: with the pack out the Roomba has no power, so its sensor data stops (a gap in the frame
//!   stamps) and comes back at least `swap_jump_v` higher. The jump is also checked at start-up
//!   against the voltage saved last time (the Jetson may have been off during a swap). A gap without a
//!   jump (a driver restart, a stream stall) is the same pack. [`Learner::manual_swap`] (the node's
//!   `~/pack_swapped` service) says "a full pack went in now"; it re-anchors at 100 % but teaches
//!   nothing unless it confirms a detected swap (merged within `manual_merge_s`).
//! - **Learning**: a pack comes out at "1 bar", `one_bar_fraction` (0.15). A pack that went in full
//!   therefore measured `capacity = used_mAh / (1 - one_bar_fraction)`; the estimates are averaged,
//!   outliers rejected. Its (open-circuit voltage, SoC) samples, labelled linearly in used mAh between
//!   its two anchors, are merged across packs into a voltage curve fitted by isotonic regression, with
//!   the stock Li-ion curve (shifted by a fitted voltage offset) as the prior where there is no data.
//! - **Voltage**: the curve gives the charge when the counter has no anchor (first run, a part-charged
//!   pack), and pulls the counter slowly towards it while the robot draws little current.
//!
//! Open-circuit voltage = measured voltage + discharge current x `resistance_ohm`, smoothed over
//! `smoothing_s`. Time comes from the frame stamps, so a replay may run faster than real time.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::battery::{li_ion_fraction, LI_ION_CURVE};
use crate::watchdog::ParamSource;

pub const STATE_FILE: &str = "battery_learner.json";
pub const STATE_VERSION: u32 = 1;

/// Voltage bins of the learned curve: 0.05 V wide from 10 V to 22 V (2-4.4 V per cell for 5S).
const BIN_V0: f64 = 10.0;
const BIN_V: f64 = 0.05;
const BINS: usize = 240;
/// A pack teaches capacity and curve only if its anchors are this far apart and it used this much.
const MIN_LEARN_SPAN: f64 = 0.3;
const MIN_LEARN_MAH: f64 = 100.0;
/// Capacity estimates outside this range (mAh) are nonsense (a mistaken manual reset, a glitch).
const CAPACITY_RANGE: (f64, f64) = (200.0, 20000.0);
/// Finished packs kept in the state file; the capacity uses the latest `ESTIMATES_USED` of them.
const HISTORY_LEN: usize = 60;
const ESTIMATES_USED: usize = 20;
/// Outlier rejection needs this many capacity estimates.
const MIN_FOR_OUTLIERS: usize = 3;
/// Readings below this are not a pack voltage (no battery packet yet).
const MIN_VOLTAGE_MV: i32 = 5000;

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Cells in series (5 for an 18 V pack).
    pub cells: f64,
    /// Pack + wiring resistance: open-circuit voltage = V + discharge current x R.
    pub resistance_ohm: f64,
    /// Time constant of the open-circuit voltage filter.
    pub smoothing_s: f64,
    /// State of charge of a pack when it is swapped out (its gauge shows 1 bar).
    pub one_bar_fraction: f64,
    /// A new pack whose voltage reads at least this charge on the curve counts as full (100 %);
    /// below it, it starts from its voltage.
    pub full_pack_min_fraction: f64,
    /// Capacity before any pack taught one (mAh), and its weight among the learned estimates.
    pub capacity_mah: f64,
    pub capacity_prior_weight: f64,
    /// Weight of a pack that started from its voltage rather than full (capacity and curve).
    pub voltage_anchor_weight: f64,
    /// With 3+ estimates, those this far (fraction) from their weighted median are rejected.
    pub outlier_fraction: f64,
    /// Weight of the stock curve in each voltage bin (a pack that started full weighs 1).
    pub curve_prior_weight: f64,
    /// A rise of the open-circuit voltage across a sensor gap that means a new pack (V).
    pub swap_jump_v: f64,
    /// Frames further apart than this (stamp time) are a gap.
    pub gap_s: f64,
    /// Readings after a gap (and at start-up) whose median is compared with the voltage before.
    pub settle_samples: usize,
    /// The robot is "at rest" below this current (mA) ...
    pub rest_current_ma: f64,
    /// ... after this long, and then the counter moves towards the voltage estimate with time constant
    /// `correction_tau_s` (pack went in full) or `uncertain_correction_tau_s` (pack started from its
    /// voltage).
    pub rest_settle_s: f64,
    pub correction_tau_s: f64,
    pub uncertain_correction_tau_s: f64,
    /// Interval of the discharge samples (CSV and curve data).
    pub sample_period_s: f64,
    /// A manual reset this soon (frame stamp time) after a detected swap only marks that pack full
    /// (no second swap).
    pub manual_merge_s: f64,
    /// First run only (no state file): a known (open-circuit voltage, charge) point of the pack in
    /// the robot, which shifts the stock curve to pass through it. seed_voltage 0 = no shift.
    pub seed_voltage: f64,
    pub seed_fraction: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            cells: 5.0,
            resistance_ohm: 0.1,
            smoothing_s: 30.0,
            one_bar_fraction: 0.15,
            full_pack_min_fraction: 0.7,
            capacity_mah: 2000.0,
            capacity_prior_weight: 0.2,
            voltage_anchor_weight: 0.3,
            outlier_fraction: 0.3,
            curve_prior_weight: 0.3,
            swap_jump_v: 0.4,
            gap_s: 2.0,
            settle_samples: 5,
            rest_current_ma: 800.0,
            rest_settle_s: 60.0,
            correction_tau_s: 7200.0,
            uncertain_correction_tau_s: 1200.0,
            sample_period_s: 10.0,
            manual_merge_s: 600.0,
            seed_voltage: 0.0,
            seed_fraction: 0.62,
        }
    }
}

impl Params {
    pub fn load(src: &mut impl ParamSource) -> anyhow::Result<Self> {
        let d = Params::default();
        let p = Params {
            cells: src.int("cells", d.cells as i64)? as f64,
            resistance_ohm: src.float("internal_resistance_ohm", d.resistance_ohm)?,
            smoothing_s: src.float("smoothing_s", d.smoothing_s)?,
            one_bar_fraction: src.float("one_bar_fraction", d.one_bar_fraction)?,
            full_pack_min_fraction: src.float("full_pack_min_fraction", d.full_pack_min_fraction)?,
            capacity_mah: src.float("capacity_mah", d.capacity_mah)?,
            capacity_prior_weight: src.float("capacity_prior_weight", d.capacity_prior_weight)?,
            voltage_anchor_weight: src.float("voltage_anchor_weight", d.voltage_anchor_weight)?,
            outlier_fraction: src.float("outlier_fraction", d.outlier_fraction)?,
            curve_prior_weight: src.float("curve_prior_weight", d.curve_prior_weight)?,
            swap_jump_v: src.float("swap_jump_v", d.swap_jump_v)?,
            gap_s: src.float("gap_s", d.gap_s)?,
            settle_samples: src.int("settle_samples", d.settle_samples as i64)?.max(1) as usize,
            rest_current_ma: src.float("rest_current_ma", d.rest_current_ma)?,
            rest_settle_s: src.float("rest_settle_s", d.rest_settle_s)?,
            correction_tau_s: src.float("correction_tau_s", d.correction_tau_s)?,
            uncertain_correction_tau_s: src.float("uncertain_correction_tau_s", d.uncertain_correction_tau_s)?,
            sample_period_s: src.float("sample_period_s", d.sample_period_s)?,
            manual_merge_s: src.float("manual_merge_s", d.manual_merge_s)?,
            seed_voltage: src.float("seed_voltage", d.seed_voltage)?,
            seed_fraction: src.float("seed_fraction", d.seed_fraction)?,
        };
        anyhow::ensure!(p.cells >= 1.0, "cells must be at least 1");
        anyhow::ensure!((0.0..0.7).contains(&p.one_bar_fraction), "one_bar_fraction must be in [0, 0.7)");
        anyhow::ensure!(p.capacity_mah > 0.0, "capacity_mah must be positive");
        anyhow::ensure!(p.gap_s > 0.0 && p.swap_jump_v > 0.0, "gap_s and swap_jump_v must be positive");
        Ok(p)
    }
}

/// How a pack's starting charge is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Anchor {
    /// It went in full (a detected swap to a full pack, or the manual reset).
    Full,
    /// From its voltage on the curve (first run, a part-charged pack).
    Voltage,
}

/// How a swap was noticed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum How {
    /// The voltage jumped across a gap in the sensor data.
    Gap,
    /// The voltage at start-up is well above the one saved last time.
    Startup,
    /// The `~/pack_swapped` service.
    Manual,
}

impl How {
    fn as_str(self) -> &'static str {
        match self {
            How::Gap => "voltage jump after a sensor gap",
            How::Startup => "voltage jump since the last run",
            How::Manual => "manual reset",
        }
    }
}

/// Discharge samples of the pack in use, by open-circuit voltage bin: count and summed used mAh.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PackBin {
    pub bin: i32,
    pub n: u32,
    pub used_sum_mah: f64,
}

/// JSON has no NaN: an unknown value is written as null.
mod nan_as_null {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_finite() {
            s.serialize_f64(*v)
        } else {
            s.serialize_none()
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
    }
}

/// The pack in the robot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pack {
    pub id: u32,
    pub inserted_unix: f64,
    pub anchor: Anchor,
    /// Charge when it went in (unknown until the first reading of the first run).
    #[serde(with = "nan_as_null")]
    pub start_soc: f64,
    #[serde(with = "nan_as_null")]
    pub start_ocv: f64,
    /// Coulomb count since it went in (mAh, discharge positive), never corrected.
    pub used_mah: f64,
    /// The published charge: counted, corrected towards the voltage at rest.
    #[serde(with = "nan_as_null")]
    pub soc: f64,
    pub bins: Vec<PackBin>,
}

impl Pack {
    fn new(id: u32, unix: f64, anchor: Anchor, start_soc: f64, start_ocv: f64) -> Self {
        Pack { id, inserted_unix: unix, anchor, start_soc, start_ocv, used_mah: 0.0, soc: start_soc, bins: Vec::new() }
    }
}

/// A point of a finished pack's discharge: voltage bin and its charge there.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    pub bin: i32,
    pub soc: f64,
}

/// A pack that came out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PackRecord {
    pub id: u32,
    pub inserted_unix: f64,
    pub removed_unix: f64,
    pub how: How,
    pub anchor: Anchor,
    #[serde(with = "nan_as_null")]
    pub start_soc: f64,
    pub end_soc: f64,
    pub used_mah: f64,
    /// used / (start - end), if it was enough to learn from.
    pub capacity_mah: Option<f64>,
    pub weight: f64,
    /// Rejected as an outlier by the latest capacity update (its curve points are left out too).
    pub outlier: bool,
    pub curve: Vec<CurvePoint>,
}

/// Everything persisted in battery_learner.json.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    pub version: u32,
    pub pack: Pack,
    /// Smoothed open-circuit voltage at the last reading, compared with the first one at start-up.
    pub last_ocv: Option<f64>,
    pub last_seen_unix: f64,
    /// Learned capacity (mAh); None until a pack taught one.
    pub learned_capacity_mah: Option<f64>,
    /// Offset added to the voltage before the stock curve (the prior).
    pub prior_offset_v: f64,
    /// The first run's seed point (V, fraction), for the record.
    pub seed: Option<[f64; 2]>,
    pub history: Vec<PackRecord>,
}

/// One discharge sample for the pack's CSV file.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub unix: f64,
    pub pack: u32,
    pub voltage: f64,
    pub current_a: f64,
    pub ocv: f64,
    pub used_mah: f64,
    pub soc: f64,
    pub soc_voltage: f64,
    pub anchor: Anchor,
}

impl Sample {
    pub const HEADER: &'static str = "unix_time,pack,voltage_v,current_a,ocv_v,used_mah,soc,soc_voltage,anchor";

    pub fn csv(&self) -> String {
        format!(
            "{:.1},{},{:.3},{:.3},{:.3},{:.2},{:.4},{:.4},{}",
            self.unix,
            self.pack,
            self.voltage,
            self.current_a,
            self.ocv,
            self.used_mah,
            self.soc,
            self.soc_voltage,
            if self.anchor == Anchor::Full { "full" } else { "voltage" }
        )
    }
}

/// What a reading produced.
#[derive(Debug, Default)]
pub struct Output {
    /// The frame was new (the driver repeats its latest frame while it waits for the next).
    pub fresh: bool,
    /// Lines for the INFO log.
    pub logs: Vec<String>,
    pub sample: Option<Sample>,
    /// Write the state file now (swap, learning, first estimate).
    pub save: bool,
}

/// The published estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub soc: f64,
    pub voltage: f64,
    pub current_a: f64,
    pub ocv: f64,
    pub soc_voltage: f64,
    pub capacity_mah: f64,
    pub pack: u32,
}

#[derive(Clone, Debug)]
enum Kind {
    Startup { saved: Option<f64> },
    Gap { before: Option<f64>, gap_s: f64 },
}

#[derive(Clone, Debug)]
struct Pending {
    kind: Kind,
    readings: Vec<f64>,
}

pub struct Learner {
    pub p: Params,
    pub state: State,
    /// Fitted charge at each voltage bin centre.
    fit: Vec<f64>,
    /// No state file: the first reading seeds the pack from the curve.
    fresh: bool,
    pending: Option<Pending>,
    last_stamp: Option<f64>,
    last_current_ma: f64,
    last_voltage: f64,
    ocv: Option<f64>,
    rest_since: Option<f64>,
    next_sample: f64,
    /// Frame stamp of the last detected (not manual) swap.
    last_detected_swap: Option<f64>,
}

/// Inverse of the stock curve: cell voltage at this charge.
pub fn li_ion_voltage(fraction: f64) -> f64 {
    let f = fraction.clamp(0.0, 1.0);
    for w in LI_ION_CURVE.windows(2) {
        let ((v0, f0), (v1, f1)) = (w[0], w[1]);
        if f <= f1 {
            return v0 + (v1 - v0) * (f - f0) / (f1 - f0);
        }
    }
    LI_ION_CURVE[LI_ION_CURVE.len() - 1].0
}

/// Pool-adjacent-violators: the non-decreasing sequence closest to `values` in weighted least squares.
pub fn isotonic(values: &[f64], weights: &[f64]) -> Vec<f64> {
    // (weighted sum, weight, count) per block
    let mut blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(values.len());
    for (&v, &w) in values.iter().zip(weights) {
        let w = w.max(1e-12);
        blocks.push((v * w, w, 1));
        while blocks.len() >= 2 {
            let (s1, w1, _) = blocks[blocks.len() - 1];
            let (s0, w0, _) = blocks[blocks.len() - 2];
            if s0 / w0 <= s1 / w1 {
                break;
            }
            let (_, _, n1) = blocks.pop().unwrap();
            let last = blocks.last_mut().unwrap();
            *last = (s0 + s1, w0 + w1, last.2 + n1);
        }
    }
    blocks.into_iter().flat_map(|(s, w, n)| std::iter::repeat(s / w).take(n)).collect()
}

/// Weighted median (the value where the cumulative weight reaches half).
fn weighted_median(mut items: Vec<(f64, f64)>) -> Option<f64> {
    items.retain(|(v, w)| v.is_finite() && *w > 0.0);
    if items.is_empty() {
        return None;
    }
    items.sort_by(|a, b| a.0.total_cmp(&b.0));
    let half = items.iter().map(|(_, w)| w).sum::<f64>() / 2.0;
    let mut acc = 0.0;
    for (v, w) in &items {
        acc += w;
        if acc >= half - 1e-12 {
            return Some(*v);
        }
    }
    items.last().map(|(v, _)| *v)
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn bin_of(ocv: f64) -> i32 {
    (((ocv - BIN_V0) / BIN_V).floor() as i32).clamp(0, BINS as i32 - 1)
}

fn bin_centre(bin: usize) -> f64 {
    BIN_V0 + (bin as f64 + 0.5) * BIN_V
}

fn pct(f: f64) -> String {
    format!("{:.0} %", f * 100.0)
}

impl Learner {
    /// `saved`: the state file's contents, None on the first run.
    pub fn new(p: Params, saved: Option<State>, unix: f64) -> Self {
        // A state saved before its first reading has no charge yet: seed it like a first run.
        let fresh = saved.as_ref().map_or(true, |s| !s.pack.soc.is_finite());
        let state = saved.unwrap_or_else(|| {
            let seeded = p.seed_voltage > 0.0 && p.seed_fraction > 0.0 && p.seed_fraction < 1.0;
            State {
                version: STATE_VERSION,
                pack: Pack::new(1, unix, Anchor::Voltage, f64::NAN, f64::NAN),
                last_ocv: None,
                last_seen_unix: unix,
                learned_capacity_mah: None,
                prior_offset_v: if seeded { p.cells * li_ion_voltage(p.seed_fraction) - p.seed_voltage } else { 0.0 },
                seed: seeded.then_some([p.seed_voltage, p.seed_fraction]),
                history: Vec::new(),
            }
        });
        let saved_ocv = state.last_ocv;
        let mut l = Learner {
            p,
            state,
            fit: Vec::new(),
            fresh,
            pending: Some(Pending { kind: Kind::Startup { saved: saved_ocv }, readings: Vec::new() }),
            last_stamp: None,
            last_current_ma: 0.0,
            last_voltage: f64::NAN,
            ocv: None,
            rest_since: None,
            next_sample: f64::NEG_INFINITY,
            last_detected_swap: None,
        };
        l.refit();
        l
    }

    /// Capacity in use (mAh): learned, or the prior.
    pub fn capacity_mah(&self) -> f64 {
        self.state.learned_capacity_mah.unwrap_or(self.p.capacity_mah)
    }

    /// The stock curve shifted by the fitted offset.
    pub fn prior_soc(&self, ocv: f64) -> f64 {
        li_ion_fraction((ocv + self.state.prior_offset_v) / self.p.cells)
    }

    /// Charge at this open-circuit voltage on the learned curve.
    pub fn curve_soc(&self, ocv: f64) -> f64 {
        if !ocv.is_finite() {
            return f64::NAN;
        }
        let x = (ocv - BIN_V0) / BIN_V - 0.5;
        if x <= 0.0 {
            return self.fit[0];
        }
        let i = x.floor() as usize;
        if i + 1 >= self.fit.len() {
            return self.fit[self.fit.len() - 1];
        }
        let t = x - i as f64;
        self.fit[i] + t * (self.fit[i + 1] - self.fit[i])
    }

    /// The fitted curve as (bin centre V, charge) pairs.
    pub fn curve(&self) -> Vec<(f64, f64)> {
        self.fit.iter().enumerate().map(|(i, &f)| (bin_centre(i), f)).collect()
    }

    pub fn estimate(&self) -> Option<Estimate> {
        let soc = self.state.pack.soc;
        let ocv = self.ocv?;
        if self.pending.is_some() || !soc.is_finite() {
            return None;
        }
        Some(Estimate {
            soc,
            voltage: self.last_voltage,
            current_a: self.last_current_ma / 1000.0,
            ocv,
            soc_voltage: self.curve_soc(ocv),
            capacity_mah: self.capacity_mah(),
            pack: self.state.pack.id,
        })
    }

    /// A /roomba/sensors frame: its stamp (s), wall-clock time, voltage and current.
    pub fn on_frame(&mut self, stamp: f64, unix: f64, voltage_mv: i32, current_ma: i32) -> Output {
        let mut out = Output::default();
        if voltage_mv < MIN_VOLTAGE_MV || self.last_stamp == Some(stamp) {
            return out;
        }
        out.fresh = true;
        let v = voltage_mv as f64 / 1000.0;
        let i = current_ma as f64;
        let ocv_now = v - i / 1000.0 * self.p.resistance_ohm;
        // A stamp going backwards (clock step) counts as a gap too.
        let dt = self.last_stamp.map(|t| stamp - t);
        let gap = dt.is_some_and(|dt| !(dt > 0.0 && dt <= self.p.gap_s));
        if gap && self.pending.is_none() {
            let gap_s = dt.unwrap_or(0.0);
            self.pending = Some(Pending { kind: Kind::Gap { before: self.ocv, gap_s }, readings: Vec::new() });
        }
        self.last_stamp = Some(stamp);
        self.last_voltage = v;
        let prev_i = std::mem::replace(&mut self.last_current_ma, i);
        self.state.last_seen_unix = unix;

        if let Some(pending) = &mut self.pending {
            pending.readings.push(ocv_now);
            if pending.readings.len() >= self.p.settle_samples.max(1) {
                let pending = self.pending.take().unwrap();
                let m = median(&pending.readings);
                self.resolve(pending.kind, m, unix, &mut out);
                self.ocv = Some(m);
                self.state.last_ocv = Some(m);
                self.rest_since = None;
                out.save = true;
            }
            return out;
        }
        let dt = dt.unwrap_or(0.0);

        let used = -(i + prev_i) / 2.0 * dt / 3600.0;
        let capacity = self.capacity_mah();
        let alpha = 1.0 - (-dt / self.p.smoothing_s.max(1e-3)).exp();
        let ocv = self.ocv.map_or(ocv_now, |o| o + alpha * (ocv_now - o));
        self.ocv = Some(ocv);
        self.state.last_ocv = Some(ocv);
        let soc_v = self.curve_soc(ocv);
        let pack = &mut self.state.pack;
        pack.used_mah += used;
        pack.soc -= used / capacity;
        if i.abs() <= self.p.rest_current_ma {
            let since = *self.rest_since.get_or_insert(stamp);
            if stamp - since >= self.p.rest_settle_s {
                let tau = if pack.anchor == Anchor::Full { self.p.correction_tau_s } else { self.p.uncertain_correction_tau_s };
                let k = (dt / tau.max(1e-3)).min(1.0);
                pack.soc += k * (soc_v - pack.soc);
            }
        } else {
            self.rest_since = None;
        }
        pack.soc = pack.soc.clamp(0.0, 1.0);

        if stamp >= self.next_sample {
            self.next_sample = stamp + self.p.sample_period_s;
            let bin = bin_of(ocv);
            match pack.bins.iter_mut().find(|b| b.bin == bin) {
                Some(b) => {
                    b.n += 1;
                    b.used_sum_mah += pack.used_mah;
                }
                None => pack.bins.push(PackBin { bin, n: 1, used_sum_mah: pack.used_mah }),
            }
            out.sample = Some(Sample {
                unix,
                pack: pack.id,
                voltage: v,
                current_a: i / 1000.0,
                ocv,
                used_mah: pack.used_mah,
                soc: pack.soc,
                soc_voltage: soc_v,
                anchor: pack.anchor,
            });
        }
        out
    }

    /// The first readings at start-up or after a gap have settled at open-circuit voltage `m`.
    fn resolve(&mut self, kind: Kind, m: f64, unix: f64, out: &mut Output) {
        match kind {
            Kind::Startup { .. } if self.fresh => {
                self.fresh = false;
                let soc = self.curve_soc(m);
                let id = self.state.pack.id;
                self.state.pack = Pack::new(id, unix, Anchor::Voltage, soc, m);
                let seed = match self.state.seed {
                    Some([v, f]) => format!(" shifted to pass through the seed point {v:.2} V = {}", pct(f)),
                    None => String::new(),
                };
                out.logs.push(format!(
                    "no saved charge: pack {id} starts at {} from its voltage ({m:.2} V on the stock Li-ion curve{seed}); \
                     the first swap at 1 bar starts the learning",
                    pct(soc)
                ));
            }
            Kind::Startup { saved: Some(before) } if m - before >= self.p.swap_jump_v => {
                self.swap(How::Startup, m, before, unix, out);
            }
            Kind::Startup { saved } => {
                let saved = saved.map_or("none".into(), |s| format!("{s:.2} V"));
                out.logs.push(format!(
                    "continuing pack {} at {} ({m:.2} V; saved {saved}), capacity {:.0} mAh{}",
                    self.state.pack.id,
                    pct(self.state.pack.soc),
                    self.capacity_mah(),
                    if self.state.learned_capacity_mah.is_some() { " (learned)" } else { " (prior)" }
                ));
            }
            Kind::Gap { before: Some(before), .. } if m - before >= self.p.swap_jump_v => {
                self.swap(How::Gap, m, before, unix, out);
            }
            Kind::Gap { before, gap_s } => {
                if gap_s.abs() >= 10.0 {
                    let change = before.map_or(String::new(), |b| format!(", {:+.2} V", m - b));
                    out.logs.push(format!("sensor data back after {gap_s:.0} s{change}: same pack"));
                }
            }
        }
    }

    fn swap(&mut self, how: How, m: f64, before: f64, unix: f64, out: &mut Output) {
        self.finish_pack(how, unix, out);
        let soc_v = self.curve_soc(m);
        let (anchor, start) = if soc_v >= self.p.full_pack_min_fraction { (Anchor::Full, 1.0) } else { (Anchor::Voltage, soc_v) };
        let id = self.state.pack.id + 1;
        self.state.pack = Pack::new(id, unix, anchor, start, m);
        self.last_detected_swap = self.last_stamp;
        out.logs.push(format!(
            "new pack ({}: {before:.2} V -> {m:.2} V): pack {id} starts at {}{}",
            how.as_str(),
            pct(start),
            if anchor == Anchor::Voltage { " from its voltage (part-charged?)" } else { " (full)" }
        ));
        out.save = true;
    }

    /// The `~/pack_swapped` service: a full pack is in the robot now. Returns the reply and the
    /// lines to log.
    pub fn manual_swap(&mut self, unix: f64) -> (String, Vec<String>) {
        let mut out = Output::default();
        let message;
        // Seconds (stamp time) since a detected swap, if within manual_merge_s.
        let since_detected = match (self.last_detected_swap, self.last_stamp) {
            (Some(swap), Some(now)) if now - swap <= self.p.manual_merge_s => Some(now - swap),
            _ => None,
        };
        if let Some(since) = since_detected {
            // The swap was already detected: only make sure it counts as a full pack.
            let pack = &mut self.state.pack;
            if pack.anchor != Anchor::Full || pack.start_soc != 1.0 {
                pack.soc = (pack.soc + 1.0 - pack.start_soc).clamp(0.0, 1.0);
                pack.anchor = Anchor::Full;
                pack.start_soc = 1.0;
            }
            message = format!("pack {} (swap detected {since:.0} s ago) is marked full: {}", pack.id, pct(pack.soc));
        } else {
            self.finish_pack(How::Manual, unix, &mut out);
            let ocv = self.ocv.unwrap_or(f64::NAN);
            let id = self.state.pack.id + 1;
            self.state.pack = Pack::new(id, unix, Anchor::Full, 1.0, ocv);
            message = format!("new pack {id} at 100 %");
            self.fresh = false;
            // Data settling after a gap or at start-up must not swap again.
            if let Some(p) = &mut self.pending {
                p.kind = Kind::Gap { before: None, gap_s: 0.0 };
            }
            self.rest_since = None;
        }
        self.last_detected_swap = None;
        out.logs.insert(0, format!("manual reset: {message}"));
        (message, out.logs)
    }

    /// The pack that comes out was at 1 bar: learn its capacity and curve.
    fn finish_pack(&mut self, how: How, unix: f64, out: &mut Output) {
        let pack = self.state.pack.clone();
        let end = self.p.one_bar_fraction;
        let span = pack.start_soc - end;
        let mut record = PackRecord {
            id: pack.id,
            inserted_unix: pack.inserted_unix,
            removed_unix: unix,
            how,
            anchor: pack.anchor,
            start_soc: pack.start_soc,
            end_soc: end,
            used_mah: pack.used_mah,
            capacity_mah: None,
            weight: 0.0,
            outlier: false,
            curve: Vec::new(),
        };
        let estimate = pack.used_mah / span;
        // A manual reset only re-anchors: without a detected swap there is no telling whether the
        // pack really came out at 1 bar (a pack can't be swapped unnoticed: the Roomba loses power).
        let enough = span.is_finite()
            && span >= MIN_LEARN_SPAN
            && pack.used_mah >= MIN_LEARN_MAH
            && (CAPACITY_RANGE.0..=CAPACITY_RANGE.1).contains(&estimate);
        let learnable = enough && how != How::Manual;
        if learnable {
            record.capacity_mah = Some(estimate);
            record.weight = if pack.anchor == Anchor::Full { 1.0 } else { self.p.voltage_anchor_weight };
            let mut bins = pack.bins.clone();
            bins.sort_by_key(|b| b.bin);
            record.curve = bins
                .iter()
                .map(|b| CurvePoint {
                    bin: b.bin,
                    soc: (pack.start_soc - span * (b.used_sum_mah / b.n as f64) / pack.used_mah).clamp(0.0, 1.0),
                })
                .collect();
        }
        self.state.history.push(record);
        if self.state.history.len() > HISTORY_LEN {
            let excess = self.state.history.len() - HISTORY_LEN;
            self.state.history.drain(..excess);
        }
        let rejected = self.update_capacity();
        self.refit();
        let start = if pack.start_soc.is_finite() { pct(pack.start_soc) } else { "unknown".into() };
        if learnable {
            let n = self.state.history.iter().filter(|r| r.capacity_mah.is_some() && !r.outlier).count();
            let rejected = if rejected.is_empty() {
                String::new()
            } else {
                format!("; rejected as outliers: {}", rejected.iter().map(|c| format!("{c:.0}")).collect::<Vec<_>>().join(", "))
            };
            out.logs.push(format!(
                "pack {} out at 1 bar ({}) after {:.0} mAh from {start} ({}): capacity estimate {estimate:.0} mAh; \
                 learned capacity {:.0} mAh from {n} pack(s){rejected}; voltage curve offset {:+.2} V",
                pack.id,
                pct(end),
                pack.used_mah,
                if pack.anchor == Anchor::Full { "went in full" } else { "started from its voltage" },
                self.capacity_mah(),
                self.state.prior_offset_v
            ));
        } else if how == How::Manual {
            out.logs.push(format!(
                "pack {} out after {:.0} mAh from {start}: not learned from (no swap was detected)",
                pack.id, pack.used_mah
            ));
        } else {
            out.logs.push(format!(
                "pack {} out after {:.0} mAh from {start}: too little to learn from (needs {:.0} mAh and {} between its anchors)",
                pack.id,
                pack.used_mah,
                MIN_LEARN_MAH,
                pct(MIN_LEARN_SPAN)
            ));
        }
        out.save = true;
    }

    /// Recomputes the learned capacity from the latest estimates; returns those rejected as outliers.
    fn update_capacity(&mut self) -> Vec<f64> {
        let n = self.state.history.len();
        let recent: Vec<usize> = (0..n).filter(|&i| self.state.history[i].capacity_mah.is_some()).collect();
        let recent = &recent[recent.len().saturating_sub(ESTIMATES_USED)..];
        for r in &mut self.state.history {
            r.outlier = false;
        }
        if recent.is_empty() {
            self.state.learned_capacity_mah = None;
            return Vec::new();
        }
        let items: Vec<(f64, f64)> = recent.iter().map(|&i| (self.state.history[i].capacity_mah.unwrap(), self.state.history[i].weight)).collect();
        let med = weighted_median(items.clone()).unwrap_or(self.p.capacity_mah);
        let mut rejected = Vec::new();
        let (mut sum, mut weight) = (self.p.capacity_mah * self.p.capacity_prior_weight, self.p.capacity_prior_weight);
        for (&i, &(c, w)) in recent.iter().zip(&items) {
            if recent.len() >= MIN_FOR_OUTLIERS && (c - med).abs() > self.p.outlier_fraction * med {
                self.state.history[i].outlier = true;
                rejected.push(c);
            } else {
                sum += c * w;
                weight += w;
            }
        }
        self.state.learned_capacity_mah = Some(sum / weight);
        rejected
    }

    /// Refits the voltage curve from the finished packs (outliers left out) over the shifted prior.
    fn refit(&mut self) {
        let mut w = vec![0.0; BINS];
        let mut s = vec![0.0; BINS];
        for r in self.state.history.iter().filter(|r| r.weight > 0.0 && !r.outlier) {
            for pt in &r.curve {
                let b = pt.bin.clamp(0, BINS as i32 - 1) as usize;
                w[b] += r.weight;
                s[b] += r.weight * pt.soc;
            }
        }
        // Shift the stock curve onto the data (weighted median offset over the sloped part), so the
        // prior meets the data where the data ends.
        let offsets: Vec<(f64, f64)> = (0..BINS)
            .filter(|&b| w[b] > 0.0)
            .filter_map(|b| {
                let label = s[b] / w[b];
                (0.05..=0.95).contains(&label).then(|| (self.p.cells * li_ion_voltage(label) - bin_centre(b), w[b]))
            })
            .collect();
        if let Some(offset) = weighted_median(offsets) {
            self.state.prior_offset_v = offset;
        }
        let w0 = self.p.curve_prior_weight.max(1e-6);
        let values: Vec<f64> = (0..BINS).map(|b| (w0 * self.prior_soc(bin_centre(b)) + s[b]) / (w0 + w[b])).collect();
        let weights: Vec<f64> = (0..BINS).map(|b| w0 + w[b]).collect();
        self.fit = isotonic(&values, &weights);
    }
}

/// Reads the state file; Ok(None) if there is none.
pub fn load_state(path: &Path) -> anyhow::Result<Option<State>> {
    match fs::read_to_string(path) {
        Ok(text) => {
            let state: State = serde_json::from_str(&text)?;
            anyhow::ensure!(state.version == STATE_VERSION, "state file version {} (expected {STATE_VERSION})", state.version);
            Ok(Some(state))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Writes the state file atomically: a temporary file in the same directory, synced, renamed over it.
pub fn save_state(path: &Path, state: &State) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir)?;
    let tmp = path.with_extension("json.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(serde_json::to_string_pretty(state).map_err(std::io::Error::other)?.as_bytes())?;
        f.write_all(b"\n")?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    if let Ok(d) = fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

/// The discharge log of a pack: battery_pack_NNNN.csv next to the state file.
pub fn csv_path(dir: &Path, pack: u32) -> PathBuf {
    dir.join(format!("battery_pack_{pack:04}.csv"))
}

/// Appends a sample to its pack's CSV file (with a header line if the file is new).
pub fn append_sample(dir: &Path, s: &Sample) -> std::io::Result<()> {
    let path = csv_path(dir, s.pack);
    let new = !path.exists();
    let mut f = fs::OpenOptions::new().create(true).append(true).open(path)?;
    if new {
        writeln!(f, "{}", Sample::HEADER)?;
    }
    writeln!(f, "{}", s.csv())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simulated pack: open-circuit voltage = stock curve shifted by `offset`, 0.2 ohm.
    struct Sim {
        offset: f64,
        capacity: f64,
        soc: f64,
        r: f64,
    }

    impl Sim {
        fn mv(&self, current_ma: f64) -> i32 {
            let ocv = 5.0 * li_ion_voltage(self.soc) - self.offset;
            ((ocv + current_ma / 1000.0 * self.r) * 1000.0).round() as i32
        }
        fn draw(&mut self, current_ma: f64, dt: f64) {
            self.soc -= -current_ma * dt / 3600.0 / self.capacity;
        }
    }

    fn params() -> Params {
        Params { resistance_ohm: 0.2, ..Params::default() }
    }

    /// Discharges `sim` at `current_ma` (negative) for `seconds` at 1 Hz from stamp `*t`.
    fn run(l: &mut Learner, sim: &mut Sim, t: &mut f64, seconds: f64, current_ma: f64) -> Vec<String> {
        let mut logs = Vec::new();
        let end = *t + seconds;
        while *t < end {
            sim.draw(current_ma, 1.0);
            *t += 1.0;
            logs.extend(l.on_frame(*t, *t, sim.mv(current_ma), current_ma as i32).logs);
        }
        logs
    }

    /// Runs until the simulated pack is at `soc`.
    fn run_to(l: &mut Learner, sim: &mut Sim, t: &mut f64, soc: f64, current_ma: f64) -> Vec<String> {
        let mut logs = Vec::new();
        while sim.soc > soc {
            logs.extend(run(l, sim, t, 1.0, current_ma));
        }
        logs
    }

    fn started(state: Option<State>, sim: &Sim, t: &mut f64) -> (Learner, Vec<String>) {
        let mut l = Learner::new(params(), state, *t);
        let mut logs = Vec::new();
        for _ in 0..5 {
            *t += 1.0;
            logs.extend(l.on_frame(*t, *t, sim.mv(-200.0), -200).logs);
        }
        assert!(l.estimate().is_some());
        (l, logs)
    }

    #[test]
    fn isotonic_pools_violators() {
        assert_eq!(isotonic(&[1.0, 3.0, 2.0, 4.0], &[1.0; 4]), vec![1.0, 2.5, 2.5, 4.0]);
        assert_eq!(isotonic(&[3.0, 1.0], &[3.0, 1.0]), vec![2.5, 2.5]);
        let fit = isotonic(&[0.5, 0.4, 0.3, 0.9, 0.8, 1.0], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert!(fit.windows(2).all(|w| w[1] >= w[0]));
        assert!((fit[0] - 0.4).abs() < 1e-12 && (fit[3] - 0.85).abs() < 1e-12);
        let sorted = [0.1, 0.2, 0.3];
        assert_eq!(isotonic(&sorted, &[1.0; 3]), sorted.to_vec());
    }

    #[test]
    fn inverse_curve() {
        for f in [0.0, 0.03, 0.15, 0.5, 0.62, 0.99, 1.0] {
            assert!((li_ion_fraction(li_ion_voltage(f)) - f).abs() < 1e-9);
        }
    }

    #[test]
    fn seed_shifts_the_prior_through_the_seed_point() {
        let p = Params { seed_voltage: 18.29, seed_fraction: 0.62, ..params() };
        let l = Learner::new(p, None, 0.0);
        assert!((l.curve_soc(18.29) - 0.62).abs() < 0.01);
        assert!((l.state.prior_offset_v - (5.0 * 3.864 - 18.29)).abs() < 1e-9);
        // The stock curve alone calls 18.29 V about 18 % (the driver's gauge said 17 %).
        assert!((Learner::new(params(), None, 0.0).curve_soc(18.29) - 0.18).abs() < 0.01);
    }

    #[test]
    fn first_run_starts_from_the_seeded_curve() {
        let p = Params { seed_voltage: 18.29, seed_fraction: 0.62, ..params() };
        let sim = Sim { offset: 1.03, capacity: 2500.0, soc: 0.55, r: 0.2 };
        let mut l = Learner::new(p, None, 0.0);
        let mut logs = Vec::new();
        for k in 1..=5 {
            assert!(l.estimate().is_none());
            logs.extend(l.on_frame(k as f64, k as f64, sim.mv(-200.0), -200).logs);
        }
        let e = l.estimate().unwrap();
        assert!((e.soc - 0.55).abs() < 0.02, "{e:?}");
        assert_eq!((l.state.pack.id, l.state.pack.anchor), (1, Anchor::Voltage));
        assert_eq!(l.state.seed, Some([18.29, 0.62]));
        assert!(logs[0].starts_with("no saved charge: pack 1 starts at 55 %"), "{logs:?}");
    }

    #[test]
    fn coulomb_counting() {
        let mut t = 0.0;
        let mut sim = Sim { offset: 0.0, capacity: 2000.0, soc: 1.0, r: 0.2 };
        let (mut l, _) = started(None, &sim, &mut t);
        l.state.pack = Pack::new(1, 0.0, Anchor::Full, 1.0, 21.0);
        // 1 A for an hour, above rest_current_ma so no voltage correction: 1000 mAh of 2000 (the
        // first step averages with the -200 mA before it).
        run(&mut l, &mut sim, &mut t, 3600.0, -1000.0);
        assert!((l.state.pack.used_mah - 1000.0).abs() < 0.2);
        assert!((l.estimate().unwrap().soc - 0.5).abs() < 1e-4);
        // Repeated frames (same stamp) count once; charging counts back (trapezoid -1000 -> +500 mA).
        let used = l.state.pack.used_mah;
        let mv = sim.mv(500.0);
        assert!(l.on_frame(t + 1.0, 0.0, mv, 500).fresh);
        assert!(!l.on_frame(t + 1.0, 0.0, mv, 500).fresh);
        assert!((l.state.pack.used_mah - (used + 250.0 / 3600.0)).abs() < 1e-9);
    }

    #[test]
    fn rest_correction_pulls_towards_the_voltage() {
        let mut t = 0.0;
        let mut sim = Sim { offset: 0.0, capacity: 2000.0, soc: 0.5, r: 0.2 };
        let (mut l, _) = started(None, &sim, &mut t);
        // A voltage-anchored pack 20 points too high converges with uncertain_correction_tau_s
        // (1200 s) once it has been at rest for rest_settle_s (60 s). No current: no counting.
        l.state.pack.soc = 0.7;
        run(&mut l, &mut sim, &mut t, 60.0 + 1200.0, 0.0);
        let soc = l.estimate().unwrap().soc;
        assert!((soc - (0.5 + 0.2 * (-1.0f64).exp())).abs() < 0.01, "{soc}");
        // Under load (above rest_current_ma) only the counter moves: 1.5 A for 10 min of 2000 mAh.
        l.state.pack.soc = 0.7;
        run(&mut l, &mut sim, &mut t, 600.0, -1500.0);
        let expected = 0.7 - (599.0 * 1500.0 + 750.0) / 3600.0 / 2000.0;
        assert!((l.estimate().unwrap().soc - expected).abs() < 1e-6);
    }

    #[test]
    fn gap_with_jump_is_a_swap_and_without_is_not() {
        let mut t = 0.0;
        let mut sim = Sim { offset: 0.0, capacity: 2000.0, soc: 0.5, r: 0.2 };
        let (mut l, _) = started(None, &sim, &mut t);
        run(&mut l, &mut sim, &mut t, 100.0, -400.0);
        // Driver restart: 30 s without data, same voltage (a little recovery): same pack.
        t += 30.0;
        let logs = run(&mut l, &mut sim, &mut t, 10.0, -200.0);
        assert_eq!(l.state.pack.id, 1);
        assert!(logs.iter().any(|m| m.contains("sensor data back after 31 s") && m.ends_with("same pack")), "{logs:?}");
        // A short stall (3 s) without a jump: nothing to say.
        t += 3.0;
        assert!(run(&mut l, &mut sim, &mut t, 10.0, -200.0).is_empty());
        // Pack pulled for 2 minutes, a full one goes in: +1.3 V.
        t += 120.0;
        sim.soc = 1.0;
        let logs = run(&mut l, &mut sim, &mut t, 10.0, -200.0);
        assert_eq!(l.state.pack.id, 2);
        assert_eq!(l.state.pack.anchor, Anchor::Full);
        assert!((l.estimate().unwrap().soc - 1.0).abs() < 0.001);
        assert!(logs.iter().any(|m| m.starts_with("new pack (voltage jump after a sensor gap") && m.contains("pack 2 starts at 100 %")), "{logs:?}");
        assert_eq!(l.state.history.len(), 1);
        assert_eq!(l.state.history[0].how, How::Gap);
    }

    #[test]
    fn part_charged_pack_starts_from_its_voltage() {
        let mut t = 0.0;
        let mut sim = Sim { offset: 0.0, capacity: 2000.0, soc: 0.2, r: 0.2 };
        let (mut l, _) = started(None, &sim, &mut t);
        t += 60.0;
        sim.soc = 0.55;
        run(&mut l, &mut sim, &mut t, 10.0, -200.0);
        assert_eq!((l.state.pack.id, l.state.pack.anchor), (2, Anchor::Voltage));
        assert!((l.estimate().unwrap().soc - 0.55).abs() < 0.02);
    }

    #[test]
    fn startup_jump_is_a_swap() {
        let mut t = 0.0;
        let mut sim = Sim { offset: 0.0, capacity: 2000.0, soc: 0.3, r: 0.2 };
        let (mut l, _) = started(None, &sim, &mut t);
        run(&mut l, &mut sim, &mut t, 60.0, -400.0);
        let saved = l.state.clone();
        // Restart with the same pack: continues where it was.
        let (l2, logs) = started(Some(saved.clone()), &sim, &mut t);
        assert_eq!(l2.state.pack.id, 1);
        assert!((l2.estimate().unwrap().soc - saved.pack.soc).abs() < 1e-9);
        assert!(logs[0].starts_with("continuing pack 1 at"), "{logs:?}");
        // The Jetson was off while the pack was swapped.
        sim.soc = 1.0;
        let (l3, logs) = started(Some(saved), &sim, &mut t);
        assert_eq!((l3.state.pack.id, l3.state.pack.anchor), (2, Anchor::Full));
        assert!(logs.iter().any(|m| m.contains("voltage jump since the last run")), "{logs:?}");
        assert_eq!(l3.state.history[0].how, How::Startup);
    }

    #[test]
    fn manual_reset() {
        let mut t = 0.0;
        let mut sim = Sim { offset: 0.0, capacity: 2000.0, soc: 0.4, r: 0.2 };
        let (mut l, _) = started(None, &sim, &mut t);
        run(&mut l, &mut sim, &mut t, 60.0, -400.0);
        let (reply, logs) = l.manual_swap(t);
        assert!(reply.starts_with("new pack 2 at 100 %"), "{reply}");
        assert!(logs[0].starts_with("manual reset: new pack 2") && logs[1].contains("not learned from"), "{logs:?}");
        assert_eq!((l.state.pack.anchor, l.state.pack.soc), (Anchor::Full, 1.0));
        assert_eq!(l.state.history[0].how, How::Manual);
        // A detected swap to a part-charged-looking pack, then the button: one swap, marked full.
        sim.soc = 0.2;
        run(&mut l, &mut sim, &mut t, 60.0, -400.0);
        t += 60.0;
        sim.soc = 0.6;
        run(&mut l, &mut sim, &mut t, 30.0, -200.0);
        assert_eq!((l.state.pack.id, l.state.pack.anchor), (3, Anchor::Voltage));
        let (reply, _) = l.manual_swap(t);
        assert!(reply.starts_with("pack 3 (swap detected"), "{reply}");
        assert_eq!((l.state.pack.id, l.state.pack.anchor, l.state.history.len()), (3, Anchor::Full, 2));
        assert!(l.state.pack.soc > 0.99);
        // Long (manual_merge_s) after a detected swap, the button is a swap of its own. It re-anchors
        // but teaches nothing: the half-used pack would have "measured" 1176 mAh.
        t += 60.0;
        sim.soc = 1.0;
        run_to(&mut l, &mut sim, &mut t, 0.5, -1000.0);
        assert_eq!(l.state.pack.id, 4);
        let (reply, logs) = l.manual_swap(t);
        assert!(reply.starts_with("new pack 5 at 100 %"), "{reply}");
        assert!(logs[1].contains("not learned from (no swap was detected)"), "{logs:?}");
        assert_eq!((l.state.learned_capacity_mah, l.state.history[3].capacity_mah), (None, None));
    }

    #[test]
    fn learns_capacity_and_rejects_outliers() {
        let mut t = 0.0;
        let mut sim = Sim { offset: 1.0, capacity: 2500.0, soc: 1.0, r: 0.2 };
        let (mut l, _) = started(None, &sim, &mut t);
        l.state.pack = Pack::new(1, 0.0, Anchor::Full, 1.0, 20.0);
        for k in 0..3 {
            run_to(&mut l, &mut sim, &mut t, 0.15, -1000.0);
            t += 60.0;
            sim.soc = 1.0;
            let logs = run(&mut l, &mut sim, &mut t, 10.0, -200.0);
            assert_eq!(l.state.pack.id, k + 2);
            assert!(logs.iter().any(|m| m.contains("capacity estimate 2500 mAh")), "{logs:?}");
        }
        // 3 estimates of 2500 and the prior 2000 at weight 0.2
        let learned = l.state.learned_capacity_mah.unwrap();
        assert!((learned - (3.0 * 2500.0 + 0.2 * 2000.0) / 3.2).abs() < 2.0, "{learned}");
        // A mistaken early swap (pack at ~70 %) gives a low estimate: rejected.
        run_to(&mut l, &mut sim, &mut t, 0.7, -1000.0);
        t += 60.0;
        sim.soc = 1.0;
        let logs = run(&mut l, &mut sim, &mut t, 10.0, -200.0);
        assert!(logs.iter().any(|m| m.contains("rejected as outliers: 882")), "{logs:?}");
        assert!((l.state.learned_capacity_mah.unwrap() - learned).abs() < 1e-9);
        assert!(l.state.history.last().unwrap().outlier);
        // Counting with the learned capacity, the next pack reads ~1 bar when it is.
        run_to(&mut l, &mut sim, &mut t, 0.15, -1000.0);
        assert!((l.estimate().unwrap().soc - 0.15).abs() < 0.03, "{:?}", l.estimate());
    }

    #[test]
    fn learns_the_voltage_curve() {
        // The pack reads 1 V below the stock curve; the prior (no seed) is off by up to ~30 points.
        let mut t = 0.0;
        let mut sim = Sim { offset: 1.0, capacity: 2000.0, soc: 1.0, r: 0.2 };
        let (mut l, _) = started(None, &sim, &mut t);
        l.state.pack = Pack::new(1, 0.0, Anchor::Full, 1.0, 20.0);
        let truth = |v: f64| li_ion_fraction((v + 1.0) / 5.0);
        let before = l.curve_soc(18.2);
        assert!((before - truth(18.2)).abs() > 0.2);
        run_to(&mut l, &mut sim, &mut t, 0.15, -600.0);
        t += 60.0;
        sim.soc = 1.0;
        run(&mut l, &mut sim, &mut t, 10.0, -200.0);
        assert!((l.state.prior_offset_v - 1.0).abs() < 0.1, "{}", l.state.prior_offset_v);
        for v in [17.2, 17.8, 18.2, 18.6, 19.0, 19.6] {
            assert!((l.curve_soc(v) - truth(v)).abs() < 0.05, "{v} V: {} vs {}", l.curve_soc(v), truth(v));
        }
        assert!(l.curve().windows(2).all(|w| w[1].1 >= w[0].1));
        // A part-charged pack is now judged right by its voltage.
        run_to(&mut l, &mut sim, &mut t, 0.2, -600.0);
        t += 60.0;
        sim.soc = 0.5;
        run(&mut l, &mut sim, &mut t, 10.0, -200.0);
        assert_eq!(l.state.pack.anchor, Anchor::Voltage);
        assert!((l.estimate().unwrap().soc - 0.5).abs() < 0.04, "{:?}", l.estimate());
    }

    #[test]
    fn short_or_unanchored_packs_teach_nothing() {
        let mut t = 0.0;
        let mut sim = Sim { offset: 0.0, capacity: 2000.0, soc: 0.3, r: 0.2 };
        let (mut l, _) = started(None, &sim, &mut t);
        run(&mut l, &mut sim, &mut t, 60.0, -400.0);
        t += 60.0;
        sim.soc = 1.0;
        let logs = run(&mut l, &mut sim, &mut t, 10.0, -200.0);
        assert!(logs.iter().any(|m| m.contains("too little to learn from")), "{logs:?}");
        assert_eq!(l.state.learned_capacity_mah, None);
        assert_eq!(l.capacity_mah(), 2000.0);
    }

    #[test]
    fn persistence_round_trip() {
        let mut t = 0.0;
        let mut sim = Sim { offset: 1.0, capacity: 2500.0, soc: 1.0, r: 0.2 };
        let (mut l, _) = started(None, &sim, &mut t);
        l.state.pack = Pack::new(1, 0.0, Anchor::Full, 1.0, 20.0);
        let mut samples = Vec::new();
        while sim.soc > 0.15 {
            sim.draw(-800.0, 1.0);
            t += 1.0;
            samples.extend(l.on_frame(t, 1.8e9 + t, sim.mv(-800.0), -800).sample);
        }
        t += 60.0;
        sim.soc = 1.0;
        run(&mut l, &mut sim, &mut t, 300.0, -800.0);

        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join(format!("battery_learner_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join(STATE_FILE);
        assert_eq!(load_state(&path).unwrap(), None);
        save_state(&path, &l.state).unwrap();
        assert!(!path.with_extension("json.tmp").exists());
        let loaded = load_state(&path).unwrap().unwrap();
        assert_eq!(loaded, l.state);
        // The reloaded learner has the same curve and estimate.
        let (l2, _) = started(Some(loaded), &sim, &mut t);
        assert_eq!(l2.curve(), l.curve());
        assert_eq!(l2.capacity_mah(), l.capacity_mah());
        assert!((l2.estimate().unwrap().soc - l.state.pack.soc).abs() < 1e-9);
        // CSV: header + one line per sample (every 10 s)
        for s in &samples[..3] {
            append_sample(&dir, s).unwrap();
        }
        let text = fs::read_to_string(csv_path(&dir, 1)).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], Sample::HEADER);
        assert_eq!(lines.len(), 4);
        assert!(lines[1].starts_with(&format!("{:.1},1,", samples[0].unix)) && lines[1].ends_with(",full"));
        assert!(samples.len() > 100 && (samples[1].unix - samples[0].unix - 10.0).abs() < 1e-9);
        // A state saved before the first reading (charge unknown, null in JSON) seeds like a first run.
        let blank = Learner::new(params(), None, 0.0).state;
        save_state(&path, &blank).unwrap();
        let loaded = load_state(&path).unwrap().unwrap();
        assert!(loaded.pack.soc.is_nan() && fs::read_to_string(&path).unwrap().contains("\"soc\": null"));
        let (_, logs) = started(Some(loaded), &sim, &mut t);
        assert!(logs[0].starts_with("no saved charge: pack 1"), "{logs:?}");
        // A corrupt or foreign file is an error, not a silent reset.
        fs::write(&path, "{\"version\": 99}").unwrap();
        assert!(load_state(&path).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
