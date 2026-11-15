//! Replays a TF event script through roomba_rs::tf_buffer for tools/ab_tf_buffer.py.
//!
//! Input (tab-separated lines):
//!   T  static(0|1)  parent  child  stamp_ns  tx ty tz  qx qy qz qw   -> set_transform (no output)
//!   L  target  source  time_ns                                        -> lookup_transform
//!   C  target  source  time_ns                                        -> can_transform
//! Output, one line per L/C: `ok stamp tx ty tz qx qy qz qw` | `err Class message` (L),
//! `ok` | `no message` (C); tab-separated, newlines in messages escaped as \n.
//!   cargo run --release --example tf_replay -- EVENTS > OUT

use std::io::{BufRead, BufReader, BufWriter, Write};

use roomba_rs::tf_buffer::{TfBuffer, Transform};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: tf_replay EVENTS"))?;
    let input = BufReader::new(std::fs::File::open(path)?);
    let mut out = BufWriter::new(std::io::stdout().lock());
    let mut buffer = TfBuffer::new();
    let esc = |s: &str| s.replace('\n', "\\n");
    for line in input.lines() {
        let line = line?;
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "T" => {
                let v: Vec<f64> = f[5..12].iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
                let t = Transform { translation: [v[0], v[1], v[2]], rotation: [v[3], v[4], v[5], v[6]] };
                let _ = buffer.set_transform(f[2], f[3], f[4].parse()?, &t, "default_authority", f[1] == "1");
            }
            "L" => match buffer.lookup_transform(f[1], f[2], f[3].parse()?) {
                Ok(t) => {
                    let (p, q) = (t.transform.translation, t.transform.rotation);
                    writeln!(out, "ok\t{}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}\t{:?}", t.stamp_ns, p[0], p[1], p[2], q[0], q[1], q[2], q[3])?;
                }
                Err(e) => writeln!(out, "err\t{}\t{}", e.kind.python_name(), esc(&e.message))?,
            },
            "C" => match buffer.can_transform(f[1], f[2], f[3].parse()?) {
                Ok(()) => writeln!(out, "ok")?,
                Err(e) => writeln!(out, "no\t{}", esc(&e))?,
            },
            other => anyhow::bail!("unknown event {other}"),
        }
    }
    Ok(())
}
