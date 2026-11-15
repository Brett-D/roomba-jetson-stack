//! Replays maps and paths through roomba_rs::gap_filler for tools/ab_map_gap_filler_offline.py.
//!
//! Input records (little-endian): `R` + params (f64 gap, i64 occupied, u8 join, f64 spacing,
//! f64 tolerance, i64 min_posts, i64 max_dot_cells, f64 clear_radius) starts a new node; `P` + u32 n +
//! n * (f64 x, f64 y) is a path; `M` + u32 w + u32 h + f32 resolution + f64 origin x + f64 origin y +
//! w*h bytes is a map. Output per map: u32 n + n bytes (the published grid), u32 log count, then per
//! log u8 throttled + u32 len + text.
//!   cargo run --release --example gap_filler_replay -- IN OUT

use std::io::{BufReader, BufWriter, Read, Write};

use roomba_rs::gap_filler::{GapFiller, GridInfo, Params};

fn take<const N: usize>(r: &mut impl Read) -> std::io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn f64_(r: &mut impl Read) -> std::io::Result<f64> {
    Ok(f64::from_le_bytes(take(r)?))
}

fn i64_(r: &mut impl Read) -> std::io::Result<i64> {
    Ok(i64::from_le_bytes(take(r)?))
}

fn u32_(r: &mut impl Read) -> std::io::Result<u32> {
    Ok(u32::from_le_bytes(take(r)?))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 3, "usage: gap_filler_replay IN OUT");
    let mut r = BufReader::new(std::fs::File::open(&args[1])?);
    let mut w = BufWriter::new(std::fs::File::create(&args[2])?);
    let mut filler = GapFiller::new(Params::default());
    loop {
        let mut tag = [0u8; 1];
        if r.read(&mut tag)? == 0 {
            break;
        }
        match tag[0] {
            b'R' => {
                let params = Params {
                    gap_m: f64_(&mut r)?,
                    occupied: i64_(&mut r)?,
                    join_fences: take::<1>(&mut r)?[0] != 0,
                    spacing_m: f64_(&mut r)?,
                    tolerance_m: f64_(&mut r)?,
                    min_posts: i64_(&mut r)?,
                    max_dot_cells: i64_(&mut r)?,
                    path_clear_radius_m: f64_(&mut r)?,
                };
                filler = GapFiller::new(params);
            }
            b'P' => {
                let n = u32_(&mut r)?;
                let poses = (0..n).map(|_| Ok((f64_(&mut r)?, f64_(&mut r)?))).collect::<std::io::Result<Vec<_>>>()?;
                filler.on_path(poses);
            }
            b'M' => {
                let (width, height) = (u32_(&mut r)? as usize, u32_(&mut r)? as usize);
                let resolution = f32::from_le_bytes(take(&mut r)?);
                let (origin_x, origin_y) = (f64_(&mut r)?, f64_(&mut r)?);
                let mut raw = vec![0u8; width * height];
                r.read_exact(&mut raw)?;
                let data: Vec<i8> = raw.iter().map(|&b| b as i8).collect();
                let info = GridInfo { width, height, resolution, origin_x, origin_y };
                let (out, logs) = match filler.on_map(&info, &data) {
                    Some(f) => (f.data, f.logs),
                    None => (data, Vec::new()),
                };
                w.write_all(&(out.len() as u32).to_le_bytes())?;
                w.write_all(&out.iter().map(|&v| v as u8).collect::<Vec<_>>())?;
                w.write_all(&(logs.len() as u32).to_le_bytes())?;
                for (throttled, text) in logs {
                    w.write_all(&[throttled as u8])?;
                    w.write_all(&(text.len() as u32).to_le_bytes())?;
                    w.write_all(text.as_bytes())?;
                }
            }
            t => anyhow::bail!("bad record {t}"),
        }
    }
    Ok(())
}
