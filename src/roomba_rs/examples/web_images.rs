//! Image half of tools/ab_web_images.py: runs the web server's preview and map image code on the
//! inputs that script writes and saves the results next to them for byte comparison with cv2.
//!   web_images DIR
//! DIR/*.gray (mono8 frames, name: <id>_<w>x<h>_<step>_<dw>x<dh>.gray) -> <id>.rs.area (INTER_AREA
//! to dw x dh), <id>.rs.jpg (preview JPEG of the resized frame, quality 70)
//! DIR/*.grid (int8 occupancy grids, name: <id>_<w>x<h>.grid) -> <id>.rs.png (map PNG)

use std::path::Path;

use anyhow::{Context, Result};
use roomba_rs::web::image::{encode_jpeg_gray, map_png, resize_area};

fn dims(s: &str) -> Result<(usize, usize)> {
    let (w, h) = s.split_once('x').context("WxH")?;
    Ok((w.parse()?, h.parse()?))
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).context("usage: web_images DIR")?;
    let (mut frames, mut grids) = (0, 0);
    for entry in std::fs::read_dir(&dir)? {
        let path = entry?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let out = |id: &str, ext: &str| Path::new(&dir).join(format!("{id}.rs.{ext}"));
        if let Some(stem) = name.strip_suffix(".gray") {
            let parts: Vec<&str> = stem.split('_').collect();
            let (w, h) = dims(parts[1])?;
            let step: usize = parts[2].parse()?;
            let (dw, dh) = dims(parts[3])?;
            let data = std::fs::read(&path)?;
            let small = if (dw, dh) == (w, h) {
                (0..h).flat_map(|y| data[y * step..y * step + w].to_vec()).collect()
            } else {
                resize_area(&data, w, h, step, dw, dh)
            };
            std::fs::write(out(parts[0], "area"), &small)?;
            std::fs::write(out(parts[0], "jpg"), encode_jpeg_gray(&small, dw, dh, dw, 70).context("jpeg")?)?;
            frames += 1;
        } else if let Some(stem) = name.strip_suffix(".grid") {
            let (id, size) = stem.split_once('_').context("id_WxH")?;
            let (w, h) = dims(size)?;
            let data: Vec<i8> = std::fs::read(&path)?.into_iter().map(|b| b as i8).collect();
            std::fs::write(out(id, "png"), map_png(w, h, &data).context("png")?)?;
            grids += 1;
        }
    }
    println!("{frames} frames, {grids} grids");
    Ok(())
}
