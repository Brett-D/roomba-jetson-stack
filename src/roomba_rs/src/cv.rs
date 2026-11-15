//! The OpenCV 4.6 and numpy 1.26 routines the Python nodes use, reproduced exactly (same pixels,
//! same label order, same floating-point operations) so the Rust ports give identical output:
//! morphological closing with a rectangular kernel, 8-connected components with stats, 1-pixel
//! and thick lines (cv2.line / cv2.polylines, LINE_8), numpy's argsort (introsort, not stable) and
//! round-half-to-even.
//!
//! Images are row-major `u8` buffers of `w * h` pixels; nonzero = set.

/// numpy.round / Python round(): halves go to the even neighbour.
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

// ---- morphology ----

/// Max (dilate) or min (erode) over a (2r+1) x (2r+1) window, ignoring pixels outside the image
/// (OpenCV's default morphology border).
fn rect_filter(img: &[u8], w: usize, h: usize, r: usize, max: bool) -> Vec<u8> {
    let pick = |a: u8, b: u8| if max { a.max(b) } else { a.min(b) };
    let mut rows = vec![0u8; img.len()];
    for y in 0..h {
        let row = &img[y * w..(y + 1) * w];
        for x in 0..w {
            let (lo, hi) = (x.saturating_sub(r), (x + r).min(w - 1));
            rows[y * w + x] = row[lo..=hi].iter().copied().reduce(pick).unwrap_or(0);
        }
    }
    let mut out = vec![0u8; img.len()];
    for y in 0..h {
        let (lo, hi) = (y.saturating_sub(r), (y + r).min(h - 1));
        for x in 0..w {
            out[y * w + x] = (lo..=hi).map(|yy| rows[yy * w + x]).reduce(pick).unwrap_or(0);
        }
    }
    out
}

/// cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2r+1, 2r+1))).
pub fn close_rect(img: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if img.is_empty() {
        return Vec::new();
    }
    let dilated = rect_filter(img, w, h, r, true);
    rect_filter(&dilated, w, h, r, false)
}

// ---- connected components ----

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Component {
    pub area: u64,
    /// Centroid (x = column, y = row), as cv2 computes it: coordinate sums / area.
    pub cx: f64,
    pub cy: f64,
}

fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        parent[i as usize] = parent[parent[i as usize] as usize];
        i = parent[i as usize];
    }
    i
}

/// cv2.connectedComponentsWithStats(img, connectivity=8) without the background: one entry per
/// component in OpenCV's label order. OpenCV's default algorithm (Spaghetti, also in its parallel
/// form) scans 2 x 2 blocks in raster order and numbers components by their first block.
pub fn components8(img: &[u8], w: usize, h: usize) -> Vec<Component> {
    let n = w * h;
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if img[i] == 0 {
                continue;
            }
            let mut join = |j: usize| {
                if img[j] != 0 {
                    let (a, b) = (find(&mut parent, i as u32), find(&mut parent, j as u32));
                    if a != b {
                        parent[a.max(b) as usize] = a.min(b);
                    }
                }
            };
            if x > 0 {
                join(i - 1);
            }
            if y > 0 {
                if x > 0 {
                    join(i - w - 1);
                }
                join(i - w);
                if x + 1 < w {
                    join(i - w + 1);
                }
            }
        }
    }
    // root -> (first block, area, sum x, sum y)
    let blocks_per_row = w.div_ceil(2);
    let mut stats: std::collections::HashMap<u32, (usize, u64, u64, u64)> = std::collections::HashMap::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if img[i] == 0 {
                continue;
            }
            let root = find(&mut parent, i as u32);
            let block = (y / 2) * blocks_per_row + x / 2;
            let e = stats.entry(root).or_insert((block, 0, 0, 0));
            e.0 = e.0.min(block);
            e.1 += 1;
            e.2 += x as u64;
            e.3 += y as u64;
        }
    }
    let mut comps: Vec<(usize, Component)> = stats
        .into_values()
        .map(|(block, area, sx, sy)| (block, Component { area, cx: sx as f64 / area as f64, cy: sy as f64 / area as f64 }))
        .collect();
    comps.sort_by_key(|c| c.0);
    comps.into_iter().map(|c| c.1).collect()
}

// ---- drawing (OpenCV 4.6 drawing.cpp, 1-channel, LINE_8) ----

const XY_SHIFT: i64 = 16;
const XY_ONE: i64 = 1 << XY_SHIFT;

/// A 1-channel image to draw on.
pub struct Canvas {
    pub w: i64,
    pub h: i64,
    pub data: Vec<u8>,
}

impl Canvas {
    pub fn new(w: usize, h: usize) -> Self {
        Canvas { w: w as i64, h: h as i64, data: vec![0; w * h] }
    }

    fn put(&mut self, x: i64, y: i64, c: u8) {
        self.data[(y * self.w + x) as usize] = c;
    }

    /// ICV_HLINE: x1..=x2 on row y (already clipped).
    fn hline(&mut self, y: i64, x1: i64, x2: i64, c: u8) {
        let row = (y * self.w) as usize;
        for x in x1..=x2 {
            self.data[row + x as usize] = c;
        }
    }

    /// cv2.line(img, p1, p2, c, thickness=1) (8-connected, drawn left to right).
    pub fn line(&mut self, p1: (i64, i64), p2: (i64, i64), c: u8) {
        for (x, y) in line_points(self.w, self.h, p1, p2) {
            self.put(x, y, c);
        }
    }

    /// cv2.polylines(img, [pts], isClosed=False, c, thickness) for thickness > 1 (LINE_8).
    pub fn polyline_thick(&mut self, pts: &[(i64, i64)], thickness: i64, c: u8) {
        if pts.is_empty() {
            return;
        }
        let mut flags = 3;
        let mut p0 = pts[0];
        for &p in &pts[1..] {
            self.thick_line(p0, p, thickness, flags, c);
            p0 = p;
            flags = 2;
        }
    }

    /// ThickLine with thickness > 1, shift 0, LINE_8: a filled rotated rectangle plus round caps.
    fn thick_line(&mut self, p0: (i64, i64), p1: (i64, i64), thickness: i64, flags: i32, c: u8) {
        const INV_XY_ONE: f64 = 1.0 / XY_ONE as f64;
        let (mut p0, p1) = ((p0.0 << XY_SHIFT, p0.1 << XY_SHIFT), (p1.0 << XY_SHIFT, p1.1 << XY_SHIFT));
        let dx = (p0.0 - p1.0) as f64 * INV_XY_ONE;
        let dy = (p1.1 - p0.1) as f64 * INV_XY_ONE;
        let r = dx * dx + dy * dy;
        let odd = thickness & 1;
        let thickness = thickness << (XY_SHIFT - 1);
        if r.abs() > f64::EPSILON {
            let r = (thickness as f64 + odd as f64 * XY_ONE as f64 * 0.5) / r.sqrt();
            let dp = ((dy * r).round_ties_even() as i32 as i64, (dx * r).round_ties_even() as i32 as i64);
            let pt = [
                (p0.0 + dp.0, p0.1 + dp.1),
                (p0.0 - dp.0, p0.1 - dp.1),
                (p1.0 - dp.0, p1.1 - dp.1),
                (p1.0 + dp.0, p1.1 + dp.1),
            ];
            self.fill_convex_poly(&pt, XY_SHIFT, c);
        }
        for i in 0..2 {
            if flags & (i + 1) != 0 {
                let center = ((p0.0 + (XY_ONE >> 1)) >> XY_SHIFT, (p0.1 + (XY_ONE >> 1)) >> XY_SHIFT);
                self.circle_filled(center, (thickness + (XY_ONE >> 1)) >> XY_SHIFT, c);
            }
            p0 = p1;
        }
    }

    /// FillConvexPoly for line_type 8.
    fn fill_convex_poly(&mut self, v: &[(i64, i64)], shift: i64, c: u8) {
        #[derive(Clone, Copy)]
        struct Edge {
            idx: usize,
            di: usize,
            x: i64,
            dx: i64,
            ye: i64,
        }
        let npts = v.len();
        let delta = (1i64 << shift) >> 1;
        let (delta1, delta2) = (XY_ONE >> 1, XY_ONE >> 1);
        let mut p0 = v[npts - 1];
        p0 = (p0.0 << (XY_SHIFT - shift), p0.1 << (XY_SHIFT - shift));
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (v[0].0, v[0].0, v[0].1, v[0].1);
        let mut imin = 0;
        for (i, &p) in v.iter().enumerate() {
            if p.1 < ymin {
                ymin = p.1;
                imin = i;
            }
            ymax = ymax.max(p.1);
            xmax = xmax.max(p.0);
            xmin = xmin.min(p.0);
            let p = (p.0 << (XY_SHIFT - shift), p.1 << (XY_SHIFT - shift));
            if shift == 0 {
                self.line((p0.0 >> XY_SHIFT, p0.1 >> XY_SHIFT), (p.0 >> XY_SHIFT, p.1 >> XY_SHIFT), c);
            } else {
                self.line2(p0, p, c);
            }
            p0 = p;
        }
        let xmin = (xmin + delta) >> shift;
        let xmax = (xmax + delta) >> shift;
        let ymin = (ymin + delta) >> shift;
        let mut ymax = (ymax + delta) >> shift;
        if npts < 3 || xmax < 0 || ymax < 0 || xmin >= self.w || ymin >= self.h {
            return;
        }
        ymax = ymax.min(self.h - 1);
        let mut edge = [Edge { idx: imin, di: 1, x: -XY_ONE, dx: 0, ye: ymin }, Edge { idx: imin, di: npts - 1, x: -XY_ONE, dx: 0, ye: ymin }];
        let mut edges = npts as i64;
        let mut y = ymin;
        loop {
            for e in edge.iter_mut() {
                if y >= e.ye {
                    let mut idx0 = e.idx;
                    let mut idx = idx0 + e.di;
                    if idx >= npts {
                        idx -= npts;
                    }
                    loop {
                        let more = edges > 0;
                        edges -= 1;
                        if !more {
                            break;
                        }
                        let ty = (v[idx].1 + delta) >> shift;
                        if ty > y {
                            let mut xs = v[idx0].0;
                            let mut xe = v[idx].0;
                            if shift != XY_SHIFT {
                                xs <<= XY_SHIFT - shift;
                                xe <<= XY_SHIFT - shift;
                            }
                            e.ye = ty;
                            e.dx = ((xe - xs) * 2 + (ty - y)) / (2 * (ty - y));
                            e.x = xs;
                            e.idx = idx;
                            break;
                        }
                        idx0 = idx;
                        idx += e.di;
                        if idx >= npts {
                            idx -= npts;
                        }
                    }
                }
            }
            if edges < 0 {
                break;
            }
            if y >= 0 {
                let (left, right) = if edge[0].x > edge[1].x { (1, 0) } else { (0, 1) };
                let mut xx1 = (edge[left].x + delta1) >> XY_SHIFT;
                let mut xx2 = (edge[right].x + delta2) >> XY_SHIFT;
                if xx2 >= 0 && xx1 < self.w {
                    xx1 = xx1.max(0);
                    xx2 = xx2.min(self.w - 1);
                    self.hline(y, xx1, xx2, c);
                }
            }
            edge[0].x += edge[0].dx;
            edge[1].x += edge[1].dx;
            y += 1;
            if y > ymax {
                break;
            }
        }
    }

    /// Line2: a line between sub-pixel (XY_SHIFT) end points.
    fn line2(&mut self, mut pt1: (i64, i64), mut pt2: (i64, i64), c: u8) {
        let size = (self.w << XY_SHIFT, self.h << XY_SHIFT);
        if !clip_line(size, &mut pt1, &mut pt2) {
            return;
        }
        let mut dx = pt2.0 - pt1.0;
        let mut dy = pt2.1 - pt1.1;
        let j = if dx < 0 { -1i64 } else { 0 };
        let ax = (dx ^ j) - j;
        let i = if dy < 0 { -1i64 } else { 0 };
        let ay = (dy ^ i) - i;
        let (x_step, y_step, mut ecount);
        if ax > ay {
            dy = (dy ^ j) - j;
            pt1.0 ^= pt2.0 & j;
            pt2.0 ^= pt1.0 & j;
            pt1.0 ^= pt2.0 & j;
            pt1.1 ^= pt2.1 & j;
            pt2.1 ^= pt1.1 & j;
            pt1.1 ^= pt2.1 & j;
            x_step = XY_ONE;
            y_step = dy * (1 << XY_SHIFT) / (ax | 1);
            ecount = (pt2.0 - pt1.0) >> XY_SHIFT;
        } else {
            dx = (dx ^ i) - i;
            pt1.0 ^= pt2.0 & i;
            pt2.0 ^= pt1.0 & i;
            pt1.0 ^= pt2.0 & i;
            pt1.1 ^= pt2.1 & i;
            pt2.1 ^= pt1.1 & i;
            pt1.1 ^= pt2.1 & i;
            x_step = dx * (1 << XY_SHIFT) / (ay | 1);
            y_step = XY_ONE;
            ecount = (pt2.1 - pt1.1) >> XY_SHIFT;
        }
        pt1.0 += XY_ONE >> 1;
        pt1.1 += XY_ONE >> 1;
        let (w, h) = (self.w, self.h);
        let put = |canvas: &mut Canvas, x: i64, y: i64| {
            if 0 <= x && x < w && 0 <= y && y < h {
                canvas.put(x, y, c);
            }
        };
        put(self, (pt2.0 + (XY_ONE >> 1)) >> XY_SHIFT, (pt2.1 + (XY_ONE >> 1)) >> XY_SHIFT);
        if ax > ay {
            pt1.0 >>= XY_SHIFT;
            while ecount >= 0 {
                put(self, pt1.0, pt1.1 >> XY_SHIFT);
                pt1.0 += 1;
                pt1.1 += y_step;
                ecount -= 1;
            }
        } else {
            pt1.1 >>= XY_SHIFT;
            while ecount >= 0 {
                put(self, pt1.0 >> XY_SHIFT, pt1.1);
                pt1.0 += x_step;
                pt1.1 += 1;
                ecount -= 1;
            }
        }
    }

    /// Circle with fill (integer midpoint circle), clipped to the image.
    fn circle_filled(&mut self, center: (i64, i64), radius: i64, c: u8) {
        let (w, h) = (self.w, self.h);
        let (mut err, mut dx, mut dy, mut plus, mut minus) = (0i64, radius, 0i64, 1i64, (radius << 1) - 1);
        let inside = center.0 >= radius && center.0 < w - radius && center.1 >= radius && center.1 < h - radius;
        while dx >= dy {
            let (y11, y12, y21, y22) = (center.1 - dy, center.1 + dy, center.1 - dx, center.1 + dx);
            let (mut x11, mut x12, mut x21, mut x22) = (center.0 - dx, center.0 + dx, center.0 - dy, center.0 + dy);
            if inside {
                self.hline(y11, x11, x12, c);
                self.hline(y12, x11, x12, c);
                self.hline(y21, x21, x22, c);
                self.hline(y22, x21, x22, c);
            } else if x11 < w && x12 >= 0 && y21 < h && y22 >= 0 {
                x11 = x11.max(0);
                x12 = x12.min(w - 1);
                if (y11 as u64) < h as u64 {
                    self.hline(y11, x11, x12, c);
                }
                if (y12 as u64) < h as u64 {
                    self.hline(y12, x11, x12, c);
                }
                if x21 < w && x22 >= 0 {
                    x21 = x21.max(0);
                    x22 = x22.min(w - 1);
                    if (y21 as u64) < h as u64 {
                        self.hline(y21, x21, x22, c);
                    }
                    if (y22 as u64) < h as u64 {
                        self.hline(y22, x21, x22, c);
                    }
                }
            }
            dy += 1;
            err += plus;
            plus += 2;
            let mask = if err <= 0 { 0 } else { -1i64 };
            err -= minus & mask;
            dx += mask;
            minus -= mask & 2;
        }
    }
}

/// cv::clipLine(Size2l, Point2l&, Point2l&).
fn clip_line(size: (i64, i64), p1: &mut (i64, i64), p2: &mut (i64, i64)) -> bool {
    let (right, bottom) = (size.0 - 1, size.1 - 1);
    if size.0 <= 0 || size.1 <= 0 {
        return false;
    }
    let (x1, y1) = (&mut p1.0, &mut p1.1);
    let (x2, y2) = (&mut p2.0, &mut p2.1);
    let code = |x: i64, y: i64| (x < 0) as i32 + (x > right) as i32 * 2 + (y < 0) as i32 * 4 + (y > bottom) as i32 * 8;
    let mut c1 = code(*x1, *y1);
    let mut c2 = code(*x2, *y2);
    if (c1 & c2) == 0 && (c1 | c2) != 0 {
        if c1 & 12 != 0 {
            let a = if c1 < 8 { 0 } else { bottom };
            *x1 += ((a - *y1) as f64 * (*x2 - *x1) as f64 / (*y2 - *y1) as f64) as i64;
            *y1 = a;
            c1 = (*x1 < 0) as i32 + (*x1 > right) as i32 * 2;
        }
        if c2 & 12 != 0 {
            let a = if c2 < 8 { 0 } else { bottom };
            *x2 += ((a - *y2) as f64 * (*x2 - *x1) as f64 / (*y2 - *y1) as f64) as i64;
            *y2 = a;
            c2 = (*x2 < 0) as i32 + (*x2 > right) as i32 * 2;
        }
        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            if c1 != 0 {
                let a = if c1 == 1 { 0 } else { right };
                *y1 += ((a - *x1) as f64 * (*y2 - *y1) as f64 / (*x2 - *x1) as f64) as i64;
                *x1 = a;
                c1 = 0;
            }
            if c2 != 0 {
                let a = if c2 == 1 { 0 } else { right };
                *y2 += ((a - *x2) as f64 * (*y2 - *y1) as f64 / (*x2 - *x1) as f64) as i64;
                *x2 = a;
                c2 = 0;
            }
        }
    }
    (c1 | c2) == 0
}

/// The pixels cv2.line (thickness 1, LINE_8) sets, via LineIterator(connectivity 8, leftToRight).
pub fn line_points(w: i64, h: i64, mut pt1: (i64, i64), mut pt2: (i64, i64)) -> Vec<(i64, i64)> {
    // OpenCV clips in int (Point) coordinates.
    let outside = |p: (i64, i64)| (p.0 as i32 as u32) >= w as u32 || (p.1 as i32 as u32) >= h as u32;
    if (outside(pt1) || outside(pt2)) && !clip_line((w, h), &mut pt1, &mut pt2) {
        return Vec::new();
    }
    let (mut delta_x, mut delta_y) = (1i64, 1i64);
    let mut dx = pt2.0 - pt1.0;
    let mut dy = pt2.1 - pt1.1;
    if dx < 0 {
        dx = -dx;
        dy = -dy;
        pt1 = pt2;
    }
    if dy < 0 {
        dy = -dy;
        delta_y = -1;
    }
    let vert = dy > dx;
    if vert {
        std::mem::swap(&mut dx, &mut dy);
        std::mem::swap(&mut delta_x, &mut delta_y);
    }
    let mut err = dx - (dy + dy);
    let plus_delta = dx + dx;
    let minus_delta = -(dy + dy);
    let (mut minus_shift, mut plus_shift, mut minus_step, mut plus_step) = (delta_x, 0i64, 0i64, delta_y);
    let count = dx + 1;
    if vert {
        std::mem::swap(&mut plus_step, &mut plus_shift);
        std::mem::swap(&mut minus_step, &mut minus_shift);
    }
    let mut p = pt1;
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        out.push(p);
        let mask = if err < 0 { -1i64 } else { 0 };
        err += minus_delta + (plus_delta & mask);
        p.0 += minus_shift + (plus_shift & mask);
        p.1 += minus_step + (plus_step & mask);
    }
    out
}

// ---- numpy argsort ----

fn np_less(a: f64, b: f64) -> bool {
    a < b || (b != b && a == a)
}

fn get_msb(mut n: usize) -> i32 {
    let mut depth = 0;
    while n > 1 {
        n >>= 1;
        depth += 1;
    }
    depth
}

/// numpy aheapsort_ on tosort[pl..pl+n].
fn aheapsort(v: &[f64], a: &mut [usize]) {
    let n0 = a.len();
    // 1-based indexing as in numpy.
    let get = |a: &[usize], i: usize| a[i - 1];
    let mut n = n0;
    let mut l = n >> 1;
    while l > 0 {
        let tmp = get(a, l);
        let (mut i, mut j) = (l, l << 1);
        while j <= n {
            if j < n && np_less(v[get(a, j)], v[get(a, j + 1)]) {
                j += 1;
            }
            if np_less(v[tmp], v[get(a, j)]) {
                a[i - 1] = a[j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i - 1] = tmp;
        l -= 1;
    }
    while n > 1 {
        let tmp = get(a, n);
        a[n - 1] = a[0];
        n -= 1;
        let (mut i, mut j) = (1, 2);
        while j <= n {
            if j < n && np_less(v[get(a, j)], v[get(a, j + 1)]) {
                j += 1;
            }
            if np_less(v[tmp], v[get(a, j)]) {
                a[i - 1] = a[j - 1];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i - 1] = tmp;
    }
}

/// np.argsort(v) (kind 'quicksort': numpy's introsort; the order of equal values follows it).
pub fn argsort(v: &[f64]) -> Vec<usize> {
    const SMALL_QUICKSORT: isize = 15;
    let num = v.len();
    let mut t: Vec<usize> = (0..num).collect();
    if num == 0 {
        return t;
    }
    let mut pl: isize = 0;
    let mut pr: isize = num as isize - 1;
    let mut stack: Vec<(isize, isize, i32)> = Vec::new();
    let mut cdepth = get_msb(num) * 2;
    loop {
        if cdepth < 0 {
            aheapsort(v, &mut t[pl as usize..=pr as usize]);
        } else {
            while pr - pl > SMALL_QUICKSORT {
                let pm = pl + ((pr - pl) >> 1);
                let (ul, um, ur) = (pl as usize, pm as usize, pr as usize);
                if np_less(v[t[um]], v[t[ul]]) {
                    t.swap(um, ul);
                }
                if np_less(v[t[ur]], v[t[um]]) {
                    t.swap(ur, um);
                }
                if np_less(v[t[um]], v[t[ul]]) {
                    t.swap(um, ul);
                }
                let vp = v[t[um]];
                let mut pi = pl;
                let mut pj = pr - 1;
                t.swap(um, pj as usize);
                loop {
                    loop {
                        pi += 1;
                        if !np_less(v[t[pi as usize]], vp) {
                            break;
                        }
                    }
                    loop {
                        pj -= 1;
                        if !np_less(vp, v[t[pj as usize]]) {
                            break;
                        }
                    }
                    if pi >= pj {
                        break;
                    }
                    t.swap(pi as usize, pj as usize);
                }
                let pk = pr - 1;
                t.swap(pi as usize, pk as usize);
                cdepth -= 1;
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth));
                    pl = pi + 1;
                }
            }
            // insertion sort
            let mut pi = pl + 1;
            while pi <= pr {
                let vi = t[pi as usize];
                let vp = v[vi];
                let mut pj = pi;
                while pj > pl && np_less(vp, v[t[(pj - 1) as usize]]) {
                    t[pj as usize] = t[(pj - 1) as usize];
                    pj -= 1;
                }
                t[pj as usize] = vi;
                pi += 1;
            }
        }
        match stack.pop() {
            Some((l, r, d)) => {
                pl = l;
                pr = r;
                cdepth = d;
            }
            None => break,
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_bridges_one_cell_gaps_only() {
        // Two dots 2 cells apart (one free cell between) get joined; 4 apart stay apart.
        let w = 11;
        let mut img = vec![0u8; w * 5];
        for x in [2, 4, 8] {
            img[2 * w + x] = 1;
        }
        let out = close_rect(&img, w, 5, 1);
        assert_eq!(&out[2 * w..3 * w], &[0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0]);
        assert!(out[w..2 * w].iter().all(|&v| v == 0));
    }

    #[test]
    fn components_in_block_order() {
        // Component A starts at row 1 col 0, B at row 0 col 4: A's block (0, 0) comes first.
        let w = 6;
        let mut img = vec![0u8; w * 2];
        img[w] = 1;
        img[4] = 1;
        img[5] = 1;
        let c = components8(&img, w, 2);
        assert_eq!(c.len(), 2);
        assert_eq!((c[0].area, c[0].cx, c[0].cy), (1, 0.0, 1.0));
        assert_eq!((c[1].area, c[1].cx, c[1].cy), (2, 4.5, 0.0));
    }

    #[test]
    fn line_is_8_connected_and_symmetric() {
        let a = line_points(20, 20, (1, 1), (10, 4));
        let b = line_points(20, 20, (10, 4), (1, 1));
        assert_eq!(a, b);
        assert_eq!(a.len(), 10);
        assert!(a.windows(2).all(|p| (p[1].0 - p[0].0).abs() <= 1 && (p[1].1 - p[0].1).abs() <= 1));
    }

    #[test]
    fn argsort_sorts() {
        let v: Vec<f64> = (0..100).map(|i| ((i * 37) % 11) as f64).collect();
        let idx = argsort(&v);
        assert!(idx.windows(2).all(|p| v[p[0]] <= v[p[1]]));
    }
}
