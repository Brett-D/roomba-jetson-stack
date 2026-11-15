//! Small vector/quaternion helpers. Quaternions are (x, y, z, w), as in ROS messages.

pub type Vec3 = [f64; 3];
pub type Quat = [f64; 4];

pub const IDENTITY: Quat = [0.0, 0.0, 0.0, 1.0];

/// Rotates vector `v` by quaternion `q` (same formula as scene_filter.quat_rotate).
pub fn quat_rotate(q: Quat, v: Vec3) -> Vec3 {
    let [x, y, z, w] = q;
    let [vx, vy, vz] = v;
    let tx = 2.0 * (y * vz - z * vy);
    let ty = 2.0 * (z * vx - x * vz);
    let tz = 2.0 * (x * vy - y * vx);
    [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]
}

/// Hamilton product a * b: rotating by the result applies b first, then a.
pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

pub fn quat_conj(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

/// a / |a|; a zero vector is returned unchanged (numpy's `x /= norm or 1.0`).
pub fn normalized(a: Vec3) -> Vec3 {
    let n = norm(a);
    let n = if n == 0.0 { 1.0 } else { n };
    [a[0] / n, a[1] / n, a[2] / n]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-9)
    }

    #[test]
    fn mul_composes_rotations() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let z90: Quat = [0.0, 0.0, s, s];
        let x90: Quat = [s, 0.0, 0.0, s];
        let v = [1.0, 2.0, 3.0];
        assert!(close(quat_rotate(quat_mul(x90, z90), v), quat_rotate(x90, quat_rotate(z90, v))));
        assert!(close(quat_rotate(quat_mul(quat_conj(z90), z90), v), v));
    }
}