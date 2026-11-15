//! Rotation between two frames of the static TF tree (/tf_static).
//!
//! floor_obstacles only needs the fixed rotation between the camera's IMU frame and its depth
//! optical frame, so a full tf2 buffer isn't needed: keep each child's parent and rotation and
//! compose along the tree.

use std::collections::HashMap;

use crate::geometry::{quat_conj, quat_mul, Quat, IDENTITY};

#[derive(Default)]
pub struct StaticTree {
    parent: HashMap<String, (String, Quat)>,
}

fn clean(frame: &str) -> &str {
    frame.trim_start_matches('/')
}

impl StaticTree {
    /// Adds a transform as published: `rotation` maps child-frame vectors into the parent frame.
    pub fn insert(&mut self, parent: &str, child: &str, rotation: Quat) {
        self.parent.insert(clean(child).to_string(), (clean(parent).to_string(), rotation));
    }

    /// (root frame, rotation mapping `frame` vectors into the root frame).
    fn to_root(&self, frame: &str) -> (String, Quat) {
        let mut r = IDENTITY;
        let mut f = clean(frame).to_string();
        for _ in 0..64 {
            match self.parent.get(&f) {
                Some((p, q)) => {
                    r = quat_mul(*q, r);
                    f = p.clone();
                }
                None => break,
            }
        }
        (f, r)
    }

    /// Rotation of tf2's `lookup_transform(target, source)`: maps source-frame vectors into the
    /// target frame. None if the frames aren't connected (yet).
    pub fn rotation(&self, target: &str, source: &str) -> Option<Quat> {
        let (root_t, r_root_target) = self.to_root(target);
        let (root_s, r_root_source) = self.to_root(source);
        if root_t != root_s {
            return None;
        }
        Some(quat_mul(quat_conj(r_root_target), r_root_source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::quat_rotate;

    #[test]
    fn rotation_through_a_shared_parent() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mut t = StaticTree::default();
        // camera_link -> optical: the usual optical rotation (z forward, x right, y down).
        t.insert("camera_link", "/camera_optical", [-0.5, 0.5, -0.5, 0.5]);
        // camera_link -> imu: rotated 90 deg about z.
        t.insert("camera_link", "camera_imu", [0.0, 0.0, s, s]);
        assert!(t.rotation("camera_optical", "unconnected").is_none());
        let q = t.rotation("camera_optical", "camera_imu").unwrap();
        // imu +x is camera_link +y (left), which is optical -x.
        let v = quat_rotate(q, [1.0, 0.0, 0.0]);
        assert!((v[0] + 1.0).abs() < 1e-9 && v[1].abs() < 1e-9 && v[2].abs() < 1e-9, "{v:?}");
        let same = t.rotation("camera_imu", "camera_imu").unwrap();
        assert!((0..4).all(|i| (same[i] - IDENTITY[i]).abs() < 1e-12), "{same:?}");
    }
}