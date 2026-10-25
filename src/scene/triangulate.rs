//! N-gon triangulation using earcutr with fan fallback.

use glam::Vec3;

/// Triangulate an ordered face into triangle indices (local to `positions`).
pub fn triangulate_face(positions: &[[f32; 3]], face: &[u32]) -> Vec<u32> {
    match face.len() {
        0..=2 => Vec::new(),
        3 => face.to_vec(),
        4.. => {
            let normal = newell_normal(positions, face);
            if normal.length_squared() <= f32::EPSILON {
                return fan(face);
            }
            let (u, v) = basis(normal);
            let mut flat: Vec<f64> = Vec::with_capacity(face.len() * 2);
            for index in face {
                let Some(position) = positions.get(*index as usize) else {
                    return fan(face);
                };
                let point = Vec3::from(*position);
                flat.push(point.dot(u) as f64);
                flat.push(point.dot(v) as f64);
            }
            match earcutr::earcut(&flat, &[], 2) {
                Ok(triangles) if triangles.len() >= 3 => {
                    triangles.into_iter().map(|local| face[local]).collect()
                }
                _ => fan(face),
            }
        }
    }
}

/// Triangulate a list of faces, concatenating their triangles.
pub fn triangulate_faces(positions: &[[f32; 3]], faces: &[Vec<u32>]) -> Vec<u32> {
    let mut out = Vec::new();
    for face in faces {
        out.extend(triangulate_face(positions, face));
    }
    out
}

fn newell_normal(positions: &[[f32; 3]], face: &[u32]) -> Vec3 {
    let mut normal = Vec3::ZERO;
    for (index, current) in face.iter().enumerate() {
        let next = face[(index + 1) % face.len()];
        let (Some(a), Some(b)) = (
            positions.get(*current as usize),
            positions.get(next as usize),
        ) else {
            continue;
        };
        let a = Vec3::from(*a);
        let b = Vec3::from(*b);
        normal.x += (a.y - b.y) * (a.z + b.z);
        normal.y += (a.z - b.z) * (a.x + b.x);
        normal.z += (a.x - b.x) * (a.y + b.y);
    }
    normal
}

fn basis(normal: Vec3) -> (Vec3, Vec3) {
    let n = normal.normalize();
    let helper = if n.x.abs() > 0.9 { Vec3::Y } else { Vec3::X };
    let u = helper.cross(n).normalize_or_zero();
    let v = n.cross(u);
    (u, v)
}

fn fan(face: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity((face.len().saturating_sub(2)) * 3);
    for index in 1..face.len().saturating_sub(1) {
        out.push(face[0]);
        out.push(face[index]);
        out.push(face[index + 1]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_becomes_two_triangles() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let triangles = triangulate_face(&positions, &[0, 1, 2, 3]);
        assert_eq!(triangles.len(), 6);
        assert!(triangles.iter().all(|index| *index < 4));
    }

    #[test]
    fn concave_polygon_triangulates_inside() {
        let positions = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let triangles = triangulate_face(&positions, &[0, 1, 2, 3, 4]);
        assert_eq!(triangles.len(), 9);
    }

    #[test]
    fn degenerate_falls_back_to_fan() {
        let positions = [[0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]];
        let triangles = triangulate_face(&positions, &[0, 1, 2, 3]);
        assert_eq!(triangles.len(), 6);
    }
}
