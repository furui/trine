//! Vertex welding, smooth/flat normal computation, and normal modes.

use std::collections::HashMap;

use glam::Vec3;

use crate::ops::gpu::GpuEngine;
use crate::scene::Primitive;

/// How to handle normals requested on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormalMode {
    #[default]
    Preserve,
    Smooth,
    Flat,
    None,
}

impl NormalMode {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "preserve" => Some(NormalMode::Preserve),
            "smooth" => Some(NormalMode::Smooth),
            "flat" => Some(NormalMode::Flat),
            "none" => Some(NormalMode::None),
            _ => None,
        }
    }
}

/// Apply a normal mode in place. Returns true if geometry was topologically changed.
pub fn apply(primitive: &mut Primitive, mode: NormalMode) -> bool {
    match mode {
        NormalMode::Preserve => false,
        NormalMode::None => {
            primitive.normals = None;
            primitive.tangents = None;
            false
        }
        NormalMode::Smooth => {
            primitive.normals = Some(smooth_normals(&primitive.positions, &primitive.indices));
            primitive.tangents = None;
            false
        }
        NormalMode::Flat => {
            let flattened = flatten(primitive);
            *primitive = flattened;
            true
        }
    }
}

/// Apply a normal mode, preferring the GPU for smoothing when available.
pub fn apply_with(primitive: &mut Primitive, mode: NormalMode, gpu: Option<&GpuEngine>) -> bool {
    if mode == NormalMode::Smooth {
        if let Some(gpu) = gpu {
            if gpu.should_accelerate(primitive.vertex_count()) && gpu.smooth_normals(primitive) {
                primitive.tangents = None;
                return false;
            }
        }
    }
    apply(primitive, mode)
}

/// Weld tolerance derived from the bounding-box diagonal.
pub fn weld_epsilon(positions: &[[f32; 3]]) -> f32 {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for position in positions {
        let point = Vec3::from(*position);
        min = min.min(point);
        max = max.max(point);
    }
    if !min.is_finite() || !max.is_finite() {
        return 1e-9;
    }
    let diagonal = (max - min).length();
    (diagonal * 1e-6).max(1e-9)
}

/// Map each vertex to a welded representative index.
pub fn weld_map(positions: &[[f32; 3]]) -> Vec<usize> {
    let epsilon = weld_epsilon(positions);
    let inverse = 1.0 / epsilon;
    let mut cells: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let mut map = Vec::with_capacity(positions.len());
    for (index, position) in positions.iter().enumerate() {
        let point = Vec3::from(*position);
        let cell = (
            (point.x * inverse).floor() as i64,
            (point.y * inverse).floor() as i64,
            (point.z * inverse).floor() as i64,
        );
        let mut found = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                    if let Some(candidates) = cells.get(&key) {
                        for candidate in candidates {
                            if Vec3::from(positions[*candidate]).distance(point) <= epsilon {
                                found = Some(*candidate);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        match found {
            Some(candidate) => map.push(candidate),
            None => {
                cells.entry(cell).or_default().push(index);
                map.push(index);
            }
        }
    }
    map
}

/// Area-weighted smooth normals across welded vertices.
pub fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let weld = weld_map(positions);
    let mut accumulated = vec![Vec3::ZERO; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let (ia, ib, ic) = (
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        );
        let (a, b, c) = (
            Vec3::from(positions[ia]),
            Vec3::from(positions[ib]),
            Vec3::from(positions[ic]),
        );
        let normal = (b - a).cross(c - a);
        for index in [ia, ib, ic] {
            accumulated[weld[index]] += normal;
        }
    }
    let mut normals = vec![[0.0f32; 3]; positions.len()];
    for index in 0..positions.len() {
        let normal = accumulated[weld[index]].try_normalize();
        normals[index] = normal.unwrap_or(Vec3::ZERO).to_array();
    }
    normals
}

/// Duplicate vertices per triangle and write face normals.
pub fn flatten(primitive: &Primitive) -> Primitive {
    let triangle_count = primitive.indices.len() / 3;
    let mut out = Primitive::new(
        Vec::with_capacity(triangle_count * 3),
        Vec::with_capacity(triangle_count * 3),
    );
    out.material = primitive.material;
    out.normals = Some(Vec::with_capacity(triangle_count * 3));
    if !primitive.uvs.is_empty() {
        out.uvs = vec![Vec::with_capacity(triangle_count * 3); primitive.uvs.len()];
    }
    if primitive.colors.is_some() {
        out.colors = Some(Vec::with_capacity(triangle_count * 3));
    }
    for triangle in primitive.indices.chunks_exact(3) {
        let (ia, ib, ic) = (
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        );
        let a = Vec3::from(primitive.positions[ia]);
        let b = Vec3::from(primitive.positions[ib]);
        let c = Vec3::from(primitive.positions[ic]);
        let normal = (b - a)
            .cross(c - a)
            .try_normalize()
            .unwrap_or(Vec3::ZERO)
            .to_array();
        for index in [ia, ib, ic] {
            out.positions.push(primitive.positions[index]);
            out.normals.as_mut().unwrap().push(normal);
            for (set, uvs) in primitive.uvs.iter().enumerate() {
                out.uvs[set].push(uvs[index]);
            }
            if let Some(colors) = &primitive.colors {
                out.colors.as_mut().unwrap().push(colors[index]);
            }
        }
        let base = out.positions.len() as u32 - 3;
        out.indices.extend([base, base + 1, base + 2]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_corner_normals_accumulate() {
        // Two triangles sharing an edge at 90 degrees.
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let indices = [0, 1, 2, 0, 3, 1];
        let normals = smooth_normals(&positions, &indices);
        let shared = Vec3::from(normals[0]);
        assert!(shared.length() > 0.99);
        assert!(shared.z > 0.0 && shared.y > 0.0);
    }

    #[test]
    fn flat_mode_duplicates_vertices() {
        let mut primitive = Primitive::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        );
        assert!(apply(&mut primitive, NormalMode::Flat));
        assert_eq!(primitive.positions.len(), 3);
        assert_eq!(primitive.normals.as_ref().unwrap().len(), 3);
    }
}
