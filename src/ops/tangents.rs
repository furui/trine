//! Per-vertex tangent generation with Gram-Schmidt orthonormalization.

use glam::{Vec2, Vec3};

use crate::ops::gpu::GpuEngine;
use crate::scene::Primitive;

/// Generate tangents, preferring the GPU when available.
pub fn generate_with(primitive: &Primitive, gpu: Option<&GpuEngine>) -> Option<Vec<[f32; 4]>> {
    if let Some(gpu) = gpu {
        if gpu.should_accelerate(primitive.vertex_count()) {
            let mut copy = primitive.clone();
            if gpu.generate_tangents(&mut copy) {
                return copy.tangents;
            }
        }
    }
    generate_tangents(primitive)
}

/// Generate `[tangent.x, tangent.y, tangent.z, handedness]` for a primitive
/// that has positions, normals, and UV set 0.
pub fn generate_tangents(primitive: &Primitive) -> Option<Vec<[f32; 4]>> {
    let normals = primitive.normals.as_ref()?;
    let uvs = primitive.uv0()?;
    if normals.len() != primitive.positions.len() || uvs.len() != primitive.positions.len() {
        return None;
    }
    let count = primitive.positions.len();
    let mut tangents = vec![Vec3::ZERO; count];
    let mut bitangents = vec![Vec3::ZERO; count];
    for triangle in primitive.indices.chunks_exact(3) {
        let (ia, ib, ic) = (
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        );
        let p0 = Vec3::from(primitive.positions[ia]);
        let p1 = Vec3::from(primitive.positions[ib]);
        let p2 = Vec3::from(primitive.positions[ic]);
        let uv0 = Vec2::from(uvs[ia]);
        let uv1 = Vec2::from(uvs[ib]);
        let uv2 = Vec2::from(uvs[ic]);
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let delta1 = uv1 - uv0;
        let delta2 = uv2 - uv0;
        let determinant = delta1.x * delta2.y - delta2.x * delta1.y;
        if determinant.abs() <= 1e-12 {
            continue;
        }
        let reciprocal = 1.0 / determinant;
        let tangent = (edge1 * delta2.y - edge2 * delta1.y) * reciprocal;
        let bitangent = (edge2 * delta1.x - edge1 * delta2.x) * reciprocal;
        for index in [ia, ib, ic] {
            tangents[index] += tangent;
            bitangents[index] += bitangent;
        }
    }
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let normal = Vec3::from(normals[index])
            .try_normalize()
            .unwrap_or(Vec3::Y);
        let mut tangent = tangents[index];
        tangent -= normal * normal.dot(tangent);
        let tangent = tangent.try_normalize().unwrap_or_else(|| {
            let helper = if normal.x.abs() > 0.9 {
                Vec3::Y
            } else {
                Vec3::X
            };
            normal.cross(helper).normalize_or_zero()
        });
        let bitangent = bitangents[index];
        let handedness = if normal.cross(tangent).dot(bitangent) < 0.0 {
            -1.0
        } else {
            1.0
        };
        out.push([tangent.x, tangent.y, tangent.z, handedness]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_tangent_points_along_u() {
        let mut primitive = Primitive::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![0, 1, 2, 0, 2, 3],
        );
        primitive.normals = Some(vec![[0.0, 0.0, 1.0]; 4]);
        primitive.uvs = vec![vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]];
        let tangents = generate_tangents(&primitive).expect("tangents");
        assert!((tangents[0][0] - 1.0).abs() < 1e-5);
        assert!((tangents[0][3] - 1.0).abs() < 1e-5);
    }
}
