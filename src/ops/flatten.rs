//! Baking transforms into vertex data for flattened or split outputs.

use glam::{Mat4, Vec3};

use crate::ops::gpu::GpuEngine;
use crate::scene::{Primitive, Scene, transform::transform_normal, transform::transform_tangent};

/// Transform positions, normals, and tangents by `matrix` in place.
/// Mirroring transforms flip triangle winding so front faces stay consistent.
pub fn transform_primitive(primitive: &mut Primitive, matrix: Mat4) {
    let mirrored = matrix.determinant() < 0.0;
    for position in &mut primitive.positions {
        *position = matrix.transform_point3(Vec3::from(*position)).to_array();
    }
    if mirrored {
        for triangle in primitive.indices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
    }
    if let Some(normals) = &mut primitive.normals {
        for normal in normals {
            *normal = transform_normal(matrix, *normal);
        }
    }
    if let Some(tangents) = &mut primitive.tangents {
        for tangent in tangents {
            *tangent = transform_tangent(matrix, *tangent);
        }
    }
}

/// Transform a primitive, using the GPU when it is available and worthwhile.
pub fn transform_primitive_with(primitive: &mut Primitive, matrix: Mat4, gpu: Option<&GpuEngine>) {
    let accelerated = gpu
        .filter(|gpu| gpu.should_accelerate(primitive.vertex_count()))
        .map(|gpu| gpu.transform_primitive(primitive, matrix))
        .unwrap_or(false);
    if accelerated {
        if matrix.determinant() < 0.0 {
            for triangle in primitive.indices.chunks_exact_mut(3) {
                triangle.swap(1, 2);
            }
        }
    } else {
        transform_primitive(primitive, matrix);
    }
}

/// Bake `global * world(node)` into every mesh node and collapse the hierarchy.
pub fn bake_flattened(scene: &mut Scene, global: Mat4, gpu: Option<&GpuEngine>) {
    let worlds = scene.world_transforms();
    for (id, node) in scene.nodes.iter_mut().enumerate() {
        let Some(mesh_id) = node.mesh else { continue };
        let matrix = global * worlds[id];
        for primitive in &mut scene.meshes[mesh_id].primitives {
            transform_primitive_with(primitive, matrix, gpu);
        }
    }
    let mesh_nodes: Vec<usize> = scene
        .traversal()
        .into_iter()
        .filter(|id| scene.nodes[*id].mesh.is_some())
        .collect();
    for node in &mut scene.nodes {
        node.local_transform = Mat4::IDENTITY;
        node.parent = None;
        node.children.clear();
    }
    scene.roots = mesh_nodes;
}

/// Multiply root transforms by the global matrix, preserving the hierarchy.
pub fn apply_global_to_roots(scene: &mut Scene, global: Mat4) {
    for root in &scene.roots {
        scene.nodes[*root].local_transform = global * scene.nodes[*root].local_transform;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> Primitive {
        let mut primitive = Primitive::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        );
        primitive.normals = Some(vec![[0.0, 0.0, 1.0]; 3]);
        primitive.tangents = Some(vec![[1.0, 0.0, 0.0, 1.0]; 3]);
        primitive
    }

    #[test]
    fn non_uniform_scale_uses_inverse_transpose() {
        let mut primitive = triangle();
        transform_primitive(&mut primitive, Mat4::from_scale(Vec3::new(2.0, 1.0, 1.0)));
        assert_eq!(primitive.indices, vec![0, 1, 2]);
        for normal in primitive.normals.as_ref().unwrap() {
            assert!((Vec3::from(*normal) - Vec3::Z).length() < 1e-6);
        }
        let tangent = primitive.tangents.as_ref().unwrap()[0];
        assert!((Vec3::new(tangent[0], tangent[1], tangent[2]) - Vec3::X).length() < 1e-6);
        assert!((tangent[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mirror_flips_winding_and_handedness() {
        let mut primitive = triangle();
        transform_primitive(&mut primitive, Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0)));
        assert_eq!(primitive.indices, vec![0, 2, 1]);
        let tangent = primitive.tangents.as_ref().unwrap()[0];
        assert!(tangent[3] < 0.0);
    }
}
