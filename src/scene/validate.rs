//! Structural and numeric scene validation.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::scene::{Primitive, Scene};

/// Validate indices, finite values, and node graph integrity.
pub fn validate(scene: &Scene) -> Result<()> {
    if scene.nodes.is_empty() {
        return Err(Error::conversion("scene contains no nodes"));
    }
    for (index, node) in scene.nodes.iter().enumerate() {
        if let Some(mesh) = node.mesh {
            if mesh >= scene.meshes.len() {
                return Err(Error::conversion(format!(
                    "node {index} references missing mesh {mesh}"
                )));
            }
        }
        if let Some(parent) = node.parent {
            if parent >= scene.nodes.len() {
                return Err(Error::conversion(format!(
                    "node {index} references missing parent {parent}"
                )));
            }
            if !scene.nodes[parent].children.contains(&index) {
                return Err(Error::conversion(format!(
                    "node {index} parent {parent} does not list it as a child"
                )));
            }
        }
        if !node.local_transform.is_finite() {
            return Err(Error::conversion(format!(
                "node {index} ('{}') has a non-finite transform",
                node.name
            )));
        }
    }
    for root in &scene.roots {
        if *root >= scene.nodes.len() {
            return Err(Error::conversion(format!(
                "root node {root} does not exist"
            )));
        }
    }

    for (material_index, material) in scene.materials.iter().enumerate() {
        for (_, slot) in material.textures.iter() {
            if slot.image >= scene.images.len() {
                return Err(Error::conversion(format!(
                    "material {material_index} references missing image {}",
                    slot.image
                )));
            }
        }
    }

    for (mesh_index, mesh) in scene.meshes.iter().enumerate() {
        for (primitive_index, primitive) in mesh.primitives.iter().enumerate() {
            if let Some(material) = primitive.material {
                if material >= scene.materials.len() {
                    return Err(Error::conversion(format!(
                        "mesh {mesh_index} ('{}') primitive {primitive_index} references missing material {material}",
                        mesh.name
                    )));
                }
            }
            validate_primitive(primitive).map_err(|message| {
                Error::conversion(format!(
                    "mesh {mesh_index} ('{}') primitive {primitive_index}: {message}",
                    mesh.name
                ))
            })?;
        }
    }

    // Reject parent cycles.
    for start in 0..scene.nodes.len() {
        let mut current = Some(start);
        let mut steps = 0usize;
        while let Some(id) = current {
            current = scene.nodes[id].parent;
            steps += 1;
            if steps > scene.nodes.len() {
                return Err(Error::conversion(format!(
                    "node hierarchy contains a cycle at node {start}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_primitive(primitive: &Primitive) -> Result<(), String> {
    if primitive.positions.is_empty() {
        return Err("has no positions".into());
    }
    if !primitive.indices.len().is_multiple_of(3) {
        return Err(format!(
            "index count {} is not a multiple of three",
            primitive.indices.len()
        ));
    }
    let vertex_count = primitive.positions.len();
    for (index, position) in primitive.positions.iter().enumerate() {
        if !position.iter().all(|value| value.is_finite()) {
            return Err(format!("position {index} is not finite"));
        }
    }
    for (triangle, chunk) in primitive.indices.chunks_exact(3).enumerate() {
        for index in chunk {
            if *index as usize >= vertex_count {
                return Err(format!(
                    "triangle {triangle} references vertex {index}, but only {vertex_count} vertices exist"
                ));
            }
        }
    }
    if let Some(normals) = &primitive.normals {
        if normals.len() != vertex_count {
            return Err(format!(
                "normal count {} does not match vertex count {vertex_count}",
                normals.len()
            ));
        }
        if normals.iter().flatten().any(|value| !value.is_finite()) {
            return Err("contains non-finite normals".into());
        }
    }
    if let Some(tangents) = &primitive.tangents {
        if tangents.len() != vertex_count {
            return Err(format!(
                "tangent count {} does not match vertex count {vertex_count}",
                tangents.len()
            ));
        }
        if tangents.iter().flatten().any(|value| !value.is_finite()) {
            return Err("contains non-finite tangents".into());
        }
    }
    for (set, uvs) in primitive.uvs.iter().enumerate() {
        if uvs.len() != vertex_count {
            return Err(format!(
                "UV set {set} has {} entries, expected {vertex_count}",
                uvs.len()
            ));
        }
        if uvs.iter().flatten().any(|value| !value.is_finite()) {
            return Err(format!("UV set {set} contains non-finite values"));
        }
    }
    if let Some(colors) = &primitive.colors {
        if colors.len() != vertex_count {
            return Err(format!(
                "color count {} does not match vertex count {vertex_count}",
                colors.len()
            ));
        }
        if colors.iter().flatten().any(|value| !value.is_finite()) {
            return Err("contains non-finite vertex colors".into());
        }
    }
    Ok(())
}

/// Edge-manifold and closure analysis for STEP output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    pub closed: bool,
    pub manifold: bool,
}

pub fn topology(indices: &[u32]) -> Topology {
    let mut directed: HashMap<(u32, u32), i32> = HashMap::new();
    let mut undirected: HashMap<(u32, u32), u32> = HashMap::new();
    for triangle in indices.chunks_exact(3) {
        let (a, b, c) = (triangle[0], triangle[1], triangle[2]);
        if a == b || b == c || c == a {
            continue;
        }
        for (from, to) in [(a, b), (b, c), (c, a)] {
            *directed.entry((from, to)).or_insert(0) += 1;
            let key = if from < to { (from, to) } else { (to, from) };
            *undirected.entry(key).or_insert(0) += 1;
        }
    }
    let manifold = undirected.values().all(|count| *count <= 2);
    let closed = manifold
        && !directed.is_empty()
        && directed
            .iter()
            .all(|((from, to), count)| *count == 1 && directed.contains_key(&(*to, *from)));
    Topology { closed, manifold }
}

/// Connected components of a triangle list, returned as sorted triangle indices.
pub fn connected_components(indices: &[u32]) -> Vec<Vec<usize>> {
    let triangle_count = indices.len() / 3;
    let mut parent: Vec<usize> = (0..triangle_count).collect();

    fn find(parent: &mut [usize], mut id: usize) -> usize {
        while parent[id] != id {
            parent[id] = parent[parent[id]];
            id = parent[id];
        }
        id
    }

    let mut edge_owner: HashMap<(u32, u32), usize> = HashMap::new();
    for triangle in 0..triangle_count {
        let base = triangle * 3;
        let (a, b, c) = (indices[base], indices[base + 1], indices[base + 2]);
        for (from, to) in [(a, b), (b, c), (c, a)] {
            let key = if from < to { (from, to) } else { (to, from) };
            if let Some(other) = edge_owner.get(&key).copied() {
                let root_a = find(&mut parent, triangle);
                let root_b = find(&mut parent, other);
                if root_a != root_b {
                    parent[root_a] = root_b;
                }
            } else {
                edge_owner.insert(key, triangle);
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for triangle in 0..triangle_count {
        let root = find(&mut parent, triangle);
        groups.entry(root).or_default().push(triangle);
    }
    let mut components: Vec<Vec<usize>> = groups.into_values().collect();
    components.sort_by_key(|component| component.first().copied().unwrap_or(0));
    components
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Format;
    use crate::scene::{Mesh, Node, Primitive, Scene};

    fn scene_with(primitive: Primitive) -> Scene {
        let mut scene = Scene::new(Format::Obj);
        let mut mesh = Mesh::new("mesh");
        mesh.primitives.push(primitive);
        let mesh_id = scene.add_mesh(mesh);
        let node = scene.add_node("node", glam::Mat4::IDENTITY);
        scene.nodes[node].mesh = Some(mesh_id);
        scene
    }

    #[test]
    fn rejects_out_of_range_indices() {
        let scene = scene_with(Primitive::new(
            vec![[0.0; 3], [1.0; 3], [2.0; 3]],
            vec![0, 1, 9],
        ));
        assert!(validate(&scene).is_err());
    }

    #[test]
    fn rejects_non_finite_positions() {
        let scene = scene_with(Primitive::new(
            vec![[0.0; 3], [f32::NAN, 0.0, 0.0], [2.0; 3]],
            vec![0, 1, 2],
        ));
        assert!(validate(&scene).is_err());
    }

    #[test]
    fn rejects_parent_cycles() {
        let mut scene = Scene::new(Format::Obj);
        scene.nodes.push(Node::new("a", glam::Mat4::IDENTITY));
        scene.nodes.push(Node::new("b", glam::Mat4::IDENTITY));
        scene.nodes[0].parent = Some(1);
        scene.nodes[1].parent = Some(0);
        scene.nodes[0].children.push(1);
        scene.nodes[1].children.push(0);
        assert!(validate(&scene).is_err());
    }

    #[test]
    fn accepts_a_valid_triangle() {
        let scene = scene_with(Primitive::new(
            vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        ));
        assert!(validate(&scene).is_ok());
    }

    #[test]
    fn open_triangle_is_not_closed() {
        let result = topology(&[0, 1, 2]);
        assert!(!result.closed);
        assert!(result.manifold);
    }

    #[test]
    fn double_sided_quad_is_closed() {
        // Two triangles covering a square, each edge used exactly twice.
        let result = topology(&[0, 1, 2, 0, 2, 3]);
        assert!(!result.closed, "open plate is not a closed solid");
        assert!(result.manifold);
    }

    #[test]
    fn components_split_by_connectivity() {
        let components = connected_components(&[0, 1, 2, 3, 4, 5]);
        assert_eq!(components.len(), 2);
    }
}
