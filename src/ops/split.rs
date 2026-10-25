//! Group enumeration, piece extraction, and output naming for `--split-groups`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use glam::Mat4;

use crate::error::{Error, Result};
use crate::io::Format;
use crate::ops::flatten::transform_primitive_with;
use crate::ops::gpu::GpuEngine;
use crate::scene::{Image, ImageId, Mesh, NodeId, Scene};

/// One output piece derived from a mesh-bearing node.
#[derive(Debug, Clone)]
pub struct Piece {
    pub node: NodeId,
    pub name: String,
    pub scene: Scene,
}

/// A source group eligible for splitting.
#[derive(Debug, Clone)]
pub struct Group {
    pub node: Option<NodeId>,
    pub name: String,
}

impl Group {
    pub fn synthetic(name: impl Into<String>) -> Self {
        Self {
            node: None,
            name: name.into(),
        }
    }
}

/// Enumerate non-empty groups in deterministic pre-order.
pub fn enumerate_groups(scene: &Scene) -> Vec<Group> {
    let mut groups = Vec::new();
    let mut used: HashSet<String> = HashSet::new();
    let mut empty_counter = 0usize;
    for node_id in scene.traversal() {
        let node = &scene.nodes[node_id];
        let Some(mesh_id) = node.mesh else { continue };
        let mesh = &scene.meshes[mesh_id];
        if mesh.triangle_count() == 0 {
            continue;
        }
        let mut name = sanitize_group_name(&node.name);
        if name.is_empty() {
            empty_counter += 1;
            name = format!("group_{empty_counter:04}");
        }
        groups.push(Group {
            node: Some(node_id),
            name,
        });
        let _ = &mut used;
    }
    dedupe_names(&mut groups);
    groups
}

/// Sanitize to `[A-Za-z0-9._-]`, collapse separators, trim, cap at 80 chars.
pub fn sanitize_group_name(name: &str) -> String {
    let mut out = String::new();
    let mut last_separator = false;
    for ch in name.chars() {
        let allowed = ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-');
        if !allowed {
            continue;
        }
        if matches!(ch, '.' | '_' | '-') {
            if last_separator {
                continue;
            }
            last_separator = true;
        } else {
            last_separator = false;
        }
        out.push(ch);
        if out.len() >= 80 {
            break;
        }
    }
    out.trim_matches(|ch| matches!(ch, '.' | '_' | '-'))
        .to_string()
}

fn dedupe_names(groups: &mut [Group]) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut used: HashSet<String> = HashSet::new();
    for group in groups {
        if used.insert(group.name.clone()) {
            counts.insert(group.name.clone(), 1);
            continue;
        }
        let base = group.name.clone();
        let mut index = counts.get(&base).copied().unwrap_or(1) + 1;
        loop {
            let candidate = format!("{base}_{index}");
            if used.insert(candidate.clone()) {
                counts.insert(base.clone(), index);
                group.name = candidate;
                break;
            }
            index += 1;
        }
    }
}

/// Extract one node's own mesh as a standalone scene with baked world transform.
pub fn extract_piece(
    scene: &Scene,
    node_id: NodeId,
    global: Mat4,
    name: &str,
    gpu: Option<&GpuEngine>,
) -> Result<Scene> {
    let node = scene
        .nodes
        .get(node_id)
        .ok_or_else(|| Error::conversion(format!("group node {node_id} does not exist")))?;
    let mesh_id = node
        .mesh
        .ok_or_else(|| Error::conversion(format!("group node {node_id} has no mesh")))?;
    let worlds = scene.world_transforms();
    let matrix = global * worlds[node_id];

    let mut piece = Scene::new(scene.source_format);
    piece.unit = scene.unit;
    piece.up_axis = scene.up_axis;
    piece.up_axis_from_file = scene.up_axis_from_file;

    let mut material_map: HashMap<usize, usize> = HashMap::new();
    let mut image_map: HashMap<ImageId, ImageId> = HashMap::new();
    let mut mesh = Mesh::new(name);
    for primitive in &scene.meshes[mesh_id].primitives {
        let mut primitive = primitive.clone();
        transform_primitive_with(&mut primitive, matrix, gpu);
        if let Some(material_id) = primitive.material {
            let new_id = match material_map.get(&material_id) {
                Some(id) => *id,
                None => {
                    let source = scene.materials.get(material_id).ok_or_else(|| {
                        Error::conversion(format!(
                            "primitive references missing material {material_id}"
                        ))
                    })?;
                    let mut material = source.clone();
                    for slot in [
                        material.textures.base_color.as_mut(),
                        material.textures.normal.as_mut(),
                        material.textures.metallic_roughness.as_mut(),
                        material.textures.emissive.as_mut(),
                        material.textures.occlusion.as_mut(),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        let source_image = slot.image;
                        let new_image = match image_map.get(&source_image) {
                            Some(id) => *id,
                            None => {
                                let image = scene.images.get(source_image).ok_or_else(|| {
                                    Error::conversion(format!(
                                        "material references missing image {source_image}"
                                    ))
                                })?;
                                let new_id = piece.add_image(Image {
                                    name: image.name.clone(),
                                    bytes: image.bytes.clone(),
                                    mime: image.mime.clone(),
                                    source_path: image.source_path.clone(),
                                });
                                image_map.insert(source_image, new_id);
                                new_id
                            }
                        };
                        slot.image = new_image;
                    }
                    let id = piece.add_material(material);
                    material_map.insert(material_id, id);
                    id
                }
            };
            primitive.material = Some(new_id);
        }
        mesh.primitives.push(primitive);
    }
    let new_mesh = piece.add_mesh(mesh);
    let root = piece.add_node(name.to_string(), Mat4::IDENTITY);
    piece.nodes[root].mesh = Some(new_mesh);
    Ok(piece)
}

fn append_extension(path: &mut PathBuf, extension: &str) {
    let mut text = path.as_os_str().to_os_string();
    text.push(".");
    text.push(extension);
    *path = PathBuf::from(text);
}

/// Resolve one output path per group.
pub fn resolve_piece_paths(
    output: &Path,
    input: &Path,
    format: Format,
    groups: &[Group],
) -> Result<Vec<PathBuf>> {
    let output_text = output.to_string_lossy().to_string();
    if groups.is_empty() {
        return Err(Error::conversion("no non-empty groups to split"));
    }
    let output_text_initial = output.to_string_lossy();
    let directory_output = output.is_dir()
        || output_text_initial.ends_with('/')
        || output_text_initial.ends_with(std::path::MAIN_SEPARATOR);
    let extension = if directory_output {
        format.extension().to_string()
    } else {
        output
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| format.extension().to_string())
    };
    let mut paths = Vec::with_capacity(groups.len());

    if output_text.contains("{group}") {
        let pattern_has_extension = output.extension().is_some();
        for group in groups {
            let replaced = output_text.replace("{group}", &group.name);
            let mut path = PathBuf::from(replaced);
            if !pattern_has_extension {
                append_extension(&mut path, &extension);
            }
            paths.push(path);
        }
        // A constant file name with only `{group}` in a directory can collide.
        let mut seen: HashSet<PathBuf> = HashSet::new();
        for (index, path) in paths.iter_mut().enumerate() {
            if seen.insert(path.clone()) {
                continue;
            }
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("output")
                .to_string();
            let name = format!("{stem}_{}", groups[index].name);
            path.set_file_name(name);
            append_extension(path, &extension);
            seen.insert(path.clone());
        }
        return Ok(paths);
    }

    let input_stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("output")
        .to_string();
    if directory_output {
        for group in groups {
            paths.push(output.join(format!("{input_stem}_{}.{}", group.name, extension)));
        }
        return Ok(paths);
    }

    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(&input_stem);
    for group in groups {
        paths.push(parent.join(format!("{stem}_{}.{}", group.name, extension)));
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_unicode_and_separators() {
        assert_eq!(sanitize_group_name("café / part: 01"), "cafpart01");
        assert_eq!(sanitize_group_name("  ...  "), "");
        assert_eq!(sanitize_group_name("a__b--c"), "a_b-c");
    }

    #[test]
    fn dedupe_names_suffixes() {
        let mut groups = vec![
            Group::synthetic("part"),
            Group::synthetic("part"),
            Group::synthetic("part"),
            Group::synthetic("other"),
        ];
        dedupe_names(&mut groups);
        assert_eq!(groups[0].name, "part");
        assert_eq!(groups[1].name, "part_2");
        assert_eq!(groups[2].name, "part_3");
        assert_eq!(groups[3].name, "other");
    }

    #[test]
    fn pattern_expands() {
        let groups = vec![Group::synthetic("a"), Group::synthetic("b")];
        let paths = resolve_piece_paths(
            Path::new("out/{group}.glb"),
            Path::new("model.obj"),
            Format::Glb,
            &groups,
        )
        .unwrap();
        assert_eq!(paths[0], PathBuf::from("out/a.glb"));
        assert_eq!(paths[1], PathBuf::from("out/b.glb"));
    }

    #[test]
    fn pattern_without_extension_uses_target() {
        let groups = vec![Group::synthetic("a")];
        let paths = resolve_piece_paths(
            Path::new("out/{group}"),
            Path::new("model.obj"),
            Format::Glb,
            &groups,
        )
        .unwrap();
        assert_eq!(paths[0], PathBuf::from("out/a.glb"));
    }

    #[test]
    fn dots_in_group_names_do_not_swallow_the_extension() {
        let groups = vec![Group::synthetic("part.1")];
        let paths = resolve_piece_paths(
            Path::new("out/{group}"),
            Path::new("model.obj"),
            Format::Glb,
            &groups,
        )
        .unwrap();
        assert_eq!(paths[0], PathBuf::from("out/part.1.glb"));
    }

    #[test]
    fn dotted_directory_names_keep_the_format_extension() {
        let groups = vec![Group::synthetic("a")];
        let paths = resolve_piece_paths(
            Path::new("out.v1/"),
            Path::new("model.obj"),
            Format::Glb,
            &groups,
        )
        .unwrap();
        assert_eq!(paths[0], PathBuf::from("out.v1/model_a.glb"));
    }

    #[test]
    fn directory_uses_input_stem() {
        let groups = vec![Group::synthetic("a")];
        let paths = resolve_piece_paths(
            Path::new("out/"),
            Path::new("model.obj"),
            Format::Glb,
            &groups,
        )
        .unwrap();
        assert_eq!(paths[0], PathBuf::from("out/model_a.glb"));
    }
}
