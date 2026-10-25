//! OBJ + MTL reader and writer.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::io::{Format, Loaded, WriteOptions};
use crate::output::staging::Staging;
use crate::output::textures::TexturePlan;
use crate::report::{WarningKind, WarningSink};
use crate::scene::{
    AlphaMode, Image, Material, Mesh, Primitive, Scene, TextureRef, UnitResolution, UpAxis,
};

pub fn read(path: &Path) -> Result<Loaded> {
    let options = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ignore_points: true,
        ignore_lines: true,
    };
    let (models, materials) = tobj::load_obj(path, &options)
        .map_err(|error| Error::path(path, format!("invalid OBJ: {error:?}")))?;
    let (materials, material_error) = materials
        .map(|materials| (materials, None))
        .unwrap_or_else(|error| (Vec::new(), Some(error)));
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

    let mut scene = Scene::new(Format::Obj);
    scene.unit = UnitResolution::Unknown;
    scene.up_axis = UpAxis::Y;
    let mut warnings = WarningSink::default();

    let mut material_ids = Vec::with_capacity(materials.len());
    for material in &materials {
        let mut converted = Material::new(if material.name.is_empty() {
            format!("material_{}", material_ids.len())
        } else {
            material.name.clone()
        });
        if let Some(diffuse) = material.diffuse {
            converted.base_color = [diffuse[0], diffuse[1], diffuse[2], 1.0];
        }
        if let Some(dissolve) = material.dissolve {
            converted.base_color[3] = dissolve;
            if dissolve < 1.0 {
                converted.alpha_mode = AlphaMode::Blend;
            }
        }
        if let Some(emissive) = material.emissive {
            converted.emissive = emissive;
        }
        let shininess = material.shininess.unwrap_or(0.0).max(0.0);
        converted.roughness = (2.0 / (shininess + 2.0)).sqrt().clamp(0.0, 1.0);
        let unknown_maps = material
            .unknown_param
            .keys()
            .filter(|key| key.to_ascii_lowercase().starts_with("map_"))
            .count();
        if unknown_maps > 0 {
            warnings.add_for(
                Format::Obj,
                WarningKind::DroppedFeature,
                "unknown-mtl-maps",
                format!(
                    "dropped {unknown_maps} unsupported MTL map(s) on material '{}'",
                    converted.name
                ),
                unknown_maps,
            );
        }
        for (slot, texture_name) in [
            (0u8, material.diffuse_texture.as_ref()),
            (1, material.normal_texture.as_ref()),
            (2, material.specular_texture.as_ref()),
        ] {
            let Some(texture_name) = texture_name else {
                continue;
            };
            let texture_path = resolve_texture(base_dir, texture_name);
            match fs::read(&texture_path) {
                Ok(bytes) => {
                    let ext = texture_path
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .unwrap_or("png");
                    let image = Image {
                        name: texture_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("texture")
                            .to_string(),
                        mime: Image::mime_for_extension(ext).map(str::to_string),
                        bytes,
                        source_path: Some(texture_path.clone()),
                    };
                    let image_id = scene.add_image(image);
                    let reference = Some(TextureRef { image: image_id });
                    match slot {
                        0 => converted.textures.base_color = reference,
                        1 => converted.textures.normal = reference,
                        2 => converted.textures.metallic_roughness = reference,
                        _ => {}
                    }
                }
                Err(_) => warnings.add(
                    WarningKind::MissingAsset,
                    "texture-missing",
                    format!(
                        "texture '{texture_name}' referenced by material '{}' was not found",
                        converted.name
                    ),
                ),
            }
        }
        material_ids.push(scene.add_material(converted));
    }
    if let Some(error) = material_error {
        warnings.add(
            WarningKind::MissingAsset,
            "mtl-missing",
            format!("could not load OBJ material library: {error:?}"),
        );
    }

    if models.is_empty() {
        return Err(Error::path(path, "OBJ contains no geometry"));
    }
    for (index, model) in models.into_iter().enumerate() {
        let mesh = model.mesh;
        if mesh.positions.is_empty() || mesh.indices.is_empty() {
            continue;
        }
        if mesh.positions.len() % 3 != 0 {
            return Err(Error::path(
                path,
                "OBJ position array is not a multiple of three",
            ));
        }
        let vertex_count = mesh.positions.len() / 3;
        let mut positions = Vec::with_capacity(vertex_count);
        for chunk in mesh.positions.chunks_exact(3) {
            if chunk.iter().any(|value| !value.is_finite()) {
                return Err(Error::path(path, "OBJ contains a non-finite position"));
            }
            positions.push([chunk[0], chunk[1], chunk[2]]);
        }
        if mesh
            .indices
            .iter()
            .any(|index| *index as usize >= vertex_count)
        {
            return Err(Error::path(
                path,
                "OBJ contains an out-of-range vertex index",
            ));
        }
        let mut primitive = Primitive::new(positions, mesh.indices);
        if !mesh.normals.is_empty() && mesh.normals.len() == vertex_count * 3 {
            primitive.normals = Some(
                mesh.normals
                    .chunks_exact(3)
                    .map(|c| [c[0], c[1], c[2]])
                    .collect(),
            );
        }
        if !mesh.texcoords.is_empty() && mesh.texcoords.len() == vertex_count * 2 {
            primitive.uvs.push(
                mesh.texcoords
                    .chunks_exact(2)
                    .map(|c| [c[0], c[1]])
                    .collect(),
            );
        }
        if !mesh.vertex_color.is_empty() && mesh.vertex_color.len() >= vertex_count * 3 {
            primitive.colors = Some(
                mesh.vertex_color
                    .chunks_exact(3)
                    .take(vertex_count)
                    .map(|c| [c[0], c[1], c[2], 1.0])
                    .collect(),
            );
        }
        primitive.material = mesh
            .material_id
            .and_then(|id| material_ids.get(id).copied());
        let name = if model.name.is_empty() {
            format!("object_{index}")
        } else {
            model.name
        };
        let mut scene_mesh = Mesh::new(name.clone());
        scene_mesh.primitives.push(primitive);
        let mesh_id = scene.add_mesh(scene_mesh);
        let node = scene.add_node(name, glam::Mat4::IDENTITY);
        scene.nodes[node].mesh = Some(mesh_id);
    }
    if scene.meshes.is_empty() {
        return Err(Error::path(path, "OBJ contains no triangle geometry"));
    }
    Ok(Loaded { scene, warnings })
}

fn resolve_texture(base_dir: &Path, name: &str) -> PathBuf {
    let candidate = Path::new(name);
    if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        base_dir.join(candidate)
    }
}

pub fn write(
    options: &WriteOptions<'_>,
    staging: &mut Staging,
    textures: &mut TexturePlan,
    warnings: &mut WarningSink,
) -> Result<()> {
    let scene = options.scene;
    if options.up_axis != Format::Obj.default_up_axis() {
        warnings.add_for(
            Format::Obj,
            WarningKind::Note,
            "obj-up-axis",
            "OBJ has no up-axis metadata; wrote non-Y-up coordinates",
            1,
        );
    }
    let temp_path = staging.stage(options.output_path)?;
    let file = fs::File::create(&temp_path).map_err(|error| Error::write(&temp_path, error))?;
    let mut writer = BufWriter::new(file);

    let use_materials = !options.geometry_only && !scene.materials.is_empty();
    let mut material_names: HashMap<usize, String> = HashMap::new();
    let mut image_paths: HashMap<usize, String> = HashMap::new();
    if use_materials {
        let mut used: Vec<usize> = scene
            .meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .filter_map(|primitive| primitive.material)
            .collect();
        used.sort_unstable();
        used.dedup();
        let image_ids: Vec<usize> = {
            let mut ids: Vec<usize> = used
                .iter()
                .filter_map(|id| scene.materials.get(*id))
                .flat_map(|material| material.textures.images())
                .collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        image_paths = textures
            .stage_for(
                staging,
                warnings,
                scene,
                options.output_path,
                options.texture_dir,
                &image_ids,
            )?
            .into_iter()
            .collect();
        let mut used_names: HashSet<String> = HashSet::new();
        for id in used {
            let material = &scene.materials[id];
            let base = sanitize_material_name(&material.name, id);
            let mut name = base.clone();
            let mut suffix = 2usize;
            while !used_names.insert(name.clone()) {
                name = format!("{base}_{suffix}");
                suffix += 1;
            }
            material_names.insert(id, name);
        }
        if !material_names.is_empty() {
            let mtl_name = options
                .output_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("scene")
                .to_string()
                + ".mtl";
            writeln!(writer, "mtllib {mtl_name}")
                .map_err(|error| Error::write(&temp_path, error))?;
        }
    }

    if let Some(parent) = options.output_path.parent() {
        let _ = parent;
    }
    let mut position_base = 1u32;
    let mut uv_base = 1u32;
    let mut normal_base = 1u32;
    let node_names: Vec<String> = scene
        .roots
        .iter()
        .map(|id| scene.nodes[*id].name.clone())
        .collect();
    for (root_index, root) in scene.roots.iter().enumerate() {
        let node = &scene.nodes[*root];
        let Some(mesh_id) = node.mesh else { continue };
        let mesh = &scene.meshes[mesh_id];
        let group_name = if node_names[root_index].is_empty() {
            format!("object_{root_index}")
        } else {
            sanitize_object_name(&node_names[root_index])
        };
        writeln!(writer, "o {group_name}").map_err(|error| Error::write(&temp_path, error))?;
        let multi_primitive = mesh
            .primitives
            .iter()
            .filter(|p| !p.indices.is_empty())
            .count()
            > 1;
        for (primitive_index, primitive) in mesh.primitives.iter().enumerate() {
            if primitive.indices.is_empty() {
                continue;
            }
            if multi_primitive {
                writeln!(writer, "g {group_name}_{primitive_index}")
                    .map_err(|error| Error::write(&temp_path, error))?;
            }
            if use_materials {
                if let Some(material_id) = primitive.material {
                    if let Some(name) = material_names.get(&material_id) {
                        writeln!(writer, "usemtl {name}")
                            .map_err(|error| Error::write(&temp_path, error))?;
                    }
                }
            }
            for position in &primitive.positions {
                writeln!(
                    writer,
                    "v {} {} {}",
                    number(position[0]),
                    number(position[1]),
                    number(position[2])
                )
                .map_err(|error| Error::write(&temp_path, error))?;
            }
            if let Some(uvs) = primitive.uv0() {
                for uv in uvs {
                    writeln!(writer, "vt {} {}", number(uv[0]), number(uv[1]))
                        .map_err(|error| Error::write(&temp_path, error))?;
                }
            }
            if let Some(normals) = &primitive.normals {
                for normal in normals {
                    writeln!(
                        writer,
                        "vn {} {} {}",
                        number(normal[0]),
                        number(normal[1]),
                        number(normal[2])
                    )
                    .map_err(|error| Error::write(&temp_path, error))?;
                }
            }
            let has_uv = primitive.uv0().is_some();
            let has_normal = primitive.normals.is_some();
            for triangle in primitive.indices.chunks_exact(3) {
                let mut face = String::from("f");
                for index in triangle {
                    let position = position_base + *index;
                    match (has_uv, has_normal) {
                        (true, true) => {
                            face.push_str(&format!(
                                " {}/{}/{}",
                                position,
                                uv_base + *index,
                                normal_base + *index
                            ));
                        }
                        (true, false) => {
                            face.push_str(&format!(" {}/{}", position, uv_base + *index));
                        }
                        (false, true) => {
                            face.push_str(&format!(" {}//{}", position, normal_base + *index));
                        }
                        (false, false) => {
                            face.push_str(&format!(" {position}"));
                        }
                    }
                }
                writeln!(writer, "{face}").map_err(|error| Error::write(&temp_path, error))?;
            }
            position_base += primitive.positions.len() as u32;
            if has_uv {
                uv_base += primitive.positions.len() as u32;
            }
            if has_normal {
                normal_base += primitive.positions.len() as u32;
            }
        }
    }
    writer
        .flush()
        .map_err(|error| Error::write(&temp_path, error))?;

    if use_materials && !material_names.is_empty() {
        write_mtl(options, staging, warnings, &material_names, &image_paths)?;
    }

    let approximated = scene
        .materials
        .iter()
        .enumerate()
        .filter(|(id, material)| material_names.contains_key(id) && material.metallic > 0.001)
        .count();
    if approximated > 0 {
        warnings.add_for(
            Format::Obj,
            WarningKind::Approximation,
            "pbr-to-phong",
            format!("OBJ/MTL has no metallic model; aproximated metallic values on {approximated} material(s)"),
            approximated,
        );
    }
    Ok(())
}

fn write_mtl(
    options: &WriteOptions<'_>,
    staging: &mut Staging,
    warnings: &mut WarningSink,
    material_names: &HashMap<usize, String>,
    image_paths: &HashMap<usize, String>,
) -> Result<()> {
    let scene = options.scene;
    let mtl_path = options.output_path.with_extension("mtl");
    let temp_path = staging.stage(&mtl_path)?;
    let file = fs::File::create(&temp_path).map_err(|error| Error::write(&temp_path, error))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "# trine material library")
        .map_err(|error| Error::write(&temp_path, error))?;
    let mut ids: Vec<usize> = material_names.keys().copied().collect();
    ids.sort_unstable();
    for id in ids {
        let material = &scene.materials[id];
        let name = &material_names[&id];
        writeln!(writer, "\nnewmtl {name}").map_err(|error| Error::write(&temp_path, error))?;
        writeln!(
            writer,
            "Kd {} {} {}",
            number(material.base_color[0]),
            number(material.base_color[1]),
            number(material.base_color[2])
        )
        .map_err(|error| Error::write(&temp_path, error))?;
        let specular = 1.0 - material.roughness;
        writeln!(
            writer,
            "Ks {} {} {}",
            number(specular),
            number(specular),
            number(specular)
        )
        .map_err(|error| Error::write(&temp_path, error))?;
        let shininess = ((2.0 / material.roughness.max(1e-4).powi(2)) - 2.0).clamp(1.0, 1000.0);
        writeln!(writer, "Ns {}", number(shininess))
            .map_err(|error| Error::write(&temp_path, error))?;
        writeln!(writer, "d {}", number(material.base_color[3]))
            .map_err(|error| Error::write(&temp_path, error))?;
        writeln!(
            writer,
            "Ke {} {} {}",
            number(material.emissive[0]),
            number(material.emissive[1]),
            number(material.emissive[2])
        )
        .map_err(|error| Error::write(&temp_path, error))?;
        writeln!(writer, "illum 2").map_err(|error| Error::write(&temp_path, error))?;
        if let Some(slot) = material.textures.base_color {
            if let Some(path) = image_paths.get(&slot.image) {
                writeln!(writer, "map_Kd {path}")
                    .map_err(|error| Error::write(&temp_path, error))?;
            }
        }
        if let Some(slot) = material.textures.normal {
            if let Some(path) = image_paths.get(&slot.image) {
                writeln!(writer, "map_Bump {path}")
                    .map_err(|error| Error::write(&temp_path, error))?;
            }
        }
        if let Some(slot) = material.textures.metallic_roughness {
            if let Some(path) = image_paths.get(&slot.image) {
                writeln!(writer, "map_Ks {path}")
                    .map_err(|error| Error::write(&temp_path, error))?;
            }
        }
        if let Some(slot) = material.textures.emissive {
            if let Some(path) = image_paths.get(&slot.image) {
                writeln!(writer, "map_Ke {path}")
                    .map_err(|error| Error::write(&temp_path, error))?;
            }
        }
    }
    writer
        .flush()
        .map_err(|error| Error::write(&temp_path, error))?;
    if scene
        .materials
        .iter()
        .any(|material| material.textures.is_empty())
    {
        warnings.add_for(
            Format::Obj,
            WarningKind::Note,
            "mtl-written",
            "wrote sibling MTL material library",
            1,
        );
    }
    Ok(())
}

fn sanitize_material_name(name: &str, id: usize) -> String {
    let cleaned: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.trim_matches('_').is_empty() {
        format!("material_{id}")
    } else {
        cleaned
    }
}

fn sanitize_object_name(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_whitespace() { '_' } else { ch })
        .filter(|ch| *ch != '\n' && *ch != '\r')
        .collect()
}

fn number(value: f32) -> String {
    if !value.is_finite() || value == 0.0 {
        "0".to_string()
    } else {
        format!("{value}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_names_are_sanitized_and_unique_by_suffix() {
        assert_eq!(sanitize_material_name("red wood", 0), "red_wood");
        assert_eq!(sanitize_material_name("", 4), "material_4");
        assert_eq!(sanitize_object_name("a b\nc"), "a_b_c");
        assert_eq!(number(f32::NAN), "0");
        assert_eq!(number(f32::INFINITY), "0");
    }
}
