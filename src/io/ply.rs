//! PLY reader and binary little-endian writer backed by `ply-rs`.

use std::fs;
use std::io::BufReader;
use std::path::Path;

use ply_rs::ply::{DefaultElement, Property};

use crate::error::{Error, Result};
use crate::io::{Format, Loaded, WriteOptions};
use crate::output::staging::Staging;
use crate::report::{WarningKind, WarningSink};
use crate::scene::{Mesh, Primitive, Scene, UnitResolution, UpAxis};

pub fn read(path: &Path) -> Result<Loaded> {
    let file = fs::File::open(path).map_err(|error| Error::read(path, error))?;
    let mut reader = BufReader::new(file);
    let parser = ply_rs::parser::Parser::<DefaultElement>::new();
    let ply = parser
        .read_ply(&mut reader)
        .map_err(|error| Error::path(path, format!("invalid PLY: {error}")))?;

    let vertices = ply
        .payload
        .get("vertex")
        .ok_or_else(|| Error::path(path, "PLY has no vertex element"))?;
    if vertices.is_empty() {
        return Err(Error::path(path, "PLY contains no vertices"));
    }
    let mut positions = Vec::with_capacity(vertices.len());
    let mut normals = Vec::with_capacity(vertices.len());
    let mut uvs = Vec::with_capacity(vertices.len());
    let mut colors = Vec::with_capacity(vertices.len());
    let mut has_normals = true;
    let mut has_uvs = true;
    let mut has_colors = true;
    for (index, vertex) in vertices.iter().enumerate() {
        let x = number(vertex, &["x"])
            .ok_or_else(|| Error::path(path, format!("vertex {index} has no x coordinate")))?;
        let y = number(vertex, &["y"]).unwrap_or(0.0);
        let z = number(vertex, &["z"]).unwrap_or(0.0);
        positions.push([x as f32, y as f32, z as f32]);
        match (
            number(vertex, &["nx", "normal_x"]),
            number(vertex, &["ny", "normal_y"]),
            number(vertex, &["nz", "normal_z"]),
        ) {
            (Some(nx), Some(ny), Some(nz)) => normals.push([nx as f32, ny as f32, nz as f32]),
            _ => has_normals = false,
        }
        match (
            number(vertex, &["s", "u", "texture_u", "tx"]),
            number(vertex, &["t", "v", "texture_v", "ty"]),
        ) {
            (Some(s), Some(t)) => uvs.push([s as f32, t as f32]),
            _ => has_uvs = false,
        }
        match (
            color_component(vertex, "red"),
            color_component(vertex, "green"),
            color_component(vertex, "blue"),
        ) {
            (Some(r), Some(g), Some(b)) => {
                let a = color_component(vertex, "alpha").unwrap_or(1.0);
                colors.push([r, g, b, a]);
            }
            _ => has_colors = false,
        }
    }

    let faces = ply
        .payload
        .get("face")
        .or_else(|| ply.payload.get("faces"))
        .ok_or_else(|| Error::path(path, "PLY has no face element"))?;
    let mut indices: Vec<u32> = Vec::new();
    for (face_index, face) in faces.iter().enumerate() {
        let list = face_indices(face).ok_or_else(|| {
            Error::path(path, format!("face {face_index} has no vertex index list"))
        })?;
        if list.len() < 3 {
            continue;
        }
        let mut local = Vec::with_capacity(list.len());
        for value in list {
            if value as usize >= positions.len() {
                return Err(Error::path(
                    path,
                    format!(
                        "face {face_index} references vertex {value}, but only {} exist",
                        positions.len()
                    ),
                ));
            }
            local.push(value);
        }
        indices.extend(crate::scene::triangulate::triangulate_face(
            &positions, &local,
        ));
    }
    if indices.is_empty() {
        return Err(Error::path(path, "PLY contains no triangles"));
    }

    let mut primitive = Primitive::new(positions, indices);
    if has_normals {
        primitive.normals = Some(normals);
    }
    if has_uvs {
        primitive.uvs.push(uvs);
    }
    if has_colors {
        primitive.colors = Some(colors);
    }

    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("PLY")
        .to_string();
    let mut scene = Scene::new(Format::Ply);
    scene.unit = UnitResolution::Unknown;
    scene.up_axis = UpAxis::Z;
    let mut mesh = Mesh::new(name.clone());
    mesh.primitives.push(primitive);
    let mesh_id = scene.add_mesh(mesh);
    let node = scene.add_node(name, glam::Mat4::IDENTITY);
    scene.nodes[node].mesh = Some(mesh_id);
    Ok(Loaded::new(scene))
}

fn number(element: &DefaultElement, names: &[&str]) -> Option<f64> {
    for name in names {
        let Some(property) = element.get(*name) else {
            continue;
        };
        match property {
            Property::Char(v) => return Some(*v as f64),
            Property::UChar(v) => return Some(*v as f64),
            Property::Short(v) => return Some(*v as f64),
            Property::UShort(v) => return Some(*v as f64),
            Property::Int(v) => return Some(*v as f64),
            Property::UInt(v) => return Some(*v as f64),
            Property::Float(v) => return Some(*v as f64),
            Property::Double(v) => return Some(*v),
            _ => {}
        }
    }
    None
}

fn color_component(element: &DefaultElement, name: &str) -> Option<f32> {
    let property = element.get(name)?;
    match property {
        Property::Char(v) => Some((*v as f32 / 127.0).clamp(0.0, 1.0)),
        Property::UChar(v) => Some(*v as f32 / 255.0),
        Property::Short(v) => Some((*v as f32 / 32767.0).clamp(0.0, 1.0)),
        Property::UShort(v) => Some(*v as f32 / 65535.0),
        Property::Int(v) => Some((*v as f32 / 2_147_483_647.0).clamp(0.0, 1.0)),
        Property::UInt(v) => Some(*v as f32 / 4_294_967_295.0),
        Property::Float(v) => Some((*v).clamp(0.0, 1.0)),
        Property::Double(v) => Some((*v as f32).clamp(0.0, 1.0)),
        _ => None,
    }
}

fn face_indices(element: &DefaultElement) -> Option<Vec<u32>> {
    for name in ["vertex_indices", "vertex_index"] {
        let Some(property) = element.get(name) else {
            continue;
        };
        let values: Vec<u32> = match property {
            Property::ListChar(values) => values.iter().map(|v| *v as u32).collect(),
            Property::ListUChar(values) => values.iter().map(|v| *v as u32).collect(),
            Property::ListShort(values) => values.iter().map(|v| *v as u32).collect(),
            Property::ListUShort(values) => values.iter().map(|v| *v as u32).collect(),
            Property::ListInt(values) => values.iter().map(|v| *v as u32).collect(),
            Property::ListUInt(values) => values.clone(),
            _ => continue,
        };
        return Some(values);
    }
    None
}

pub fn write(
    options: &WriteOptions<'_>,
    staging: &mut Staging,
    warnings: &mut WarningSink,
) -> Result<()> {
    use std::io::Write;

    let scene = options.scene;
    if options.up_axis != Format::Ply.default_up_axis() {
        warnings.add_for(
            Format::Ply,
            WarningKind::Note,
            "ply-up-axis",
            "PLY has no up-axis metadata; wrote non-Z-up coordinates",
            1,
        );
    }
    let primitives: Vec<&Primitive> = scene
        .referenced_meshes()
        .into_iter()
        .flat_map(|mesh_id| &scene.meshes[mesh_id].primitives)
        .filter(|primitive| !primitive.indices.is_empty())
        .collect();
    if primitives.is_empty() {
        return Err(Error::conversion("no triangles to write to PLY"));
    }
    let write_normals = primitives
        .iter()
        .all(|primitive| primitive.normals.is_some());
    let write_uvs = primitives.iter().all(|primitive| primitive.uv0().is_some());
    let write_colors = primitives
        .iter()
        .all(|primitive| primitive.colors.is_some());

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for primitive in &primitives {
        let base = positions.len() as u32;
        positions.extend_from_slice(&primitive.positions);
        if write_normals {
            normals.extend_from_slice(primitive.normals.as_ref().expect("checked"));
        }
        if write_uvs {
            uvs.extend_from_slice(primitive.uv0().expect("checked"));
        }
        if write_colors {
            colors.extend_from_slice(primitive.colors.as_ref().expect("checked"));
        }
        indices.extend(primitive.indices.iter().map(|index| index + base));
    }

    let temp_path = staging.stage(options.output_path)?;
    let file = fs::File::create(&temp_path).map_err(|error| Error::write(&temp_path, error))?;
    let mut writer = std::io::BufWriter::new(file);

    let mut header = String::new();
    header.push_str("ply\nformat binary_little_endian 1.0\n");
    header.push_str("comment trine PLY output\n");
    header.push_str(&format!("element vertex {}\n", positions.len()));
    header.push_str("property float x\nproperty float y\nproperty float z\n");
    if write_normals {
        header.push_str("property float nx\nproperty float ny\nproperty float nz\n");
    }
    if write_uvs {
        header.push_str("property float s\nproperty float t\n");
    }
    if write_colors {
        header.push_str(
            "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n",
        );
    }
    header.push_str(&format!("element face {}\n", indices.len() / 3));
    header.push_str("property list uchar uint vertex_indices\n");
    header.push_str("end_header\n");
    writer
        .write_all(header.as_bytes())
        .map_err(|error| Error::write(&temp_path, error))?;

    for (index, position) in positions.iter().enumerate() {
        for value in position {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|error| Error::write(&temp_path, error))?;
        }
        if write_normals {
            for value in &normals[index] {
                writer
                    .write_all(&value.to_le_bytes())
                    .map_err(|error| Error::write(&temp_path, error))?;
            }
        }
        if write_uvs {
            for value in &uvs[index] {
                writer
                    .write_all(&value.to_le_bytes())
                    .map_err(|error| Error::write(&temp_path, error))?;
            }
        }
        if write_colors {
            for value in &colors[index] {
                writer
                    .write_all(&[to_u8(*value)])
                    .map_err(|error| Error::write(&temp_path, error))?;
            }
        }
    }
    for triangle in indices.chunks_exact(3) {
        writer
            .write_all(&[3u8])
            .map_err(|error| Error::write(&temp_path, error))?;
        for index in triangle {
            writer
                .write_all(&index.to_le_bytes())
                .map_err(|error| Error::write(&temp_path, error))?;
        }
    }
    writer
        .flush()
        .map_err(|error| Error::write(&temp_path, error))?;

    if (!write_normals
        && primitives
            .iter()
            .any(|primitive| primitive.normals.is_some()))
        || (!write_uvs && primitives.iter().any(|primitive| primitive.uv0().is_some()))
        || (!write_colors
            && primitives
                .iter()
                .any(|primitive| primitive.colors.is_some()))
    {
        warnings.add_for(
            Format::Ply,
            WarningKind::DroppedFeature,
            "ply-mixed-attributes",
            "PLY output needs uniform vertex attributes; dropped attributes that were missing on some primitives",
            1,
        );
    }
    Ok(())
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colors_clamp_to_byte_range() {
        assert_eq!(to_u8(-1.0), 0);
        assert_eq!(to_u8(0.5), 128);
        assert_eq!(to_u8(2.0), 255);
    }
}
