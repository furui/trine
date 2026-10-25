//! 3MF reader and writer using `zip` + `quick-xml` directly.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::io::{Read as _, Write as _};
use std::path::Path;

use glam::{Mat4, Vec4};
use quick_xml::Reader;
use quick_xml::events::Event;
use zip::write::SimpleFileOptions;

use crate::error::{Error, Result};
use crate::io::{Format, Loaded, WriteOptions};
use crate::output::staging::Staging;
use crate::output::textures::TexturePlan;
use crate::report::{WarningKind, WarningSink};
use crate::scene::{Material, Mesh, NodeId, Primitive, Scene, Unit, UnitResolution, UpAxis};

const MODEL_PATH: &str = "3D/3dmodel.model";
const MATERIAL_ID: usize = 1;

#[derive(Default)]
struct RawObject {
    id: usize,
    name: Option<String>,
    pid: Option<usize>,
    pindex: Option<usize>,
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    components: Vec<(usize, Option<[f64; 12]>)>,
}

#[derive(Default)]
struct RawModel {
    unit: Unit,
    unit_present: bool,
    objects: HashMap<usize, RawObject>,
    order: Vec<usize>,
    materials: HashMap<usize, Vec<(String, [f32; 4])>>,
    build: Vec<(usize, Option<[f64; 12]>)>,
    unsupported: usize,
}

pub fn read(path: &Path) -> Result<Loaded> {
    let file = fs::File::open(path).map_err(|error| Error::read(path, error))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|error| Error::path(path, format!("invalid 3MF zip: {error}")))?;
    let mut model_name = None;
    for index in 0..archive.len() {
        let name = archive
            .name_for_index(index)
            .map(str::to_string)
            .unwrap_or_default();
        if name.to_ascii_lowercase().ends_with(".model") {
            if name.ends_with("3dmodel.model") {
                model_name = Some(name);
                break;
            }
            if model_name.is_none() {
                model_name = Some(name);
            }
        }
    }
    let Some(model_name) = model_name else {
        return Err(Error::path(path, "3MF archive contains no .model part"));
    };
    let mut entry = archive
        .by_name(&model_name)
        .map_err(|error| Error::path(path, format!("cannot read 3MF model: {error}")))?;
    let mut xml = String::new();
    entry
        .read_to_string(&mut xml)
        .map_err(|error| Error::path(path, format!("cannot read 3MF model: {error}")))?;
    let raw = parse_model(&xml).map_err(|message| Error::path(path, message))?;
    if raw.objects.is_empty() {
        return Err(Error::path(path, "3MF model contains no objects"));
    }

    let mut scene = Scene::new(Format::ThreeMf);
    scene.unit = if raw.unit_present {
        UnitResolution::Known(raw.unit)
    } else {
        UnitResolution::Unknown
    };
    scene.up_axis = UpAxis::Z;
    let mut warnings = WarningSink::default();
    let mut material_cache: HashMap<(usize, usize), usize> = HashMap::new();
    let mut visited = Vec::new();
    for (object_id, transform) in &raw.build {
        instantiate(
            *object_id,
            transform.map(transform_from_3mf).unwrap_or(Mat4::IDENTITY),
            &raw,
            &mut material_cache,
            &mut scene,
            None,
            &mut visited,
            &mut warnings,
        );
    }
    if raw.unsupported > 0 {
        warnings.add_for(
            Format::ThreeMf,
            WarningKind::UnsupportedEntity,
            "3mf-extension",
            format!(
                "ignored {} unsupported 3MF material/extension element(s)",
                raw.unsupported
            ),
            raw.unsupported,
        );
    }
    if scene.meshes.is_empty() {
        return Err(Error::path(path, "3MF contains no mesh geometry"));
    }
    Ok(Loaded { scene, warnings })
}

#[allow(clippy::too_many_arguments)]
fn instantiate(
    object_id: usize,
    transform: Mat4,
    raw: &RawModel,
    material_cache: &mut HashMap<(usize, usize), usize>,
    scene: &mut Scene,
    parent: Option<NodeId>,
    visited: &mut Vec<usize>,
    warnings: &mut WarningSink,
) -> Option<NodeId> {
    if visited.contains(&object_id) {
        warnings.add(
            WarningKind::UnsupportedEntity,
            "3mf-component-cycle",
            format!("3MF object {object_id} forms a component cycle and was skipped"),
        );
        return None;
    }
    let object = raw.objects.get(&object_id)?;
    let name = object
        .name
        .clone()
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| format!("object_{object_id}"));
    let node = match parent {
        Some(parent) => scene.add_child(parent, name.clone(), transform),
        None => scene.add_node(name.clone(), transform),
    };

    if !object.positions.is_empty() && !object.indices.is_empty() {
        let vertex_count = object.positions.len();
        let mut indices = Vec::with_capacity(object.indices.len());
        for index in &object.indices {
            if *index as usize >= vertex_count {
                warnings.add(
                    WarningKind::UnsupportedEntity,
                    "3mf-index",
                    format!("3MF object {object_id} references an out-of-range vertex"),
                );
                continue;
            }
            indices.push(*index);
        }
        if !indices.is_empty() {
            let mut primitive = Primitive::new(object.positions.clone(), indices);
            if let (Some(pid), Some(pindex)) = (object.pid, object.pindex) {
                if let Some(bases) = raw.materials.get(&pid) {
                    if let Some((base_name, color)) = bases.get(pindex) {
                        let material_id =
                            *material_cache.entry((pid, pindex)).or_insert_with(|| {
                                let mut material = Material::new(if base_name.is_empty() {
                                    "Material".to_string()
                                } else {
                                    base_name.clone()
                                });
                                material.base_color = *color;
                                scene.add_material(material)
                            });
                        primitive.material = Some(material_id);
                    }
                }
            }
            let mut mesh = Mesh::new(name);
            mesh.primitives.push(primitive);
            let mesh_id = scene.add_mesh(mesh);
            scene.nodes[node].mesh = Some(mesh_id);
        }
    }

    visited.push(object_id);
    for (child_id, child_transform) in &object.components {
        if !raw.objects.contains_key(child_id) {
            warnings.add(
                WarningKind::UnsupportedEntity,
                "3mf-component",
                format!("3MF object {object_id} references missing component {child_id}"),
            );
            continue;
        }
        instantiate(
            *child_id,
            child_transform
                .map(transform_from_3mf)
                .unwrap_or(Mat4::IDENTITY),
            raw,
            material_cache,
            scene,
            Some(node),
            visited,
            warnings,
        );
    }
    visited.pop();
    Some(node)
}

fn parse_model(xml: &str) -> std::result::Result<RawModel, String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut model = RawModel {
        unit: Unit::Millimeter,
        ..RawModel::default()
    };
    let mut current: Option<RawObject> = None;
    let mut current_material_id: Option<usize> = None;
    loop {
        match reader.read_event() {
            Ok(Event::Start(event)) | Ok(Event::Empty(event)) => {
                let name = local_name(&event);
                match name.as_str() {
                    "texture2d" | "multiproperties" | "composite" | "beamset" => {
                        model.unsupported += 1;
                    }
                    "model" => {
                        if let Some(unit) = attribute(&event, "unit") {
                            model.unit = parse_unit(&unit);
                            model.unit_present = true;
                        }
                    }
                    "object" => {
                        current = Some(RawObject {
                            id: attribute(&event, "id")
                                .and_then(|id| id.parse().ok())
                                .ok_or("3MF object has no id")?,
                            name: attribute(&event, "name"),
                            pid: attribute(&event, "pid").and_then(|id| id.parse().ok()),
                            pindex: attribute(&event, "pindex").and_then(|id| id.parse().ok()),
                            ..RawObject::default()
                        });
                    }
                    "vertex" => {
                        if let Some(object) = current.as_mut() {
                            let x = attribute(&event, "x")
                                .and_then(|v| v.parse::<f64>().ok())
                                .unwrap_or(0.0);
                            let y = attribute(&event, "y")
                                .and_then(|v| v.parse::<f64>().ok())
                                .unwrap_or(0.0);
                            let z = attribute(&event, "z")
                                .and_then(|v| v.parse::<f64>().ok())
                                .unwrap_or(0.0);
                            object.positions.push([x as f32, y as f32, z as f32]);
                        }
                    }
                    "triangle" => {
                        if let Some(object) = current.as_mut() {
                            for name in ["v1", "v2", "v3"] {
                                if let Some(value) =
                                    attribute(&event, name).and_then(|v| v.parse::<u32>().ok())
                                {
                                    object.indices.push(value);
                                }
                            }
                        }
                    }
                    "component" => {
                        if let Some(object) = current.as_mut() {
                            if let Some(child) =
                                attribute(&event, "objectid").and_then(|id| id.parse().ok())
                            {
                                let transform = attribute(&event, "transform")
                                    .and_then(|text| parse_transform(&text));
                                object.components.push((child, transform));
                            }
                        }
                    }
                    "basematerials" | "colorgroup" => {
                        current_material_id =
                            attribute(&event, "id").and_then(|id| id.parse().ok());
                        if let Some(id) = current_material_id {
                            model.materials.entry(id).or_default();
                        }
                    }
                    "base" | "color" => {
                        if let Some(id) = current_material_id {
                            let name = attribute(&event, "name").unwrap_or_default();
                            let color = attribute(&event, "displaycolor")
                                .or_else(|| attribute(&event, "color"))
                                .and_then(|text| parse_color(&text))
                                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                            model.materials.entry(id).or_default().push((name, color));
                        }
                    }
                    "item" => {
                        if let Some(object_id) =
                            attribute(&event, "objectid").and_then(|id| id.parse().ok())
                        {
                            let transform = attribute(&event, "transform")
                                .and_then(|text| parse_transform(&text));
                            model.build.push((object_id, transform));
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(event)) => {
                let name = local_name_end(&event);
                if name == "object" {
                    if let Some(object) = current.take() {
                        model.order.push(object.id);
                        model.objects.insert(object.id, object);
                    }
                } else if name == "basematerials" || name == "colorgroup" {
                    current_material_id = None;
                }
            }
            Ok(Event::Eof) => break,
            Ok(_) => {}
            Err(error) => return Err(format!("invalid 3MF XML: {error}")),
        }
    }
    Ok(model)
}

fn local_name(event: &quick_xml::events::BytesStart<'_>) -> String {
    String::from_utf8_lossy(event.local_name().as_ref()).to_ascii_lowercase()
}

fn local_name_end(event: &quick_xml::events::BytesEnd<'_>) -> String {
    String::from_utf8_lossy(event.local_name().as_ref()).to_ascii_lowercase()
}

fn attribute(event: &quick_xml::events::BytesStart<'_>, name: &str) -> Option<String> {
    for attribute in event.attributes().flatten() {
        if attribute.key.local_name().as_ref() == name.as_bytes() {
            if let Ok(value) = attribute.unescape_value() {
                return Some(value.into_owned());
            }
            return Some(String::from_utf8_lossy(&attribute.value).to_string());
        }
    }
    None
}

fn parse_unit(text: &str) -> Unit {
    match text.to_ascii_lowercase().as_str() {
        "micron" => Unit::Micrometer,
        "centimeter" => Unit::Centimeter,
        "inch" => Unit::Inch,
        "foot" => Unit::Foot,
        "meter" => Unit::Meter,
        _ => Unit::Millimeter,
    }
}

fn parse_color(text: &str) -> Option<[f32; 4]> {
    let hex = text.trim().trim_start_matches('#');
    if hex.len() != 6 && hex.len() != 8 {
        return None;
    }
    let byte = |index: usize| u8::from_str_radix(&hex[index..index + 2], 16).ok();
    let r = byte(0)?;
    let g = byte(2)?;
    let b = byte(4)?;
    let a = if hex.len() == 8 { byte(6)? } else { 255 };
    Some([
        r as f32 / 255.0,
        g as f32 / 255.0,
        b as f32 / 255.0,
        a as f32 / 255.0,
    ])
}

fn format_color(color: [f32; 4]) -> String {
    let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!(
        "#{:02X}{:02X}{:02X}{:02X}",
        channel(color[0]),
        channel(color[1]),
        channel(color[2]),
        channel(color[3])
    )
}

fn parse_transform(text: &str) -> Option<[f64; 12]> {
    let values: Vec<f64> = text
        .split_whitespace()
        .filter_map(|value| value.parse().ok())
        .collect();
    values.try_into().ok()
}

fn transform_from_3mf(values: [f64; 12]) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(values[0] as f32, values[1] as f32, values[2] as f32, 0.0),
        Vec4::new(values[3] as f32, values[4] as f32, values[5] as f32, 0.0),
        Vec4::new(values[6] as f32, values[7] as f32, values[8] as f32, 0.0),
        Vec4::new(values[9] as f32, values[10] as f32, values[11] as f32, 1.0),
    )
}

fn transform_to_3mf(matrix: Mat4) -> String {
    [
        matrix.x_axis.x,
        matrix.x_axis.y,
        matrix.x_axis.z,
        matrix.y_axis.x,
        matrix.y_axis.y,
        matrix.y_axis.z,
        matrix.z_axis.x,
        matrix.z_axis.y,
        matrix.z_axis.z,
        matrix.w_axis.x,
        matrix.w_axis.y,
        matrix.w_axis.z,
    ]
    .iter()
    .map(|value| number(*value))
    .collect::<Vec<_>>()
    .join(" ")
}

fn postorder(scene: &Scene) -> Vec<NodeId> {
    fn visit(id: NodeId, scene: &Scene, out: &mut Vec<NodeId>, seen: &mut [bool]) {
        if seen[id] {
            return;
        }
        seen[id] = true;
        for child in &scene.nodes[id].children {
            visit(*child, scene, out, seen);
        }
        out.push(id);
    }
    let mut out = Vec::new();
    let mut seen = vec![false; scene.nodes.len()];
    for root in &scene.roots {
        visit(*root, scene, &mut out, &mut seen);
    }
    for id in 0..scene.nodes.len() {
        if !seen[id] {
            visit(id, scene, &mut out, &mut seen);
        }
    }
    out
}

pub fn write(
    options: &WriteOptions<'_>,
    staging: &mut Staging,
    _textures: &mut TexturePlan,
    warnings: &mut WarningSink,
) -> Result<()> {
    let scene = options.scene;
    if options.up_axis != UpAxis::Z {
        warnings.add_for(
            Format::ThreeMf,
            WarningKind::Note,
            "3mf-up-axis",
            "3MF assumes Z-up; wrote non-Z-up coordinates",
            1,
        );
    }
    let mut next_object = MATERIAL_ID + 1;
    let mut object_ids: HashMap<NodeId, usize> = HashMap::new();
    let mut objects_xml = String::new();
    let mut split_objects = 0usize;

    for node_id in postorder(scene) {
        let node = &scene.nodes[node_id];
        let mut own_objects: Vec<usize> = Vec::new();
        if let Some(mesh_id) = node.mesh {
            let mesh = &scene.meshes[mesh_id];
            for (primitive_index, primitive) in mesh.primitives.iter().enumerate() {
                if primitive.indices.is_empty() {
                    continue;
                }
                let id = next_object;
                next_object += 1;
                let mut mesh_xml = String::new();
                for position in &primitive.positions {
                    writeln!(
                        mesh_xml,
                        "          <vertex x=\"{}\" y=\"{}\" z=\"{}\"/>",
                        number(position[0]),
                        number(position[1]),
                        number(position[2])
                    )
                    .unwrap();
                }
                let mut triangles_xml = String::new();
                for triangle in primitive.indices.chunks_exact(3) {
                    writeln!(
                        triangles_xml,
                        "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>",
                        triangle[0], triangle[1], triangle[2]
                    )
                    .unwrap();
                }
                let pid = match primitive.material {
                    Some(material) if material < scene.materials.len() => {
                        format!(" pid=\"{MATERIAL_ID}\" pindex=\"{material}\"")
                    }
                    _ => String::new(),
                };
                let name = sanitize_name(&node.name, node_id);
                writeln!(
                    objects_xml,
                    "    <object id=\"{id}\" name=\"{name}_{primitive_index}\"{pid}>"
                )
                .unwrap();
                writeln!(objects_xml, "      <mesh>").unwrap();
                writeln!(objects_xml, "        <vertices>").unwrap();
                objects_xml.push_str(&mesh_xml);
                writeln!(objects_xml, "        </vertices>").unwrap();
                writeln!(objects_xml, "        <triangles>").unwrap();
                objects_xml.push_str(&triangles_xml);
                writeln!(objects_xml, "        </triangles>").unwrap();
                writeln!(objects_xml, "      </mesh>").unwrap();
                writeln!(objects_xml, "    </object>").unwrap();
                own_objects.push(id);
            }
        }

        let mut components = String::new();
        let mut component_count = 0usize;
        for id in &own_objects {
            writeln!(components, "        <component objectid=\"{id}\"/>").unwrap();
            component_count += 1;
        }
        for child in &node.children {
            let Some(child_object) = object_ids.get(child).copied() else {
                continue;
            };
            writeln!(
                components,
                "        <component objectid=\"{child_object}\" transform=\"{}\"/>",
                transform_to_3mf(scene.nodes[*child].local_transform)
            )
            .unwrap();
            component_count += 1;
        }
        if component_count == 0 {
            continue;
        }
        if own_objects.len() == 1 && node.children.is_empty() {
            object_ids.insert(node_id, own_objects[0]);
            continue;
        }
        if own_objects.len() > 1 {
            split_objects += own_objects.len() - 1;
        }
        let id = next_object;
        next_object += 1;
        let name = sanitize_name(&node.name, node_id);
        writeln!(objects_xml, "    <object id=\"{id}\" name=\"{name}\">").unwrap();
        writeln!(objects_xml, "      <components>").unwrap();
        objects_xml.push_str(&components);
        writeln!(objects_xml, "      </components>").unwrap();
        writeln!(objects_xml, "    </object>").unwrap();
        object_ids.insert(node_id, id);
    }

    let mut build = String::new();
    for root in &scene.roots {
        let Some(object_id) = object_ids.get(root).copied() else {
            continue;
        };
        let transform = scene.nodes[*root].local_transform;
        if transform == Mat4::IDENTITY {
            writeln!(build, "    <item objectid=\"{object_id}\"/>").unwrap();
        } else {
            writeln!(
                build,
                "    <item objectid=\"{object_id}\" transform=\"{}\"/>",
                transform_to_3mf(transform)
            )
            .unwrap();
        }
    }
    if build.is_empty() {
        return Err(Error::conversion("no geometry to write to 3MF"));
    }

    let mut materials = String::new();
    if !scene.materials.is_empty() {
        writeln!(materials, "    <m:basematerials id=\"{MATERIAL_ID}\">").unwrap();
        for (index, material) in scene.materials.iter().enumerate() {
            writeln!(
                materials,
                "      <m:base name=\"{}\" displaycolor=\"{}\"/>",
                escape(&sanitize_name(&material.name, index)),
                format_color(material.base_color)
            )
            .unwrap();
        }
        writeln!(materials, "    </m:basematerials>").unwrap();
    }

    let unit = match options.unit {
        Unit::Micrometer => "micron",
        Unit::Millimeter => "millimeter",
        Unit::Centimeter => "centimeter",
        Unit::Meter => "meter",
        Unit::Inch => "inch",
        Unit::Foot => "foot",
    };
    let model = format!(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n\
<model xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\" xmlns:m=\"http://schemas.microsoft.com/3dmanufacturing/material/2015/02\" unit=\"{unit}\">\n\
  <resources>\n{objects_xml}{materials}  </resources>\n\
  <build>\n{build}  </build>\n\
</model>\n"
    );

    let temp_path = staging.stage(options.output_path)?;
    let file = fs::File::create(&temp_path).map_err(|error| Error::write(&temp_path, error))?;
    let mut zip = zip::ZipWriter::new(file);
    let options_zip = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .unix_permissions(0o644);
    let content_types = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\n\
  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\n\
  <Default Extension=\"model\" ContentType=\"application/vnd.ms-package.3dmanufacturing-3dmodel+xml\"/>\n\
</Types>\n";
    let rels = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n\
  <Relationship Target=\"/3D/3dmodel.model\" Id=\"rel0\" Type=\"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel\"/>\n\
</Relationships>\n";
    for (name, bytes) in [
        ("[Content_Types].xml", content_types.as_bytes()),
        ("_rels/.rels", rels.as_bytes()),
        (MODEL_PATH, model.as_bytes()),
    ] {
        zip.start_file(name, options_zip)
            .map_err(|error| Error::path(&temp_path, format!("cannot write 3MF: {error}")))?;
        zip.write_all(bytes)
            .map_err(|error| Error::write(&temp_path, error))?;
    }
    zip.finish()
        .map_err(|error| Error::path(&temp_path, format!("cannot finish 3MF: {error}")))?;

    let approximated = scene
        .materials
        .iter()
        .filter(|material| {
            material.metallic > 0.001
                || (material.roughness - 1.0).abs() > 0.001
                || material.emissive.iter().any(|value| *value > 0.001)
        })
        .count();
    if approximated > 0 {
        warnings.add_for(
            Format::ThreeMf,
            WarningKind::Approximation,
            "3mf-material",
            format!(
                "3MF core stores only base colors; approximated metallic/roughness/emissive on {approximated} material(s)"
            ),
            approximated,
        );
    }
    if split_objects > 0 {
        warnings.add_for(
            Format::ThreeMf,
            WarningKind::Note,
            "3mf-object-split",
            format!("3MF output used {split_objects} extra component object(s) to keep per-material colors"),
            split_objects,
        );
    }
    Ok(())
}

fn sanitize_name(name: &str, fallback_index: usize) -> String {
    let cleaned: String = name
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('_').to_string();
    if trimmed.is_empty() {
        format!("object_{fallback_index}")
    } else {
        trimmed
    }
}

fn number(value: f32) -> String {
    if !value.is_finite() || value == 0.0 {
        "0".to_string()
    } else {
        format!("{value}")
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colour_roundtrip() {
        let parsed = parse_color("#FF800080").unwrap();
        assert!((parsed[0] - 1.0).abs() < 1e-6);
        assert!((parsed[1] - 128.0 / 255.0).abs() < 1e-6);
        assert!((parsed[3] - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(format_color(parsed), "#FF800080");
    }

    #[test]
    fn names_are_sanitized() {
        assert_eq!(sanitize_name("bad/name\"", 3), "bad_name");
    }
}
