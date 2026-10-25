//! glTF 2.0 and binary glTF (GLB) reader and writer.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use glam::Mat4;
use gltf::json;
use gltf::json::accessor::{ComponentType, GenericComponentType, Type as AccessorType};
use gltf::json::buffer::Target;
use gltf::json::mesh::{Mode, Semantic};
use gltf::json::validation::{Checked, USize64};
use gltf::json::{Index, Value};

use crate::error::{Error, Result};
use crate::io::{Format, Loaded, WriteOptions};
use crate::output::staging::Staging;
use crate::output::textures::TexturePlan;
use crate::report::{WarningKind, WarningSink};
use crate::scene::{
    AlphaMode, Image, ImageId, Material, Mesh, Node, Primitive, Scene, TextureRef, Unit,
    UnitResolution, UpAxis,
};

const SUPPORTED_EXTENSIONS: &[&str] = &[];

pub fn read(path: &Path, _format: Format) -> Result<Loaded> {
    let gltf = gltf::Gltf::open(path)
        .map_err(|error| Error::path(path, format!("invalid glTF: {error}")))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let buffers = gltf::import_buffers(&gltf.document, Some(base_dir), gltf.blob.clone())
        .map_err(|error| Error::path(path, format!("cannot load glTF buffers: {error}")))?;
    let mut warnings = WarningSink::default();
    let mut scene = Scene::new(Format::Gltf);
    scene.unit = UnitResolution::Known(Unit::Meter);
    scene.up_axis = UpAxis::Y;

    // Images: keep encoded bytes exactly as found.
    let mut image_ids: Vec<Option<ImageId>> = Vec::new();
    for (index, image) in gltf.document.images().enumerate() {
        let mut bytes: Option<Vec<u8>> = None;
        let mut mime: Option<String>;
        let mut source_path: Option<PathBuf> = None;
        match image.source() {
            gltf::image::Source::View { view, mime_type } => {
                mime = Some(mime_type.to_string());
                let data = &buffers[view.buffer().index()].0;
                let start = view.offset();
                let end = start.saturating_add(view.length());
                if end <= data.len() {
                    bytes = Some(data[start..end].to_vec());
                } else {
                    warnings.add(
                        WarningKind::UnsupportedEntity,
                        "gltf-image-view",
                        format!("glTF image {index} has an out-of-range buffer view"),
                    );
                }
            }
            gltf::image::Source::Uri { uri, mime_type } => {
                mime = mime_type.map(str::to_string);
                if let Some(rest) = uri.strip_prefix("data:") {
                    if let Some((meta, payload)) = rest.split_once(";base64,") {
                        if mime.is_none() {
                            mime = Some(meta.to_string());
                        }
                        match base64::engine::general_purpose::STANDARD.decode(payload) {
                            Ok(decoded) => bytes = Some(decoded),
                            Err(_) => warnings.add(
                                WarningKind::UnsupportedEntity,
                                "gltf-image-data",
                                format!("glTF image {index} has invalid base64 data"),
                            ),
                        }
                    } else {
                        warnings.add(
                            WarningKind::UnsupportedEntity,
                            "gltf-image-data",
                            format!("glTF image {index} uses a non-base64 data URI"),
                        );
                    }
                } else {
                    let image_path = base_dir.join(uri);
                    match fs::read(&image_path) {
                        Ok(read) => {
                            bytes = Some(read);
                            source_path = Some(image_path.clone());
                        }
                        Err(_) => warnings.add(
                            WarningKind::MissingAsset,
                            "texture-missing",
                            format!("texture '{uri}' referenced by image {index} was not found"),
                        ),
                    }
                }
            }
        }
        let Some(bytes) = bytes else {
            image_ids.push(None);
            continue;
        };
        let name = image
            .name()
            .map(str::to_string)
            .or_else(|| {
                source_path
                    .as_ref()
                    .and_then(|path| path.file_name())
                    .and_then(|name| name.to_str())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| format!("image_{index}"));
        let id = scene.add_image(Image {
            name,
            bytes,
            mime,
            source_path,
        });
        image_ids.push(Some(id));
    }

    // Materials.
    let mut material_ids = Vec::new();
    for material in gltf.document.materials() {
        let pbr = material.pbr_metallic_roughness();
        let mut converted = Material::new(material.name().unwrap_or("Material"));
        let base = pbr.base_color_factor();
        converted.base_color = base;
        converted.metallic = pbr.metallic_factor();
        converted.roughness = pbr.roughness_factor();
        converted.emissive = material.emissive_factor();
        converted.double_sided = material.double_sided();
        converted.alpha_mode = match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        };
        converted.alpha_cutoff = material.alpha_cutoff().unwrap_or(0.5);

        let slot = |info: Option<gltf::texture::Info<'_>>,
                    warnings: &mut WarningSink,
                    label: &str|
         -> Option<TextureRef> {
            let info = info?;
            if info.tex_coord() != 0 {
                warnings.add_for(
                    Format::Gltf,
                    WarningKind::DroppedFeature,
                    "uv-extra",
                    format!(
                        "texture slot {label} uses TEXCOORD_{}; only TEXCOORD_0 is kept",
                        info.tex_coord()
                    ),
                    1,
                );
                return None;
            }
            let image_index = info.texture().source().index();
            image_ids
                .get(image_index)
                .copied()
                .flatten()
                .map(|image| TextureRef { image })
        };
        converted.textures.base_color = slot(pbr.base_color_texture(), &mut warnings, "base color");
        converted.textures.metallic_roughness = slot(
            pbr.metallic_roughness_texture(),
            &mut warnings,
            "metallic-roughness",
        );
        converted.textures.normal = material.normal_texture().and_then(|texture| {
            if texture.tex_coord() != 0 {
                warnings.add_for(
                    Format::Gltf,
                    WarningKind::DroppedFeature,
                    "uv-extra",
                    format!(
                        "normal texture uses TEXCOORD_{}; only TEXCOORD_0 is kept",
                        texture.tex_coord()
                    ),
                    1,
                );
                return None;
            }
            let image_index = texture.texture().source().index();
            image_ids
                .get(image_index)
                .copied()
                .flatten()
                .map(|image| TextureRef { image })
        });
        converted.textures.occlusion = material.occlusion_texture().and_then(|texture| {
            if texture.tex_coord() != 0 {
                warnings.add_for(
                    Format::Gltf,
                    WarningKind::DroppedFeature,
                    "uv-extra",
                    format!(
                        "occlusion texture uses TEXCOORD_{}; only TEXCOORD_0 is kept",
                        texture.tex_coord()
                    ),
                    1,
                );
                return None;
            }
            let image_index = texture.texture().source().index();
            image_ids
                .get(image_index)
                .copied()
                .flatten()
                .map(|image| TextureRef { image })
        });
        converted.textures.emissive = slot(material.emissive_texture(), &mut warnings, "emissive");
        material_ids.push(scene.add_material(converted));
    }

    // Meshes and primitives.
    let mut skipped_primitives = 0usize;
    for (mesh_index, mesh) in gltf.document.meshes().enumerate() {
        let mut mesh_out = Mesh::new(mesh.name().unwrap_or("mesh"));
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));
            let Some(positions) = reader.read_positions() else {
                skipped_primitives += 1;
                continue;
            };
            let positions: Vec<[f32; 3]> = positions.collect();
            if positions.is_empty() {
                skipped_primitives += 1;
                continue;
            }
            let vertex_count = positions.len();
            let raw_indices: Vec<u32> = match reader.read_indices() {
                Some(indices) => indices.into_u32().collect(),
                None => (0..vertex_count as u32).collect(),
            };
            let indices = match primitive.mode() {
                Mode::Triangles => raw_indices
                    .chunks_exact(3)
                    .flatten()
                    .copied()
                    .collect::<Vec<u32>>(),
                Mode::TriangleStrip => {
                    let mut out = Vec::new();
                    for i in 0..raw_indices.len().saturating_sub(2) {
                        if i % 2 == 0 {
                            out.extend([raw_indices[i], raw_indices[i + 1], raw_indices[i + 2]]);
                        } else {
                            out.extend([raw_indices[i + 1], raw_indices[i], raw_indices[i + 2]]);
                        }
                    }
                    out
                }
                Mode::TriangleFan => {
                    let mut out = Vec::new();
                    for i in 1..raw_indices.len().saturating_sub(1) {
                        out.extend([raw_indices[0], raw_indices[i], raw_indices[i + 1]]);
                    }
                    out
                }
                _ => {
                    skipped_primitives += 1;
                    continue;
                }
            };
            if indices.is_empty() {
                skipped_primitives += 1;
                continue;
            }
            if indices.iter().any(|index| *index as usize >= vertex_count) {
                return Err(Error::path(
                    path,
                    format!("glTF mesh {mesh_index} contains an out-of-range index"),
                ));
            }
            let mut out = Primitive::new(positions, indices);
            if let Some(normals) = reader.read_normals() {
                out.normals = Some(normals.collect());
            }
            if let Some(tangents) = reader.read_tangents() {
                out.tangents = Some(tangents.collect());
            }
            if let Some(uvs) = reader.read_tex_coords(0) {
                out.uvs.push(uvs.into_f32().collect());
            }
            if let Some(colors) = reader.read_colors(0) {
                out.colors = Some(read_colors_to_rgba(colors));
            }
            let mut extra_attributes = 0usize;
            for (semantic, _) in primitive.attributes() {
                match semantic {
                    Semantic::TexCoords(set) if set > 0 => extra_attributes += 1,
                    Semantic::Colors(set) if set > 0 => extra_attributes += 1,
                    _ => {}
                }
            }
            if extra_attributes > 0 {
                warnings.add_for(
                    Format::Gltf,
                    WarningKind::DroppedFeature,
                    "uv-extra",
                    format!("kept only TEXCOORD_0/COLOR_0; dropped {extra_attributes} additional attribute set(s)"),
                    extra_attributes,
                );
            }
            out.material = primitive
                .material()
                .index()
                .and_then(|index| material_ids.get(index).copied());
            mesh_out.primitives.push(out);
        }
        scene.add_mesh(mesh_out);
    }
    if skipped_primitives > 0 {
        warnings.add_for(
            Format::Gltf,
            WarningKind::UnsupportedEntity,
            "gltf-primitive-mode",
            format!("skipped {skipped_primitives} glTF primitive(s) that are not triangle meshes"),
            skipped_primitives,
        );
    }

    // Nodes and hierarchy.
    let node_count = gltf.document.nodes().len();
    let mut children_map: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    let mut parent_map: Vec<Option<usize>> = vec![None; node_count];
    for node in gltf.document.nodes() {
        for child in node.children() {
            let child_index = child.index();
            if parent_map[child_index].is_some() {
                warnings.add(
                    WarningKind::UnsupportedEntity,
                    "gltf-multi-parent",
                    format!("glTF node {child_index} has multiple parents; kept the first"),
                );
            } else {
                parent_map[child_index] = Some(node.index());
            }
            children_map[node.index()].push(child_index);
        }
    }
    for node in gltf.document.nodes() {
        let local = Mat4::from_cols_array_2d(&node.transform().matrix());
        scene.nodes.push(Node {
            name: node.name().unwrap_or("node").to_string(),
            parent: parent_map[node.index()],
            children: children_map[node.index()].clone(),
            local_transform: local,
            mesh: node.mesh().map(|mesh| mesh.index()),
        });
    }
    scene.roots = match gltf.document.default_scene() {
        Some(default_scene) => default_scene.nodes().map(|node| node.index()).collect(),
        None => (0..node_count)
            .filter(|index| parent_map[*index].is_none())
            .collect(),
    };

    let animations = gltf.document.animations().len();
    if animations > 0 {
        warnings.add_for(
            Format::Gltf,
            WarningKind::DroppedFeature,
            "animation",
            format!("glTF input contains animation; dropped {animations} animation clip(s)"),
            animations,
        );
    }
    let skins = gltf.document.skins().len();
    if skins > 0 {
        warnings.add_for(
            Format::Gltf,
            WarningKind::DroppedFeature,
            "skinning",
            format!("glTF input contains skinning; dropped {skins} skin(s)"),
            skins,
        );
    }
    let cameras = gltf.document.cameras().len();
    if cameras > 0 {
        warnings.add_for(
            Format::Gltf,
            WarningKind::DroppedFeature,
            "camera",
            format!("glTF input contains cameras; dropped {cameras} camera(s)"),
            cameras,
        );
    }
    let morphs: usize = gltf
        .document
        .meshes()
        .flat_map(|mesh| {
            mesh.primitives()
                .map(|primitive| primitive.morph_targets().len())
                .collect::<Vec<_>>()
        })
        .sum();
    if morphs > 0 {
        warnings.add_for(
            Format::Gltf,
            WarningKind::DroppedFeature,
            "morph",
            format!("glTF input contains morph targets; dropped {morphs} target set(s)"),
            morphs,
        );
    }
    let mut unknown_extensions = 0usize;
    for extension in gltf.document.extensions_used() {
        if !SUPPORTED_EXTENSIONS.contains(&extension) {
            unknown_extensions += 1;
        }
    }
    if unknown_extensions > 0 {
        warnings.add_for(
            Format::Gltf,
            WarningKind::UnsupportedEntity,
            "gltf-extension",
            format!("ignored {unknown_extensions} glTF extension(s)"),
            unknown_extensions,
        );
    }

    if scene.referenced_triangle_count() == 0 {
        return Err(Error::path(path, "glTF contains no triangle geometry"));
    }
    Ok(Loaded { scene, warnings })
}

fn read_colors_to_rgba<'a>(colors: gltf::mesh::util::ReadColors<'a>) -> Vec<[f32; 4]> {
    use gltf::mesh::util::ReadColors;
    match colors {
        ReadColors::RgbU8(iter) => iter
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    1.0,
                ]
            })
            .collect(),
        ReadColors::RgbU16(iter) => iter
            .map(|c| {
                [
                    c[0] as f32 / 65535.0,
                    c[1] as f32 / 65535.0,
                    c[2] as f32 / 65535.0,
                    1.0,
                ]
            })
            .collect(),
        ReadColors::RgbF32(iter) => iter.map(|c| [c[0], c[1], c[2], 1.0]).collect(),
        ReadColors::RgbaU8(iter) => iter
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    c[3] as f32 / 255.0,
                ]
            })
            .collect(),
        ReadColors::RgbaU16(iter) => iter
            .map(|c| {
                [
                    c[0] as f32 / 65535.0,
                    c[1] as f32 / 65535.0,
                    c[2] as f32 / 65535.0,
                    c[3] as f32 / 65535.0,
                ]
            })
            .collect(),
        ReadColors::RgbaF32(iter) => iter.map(|c| [c[0], c[1], c[2], c[3]]).collect(),
    }
}

struct BinaryBuilder {
    blob: Vec<u8>,
    views: Vec<json::buffer::View>,
}

impl BinaryBuilder {
    fn new() -> Self {
        Self {
            blob: Vec::new(),
            views: Vec::new(),
        }
    }

    fn align(&mut self) {
        while !self.blob.len().is_multiple_of(4) {
            self.blob.push(0);
        }
    }

    fn push_view(&mut self, bytes: &[u8], target: Option<Target>) -> usize {
        self.align();
        let offset = self.blob.len();
        self.blob.extend_from_slice(bytes);
        self.align();
        self.views.push(json::buffer::View {
            buffer: Index::new(0),
            byte_length: USize64::from(bytes.len()),
            byte_offset: Some(USize64::from(offset)),
            byte_stride: None,
            name: None,
            target: target.map(Checked::Valid),
            extensions: None,
            extras: Default::default(),
        });
        self.views.len() - 1
    }

    fn push_floats(&mut self, values: &[f32], target: Option<Target>) -> usize {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.push_view(&bytes, target)
    }

    fn push_floats2(&mut self, values: &[[f32; 2]], target: Option<Target>) -> usize {
        let flat: Vec<f32> = values.iter().flatten().copied().collect();
        self.push_floats(&flat, target)
    }

    fn push_floats3(&mut self, values: &[[f32; 3]], target: Option<Target>) -> usize {
        let flat: Vec<f32> = values.iter().flatten().copied().collect();
        self.push_floats(&flat, target)
    }

    fn push_floats4(&mut self, values: &[[f32; 4]], target: Option<Target>) -> usize {
        let flat: Vec<f32> = values.iter().flatten().copied().collect();
        self.push_floats(&flat, target)
    }

    fn push_indices(&mut self, values: &[u32]) -> usize {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.push_view(&bytes, Some(Target::ElementArrayBuffer))
    }
}

fn accessor(
    view: usize,
    count: usize,
    accessor_type: AccessorType,
    min: Option<Value>,
    max: Option<Value>,
) -> json::Accessor {
    json::Accessor {
        buffer_view: Some(Index::new(view as u32)),
        byte_offset: None,
        count: USize64::from(count),
        component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
        extensions: None,
        extras: Default::default(),
        type_: Checked::Valid(accessor_type),
        min,
        max,
        name: None,
        normalized: false,
        sparse: None,
    }
}

fn index_accessor(view: usize, count: usize) -> json::Accessor {
    json::Accessor {
        buffer_view: Some(Index::new(view as u32)),
        byte_offset: None,
        count: USize64::from(count),
        component_type: Checked::Valid(GenericComponentType(ComponentType::U32)),
        extensions: None,
        extras: Default::default(),
        type_: Checked::Valid(AccessorType::Scalar),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    }
}

fn min_max3(positions: &[[f32; 3]]) -> (Value, Value) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for position in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    (
        Value::Array(min.iter().map(|v| Value::from(*v)).collect()),
        Value::Array(max.iter().map(|v| Value::from(*v)).collect()),
    )
}

pub fn write(
    options: &WriteOptions<'_>,
    format: Format,
    staging: &mut Staging,
    textures: &mut TexturePlan,
    warnings: &mut WarningSink,
) -> Result<()> {
    let scene = options.scene;
    if options.up_axis == crate::scene::UpAxis::Z {
        warnings.add_for(
            format,
            WarningKind::Note,
            "gltf-up-axis",
            "glTF has no up-axis metadata; wrote Z-up coordinates that consumers will read as Y-up",
            1,
        );
    }
    let embed = format == Format::Glb;
    let mut builder = BinaryBuilder::new();
    let mut root = json::Root {
        accessors: Vec::new(),
        animations: Vec::new(),
        asset: json::Asset {
            generator: Some(format!("trine {}", env!("CARGO_PKG_VERSION"))),
            ..Default::default()
        },
        buffers: Vec::new(),
        buffer_views: Vec::new(),
        scene: Some(Index::new(0)),
        extensions: None,
        extras: Default::default(),
        extensions_used: Vec::new(),
        extensions_required: Vec::new(),
        cameras: Vec::new(),
        images: Vec::new(),
        materials: Vec::new(),
        meshes: Vec::new(),
        nodes: Vec::new(),
        samplers: Vec::new(),
        scenes: vec![json::Scene {
            extensions: None,
            extras: Default::default(),
            name: Some("Scene".to_string()),
            nodes: scene
                .roots
                .iter()
                .map(|root| Index::new(*root as u32))
                .collect(),
        }],
        skins: Vec::new(),
        textures: Vec::new(),
    };

    // Texture images: embed in GLB or stage as external files for glTF.
    let mut used_images: Vec<ImageId> = scene
        .materials
        .iter()
        .flat_map(|material| material.textures.images())
        .filter(|id| scene.images.get(*id).is_some())
        .collect();
    used_images.sort_unstable();
    used_images.dedup();
    let mut image_texture_indices: HashMap<ImageId, usize> = HashMap::new();
    let external_paths = if embed {
        HashMap::new()
    } else {
        textures.stage_for(
            staging,
            warnings,
            scene,
            options.output_path,
            options.texture_dir,
            &used_images,
        )?
    };
    for image_id in &used_images {
        let image = &scene.images[*image_id];
        let mime = image
            .mime
            .clone()
            .or_else(|| Image::mime_for_extension(image.extension()).map(str::to_string));
        let mime = mime.or_else(|| embed.then(|| "image/png".to_string()));
        let json_image = if embed {
            let view = builder.push_view(&image.bytes, None);
            json::Image {
                buffer_view: Some(Index::new(view as u32)),
                mime_type: mime.map(json::image::MimeType),
                name: Some(image.name.clone()),
                uri: None,
                extensions: None,
                extras: Default::default(),
            }
        } else {
            let Some(path) = external_paths.get(image_id) else {
                continue;
            };
            json::Image {
                buffer_view: None,
                mime_type: mime.map(json::image::MimeType),
                name: Some(image.name.clone()),
                uri: Some(path.clone()),
                extensions: None,
                extras: Default::default(),
            }
        };
        let image_index = root.images.len();
        root.images.push(json_image);
        root.textures.push(json::Texture {
            name: Some(image.name.clone()),
            sampler: None,
            source: Index::new(image_index as u32),
            extensions: None,
            extras: Default::default(),
        });
        image_texture_indices.insert(*image_id, root.textures.len() - 1);
    }

    let texture_info = |slot: Option<TextureRef>,
                        image_texture_indices: &HashMap<ImageId, usize>|
     -> Option<json::texture::Info> {
        let slot = slot?;
        let index = image_texture_indices.get(&slot.image)?;
        Some(json::texture::Info {
            index: Index::new(*index as u32),
            tex_coord: 0,
            extensions: None,
            extras: Default::default(),
        })
    };

    for material in &scene.materials {
        let base_color_texture = texture_info(material.textures.base_color, &image_texture_indices);
        let metallic_roughness_texture =
            texture_info(material.textures.metallic_roughness, &image_texture_indices);
        let normal_texture = image_texture_indices
            .get(
                &material
                    .textures
                    .normal
                    .map(|slot| slot.image)
                    .unwrap_or(usize::MAX),
            )
            .map(|index| json::material::NormalTexture {
                index: Index::new(*index as u32),
                scale: 1.0,
                tex_coord: 0,
                extensions: None,
                extras: Default::default(),
            });
        let occlusion_texture = image_texture_indices
            .get(
                &material
                    .textures
                    .occlusion
                    .map(|slot| slot.image)
                    .unwrap_or(usize::MAX),
            )
            .map(|index| json::material::OcclusionTexture {
                index: Index::new(*index as u32),
                strength: json::material::StrengthFactor(1.0),
                tex_coord: 0,
                extensions: None,
                extras: Default::default(),
            });
        root.materials.push(json::Material {
            alpha_cutoff: (material.alpha_mode == AlphaMode::Mask)
                .then_some(json::material::AlphaCutoff(material.alpha_cutoff)),
            alpha_mode: Checked::Valid(match material.alpha_mode {
                AlphaMode::Opaque => json::material::AlphaMode::Opaque,
                AlphaMode::Mask => json::material::AlphaMode::Mask,
                AlphaMode::Blend => json::material::AlphaMode::Blend,
            }),
            double_sided: material.double_sided,
            name: Some(material.name.clone()),
            pbr_metallic_roughness: json::material::PbrMetallicRoughness {
                base_color_factor: json::material::PbrBaseColorFactor(material.base_color),
                base_color_texture,
                metallic_factor: json::material::StrengthFactor(material.metallic),
                roughness_factor: json::material::StrengthFactor(material.roughness),
                metallic_roughness_texture,
                extensions: None,
                extras: Default::default(),
            },
            normal_texture,
            occlusion_texture,
            emissive_texture: texture_info(material.textures.emissive, &image_texture_indices),
            emissive_factor: json::material::EmissiveFactor(material.emissive),
            extensions: None,
            extras: Default::default(),
        });
    }

    let mut mesh_indices: Vec<Option<usize>> = vec![None; scene.meshes.len()];
    for mesh_id in scene.referenced_meshes() {
        let mesh = &scene.meshes[mesh_id];
        let mut json_primitives = Vec::new();
        for primitive in &mesh.primitives {
            if primitive.indices.is_empty() {
                continue;
            }
            let mut attributes: BTreeMap<Checked<Semantic>, Index<json::accessor::Accessor>> =
                BTreeMap::new();
            let view = builder.push_floats3(&primitive.positions, Some(Target::ArrayBuffer));
            let (min, max) = min_max3(&primitive.positions);
            let accessor_index = json::Index::push(
                &mut root.accessors,
                accessor(
                    view,
                    primitive.positions.len(),
                    AccessorType::Vec3,
                    Some(min),
                    Some(max),
                ),
            );
            attributes.insert(Checked::Valid(Semantic::Positions), accessor_index);
            if let Some(normals) = &primitive.normals {
                let view = builder.push_floats3(normals, Some(Target::ArrayBuffer));
                let index = json::Index::push(
                    &mut root.accessors,
                    accessor(view, normals.len(), AccessorType::Vec3, None, None),
                );
                attributes.insert(Checked::Valid(Semantic::Normals), index);
            }
            if let Some(tangents) = &primitive.tangents {
                let view = builder.push_floats4(tangents, Some(Target::ArrayBuffer));
                let index = json::Index::push(
                    &mut root.accessors,
                    accessor(view, tangents.len(), AccessorType::Vec4, None, None),
                );
                attributes.insert(Checked::Valid(Semantic::Tangents), index);
            }
            if let Some(uvs) = primitive.uv0() {
                let view = builder.push_floats2(uvs, Some(Target::ArrayBuffer));
                let index = json::Index::push(
                    &mut root.accessors,
                    accessor(view, uvs.len(), AccessorType::Vec2, None, None),
                );
                attributes.insert(Checked::Valid(Semantic::TexCoords(0)), index);
            }
            if let Some(colors) = &primitive.colors {
                let view = builder.push_floats4(colors, Some(Target::ArrayBuffer));
                let index = json::Index::push(
                    &mut root.accessors,
                    accessor(view, colors.len(), AccessorType::Vec4, None, None),
                );
                attributes.insert(Checked::Valid(Semantic::Colors(0)), index);
            }
            let index_view = builder.push_indices(&primitive.indices);
            let index_accessor_index = json::Index::push(
                &mut root.accessors,
                index_accessor(index_view, primitive.indices.len()),
            );
            json_primitives.push(json::mesh::Primitive {
                attributes,
                extensions: None,
                extras: Default::default(),
                indices: Some(index_accessor_index),
                material: primitive.material.map(|id| Index::new(id as u32)),
                mode: Checked::Valid(Mode::Triangles),
                targets: None,
            });
        }
        if json_primitives.is_empty() {
            continue;
        }
        mesh_indices[mesh_id] = Some(root.meshes.len());
        root.meshes.push(json::Mesh {
            extensions: None,
            extras: Default::default(),
            name: Some(if mesh.name.is_empty() {
                format!("mesh_{mesh_id}")
            } else {
                mesh.name.clone()
            }),
            primitives: json_primitives,
            weights: None,
        });
    }
    if root.meshes.is_empty() {
        return Err(Error::conversion("no triangle geometry to write to glTF"));
    }

    for (index, node) in scene.nodes.iter().enumerate() {
        let matrix = node.local_transform;
        root.nodes.push(json::Node {
            camera: None,
            children: if node.children.is_empty() {
                None
            } else {
                Some(
                    node.children
                        .iter()
                        .map(|child| Index::new(*child as u32))
                        .collect(),
                )
            },
            extensions: None,
            extras: Default::default(),
            matrix: if matrix == Mat4::IDENTITY {
                None
            } else {
                Some(matrix.to_cols_array())
            },
            mesh: node
                .mesh
                .and_then(|mesh| mesh_indices.get(mesh).copied().flatten())
                .map(|mesh| Index::new(mesh as u32)),
            name: Some(if node.name.is_empty() {
                format!("node_{index}")
            } else {
                node.name.clone()
            }),
            rotation: None,
            scale: None,
            translation: None,
            skin: None,
            weights: None,
        });
    }

    builder.align();
    root.buffer_views = builder.views;
    let temp_path = staging.stage(options.output_path)?;
    if embed {
        root.buffers.push(json::Buffer {
            byte_length: USize64::from(builder.blob.len()),
            name: None,
            uri: None,
            extensions: None,
            extras: Default::default(),
        });
        let json_bytes = root.to_vec().map_err(|error| {
            Error::path(&temp_path, format!("cannot serialize glTF JSON: {error}"))
        })?;
        let glb = gltf::binary::Glb {
            header: gltf::binary::Header {
                magic: *b"glTF",
                version: 2,
                length: 0,
            },
            json: Cow::Owned(json_bytes),
            bin: Some(Cow::Owned(builder.blob)),
        };
        let bytes = glb
            .to_vec()
            .map_err(|error| Error::path(&temp_path, format!("cannot build GLB: {error}")))?;
        fs::write(&temp_path, bytes).map_err(|error| Error::write(&temp_path, error))?;
    } else {
        let bin_name = options
            .output_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("scene")
            .to_string()
            + ".bin";
        let bin_path = options
            .output_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&bin_name);
        root.buffers.push(json::Buffer {
            byte_length: USize64::from(builder.blob.len()),
            name: None,
            uri: Some(bin_name),
            extensions: None,
            extras: Default::default(),
        });
        let json_bytes = root.to_vec_pretty().map_err(|error| {
            Error::path(&temp_path, format!("cannot serialize glTF JSON: {error}"))
        })?;
        fs::write(&temp_path, json_bytes).map_err(|error| Error::write(&temp_path, error))?;
        staging.stage_bytes(&bin_path, &builder.blob)?;
    }
    Ok(())
}
