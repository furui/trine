//! FBX import through the vendored `ufbx` C library.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use glam::{Mat4, Vec4};
use ufbx::{MaterialMap, Mesh as UfbxMesh, Texture as UfbxTexture};

use crate::error::{Error, Result};
use crate::io::{Format, Loaded};
use crate::report::{WarningKind, WarningSink};
use crate::scene::{
    AlphaMode, Image, ImageId, Material, Mesh, Node, Primitive, Scene, TextureRef, Unit,
    UnitResolution, UpAxis,
};

pub fn read(path: &Path) -> Result<Loaded> {
    let filename = path
        .to_str()
        .ok_or_else(|| Error::path(path, "FBX path is not valid UTF-8"))?;
    let root = ufbx::load_file(filename, ufbx::LoadOpts::default())
        .map_err(|error| Error::path(path, format!("invalid FBX: {}", error.description)))?;
    let fbx: &ufbx::Scene = &root;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut warnings = WarningSink::default();
    let mut scene = Scene::new(Format::Fbx);

    let unit_meters = fbx.settings.unit_meters;
    scene.unit = match unit_from_meters(unit_meters) {
        Some(unit) => UnitResolution::Known(unit),
        None => {
            if unit_meters > 0.0 {
                warnings.add(
                    WarningKind::UnsupportedEntity,
                    "fbx-unit",
                    format!(
                        "FBX unit scale {} m/unit is not one of the supported units; assuming the FBX default (cm)",
                        unit_meters
                    ),
                );
            }
            UnitResolution::Unknown
        }
    };
    let (axis, axis_from_file) = axis_from_coordinate(fbx.settings.original_axis_up);
    scene.up_axis = axis;
    scene.up_axis_from_file = axis_from_file;

    let mut image_cache: HashMap<usize, Option<ImageId>> = HashMap::new();
    let mut material_ids: Vec<Option<usize>> = Vec::new();
    let mut material_pointer_map: HashMap<usize, usize> = HashMap::new();
    let mut unsupported_shaders = 0usize;

    for material in &fbx.materials {
        let mut converted = Material::new(if material.element.name.is_empty() {
            "Material"
        } else {
            &material.element.name
        });
        let base_color = color_value(
            &material.pbr.base_color,
            &material.fbx.diffuse_color,
            [1.0, 1.0, 1.0, 1.0],
        );
        converted.base_color = base_color;
        let metallic = value(&material.pbr.metalness, 0.0);
        converted.metallic = metallic.clamp(0.0, 1.0);
        let mut roughness = value(&material.pbr.roughness, -1.0);
        if roughness < 0.0 {
            let exponent = value(&material.fbx.specular_exponent, 0.0);
            roughness = if exponent > 0.0 {
                (2.0 / (exponent + 2.0)).sqrt()
            } else {
                1.0
            };
        }
        converted.roughness = roughness.clamp(0.0, 1.0);
        let emissive = color_value(
            &material.pbr.emission_color,
            &material.fbx.emission_color,
            [0.0, 0.0, 0.0, 1.0],
        );
        converted.emissive = [emissive[0], emissive[1], emissive[2]];
        let opacity = value(&material.pbr.opacity, 1.0);
        let transparency = value(&material.fbx.transparency_factor, 0.0);
        converted.base_color[3] = opacity.min(1.0 - transparency).clamp(0.0, 1.0);
        if converted.base_color[3] < 0.999 {
            converted.alpha_mode = AlphaMode::Blend;
        }
        converted.double_sided = material.features.double_sided.enabled;
        converted.textures.base_color = texture_ref(
            &mut scene,
            &mut image_cache,
            &mut warnings,
            base_dir,
            first_texture(&material.pbr.base_color, &material.fbx.diffuse_color),
        );
        converted.textures.normal = texture_ref(
            &mut scene,
            &mut image_cache,
            &mut warnings,
            base_dir,
            first_texture(&material.pbr.normal_map, &material.fbx.normal_map)
                .or_else(|| texture_option(&material.fbx.bump)),
        );
        converted.textures.metallic_roughness = texture_ref(
            &mut scene,
            &mut image_cache,
            &mut warnings,
            base_dir,
            texture_option(&material.pbr.roughness)
                .or_else(|| texture_option(&material.pbr.metalness)),
        );
        converted.textures.emissive = texture_ref(
            &mut scene,
            &mut image_cache,
            &mut warnings,
            base_dir,
            first_texture(&material.pbr.emission_color, &material.fbx.emission_color),
        );
        converted.textures.occlusion = texture_ref(
            &mut scene,
            &mut image_cache,
            &mut warnings,
            base_dir,
            texture_option(&material.pbr.ambient_occlusion),
        );

        if !matches!(
            material.shader_type,
            ufbx::ShaderType::FbxLambert
                | ufbx::ShaderType::FbxPhong
                | ufbx::ShaderType::BlenderPhong
                | ufbx::ShaderType::WavefrontMtl
                | ufbx::ShaderType::GltfMaterial
        ) {
            unsupported_shaders += 1;
        }
        let pointer = material as *const ufbx::Material as usize;
        let id = scene.add_material(converted);
        material_pointer_map.insert(pointer, id);
        material_ids.push(Some(id));
    }
    if unsupported_shaders > 0 {
        warnings.add_for(
            Format::Fbx,
            WarningKind::Approximation,
            "fbx-shader",
            format!("FBX material shader graphs were flattened to core factors on {unsupported_shaders} material(s)"),
            unsupported_shaders,
        );
    }

    // Meshes.
    let mut mesh_pointer_map: HashMap<usize, usize> = HashMap::new();
    for mesh in &fbx.meshes {
        let mesh_id = build_mesh(&mut scene, &mut warnings, mesh, &material_pointer_map);
        mesh_pointer_map.insert(mesh as *const UfbxMesh as usize, mesh_id);
    }

    // Nodes.
    let mut node_pointer_map: HashMap<usize, Option<usize>> = HashMap::new();
    let mut node_indices: Vec<usize> = Vec::new();
    for node in &fbx.nodes {
        if node.is_root && node.parent.is_none() {
            node_pointer_map.insert(node as *const ufbx::Node as usize, None);
            continue;
        }
        let local = matrix_to_mat4(&node.node_to_parent) * matrix_to_mat4(&node.geometry_to_node);
        let id = scene.nodes.len();
        scene.nodes.push(Node {
            name: if node.element.name.is_empty() {
                format!("node_{id}")
            } else {
                node.element.name.to_string()
            },
            parent: None,
            children: Vec::new(),
            local_transform: local,
            mesh: node.mesh.as_ref().and_then(|mesh| {
                mesh_pointer_map
                    .get(&(&**mesh as *const UfbxMesh as usize))
                    .copied()
            }),
        });
        node_pointer_map.insert(node as *const ufbx::Node as usize, Some(id));
        node_indices.push(id);
    }
    for node in &fbx.nodes {
        let Some(Some(id)) = node_pointer_map.get(&(node as *const ufbx::Node as usize)) else {
            continue;
        };
        let id = *id;
        let parent = node.parent.as_ref().and_then(|parent| {
            node_pointer_map
                .get(&(&**parent as *const ufbx::Node as usize))
                .copied()
                .flatten()
        });
        scene.nodes[id].parent = parent;
        if let Some(parent) = parent {
            scene.nodes[parent].children.push(id);
        } else {
            scene.roots.push(id);
        }
    }
    scene.roots.sort_unstable();
    scene.roots.dedup();

    if !fbx.anim_stacks.is_empty() {
        warnings.add_for(
            Format::Fbx,
            WarningKind::DroppedFeature,
            "animation",
            format!(
                "FBX input contains animation; dropped {} animation stack(s)",
                fbx.anim_stacks.len()
            ),
            fbx.anim_stacks.len(),
        );
    }
    if !fbx.skin_deformers.is_empty() {
        warnings.add_for(
            Format::Fbx,
            WarningKind::DroppedFeature,
            "skinning",
            format!(
                "FBX input contains skinning; dropped {} skin deformer(s)",
                fbx.skin_deformers.len()
            ),
            fbx.skin_deformers.len(),
        );
    }
    if !fbx.blend_deformers.is_empty() {
        warnings.add_for(
            Format::Fbx,
            WarningKind::DroppedFeature,
            "morph",
            format!(
                "FBX input contains morph targets; dropped {} blend deformer(s)",
                fbx.blend_deformers.len()
            ),
            fbx.blend_deformers.len(),
        );
    }
    if !fbx.cameras.is_empty() {
        warnings.add_for(
            Format::Fbx,
            WarningKind::DroppedFeature,
            "camera",
            format!(
                "FBX input contains cameras; dropped {} camera(s)",
                fbx.cameras.len()
            ),
            fbx.cameras.len(),
        );
    }
    if !fbx.lights.is_empty() {
        warnings.add_for(
            Format::Fbx,
            WarningKind::DroppedFeature,
            "light",
            format!(
                "FBX input contains lights; dropped {} light(s)",
                fbx.lights.len()
            ),
            fbx.lights.len(),
        );
    }
    if !fbx.constraints.is_empty() {
        warnings.add_for(
            Format::Fbx,
            WarningKind::DroppedFeature,
            "constraints",
            format!(
                "FBX input contains constraints; dropped {} constraint(s)",
                fbx.constraints.len()
            ),
            fbx.constraints.len(),
        );
    }

    if scene.referenced_triangle_count() == 0 {
        return Err(Error::path(path, "FBX contains no triangle geometry"));
    }
    Ok(Loaded { scene, warnings })
}

fn build_mesh(
    scene: &mut Scene,
    warnings: &mut WarningSink,
    mesh: &UfbxMesh,
    material_pointer_map: &HashMap<usize, usize>,
) -> usize {
    let has_normals = mesh.vertex_normal.exists;
    let has_uvs = mesh.vertex_uv.exists;
    let has_colors = mesh.vertex_color.exists;
    let has_tangents = mesh.vertex_tangent.exists && mesh.vertex_bitangent.exists;
    let extra_uv_sets = mesh.uv_sets.len().saturating_sub(1);
    if extra_uv_sets > 0 {
        warnings.add_n(
            WarningKind::DroppedFeature,
            "extra-uv-sets",
            format!("kept only the first FBX UV set; discarded {extra_uv_sets} additional set(s)"),
            extra_uv_sets,
        );
    }
    let extra_color_sets = mesh.color_sets.len().saturating_sub(1);
    if extra_color_sets > 0 {
        warnings.add_n(
            WarningKind::DroppedFeature,
            "extra-color-sets",
            format!(
                "kept only the first FBX color set; discarded {extra_color_sets} additional set(s)"
            ),
            extra_color_sets,
        );
    }

    struct Bucket {
        primitive: Primitive,
        welded: HashMap<(u32, u32, u32, u32), u32>,
    }
    let mut buckets: Vec<Bucket> = Vec::new();
    let mut bucket_materials: Vec<Option<usize>> = Vec::new();
    let mut face_triangle_count = 0usize;
    let mut num_corners_total = 0usize;

    for (face_index, face) in mesh.faces.iter().enumerate() {
        if face.num_indices < 3 {
            continue;
        }
        let required = (face.num_indices as usize - 2) * 3;
        let mut corners = vec![0u32; required];
        let triangles = mesh.triangulate_face(&mut corners, *face);
        corners.truncate(triangles as usize * 3);
        if corners.is_empty() {
            continue;
        }
        face_triangle_count += triangles as usize;
        let material_slot = mesh.face_material.get(face_index).copied().unwrap_or(0) as usize;
        let material_id = mesh.materials.get(material_slot).and_then(|material| {
            material_pointer_map
                .get(&(&**material as *const ufbx::Material as usize))
                .copied()
        });
        let bucket_index = match bucket_materials.iter().position(|id| *id == material_id) {
            Some(index) => index,
            None => {
                let mut primitive = Primitive::new(Vec::new(), Vec::new());
                primitive.material = material_id;
                buckets.push(Bucket {
                    primitive,
                    welded: HashMap::new(),
                });
                bucket_materials.push(material_id);
                buckets.len() - 1
            }
        };
        let bucket = &mut buckets[bucket_index];
        for corner in corners {
            let corner_index = corner as usize;
            let position_index = mesh.vertex_position.indices[corner_index];
            let normal_index = if has_normals {
                mesh.vertex_normal.indices[corner_index]
            } else {
                u32::MAX
            };
            let uv_index = if has_uvs {
                mesh.uv_sets
                    .first()
                    .map(|set| set.vertex_uv.indices[corner_index])
                    .unwrap_or(u32::MAX)
            } else {
                u32::MAX
            };
            let color_index = if has_colors {
                mesh.color_sets
                    .first()
                    .map(|set| set.vertex_color.indices[corner_index])
                    .unwrap_or_else(|| mesh.vertex_color.indices[corner_index])
            } else {
                u32::MAX
            };
            let key = (position_index, normal_index, uv_index, color_index);
            let vertex_index = match bucket.welded.get(&key) {
                Some(index) => *index,
                None => {
                    let output_index = bucket.primitive.positions.len() as u32;
                    let position = mesh.vertex_position[corner_index];
                    bucket.primitive.positions.push([
                        position.x as f32,
                        position.y as f32,
                        position.z as f32,
                    ]);
                    if has_normals {
                        let normal = mesh.vertex_normal[corner_index];
                        bucket.primitive.normals.get_or_insert_with(Vec::new).push([
                            normal.x as f32,
                            normal.y as f32,
                            normal.z as f32,
                        ]);
                    }
                    if has_uvs {
                        let uv = mesh
                            .uv_sets
                            .first()
                            .map(|set| set.vertex_uv[corner_index])
                            .unwrap_or_else(|| mesh.vertex_uv[corner_index]);
                        if bucket.primitive.uvs.is_empty() {
                            bucket.primitive.uvs.push(Vec::new());
                        }
                        bucket.primitive.uvs[0].push([uv.x as f32, uv.y as f32]);
                    }
                    if has_colors {
                        let color = mesh
                            .color_sets
                            .first()
                            .map(|set| set.vertex_color[corner_index])
                            .unwrap_or_else(|| mesh.vertex_color[corner_index]);
                        bucket.primitive.colors.get_or_insert_with(Vec::new).push([
                            color.x as f32,
                            color.y as f32,
                            color.z as f32,
                            color.w as f32,
                        ]);
                    }
                    if has_tangents {
                        let tangent = mesh.vertex_tangent[corner_index];
                        let bitangent = mesh.vertex_bitangent[corner_index];
                        let normal = if has_normals {
                            let normal = mesh.vertex_normal[corner_index];
                            glam::Vec3::new(normal.x as f32, normal.y as f32, normal.z as f32)
                        } else {
                            glam::Vec3::Y
                        };
                        let t =
                            glam::Vec3::new(tangent.x as f32, tangent.y as f32, tangent.z as f32);
                        let b = glam::Vec3::new(
                            bitangent.x as f32,
                            bitangent.y as f32,
                            bitangent.z as f32,
                        );
                        let handedness = if normal.cross(t).dot(b) < 0.0 {
                            -1.0
                        } else {
                            1.0
                        };
                        bucket
                            .primitive
                            .tangents
                            .get_or_insert_with(Vec::new)
                            .push([t.x, t.y, t.z, handedness]);
                    }
                    bucket.welded.insert(key, output_index);
                    output_index
                }
            };
            bucket.primitive.indices.push(vertex_index);
            num_corners_total += 1;
        }
    }
    let _ = num_corners_total;

    let mut mesh_out = Mesh::new(if mesh.element.name.is_empty() {
        "mesh"
    } else {
        &mesh.element.name
    });
    for bucket in buckets {
        if !bucket.primitive.indices.is_empty() {
            mesh_out.primitives.push(bucket.primitive);
        }
    }
    if face_triangle_count == 0 {
        warnings.add(
            WarningKind::UnsupportedEntity,
            "fbx-empty-mesh",
            format!(
                "FBX mesh '{}' has no triangles and was skipped",
                mesh_out.name
            ),
        );
    }
    scene.add_mesh(mesh_out)
}

fn color_value(primary: &MaterialMap, fallback: &MaterialMap, default: [f32; 4]) -> [f32; 4] {
    let map = if primary.has_value {
        primary
    } else if fallback.has_value {
        fallback
    } else {
        return default;
    };
    [
        map.value_vec4.x as f32,
        map.value_vec4.y as f32,
        map.value_vec4.z as f32,
        map.value_vec4.w as f32,
    ]
}

fn value(map: &MaterialMap, fallback: f32) -> f32 {
    if map.has_value {
        map.value_vec4.x as f32
    } else {
        fallback
    }
}

fn texture_option(map: &MaterialMap) -> Option<&UfbxTexture> {
    map.texture.as_deref()
}

fn first_texture<'a>(
    primary: &'a MaterialMap,
    fallback: &'a MaterialMap,
) -> Option<&'a UfbxTexture> {
    texture_option(primary).or_else(|| texture_option(fallback))
}

fn texture_ref(
    scene: &mut Scene,
    cache: &mut HashMap<usize, Option<ImageId>>,
    warnings: &mut WarningSink,
    base_dir: &Path,
    texture: Option<&UfbxTexture>,
) -> Option<TextureRef> {
    let texture = texture?;
    let pointer = texture as *const UfbxTexture as usize;
    if let Some(cached) = cache.get(&pointer) {
        return cached.map(|image| TextureRef { image });
    }
    let image = load_texture(scene, warnings, base_dir, texture);
    let id = image.map(|image| scene.add_image(image));
    cache.insert(pointer, id);
    id.map(|image| TextureRef { image })
}

fn load_texture(
    _scene: &mut Scene,
    warnings: &mut WarningSink,
    base_dir: &Path,
    texture: &UfbxTexture,
) -> Option<Image> {
    let mut bytes: Option<Vec<u8>> = None;
    let mut source_path = None;
    let mut name = texture.filename.to_string();
    if !texture.content.is_empty() {
        bytes = Some(texture.content.to_vec());
    } else {
        let candidates = [
            texture.absolute_filename.to_string(),
            texture.filename.to_string(),
            texture.relative_filename.to_string(),
        ];
        for candidate in candidates.iter().filter(|candidate| !candidate.is_empty()) {
            let path = if Path::new(candidate).is_absolute() {
                Path::new(candidate).to_path_buf()
            } else {
                base_dir.join(candidate)
            };
            if let Ok(read) = fs::read(&path) {
                bytes = Some(read);
                source_path = Some(path);
                if name.is_empty() {
                    name = candidate.clone();
                }
                break;
            }
        }
    }
    let Some(bytes) = bytes else {
        warnings.add(
            WarningKind::MissingAsset,
            "texture-missing",
            format!(
                "texture '{}' referenced by the FBX file was not found",
                if name.is_empty() { "<unnamed>" } else { &name }
            ),
        );
        return None;
    };
    let file_name = Path::new(&name)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("texture");
    let ext = Path::new(&name)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("png");
    Some(Image {
        name: file_name.to_string(),
        mime: Image::mime_for_extension(ext).map(str::to_string),
        bytes,
        source_path,
    })
}

fn unit_from_meters(value: f64) -> Option<Unit> {
    if value <= 0.0 {
        return None;
    }
    [
        Unit::Micrometer,
        Unit::Millimeter,
        Unit::Centimeter,
        Unit::Meter,
        Unit::Inch,
        Unit::Foot,
    ]
    .into_iter()
    .find(|&unit| (value / unit.meters_per_unit() - 1.0).abs() < 1e-4)
}

fn axis_from_coordinate(axis: ufbx::CoordinateAxis) -> (UpAxis, bool) {
    match axis {
        ufbx::CoordinateAxis::PositiveY | ufbx::CoordinateAxis::NegativeY => (UpAxis::Y, true),
        ufbx::CoordinateAxis::PositiveZ | ufbx::CoordinateAxis::NegativeZ => (UpAxis::Z, true),
        _ => (UpAxis::Y, false),
    }
}

fn matrix_to_mat4(matrix: &ufbx::Matrix) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(matrix.m00 as f32, matrix.m10 as f32, matrix.m20 as f32, 0.0),
        Vec4::new(matrix.m01 as f32, matrix.m11 as f32, matrix.m21 as f32, 0.0),
        Vec4::new(matrix.m02 as f32, matrix.m12 as f32, matrix.m22 as f32, 0.0),
        Vec4::new(matrix.m03 as f32, matrix.m13 as f32, matrix.m23 as f32, 1.0),
    )
}
