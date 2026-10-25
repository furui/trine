//! Compact FBX 7.4 ASCII writer for static triangle meshes.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;

use glam::{EulerRot, Mat4, Quat, Vec3};

use crate::error::{Error, Result};
use crate::io::WriteOptions;
use crate::ops::flatten::bake_flattened;
use crate::output::staging::Staging;
use crate::output::textures::TexturePlan;
use crate::report::{WarningKind, WarningSink};
use crate::scene::transform::decompose_trs;
use crate::scene::{ImageId, MaterialId};

const MODEL_BASE: u32 = 1_000_000;
const GEOMETRY_BASE: u32 = 2_000_000;
const MATERIAL_BASE: u32 = 3_000_000;
const DEFAULT_MATERIAL_ID: u32 = 3_999_999;
const TEXTURE_BASE: u32 = 4_000_000;
const VIDEO_BASE: u32 = 5_000_000;
const DOCUMENT_ID: u32 = 9_000_000;

pub fn write(
    options: &WriteOptions<'_>,
    staging: &mut Staging,
    textures: &mut TexturePlan,
    warnings: &mut WarningSink,
) -> Result<()> {
    let mut scene = options.scene.clone();
    let needs_flatten = scene
        .nodes
        .iter()
        .any(|node| decompose_trs(node.local_transform).is_none());
    if needs_flatten {
        bake_flattened(&mut scene, Mat4::IDENTITY, None);
        warnings.add_for(
            crate::io::Format::Fbx,
            WarningKind::DroppedFeature,
            "hierarchy-shear",
            "FBX output cannot represent a sheared/mirrored node transform; baked the hierarchy into world-space meshes",
            1,
        );
    }
    if scene.referenced_triangle_count() == 0 {
        return Err(Error::conversion("no triangle geometry to write to FBX"));
    }

    let used_materials: Vec<MaterialId> = {
        let mut ids: Vec<MaterialId> = scene
            .meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .filter_map(|primitive| primitive.material)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    let mut material_slot: HashMap<MaterialId, u32> = HashMap::new();
    for (index, id) in used_materials.iter().enumerate() {
        material_slot.insert(*id, index as u32);
    }
    let needs_default_material = scene
        .meshes
        .iter()
        .flat_map(|mesh| &mesh.primitives)
        .any(|primitive| primitive.material.is_none());

    let image_ids: Vec<ImageId> = {
        let mut ids: Vec<ImageId> = used_materials
            .iter()
            .filter_map(|id| scene.materials.get(*id))
            .flat_map(|material| material.textures.images())
            .filter(|id| scene.images.get(*id).is_some())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    let image_paths = textures.stage_for(
        staging,
        warnings,
        &scene,
        options.output_path,
        options.texture_dir,
        &image_ids,
    )?;

    let mut objects = String::new();
    let mut connections = String::new();
    let mut definitions: HashMap<&'static str, usize> = HashMap::new();

    let mut partial_attributes = 0usize;
    let mut geometry_written: HashSet<usize> = HashSet::new();
    // Geometry objects.
    for mesh_id in scene.referenced_meshes() {
        let mesh = &scene.meshes[mesh_id];
        if mesh.triangle_count() == 0 {
            continue;
        }
        let geometry_id = GEOMETRY_BASE + mesh_id as u32;
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut colors: Vec<[f32; 4]> = Vec::new();
        let mut tangents: Vec<[f32; 3]> = Vec::new();
        let mut binormals: Vec<[f32; 3]> = Vec::new();
        let mut polygon_indices: Vec<i64> = Vec::new();
        let mut polygon_materials: Vec<u32> = Vec::new();
        let all_normals = mesh
            .primitives
            .iter()
            .filter(|primitive| !primitive.indices.is_empty())
            .all(|primitive| primitive.normals.is_some());
        let all_uvs = mesh
            .primitives
            .iter()
            .filter(|primitive| !primitive.indices.is_empty())
            .all(|primitive| primitive.uv0().is_some());
        let all_colors = mesh
            .primitives
            .iter()
            .filter(|primitive| !primitive.indices.is_empty())
            .all(|primitive| primitive.colors.is_some());
        let all_tangents = mesh
            .primitives
            .iter()
            .filter(|primitive| !primitive.indices.is_empty())
            .all(|primitive| primitive.tangents.is_some());
        for (all, any) in [
            (
                all_normals,
                mesh.primitives
                    .iter()
                    .any(|primitive| primitive.normals.is_some()),
            ),
            (
                all_uvs,
                mesh.primitives
                    .iter()
                    .any(|primitive| primitive.uv0().is_some()),
            ),
            (
                all_colors,
                mesh.primitives
                    .iter()
                    .any(|primitive| primitive.colors.is_some()),
            ),
            (
                all_tangents,
                mesh.primitives
                    .iter()
                    .any(|primitive| primitive.tangents.is_some()),
            ),
        ] {
            if any && !all {
                partial_attributes += 1;
            }
        }
        let mut geometry_materials: Vec<Option<MaterialId>> = Vec::new();

        for primitive in &mesh.primitives {
            if primitive.indices.is_empty() {
                continue;
            }
            let slot = match geometry_materials
                .iter()
                .position(|id| *id == primitive.material)
            {
                Some(index) => index as u32,
                None => {
                    geometry_materials.push(primitive.material);
                    (geometry_materials.len() - 1) as u32
                }
            };
            let base = positions.len() as u32;
            positions.extend_from_slice(&primitive.positions);
            if all_normals {
                normals.extend_from_slice(primitive.normals.as_ref().expect("checked"));
            }
            if all_uvs {
                uvs.extend_from_slice(primitive.uv0().expect("checked"));
            }
            if all_colors {
                colors.extend_from_slice(primitive.colors.as_ref().expect("checked"));
            }
            if all_tangents {
                for (index, tangent) in primitive
                    .tangents
                    .as_ref()
                    .expect("checked")
                    .iter()
                    .enumerate()
                {
                    let t = Vec3::new(tangent[0], tangent[1], tangent[2]);
                    let n = primitive
                        .normals
                        .as_ref()
                        .and_then(|normals| normals.get(index))
                        .map(|normal| Vec3::from(*normal))
                        .unwrap_or(Vec3::Z);
                    let bitangent = n.cross(t) * tangent[3];
                    tangents.push([t.x, t.y, t.z]);
                    binormals.push(bitangent.to_array());
                }
            }
            for triangle in primitive.indices.chunks_exact(3) {
                polygon_indices.push((base + triangle[0]) as i64);
                polygon_indices.push((base + triangle[1]) as i64);
                polygon_indices.push(-((base + triangle[2]) as i64) - 1);
                polygon_materials.push(slot);
            }
        }
        if polygon_indices.is_empty() {
            continue;
        }
        geometry_written.insert(mesh_id);

        *definitions.entry("Geometry").or_default() += 1;
        let name = escape_name(if mesh.name.is_empty() {
            "Geometry"
        } else {
            &mesh.name
        });
        writeln!(
            objects,
            "    Geometry: {geometry_id}, \"Geometry::{name}\", \"Mesh\" {{"
        )
        .unwrap();
        writeln!(objects, "        Properties70:  {{").unwrap();
        writeln!(
            objects,
            "            P: \"Primary Visibility\", \"bool\", \"Bool\", \"\",1"
        )
        .unwrap();
        writeln!(objects, "        }}").unwrap();
        writeln!(
            objects,
            "        Vertices: *{} {{ a: {} }}",
            positions.len() * 3,
            floats3(&positions)
        )
        .unwrap();
        let polygon_values: Vec<String> = polygon_indices.iter().map(|v| v.to_string()).collect();
        writeln!(
            objects,
            "        PolygonVertexIndex: *{} {{ a: {} }}",
            polygon_values.len(),
            polygon_values.join(",")
        )
        .unwrap();
        writeln!(objects, "        GeometryVersion: 124").unwrap();
        if all_normals {
            writeln!(objects, "        LayerElementNormal: 0 {{").unwrap();
            writeln!(objects, "            Version: 101").unwrap();
            writeln!(objects, "            Name: \"\"").unwrap();
            writeln!(
                objects,
                "            MappingInformationType: \"ByPolygonVertex\""
            )
            .unwrap();
            writeln!(objects, "            ReferenceInformationType: \"Direct\"").unwrap();
            writeln!(
                objects,
                "            Normals: *{} {{ a: {} }}",
                normals.len() * 3,
                floats3(&normals)
            )
            .unwrap();
            writeln!(objects, "        }}").unwrap();
        }
        if all_uvs {
            writeln!(objects, "        LayerElementUV: 0 {{").unwrap();
            writeln!(objects, "            Version: 101").unwrap();
            writeln!(objects, "            Name: \"UVMap\"").unwrap();
            writeln!(
                objects,
                "            MappingInformationType: \"ByPolygonVertex\""
            )
            .unwrap();
            writeln!(objects, "            ReferenceInformationType: \"Direct\"").unwrap();
            writeln!(
                objects,
                "            UV: *{} {{ a: {} }}",
                uvs.len() * 2,
                floats2(&uvs)
            )
            .unwrap();
            writeln!(objects, "        }}").unwrap();
        }
        if all_colors {
            writeln!(objects, "        LayerElementColor: 0 {{").unwrap();
            writeln!(objects, "            Version: 101").unwrap();
            writeln!(objects, "            Name: \"Colors\"").unwrap();
            writeln!(
                objects,
                "            MappingInformationType: \"ByPolygonVertex\""
            )
            .unwrap();
            writeln!(objects, "            ReferenceInformationType: \"Direct\"").unwrap();
            writeln!(
                objects,
                "            Colors: *{} {{ a: {} }}",
                colors.len() * 4,
                floats4(&colors)
            )
            .unwrap();
            writeln!(
                objects,
                "            ColorIndex: *{} {{ a: {} }}",
                colors.len(),
                sequence(colors.len())
            )
            .unwrap();
            writeln!(objects, "        }}").unwrap();
        }
        if all_tangents {
            writeln!(objects, "        LayerElementTangent: 0 {{").unwrap();
            writeln!(objects, "            Version: 101").unwrap();
            writeln!(objects, "            Name: \"\"").unwrap();
            writeln!(
                objects,
                "            MappingInformationType: \"ByPolygonVertex\""
            )
            .unwrap();
            writeln!(objects, "            ReferenceInformationType: \"Direct\"").unwrap();
            writeln!(
                objects,
                "            Tangents: *{} {{ a: {} }}",
                tangents.len() * 3,
                floats3(&tangents)
            )
            .unwrap();
            writeln!(objects, "        }}").unwrap();
            writeln!(objects, "        LayerElementBinormal: 0 {{").unwrap();
            writeln!(objects, "            Version: 101").unwrap();
            writeln!(objects, "            Name: \"\"").unwrap();
            writeln!(
                objects,
                "            MappingInformationType: \"ByPolygonVertex\""
            )
            .unwrap();
            writeln!(objects, "            ReferenceInformationType: \"Direct\"").unwrap();
            writeln!(
                objects,
                "            Binormals: *{} {{ a: {} }}",
                binormals.len() * 3,
                floats3(&binormals)
            )
            .unwrap();
            writeln!(objects, "        }}").unwrap();
        }
        writeln!(objects, "        LayerElementMaterial: 0 {{").unwrap();
        writeln!(objects, "            Version: 101").unwrap();
        writeln!(objects, "            Name: \"\"").unwrap();
        writeln!(objects, "            MappingInformationType: \"ByPolygon\"").unwrap();
        writeln!(
            objects,
            "            ReferenceInformationType: \"IndexToDirect\""
        )
        .unwrap();
        let material_values: Vec<String> = polygon_materials
            .iter()
            .map(|slot| slot.to_string())
            .collect();
        writeln!(
            objects,
            "            Materials: *{} {{ a: {} }}",
            material_values.len(),
            material_values.join(",")
        )
        .unwrap();
        writeln!(objects, "        }}").unwrap();
        writeln!(objects, "        Layer: 0 {{").unwrap();
        writeln!(objects, "            Version: 100").unwrap();
        if all_normals {
            writeln!(
                objects,
                "            LayerElement:  {{ Type: \"LayerElementNormal\" TypedIndex: 0 }}"
            )
            .unwrap();
        }
        if all_uvs {
            writeln!(
                objects,
                "            LayerElement:  {{ Type: \"LayerElementUV\" TypedIndex: 0 }}"
            )
            .unwrap();
        }
        if all_colors {
            writeln!(
                objects,
                "            LayerElement:  {{ Type: \"LayerElementColor\" TypedIndex: 0 }}"
            )
            .unwrap();
        }
        if all_tangents {
            writeln!(
                objects,
                "            LayerElement:  {{ Type: \"LayerElementTangent\" TypedIndex: 0 }}"
            )
            .unwrap();
            writeln!(
                objects,
                "            LayerElement:  {{ Type: \"LayerElementBinormal\" TypedIndex: 0 }}"
            )
            .unwrap();
        }
        writeln!(
            objects,
            "            LayerElement:  {{ Type: \"LayerElementMaterial\" TypedIndex: 0 }}"
        )
        .unwrap();
        writeln!(objects, "        }}").unwrap();
        writeln!(objects, "    }}").unwrap();

        for material in &geometry_materials {
            let id = material
                .map(|id| MATERIAL_BASE + id as u32)
                .unwrap_or(DEFAULT_MATERIAL_ID);
            writeln!(connections, "    C: \"OO\", {id}, {geometry_id}").unwrap();
        }
    }

    // Models.
    let node_count = scene.nodes.len();
    for (node_id, node) in scene.nodes.iter().enumerate() {
        if node.mesh.is_none() && node.children.is_empty() {
            continue;
        }
        *definitions.entry("Model").or_default() += 1;
        let model_id = MODEL_BASE + node_id as u32;
        let name = escape_name(if node.name.is_empty() {
            "Node"
        } else {
            &node.name
        });
        let (translation, rotation, scale) =
            decompose_trs(node.local_transform).unwrap_or((Vec3::ZERO, Quat::IDENTITY, Vec3::ONE));
        let (rx, ry, rz) = rotation.to_euler(EulerRot::XYZ);
        writeln!(
            objects,
            "    Model: {model_id}, \"Model::{name}\", \"Mesh\" {{"
        )
        .unwrap();
        writeln!(objects, "        Version: 232").unwrap();
        writeln!(objects, "        Properties70:  {{").unwrap();
        writeln!(
            objects,
            "            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",{},{},{}",
            number(translation.x),
            number(translation.y),
            number(translation.z)
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",{},{},{}",
            number(rx.to_degrees()),
            number(ry.to_degrees()),
            number(rz.to_degrees())
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",{},{},{}",
            number(scale.x),
            number(scale.y),
            number(scale.z)
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"RotationOrder\", \"enum\", \"\", \"\",0"
        )
        .unwrap();
        writeln!(objects, "        }}").unwrap();
        writeln!(objects, "        Shading: T").unwrap();
        writeln!(
            objects,
            "        Culling: \"{}\"",
            if scene.materials.iter().any(|material| material.double_sided) {
                "CullingOff"
            } else {
                "CullingOn"
            }
        )
        .unwrap();
        writeln!(objects, "    }}").unwrap();

        let parent_model = node
            .parent
            .map(|parent| MODEL_BASE + parent as u32)
            .unwrap_or(0);
        writeln!(connections, "    C: \"OO\", {model_id}, {parent_model}").unwrap();
        if let Some(mesh_id) = node.mesh {
            writeln!(
                connections,
                "    C: \"OO\", {}, {model_id}",
                GEOMETRY_BASE + mesh_id as u32
            )
            .unwrap();
        }
    }
    let _ = node_count;

    // Materials.
    for id in &used_materials {
        let material = &scene.materials[*id];
        *definitions.entry("Material").or_default() += 1;
        let object_id = MATERIAL_BASE + *id as u32;
        let name = escape_name(if material.name.is_empty() {
            "Material"
        } else {
            &material.name
        });
        let exponent = ((2.0 / material.roughness.max(1e-4).powi(2)) - 2.0).clamp(1.0, 1000.0);
        writeln!(
            objects,
            "    Material: {object_id}, \"Material::{name}\", \"\" {{"
        )
        .unwrap();
        writeln!(objects, "        Version: 102").unwrap();
        writeln!(objects, "        ShadingModel: \"phong\"").unwrap();
        writeln!(objects, "        MultiLayer: 0").unwrap();
        writeln!(objects, "        Properties70:  {{").unwrap();
        writeln!(
            objects,
            "            P: \"AmbientColor\", \"Color\", \"\", \"A\",0,0,0"
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"DiffuseColor\", \"Color\", \"\", \"A\",{},{},{}",
            number(material.base_color[0]),
            number(material.base_color[1]),
            number(material.base_color[2])
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"DiffuseFactor\", \"Number\", \"\", \"A\",1"
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"SpecularColor\", \"Color\", \"\", \"A\",1,1,1"
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"SpecularFactor\", \"Number\", \"\", \"A\",1"
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"ShininessExponent\", \"Number\", \"\", \"A\",{}",
            number(exponent)
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"EmissiveColor\", \"Color\", \"\", \"A\",{},{},{}",
            number(material.emissive[0]),
            number(material.emissive[1]),
            number(material.emissive[2])
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"EmissiveFactor\", \"Number\", \"\", \"A\",1"
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"Opacity\", \"Number\", \"\", \"A\",{}",
            number(material.base_color[3])
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"TransparencyFactor\", \"Number\", \"\", \"A\",{}",
            number(1.0 - material.base_color[3])
        )
        .unwrap();
        writeln!(objects, "        }}").unwrap();
        writeln!(objects, "    }}").unwrap();
    }
    if needs_default_material {
        *definitions.entry("Material").or_default() += 1;
        writeln!(
            objects,
            "    Material: {DEFAULT_MATERIAL_ID}, \"Material::trine_default\", \"\" {{"
        )
        .unwrap();
        writeln!(objects, "        Version: 102").unwrap();
        writeln!(objects, "        ShadingModel: \"phong\"").unwrap();
        writeln!(objects, "        MultiLayer: 0").unwrap();
        writeln!(objects, "        Properties70:  {{").unwrap();
        writeln!(
            objects,
            "            P: \"DiffuseColor\", \"Color\", \"\", \"A\",0.8,0.8,0.8"
        )
        .unwrap();
        writeln!(
            objects,
            "            P: \"DiffuseFactor\", \"Number\", \"\", \"A\",1"
        )
        .unwrap();
        writeln!(objects, "        }}").unwrap();
        writeln!(objects, "    }}").unwrap();
    }

    // Textures and videos: objects once per image, connections per slot.
    let mut written_images: HashSet<ImageId> = HashSet::new();
    for id in &used_materials {
        let material = &scene.materials[*id];
        for (_, slot) in material.textures.iter() {
            if !written_images.insert(slot.image) {
                continue;
            }
            let Some(relative) = image_paths.get(&slot.image) else {
                continue;
            };
            let image = &scene.images[slot.image];
            let texture_id = TEXTURE_BASE + slot.image as u32;
            let video_id = VIDEO_BASE + slot.image as u32;
            let name = escape_name(&image.name);
            let file_name = escape_name(
                std::path::Path::new(relative)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(relative),
            );
            *definitions.entry("Texture").or_default() += 1;
            *definitions.entry("Video").or_default() += 1;
            writeln!(
                objects,
                "    Texture: {texture_id}, \"Texture::{name}\", \"\" {{"
            )
            .unwrap();
            writeln!(objects, "        Type: \"TextureVideoClip\"").unwrap();
            writeln!(objects, "        Version: 202").unwrap();
            writeln!(objects, "        TextureName: \"Texture::{name}\"").unwrap();
            writeln!(objects, "        Media: \"Video::{name}\"").unwrap();
            writeln!(objects, "        FileName: \"{relative}\"").unwrap();
            writeln!(objects, "        RelativeFilename: \"{relative}\"").unwrap();
            writeln!(objects, "        ModelUVTranslation: 0,0").unwrap();
            writeln!(objects, "        ModelUVScaling: 1,1").unwrap();
            writeln!(objects, "        Texture_Alpha_Source: \"Alpha_Black\"").unwrap();
            writeln!(objects, "        Cropping: 0,0,0,0").unwrap();
            writeln!(objects, "    }}").unwrap();
            writeln!(
                objects,
                "    Video: {video_id}, \"Video::{name}\", \"Clip\" {{"
            )
            .unwrap();
            writeln!(objects, "        Type: \"Clip\"").unwrap();
            writeln!(objects, "        Version: 202").unwrap();
            writeln!(objects, "        UseMipMap: 0").unwrap();
            writeln!(objects, "        Filename: \"{file_name}\"").unwrap();
            writeln!(objects, "        RelativeFilename: \"{relative}\"").unwrap();
            writeln!(objects, "    }}").unwrap();
            writeln!(connections, "    C: \"OO\", {video_id}, {texture_id}").unwrap();
        }
    }
    for id in &used_materials {
        let material = &scene.materials[*id];
        let material_object = MATERIAL_BASE + *id as u32;
        for (slot_name, slot) in material.textures.iter() {
            if !image_paths.contains_key(&slot.image) {
                continue;
            }
            let texture_id = TEXTURE_BASE + slot.image as u32;
            let property = match slot_name {
                "base_color" => "DiffuseColor",
                "normal" => "NormalMap",
                "metallic_roughness" => "ShininessExponent",
                "emissive" => "EmissiveColor",
                "occlusion" => "AmbientColor",
                _ => "DiffuseColor",
            };
            writeln!(
                connections,
                "    C: \"OP\", {texture_id}, {material_object}, \"{property}\""
            )
            .unwrap();
        }
    }

    let object_count: usize = definitions.values().sum();
    let mut output = String::new();
    writeln!(output, "; FBX 7.4.0 project file").unwrap();
    writeln!(output, "; Generated by trine {}", env!("CARGO_PKG_VERSION")).unwrap();
    writeln!(output, "FBXHeaderExtension:  {{").unwrap();
    writeln!(output, "    FBXHeaderVersion: 1003").unwrap();
    writeln!(output, "    FBXVersion: 7400").unwrap();
    writeln!(output, "    CreationTimeStamp:  {{").unwrap();
    writeln!(output, "        Version: 1000").unwrap();
    writeln!(output, "        Year: 2026").unwrap();
    writeln!(output, "        Month: 1").unwrap();
    writeln!(output, "        Day: 1").unwrap();
    writeln!(output, "        Hour: 0").unwrap();
    writeln!(output, "        Minute: 0").unwrap();
    writeln!(output, "        Second: 0").unwrap();
    writeln!(output, "        Millisecond: 0").unwrap();
    writeln!(output, "    }}").unwrap();
    writeln!(output, "    Creator: \"trine\"").unwrap();
    writeln!(output, "}}").unwrap();
    writeln!(output, "GlobalSettings:  {{").unwrap();
    writeln!(output, "    Version: 1000").unwrap();
    let unit_scale = options.unit.meters_per_unit() / 0.01;
    let (up_axis, front_axis) = match options.up_axis {
        crate::scene::UpAxis::Y => (1, 2),
        crate::scene::UpAxis::Z => (2, 1),
    };
    writeln!(output, "    Properties70:  {{").unwrap();
    writeln!(
        output,
        "        P: \"UpAxis\", \"int\", \"Integer\", \"\",{up_axis}"
    )
    .unwrap();
    writeln!(
        output,
        "        P: \"UpAxisSign\", \"int\", \"Integer\", \"\",1"
    )
    .unwrap();
    writeln!(
        output,
        "        P: \"FrontAxis\", \"int\", \"Integer\", \"\",{front_axis}"
    )
    .unwrap();
    writeln!(
        output,
        "        P: \"FrontAxisSign\", \"int\", \"Integer\", \"\",1"
    )
    .unwrap();
    writeln!(
        output,
        "        P: \"CoordAxis\", \"int\", \"Integer\", \"\",0"
    )
    .unwrap();
    writeln!(
        output,
        "        P: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1"
    )
    .unwrap();
    writeln!(
        output,
        "        P: \"OriginalUpAxis\", \"int\", \"Integer\", \"\",{up_axis}"
    )
    .unwrap();
    writeln!(
        output,
        "        P: \"OriginalUpAxisSign\", \"int\", \"Integer\", \"\",1"
    )
    .unwrap();
    writeln!(
        output,
        "        P: \"UnitScaleFactor\", \"double\", \"Number\", \"\",{}",
        number64(unit_scale)
    )
    .unwrap();
    writeln!(
        output,
        "        P: \"OriginalUnitScaleFactor\", \"double\", \"Number\", \"\",{}",
        number64(unit_scale)
    )
    .unwrap();
    writeln!(output, "    }}").unwrap();
    writeln!(output, "}}").unwrap();
    writeln!(output, "Documents:  {{").unwrap();
    writeln!(output, "    Count: 1").unwrap();
    writeln!(output, "    Document: {DOCUMENT_ID}, \"\", \"Scene\" {{").unwrap();
    writeln!(output, "        Properties70:  {{").unwrap();
    writeln!(
        output,
        "            P: \"SourceObject\", \"object\", \"\", \"\""
    )
    .unwrap();
    writeln!(
        output,
        "            P: \"ActiveAnimStackName\", \"KString\", \"\", \"\", \"\""
    )
    .unwrap();
    writeln!(output, "        }}").unwrap();
    writeln!(output, "        RootNode: 0").unwrap();
    writeln!(output, "    }}").unwrap();
    writeln!(output, "}}").unwrap();
    writeln!(output, "References:  {{").unwrap();
    writeln!(output, "}}").unwrap();
    writeln!(output, "Definitions:  {{").unwrap();
    writeln!(output, "    Version: 100").unwrap();
    writeln!(output, "    Count: {object_count}").unwrap();
    writeln!(output, "    ObjectType: \"GlobalSettings\" {{ Count: 1 }}").unwrap();
    for kind in ["Geometry", "Model", "Material", "Texture", "Video"] {
        writeln!(
            output,
            "    ObjectType: \"{kind}\" {{ Count: {} }}",
            definitions.get(kind).copied().unwrap_or(0)
        )
        .unwrap();
    }
    writeln!(output, "}}").unwrap();
    writeln!(output, "Objects:  {{").unwrap();
    output.push_str(&objects);
    writeln!(output, "}}").unwrap();
    writeln!(output, "Connections:  {{").unwrap();
    output.push_str(&connections);
    writeln!(output, "}}").unwrap();
    writeln!(output, "Takes:  {{").unwrap();
    writeln!(output, "    Current: \"\"").unwrap();
    writeln!(output, "}}").unwrap();

    let approximated = used_materials
        .iter()
        .filter(|id| scene.materials[**id].metallic > 0.001)
        .count();
    if approximated > 0 {
        warnings.add_for(
            crate::io::Format::Fbx,
            WarningKind::Approximation,
            "pbr-to-phong",
            format!(
                "FBX Phong materials approximate metallic/roughness on {approximated} material(s)"
            ),
            approximated,
        );
    }
    if partial_attributes > 0 {
        warnings.add_for(
            crate::io::Format::Fbx,
            WarningKind::DroppedFeature,
            "fbx-partial-attributes",
            "FBX output needs uniform geometry layers; dropped attributes that were missing on some primitives",
            partial_attributes,
        );
    }

    let temp_path = staging.stage(options.output_path)?;
    fs::write(&temp_path, output).map_err(|error| Error::write(&temp_path, error))?;
    Ok(())
}

fn floats2(values: &[[f32; 2]]) -> String {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.push(number(value[0]));
        out.push(number(value[1]));
    }
    out.join(",")
}

fn floats3(values: &[[f32; 3]]) -> String {
    let mut out = Vec::with_capacity(values.len() * 3);
    for value in values {
        out.push(number(value[0]));
        out.push(number(value[1]));
        out.push(number(value[2]));
    }
    out.join(",")
}

fn floats4(values: &[[f32; 4]]) -> String {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.push(number(value[0]));
        out.push(number(value[1]));
        out.push(number(value[2]));
        out.push(number(value[3]));
    }
    out.join(",")
}

fn sequence(count: usize) -> String {
    let mut out = Vec::with_capacity(count);
    for value in 0..count {
        out.push(value.to_string());
    }
    out.join(",")
}

fn number(value: f32) -> String {
    if !value.is_finite() || value == 0.0 {
        "0".to_string()
    } else {
        format!("{value}")
    }
}

fn number64(value: f64) -> String {
    if !value.is_finite() || value == 0.0 {
        "0".to_string()
    } else {
        format!("{value}")
    }
}

fn escape_name(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            '"' => '\'',
            '\n' | '\r' => ' ',
            _ => ch,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_numbers_become_zero() {
        assert_eq!(number(f32::NAN), "0");
        assert_eq!(number(f32::INFINITY), "0");
        assert_eq!(number(-0.0), "0");
        assert_eq!(number64(f64::NEG_INFINITY), "0");
    }

    #[test]
    fn names_escape_quotes_and_newlines() {
        assert_eq!(escape_name("a\"b\nc"), "a'b c");
    }
}
