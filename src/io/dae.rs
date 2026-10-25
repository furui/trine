//! COLLADA (DAE) import via `dae-parser` and COLLADA 1.4.1 export.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use dae_parser::source::{InputKind, ST, SourceRead, SourceReader, XYZ};
use dae_parser::{
    Animation, ColorParam, Document, Effect, FloatParam, Geometry, GeometryElement,
    Image as DaeImage, ImageSource, InputS, Material as DaeMaterial, Mesh as DaeMesh, NewParam,
    Node as DaeNode, PolyList, Polygons, Primitive as DaePrimitive, Scene as DaeScene, Semantic,
    Shader, SurfaceInit, Transform, TriFans, TriStrips, Triangles, UpAxis as DaeUpAxis, Url,
    VisualScene,
};
use glam::{Mat4, Vec3};

use crate::error::{Error, Result};
use crate::io::{Format, Loaded, WriteOptions};
use crate::output::staging::Staging;
use crate::output::textures::TexturePlan;
use crate::report::{WarningKind, WarningSink};
use crate::scene::{
    AlphaMode, Image, ImageId, Material, Mesh, Primitive, Scene, TextureRef, Unit, UnitResolution,
    UpAxis,
};

#[derive(Debug, Clone, Copy)]
struct RGBAColor;

#[derive(Debug, Clone, Copy)]
struct RgbaReader {
    r: usize,
    g: usize,
    b: usize,
    a: Option<usize>,
}

impl InputKind for RGBAColor {
    type Array = dae_parser::FloatArray;
    type Reader = RgbaReader;

    fn new_reader(&self, accessor: &dae_parser::Accessor) -> Option<Self::Reader> {
        let channel = |long: &str, short: &str| {
            accessor
                .param_offset(long, "float")
                .or_else(|| accessor.param_offset(short, "float"))
        };
        Some(RgbaReader {
            r: channel("R", "RED")?,
            g: channel("G", "GREEN")?,
            b: channel("B", "BLUE")?,
            a: channel("A", "ALPHA"),
        })
    }
}

impl SourceRead<RGBAColor> for RgbaReader {
    type Output = [f32; 4];

    fn load(&self, data: &[f32]) -> Self::Output {
        [
            data[self.r],
            data[self.g],
            data[self.b],
            self.a.map(|index| data[index]).unwrap_or(1.0),
        ]
    }
}

pub fn read(path: &Path) -> Result<Loaded> {
    let document = Document::from_file(path)
        .map_err(|error| Error::path(path, format!("invalid DAE: {error:?}")))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut warnings = WarningSink::default();
    let mut scene = Scene::new(Format::Dae);

    let unit_meters = document.asset.unit.meter as f64;
    scene.unit = match unit_from_meters(unit_meters) {
        Some(unit) => UnitResolution::Known(unit),
        None => {
            warnings.add(
                WarningKind::UnsupportedEntity,
                "dae-unit",
                format!(
                    "COLLADA unit scale {unit_meters} m/unit is not supported; assuming meters"
                ),
            );
            UnitResolution::Unknown
        }
    };
    match document.asset.up_axis {
        DaeUpAxis::YUp => {
            scene.up_axis = UpAxis::Y;
            scene.up_axis_from_file = true;
        }
        DaeUpAxis::ZUp => {
            scene.up_axis = UpAxis::Z;
            scene.up_axis_from_file = true;
        }
        DaeUpAxis::XUp => {
            scene.up_axis = UpAxis::Y;
            warnings.add(
                WarningKind::UnsupportedEntity,
                "dae-up-axis",
                "COLLADA X_UP is not supported; assuming Y_UP",
            );
        }
    }

    let mut image_records: Vec<(&str, &DaeImage)> = Vec::new();
    for image in document.iter::<DaeImage>() {
        if let Some(id) = image.id.as_deref() {
            image_records.push((id, image));
        }
    }
    let mut image_cache: HashMap<String, Option<ImageId>> = HashMap::new();

    let mut materials_by_id: HashMap<String, usize> = HashMap::new();
    for material in document.iter::<DaeMaterial>() {
        let effect_id = match &material.instance_effect.url.val {
            Url::Fragment(id) => id.clone(),
            _ => continue,
        };
        let mut converted = Material::new(
            material
                .name
                .clone()
                .unwrap_or_else(|| "Material".to_string()),
        );
        if let Some(effect) = document
            .iter::<Effect>()
            .find(|effect| effect.id == effect_id)
        {
            if let Some(common) = effect.get_common_profile() {
                if let Some(shader) = common.technique.data.shaders.first() {
                    apply_shader(
                        shader,
                        effect,
                        &mut converted,
                        &mut warnings,
                        &image_records,
                        &mut image_cache,
                        &mut scene,
                        base_dir,
                    );
                }
            } else {
                warnings.add(
                    WarningKind::Approximation,
                    "dae-profile",
                    format!("COLLADA effect '{effect_id}' has no profile_COMMON; defaults used"),
                );
            }
        }
        let id = scene.add_material(converted);
        if let Some(id_text) = &material.id {
            materials_by_id.insert(id_text.clone(), id);
        }
    }

    let mut symbol_material: HashMap<String, usize> = HashMap::new();
    for visual_scene in document.iter::<VisualScene>() {
        for node in &visual_scene.nodes {
            collect_bindings(node, &materials_by_id, &mut symbol_material, &mut warnings);
        }
    }

    let mut geometry_meshes: HashMap<String, Option<usize>> = HashMap::new();
    for geometry in document.iter::<Geometry>() {
        let Some(id) = geometry.id.as_deref() else {
            continue;
        };
        let built = match &geometry.element {
            GeometryElement::Mesh(mesh) if !mesh.convex => {
                build_geometry(&mut scene, geometry, mesh, &symbol_material, &mut warnings)
            }
            _ => {
                warnings.add(
                    WarningKind::UnsupportedEntity,
                    "dae-geometry",
                    format!("COLLADA geometry '{id}' is not a triangle mesh and was skipped"),
                );
                None
            }
        };
        geometry_meshes.insert(id.to_string(), built);
    }

    let visual_scene = document
        .scene
        .as_ref()
        .and_then(|scene: &DaeScene| scene.instance_visual_scene.as_ref())
        .and_then(|instance| match &instance.url.val {
            Url::Fragment(id) => document
                .iter::<VisualScene>()
                .find(|scene| scene.id.as_deref() == Some(id)),
            _ => None,
        })
        .or_else(|| document.iter::<VisualScene>().next());
    if let Some(visual_scene) = visual_scene {
        let mut dropped = 0usize;
        for node in &visual_scene.nodes {
            build_node(
                node,
                &geometry_meshes,
                &mut scene,
                None,
                &mut warnings,
                &mut dropped,
            );
        }
        if dropped > 0 {
            warnings.add_for(
                Format::Dae,
                WarningKind::DroppedFeature,
                "dae-instances",
                format!("dropped {dropped} camera/light/controller/instance_node element(s)"),
                dropped,
            );
        }
    }
    let animations = document.iter::<Animation>().count();
    if animations > 0 {
        warnings.add_for(
            Format::Dae,
            WarningKind::DroppedFeature,
            "animation",
            format!("COLLADA input contains animation; dropped {animations} animation clip(s)"),
            animations,
        );
    }
    if scene.referenced_triangle_count() == 0 {
        return Err(Error::path(path, "DAE contains no triangle geometry"));
    }
    Ok(Loaded { scene, warnings })
}

fn build_geometry(
    scene: &mut Scene,
    geometry: &Geometry,
    mesh: &DaeMesh,
    symbol_material: &HashMap<String, usize>,
    warnings: &mut WarningSink,
) -> Option<usize> {
    let id = geometry.id.as_deref().unwrap_or("geometry");
    let vertices = mesh.vertices.as_ref()?;
    let mut sources = dae_parser::LocalMap::default();
    for source in &mesh.sources {
        let _ = sources.push(source);
    }
    let position_reader = vertices
        .inputs
        .iter()
        .find(|input| input.semantic == Semantic::Position)
        .and_then(|input| sources.get_raw(&input.source))
        .and_then(|source| source.reader(XYZ));
    let Some(position_reader) = position_reader else {
        warnings.add(
            WarningKind::UnsupportedEntity,
            "dae-positions",
            format!("COLLADA geometry '{id}' has no readable POSITION source and was skipped"),
        );
        return None;
    };
    let vertex_normal = vertices
        .inputs
        .iter()
        .find(|input| input.semantic == Semantic::Normal)
        .and_then(|input| sources.get_raw(&input.source))
        .and_then(|source| source.reader(XYZ));
    let vertex_color = vertices
        .inputs
        .iter()
        .find(|input| input.semantic == Semantic::Color)
        .and_then(|input| sources.get_raw(&input.source))
        .and_then(|source| source.reader(RGBAColor));

    let mut buckets: HashMap<Option<String>, Bucket> = HashMap::new();
    for primitive in &mesh.elements {
        let symbol = primitive_symbol(primitive);
        let bucket = buckets.entry(symbol).or_default();
        let inputs = inputs_of(primitive);
        let stride = inputs
            .iter()
            .map(|input| input.offset as usize)
            .max()
            .unwrap_or(0)
            + 1;
        let vertex_offset = inputs
            .iter()
            .find(|input| input.semantic == Semantic::Vertex)
            .map(|input| input.offset as usize)
            .unwrap_or(0);
        let normal_reader = inputs
            .iter()
            .find(|input| input.semantic == Semantic::Normal)
            .and_then(|input| sources.get_raw(&input.source))
            .and_then(|source| source.reader(XYZ));
        let uv_reader = inputs
            .iter()
            .find(|input| input.semantic == Semantic::TexCoord && input.set.unwrap_or(0) == 0)
            .and_then(|input| sources.get_raw(&input.source))
            .and_then(|source| source.reader(ST));
        let color_reader = inputs
            .iter()
            .find(|input| input.semantic == Semantic::Color && input.set.unwrap_or(0) == 0)
            .and_then(|input| sources.get_raw(&input.source))
            .and_then(|source| source.reader(RGBAColor));
        let normal_offset = inputs
            .iter()
            .find(|input| input.semantic == Semantic::Normal)
            .map(|input| input.offset as usize);
        let uv_offset = inputs
            .iter()
            .find(|input| input.semantic == Semantic::TexCoord && input.set.unwrap_or(0) == 0)
            .map(|input| input.offset as usize);
        let color_offset = inputs
            .iter()
            .find(|input| input.semantic == Semantic::Color && input.set.unwrap_or(0) == 0)
            .map(|input| input.offset as usize);
        let extra_uvs = inputs
            .iter()
            .filter(|input| input.semantic == Semantic::TexCoord && input.set.unwrap_or(0) > 0)
            .count();
        if extra_uvs > 0 {
            warnings.add_n(
                WarningKind::DroppedFeature,
                "extra-uv-sets",
                format!(
                    "kept only the first COLLADA UV set; discarded {extra_uvs} additional set(s)"
                ),
                extra_uvs,
            );
        }
        let extra_colors = inputs
            .iter()
            .filter(|input| input.semantic == Semantic::Color && input.set.unwrap_or(0) > 0)
            .count();
        if extra_colors > 0 {
            warnings.add_n(
                WarningKind::DroppedFeature,
                "extra-color-sets",
                format!(
                    "kept only the first COLLADA color set; discarded {extra_colors} additional set(s)"
                ),
                extra_colors,
            );
        }
        let normal_from_vertex = normal_reader.is_none() && vertex_normal.is_some();
        let color_from_vertex = color_reader.is_none() && vertex_color.is_some();
        let corners = Corners {
            positions: &position_reader,
            normals: normal_reader.as_ref().or(vertex_normal.as_ref()),
            uvs: uv_reader.as_ref(),
            colors: color_reader.as_ref().or(vertex_color.as_ref()),
            stride,
            vertex_offset,
            normal_offset: normal_offset.unwrap_or(vertex_offset),
            uv_offset: uv_offset.unwrap_or(vertex_offset),
            color_offset: color_offset.unwrap_or(vertex_offset),
            normal_from_vertex,
            uv_from_vertex: uv_offset.is_none(),
            color_from_vertex,
        };
        let read = match primitive {
            DaePrimitive::Triangles(geom) => read_triangles(&corners, geom, bucket),
            DaePrimitive::PolyList(geom) => read_polylist(&corners, geom, bucket),
            DaePrimitive::Polygons(geom) => read_polygons(&corners, geom, bucket),
            DaePrimitive::TriFans(geom) => read_fans(&corners, geom, bucket),
            DaePrimitive::TriStrips(geom) => read_strips(&corners, geom, bucket),
            _ => 0,
        };
        if read == 0 {
            warnings.add(
                WarningKind::UnsupportedEntity,
                "dae-primitive",
                format!("COLLADA geometry '{id}' has an unsupported or empty primitive"),
            );
        }
    }

    let mut mesh_out = Mesh::new(geometry.name.clone().unwrap_or_else(|| id.to_string()));
    for (symbol, mut bucket) in buckets {
        if bucket.indices.is_empty() {
            continue;
        }
        let material = symbol
            .as_deref()
            .and_then(|symbol| symbol_material.get(symbol))
            .copied();
        let mut primitive = Primitive::new(
            std::mem::take(&mut bucket.positions),
            std::mem::take(&mut bucket.indices),
        );
        primitive.normals = bucket.normals.take();
        primitive.uvs = bucket.uvs.take().into_iter().collect();
        primitive.colors = bucket.colors.take();
        primitive.material = material;
        mesh_out.primitives.push(primitive);
    }
    if mesh_out.triangle_count() == 0 {
        return None;
    }
    Some(scene.add_mesh(mesh_out))
}

fn primitive_symbol(primitive: &DaePrimitive) -> Option<String> {
    match primitive {
        DaePrimitive::Triangles(g) => g.material.clone(),
        DaePrimitive::PolyList(g) => g.material.clone(),
        DaePrimitive::Polygons(g) => g.material.clone(),
        DaePrimitive::TriFans(g) => g.material.clone(),
        DaePrimitive::TriStrips(g) => g.material.clone(),
        _ => None,
    }
}

fn inputs_of(primitive: &DaePrimitive) -> Vec<InputS> {
    match primitive {
        DaePrimitive::Triangles(g) => g.inputs.to_vec(),
        DaePrimitive::PolyList(g) => g.inputs.to_vec(),
        DaePrimitive::Polygons(g) => g.inputs.to_vec(),
        DaePrimitive::TriFans(g) => g.inputs.to_vec(),
        DaePrimitive::TriStrips(g) => g.inputs.to_vec(),
        _ => Vec::new(),
    }
}

type CornersTuple = (
    [f32; 3],
    Option<[f32; 3]>,
    Option<[f32; 2]>,
    Option<[f32; 4]>,
);

struct Corners<'a> {
    positions: &'a SourceReader<'a, XYZ>,
    normals: Option<&'a SourceReader<'a, XYZ>>,
    uvs: Option<&'a SourceReader<'a, ST>>,
    colors: Option<&'a SourceReader<'a, RGBAColor>>,
    stride: usize,
    vertex_offset: usize,
    normal_offset: usize,
    uv_offset: usize,
    color_offset: usize,
    normal_from_vertex: bool,
    uv_from_vertex: bool,
    color_from_vertex: bool,
}

impl Corners<'_> {
    fn at(&self, data: &[u32], corner: usize) -> Option<CornersTuple> {
        let base = corner.checked_mul(self.stride)?;
        let vertex_slot = base + self.vertex_offset;
        if vertex_slot >= data.len() {
            return None;
        }
        let vertex_index = data[vertex_slot] as usize;
        if vertex_index >= self.positions.len() {
            return None;
        }
        let position = self.positions.get(vertex_index);
        let normal = self.normals.and_then(|reader| {
            let slot = if self.normal_from_vertex {
                vertex_slot
            } else {
                base + self.normal_offset
            };
            let index = *data.get(slot)? as usize;
            (index < reader.len()).then(|| reader.get(index))
        });
        let uv = self.uvs.and_then(|reader| {
            let slot = if self.uv_from_vertex {
                vertex_slot
            } else {
                base + self.uv_offset
            };
            let index = *data.get(slot)? as usize;
            (index < reader.len()).then(|| reader.get(index))
        });
        let color = self.colors.and_then(|reader| {
            let slot = if self.color_from_vertex {
                vertex_slot
            } else {
                base + self.color_offset
            };
            let index = *data.get(slot)? as usize;
            (index < reader.len()).then(|| reader.get(index))
        });
        Some((position, normal, uv, color))
    }

    fn triangle(&self, data: &[u32], corner: usize, bucket: &mut Bucket, flip: bool) -> bool {
        let (Some(a), Some(b), Some(c)) = (
            self.at(data, corner),
            self.at(data, corner + 1),
            self.at(data, corner + 2),
        ) else {
            return false;
        };
        if flip {
            bucket.triangle(a, c, b);
        } else {
            bucket.triangle(a, b, c);
        }
        true
    }

    fn polygon(&self, data: &[u32], bucket: &mut Bucket) -> bool {
        let count = data.len() / self.stride.max(1);
        let mut vertices = Vec::with_capacity(count);
        for corner in 0..count {
            match self.at(data, corner) {
                Some(vertex) => vertices.push(vertex),
                None => return false,
            }
        }
        for index in 1..vertices.len().saturating_sub(1) {
            bucket.triangle(vertices[0], vertices[index], vertices[index + 1]);
        }
        true
    }
}

fn read_triangles(corners: &Corners<'_>, geom: &Triangles, bucket: &mut Bucket) -> usize {
    let Some(data) = geom.data.prim.as_deref() else {
        return 0;
    };
    let mut triangles = 0usize;
    let stride = corners.stride.max(1);
    for triangle in 0..data.len() / (stride * 3) {
        if corners.triangle(data, triangle * 3, bucket, false) {
            triangles += 1;
        }
    }
    triangles
}

fn read_polylist(corners: &Corners<'_>, geom: &PolyList, bucket: &mut Bucket) -> usize {
    let stride = corners.stride.max(1);
    let mut offset = 0usize;
    let mut triangles = 0usize;
    for &count in &geom.data.vcount {
        let count = count as usize;
        let end = offset + count * stride;
        if end > geom.data.prim.len() {
            return triangles;
        }
        if corners.polygon(&geom.data.prim[offset..end], bucket) {
            triangles += count.saturating_sub(2);
        }
        offset = end;
    }
    triangles
}

fn read_polygons(corners: &Corners<'_>, geom: &Polygons, bucket: &mut Bucket) -> usize {
    let mut triangles = 0usize;
    for polygon in &geom.data.0 {
        if !polygon.hole.is_empty() {
            continue;
        }
        if corners.polygon(&polygon.verts, bucket) {
            triangles += polygon.verts.len() / corners.stride.max(1);
        }
    }
    triangles
}

fn read_fans(corners: &Corners<'_>, geom: &TriFans, bucket: &mut Bucket) -> usize {
    let mut triangles = 0usize;
    for fan in &geom.data.prim {
        let mut vertices = Vec::new();
        for corner in 0..fan.len() / corners.stride.max(1) {
            match corners.at(fan, corner) {
                Some(vertex) => vertices.push(vertex),
                None => break,
            }
        }
        for index in 1..vertices.len().saturating_sub(1) {
            bucket.triangle(vertices[0], vertices[index], vertices[index + 1]);
            triangles += 1;
        }
    }
    triangles
}

fn read_strips(corners: &Corners<'_>, geom: &TriStrips, bucket: &mut Bucket) -> usize {
    let mut triangles = 0usize;
    for strip in &geom.data.prim {
        let mut vertices = Vec::new();
        for corner in 0..strip.len() / corners.stride.max(1) {
            match corners.at(strip, corner) {
                Some(vertex) => vertices.push(vertex),
                None => break,
            }
        }
        for index in 0..vertices.len().saturating_sub(2) {
            if index % 2 == 0 {
                bucket.triangle(vertices[index], vertices[index + 1], vertices[index + 2]);
            } else {
                bucket.triangle(vertices[index + 1], vertices[index], vertices[index + 2]);
            }
            triangles += 1;
        }
    }
    triangles
}

#[derive(Default)]
struct Bucket {
    positions: Vec<[f32; 3]>,
    normals: Option<Vec<[f32; 3]>>,
    uvs: Option<Vec<[f32; 2]>>,
    colors: Option<Vec<[f32; 4]>>,
    indices: Vec<u32>,
    welded: HashMap<Vec<u32>, u32>,
}

impl Bucket {
    fn push(&mut self, vertex: CornersTuple) -> u32 {
        let (position, normal_opt, uv_opt, color_opt) = vertex;
        let normal = normal_opt.unwrap_or([0.0, 0.0, 0.0]);
        let uv = uv_opt.unwrap_or([0.0, 0.0]);
        let color = color_opt.unwrap_or([1.0, 1.0, 1.0, 1.0]);
        if self.normals.is_none() && normal_opt.is_some() {
            let existing = self.positions.len();
            self.normals = Some(vec![[0.0, 0.0, 0.0]; existing]);
        }
        if self.uvs.is_none() && uv_opt.is_some() {
            let existing = self.positions.len();
            self.uvs = Some(vec![[0.0, 0.0]; existing]);
        }
        if self.colors.is_none() && color_opt.is_some() {
            let existing = self.positions.len();
            self.colors = Some(vec![[1.0, 1.0, 1.0, 1.0]; existing]);
        }
        let mut key = Vec::with_capacity(12);
        key.extend(position.iter().map(|value| value.to_bits()));
        key.extend(normal.iter().map(|value| value.to_bits()));
        key.extend(uv.iter().map(|value| value.to_bits()));
        key.extend(color.iter().map(|value| value.to_bits()));
        if let Some(existing) = self.welded.get(&key) {
            return *existing;
        }
        let index = self.positions.len() as u32;
        self.positions.push(position);
        if let Some(normals) = &mut self.normals {
            normals.push(normal);
        }
        if let Some(uvs) = &mut self.uvs {
            uvs.push(uv);
        }
        if let Some(colors) = &mut self.colors {
            colors.push(color);
        }
        self.welded.insert(key, index);
        index
    }

    fn triangle(&mut self, a: CornersTuple, b: CornersTuple, c: CornersTuple) {
        let a = self.push(a);
        let b = self.push(b);
        let c = self.push(c);
        self.indices.extend([a, b, c]);
    }
}

fn collect_bindings(
    node: &DaeNode,
    materials: &HashMap<String, usize>,
    symbol_material: &mut HashMap<String, usize>,
    warnings: &mut WarningSink,
) {
    for instance in &node.instance_geometry {
        for bound in instance.instance_materials() {
            let Some(target) = url_fragment(&bound.target) else {
                continue;
            };
            let Some(material) = materials.get(target) else {
                continue;
            };
            if let Some(existing) = symbol_material.insert(bound.symbol.clone(), *material) {
                if existing != *material {
                    warnings.add(
                        WarningKind::UnsupportedEntity,
                        "dae-symbol",
                        format!(
                            "COLLADA material symbol '{}' is bound to different materials; kept the last binding",
                            bound.symbol
                        ),
                    );
                }
            }
        }
    }
    for child in &node.children {
        collect_bindings(child, materials, symbol_material, warnings);
    }
}

fn url_fragment<T>(url: &dae_parser::UrlRef<T>) -> Option<&str> {
    match &url.val {
        Url::Fragment(id) => Some(id.as_str()),
        Url::Other(id) => Some(id.as_str()),
    }
}

fn build_node(
    node: &DaeNode,
    geometry_meshes: &HashMap<String, Option<usize>>,
    scene: &mut Scene,
    parent: Option<usize>,
    warnings: &mut WarningSink,
    dropped: &mut usize,
) {
    let name = node
        .name
        .clone()
        .or_else(|| node.id.clone())
        .unwrap_or_else(|| "node".to_string());
    let local = node_transform(node, warnings);
    let id = match parent {
        Some(parent) => scene.add_child(parent, name, local),
        None => scene.add_node(name, local),
    };
    *dropped += node.instance_camera.len()
        + node.instance_light.len()
        + node.instance_controller.len()
        + node.instance_node.len();
    let mut meshes = Vec::new();
    for instance in &node.instance_geometry {
        if let Some(id) = url_fragment(&instance.url) {
            if let Some(Some(mesh)) = geometry_meshes.get(id) {
                meshes.push(*mesh);
            }
        }
    }
    if let Some(first) = meshes.first() {
        scene.nodes[id].mesh = Some(*first);
        for mesh in &meshes[1..] {
            let child = scene.add_child(id, format!("geometry_{mesh}"), Mat4::IDENTITY);
            scene.nodes[child].mesh = Some(*mesh);
        }
    }
    for child in &node.children {
        build_node(child, geometry_meshes, scene, Some(id), warnings, dropped);
    }
}

fn node_transform(node: &DaeNode, warnings: &mut WarningSink) -> Mat4 {
    let mut matrix = Mat4::IDENTITY;
    for transform in &node.transforms {
        matrix *= transform_matrix(transform, warnings);
    }
    matrix
}

fn transform_matrix(transform: &Transform, warnings: &mut WarningSink) -> Mat4 {
    match transform {
        Transform::Matrix(matrix) => Mat4::from_cols_array(&matrix.0),
        Transform::Translate(translate) => {
            Mat4::from_translation(Vec3::from_array(translate.0[..3].try_into().unwrap()))
        }
        Transform::Scale(scale) => {
            Mat4::from_scale(Vec3::from_array(scale.0[..3].try_into().unwrap()))
        }
        Transform::Rotate(rotate) => Mat4::from_axis_angle(
            Vec3::from_array(*rotate.axis()),
            rotate.angle().to_radians(),
        ),
        Transform::LookAt(_) | Transform::Skew(_) => {
            warnings.add(
                WarningKind::Approximation,
                "dae-transform",
                "COLLADA look-at/skew transforms are approximated as identity",
            );
            Mat4::IDENTITY
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_shader(
    shader: &Shader,
    effect: &Effect,
    material: &mut Material,
    warnings: &mut WarningSink,
    images: &[(&str, &DaeImage)],
    image_cache: &mut HashMap<String, Option<ImageId>>,
    scene: &mut Scene,
    base_dir: &Path,
) {
    let (diffuse, emission, shininess, transparency) = match shader {
        Shader::Phong(phong) => (
            phong.diffuse.as_deref(),
            phong.emission.as_deref(),
            phong.shininess.as_deref(),
            phong.transparency.as_deref(),
        ),
        Shader::Blinn(blinn) => (
            blinn.diffuse.as_deref(),
            blinn.emission.as_deref(),
            blinn.shininess.as_deref(),
            blinn.transparency.as_deref(),
        ),
        Shader::Lambert(lambert) => (
            lambert.diffuse.as_deref(),
            lambert.emission.as_deref(),
            None,
            None,
        ),
        Shader::Constant(constant) => (
            constant.emission.as_deref(),
            constant.emission.as_deref(),
            None,
            None,
        ),
    };
    if let Some(color) = diffuse {
        match color {
            ColorParam::Color(color) => material.base_color = **color,
            ColorParam::Texture(texture) => {
                material.base_color = [1.0, 1.0, 1.0, 1.0];
                material.textures.base_color = resolve_texture(
                    &texture.texture,
                    effect,
                    images,
                    image_cache,
                    scene,
                    base_dir,
                    warnings,
                );
            }
            ColorParam::Param(_) => {}
        }
    }
    if let Some(ColorParam::Color(color)) = emission.as_ref() {
        material.emissive = [color[0], color[1], color[2]];
    }
    if let Some(FloatParam::Float(value)) = shininess.as_ref() {
        material.roughness = (2.0 / (value + 2.0)).sqrt().clamp(0.0, 1.0);
    }
    if let Some(FloatParam::Float(value)) = transparency.as_ref() {
        material.base_color[3] = (1.0 - value).clamp(0.0, 1.0);
        if material.base_color[3] < 0.999 {
            material.alpha_mode = AlphaMode::Blend;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_texture(
    sid: &str,
    effect: &Effect,
    images: &[(&str, &DaeImage)],
    image_cache: &mut HashMap<String, Option<ImageId>>,
    scene: &mut Scene,
    base_dir: &Path,
    warnings: &mut WarningSink,
) -> Option<TextureRef> {
    let sampler = find_param(effect, sid)?.ty.as_sampler2d()?;
    let surface_sid = &sampler.source.val;
    let surface = find_param(effect, surface_sid)?.ty.as_surface()?;
    let image_id = match &surface.init {
        SurfaceInit::From { image, .. } => image.val.clone(),
        _ => return None,
    };
    if let Some(cached) = image_cache.get(&image_id) {
        return cached.map(|image| TextureRef { image });
    }
    let record = images.iter().find(|(id, _)| *id == image_id);
    let loaded = record.and_then(|(_, image)| load_image(image, base_dir, warnings));
    let id = loaded.map(|image| scene.add_image(image));
    image_cache.insert(image_id, id);
    id.map(|image| TextureRef { image })
}

fn find_param<'a>(effect: &'a Effect, sid: &str) -> Option<&'a NewParam> {
    effect
        .new_param
        .iter()
        .find(|param| param.sid == sid)
        .or_else(|| {
            effect
                .get_common_profile()
                .into_iter()
                .flat_map(|common| common.new_param.iter())
                .find(|param| param.sid == sid)
        })
}

fn load_image(image: &DaeImage, base_dir: &Path, warnings: &mut WarningSink) -> Option<Image> {
    let (bytes, source_path) = match &image.source {
        ImageSource::Data(data) => (data.to_vec(), None),
        ImageSource::InitFrom(url) => {
            let path = match url {
                Url::Other(path) => {
                    let path = PathBuf::from(path);
                    if path.is_absolute() {
                        path
                    } else {
                        base_dir.join(path)
                    }
                }
                Url::Fragment(_) => return None,
            };
            match fs::read(&path) {
                Ok(bytes) => (bytes, Some(path)),
                Err(_) => {
                    warnings.add(
                        WarningKind::MissingAsset,
                        "texture-missing",
                        format!("texture '{}' referenced by the DAE file was not found", url),
                    );
                    return None;
                }
            }
        }
    };
    let name = image
        .name
        .clone()
        .or_else(|| {
            source_path
                .as_ref()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "texture".to_string());
    let extension = source_path
        .as_ref()
        .and_then(|path| path.extension())
        .and_then(|ext| ext.to_str())
        .or(image.format.as_deref())
        .unwrap_or("png");
    Some(Image {
        name,
        bytes,
        mime: Image::mime_for_extension(extension).map(str::to_string),
        source_path,
    })
}

fn unit_from_meters(value: f64) -> Option<Unit> {
    [
        Unit::Micrometer,
        Unit::Millimeter,
        Unit::Centimeter,
        Unit::Meter,
        Unit::Inch,
        Unit::Foot,
    ]
    .into_iter()
    .find(|&unit| (value / unit.meters_per_unit() - 1.0).abs() < 1e-3)
}

pub fn write(
    options: &WriteOptions<'_>,
    staging: &mut Staging,
    textures: &mut TexturePlan,
    warnings: &mut WarningSink,
) -> Result<()> {
    let scene = options.scene;
    if scene.referenced_triangle_count() == 0 {
        return Err(Error::conversion("no triangle geometry to write to DAE"));
    }

    let mut used_materials: Vec<usize> = {
        let mut ids: Vec<usize> = scene
            .meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .filter_map(|primitive| primitive.material)
            .filter(|id| *id < scene.materials.len())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    used_materials.retain(|id| *id < scene.materials.len());
    let images: Vec<ImageId> = {
        let mut ids: Vec<ImageId> = used_materials
            .iter()
            .filter_map(|id| scene.materials[*id].textures.base_color)
            .map(|slot| slot.image)
            .filter(|id| scene.images.get(*id).is_some())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    let image_paths = textures.stage_for(
        staging,
        warnings,
        scene,
        options.output_path,
        options.texture_dir,
        &images,
    )?;

    let mut partial_attributes = 0usize;
    let mut geometry_written: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut xml = String::new();
    writeln!(xml, "<?xml version=\"1.0\" encoding=\"utf-8\"?>").unwrap();
    writeln!(
        xml,
        "<COLLADA xmlns=\"http://www.collada.org/2005/11/COLLADASchema\" version=\"1.4.1\">"
    )
    .unwrap();
    writeln!(xml, "  <asset>").unwrap();
    writeln!(
        xml,
        "    <contributor><authoring_tool>trine</authoring_tool></contributor>"
    )
    .unwrap();
    writeln!(xml, "    <created>2026-01-01T00:00:00Z</created>").unwrap();
    writeln!(xml, "    <modified>2026-01-01T00:00:00Z</modified>").unwrap();
    writeln!(
        xml,
        "    <unit name=\"{}\" meter=\"{}\"/>",
        options.unit.as_str(),
        number64(options.unit.meters_per_unit())
    )
    .unwrap();
    writeln!(
        xml,
        "    <up_axis>{}</up_axis>",
        if options.up_axis == UpAxis::Z {
            "Z_UP"
        } else {
            "Y_UP"
        }
    )
    .unwrap();
    writeln!(xml, "  </asset>").unwrap();

    if !images.is_empty() {
        writeln!(xml, "  <library_images>").unwrap();
        for (index, image_id) in images.iter().enumerate() {
            let Some(path) = image_paths.get(image_id) else {
                continue;
            };
            writeln!(
                xml,
                "    <image id=\"image_{index}\" name=\"{}\"><init_from>{}</init_from></image>",
                escape(&scene.images[*image_id].name),
                escape(path)
            )
            .unwrap();
        }
        writeln!(xml, "  </library_images>").unwrap();
    }

    writeln!(xml, "  <library_effects>").unwrap();
    for (index, material_id) in used_materials.iter().enumerate() {
        let material = &scene.materials[*material_id];
        writeln!(xml, "    <effect id=\"effect_{index}\">").unwrap();
        writeln!(xml, "      <profile_COMMON>").unwrap();
        if let Some(slot) = material.textures.base_color {
            if let Some(image_index) = images.iter().position(|id| *id == slot.image) {
                writeln!(xml, "        <newparam sid=\"surface_{index}\"><surface type=\"2D\"><init_from>image_{image_index}</init_from></surface></newparam>").unwrap();
                writeln!(xml, "        <newparam sid=\"sampler_{index}\"><sampler2D><source>surface_{index}</source></sampler2D></newparam>").unwrap();
            }
        }
        writeln!(xml, "        <technique sid=\"common\">").unwrap();
        writeln!(xml, "          <phong>").unwrap();
        writeln!(
            xml,
            "            <emission><color>{} {} {} 1</color></emission>",
            number(material.emissive[0]),
            number(material.emissive[1]),
            number(material.emissive[2])
        )
        .unwrap();
        writeln!(xml, "            <ambient><color>0 0 0 1</color></ambient>").unwrap();
        let has_texture = material
            .textures
            .base_color
            .map(|slot| images.contains(&slot.image))
            .unwrap_or(false);
        if has_texture {
            writeln!(xml, "            <diffuse><texture texture=\"sampler_{index}\" texcoord=\"UVMap\"/></diffuse>").unwrap();
        } else {
            writeln!(
                xml,
                "            <diffuse><color>{} {} {} {}</color></diffuse>",
                number(material.base_color[0]),
                number(material.base_color[1]),
                number(material.base_color[2]),
                number(material.base_color[3])
            )
            .unwrap();
        }
        let specular = 1.0 - material.roughness;
        writeln!(
            xml,
            "            <specular><color>{} {} {} 1</color></specular>",
            number(specular),
            number(specular),
            number(specular)
        )
        .unwrap();
        let shininess = (2.0 / material.roughness.max(1e-4).powi(2) - 2.0).clamp(1.0, 1000.0);
        writeln!(
            xml,
            "            <shininess><float>{}</float></shininess>",
            number(shininess)
        )
        .unwrap();
        if material.base_color[3] < 0.999 {
            writeln!(
                xml,
                "            <transparent opaque=\"A_ONE\"><color>1 1 1 {}</color></transparent>",
                number(material.base_color[3])
            )
            .unwrap();
            writeln!(
                xml,
                "            <transparency><float>{}</float></transparency>",
                number(1.0 - material.base_color[3])
            )
            .unwrap();
        }
        writeln!(xml, "          </phong>").unwrap();
        writeln!(xml, "        </technique>").unwrap();
        writeln!(xml, "      </profile_COMMON>").unwrap();
        writeln!(xml, "    </effect>").unwrap();
    }
    writeln!(xml, "  </library_effects>").unwrap();

    writeln!(xml, "  <library_materials>").unwrap();
    for (index, material_id) in used_materials.iter().enumerate() {
        writeln!(xml, "    <material id=\"material_{index}\" name=\"{}\"><instance_effect url=\"#effect_{index}\"/></material>",
            escape(&scene.materials[*material_id].name)).unwrap();
    }
    writeln!(xml, "  </library_materials>").unwrap();

    writeln!(xml, "  <library_geometries>").unwrap();
    for mesh_id in scene.referenced_meshes() {
        let mesh = &scene.meshes[mesh_id];
        if mesh.triangle_count() == 0 {
            continue;
        }
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut colors: Vec<[f32; 4]> = Vec::new();
        let used = || {
            mesh.primitives
                .iter()
                .filter(|primitive| !primitive.indices.is_empty())
        };
        let all_normals = used().all(|primitive| primitive.normals.is_some());
        let all_uvs = used().all(|primitive| primitive.uv0().is_some());
        let all_colors = used().all(|primitive| primitive.colors.is_some());
        if used().any(|primitive| primitive.normals.is_some()) && !all_normals {
            partial_attributes += 1;
        }
        if used().any(|primitive| primitive.uv0().is_some()) && !all_uvs {
            partial_attributes += 1;
        }
        if used().any(|primitive| primitive.colors.is_some()) && !all_colors {
            partial_attributes += 1;
        }
        struct Group {
            material: Option<usize>,
            indices: Vec<u32>,
        }
        let mut groups = Vec::new();
        for primitive in &mesh.primitives {
            if primitive.indices.is_empty() {
                continue;
            }
            let base = positions.len();
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
            groups.push(Group {
                material: primitive.material,
                indices: primitive
                    .indices
                    .iter()
                    .map(|index| *index + base as u32)
                    .collect(),
            });
        }
        if positions.is_empty() {
            continue;
        }
        geometry_written.insert(mesh_id);
        writeln!(
            xml,
            "    <geometry id=\"geometry_{mesh_id}\" name=\"{}\">",
            escape(&mesh.name)
        )
        .unwrap();
        writeln!(xml, "      <mesh>").unwrap();
        writeln!(xml, "        <source id=\"geometry_{mesh_id}_positions\">").unwrap();
        writeln!(xml, "          <float_array id=\"geometry_{mesh_id}_positions_array\" count=\"{}\">{}</float_array>",
            positions.len() * 3, floats3(&positions)).unwrap();
        writeln!(xml, "          <technique_common><accessor source=\"#geometry_{mesh_id}_positions_array\" count=\"{}\" stride=\"3\">", positions.len()).unwrap();
        writeln!(xml, "            <param name=\"X\" type=\"float\"/><param name=\"Y\" type=\"float\"/><param name=\"Z\" type=\"float\"/>").unwrap();
        writeln!(xml, "          </accessor></technique_common>").unwrap();
        writeln!(xml, "        </source>").unwrap();
        if all_normals {
            writeln!(xml, "        <source id=\"geometry_{mesh_id}_normals\">").unwrap();
            writeln!(xml, "          <float_array id=\"geometry_{mesh_id}_normals_array\" count=\"{}\">{}</float_array>",
                normals.len() * 3, floats3(&normals)).unwrap();
            writeln!(xml, "          <technique_common><accessor source=\"#geometry_{mesh_id}_normals_array\" count=\"{}\" stride=\"3\">", normals.len()).unwrap();
            writeln!(xml, "            <param name=\"X\" type=\"float\"/><param name=\"Y\" type=\"float\"/><param name=\"Z\" type=\"float\"/>").unwrap();
            writeln!(xml, "          </accessor></technique_common>").unwrap();
            writeln!(xml, "        </source>").unwrap();
        }
        if all_uvs {
            writeln!(xml, "        <source id=\"geometry_{mesh_id}_uvs\">").unwrap();
            writeln!(xml, "          <float_array id=\"geometry_{mesh_id}_uvs_array\" count=\"{}\">{}</float_array>",
                uvs.len() * 2, floats2(&uvs)).unwrap();
            writeln!(xml, "          <technique_common><accessor source=\"#geometry_{mesh_id}_uvs_array\" count=\"{}\" stride=\"2\">", uvs.len()).unwrap();
            writeln!(
                xml,
                "            <param name=\"S\" type=\"float\"/><param name=\"T\" type=\"float\"/>"
            )
            .unwrap();
            writeln!(xml, "          </accessor></technique_common>").unwrap();
            writeln!(xml, "        </source>").unwrap();
        }
        if all_colors {
            writeln!(xml, "        <source id=\"geometry_{mesh_id}_colors\">").unwrap();
            writeln!(xml, "          <float_array id=\"geometry_{mesh_id}_colors_array\" count=\"{}\">{}</float_array>",
                colors.len() * 4, floats4(&colors)).unwrap();
            writeln!(xml, "          <technique_common><accessor source=\"#geometry_{mesh_id}_colors_array\" count=\"{}\" stride=\"4\">", colors.len()).unwrap();
            writeln!(xml, "            <param name=\"R\" type=\"float\"/><param name=\"G\" type=\"float\"/><param name=\"B\" type=\"float\"/><param name=\"A\" type=\"float\"/>").unwrap();
            writeln!(xml, "          </accessor></technique_common>").unwrap();
            writeln!(xml, "        </source>").unwrap();
        }
        writeln!(xml, "        <vertices id=\"geometry_{mesh_id}_vertices\">").unwrap();
        writeln!(
            xml,
            "          <input semantic=\"POSITION\" source=\"#geometry_{mesh_id}_positions\"/>"
        )
        .unwrap();
        writeln!(xml, "        </vertices>").unwrap();
        let mut inputs = format!(
            "<input semantic=\"VERTEX\" source=\"#geometry_{mesh_id}_vertices\" offset=\"0\"/>"
        );
        let mut stride = 1usize;
        if all_normals {
            inputs.push_str(&format!("<input semantic=\"NORMAL\" source=\"#geometry_{mesh_id}_normals\" offset=\"{stride}\"/>"));
            stride += 1;
        }
        if all_uvs {
            inputs.push_str(&format!("<input semantic=\"TEXCOORD\" source=\"#geometry_{mesh_id}_uvs\" offset=\"{stride}\" set=\"0\"/>"));
            stride += 1;
        }
        if all_colors {
            inputs.push_str(&format!("<input semantic=\"COLOR\" source=\"#geometry_{mesh_id}_colors\" offset=\"{stride}\" set=\"0\"/>"));
            stride += 1;
        }
        for group in &groups {
            let symbol = group
                .material
                .and_then(|material_id| used_materials.iter().position(|id| *id == material_id));
            let mut p = Vec::with_capacity(group.indices.len() * stride);
            for vertex in &group.indices {
                p.push(*vertex);
                if all_normals {
                    p.push(*vertex);
                }
                if all_uvs {
                    p.push(*vertex);
                }
                if all_colors {
                    p.push(*vertex);
                }
            }
            if p.is_empty() {
                continue;
            }
            let material_attr = symbol
                .map(|symbol| format!(" material=\"mat_{symbol}\""))
                .unwrap_or_default();
            writeln!(
                xml,
                "        <triangles{material_attr} count=\"{}\">{inputs}<p>{}</p></triangles>",
                group.indices.len() / 3,
                ints(&p)
            )
            .unwrap();
        }
        writeln!(xml, "      </mesh>").unwrap();
        writeln!(xml, "    </geometry>").unwrap();
    }
    writeln!(xml, "  </library_geometries>").unwrap();

    writeln!(xml, "  <library_visual_scenes>").unwrap();
    writeln!(xml, "    <visual_scene id=\"scene\" name=\"Scene\">").unwrap();
    write_nodes(
        scene,
        &scene.roots,
        2,
        &mut xml,
        &used_materials,
        &geometry_written,
    );
    writeln!(xml, "    </visual_scene>").unwrap();
    writeln!(xml, "  </library_visual_scenes>").unwrap();
    writeln!(
        xml,
        "  <scene><instance_visual_scene url=\"#scene\"/></scene>"
    )
    .unwrap();
    writeln!(xml, "</COLLADA>").unwrap();

    if partial_attributes > 0 {
        warnings.add_for(
            Format::Dae,
            WarningKind::DroppedFeature,
            "dae-partial-attributes",
            "COLLADA output needs uniform vertex attributes; dropped attributes that were missing on some primitives",
            partial_attributes,
        );
    }
    let metallic_materials = used_materials
        .iter()
        .filter(|id| scene.materials[**id].metallic > 0.001)
        .count();
    if metallic_materials > 0 {
        warnings.add_for(
            Format::Dae,
            WarningKind::Approximation,
            "pbr-to-phong",
            format!(
                "COLLADA Phong materials approximate metallic values on {metallic_materials} material(s)"
            ),
            metallic_materials,
        );
    }
    let dropped_texture_slots: usize = used_materials
        .iter()
        .map(|id| {
            scene.materials[*id]
                .textures
                .iter()
                .filter(|(slot, _)| *slot != "base_color")
                .count()
        })
        .sum();
    if dropped_texture_slots > 0 {
        warnings.add_for(
            Format::Dae,
            WarningKind::DroppedFeature,
            "dae-texture-slots",
            format!("COLLADA output currently writes only base-color textures; dropped {dropped_texture_slots} other texture slot(s)"),
            dropped_texture_slots,
        );
    }

    let temp_path = staging.stage(options.output_path)?;
    fs::write(&temp_path, xml).map_err(|error| Error::write(&temp_path, error))?;
    Ok(())
}

fn write_nodes(
    scene: &Scene,
    nodes: &[usize],
    depth: usize,
    xml: &mut String,
    used_materials: &[usize],
    geometry_written: &std::collections::HashSet<usize>,
) {
    let indent = "  ".repeat(depth);
    for id in nodes {
        let node = &scene.nodes[*id];
        writeln!(
            xml,
            "{indent}<node id=\"node_{id}\" name=\"{}\">",
            escape(&node.name)
        )
        .unwrap();
        writeln!(
            xml,
            "{indent}  <matrix sid=\"transform\">{}</matrix>",
            matrix_values(node.local_transform)
        )
        .unwrap();
        if let Some(mesh_id) = node
            .mesh
            .filter(|mesh_id| geometry_written.contains(mesh_id))
        {
            writeln!(
                xml,
                "{indent}  <instance_geometry url=\"#geometry_{mesh_id}\">"
            )
            .unwrap();
            let mut materials: Vec<usize> = scene.meshes[mesh_id]
                .primitives
                .iter()
                .filter_map(|primitive| primitive.material)
                .collect();
            materials.sort_unstable();
            materials.dedup();
            if !materials.is_empty() {
                writeln!(xml, "{indent}    <bind_material><technique_common>").unwrap();
                for material_id in materials {
                    if let Some(symbol) = used_materials.iter().position(|id| *id == material_id) {
                        writeln!(xml, "{indent}      <instance_material symbol=\"mat_{symbol}\" target=\"#material_{symbol}\"/>").unwrap();
                    }
                }
                writeln!(xml, "{indent}    </technique_common></bind_material>").unwrap();
            }
            writeln!(xml, "{indent}  </instance_geometry>").unwrap();
        }
        write_nodes(
            scene,
            &node.children,
            depth + 1,
            xml,
            used_materials,
            geometry_written,
        );
        writeln!(xml, "{indent}</node>").unwrap();
    }
}

fn matrix_values(matrix: Mat4) -> String {
    matrix
        .to_cols_array()
        .iter()
        .map(|value| number(*value))
        .collect::<Vec<_>>()
        .join(" ")
}

fn floats2(values: &[[f32; 2]]) -> String {
    values
        .iter()
        .flatten()
        .map(|v| number(*v))
        .collect::<Vec<_>>()
        .join(" ")
}

fn floats4(values: &[[f32; 4]]) -> String {
    values
        .iter()
        .flatten()
        .map(|value| number(*value))
        .collect::<Vec<_>>()
        .join(" ")
}

fn floats3(values: &[[f32; 3]]) -> String {
    values
        .iter()
        .flatten()
        .map(|v| number(*v))
        .collect::<Vec<_>>()
        .join(" ")
}

fn ints(values: &[u32]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(" ")
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
    fn numbers_never_emit_nan_or_double_dots() {
        assert_eq!(number(f32::NAN), "0");
        assert_eq!(number64(f64::INFINITY), "0");
        assert_eq!(number(-0.5), "-0.5");
    }

    #[test]
    fn xml_escaping_covers_all_markup_characters() {
        assert_eq!(escape("a&<>\"b"), "a&amp;&lt;&gt;&quot;b");
    }
}
