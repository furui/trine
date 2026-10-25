//! Canonical in-memory scene representation shared by every reader and writer.

pub mod transform;
pub mod triangulate;
pub mod validate;

use std::collections::HashSet;
use std::fmt;
use std::path::PathBuf;

use glam::Mat4;

/// Index of a node in [`Scene::nodes`].
pub type NodeId = usize;
/// Index of a mesh in [`Scene::meshes`].
pub type MeshId = usize;
/// Index of a material in [`Scene::materials`].
pub type MaterialId = usize;
/// Index of an image in [`Scene::images`].
pub type ImageId = usize;

/// Physical length units understood by the converter.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum Unit {
    Micrometer,
    #[default]
    Millimeter,
    Centimeter,
    Meter,
    Inch,
    Foot,
}

impl Unit {
    /// Length of one unit in meters.
    pub const fn meters_per_unit(self) -> f64 {
        match self {
            Unit::Micrometer => 1e-6,
            Unit::Millimeter => 1e-3,
            Unit::Centimeter => 1e-2,
            Unit::Meter => 1.0,
            Unit::Inch => 0.0254,
            Unit::Foot => 0.3048,
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "um" | "µm" | "micron" | "microns" | "micrometer" => Some(Unit::Micrometer),
            "mm" | "millimeter" | "millimeters" => Some(Unit::Millimeter),
            "cm" | "centimeter" | "centimeters" => Some(Unit::Centimeter),
            "m" | "meter" | "meters" | "metre" | "metres" => Some(Unit::Meter),
            "in" | "inch" | "inches" => Some(Unit::Inch),
            "ft" | "foot" | "feet" => Some(Unit::Foot),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Unit::Micrometer => "um",
            Unit::Millimeter => "mm",
            Unit::Centimeter => "cm",
            Unit::Meter => "m",
            Unit::Inch => "in",
            Unit::Foot => "ft",
        }
    }

    pub const fn long_name(self) -> &'static str {
        match self {
            Unit::Micrometer => "micrometers",
            Unit::Millimeter => "millimeters",
            Unit::Centimeter => "centimeters",
            Unit::Meter => "meters",
            Unit::Inch => "inches",
            Unit::Foot => "feet",
        }
    }

    /// Scale relative to millimeters.
    pub fn in_millimeters(self) -> f64 {
        self.meters_per_unit() / Unit::Millimeter.meters_per_unit()
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Unit information as it appeared in the source file, before defaults are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitResolution {
    /// The source file stated a unit.
    Known(Unit),
    /// The source file had no unit metadata; a default must be applied at conversion time.
    Unknown,
}

impl UnitResolution {
    pub fn known(self) -> Option<Unit> {
        match self {
            UnitResolution::Known(unit) => Some(unit),
            UnitResolution::Unknown => None,
        }
    }

    pub fn resolve(self, default: Unit) -> Unit {
        self.known().unwrap_or(default)
    }
}

/// Up-axis convention of a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum UpAxis {
    Y,
    Z,
}

impl UpAxis {
    pub const fn as_str(self) -> &'static str {
        match self {
            UpAxis::Y => "y",
            UpAxis::Z => "z",
        }
    }
}

impl fmt::Display for UpAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How the alpha channel of a base color should be interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

impl AlphaMode {
    pub fn as_gltf_str(self) -> &'static str {
        match self {
            AlphaMode::Opaque => "OPAQUE",
            AlphaMode::Mask => "MASK",
            AlphaMode::Blend => "BLEND",
        }
    }

    pub fn parse_gltf(text: &str) -> Self {
        match text {
            "MASK" => AlphaMode::Mask,
            "BLEND" => AlphaMode::Blend,
            _ => AlphaMode::Opaque,
        }
    }
}

/// A reference to an image used by a material texture slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureRef {
    pub image: ImageId,
}

/// The core PBR texture slots tracked by the canonical scene.
#[derive(Debug, Clone, Copy, Default)]
pub struct TextureSlots {
    pub base_color: Option<TextureRef>,
    pub normal: Option<TextureRef>,
    pub metallic_roughness: Option<TextureRef>,
    pub emissive: Option<TextureRef>,
    pub occlusion: Option<TextureRef>,
}

impl TextureSlots {
    pub fn is_empty(&self) -> bool {
        self.base_color.is_none()
            && self.normal.is_none()
            && self.metallic_roughness.is_none()
            && self.emissive.is_none()
            && self.occlusion.is_none()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, TextureRef)> {
        [
            ("base_color", self.base_color),
            ("normal", self.normal),
            ("metallic_roughness", self.metallic_roughness),
            ("emissive", self.emissive),
            ("occlusion", self.occlusion),
        ]
        .into_iter()
        .filter_map(|(name, slot)| slot.map(|slot| (name, slot)))
    }

    pub fn images(&self) -> impl Iterator<Item = ImageId> {
        self.iter().map(|(_, slot)| slot.image)
    }
}

/// Flattened core material. Format-specific approximations are resolved by writers.
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub textures: TextureSlots,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "Material".into(),
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0, 0.0, 0.0],
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            textures: TextureSlots::default(),
        }
    }
}

impl Material {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }
}

/// Texture bytes as found in the source file. Images are never decoded.
#[derive(Debug, Clone)]
pub struct Image {
    pub name: String,
    pub bytes: Vec<u8>,
    pub mime: Option<String>,
    pub source_path: Option<PathBuf>,
}

impl Image {
    /// Guess the file extension from the declared mime type or the source path.
    pub fn extension(&self) -> &'static str {
        if let Some(mime) = &self.mime {
            match mime.as_str() {
                "image/png" => return "png",
                "image/jpeg" | "image/jpg" => return "jpg",
                "image/webp" => return "webp",
                "image/ktx2" => return "ktx2",
                "image/bmp" => return "bmp",
                "image/tga" => return "tga",
                "image/tiff" => return "tiff",
                _ => {}
            }
        }
        if let Some(path) = &self.source_path {
            if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
                match ext.to_ascii_lowercase().as_str() {
                    "png" => return "png",
                    "jpg" | "jpeg" => return "jpg",
                    "webp" => return "webp",
                    "ktx2" => return "ktx2",
                    "bmp" => return "bmp",
                    "tga" => return "tga",
                    "tif" | "tiff" => return "tiff",
                    _ => {}
                }
            }
        }
        "png"
    }

    pub fn mime_for_extension(ext: &str) -> Option<&'static str> {
        match ext.to_ascii_lowercase().as_str() {
            "png" => Some("image/png"),
            "jpg" | "jpeg" => Some("image/jpeg"),
            "webp" => Some("image/webp"),
            "ktx2" => Some("image/ktx2"),
            "bmp" => Some("image/bmp"),
            "tga" => Some("image/tga"),
            "tif" | "tiff" => Some("image/tiff"),
            _ => None,
        }
    }
}

/// One drawable triangle primitive with its own vertex attributes.
#[derive(Debug, Clone)]
pub struct Primitive {
    pub positions: Vec<[f32; 3]>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub tangents: Option<Vec<[f32; 4]>>,
    /// UV sets; only set 0 is canonical, additional sets are dropped by writers.
    pub uvs: Vec<Vec<[f32; 2]>>,
    pub colors: Option<Vec<[f32; 4]>>,
    pub indices: Vec<u32>,
    pub material: Option<MaterialId>,
}

impl Primitive {
    pub fn new(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Self {
        Self {
            positions,
            normals: None,
            tangents: None,
            uvs: Vec::new(),
            colors: None,
            indices,
            material: None,
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn uv0(&self) -> Option<&[[f32; 2]]> {
        self.uvs.first().map(|uvs| uvs.as_slice())
    }

    pub fn uv_set_count(&self) -> usize {
        self.uvs.len()
    }
}

/// A named collection of primitives.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub name: String,
    pub primitives: Vec<Primitive>,
}

impl Mesh {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            primitives: Vec::new(),
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.primitives.iter().map(Primitive::triangle_count).sum()
    }

    pub fn vertex_count(&self) -> usize {
        self.primitives.iter().map(Primitive::vertex_count).sum()
    }
}

/// A scene node with a local transform and an optional mesh.
#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub parent: Option<NodeId>,
    pub children: Vec<NodeId>,
    pub local_transform: Mat4,
    pub mesh: Option<MeshId>,
}

impl Node {
    pub fn new(name: impl Into<String>, local_transform: Mat4) -> Self {
        Self {
            name: name.into(),
            parent: None,
            children: Vec::new(),
            local_transform,
            mesh: None,
        }
    }
}

/// Canonical scene.
#[derive(Debug, Clone)]
pub struct Scene {
    pub nodes: Vec<Node>,
    pub roots: Vec<NodeId>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
    pub unit: UnitResolution,
    pub up_axis: UpAxis,
    /// True when `up_axis` came from file metadata (rather than a format default).
    pub up_axis_from_file: bool,
    pub source_format: crate::io::Format,
}

impl Scene {
    pub fn new(source_format: crate::io::Format) -> Self {
        Self {
            nodes: Vec::new(),
            roots: Vec::new(),
            meshes: Vec::new(),
            materials: Vec::new(),
            images: Vec::new(),
            unit: UnitResolution::Unknown,
            up_axis: UpAxis::Y,
            up_axis_from_file: false,
            source_format,
        }
    }

    pub fn add_node(&mut self, name: impl Into<String>, local_transform: Mat4) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node::new(name, local_transform));
        self.roots.push(id);
        id
    }

    pub fn add_child(
        &mut self,
        parent: NodeId,
        name: impl Into<String>,
        local_transform: Mat4,
    ) -> NodeId {
        let id = self.nodes.len();
        let mut node = Node::new(name, local_transform);
        node.parent = Some(parent);
        self.nodes.push(node);
        self.nodes[parent].children.push(id);
        self.roots.retain(|root| *root != id);
        id
    }

    pub fn add_mesh(&mut self, mesh: Mesh) -> MeshId {
        let id = self.meshes.len();
        self.meshes.push(mesh);
        id
    }

    pub fn add_material(&mut self, material: Material) -> MaterialId {
        let id = self.materials.len();
        self.materials.push(material);
        id
    }

    pub fn add_image(&mut self, image: Image) -> ImageId {
        let id = self.images.len();
        self.images.push(image);
        id
    }

    /// Deterministic pre-order traversal of all nodes.
    pub fn traversal(&self) -> Vec<NodeId> {
        let mut out = Vec::with_capacity(self.nodes.len());
        let mut stack: Vec<NodeId> = self.roots.iter().rev().copied().collect();
        let mut seen = HashSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            out.push(id);
            for child in self.nodes[id].children.iter().rev() {
                stack.push(*child);
            }
        }
        // Defensive: include orphaned nodes in index order.
        for id in 0..self.nodes.len() {
            if seen.insert(id) {
                out.push(id);
            }
        }
        out
    }

    /// Local-to-world matrices for every node, in index order.
    pub fn world_transforms(&self) -> Vec<Mat4> {
        let mut worlds = vec![Mat4::IDENTITY; self.nodes.len()];
        for id in self.traversal() {
            let local = self.nodes[id].local_transform;
            worlds[id] = match self.nodes[id].parent {
                Some(parent) => worlds[parent] * local,
                None => local,
            };
        }
        worlds
    }

    /// Meshes referenced by at least one node, in traversal order.
    pub fn referenced_meshes(&self) -> Vec<MeshId> {
        let mut out = Vec::new();
        for node_id in self.traversal() {
            if let Some(mesh) = self.nodes[node_id].mesh {
                if !out.contains(&mesh) {
                    out.push(mesh);
                }
            }
        }
        out
    }

    pub fn referenced_triangle_count(&self) -> usize {
        self.referenced_meshes()
            .into_iter()
            .map(|mesh| self.meshes[mesh].triangle_count())
            .sum()
    }

    pub fn referenced_vertex_count(&self) -> usize {
        self.referenced_meshes()
            .into_iter()
            .map(|mesh| self.meshes[mesh].vertex_count())
            .sum()
    }

    pub fn triangle_count(&self) -> usize {
        self.meshes.iter().map(Mesh::triangle_count).sum()
    }

    pub fn vertex_count(&self) -> usize {
        self.meshes.iter().map(Mesh::vertex_count).sum()
    }

    /// Bounding box of all positions transformed into world space.
    pub fn world_bounds(&self) -> Option<(glam::Vec3, glam::Vec3)> {
        let worlds = self.world_transforms();
        let mut min = glam::Vec3::splat(f32::INFINITY);
        let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
        let mut any = false;
        for (id, node) in self.nodes.iter().enumerate() {
            let Some(mesh_id) = node.mesh else { continue };
            let transform = worlds[id];
            for primitive in &self.meshes[mesh_id].primitives {
                for position in &primitive.positions {
                    let world = transform.transform_point3(glam::Vec3::from(*position));
                    min = min.min(world);
                    max = max.max(world);
                    any = true;
                }
            }
        }
        any.then_some((min, max))
    }

    /// Ensure every primitive has `uvs.len() <= 1` and strip extra sets.
    pub fn drop_extra_uv_sets(&mut self) -> usize {
        let mut dropped = 0;
        for mesh in &mut self.meshes {
            for primitive in &mut mesh.primitives {
                if primitive.uvs.len() > 1 {
                    dropped += primitive.uvs.len() - 1;
                    primitive.uvs.truncate(1);
                }
            }
        }
        dropped
    }

    pub fn has_textures(&self) -> bool {
        self.materials
            .iter()
            .any(|material| !material.textures.is_empty())
    }

    pub fn has_vertex_colors(&self) -> bool {
        self.meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .any(|primitive| primitive.colors.is_some())
    }

    pub fn has_normals(&self) -> bool {
        self.meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .any(|primitive| primitive.normals.is_some())
    }

    pub fn has_uvs(&self) -> bool {
        self.meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .any(|primitive| !primitive.uvs.is_empty())
    }

    pub fn has_tangents(&self) -> bool {
        self.meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .any(|primitive| primitive.tangents.is_some())
    }

    /// Every distinct material referenced by any primitive.
    pub fn used_materials(&self) -> Vec<MaterialId> {
        let mut used: Vec<MaterialId> = self
            .meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .filter_map(|primitive| primitive.material)
            .collect();
        used.sort_unstable();
        used.dedup();
        used
    }
}
