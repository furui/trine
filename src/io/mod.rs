//! Format registry, detection, and reader/writer dispatch.

use std::path::Path;

use crate::error::{Error, Result};
use crate::output::staging::Staging;
use crate::output::textures::TexturePlan;
use crate::report::WarningSink;
use crate::scene::{Scene, UpAxis};

pub mod dae;
pub mod fbx;
pub mod gltf;
pub mod obj;
pub mod off;
pub mod ply;
pub mod step;
pub mod stl;
pub mod three_mf;

/// Supported static mesh formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Format {
    Fbx,
    ThreeMf,
    Stl,
    Obj,
    Off,
    Gltf,
    Glb,
    Dae,
    Ply,
    Step,
}

impl Format {
    pub const ALL: [Format; 10] = [
        Format::Fbx,
        Format::ThreeMf,
        Format::Stl,
        Format::Obj,
        Format::Off,
        Format::Gltf,
        Format::Glb,
        Format::Dae,
        Format::Ply,
        Format::Step,
    ];

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "fbx" => Some(Format::Fbx),
            "3mf" => Some(Format::ThreeMf),
            "stl" => Some(Format::Stl),
            "obj" => Some(Format::Obj),
            "off" => Some(Format::Off),
            "gltf" => Some(Format::Gltf),
            "glb" => Some(Format::Glb),
            "dae" | "collada" => Some(Format::Dae),
            "ply" => Some(Format::Ply),
            "step" | "stp" => Some(Format::Step),
            _ => None,
        }
    }

    pub fn from_extension(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?;
        Self::parse(ext)
    }

    /// Canonical file extension.
    pub const fn extension(self) -> &'static str {
        match self {
            Format::Fbx => "fbx",
            Format::ThreeMf => "3mf",
            Format::Stl => "stl",
            Format::Obj => "obj",
            Format::Off => "off",
            Format::Gltf => "gltf",
            Format::Glb => "glb",
            Format::Dae => "dae",
            Format::Ply => "ply",
            Format::Step => "step",
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Format::Fbx => "FBX",
            Format::ThreeMf => "3MF",
            Format::Stl => "STL",
            Format::Obj => "OBJ",
            Format::Off => "OFF",
            Format::Gltf => "glTF",
            Format::Glb => "GLB",
            Format::Dae => "DAE (COLLADA)",
            Format::Ply => "PLY",
            Format::Step => "STEP",
        }
    }

    pub const fn default_up_axis(self) -> UpAxis {
        match self {
            Format::Fbx | Format::Dae | Format::Gltf | Format::Glb | Format::Obj => UpAxis::Y,
            Format::ThreeMf | Format::Step | Format::Stl | Format::Ply | Format::Off => UpAxis::Z,
        }
    }

    pub const fn stores_units(self) -> bool {
        matches!(
            self,
            Format::Fbx | Format::ThreeMf | Format::Step | Format::Dae | Format::Gltf | Format::Glb
        )
    }

    /// Whether `--split-groups` can produce more than one meaningful piece.
    pub const fn supports_split(self) -> bool {
        !matches!(self, Format::Stl | Format::Ply | Format::Off)
    }

    pub fn capabilities(self) -> crate::ops::capabilities::OutputCapabilities {
        crate::ops::capabilities::OutputCapabilities::for_format(self)
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}

/// Scene plus reader warnings.
pub struct Loaded {
    pub scene: Scene,
    pub warnings: WarningSink,
}

impl Loaded {
    pub fn new(scene: Scene) -> Self {
        Self {
            scene,
            warnings: WarningSink::default(),
        }
    }
}

/// Detected or overridden input format.
pub fn resolve_input_format(path: &Path, from: Option<Format>) -> Result<Format> {
    if let Some(format) = from {
        return Ok(format);
    }
    Format::from_extension(path).ok_or_else(|| {
        Error::usage(format!(
            "cannot infer input format from '{}'; use --from <FORMAT>",
            path.display()
        ))
    })
}

/// Detected or overridden output format.
pub fn resolve_output_format(path: &Path, to: Option<Format>) -> Result<Format> {
    if let Some(format) = to {
        return Ok(format);
    }
    Format::from_extension(path).ok_or_else(|| {
        Error::usage(format!(
            "cannot infer output format from '{}'; use --to <FORMAT>",
            path.display()
        ))
    })
}

/// Read a file into the canonical scene.
pub fn read(path: &Path, format: Format, tessellation_tolerance: Option<f64>) -> Result<Loaded> {
    match format {
        Format::Obj => obj::read(path),
        Format::Stl => stl::read(path),
        Format::Ply => ply::read(path),
        Format::Off => off::read(path),
        Format::Gltf | Format::Glb => gltf::read(path, format),
        Format::ThreeMf => three_mf::read(path),
        Format::Dae => dae::read(path),
        Format::Fbx => fbx::read(path),
        Format::Step => step::read(path, tessellation_tolerance),
    }
}

/// Options shared by all writers.
pub struct WriteOptions<'a> {
    pub scene: &'a Scene,
    pub output_path: &'a Path,
    /// Texture directory name relative to the output file's parent.
    pub texture_dir: &'a str,
    /// Resolved target unit, written as metadata where the format stores units.
    pub unit: crate::scene::Unit,
    /// Resolved target up axis, written as metadata where the format stores it.
    pub up_axis: crate::scene::UpAxis,
    /// Do not emit materials, only geometry.
    pub geometry_only: bool,
}

/// Write a scene to `output_path` through the staging area.
pub fn write(
    format: Format,
    options: &WriteOptions<'_>,
    staging: &mut Staging,
    textures: &mut TexturePlan,
    warnings: &mut WarningSink,
) -> Result<()> {
    match format {
        Format::Obj => obj::write(options, staging, textures, warnings),
        Format::Stl => stl::write(options, staging, warnings),
        Format::Ply => ply::write(options, staging, warnings),
        Format::Off => off::write(options, staging, warnings),
        Format::Gltf | Format::Glb => gltf::write(options, format, staging, textures, warnings),
        Format::ThreeMf => three_mf::write(options, staging, textures, warnings),
        Format::Dae => dae::write(options, staging, textures, warnings),
        Format::Fbx => fbx::write(options, staging, textures, warnings),
        Format::Step => step::write(options, staging, warnings),
    }
}
