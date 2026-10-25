//! Command-line definition and validation.

use std::path::PathBuf;

use clap::Parser;

use crate::error::{Error, Result};
use crate::io::Format;
use crate::ops::gpu::GpuMode;
use crate::ops::normals::NormalMode;
use crate::scene::Unit;
use crate::scene::transform::parse_scale_axes;

const BUILD_VERSION: &str = env!("TRINE_VERSION");

const EXAMPLES: &str = "\
Examples:
  trine model.fbx model.glb --scale 2
  trine part.step part.stl --unit-in mm --unit-out mm
  trine print.stl print.glb
  trine scene.obj scene.obj --no-unit-conversion --force
  trine scene.obj out/{group}.glb --split-groups --force
  trine scene.fbx out/ --split-groups --to glb
  trine cube.glb cube.dae

Units: um, mm, cm, m, in, ft
Formats: fbx, 3mf, stl, obj, off, gltf, glb, dae, ply, step
Unit defaults: OBJ/STL/PLY/OFF assume mm (3D printing); FBX cm; DAE m;
3MF/STEP mm; glTF/GLB m. Missing units never disable conversion.
Targets that cannot store units are written in mm unless --unit-out or
--no-unit-conversion says otherwise. glTF/GLB always use meters.";

#[derive(Debug, Parser)]
#[command(
    name = "trine",
    version = BUILD_VERSION,
    about = "GPU-accelerated converter for static 3D triangle-mesh scenes",
    long_about = "Convert static triangle-mesh scenes between FBX, 3MF, STL, OBJ, OFF, glTF, \
GLB, DAE, PLY, and STEP. Units, scale, and up-axis are resolved automatically and can be \
overridden. Lossy features are reported; --strict turns data loss into an error.",
    after_help = EXAMPLES,
    disable_help_subcommand = true
)]
pub struct Cli {
    /// Input 3D file
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output file, directory, or pattern containing {group}
    #[arg(value_name = "OUTPUT")]
    pub output: PathBuf,

    /// Input format (auto detects from the extension)
    #[arg(short = 'f', long = "from", value_name = "FORMAT", value_parser = ["auto", "fbx", "3mf", "stl", "obj", "off", "gltf", "glb", "dae", "ply", "step", "stp"])]
    pub from: Option<String>,

    /// Output format (defaults to the output extension)
    #[arg(short = 't', long = "to", value_name = "FORMAT", value_parser = ["fbx", "3mf", "stl", "obj", "off", "gltf", "glb", "dae", "ply", "step", "stp"])]
    pub to: Option<String>,

    /// Write one output per source group/node/object
    #[arg(long = "split-groups")]
    pub split_groups: bool,

    /// Uniform scale factor
    #[arg(long, default_value_t = 1.0, value_name = "FACTOR")]
    pub scale: f64,

    /// Per-axis multipliers X,Y,Z
    #[arg(long = "scale-axes", default_value = "1,1,1", value_name = "X,Y,Z")]
    pub scale_axes: String,

    /// Input units
    #[arg(long = "unit-in", value_name = "UNIT", value_parser = ["um", "mm", "cm", "m", "in", "ft"])]
    pub unit_in: Option<String>,

    /// Output units
    #[arg(long = "unit-out", value_name = "UNIT", value_parser = ["um", "mm", "cm", "m", "in", "ft"])]
    pub unit_out: Option<String>,

    /// Preserve raw coordinates; disable unit defaults and conversion
    #[arg(long = "no-unit-conversion", conflicts_with_all = ["unit_in", "unit_out"])]
    pub no_unit_conversion: bool,

    /// Target up axis
    #[arg(long = "up-axis", value_name = "AXIS", value_parser = ["auto", "y", "z"])]
    pub up_axis: Option<String>,

    /// Override the detected source up axis
    #[arg(long = "source-up-axis", value_name = "AXIS", value_parser = ["auto", "y", "z"])]
    pub source_up_axis: Option<String>,

    /// Disable up-axis conversion
    #[arg(long = "keep-axes", conflicts_with_all = ["up_axis", "source_up_axis"])]
    pub keep_axes: bool,

    /// Normal handling
    #[arg(long = "normals", value_name = "MODE", default_value = "preserve", value_parser = ["preserve", "smooth", "flat", "none"])]
    pub normals: String,

    /// GPU acceleration mode
    #[arg(long = "gpu", value_name = "MODE", default_value = "auto", value_parser = ["auto", "on", "off"])]
    pub gpu: String,

    /// STEP tessellation tolerance in source units
    #[arg(long = "tessellation-tolerance", value_name = "VALUE")]
    pub tessellation_tolerance: Option<f64>,

    /// Overwrite existing outputs
    #[arg(long)]
    pub force: bool,

    /// Fail if any feature would be dropped
    #[arg(long)]
    pub strict: bool,

    /// Suppress progress and non-fatal warnings
    #[arg(short = 'q', long)]
    pub quiet: bool,

    /// Verbose diagnostics
    #[arg(short = 'v', long)]
    pub verbose: bool,
}

/// Validated conversion options.
#[derive(Debug, Clone)]
pub struct Options {
    pub input: PathBuf,
    pub output: PathBuf,
    pub from: Option<Format>,
    pub to: Option<Format>,
    pub split_groups: bool,
    pub scale: f64,
    pub scale_axes: [f64; 3],
    pub unit_in: Option<Unit>,
    pub unit_out: Option<Unit>,
    pub no_unit_conversion: bool,
    pub up_axis: Option<crate::scene::UpAxis>,
    pub source_up_axis: Option<crate::scene::UpAxis>,
    pub keep_axes: bool,
    pub normals: NormalMode,
    pub gpu: GpuMode,
    pub tessellation_tolerance: Option<f64>,
    pub force: bool,
    pub strict: bool,
    pub quiet: bool,
    pub verbose: bool,
}

fn parse_axis(text: &str) -> Option<crate::scene::UpAxis> {
    match text {
        "y" => Some(crate::scene::UpAxis::Y),
        "z" => Some(crate::scene::UpAxis::Z),
        _ => None,
    }
}

impl Cli {
    pub fn into_options(self) -> Result<Options> {
        let from = match self.from.as_deref() {
            None | Some("auto") => None,
            Some(text) => Some(
                Format::parse(text)
                    .ok_or_else(|| Error::usage(format!("unknown input format '{text}'")))?,
            ),
        };
        let to = match self.to.as_deref() {
            None => None,
            Some(text) => Some(
                Format::parse(text)
                    .ok_or_else(|| Error::usage(format!("unknown output format '{text}'")))?,
            ),
        };
        let unit_in = self
            .unit_in
            .as_deref()
            .map(|text| Unit::parse(text).expect("clap validated unit"));
        let unit_out = self
            .unit_out
            .as_deref()
            .map(|text| Unit::parse(text).expect("clap validated unit"));
        let up_axis = self.up_axis.as_deref().and_then(parse_axis);
        let source_up_axis = self.source_up_axis.as_deref().and_then(parse_axis);
        let normals = NormalMode::parse(&self.normals)
            .ok_or_else(|| Error::usage(format!("unknown normal mode '{}'", self.normals)))?;
        let gpu = GpuMode::parse(&self.gpu)
            .ok_or_else(|| Error::usage(format!("unknown GPU mode '{}'", self.gpu)))?;
        let scale_axes = parse_scale_axes(&self.scale_axes)?;

        if !self.scale.is_finite() {
            return Err(Error::usage("--scale must be finite"));
        }
        if self.no_unit_conversion && (unit_in.is_some() || unit_out.is_some()) {
            return Err(Error::usage(
                "--no-unit-conversion cannot be combined with --unit-in or --unit-out",
            ));
        }
        if self.keep_axes && (up_axis.is_some() || source_up_axis.is_some()) {
            return Err(Error::usage(
                "--keep-axes cannot be combined with --up-axis or --source-up-axis",
            ));
        }
        if let Some(tolerance) = self.tessellation_tolerance {
            if !tolerance.is_finite() || tolerance <= 0.0 {
                return Err(Error::usage(
                    "--tessellation-tolerance must be a positive finite number",
                ));
            }
        }

        Ok(Options {
            input: self.input,
            output: self.output,
            from,
            to,
            split_groups: self.split_groups,
            scale: self.scale,
            scale_axes,
            unit_in,
            unit_out,
            no_unit_conversion: self.no_unit_conversion,
            up_axis,
            source_up_axis,
            keep_axes: self.keep_axes,
            normals,
            gpu,
            tessellation_tolerance: self.tessellation_tolerance,
            force: self.force,
            strict: self.strict,
            quiet: self.quiet,
            verbose: self.verbose,
        })
    }
}

pub fn parse() -> Result<Options> {
    Cli::parse().into_options()
}
