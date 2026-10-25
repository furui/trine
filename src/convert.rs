//! End-to-end conversion pipeline.

use std::path::{Path, PathBuf};

use crate::cli::Options;
use crate::error::{Error, Result};
use crate::io::{self, Format, WriteOptions};
use crate::ops::capabilities;
use crate::ops::flatten::{apply_global_to_roots, bake_flattened};
use crate::ops::gpu::{self, GpuMode};
use crate::ops::normals::{self, NormalMode};
use crate::ops::split::{enumerate_groups, extract_piece, resolve_piece_paths};
use crate::ops::tangents;
use crate::output::staging::Staging;
use crate::output::textures::TexturePlan;
use crate::report::{WarningKind, WarningSink};
use crate::scene::Scene;
use crate::scene::transform::{compose_global, resolve_unit_plan_for_scene, resolve_up_axes};

/// Result of a successful conversion.
#[derive(Debug, Clone)]
pub struct Summary {
    pub input: PathBuf,
    pub outputs: Vec<PathBuf>,
    pub triangles: usize,
    pub vertices: usize,
    pub materials: usize,
    pub images: usize,
    pub warning_count: usize,
}

/// Run one conversion end to end.
pub fn run(options: &Options) -> Result<Summary> {
    if !options.input.exists() {
        return Err(Error::conversion(format!(
            "input file {} does not exist",
            options.input.display()
        )));
    }
    if !options.input.is_file() {
        return Err(Error::conversion(format!(
            "input {} is not a regular file",
            options.input.display()
        )));
    }

    let source_format = io::resolve_input_format(&options.input, options.from)?;
    let target_format = io::resolve_output_format(&options.output, options.to)?;
    if matches!(target_format, Format::Gltf | Format::Glb)
        && options.up_axis == Some(crate::scene::UpAxis::Z)
    {
        return Err(Error::usage(
            "glTF/GLB output is always Y-up; --up-axis z is not allowed",
        ));
    }
    if target_format == Format::ThreeMf && options.up_axis == Some(crate::scene::UpAxis::Y) {
        return Err(Error::usage(
            "3MF output is always Z-up; --up-axis y is not allowed",
        ));
    }
    crate::ops::gpu::check_build_support(options.gpu)?;

    let mut loaded = io::read(
        &options.input,
        source_format,
        options.tessellation_tolerance,
    )?;
    let mut warnings = WarningSink::new(options.quiet);
    warnings.merge(&loaded.warnings);
    let unit_plan = resolve_unit_plan_for_scene(
        &loaded.scene,
        source_format,
        target_format,
        options.unit_in,
        options.unit_out,
        options.no_unit_conversion,
    )?;

    if unit_plan.assumed_source {
        let default_note = match source_format {
            Format::Stl | Format::Obj | Format::Ply | Format::Off => "(3D-printing default)",
            _ => "(format default)",
        };
        warnings.add(
            WarningKind::Assumption,
            "unit-assumed",
            format!(
                "{} has no unit metadata; assuming {} {}",
                options
                    .input
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("input"),
                unit_plan.source_unit.unwrap_or(unit_plan.target_unit),
                default_note
            ),
        );
    }

    if options.no_unit_conversion
        && matches!(target_format, Format::Gltf | Format::Glb)
        && unit_plan
            .source_unit
            .is_some_and(|unit| unit != crate::scene::Unit::Meter)
    {
        warnings.add(
            WarningKind::Note,
            "raw-gltf-units",
            "raw coordinates were preserved, but glTF/GLB always declares meters",
        );
    }

    if options.verbose {
        eprintln!(
            "input: {} ({})  output: {} ({})",
            options.input.display(),
            source_format.display_name(),
            options.output.display(),
            target_format.display_name()
        );
        if unit_plan.conversion_enabled {
            eprintln!(
                "units: {} -> {} (factor {})",
                unit_plan
                    .source_unit
                    .map(|unit| unit.as_str())
                    .unwrap_or("raw"),
                unit_plan.target_unit,
                unit_plan.factor
            );
        } else {
            eprintln!("units: conversion disabled (raw coordinates preserved)");
        }
    }

    let (source_up, target_up) = resolve_up_axes(
        loaded.scene.up_axis,
        options.source_up_axis,
        options.up_axis,
        options.keep_axes,
        target_format,
    );
    if options.verbose {
        eprintln!("up axis: {} -> {}", source_up, target_up);
    }
    let global = compose_global(
        options.scale,
        options.scale_axes,
        unit_plan.factor,
        source_up,
        target_up,
    );

    let gpu = if options.gpu == GpuMode::Off
        || (options.gpu == GpuMode::Auto && loaded.scene.vertex_count() < gpu::GPU_VERTEX_THRESHOLD)
    {
        None
    } else {
        gpu::create(options.gpu, options.verbose, &mut warnings)?
    };

    // Normal handling happens before capability filtering so that user-requested
    // computations are not mistaken for source data.
    for mesh in &mut loaded.scene.meshes {
        for primitive in &mut mesh.primitives {
            normals::apply_with(primitive, options.normals, gpu.as_ref());
        }
    }

    let capabilities = target_format.capabilities();
    if capabilities.tangents {
        let needs_tangents = loaded
            .scene
            .materials
            .iter()
            .any(|material| material.textures.normal.is_some());
        if needs_tangents && options.normals != NormalMode::None {
            for mesh in &mut loaded.scene.meshes {
                for primitive in &mut mesh.primitives {
                    if primitive.tangents.is_none()
                        && primitive.normals.is_some()
                        && primitive.uv0().is_some()
                    {
                        primitive.tangents = tangents::generate_with(primitive, gpu.as_ref());
                    }
                }
            }
        }
    }

    let mut scene = capabilities::filter(
        &loaded.scene,
        target_format,
        unit_plan.target_unit,
        &mut warnings,
    );
    crate::scene::validate::validate(&scene)?;

    let source_has_groups = !matches!(source_format, Format::Stl | Format::Ply | Format::Off);
    let split = options.split_groups && target_format.supports_split() && source_has_groups;
    if options.split_groups && !source_has_groups {
        warnings.add_for(
            target_format,
            WarningKind::Note,
            "split-single-source-group",
            format!(
                "{} input has a single synthetic group; --split-groups writes one file",
                source_format.display_name()
            ),
            1,
        );
    } else if options.split_groups && !target_format.supports_split() {
        warnings.add_for(
            target_format,
            WarningKind::Note,
            "split-single-group",
            format!(
                "{} output has no group concept; --split-groups writes one file",
                target_format.display_name()
            ),
            1,
        );
    }

    let output_paths: Vec<PathBuf> = if split {
        let groups = enumerate_groups(&scene);
        if groups.is_empty() {
            return Err(Error::conversion(
                "--split-groups found no non-empty groups in the input",
            ));
        }
        resolve_piece_paths(&options.output, &options.input, target_format, &groups)?
    } else {
        let single = if options.split_groups {
            fallback_single_path(&options.output, &options.input, target_format)
        } else {
            options.output.clone()
        };
        if looks_like_dir(&single) {
            return Err(Error::usage(format!(
                "output {} is a directory; provide a file name or use --split-groups",
                single.display()
            )));
        }
        vec![single]
    };
    preflight(&output_paths, options.force)?;

    let staging_parent = output_paths
        .first()
        .and_then(|path| path.parent())
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut staging = Staging::new(staging_parent)?;
    let mut textures = TexturePlan::new();
    let mut summary = Summary {
        input: options.input.clone(),
        outputs: output_paths.clone(),
        triangles: 0,
        vertices: 0,
        materials: 0,
        images: 0,
        warning_count: 0,
    };

    if split {
        let groups = enumerate_groups(&scene);
        let input_stem = options
            .input
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("scene");
        let texture_dir = format!("{input_stem}.textures");
        for (group, path) in groups.iter().zip(&output_paths) {
            let Some(node) = group.node else { continue };
            let piece = extract_piece(&scene, node, global, &group.name, gpu.as_ref())?;
            crate::scene::validate::validate(&piece)?;
            let write_options = WriteOptions {
                scene: &piece,
                output_path: path,
                texture_dir: &texture_dir,
                unit: unit_plan.target_unit,
                up_axis: target_up,
                geometry_only: false,
            };
            io::write(
                target_format,
                &write_options,
                &mut staging,
                &mut textures,
                &mut warnings,
            )?;
            summary.triangles += piece.triangle_count();
            summary.vertices += piece.vertex_count();
            summary.materials += piece.materials.len();
            summary.images += piece.images.len();
        }
    } else {
        if capabilities.hierarchy {
            apply_global_to_roots(&mut scene, global);
        } else {
            bake_flattened(&mut scene, global, gpu.as_ref());
        }
        let output_stem = options
            .output
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("scene");
        let texture_dir = format!("{output_stem}.textures");
        let write_options = WriteOptions {
            scene: &scene,
            output_path: &output_paths[0],
            texture_dir: &texture_dir,
            unit: unit_plan.target_unit,
            up_axis: target_up,
            geometry_only: false,
        };
        io::write(
            target_format,
            &write_options,
            &mut staging,
            &mut textures,
            &mut warnings,
        )?;
        summary.triangles = scene.referenced_triangle_count();
        summary.vertices = scene.referenced_vertex_count();
        summary.materials = scene.materials.len();
        summary.images = scene.images.len();
    }

    summary.warning_count = warnings.len();
    warnings.emit();
    if options.strict {
        if let Some(error) = warnings.strict_error() {
            return Err(error);
        }
    }
    staging.finalize(options.force)?;
    Ok(summary)
}

fn fallback_single_path(output: &Path, input: &Path, format: Format) -> PathBuf {
    let input_stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("output");
    if looks_like_dir(output) {
        return output.join(format!("{input_stem}.{}", format.extension()));
    }
    let text = output.to_string_lossy();
    if text.contains("{group}") {
        let replaced = text.replace("{group}", input_stem);
        let mut path = PathBuf::from(replaced);
        if path.extension().is_none() {
            path.set_extension(format.extension());
        }
        path
    } else {
        output.to_path_buf()
    }
}

fn looks_like_dir(path: &Path) -> bool {
    let text = path.to_string_lossy();
    path.is_dir() || text.ends_with('/') || text.ends_with(std::path::MAIN_SEPARATOR)
}

fn preflight(paths: &[PathBuf], force: bool) -> Result<()> {
    if force {
        return Ok(());
    }
    for path in paths {
        if path.exists() {
            return Err(Error::conversion(format!(
                "output {} already exists; pass --force to overwrite",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Utility used by tests and the library API to read a scene without writing.
pub fn load(path: &Path, format: Option<Format>) -> Result<Scene> {
    let format = io::resolve_input_format(path, format)?;
    Ok(io::read(path, format, None)?.scene)
}
