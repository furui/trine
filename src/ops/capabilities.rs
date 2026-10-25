//! Target capability table and lossy-feature stripping.

use crate::io::Format;
use crate::report::{WarningKind, WarningSink};
use crate::scene::{Scene, Unit, UnitResolution};

/// What a writer can represent. One table drives filtering for every format.
#[derive(Debug, Clone, Copy)]
pub struct OutputCapabilities {
    pub normals: bool,
    pub tangents: bool,
    pub uvs: bool,
    pub colors: bool,
    pub materials: bool,
    pub textures: bool,
    pub units: bool,
    pub hierarchy: bool,
    pub groups: bool,
}

impl OutputCapabilities {
    pub const fn for_format(format: Format) -> Self {
        match format {
            Format::Fbx => Self {
                normals: true,
                tangents: true,
                uvs: true,
                colors: true,
                materials: true,
                textures: true,
                units: true,
                hierarchy: true,
                groups: true,
            },
            Format::ThreeMf => Self {
                normals: false,
                tangents: false,
                uvs: false,
                colors: false,
                materials: true,
                textures: false,
                units: true,
                hierarchy: true,
                groups: true,
            },
            Format::Stl => Self {
                normals: false,
                tangents: false,
                uvs: false,
                colors: false,
                materials: false,
                textures: false,
                units: false,
                hierarchy: false,
                groups: false,
            },
            Format::Obj => Self {
                normals: true,
                tangents: false,
                uvs: true,
                colors: false,
                materials: true,
                textures: true,
                units: false,
                hierarchy: false,
                groups: true,
            },
            Format::Off => Self {
                normals: false,
                tangents: false,
                uvs: false,
                colors: true,
                materials: false,
                textures: false,
                units: false,
                hierarchy: false,
                groups: false,
            },
            Format::Gltf | Format::Glb => Self {
                normals: true,
                tangents: true,
                uvs: true,
                colors: true,
                materials: true,
                textures: true,
                units: true,
                hierarchy: true,
                groups: true,
            },
            Format::Dae => Self {
                normals: true,
                tangents: false,
                uvs: true,
                colors: true,
                materials: true,
                textures: true,
                units: true,
                hierarchy: true,
                groups: true,
            },
            Format::Ply => Self {
                normals: true,
                tangents: false,
                uvs: true,
                colors: true,
                materials: false,
                textures: false,
                units: false,
                hierarchy: false,
                groups: false,
            },
            Format::Step => Self {
                normals: false,
                tangents: false,
                uvs: false,
                colors: false,
                materials: false,
                textures: false,
                units: true,
                hierarchy: false,
                groups: true,
            },
        }
    }
}

/// Strip every feature the target cannot represent, recording one warning per
/// lost feature category (counts accumulate through de-duplication).
pub fn filter(
    scene: &Scene,
    format: Format,
    target_unit: Unit,
    warnings: &mut WarningSink,
) -> Scene {
    let caps = format.capabilities();
    let mut out = scene.clone();
    let ext = format.extension().to_ascii_uppercase();

    // Only UV set 0 is canonical; extra sets are lost for every target.
    let mut extra_uvs = 0usize;
    for mesh in &mut out.meshes {
        for primitive in &mut mesh.primitives {
            if primitive.uvs.len() > 1 {
                extra_uvs += primitive.uvs.len() - 1;
                primitive.uvs.truncate(1);
            }
        }
    }
    if extra_uvs > 0 {
        warnings.add_n(
            WarningKind::DroppedFeature,
            "extra-uv-sets",
            format!("kept only UV set 0; discarded {extra_uvs} additional UV set(s)"),
            extra_uvs,
        );
    }

    if !caps.normals {
        let count = out
            .meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .filter(|primitive| primitive.normals.is_some())
            .count();
        if count > 0 {
            warnings.add_for(
                format,
                WarningKind::DroppedFeature,
                "normals",
                format!("{ext} output does not support explicit vertex normals; dropped normals on {count} primitive(s)"),
                count,
            );
        }
        for mesh in &mut out.meshes {
            for primitive in &mut mesh.primitives {
                primitive.normals = None;
            }
        }
    }

    let orphan_tangents = out
        .meshes
        .iter()
        .flat_map(|mesh| &mesh.primitives)
        .filter(|primitive| primitive.normals.is_none() && primitive.tangents.is_some())
        .count();
    if caps.tangents && orphan_tangents > 0 {
        warnings.add_for(
            format,
            WarningKind::DroppedFeature,
            "tangents-without-normals",
            format!(
                "{ext} output cannot express tangents without normals; dropped tangents on {orphan_tangents} primitive(s)"
            ),
            orphan_tangents,
        );
        for mesh in &mut out.meshes {
            for primitive in &mut mesh.primitives {
                if primitive.normals.is_none() {
                    primitive.tangents = None;
                }
            }
        }
    }

    if !caps.tangents {
        let count = out
            .meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .filter(|primitive| primitive.tangents.is_some())
            .count();
        if count > 0 {
            warnings.add_for(
                format,
                WarningKind::DroppedFeature,
                "tangents",
                format!("{ext} output does not support tangents; dropped tangents on {count} primitive(s)"),
                count,
            );
        }
        for mesh in &mut out.meshes {
            for primitive in &mut mesh.primitives {
                primitive.tangents = None;
            }
        }
    }

    if !caps.uvs {
        let count = out
            .meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .filter(|primitive| !primitive.uvs.is_empty())
            .count();
        if count > 0 {
            warnings.add_for(
                format,
                WarningKind::DroppedFeature,
                "uvs",
                format!("{ext} output does not support UV coordinates; dropped UVs on {count} primitive(s)"),
                count,
            );
        }
        for mesh in &mut out.meshes {
            for primitive in &mut mesh.primitives {
                primitive.uvs.clear();
            }
        }
    }

    if !caps.colors {
        let count = out
            .meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .filter(|primitive| primitive.colors.is_some())
            .count();
        if count > 0 {
            warnings.add_for(
                format,
                WarningKind::DroppedFeature,
                "vertex-colors",
                format!("{ext} output does not support vertex colors; dropped colors on {count} primitive(s)"),
                count,
            );
        }
        for mesh in &mut out.meshes {
            for primitive in &mut mesh.primitives {
                primitive.colors = None;
            }
        }
    }

    if !caps.materials {
        let assignments = out
            .meshes
            .iter()
            .flat_map(|mesh| &mesh.primitives)
            .filter(|primitive| primitive.material.is_some())
            .count();
        if assignments > 0 {
            warnings.add_for(
                format,
                WarningKind::DroppedFeature,
                "materials",
                format!("{ext} output does not support materials; dropped {assignments} material assignment(s)"),
                assignments,
            );
        }
        for mesh in &mut out.meshes {
            for primitive in &mut mesh.primitives {
                primitive.material = None;
            }
        }
        out.materials.clear();
    } else if !caps.textures {
        let mut slots = 0usize;
        let mut touched = 0usize;
        for material in &mut out.materials {
            let count = material.textures.iter().count();
            if count > 0 {
                touched += 1;
                slots += count;
                material.textures = Default::default();
            }
        }
        if slots > 0 {
            warnings.add_for(
                format,
                WarningKind::DroppedFeature,
                "textures",
                format!("{ext} output does not support textures; dropped {slots} texture assignment(s) on {touched} material(s)"),
                slots,
            );
        }
    }

    if !caps.hierarchy && scene.nodes.len() > 1 {
        let links = scene.nodes.len().saturating_sub(1);
        warnings.add_for(
            format,
            WarningKind::DroppedFeature,
            "hierarchy",
            format!(
                "{ext} output does not support a node hierarchy; flattened {links} node link(s)"
            ),
            links,
        );
    }

    if !caps.units {
        let message = format!(
            "output.{} cannot store units; coordinates written in {}",
            format.extension(),
            target_unit.long_name()
        );
        match scene.unit {
            UnitResolution::Known(_) => {
                warnings.add_for(format, WarningKind::DroppedFeature, "units", message, 1);
            }
            UnitResolution::Unknown => {
                warnings.add_for(format, WarningKind::Note, "units", message, 1);
            }
        }
    }

    out
}
