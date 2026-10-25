//! Unit/axis/scale resolution and matrix composition.

use glam::{Mat3, Mat4, Quat, Vec3};

use crate::error::{Error, Result};
use crate::io::Format;
use crate::scene::{Scene, Unit, UnitResolution, UpAxis};

/// Unit handling resolved for one conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitPlan {
    /// Physical unit of the source numbers, if known or assumed. `None` with
    /// `--no-unit-conversion` when the source had no unit metadata.
    pub source_unit: Option<Unit>,
    /// Unit that the target format is told about / uses.
    pub target_unit: Unit,
    /// Multiplier applied to coordinates (`source_meters / target_meters`).
    pub factor: f64,
    /// The source unit came from the documented format default, not the file.
    pub assumed_source: bool,
    /// Unit metadata will be lost by the target format.
    pub target_loses_units: bool,
    pub conversion_enabled: bool,
}

impl UnitPlan {
    pub fn source_or_default(&self, default: Unit) -> Unit {
        self.source_unit.unwrap_or(default)
    }
}

/// Source unit default per format (used when a file has no unit metadata).
pub fn source_default_unit(format: Format) -> Unit {
    match format {
        Format::Fbx => Unit::Centimeter,
        Format::ThreeMf => Unit::Millimeter,
        Format::Step => Unit::Millimeter,
        Format::Dae => Unit::Meter,
        Format::Gltf | Format::Glb => Unit::Meter,
        Format::Stl | Format::Obj | Format::Ply | Format::Off => Unit::Millimeter,
    }
}

/// Unit that a target format uses when `--unit-out` is omitted.
pub fn target_default_unit(format: Format) -> Unit {
    match format {
        Format::Fbx => Unit::Centimeter,
        Format::Dae => Unit::Meter,
        Format::Gltf | Format::Glb => Unit::Meter,
        Format::ThreeMf | Format::Step | Format::Stl | Format::Obj | Format::Ply | Format::Off => {
            Unit::Millimeter
        }
    }
}

/// Resolve source and target units and the numeric conversion factor.
pub fn resolve_units(
    scene: UnitResolution,
    source_format: Format,
    target_format: Format,
    unit_in: Option<Unit>,
    unit_out: Option<Unit>,
    no_unit_conversion: bool,
) -> Result<UnitPlan> {
    if let Some(unit) = unit_out {
        if matches!(target_format, Format::Gltf | Format::Glb) && unit != Unit::Meter {
            return Err(Error::usage(format!(
                "{} output always uses meters; --unit-out {} is not allowed",
                target_format.extension(),
                unit
            )));
        }
    }

    let mut target_unit = unit_out.unwrap_or_else(|| target_default_unit(target_format));
    let source_default = source_default_unit(source_format);
    let source_unit = unit_in.or_else(|| scene.known());
    let assumed_source = unit_in.is_none() && scene.known().is_none() && !no_unit_conversion;

    // With conversion disabled, metadata (where the target can store it) must
    // describe the untouched raw numbers, so keep the source unit.
    if no_unit_conversion
        && unit_out.is_none()
        && !matches!(target_format, Format::Gltf | Format::Glb)
    {
        if let Some(source) = source_unit {
            target_unit = source;
        }
    }

    let resolved_source = source_unit.unwrap_or(source_default);
    let factor = if no_unit_conversion {
        1.0
    } else {
        resolved_source.meters_per_unit() / target_unit.meters_per_unit()
    };

    Ok(UnitPlan {
        source_unit: if no_unit_conversion {
            source_unit
        } else {
            Some(resolved_source)
        },
        target_unit,
        factor,
        assumed_source,
        target_loses_units: !target_format.stores_units(),
        conversion_enabled: !no_unit_conversion,
    })
}

/// Resolve units for a scene against a source and target format.
pub fn resolve_unit_plan_for_scene(
    scene: &Scene,
    source_format: Format,
    target_format: Format,
    unit_in: Option<Unit>,
    unit_out: Option<Unit>,
    no_unit_conversion: bool,
) -> Result<UnitPlan> {
    resolve_units(
        scene.unit,
        source_format,
        target_format,
        unit_in,
        unit_out,
        no_unit_conversion,
    )
}

/// Resolve the source and target up axes.
pub fn resolve_up_axes(
    scene_axis: UpAxis,
    source_override: Option<UpAxis>,
    target_override: Option<UpAxis>,
    keep_axes: bool,
    target_format: Format,
) -> (UpAxis, UpAxis) {
    let source = source_override.unwrap_or(scene_axis);
    let target = if keep_axes {
        source
    } else {
        target_override.unwrap_or_else(|| target_format.default_up_axis())
    };
    (source, target)
}

/// Rotation converting `source_up` to `target_up`.
pub fn axis_rotation(source_up: UpAxis, target_up: UpAxis) -> Mat4 {
    match (source_up, target_up) {
        (UpAxis::Z, UpAxis::Y) => Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2),
        (UpAxis::Y, UpAxis::Z) => Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
        _ => Mat4::IDENTITY,
    }
}

/// Build `G = S * R * U` where `S` is uniform/per-axis scale, `R` the axis
/// rotation, and `U` the unit factor.
pub fn compose_global(
    uniform_scale: f64,
    axis_scale: [f64; 3],
    unit_factor: f64,
    source_up: UpAxis,
    target_up: UpAxis,
) -> Mat4 {
    let scale = Vec3::new(
        (uniform_scale * axis_scale[0]) as f32,
        (uniform_scale * axis_scale[1]) as f32,
        (uniform_scale * axis_scale[2]) as f32,
    );
    let rotation = axis_rotation(source_up, target_up);
    let unit = unit_factor as f32;
    Mat4::from_scale(scale) * rotation * Mat4::from_scale(Vec3::splat(unit))
}

/// Parse a comma-separated per-axis scale such as `1,2,3`.
pub fn parse_scale_axes(text: &str) -> Result<[f64; 3]> {
    let parts: Vec<&str> = text.split(',').map(str::trim).collect();
    if parts.len() != 3 {
        return Err(Error::usage(format!(
            "--scale-axes expects three comma-separated numbers, got '{text}'"
        )));
    }
    let mut out = [1.0; 3];
    for (index, part) in parts.iter().enumerate() {
        let value: f64 = part
            .parse()
            .map_err(|_| Error::usage(format!("invalid --scale-axes component '{part}'")))?;
        if !value.is_finite() {
            return Err(Error::usage("--scale-axes components must be finite"));
        }
        out[index] = value;
    }
    Ok(out)
}

/// Normal matrix (inverse transpose of the upper 3x3) for transforming normals.
pub fn normal_matrix(matrix: Mat4) -> Mat3 {
    let upper = Mat3::from_mat4(matrix);
    if upper.determinant().abs() <= f32::EPSILON {
        return Mat3::IDENTITY;
    }
    upper.inverse().transpose()
}

/// Transform a normal and renormalize.
pub fn transform_normal(matrix: Mat4, normal: [f32; 3]) -> [f32; 3] {
    let transformed = normal_matrix(matrix) * Vec3::from(normal);
    let normalized = transformed.try_normalize().unwrap_or(Vec3::Y);
    normalized.to_array()
}

/// Transform a tangent, rotating the handedness sign when the matrix mirrors.
pub fn transform_tangent(matrix: Mat4, tangent: [f32; 4]) -> [f32; 4] {
    let transformed = Mat3::from_mat4(matrix) * Vec3::new(tangent[0], tangent[1], tangent[2]);
    let normalized = transformed.try_normalize().unwrap_or(Vec3::X);
    let mirrored = Mat3::from_mat4(matrix).determinant() < 0.0;
    let w = if mirrored { -tangent[3] } else { tangent[3] };
    [normalized.x, normalized.y, normalized.z, w]
}

/// Decompose a shear-free affine matrix into translation, rotation, and scale.
///
/// Returns `None` for matrices containing shear or mirroring; callers bake
/// such transforms instead of writing an incorrect TRS triple.
pub fn decompose_trs(matrix: Mat4) -> Option<(Vec3, Quat, Vec3)> {
    let translation = matrix.w_axis.truncate();
    let x = matrix.x_axis.truncate();
    let y = matrix.y_axis.truncate();
    let z = matrix.z_axis.truncate();
    let sx = x.length();
    let sy = y.length();
    let sz = z.length();
    if sx <= 1e-9 || sy <= 1e-9 || sz <= 1e-9 {
        return None;
    }
    let xn = x / sx;
    let yn = y / sy;
    let zn = z / sz;
    let shear_tolerance = 1e-4;
    if xn.dot(yn).abs() > shear_tolerance
        || xn.dot(zn).abs() > shear_tolerance
        || yn.dot(zn).abs() > shear_tolerance
    {
        return None;
    }
    let rotation_matrix = Mat3::from_cols(xn, yn, zn);
    if rotation_matrix.determinant() <= 0.0 {
        return None;
    }
    let rotation = Quat::from_mat3(&rotation_matrix);
    if !rotation.is_finite() {
        return None;
    }
    Some((translation, rotation.normalize(), Vec3::new(sx, sy, sz)))
}

/// A single output node transform: retained hierarchy or baked world space.
#[derive(Debug, Clone, Copy)]
pub enum TransformMode {
    /// Keep hierarchy; multiply root transforms by the global matrix only.
    Hierarchy,
    /// Bake world transforms into vertex data.
    Flatten,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn assert_vec3(actual: Vec3, expected: Vec3, epsilon: f32) {
        assert!(
            (actual - expected).length() <= epsilon,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn axis_rotation_z_to_y_maps_up_vector() {
        let rotation = axis_rotation(UpAxis::Z, UpAxis::Y);
        let up = rotation.transform_point3(Vec3::Z);
        assert_vec3(up, Vec3::Y, 1e-6);
    }

    #[test]
    fn axis_rotation_y_to_z_maps_up_vector() {
        let rotation = axis_rotation(UpAxis::Y, UpAxis::Z);
        let up = rotation.transform_point3(Vec3::Y);
        assert_vec3(up, Vec3::Z, 1e-6);
    }

    #[test]
    fn compose_scales_and_units() {
        let matrix = compose_global(2.0, [1.0, 3.0, 1.0], 0.001, UpAxis::Y, UpAxis::Y);
        let point = matrix.transform_point3(Vec3::new(1.0, 1.0, 1.0));
        assert_vec3(point, Vec3::new(0.002, 0.006, 0.002), 1e-9);
    }

    #[test]
    fn unit_defaults() {
        let plan = resolve_units(
            UnitResolution::Unknown,
            Format::Stl,
            Format::Glb,
            None,
            None,
            false,
        )
        .unwrap();
        assert_eq!(plan.source_unit, Some(Unit::Millimeter));
        assert_eq!(plan.target_unit, Unit::Meter);
        assert_relative_eq!(plan.factor, 1e-3, epsilon = 1e-12);
        assert!(plan.assumed_source);
    }

    #[test]
    fn explicit_units_override_metadata() {
        let plan = resolve_units(
            UnitResolution::Known(Unit::Meter),
            Format::Obj,
            Format::Stl,
            Some(Unit::Centimeter),
            Some(Unit::Inch),
            false,
        )
        .unwrap();
        assert_eq!(plan.source_unit, Some(Unit::Centimeter));
        assert_eq!(plan.target_unit, Unit::Inch);
        assert_relative_eq!(plan.factor, 1.0 / 2.54, epsilon = 1e-12);
        assert!(!plan.assumed_source);
    }

    #[test]
    fn no_conversion_preserves_raw_numbers() {
        let plan = resolve_units(
            UnitResolution::Known(Unit::Millimeter),
            Format::Obj,
            Format::Step,
            None,
            None,
            true,
        )
        .unwrap();
        assert_relative_eq!(plan.factor, 1.0);
        assert!(!plan.assumed_source);
    }

    #[test]
    fn gltf_rejects_non_meter_output_unit() {
        let error = resolve_units(
            UnitResolution::Unknown,
            Format::Obj,
            Format::Glb,
            None,
            Some(Unit::Millimeter),
            false,
        )
        .unwrap_err();
        assert!(error.is_usage());
    }

    #[test]
    fn trs_roundtrip() {
        let translation = Vec3::new(1.0, 2.0, 3.0);
        let rotation = Quat::from_rotation_y(0.7);
        let scale = Vec3::new(1.0, 2.0, 3.0);
        let matrix = Mat4::from_scale_rotation_translation(scale, rotation, translation);
        let (t, r, s) = decompose_trs(matrix).expect("shear-free");
        assert_vec3(t, translation, 1e-6);
        assert_vec3(s, scale, 1e-6);
        assert_relative_eq!(r.dot(rotation).abs(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn shear_is_rejected() {
        let mut matrix = Mat4::from_scale(Vec3::new(1.0, 2.0, 1.0));
        matrix.y_axis.x = 1.0;
        assert!(decompose_trs(matrix).is_none());
    }
}
