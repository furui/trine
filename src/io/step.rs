//! STEP (AP214) import via `truck-stepio` and faceted AP214 export.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use glam::{Mat4, Vec3};

use crate::error::{Error, Result};
use crate::io::{Format, Loaded, WriteOptions};
use crate::output::staging::Staging;
use crate::report::{WarningKind, WarningSink};
use crate::scene::{Mesh, Primitive, Scene, Unit, UnitResolution, UpAxis};

pub fn read(path: &Path, tessellation_tolerance: Option<f64>) -> Result<Loaded> {
    use truck_meshalgo::prelude::*;
    use truck_stepio::r#in::Table;

    let text = std::fs::read_to_string(path).map_err(|error| Error::read(path, error))?;
    let exchange = ruststep::parser::parse(&text)
        .map_err(|error| Error::path(path, format!("invalid STEP: {error:?}")))?;
    let Some(data_section) = exchange.data.first() else {
        return Err(Error::path(path, "STEP file has no DATA section"));
    };
    let table = Table::from_data_section(data_section);
    let mut warnings = WarningSink::default();
    let mut scene = Scene::new(Format::Step);
    scene.unit = match detect_unit(&text) {
        Some(unit) => UnitResolution::Known(unit),
        None => UnitResolution::Unknown,
    };
    scene.up_axis = UpAxis::Z;

    let mut shell_ids: Vec<u64> = table.shell.keys().copied().collect();
    shell_ids.sort_unstable();
    let mut tessellation_warned = false;
    let mut failed_shells = 0usize;
    for shell_id in shell_ids {
        let Some(shell) = table.shell.get(&shell_id) else {
            continue;
        };
        let compressed = match table.to_compressed_shell(shell) {
            Ok(compressed) => compressed,
            Err(_) => {
                failed_shells += 1;
                continue;
            }
        };
        let preview = compressed.robust_triangulation(0.01).to_polygon();
        let diagonal = preview.bounding_box().diameter();
        let tolerance = tessellation_tolerance.unwrap_or_else(|| (diagonal * 0.001).max(1e-6));
        let tessellated = compressed.robust_triangulation(tolerance);
        let polygon = tessellated.to_polygon();
        if !tessellation_warned {
            warnings.add(
                WarningKind::Approximation,
                "step-tessellation",
                format!(
                    "STEP input was tessellated (tolerance {tolerance:.6}); curved surfaces are approximated by triangles"
                ),
            );
            tessellation_warned = true;
        }

        let positions: Vec<[f32; 3]> = polygon
            .positions()
            .iter()
            .map(|point| [point.x as f32, point.y as f32, point.z as f32])
            .collect();
        if positions.is_empty() {
            continue;
        }
        let mut indices = Vec::new();
        for face in polygon.face_iter() {
            let face_indices: Vec<u32> = face.iter().map(|vertex| vertex.pos as u32).collect();
            for index in 1..face_indices.len().saturating_sub(1) {
                indices.extend([
                    face_indices[0],
                    face_indices[index],
                    face_indices[index + 1],
                ]);
            }
        }
        if indices.is_empty() {
            continue;
        }
        let mut mesh = Mesh::new(format!("shell_{shell_id}"));
        mesh.primitives.push(Primitive::new(positions, indices));
        let mesh_id = scene.add_mesh(mesh);
        let node = scene.add_node(format!("shell_{shell_id}"), Mat4::IDENTITY);
        scene.nodes[node].mesh = Some(mesh_id);
    }
    if failed_shells > 0 {
        warnings.add_for(
            Format::Step,
            WarningKind::UnsupportedEntity,
            "step-shell",
            format!("skipped {failed_shells} STEP shell(s) that could not be tessellated"),
            failed_shells,
        );
    }
    if scene.referenced_triangle_count() == 0 {
        return Err(Error::path(path, "STEP contains no tessellatable shells"));
    }
    Ok(Loaded { scene, warnings })
}

fn detect_unit(text: &str) -> Option<Unit> {
    let compact: String = text
        .to_ascii_uppercase()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();
    let upper = compact.as_str();
    if upper.contains("CONVERSION_BASED_UNIT('INCH'")
        || upper.contains("CONVERSION_BASED_UNIT(\"INCH\"")
    {
        return Some(Unit::Inch);
    }
    if upper.contains("CONVERSION_BASED_UNIT('FOOT'")
        || upper.contains("CONVERSION_BASED_UNIT(\"FOOT\"")
    {
        return Some(Unit::Foot);
    }
    if upper.contains(".MILLI.,.METRE.") {
        return Some(Unit::Millimeter);
    }
    if upper.contains(".CENTI.,.METRE.") {
        return Some(Unit::Centimeter);
    }
    if upper.contains(".MICRO.,.METRE.") {
        return Some(Unit::Micrometer);
    }
    if upper.contains("SI_UNIT($,.METRE.)") || upper.contains("SI_UNIT(*,.METRE.)") {
        return Some(Unit::Meter);
    }
    None
}

struct StepWriter {
    next: u64,
    data: String,
}

impl StepWriter {
    fn new() -> Self {
        Self {
            next: 1,
            data: String::new(),
        }
    }

    fn id(&mut self) -> u64 {
        let id = self.next;
        self.next += 1;
        id
    }

    fn push(&mut self, id: u64, body: impl std::fmt::Display) {
        writeln!(self.data, "#{id}={body};").unwrap();
    }
}

pub fn write(
    options: &WriteOptions<'_>,
    staging: &mut Staging,
    warnings: &mut WarningSink,
) -> Result<()> {
    let scene = options.scene;
    if scene.referenced_triangle_count() == 0 {
        return Err(Error::conversion("no triangles to write to STEP"));
    }
    if options.up_axis == UpAxis::Y {
        warnings.add_for(
            Format::Step,
            WarningKind::Note,
            "step-up-axis",
            "STEP has no up-axis metadata; wrote Y-up coordinates by convention",
            1,
        );
    }

    let mut writer = StepWriter::new();
    let app_context = writer.id();
    writer.push(app_context, "APPLICATION_CONTEXT('automotive design')");
    let protocol = writer.id();
    writer.push(
        protocol,
        format!("APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app_context})"),
    );
    let product_context = writer.id();
    writer.push(
        product_context,
        format!("PRODUCT_CONTEXT('',#{app_context},'mechanical')"),
    );
    let product = writer.id();
    writer.push(
        product,
        format!("PRODUCT('trine','trine','',(#{product_context}))"),
    );
    let category = writer.id();
    writer.push(
        category,
        format!("PRODUCT_RELATED_PRODUCT_CATEGORY('part','',(#{product}))"),
    );
    let formation = writer.id();
    writer.push(
        formation,
        format!("PRODUCT_DEFINITION_FORMATION('','',#{product})"),
    );
    let definition_context = writer.id();
    writer.push(
        definition_context,
        format!("PRODUCT_DEFINITION_CONTEXT('part definition',#{app_context},'design')"),
    );
    let definition = writer.id();
    writer.push(
        definition,
        format!("PRODUCT_DEFINITION('design','',#{formation},#{definition_context})"),
    );
    let definition_shape = writer.id();
    writer.push(
        definition_shape,
        format!("PRODUCT_DEFINITION_SHAPE('','',#{definition})"),
    );

    let unit_id = write_units(&mut writer, options.unit);
    let context = writer.id();
    writer.push(
        context,
        format!(
            "(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNIT_ASSIGNED_CONTEXT((#{unit_id}))REPRESENTATION_CONTEXT('',''))"
        ),
    );

    let origin_point = writer.id();
    writer.push(origin_point, "CARTESIAN_POINT('',(0.,0.,0.))");
    let z_direction = writer.id();
    writer.push(z_direction, "DIRECTION('',(0.,0.,1.))");
    let x_direction = writer.id();
    writer.push(x_direction, "DIRECTION('',(1.,0.,0.))");
    let root_axis = writer.id();
    writer.push(
        root_axis,
        format!("AXIS2_PLACEMENT_3D('',#{origin_point},#{z_direction},#{x_direction})"),
    );

    let mut representation_items: Vec<u64> = vec![root_axis];
    let mut open_shells = 0usize;
    let mut solids = 0usize;
    for mesh_index in scene.referenced_meshes() {
        let mesh = &scene.meshes[mesh_index];
        for (primitive_index, primitive) in mesh.primitives.iter().enumerate() {
            if primitive.indices.is_empty() {
                continue;
            }
            let mut weld_ids: Vec<u32> = Vec::with_capacity(primitive.positions.len());
            let mut weld_map: HashMap<(u32, u32, u32), u32> = HashMap::new();
            for position in &primitive.positions {
                let key = (
                    position[0].to_bits(),
                    position[1].to_bits(),
                    position[2].to_bits(),
                );
                let next = weld_map.len() as u32;
                let id = *weld_map.entry(key).or_insert(next);
                weld_ids.push(id);
            }
            let welded_indices: Vec<u32> = primitive
                .indices
                .iter()
                .map(|index| weld_ids[*index as usize])
                .collect();
            let components = crate::scene::validate::connected_components(&welded_indices);
            for (component_index, component) in components.iter().enumerate() {
                let mut indices = Vec::with_capacity(component.len() * 3);
                for triangle in component {
                    let base = triangle * 3;
                    indices.extend_from_slice(&primitive.indices[base..base + 3]);
                }
                let label = format!("{mesh_index}_{primitive_index}_{component_index}");
                match write_primitive_shell(&mut writer, primitive, &indices, &label) {
                    Ok(Some((item, closed))) => {
                        representation_items.push(item);
                        if closed {
                            solids += 1;
                        } else {
                            open_shells += 1;
                        }
                    }
                    Ok(None) => {}
                    Err(message) => {
                        warnings.add(
                            WarningKind::UnsupportedEntity,
                            "step-primitive",
                            format!("STEP output skipped a component: {message}"),
                        );
                    }
                }
            }
        }
    }
    if representation_items.len() <= 1 {
        return Err(Error::conversion("no valid triangles to write to STEP"));
    }
    let representation = writer.id();
    let items: Vec<String> = representation_items
        .iter()
        .map(|id| format!("#{id}"))
        .collect();
    writer.push(
        representation,
        format!(
            "ADVANCED_BREP_SHAPE_REPRESENTATION('',({}),#{context})",
            items.join(",")
        ),
    );
    let shape_rep = writer.id();
    writer.push(
        shape_rep,
        format!("SHAPE_DEFINITION_REPRESENTATION(#{definition_shape},#{representation})"),
    );

    let file_name = options
        .output_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("trine.step");
    let mut output = String::new();
    writeln!(output, "ISO-10303-21;").unwrap();
    writeln!(output, "HEADER;").unwrap();
    writeln!(output, "FILE_DESCRIPTION(('trine faceted mesh'),'2;1');").unwrap();
    writeln!(
        output,
        "FILE_NAME('{}','2026-01-01T00:00:00',(''),(''),'trine','trine','');",
        escape(file_name)
    )
    .unwrap();
    writeln!(
        output,
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN {{ 1 0 10303 214 1 1 1 1 }}'));"
    )
    .unwrap();
    writeln!(output, "ENDSEC;").unwrap();
    writeln!(output, "DATA;").unwrap();
    output.push_str(&writer.data);
    writeln!(output, "ENDSEC;").unwrap();
    writeln!(output, "END-ISO-10303-21;").unwrap();

    if open_shells > 0 {
        warnings.add_for(
            Format::Step,
            WarningKind::Note,
            "step-open-shell",
            format!(
                "{open_shells} primitive(s) are not closed solids; wrote SHELL_BASED_SURFACE_MODEL"
            ),
            open_shells,
        );
    }
    let _ = solids;

    let temp_path = staging.stage(options.output_path)?;
    std::fs::write(&temp_path, output).map_err(|error| Error::write(&temp_path, error))?;
    Ok(())
}

fn write_units(writer: &mut StepWriter, unit: Unit) -> u64 {
    match unit {
        Unit::Millimeter => {
            let id = writer.id();
            writer.push(id, "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))");
            id
        }
        Unit::Centimeter => {
            let id = writer.id();
            writer.push(id, "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.CENTI.,.METRE.))");
            id
        }
        Unit::Micrometer => {
            let id = writer.id();
            writer.push(id, "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MICRO.,.METRE.))");
            id
        }
        Unit::Meter => {
            let id = writer.id();
            writer.push(id, "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))");
            id
        }
        Unit::Inch | Unit::Foot => {
            let (name, millimeters) = if unit == Unit::Inch {
                ("INCH", 25.4)
            } else {
                ("FOOT", 304.8)
            };
            let base = writer.id();
            writer.push(base, "(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))");
            let dimensions = writer.id();
            writer.push(dimensions, "DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)");
            let measure = writer.id();
            writer.push(
                measure,
                format!("LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE({millimeters}),#{base})"),
            );
            let conversion = writer.id();
            writer.push(
                conversion,
                format!(
                    "(CONVERSION_BASED_UNIT('{name}',#{measure})LENGTH_UNIT()NAMED_UNIT(#{dimensions}))"
                ),
            );
            conversion
        }
    }
}

fn write_primitive_shell(
    writer: &mut StepWriter,
    primitive: &Primitive,
    indices: &[u32],
    label: &str,
) -> std::result::Result<Option<(u64, bool)>, String> {
    let positions = &primitive.positions;
    let mut vertex_points: HashMap<(u32, u32, u32), (u64, u64, u32)> = HashMap::new();
    let mut dedup_indices: Vec<u32> = Vec::with_capacity(positions.len());
    let mut dedup_vertex_entities: Vec<u64> = Vec::new();
    let mut dedup_cartesians: Vec<u64> = Vec::new();
    let mut dedup_positions: Vec<[f32; 3]> = Vec::new();
    for position in positions {
        let key = (
            position[0].to_bits(),
            position[1].to_bits(),
            position[2].to_bits(),
        );
        let (_, _, dedup) = match vertex_points.get(&key) {
            Some(ids) => *ids,
            None => {
                let cartesian = writer.id();
                writer.push(
                    cartesian,
                    format!(
                        "CARTESIAN_POINT('',({},{},{}))",
                        number64(position[0] as f64),
                        number64(position[1] as f64),
                        number64(position[2] as f64)
                    ),
                );
                let vertex = writer.id();
                writer.push(vertex, format!("VERTEX_POINT('',#{cartesian})"));
                let dedup = dedup_vertex_entities.len() as u32;
                dedup_vertex_entities.push(vertex);
                dedup_cartesians.push(cartesian);
                dedup_positions.push(*position);
                vertex_points.insert(key, (vertex, cartesian, dedup));
                (vertex, cartesian, dedup)
            }
        };
        dedup_indices.push(dedup);
    }

    let mut edges: HashMap<(u32, u32), u64> = HashMap::new();
    let mut faces: Vec<u64> = Vec::new();
    let mut kept_indices: Vec<u32> = Vec::new();
    for triangle in indices.chunks_exact(3) {
        let (a, b, c) = (triangle[0], triangle[1], triangle[2]);
        let pa = Vec3::from(positions[a as usize]);
        let pb = Vec3::from(positions[b as usize]);
        let pc = Vec3::from(positions[c as usize]);
        let normal = (pb - pa).cross(pc - pa);
        if normal.length_squared() <= 1e-20 {
            continue;
        }
        let (da, db, dc) = (
            dedup_indices[a as usize],
            dedup_indices[b as usize],
            dedup_indices[c as usize],
        );
        if da == db || db == dc || dc == da {
            continue;
        }
        kept_indices.extend([da, db, dc]);
        let mut oriented = Vec::new();
        for (from, to) in [(da, db), (db, dc), (dc, da)] {
            let key = if from < to { (from, to) } else { (to, from) };
            let edge = match edges.get(&key) {
                Some(id) => *id,
                None => {
                    let origin = dedup_cartesians[key.0 as usize];
                    let p_from = Vec3::from(dedup_positions[key.0 as usize]);
                    let p_to = Vec3::from(dedup_positions[key.1 as usize]);
                    let delta = p_to - p_from;
                    let length = delta.length();
                    if length <= 1e-12 {
                        return Err("zero-length edge".into());
                    }
                    let direction = (delta / length).to_array();
                    let dir_id = writer.id();
                    writer.push(
                        dir_id,
                        format!(
                            "DIRECTION('',({},{},{}))",
                            number64(direction[0] as f64),
                            number64(direction[1] as f64),
                            number64(direction[2] as f64)
                        ),
                    );
                    let vector = writer.id();
                    writer.push(
                        vector,
                        format!("VECTOR('',#{dir_id},{})", number64(length as f64)),
                    );
                    let line = writer.id();
                    writer.push(line, format!("LINE('',#{origin},#{vector})"));
                    let edge_id = writer.id();
                    writer.push(
                        edge_id,
                        format!(
                            "EDGE_CURVE('',#{},#{},#{line},.T.)",
                            dedup_vertex_entities[key.0 as usize],
                            dedup_vertex_entities[key.1 as usize]
                        ),
                    );
                    edges.insert(key, edge_id);
                    edge_id
                }
            };
            let orientation = if from < to { ".T." } else { ".F." };
            let oriented_id = writer.id();
            writer.push(
                oriented_id,
                format!("ORIENTED_EDGE('',*,*,#{edge},{orientation})"),
            );
            oriented.push(oriented_id);
        }
        let loop_id = writer.id();
        let oriented_refs: Vec<String> = oriented.iter().map(|id| format!("#{id}")).collect();
        writer.push(
            loop_id,
            format!("EDGE_LOOP('',({}))", oriented_refs.join(",")),
        );
        let bound = writer.id();
        writer.push(bound, format!("FACE_OUTER_BOUND('',#{loop_id},.T.)"));

        let unit_normal = normal.normalize();
        let ref_direction = (pb - pa).try_normalize().unwrap_or(Vec3::X);
        let origin = writer.id();
        writer.push(
            origin,
            format!(
                "CARTESIAN_POINT('',({},{},{}))",
                number64(pa.x as f64),
                number64(pa.y as f64),
                number64(pa.z as f64)
            ),
        );
        let axis_dir = writer.id();
        writer.push(
            axis_dir,
            format!(
                "DIRECTION('',({},{},{}))",
                number64(unit_normal.x as f64),
                number64(unit_normal.y as f64),
                number64(unit_normal.z as f64)
            ),
        );
        let ref_dir = writer.id();
        writer.push(
            ref_dir,
            format!(
                "DIRECTION('',({},{},{}))",
                number64(ref_direction.x as f64),
                number64(ref_direction.y as f64),
                number64(ref_direction.z as f64)
            ),
        );
        let placement = writer.id();
        writer.push(
            placement,
            format!("AXIS2_PLACEMENT_3D('',#{origin},#{axis_dir},#{ref_dir})"),
        );
        let plane = writer.id();
        writer.push(plane, format!("PLANE('',#{placement})"));
        let face = writer.id();
        writer.push(face, format!("ADVANCED_FACE('',(#{bound}),#{plane},.T.)"));
        faces.push(face);
    }
    if faces.is_empty() {
        return Ok(None);
    }
    let closed = crate::scene::validate::topology(&kept_indices).closed;
    let face_refs: Vec<String> = faces.iter().map(|id| format!("#{id}")).collect();
    let shell = writer.id();
    if closed {
        writer.push(
            shell,
            format!("CLOSED_SHELL('{label}',({}))", face_refs.join(",")),
        );
        let solid = writer.id();
        writer.push(solid, format!("MANIFOLD_SOLID_BREP('{label}',#{shell})"));
        Ok(Some((solid, true)))
    } else {
        writer.push(
            shell,
            format!("OPEN_SHELL('{label}',({}))", face_refs.join(",")),
        );
        let model = writer.id();
        writer.push(
            model,
            format!("SHELL_BASED_SURFACE_MODEL('{label}',(#{shell}))"),
        );
        Ok(Some((model, false)))
    }
}

fn number64(value: f64) -> String {
    if !value.is_finite() || value == 0.0 {
        "0.".to_string()
    } else {
        let text = format!("{value:.9}");
        let trimmed = text.trim_end_matches('0').trim_end_matches('.');
        if trimmed.contains('.') {
            trimmed.to_string()
        } else {
            format!("{trimmed}.")
        }
    }
}

fn escape(text: &str) -> String {
    text.replace('\'', "''")
}
