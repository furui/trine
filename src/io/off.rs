//! OFF / COFF reader and writer (custom, small and strict).

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{Error, Result};
use crate::io::{Format, Loaded, WriteOptions};
use crate::output::staging::Staging;
use crate::report::{WarningKind, WarningSink};
use crate::scene::{Mesh, Primitive, Scene, UnitResolution, UpAxis};

pub fn read(path: &Path) -> Result<Loaded> {
    let text = fs::read_to_string(path).map_err(|error| Error::read(path, error))?;
    let mut warnings = WarningSink::default();
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let mut lines = text
        .lines()
        .map(|line| line.split('#').next().unwrap_or("").trim())
        .filter(|line| !line.is_empty())
        .peekable();

    let Some(first_line) = lines.next() else {
        return Err(Error::path(path, "empty OFF file"));
    };
    let first_tokens: Vec<&str> = first_line.split_whitespace().collect();
    let mut is_coff = false;
    let mut is_normals = false;
    let mut is_cnoff = false;
    let mut header_tokens: Vec<String> = Vec::new();
    if let Some(first) = first_tokens.first() {
        let upper = first.to_ascii_uppercase();
        if upper.ends_with("OFF") {
            is_coff = upper.contains("COFF") || upper.contains("CNOFF");
            is_cnoff = upper.contains("CNOFF");
            is_normals = upper.contains("NOFF") && !is_coff;
            header_tokens = first_tokens.iter().skip(1).map(|s| s.to_string()).collect();
        } else {
            header_tokens = first_tokens.iter().map(|s| s.to_string()).collect();
        }
    }
    let counts: Vec<usize> = if header_tokens.len() >= 3 {
        parse_counts(&header_tokens, path)?
    } else {
        let Some(line) = lines.next() else {
            return Err(Error::path(path, "missing OFF element counts"));
        };
        let tokens: Vec<String> = line.split_whitespace().map(str::to_string).collect();
        parse_counts(&tokens, path)?
    };
    let (vertex_count, face_count, _edge_count) = (counts[0], counts[1], counts[2]);

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vertex_count);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(vertex_count);
    let mut has_colors = is_coff;
    for index in 0..vertex_count {
        let Some(line) = lines.next() else {
            return Err(Error::path(
                path,
                format!("expected {vertex_count} vertices, found {index}"),
            ));
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 3 {
            return Err(Error::path(
                path,
                format!("vertex {index} has fewer than three coordinates"),
            ));
        }
        let mut position = [0.0f32; 3];
        for axis in 0..3 {
            position[axis] = parse_f32(tokens[axis], path, line)?;
        }
        positions.push(position);
        if is_normals {
            colors.push([1.0, 1.0, 1.0, 1.0]);
            continue;
        }
        let color_offset = if is_cnoff { 6 } else { 3 };
        if tokens.len() >= color_offset + 3 {
            has_colors = true;
            let mut color = [1.0f32; 4];
            for channel in 0..3 {
                color[channel] = parse_color(tokens[color_offset + channel]);
            }
            if tokens.len() >= color_offset + 4 {
                color[3] = parse_color(tokens[color_offset + 3]);
            }
            colors.push(color);
        } else {
            colors.push([1.0, 1.0, 1.0, 1.0]);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity(face_count * 3);
    let mut ignored_face_colors = 0usize;
    for face in 0..face_count {
        let Some(line) = lines.next() else {
            return Err(Error::path(
                path,
                format!("expected {face_count} faces, found {face}"),
            ));
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some((&count_text, rest)) = tokens.split_first() else {
            return Err(Error::path(path, format!("face {face} is empty")));
        };
        let count: usize = count_text
            .parse()
            .map_err(|_| Error::path(path, format!("face {face} has an invalid vertex count")))?;
        if rest.len() < count || count < 3 {
            return Err(Error::path(
                path,
                format!(
                    "face {face} declares {count} vertices but has {} indices",
                    rest.len()
                ),
            ));
        }
        if rest.len() > count && !has_colors {
            ignored_face_colors += 1;
        }
        let mut face_indices = Vec::with_capacity(count);
        for token in &rest[..count] {
            let value: usize = token.parse().map_err(|_| {
                Error::path(path, format!("face {face} has an invalid index '{token}'"))
            })?;
            if value >= vertex_count {
                return Err(Error::path(
                    path,
                    format!("face {face} references vertex {value}, but only {vertex_count} exist"),
                ));
            }
            face_indices.push(value as u32);
        }
        indices.extend(crate::scene::triangulate::triangulate_face(
            &positions,
            &face_indices,
        ));
    }

    let mut scene = Scene::new(Format::Off);
    scene.unit = UnitResolution::Unknown;
    scene.up_axis = UpAxis::Z;
    let mut primitive = Primitive::new(positions, indices);
    if has_colors {
        primitive.colors = Some(colors);
    }
    let mut mesh = Mesh::new(
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("OFF"),
    );
    mesh.primitives.push(primitive);
    let mesh_id = scene.add_mesh(mesh);
    let node = scene.add_node(
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("OFF"),
        glam::Mat4::IDENTITY,
    );
    scene.nodes[node].mesh = Some(mesh_id);
    if is_normals || is_cnoff {
        warnings.add_for(
            Format::Off,
            WarningKind::DroppedFeature,
            "off-normals",
            format!("OFF input contains normals; dropped {vertex_count} vertex normal(s)"),
            vertex_count,
        );
    }
    if ignored_face_colors > 0 {
        warnings.add_for(
            Format::Off,
            WarningKind::DroppedFeature,
            "off-face-colors",
            format!("dropped per-face colors on {ignored_face_colors} face(s)"),
            ignored_face_colors,
        );
    }
    Ok(Loaded { scene, warnings })
}

fn parse_counts(tokens: &[String], path: &Path) -> Result<Vec<usize>> {
    if tokens.len() < 3 {
        return Err(Error::path(
            path,
            "OFF header needs vertex, face, and edge counts",
        ));
    }
    let mut counts = Vec::with_capacity(3);
    for token in &tokens[..3] {
        counts.push(
            token
                .parse()
                .map_err(|_| Error::path(path, format!("invalid OFF count '{token}'")))?,
        );
    }
    Ok(counts)
}

fn parse_f32(token: &str, path: &Path, line: &str) -> Result<f32> {
    let value: f32 = token
        .parse()
        .map_err(|_| Error::path(path, format!("invalid number in '{line}'")))?;
    if !value.is_finite() {
        return Err(Error::path(path, "non-finite coordinate"));
    }
    Ok(value)
}

fn parse_color(token: &str) -> f32 {
    match token.parse::<f32>() {
        Ok(value) if value > 1.0 => (value / 255.0).clamp(0.0, 1.0),
        Ok(value) => value.clamp(0.0, 1.0),
        Err(_) => 1.0,
    }
}

pub fn write(
    options: &WriteOptions<'_>,
    staging: &mut Staging,
    warnings: &mut WarningSink,
) -> Result<()> {
    let scene = options.scene;
    if options.up_axis != Format::Off.default_up_axis() {
        warnings.add_for(
            Format::Off,
            WarningKind::Note,
            "off-up-axis",
            "OFF has no up-axis metadata; wrote non-Z-up coordinates",
            1,
        );
    }
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut has_colors = false;
    for mesh_id in scene.referenced_meshes() {
        let mesh = &scene.meshes[mesh_id];
        for primitive in &mesh.primitives {
            let base = positions.len() as u32;
            positions.extend_from_slice(&primitive.positions);
            if let Some(primitive_colors) = &primitive.colors {
                has_colors = true;
                colors.extend_from_slice(primitive_colors);
            } else {
                colors.extend(std::iter::repeat_n(
                    [1.0, 1.0, 1.0, 1.0],
                    primitive.positions.len(),
                ));
            }
            indices.extend(primitive.indices.iter().map(|index| index + base));
        }
    }
    if positions.is_empty() {
        return Err(Error::conversion("no geometry to write to OFF"));
    }
    let temp_path = staging.stage(options.output_path)?;
    let file = fs::File::create(&temp_path).map_err(|error| Error::write(&temp_path, error))?;
    let mut writer = BufWriter::new(file);
    if has_colors {
        writeln!(writer, "COFF").map_err(|error| Error::write(&temp_path, error))?;
    } else {
        writeln!(writer, "OFF").map_err(|error| Error::write(&temp_path, error))?;
    }
    writeln!(writer, "{} {} 0", positions.len(), indices.len() / 3)
        .map_err(|error| Error::write(&temp_path, error))?;
    for (vertex, position) in positions.iter().enumerate() {
        if has_colors {
            let color = colors[vertex];
            writeln!(
                writer,
                "{} {} {} {} {} {} {}",
                fmt(position[0]),
                fmt(position[1]),
                fmt(position[2]),
                fmt(color[0]),
                fmt(color[1]),
                fmt(color[2]),
                fmt(color[3])
            )
            .map_err(|error| Error::write(&temp_path, error))?;
        } else {
            writeln!(
                writer,
                "{} {} {}",
                fmt(position[0]),
                fmt(position[1]),
                fmt(position[2])
            )
            .map_err(|error| Error::write(&temp_path, error))?;
        }
    }
    for triangle in indices.chunks_exact(3) {
        writeln!(writer, "3 {} {} {}", triangle[0], triangle[1], triangle[2])
            .map_err(|error| Error::write(&temp_path, error))?;
    }
    writer
        .flush()
        .map_err(|error| Error::write(&temp_path, error))?;
    if has_colors {
        warnings.add(
            crate::report::WarningKind::Note,
            "off-coff",
            "OFF output contains vertex colors and uses the COFF dialect",
        );
    }
    Ok(())
}

fn fmt(value: f32) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{value}")
    }
}
