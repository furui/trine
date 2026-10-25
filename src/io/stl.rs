//! STL reader and binary writer backed by `stl_io`.

use std::fs;
use std::path::Path;

use glam::Vec3;

use crate::error::{Error, Result};
use crate::io::{Format, Loaded, WriteOptions};
use crate::output::staging::Staging;
use crate::report::WarningSink;
use crate::scene::{Mesh, Primitive, Scene, UnitResolution, UpAxis};

pub fn read(path: &Path) -> Result<Loaded> {
    let mut file = fs::File::open(path).map_err(|error| Error::read(path, error))?;
    let mesh = stl_io::read_stl(&mut file)
        .map_err(|error| Error::path(path, format!("invalid STL: {error}")))?;
    if mesh.faces.is_empty() {
        return Err(Error::path(path, "STL contains no triangles"));
    }
    let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|vertex| vertex.0).collect();
    let mut indices = Vec::with_capacity(mesh.faces.len() * 3);
    for face in &mesh.faces {
        for index in face.vertices {
            indices.push(index as u32);
        }
    }
    let mut scene = Scene::new(Format::Stl);
    scene.unit = UnitResolution::Unknown;
    scene.up_axis = UpAxis::Z;
    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("STL")
        .to_string();
    let mut mesh_out = Mesh::new(name.clone());
    mesh_out.primitives.push(Primitive::new(positions, indices));
    let mesh_id = scene.add_mesh(mesh_out);
    let node = scene.add_node(name, glam::Mat4::IDENTITY);
    scene.nodes[node].mesh = Some(mesh_id);
    Ok(Loaded::new(scene))
}

pub fn write(
    options: &WriteOptions<'_>,
    staging: &mut Staging,
    warnings: &mut WarningSink,
) -> Result<()> {
    let scene = options.scene;
    if options.up_axis != Format::Stl.default_up_axis() {
        warnings.add_for(
            Format::Stl,
            crate::report::WarningKind::Note,
            "stl-up-axis",
            "STL has no up-axis metadata; wrote non-Z-up coordinates",
            1,
        );
    }
    let mut triangles: Vec<stl_io::Triangle> = Vec::new();
    for mesh_id in scene.referenced_meshes() {
        let mesh = &scene.meshes[mesh_id];
        for primitive in &mesh.primitives {
            for triangle in primitive.indices.chunks_exact(3) {
                let a = Vec3::from(primitive.positions[triangle[0] as usize]);
                let b = Vec3::from(primitive.positions[triangle[1] as usize]);
                let c = Vec3::from(primitive.positions[triangle[2] as usize]);
                let normal = (b - a).cross(c - a).try_normalize().unwrap_or(Vec3::ZERO);
                triangles.push(stl_io::Triangle {
                    normal: stl_io::Normal::new(normal.to_array()),
                    vertices: [
                        stl_io::Vertex::new(a.to_array()),
                        stl_io::Vertex::new(b.to_array()),
                        stl_io::Vertex::new(c.to_array()),
                    ],
                });
            }
        }
    }
    if triangles.is_empty() {
        return Err(Error::conversion("no triangles to write to STL"));
    }
    let temp_path = staging.stage(options.output_path)?;
    let file = fs::File::create(&temp_path).map_err(|error| Error::write(&temp_path, error))?;
    let mut writer = std::io::BufWriter::new(file);
    stl_io::write_stl(&mut writer, triangles.iter())
        .map_err(|error| Error::path(&temp_path, format!("cannot write STL: {error}")))?;
    Ok(())
}
