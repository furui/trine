use std::path::Path;

use glam::Mat4;
use trine::cli::Options;
use trine::convert;
use trine::io::{self, Format, WriteOptions};
use trine::ops::gpu::GpuMode;
use trine::ops::normals::NormalMode;
use trine::output::staging::Staging;
use trine::output::textures::TexturePlan;
use trine::report::WarningSink;
use trine::scene::{Mesh, Primitive, Scene, Unit, UnitResolution, UpAxis};

fn cube(unit: Unit) -> Scene {
    let mut scene = Scene::new(Format::Obj);
    scene.unit = UnitResolution::Known(unit);
    scene.up_axis = UpAxis::Y;
    let positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let indices = vec![
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 6, 2, 3, 7, 6, 0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    ];
    let mut mesh = Mesh::new("Cube");
    mesh.primitives.push(Primitive::new(positions, indices));
    let mesh_id = scene.add_mesh(mesh);
    let node = scene.add_node("Cube", Mat4::IDENTITY);
    scene.nodes[node].mesh = Some(mesh_id);
    scene
}

fn write_scene(scene: &Scene, path: &Path, format: Format) {
    let mut staging = Staging::new(path.parent().unwrap()).unwrap();
    let mut textures = TexturePlan::new();
    let mut warnings = WarningSink::new(true);
    io::write(
        format,
        &WriteOptions {
            scene,
            output_path: path,
            texture_dir: "tex",
            unit: scene.unit.known().unwrap_or(Unit::Millimeter),
            up_axis: scene.up_axis,
            geometry_only: false,
        },
        &mut staging,
        &mut textures,
        &mut warnings,
    )
    .unwrap();
    staging.finalize(true).unwrap();
}

fn options(input: &Path, output: &Path) -> Options {
    Options {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        from: None,
        to: None,
        split_groups: false,
        scale: 1.0,
        scale_axes: [1.0, 1.0, 1.0],
        unit_in: None,
        unit_out: None,
        no_unit_conversion: false,
        up_axis: None,
        source_up_axis: None,
        keep_axes: false,
        normals: NormalMode::Preserve,
        gpu: GpuMode::Off,
        tessellation_tolerance: None,
        force: true,
        strict: false,
        quiet: true,
        verbose: false,
    }
}

#[test]
fn every_format_converts_to_glb() {
    let dir = tempfile::tempdir().unwrap();
    for format in Format::ALL {
        let source = dir.path().join(format!("source.{}", format.extension()));
        let unit = if matches!(format, Format::Gltf | Format::Glb) {
            Unit::Meter
        } else {
            Unit::Millimeter
        };
        write_scene(&cube(unit), &source, format);
        let output = dir.path().join(format!("to-{}.glb", format.extension()));
        convert::run(&options(&source, &output)).unwrap();
        let scene = io::read(&output, Format::Glb, None).unwrap().scene;
        assert_eq!(
            scene.triangle_count(),
            12,
            "converting {format} to GLB lost triangles"
        );
    }
}

#[test]
fn glb_converts_to_every_format() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.glb");
    write_scene(&cube(Unit::Meter), &source, Format::Glb);
    for format in Format::ALL {
        if matches!(format, Format::Gltf | Format::Glb) {
            continue;
        }
        let output = dir.path().join(format!("out.{}", format.extension()));
        convert::run(&options(&source, &output)).unwrap();
        let scene = io::read(&output, format, None).unwrap().scene;
        assert_eq!(
            scene.triangle_count(),
            12,
            "converting GLB to {format} lost triangles"
        );
    }
}

#[test]
fn every_format_accepts_a_second_conversion() {
    let dir = tempfile::tempdir().unwrap();
    let glb = dir.path().join("source.glb");
    write_scene(&cube(Unit::Meter), &glb, Format::Glb);
    for format in Format::ALL {
        let intermediate = dir.path().join(format!("mid.{}", format.extension()));
        convert::run(&options(&glb, &intermediate)).unwrap();
        let final_path = dir.path().join(format!("final-{}.glb", format.extension()));
        convert::run(&options(&intermediate, &final_path)).unwrap();
        let scene = io::read(&final_path, Format::Glb, None).unwrap().scene;
        assert_eq!(scene.triangle_count(), 12, "second conversion via {format}");
    }
}
