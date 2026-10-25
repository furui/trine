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
use trine::scene::{
    AlphaMode, Image, Material, Mesh, Primitive, Scene, TextureRef, Unit, UnitResolution, UpAxis,
};

fn cube_positions() -> Vec<[f32; 3]> {
    vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ]
}

fn cube_indices() -> Vec<u32> {
    vec![
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 6, 2, 3, 7, 6, 0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    ]
}

fn fixture() -> Scene {
    let mut scene = Scene::new(Format::Obj);
    scene.unit = UnitResolution::Known(Unit::Millimeter);
    scene.up_axis = UpAxis::Y;
    let image = scene.add_image(Image {
        name: "wood.png".into(),
        bytes: vec![
            0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, b'I', b'H', b'D', b'R',
        ],
        mime: Some("image/png".into()),
        source_path: None,
    });
    let mut material = Material::new("Red");
    material.base_color = [0.8, 0.1, 0.1, 1.0];
    material.metallic = 0.2;
    material.roughness = 0.4;
    material.alpha_mode = AlphaMode::Opaque;
    material.textures.base_color = Some(TextureRef { image });
    let material_id = scene.add_material(material);

    let positions = cube_positions();
    let mut primitive = Primitive::new(positions.clone(), cube_indices());
    primitive.normals = Some(
        positions
            .iter()
            .map(|position| {
                let v = glam::Vec3::from(*position);
                v.try_normalize().unwrap_or(glam::Vec3::Y).to_array()
            })
            .collect(),
    );
    primitive.uvs.push(
        positions
            .iter()
            .map(|position| [position[0] + 0.5, position[1] + 0.5])
            .collect(),
    );
    primitive.colors = Some(positions.iter().map(|_| [1.0, 1.0, 1.0, 1.0]).collect());
    primitive.material = Some(material_id);
    primitive.tangents = trine::ops::tangents::generate_tangents(&primitive);
    let mut mesh = Mesh::new("Cube");
    mesh.primitives.push(primitive);
    let mesh_id = scene.add_mesh(mesh);

    let root = scene.add_node(
        "Root",
        Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0)),
    );
    scene.nodes[root].mesh = Some(mesh_id);
    let child = scene.add_child(root, "Child", Mat4::IDENTITY);
    let positions: Vec<[f32; 3]> = cube_positions()
        .into_iter()
        .map(|mut position| {
            position[0] += 2.0;
            position
        })
        .collect();
    let mut second = Primitive::new(positions.clone(), cube_indices());
    second.normals = Some(
        positions
            .iter()
            .map(|position| {
                let v = glam::Vec3::from(*position);
                v.try_normalize().unwrap_or(glam::Vec3::Y).to_array()
            })
            .collect(),
    );
    second.uvs.push(
        positions
            .iter()
            .map(|position| [position[0] + 0.5, position[1] + 0.5])
            .collect(),
    );
    second.colors = Some(positions.iter().map(|_| [1.0, 1.0, 1.0, 1.0]).collect());
    second.material = Some(material_id);
    second.tangents = trine::ops::tangents::generate_tangents(&second);
    let mut mesh = Mesh::new("Cube2");
    mesh.primitives.push(second);
    let mesh_id = scene.add_mesh(mesh);
    scene.nodes[child].mesh = Some(mesh_id);
    scene
}

fn simple_scene(unit: Unit) -> Scene {
    let mut scene = Scene::new(Format::Obj);
    scene.unit = UnitResolution::Known(unit);
    scene.up_axis = UpAxis::Y;
    let positions = cube_positions();
    let primitive = Primitive::new(positions, cube_indices());
    let mut mesh = Mesh::new("Cube");
    mesh.primitives.push(primitive);
    let mesh_id = scene.add_mesh(mesh);
    let node = scene.add_node("Cube", Mat4::IDENTITY);
    scene.nodes[node].mesh = Some(mesh_id);
    scene
}

fn write_and_read(scene: &Scene, format: Format) -> Scene {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().to_path_buf();
    let path = base.join(format!("roundtrip.{}", format.extension()));
    let mut staging = Staging::new(&base).unwrap();
    let mut textures = TexturePlan::new();
    let mut warnings = WarningSink::new(true);
    let options = WriteOptions {
        scene,
        output_path: &path,
        texture_dir: "roundtrip.textures",
        unit: Unit::Millimeter,
        up_axis: scene.up_axis,
        geometry_only: false,
    };
    // Flattened writers expect baked transforms; provide a scene with identity
    // transforms by baking the hierarchy first.
    let mut prepared = scene.clone();
    if !format.capabilities().hierarchy {
        trine::ops::flatten::bake_flattened(&mut prepared, Mat4::IDENTITY, None);
    }
    let options = WriteOptions {
        scene: &prepared,
        ..options
    };
    io::write(format, &options, &mut staging, &mut textures, &mut warnings).unwrap();
    staging.finalize(true).unwrap();
    let loaded = io::read(&path, format, None).unwrap();
    loaded.scene
}

fn assert_fixture_bounds(scene: &Scene) {
    let (min, max) = bounds(scene);
    assert!((min.x - 0.5).abs() < 1e-4, "min {min:?}");
    assert!((max.x - 3.5).abs() < 1e-4, "max {max:?}");
    assert!((min.y + 0.5).abs() < 1e-4, "min {min:?}");
    assert!((max.y - 0.5).abs() < 1e-4, "max {max:?}");
}

#[test]
fn obj_roundtrip() {
    let scene = write_and_read(&fixture(), Format::Obj);
    assert_eq!(scene.triangle_count(), 24);
    assert!(!scene.materials.is_empty());
    assert!(scene.has_textures());
    assert_eq!(scene.unit.known(), None);
    assert_fixture_bounds(&scene);
}

#[test]
fn ply_roundtrip() {
    let scene = write_and_read(&fixture(), Format::Ply);
    assert_eq!(scene.triangle_count(), 24);
    assert!(scene.has_normals());
    assert!(scene.has_uvs());
    assert_fixture_bounds(&scene);
}

#[test]
fn off_roundtrip() {
    let scene = write_and_read(&fixture(), Format::Off);
    assert_eq!(scene.triangle_count(), 24);
    assert!(scene.has_vertex_colors());
    assert_fixture_bounds(&scene);
}

#[test]
fn stl_roundtrip() {
    let scene = write_and_read(&fixture(), Format::Stl);
    assert_eq!(scene.triangle_count(), 24);
    assert_fixture_bounds(&scene);
}

#[test]
fn gltf_roundtrip() {
    let scene = write_and_read(&fixture(), Format::Gltf);
    assert_eq!(scene.triangle_count(), 24);
    assert_eq!(scene.materials.len(), 1);
    assert!(scene.has_textures());
    assert!(scene.has_tangents());
    assert!(scene.has_vertex_colors());
    assert_eq!(scene.unit.known(), Some(Unit::Meter));
    assert_fixture_bounds(&scene);
}

#[test]
fn glb_roundtrip() {
    let scene = write_and_read(&fixture(), Format::Glb);
    assert_eq!(scene.triangle_count(), 24);
    assert!(scene.has_textures());
    assert!(scene.has_tangents());
    assert!(scene.has_vertex_colors());
    assert_fixture_bounds(&scene);
}

#[test]
fn three_mf_roundtrip() {
    let scene = write_and_read(&fixture(), Format::ThreeMf);
    assert_eq!(scene.triangle_count(), 24);
    assert_eq!(scene.materials.len(), 1);
    let color = scene.materials[0].base_color;
    assert!((color[0] - 0.8).abs() < 0.01);
    assert_eq!(scene.unit.known(), Some(Unit::Millimeter));
    assert_fixture_bounds(&scene);
}

#[test]
fn dae_roundtrip() {
    let scene = write_and_read(&fixture(), Format::Dae);
    assert_eq!(scene.triangle_count(), 24);
    assert!(!scene.materials.is_empty());
    assert!(scene.has_textures());
    assert!(scene.has_vertex_colors());
    assert_eq!(scene.unit.known(), Some(Unit::Millimeter));
    assert_fixture_bounds(&scene);
}

#[test]
fn fbx_roundtrip() {
    let scene = write_and_read(&fixture(), Format::Fbx);
    assert_eq!(scene.triangle_count(), 24);
    assert!(!scene.materials.is_empty());
    assert!(scene.has_textures());
    assert!(scene.has_tangents());
    assert!(scene.has_vertex_colors());
    assert_eq!(scene.unit.known(), Some(Unit::Millimeter));
    assert_fixture_bounds(&scene);
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

fn bounds(scene: &Scene) -> (glam::Vec3, glam::Vec3) {
    scene.world_bounds().expect("bounds")
}

fn write_scene(scene: &Scene, path: &Path, format: Format) {
    let mut staging = Staging::new(path.parent().unwrap()).unwrap();
    let mut textures = TexturePlan::new();
    let mut warnings = WarningSink::new(true);
    let options = WriteOptions {
        scene,
        output_path: path,
        texture_dir: "tex",
        unit: scene.unit.known().unwrap_or(Unit::Millimeter),
        up_axis: scene.up_axis,
        geometry_only: false,
    };
    io::write(format, &options, &mut staging, &mut textures, &mut warnings).unwrap();
    staging.finalize(true).unwrap();
}

#[test]
fn stl_mm_to_glb_meters() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("cube.stl");
    write_scene(&simple_scene(Unit::Millimeter), &source, Format::Stl);
    let output = dir.path().join("cube.glb");
    convert::run(&options(&source, &output)).unwrap();
    let scene = io::read(&output, Format::Glb, None).unwrap().scene;
    let (min, max) = bounds(&scene);
    assert!((min.x + 0.0005).abs() < 1e-6, "min {min:?}");
    assert!((max.x - 0.0005).abs() < 1e-6, "max {max:?}");
}

#[test]
fn glb_meters_to_stl_millimeters() {
    let dir = tempfile::tempdir().unwrap();
    let scene = simple_scene(Unit::Meter);
    let source = dir.path().join("cube.glb");
    write_scene(&scene, &source, Format::Glb);
    let output = dir.path().join("cube.stl");
    convert::run(&options(&source, &output)).unwrap();
    let loaded = io::read(&output, Format::Stl, None).unwrap().scene;
    let (min, max) = bounds(&loaded);
    assert!((min.x + 500.0).abs() < 1e-2, "min {min:?}");
    assert!((max.x - 500.0).abs() < 1e-2, "max {max:?}");
}

#[test]
fn fbx_centimeters_to_stl_millimeters() {
    let dir = tempfile::tempdir().unwrap();
    let scene = simple_scene(Unit::Centimeter);
    let source = dir.path().join("cube.fbx");
    write_scene(&scene, &source, Format::Fbx);
    let output = dir.path().join("cube.stl");
    convert::run(&options(&source, &output)).unwrap();
    let loaded = io::read(&output, Format::Stl, None).unwrap().scene;
    let (min, max) = bounds(&loaded);
    assert!((min.x + 5.0).abs() < 1e-2, "min {min:?}");
    assert!((max.x - 5.0).abs() < 1e-2, "max {max:?}");
}

#[test]
fn dae_meters_to_stl_millimeters() {
    let dir = tempfile::tempdir().unwrap();
    let scene = simple_scene(Unit::Meter);
    let source = dir.path().join("cube.dae");
    write_scene(&scene, &source, Format::Dae);
    let output = dir.path().join("cube.stl");
    convert::run(&options(&source, &output)).unwrap();
    let loaded = io::read(&output, Format::Stl, None).unwrap().scene;
    let (min, max) = bounds(&loaded);
    assert!((min.x + 500.0).abs() < 1e-2, "min {min:?}");
    assert!((max.x - 500.0).abs() < 1e-2, "max {max:?}");
}

#[test]
fn step_roundtrip() {
    let scene = write_and_read(&simple_scene(Unit::Millimeter), Format::Step);
    assert_eq!(scene.triangle_count(), 12);
    let (min, max) = bounds(&scene);
    assert!((min.x + 0.5).abs() < 0.01, "min {min:?}");
    assert!((max.x - 0.5).abs() < 0.01, "max {max:?}");
}

#[test]
fn step_millimeters_to_glb_meters() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("cube.step");
    write_scene(&simple_scene(Unit::Millimeter), &source, Format::Step);
    let output = dir.path().join("cube.glb");
    convert::run(&options(&source, &output)).unwrap();
    let scene = io::read(&output, Format::Glb, None).unwrap().scene;
    let (min, max) = bounds(&scene);
    assert!((min.x + 0.0005).abs() < 1e-6, "min {min:?}");
    assert!((max.x - 0.0005).abs() < 1e-6, "max {max:?}");
}

#[test]
fn step_open_surface_roundtrip() {
    let mut scene = Scene::new(Format::Obj);
    scene.unit = UnitResolution::Known(Unit::Millimeter);
    scene.up_axis = UpAxis::Z;
    let mut primitive = Primitive::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        vec![0, 1, 2, 0, 2, 3],
    );
    primitive.normals = Some(vec![[0.0, 0.0, 1.0]; 4]);
    let mut mesh = Mesh::new("Plate");
    mesh.primitives.push(primitive);
    let mesh_id = scene.add_mesh(mesh);
    let node = scene.add_node("Plate", Mat4::IDENTITY);
    scene.nodes[node].mesh = Some(mesh_id);
    let roundtripped = write_and_read(&scene, Format::Step);
    assert_eq!(roundtripped.triangle_count(), 2);
}

#[test]
fn step_roundtrip_keeps_all_primitives() {
    let scene = write_and_read(&fixture(), Format::Step);
    assert_eq!(scene.triangle_count(), 24);
    assert!(scene.materials.is_empty());
    assert_eq!(scene.unit.known(), Some(Unit::Millimeter));
}

#[test]
fn split_textures_share_one_directory() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("src.glb");
    write_scene(&fixture(), &source, Format::Glb);
    let output = dir.path().join("{group}.gltf");
    let mut opts = options(&source, &output);
    opts.split_groups = true;
    convert::run(&opts).unwrap();
    let root = io::read(&dir.path().join("Root.gltf"), Format::Gltf, None)
        .unwrap()
        .scene;
    let child = io::read(&dir.path().join("Child.gltf"), Format::Gltf, None)
        .unwrap()
        .scene;
    assert!(root.has_textures());
    assert!(child.has_textures());
    let texture_dir = dir.path().join("src.textures");
    let files: Vec<_> = std::fs::read_dir(&texture_dir).unwrap().collect();
    assert_eq!(files.len(), 1, "one shared texture copy expected");
}

#[test]
fn split_textures_do_not_collide_across_pieces() {
    let mut scene = Scene::new(Format::Obj);
    scene.unit = UnitResolution::Known(Unit::Millimeter);
    scene.up_axis = UpAxis::Y;
    for (index, byte) in [1u8, 2u8].into_iter().enumerate() {
        let image = scene.add_image(Image {
            name: "same.png".into(),
            bytes: vec![0x89, b'P', b'N', b'G', byte, byte, byte],
            mime: Some("image/png".into()),
            source_path: None,
        });
        let mut material = Material::new(format!("M{index}"));
        material.textures.base_color = Some(TextureRef { image });
        let material_id = scene.add_material(material);
        let mut primitive = Primitive::new(cube_positions(), cube_indices());
        primitive.material = Some(material_id);
        let mut mesh = Mesh::new(format!("Mesh{index}"));
        mesh.primitives.push(primitive);
        let mesh_id = scene.add_mesh(mesh);
        let node = scene.add_node(format!("Part{index}"), Mat4::IDENTITY);
        scene.nodes[node].mesh = Some(mesh_id);
    }
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("src.glb");
    write_scene(&scene, &source, Format::Glb);
    let output = dir.path().join("{group}.gltf");
    let mut opts = options(&source, &output);
    opts.split_groups = true;
    convert::run(&opts).unwrap();
    let part0 = io::read(&dir.path().join("Part0.gltf"), Format::Gltf, None)
        .unwrap()
        .scene;
    let part1 = io::read(&dir.path().join("Part1.gltf"), Format::Gltf, None)
        .unwrap()
        .scene;
    assert_ne!(part0.images[0].bytes, part1.images[0].bytes);
    let files: Vec<_> = std::fs::read_dir(dir.path().join("src.textures"))
        .unwrap()
        .collect();
    assert_eq!(files.len(), 2);
}

#[test]
fn obj_mm_to_fbx_centimeters_preserves_physical_size() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("cube.obj");
    write_scene(&simple_scene(Unit::Millimeter), &source, Format::Obj);
    let fbx = dir.path().join("cube.fbx");
    convert::run(&options(&source, &fbx)).unwrap();
    let fbx_scene = io::read(&fbx, Format::Fbx, None).unwrap().scene;
    assert_eq!(fbx_scene.unit.known(), Some(Unit::Centimeter));
    let (min, max) = bounds(&fbx_scene);
    assert!((min.x + 0.05).abs() < 1e-4, "{min:?}");
    assert!((max.x - 0.05).abs() < 1e-4, "{max:?}");

    let stl = dir.path().join("back.stl");
    convert::run(&options(&fbx, &stl)).unwrap();
    let stl_scene = io::read(&stl, Format::Stl, None).unwrap().scene;
    let (min, max) = bounds(&stl_scene);
    assert!((min.x + 0.5).abs() < 1e-4, "{min:?}");
    assert!((max.x - 0.5).abs() < 1e-4, "{max:?}");
}

#[test]
fn three_mf_millimeters_to_stl_millimeters_is_identity() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("cube.3mf");
    write_scene(&simple_scene(Unit::Millimeter), &source, Format::ThreeMf);
    let stl = dir.path().join("cube.stl");
    convert::run(&options(&source, &stl)).unwrap();
    let scene = io::read(&stl, Format::Stl, None).unwrap().scene;
    let (min, max) = bounds(&scene);
    assert!((min.x + 0.5).abs() < 1e-4, "{min:?}");
    assert!((max.x - 0.5).abs() < 1e-4, "{max:?}");
}

#[test]
fn empty_meshes_do_not_break_writers() {
    let mut scene = simple_scene(Unit::Millimeter);
    let empty_mesh = scene.add_mesh(Mesh::new("Empty"));
    let empty_node = scene.add_node("EmptyNode", Mat4::IDENTITY);
    scene.nodes[empty_node].mesh = Some(empty_mesh);
    for format in [
        Format::Fbx,
        Format::Dae,
        Format::Gltf,
        Format::Glb,
        Format::ThreeMf,
        Format::Obj,
        Format::Step,
    ] {
        let roundtripped = write_and_read(&scene, format);
        assert_eq!(
            roundtripped.triangle_count(),
            12,
            "{format} mishandled an empty mesh"
        );
    }
}

#[test]
fn step_inch_conversion_based_units_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("cube.obj");
    write_scene(&simple_scene(Unit::Millimeter), &source, Format::Obj);
    let step = dir.path().join("cube.step");
    let mut opts = options(&source, &step);
    opts.unit_out = Some(Unit::Inch);
    convert::run(&opts).unwrap();
    let scene = io::read(&step, Format::Step, None).unwrap().scene;
    assert_eq!(scene.unit.known(), Some(Unit::Inch));
    let (min, max) = bounds(&scene);
    assert!((min.x + 0.5 / 25.4).abs() < 1e-5, "{min:?}");
    assert!((max.x - 0.5 / 25.4).abs() < 1e-5, "{max:?}");

    let stl = dir.path().join("back.stl");
    convert::run(&options(&step, &stl)).unwrap();
    let scene = io::read(&stl, Format::Stl, None).unwrap().scene;
    let (min, max) = bounds(&scene);
    assert!((min.x + 0.5).abs() < 1e-4, "{min:?}");
    assert!((max.x - 0.5).abs() < 1e-4, "{max:?}");
}

#[test]
fn fbx_bakes_sheared_transforms() {
    let mut scene = simple_scene(Unit::Millimeter);
    let mut matrix = Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0));
    matrix.y_axis.x = 1.0;
    let node = scene.roots[0];
    scene.nodes[node].local_transform = matrix;
    let (expected_min, expected_max) = bounds(&scene);
    let roundtripped = write_and_read(&scene, Format::Fbx);
    let (min, max) = bounds(&roundtripped);
    assert!(
        (min - expected_min).length() < 1e-4,
        "{min:?} vs {expected_min:?}"
    );
    assert!(
        (max - expected_max).length() < 1e-4,
        "{max:?} vs {expected_max:?}"
    );
}

#[test]
fn three_mf_splits_multi_material_meshes_without_losing_colors() {
    let mut scene = Scene::new(Format::Obj);
    scene.unit = UnitResolution::Known(Unit::Millimeter);
    scene.up_axis = UpAxis::Z;
    let mut materials = Vec::new();
    for (index, color) in [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
        .into_iter()
        .enumerate()
    {
        let mut material = Material::new(format!("M{index}"));
        material.base_color = color;
        materials.push(scene.add_material(material));
    }
    let mut mesh = Mesh::new("TwoMaterials");
    for (index, material) in materials.iter().enumerate() {
        let positions = cube_positions()
            .into_iter()
            .map(|mut position| {
                position[0] += index as f32 * 2.0;
                position
            })
            .collect();
        let mut primitive = Primitive::new(positions, cube_indices());
        primitive.material = Some(*material);
        mesh.primitives.push(primitive);
    }
    let mesh_id = scene.add_mesh(mesh);
    let node = scene.add_node("TwoMaterials", Mat4::IDENTITY);
    scene.nodes[node].mesh = Some(mesh_id);

    let roundtripped = write_and_read(&scene, Format::ThreeMf);
    assert_eq!(roundtripped.triangle_count(), 24);
    assert_eq!(roundtripped.materials.len(), 2);
    let mut colors: Vec<[f32; 4]> = roundtripped
        .materials
        .iter()
        .map(|material| material.base_color)
        .collect();
    colors.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
    assert!((colors[0][1] - 1.0).abs() < 0.01);
    assert!((colors[1][0] - 1.0).abs() < 0.01);
}

#[test]
fn z_up_metadata_survives_fbx_and_dae() {
    let mut scene = simple_scene(Unit::Millimeter);
    scene.up_axis = UpAxis::Z;
    scene.up_axis_from_file = true;
    for format in [Format::Fbx, Format::Dae] {
        let roundtripped = write_and_read(&scene, format);
        assert_eq!(roundtripped.up_axis, UpAxis::Z, "{format}");
    }
}

#[test]
fn unreferenced_meshes_are_not_exported_by_merge_formats() {
    let mut scene = simple_scene(Unit::Millimeter);
    let mut unused = Mesh::new("Unused");
    unused
        .primitives
        .push(Primitive::new(cube_positions(), cube_indices()));
    scene.add_mesh(unused);
    for format in [
        Format::Stl,
        Format::Ply,
        Format::Off,
        Format::Step,
        Format::Obj,
    ] {
        let roundtripped = write_and_read(&scene, format);
        assert_eq!(roundtripped.triangle_count(), 12, "{format}");
    }
}

#[test]
fn dae_preserves_vertex_color_values() {
    let mut scene = simple_scene(Unit::Millimeter);
    let mesh_id = scene.roots[0];
    let mesh_id = scene.nodes[mesh_id].mesh.unwrap();
    let primitive = &mut scene.meshes[mesh_id].primitives[0];
    primitive.colors = Some(vec![[0.25, 0.5, 0.75, 1.0]; primitive.positions.len()]);
    let roundtripped = write_and_read(&scene, Format::Dae);
    let colors = roundtripped.meshes[0].primitives[0]
        .colors
        .as_ref()
        .expect("vertex colors");
    assert!((colors[0][0] - 0.25).abs() < 1e-5);
    assert!((colors[0][1] - 0.5).abs() < 1e-5);
    assert!((colors[0][2] - 0.75).abs() < 1e-5);
    assert!((colors[0][3] - 1.0).abs() < 1e-5);
}

#[test]
fn split_groups_writes_one_file_per_node() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("src.glb");
    write_scene(&fixture(), &source, Format::Glb);
    let output = dir.path().join("{group}.glb");
    let mut opts = options(&source, &output);
    opts.split_groups = true;
    convert::run(&opts).unwrap();
    let root = io::read(&dir.path().join("Root.glb"), Format::Glb, None)
        .unwrap()
        .scene;
    let child = io::read(&dir.path().join("Child.glb"), Format::Glb, None)
        .unwrap()
        .scene;
    assert_eq!(root.triangle_count(), 12);
    assert_eq!(child.triangle_count(), 12);
}
