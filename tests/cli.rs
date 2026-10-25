use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use trine::io::{self, Format};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_trine")
}

fn run(args: &[&str]) -> Output {
    Command::new(binary())
        .args(args)
        .output()
        .expect("run trine")
}

fn write_obj(path: &Path) {
    let mut text = String::from("o Part0\n");
    for (x, y, z) in [
        (-0.5, -0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 0.0),
        (-0.5, 0.5, 0.0),
    ] {
        text.push_str(&format!("v {x} {y} {z}\n"));
    }
    text.push_str("vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n");
    text.push_str("f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n");
    text.push_str("o Part1\n");
    for (x, y, z) in [
        (1.5, -0.5, 0.0),
        (2.5, -0.5, 0.0),
        (2.5, 0.5, 0.0),
        (1.5, 0.5, 0.0),
    ] {
        text.push_str(&format!("v {x} {y} {z}\n"));
    }
    text.push_str("vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n");
    text.push_str("f 5/5 6/6 7/7\nf 5/5 7/7 8/8\n");
    fs::write(path, text).unwrap();
}

fn bounds(path: &Path, format: Format) -> (glam::Vec3, glam::Vec3) {
    let scene = io::read(path, format, None).unwrap().scene;
    scene.world_bounds().unwrap()
}

#[test]
fn help_lists_examples_and_exits_zero() {
    let output = run(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Examples:"));
    assert!(stdout.contains("--split-groups"));
    assert!(stdout.contains("step"));
}

#[test]
fn clap_rejects_unknown_flags_with_exit_two() {
    let output = run(&["a.obj", "b.glb", "--not-a-flag"]);
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn missing_input_exits_one() {
    let dir = tempfile::tempdir().unwrap();
    let output = run(&[
        dir.path().join("missing.obj").to_str().unwrap(),
        dir.path().join("out.glb").to_str().unwrap(),
        "--force",
        "--gpu",
        "off",
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("does not exist"));
}

#[test]
fn gltf_rejects_non_meter_unit_out_with_exit_two() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let output = run(&[
        source.to_str().unwrap(),
        dir.path().join("out.glb").to_str().unwrap(),
        "--unit-out",
        "mm",
        "--force",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("always uses meters"));
}

#[test]
fn gltf_rejects_non_y_up_with_exit_two() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let output = run(&[
        source.to_str().unwrap(),
        dir.path().join("out.glb").to_str().unwrap(),
        "--up-axis",
        "z",
        "--force",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Y-up"));
}

#[test]
fn three_mf_rejects_non_z_up_with_exit_two() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let output = run(&[
        source.to_str().unwrap(),
        dir.path().join("out.3mf").to_str().unwrap(),
        "--up-axis",
        "y",
        "--force",
    ]);
    assert_eq!(output.status.code(), Some(2));
}

fn write_plain_triangle(path: &Path) {
    fs::write(path, "o Tri\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
}

#[test]
fn strict_fails_when_unit_metadata_is_lost() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("tri.obj");
    write_plain_triangle(&source);
    let glb = dir.path().join("tri.glb");
    assert!(
        run(&[
            source.to_str().unwrap(),
            glb.to_str().unwrap(),
            "--force",
            "--gpu",
            "off",
        ])
        .status
        .success()
    );
    let stl = dir.path().join("tri.stl");
    let output = run(&[
        glb.to_str().unwrap(),
        stl.to_str().unwrap(),
        "--strict",
        "--force",
        "--gpu",
        "off",
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(!stl.exists());
}

#[test]
fn quiet_suppresses_warnings_and_summary() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let target = dir.path().join("out.glb");
    let output = run(&[
        source.to_str().unwrap(),
        target.to_str().unwrap(),
        "--force",
        "--gpu",
        "off",
        "--quiet",
    ]);
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
    assert!(target.exists());
}

#[test]
fn warnings_are_reported_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let target = dir.path().join("out.stl");
    let output = run(&[
        source.to_str().unwrap(),
        target.to_str().unwrap(),
        "--force",
        "--gpu",
        "off",
    ]);
    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("assumption"));
    assert!(stderr.contains("hierarchy"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Converted"));
    assert!(stdout.contains("triangles"));
}

#[test]
fn strict_fails_on_lost_input_data() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let target = dir.path().join("out.stl");
    let output = run(&[
        source.to_str().unwrap(),
        target.to_str().unwrap(),
        "--force",
        "--gpu",
        "off",
        "--strict",
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--strict"));
    assert!(!target.exists());
}

#[test]
fn strict_allows_lossless_conversions_and_assumptions() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let target = dir.path().join("out.glb");
    let output = run(&[
        source.to_str().unwrap(),
        target.to_str().unwrap(),
        "--strict",
        "--force",
        "--gpu",
        "off",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(target.exists());
}

#[test]
fn existing_output_requires_force() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let target = dir.path().join("out.glb");
    let args = [
        source.to_str().unwrap(),
        target.to_str().unwrap(),
        "--gpu",
        "off",
    ];
    assert!(run(&args).status.success());
    let second = run(&args);
    assert_eq!(second.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&second.stderr).contains("--force"));
    let third = run(&[
        source.to_str().unwrap(),
        target.to_str().unwrap(),
        "--gpu",
        "off",
        "--force",
    ]);
    assert!(third.status.success());
}

#[test]
fn unit_in_unit_out_scale_coordinates() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let target = dir.path().join("out.stl");
    let output = run(&[
        source.to_str().unwrap(),
        target.to_str().unwrap(),
        "--unit-in",
        "cm",
        "--unit-out",
        "mm",
        "--force",
        "--gpu",
        "off",
    ]);
    assert!(output.status.success());
    let (min, max) = bounds(&target, Format::Stl);
    assert!((min.x + 5.0).abs() < 1e-3, "{min:?}");
    assert!((max.x - 25.0).abs() < 1e-3, "{max:?}");
}

#[test]
fn no_unit_conversion_preserves_raw_numbers() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let target = dir.path().join("raw.glb");
    let output = run(&[
        source.to_str().unwrap(),
        target.to_str().unwrap(),
        "--no-unit-conversion",
        "--force",
        "--gpu",
        "off",
    ]);
    assert!(output.status.success());
    let (min, max) = bounds(&target, Format::Glb);
    assert!((min.x + 0.5).abs() < 1e-6, "{min:?}");
    assert!((max.x - 2.5).abs() < 1e-6, "{max:?}");

    let scaled = dir.path().join("scaled.glb");
    let output = run(&[
        source.to_str().unwrap(),
        scaled.to_str().unwrap(),
        "--force",
        "--gpu",
        "off",
    ]);
    assert!(output.status.success());
    let (min, max) = bounds(&scaled, Format::Glb);
    assert!((min.x + 0.0005).abs() < 1e-6, "{min:?}");
    assert!((max.x - 0.0025).abs() < 1e-6, "{max:?}");
}

#[test]
fn split_groups_writes_one_file_per_object() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let pattern = dir.path().join("{group}.glb");
    let output = run(&[
        source.to_str().unwrap(),
        pattern.to_str().unwrap(),
        "--split-groups",
        "--force",
        "--gpu",
        "off",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(dir.path().join("Part0.glb").exists());
    assert!(dir.path().join("Part1.glb").exists());
    let (min, _) = bounds(&dir.path().join("Part0.glb"), Format::Glb);
    assert!((min.x + 0.0005).abs() < 1e-6, "{min:?}");
}

#[test]
fn split_into_a_directory_uses_the_input_stem() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let outdir = dir.path().join("pieces");
    fs::create_dir(&outdir).unwrap();
    let output = run(&[
        source.to_str().unwrap(),
        outdir.to_str().unwrap(),
        "--split-groups",
        "--to",
        "glb",
        "--force",
        "--gpu",
        "off",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(outdir.join("in_Part0.glb").exists());
    assert!(outdir.join("in_Part1.glb").exists());
}

#[test]
fn split_from_single_group_sources_writes_one_file() {
    let dir = tempfile::tempdir().unwrap();
    let obj = dir.path().join("cube.obj");
    write_obj(&obj);
    let stl = dir.path().join("cube.stl");
    assert!(
        run(&[
            obj.to_str().unwrap(),
            stl.to_str().unwrap(),
            "--force",
            "--gpu",
            "off",
        ])
        .status
        .success()
    );
    let output = run(&[
        stl.to_str().unwrap(),
        dir.path().join("{group}.glb").to_str().unwrap(),
        "--split-groups",
        "--force",
        "--gpu",
        "off",
    ]);
    assert!(output.status.success());
    assert!(dir.path().join("cube.glb").exists());
    assert!(String::from_utf8_lossy(&output.stderr).contains("single synthetic group"));
}

#[test]
fn split_falls_back_to_single_file_for_stl() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let pattern = dir.path().join("{group}.stl");
    let output = run(&[
        source.to_str().unwrap(),
        pattern.to_str().unwrap(),
        "--split-groups",
        "--force",
        "--gpu",
        "off",
    ]);
    assert!(output.status.success());
    assert!(dir.path().join("in.stl").exists());
    assert!(!dir.path().join("Part0.stl").exists());
    assert!(String::from_utf8_lossy(&output.stderr).contains("one file"));
}

#[test]
fn invalid_tolerance_is_usage_error() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let output = run(&[
        source.to_str().unwrap(),
        dir.path().join("out.step").to_str().unwrap(),
        "--tessellation-tolerance",
        "0",
        "--force",
    ]);
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn axis_conversion_rotates_geometry() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);

    let converted = dir.path().join("converted.step");
    assert!(
        run(&[
            source.to_str().unwrap(),
            converted.to_str().unwrap(),
            "--force",
            "--gpu",
            "off",
        ])
        .status
        .success()
    );
    let (converted_min, converted_max) = bounds(&converted, Format::Step);
    assert!((converted_max.z - converted_min.z - 1.0).abs() < 1e-3);
    assert!((converted_max.y - converted_min.y).abs() < 1e-3);

    let kept = dir.path().join("kept.step");
    assert!(
        run(&[
            source.to_str().unwrap(),
            kept.to_str().unwrap(),
            "--keep-axes",
            "--force",
            "--gpu",
            "off",
        ])
        .status
        .success()
    );
    let (kept_min, kept_max) = bounds(&kept, Format::Step);
    assert!((kept_max.y - kept_min.y - 1.0).abs() < 1e-3);
    assert!((kept_max.z - kept_min.z).abs() < 1e-3);
}

#[test]
fn unknown_extensions_are_usage_errors() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("model.xyz");
    fs::write(&source, "whatever").unwrap();
    let output = run(&[
        source.to_str().unwrap(),
        dir.path().join("out.glb").to_str().unwrap(),
        "--force",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot infer input format"));
}

#[test]
fn corrupt_indices_are_conversion_errors() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("bad.obj");
    fs::write(&source, "o Bad\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99\n").unwrap();
    let output = run(&[
        source.to_str().unwrap(),
        dir.path().join("out.glb").to_str().unwrap(),
        "--force",
        "--gpu",
        "off",
    ]);
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn empty_files_report_no_geometry() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("empty.obj");
    fs::write(&source, "# nothing here\n").unwrap();
    let output = run(&[
        source.to_str().unwrap(),
        dir.path().join("out.glb").to_str().unwrap(),
        "--force",
        "--gpu",
        "off",
    ]);
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn missing_material_library_warns_and_fails_strict() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    fs::write(
        &source,
        "mtllib missing.mtl\nusemtl Oak\no Tri\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
    )
    .unwrap();
    let target = dir.path().join("out.glb");
    let output = run(&[
        source.to_str().unwrap(),
        target.to_str().unwrap(),
        "--force",
        "--gpu",
        "off",
    ]);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("material library"));

    let strict_target = dir.path().join("strict.glb");
    let output = run(&[
        source.to_str().unwrap(),
        strict_target.to_str().unwrap(),
        "--strict",
        "--force",
        "--gpu",
        "off",
    ]);
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn missing_texture_warns_and_fails_strict() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("materials.mtl"),
        "newmtl Oak\nKd 0.5 0.3 0.1\nmap_Kd missing.png\n",
    )
    .unwrap();
    let source = dir.path().join("in.obj");
    fs::write(
        &source,
        "mtllib materials.mtl\nusemtl Oak\no Tri\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
    )
    .unwrap();
    let target = dir.path().join("out.glb");
    let output = run(&[
        source.to_str().unwrap(),
        target.to_str().unwrap(),
        "--force",
        "--gpu",
        "off",
    ]);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("texture"));

    let strict_target = dir.path().join("strict.glb");
    let output = run(&[
        source.to_str().unwrap(),
        strict_target.to_str().unwrap(),
        "--strict",
        "--force",
        "--gpu",
        "off",
    ]);
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn conflicting_unit_flags_are_usage_errors() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let output = run(&[
        source.to_str().unwrap(),
        dir.path().join("out.glb").to_str().unwrap(),
        "--no-unit-conversion",
        "--unit-in",
        "mm",
        "--force",
    ]);
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn conflicting_axis_flags_are_usage_errors() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let output = run(&[
        source.to_str().unwrap(),
        dir.path().join("out.glb").to_str().unwrap(),
        "--keep-axes",
        "--up-axis",
        "y",
        "--force",
    ]);
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn relative_output_paths_work() {
    let dir = tempfile::tempdir().unwrap();
    write_obj(&dir.path().join("in.obj"));
    let output = Command::new(binary())
        .current_dir(dir.path())
        .args(["in.obj", "out.glb", "--force", "--gpu", "off"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(dir.path().join("out.glb").exists());
}

#[test]
fn output_paths_are_reported() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("in.obj");
    write_obj(&source);
    let target = dir.path().join("out.glb");
    let output = run(&[
        source.to_str().unwrap(),
        target.to_str().unwrap(),
        "--force",
        "--gpu",
        "off",
        "--verbose",
    ]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(&PathBuf::from(&target).display().to_string()));
}
