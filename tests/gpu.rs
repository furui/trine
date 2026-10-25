#![cfg(feature = "gpu")]

use glam::{Mat4, Quat, Vec3};
use trine::ops::gpu::{GpuMode, create};
use trine::ops::normals::smooth_normals;
use trine::ops::tangents::generate_tangents;
use trine::report::WarningSink;
use trine::scene::Primitive;

fn cube() -> Primitive {
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
    let mut primitive = Primitive::new(positions.clone(), indices);
    primitive.normals = Some(
        positions
            .iter()
            .map(|p| Vec3::from(*p).normalize().to_array())
            .collect(),
    );
    primitive
        .uvs
        .push(positions.iter().map(|p| [p[0] + 0.5, p[1] + 0.5]).collect());
    primitive
}

#[test]
fn gpu_matches_cpu() {
    if std::env::var("TRINE_TEST_GPU").as_deref() != Ok("1") {
        eprintln!("skipping GPU parity test: set TRINE_TEST_GPU=1 to enable");
        return;
    }
    let mut warnings = WarningSink::new(true);
    let engine = match create(GpuMode::On, false, &mut warnings) {
        Ok(Some(engine)) => engine,
        _ => {
            eprintln!("skipping GPU parity test: no adapter available");
            return;
        }
    };

    let matrix = Mat4::from_scale_rotation_translation(
        Vec3::new(2.0, 0.5, 1.5),
        Quat::from_rotation_y(0.7),
        Vec3::new(1.0, 2.0, 3.0),
    );
    let mut cpu = cube();
    trine::ops::flatten::transform_primitive(&mut cpu, matrix);
    let mut gpu = cube();
    assert!(engine.transform_primitive(&mut gpu, matrix));
    for (a, b) in cpu.positions.iter().zip(&gpu.positions) {
        assert!((Vec3::from(*a) - Vec3::from(*b)).length() < 1e-4);
    }
    for (a, b) in cpu
        .normals
        .as_ref()
        .unwrap()
        .iter()
        .zip(gpu.normals.as_ref().unwrap())
    {
        assert!(Vec3::from(*a).dot(Vec3::from(*b)) > 0.999);
    }

    let mut gpu_smooth = cube();
    assert!(engine.smooth_normals(&mut gpu_smooth));
    let cpu_smooth = smooth_normals(&gpu_smooth.positions, &gpu_smooth.indices);
    for (a, b) in cpu_smooth.iter().zip(gpu_smooth.normals.as_ref().unwrap()) {
        assert!(Vec3::from(*a).dot(Vec3::from(*b)) > 0.999, "{a:?} vs {b:?}");
    }

    let mut gpu_tangents = cube();
    assert!(engine.smooth_normals(&mut gpu_tangents));
    assert!(engine.generate_tangents(&mut gpu_tangents));
    let cpu_tangents = generate_tangents(&gpu_tangents).expect("tangents");
    for (a, b) in cpu_tangents
        .iter()
        .zip(gpu_tangents.tangents.as_ref().unwrap())
    {
        assert!(Vec3::new(a[0], a[1], a[2]).dot(Vec3::new(b[0], b[1], b[2])) > 0.99);
    }
}
