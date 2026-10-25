use std::path::Path;

#[test]
fn required_repository_files_exist() {
    for path in [
        "README.md",
        "LICENSE",
        ".gitignore",
        "rust-toolchain.toml",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        "Cargo.lock",
        "build.rs",
    ] {
        assert!(Path::new(path).exists(), "missing required file {path}");
    }
}

#[test]
fn readme_documents_formats_units_and_gpu() {
    let readme = std::fs::read_to_string("README.md").unwrap();
    for needle in [
        "FBX",
        "3MF",
        "STL",
        "OBJ",
        "OFF",
        "glTF",
        "GLB",
        "DAE",
        "PLY",
        "STEP",
        "--split-groups",
        "--no-unit-conversion",
        "--strict",
        "GPU acceleration",
        "MIT",
    ] {
        assert!(readme.contains(needle), "README is missing {needle}");
    }
}

#[test]
fn license_is_mit() {
    let license = std::fs::read_to_string("LICENSE").unwrap();
    assert!(license.contains("MIT License"));
    assert!(license.contains("2026 trine contributors"));
}
