fn main() {
    println!("cargo:rerun-if-env-changed=TRINE_BUILD_VERSION");
    println!("cargo:rerun-if-changed=build.rs");

    let version = std::env::var("TRINE_BUILD_VERSION")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| std::env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION"));

    println!("cargo:rustc-env=TRINE_VERSION={}", version.trim());
}
