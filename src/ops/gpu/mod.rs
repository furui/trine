//! GPU acceleration mode selection and availability checks.
//!
//! The actual wgpu implementation lives behind the `gpu` feature. CPU-only
//! builds accept `--gpu auto` (and fall back), but `--gpu on` is an error.

#[cfg(feature = "gpu")]
mod backend;
#[cfg(not(feature = "gpu"))]
mod stub;

#[cfg(feature = "gpu")]
pub use backend::{GpuEngine, create};
#[cfg(not(feature = "gpu"))]
pub use stub::{GpuEngine, create};

use crate::error::{Error, Result};

/// GPU acceleration mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuMode {
    /// Use the GPU when an adapter is available and the work is large enough.
    #[default]
    Auto,
    /// Require a GPU adapter.
    On,
    /// Always use the CPU path.
    Off,
}

impl GpuMode {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "auto" => Some(GpuMode::Auto),
            "on" => Some(GpuMode::On),
            "off" => Some(GpuMode::Off),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            GpuMode::Auto => "auto",
            GpuMode::On => "on",
            GpuMode::Off => "off",
        }
    }
}

/// Minimum vertex count before `auto` considers the GPU worthwhile.
pub const GPU_VERTEX_THRESHOLD: usize = 50_000;

/// Fail early when `--gpu on` was requested but this build has no GPU support.
pub fn check_build_support(mode: GpuMode) -> Result<()> {
    if mode == GpuMode::On && !cfg!(feature = "gpu") {
        return Err(Error::conversion(
            "--gpu on requires a build with the 'gpu' feature enabled; rebuild without --no-default-features or use --gpu off",
        ));
    }
    Ok(())
}

/// Whether the GPU feature is compiled in.
pub const fn compiled_in() -> bool {
    cfg!(feature = "gpu")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn gpu_on_requires_the_feature() {
        assert!(check_build_support(GpuMode::On).is_err());
        assert!(check_build_support(GpuMode::Auto).is_ok());
        assert!(check_build_support(GpuMode::Off).is_ok());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_on_is_allowed_with_the_feature() {
        assert!(check_build_support(GpuMode::On).is_ok());
    }

    #[test]
    fn parses_modes() {
        assert_eq!(GpuMode::parse("auto"), Some(GpuMode::Auto));
        assert_eq!(GpuMode::parse("on"), Some(GpuMode::On));
        assert_eq!(GpuMode::parse("off"), Some(GpuMode::Off));
        assert_eq!(GpuMode::parse("maybe"), None);
    }
}
