//! CPU-only builds: no wgpu symbols are referenced.

use glam::Mat4;

use crate::error::Result;
use crate::ops::gpu::GpuMode;
use crate::report::WarningSink;
use crate::scene::Primitive;

pub struct GpuEngine;

impl GpuEngine {
    pub fn should_accelerate(&self, _vertices: usize) -> bool {
        false
    }

    pub fn transform_primitive(&self, _primitive: &mut Primitive, _matrix: Mat4) -> bool {
        false
    }

    pub fn smooth_normals(&self, _primitive: &mut Primitive) -> bool {
        false
    }

    pub fn generate_tangents(&self, _primitive: &mut Primitive) -> bool {
        false
    }
}

pub fn create(
    _mode: GpuMode,
    _verbose: bool,
    _warnings: &mut WarningSink,
) -> Result<Option<GpuEngine>> {
    Ok(None)
}
