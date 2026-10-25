//! wgpu compute implementation for vertex transforms, smooth normals, and tangents.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4};
use wgpu::util::DeviceExt;

use crate::error::{Error, Result};
use crate::ops::gpu::{GPU_VERTEX_THRESHOLD, GpuMode};
use crate::report::{WarningKind, WarningSink};
use crate::scene::Primitive;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransformUniforms {
    model: [[f32; 4]; 4],
    normal: [[f32; 4]; 3],
    tangent: [[f32; 4]; 3],
    flags: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Counts {
    triangles: u32,
    vertices: u32,
    pad0: u32,
    pad1: u32,
}

pub struct GpuEngine {
    mode: GpuMode,
    device: wgpu::Device,
    queue: wgpu::Queue,
    transform_pipeline: wgpu::ComputePipeline,
    transform_layout: wgpu::BindGroupLayout,
    normals_accum_pipeline: wgpu::ComputePipeline,
    normals_accum_layout: wgpu::BindGroupLayout,
    normals_final_pipeline: wgpu::ComputePipeline,
    normals_final_layout: wgpu::BindGroupLayout,
    tangents_accum_pipeline: wgpu::ComputePipeline,
    tangents_accum_layout: wgpu::BindGroupLayout,
    tangents_final_pipeline: wgpu::ComputePipeline,
    tangents_final_layout: wgpu::BindGroupLayout,
    max_storage: u64,
    max_workgroups: u32,
}

impl GpuEngine {
    pub fn should_accelerate(&self, vertices: usize) -> bool {
        match self.mode {
            GpuMode::On => true,
            GpuMode::Auto => vertices >= GPU_VERTEX_THRESHOLD,
            GpuMode::Off => false,
        }
    }

    pub fn transform_primitive(&self, primitive: &mut Primitive, matrix: Mat4) -> bool {
        if primitive.positions.is_empty() {
            return false;
        }
        let count = primitive.positions.len();
        if !self.fits_dispatch(count) {
            return false;
        }
        let positions: Vec<[f32; 4]> = primitive
            .positions
            .iter()
            .map(|p| [p[0], p[1], p[2], 1.0])
            .collect();
        let normals: Vec<[f32; 4]> = primitive
            .normals
            .as_ref()
            .map(|normals| normals.iter().map(|n| [n[0], n[1], n[2], 0.0]).collect())
            .unwrap_or_default();
        let tangents: Vec<[f32; 4]> = primitive.tangents.clone().unwrap_or_default();
        if !self.fits_storage(positions.len() * 16)
            || !self.fits_storage(normals.len().max(1) * 16)
            || !self.fits_storage(tangents.len().max(1) * 16)
        {
            return false;
        }

        let position_buffer = self.storage_named("positions", bytemuck::cast_slice(&positions));
        let normal_buffer = if normals.is_empty() {
            self.dummy()
        } else {
            self.storage_named("normals", bytemuck::cast_slice(&normals))
        };
        let tangent_buffer = if tangents.is_empty() {
            self.dummy()
        } else {
            self.storage_named("tangents", bytemuck::cast_slice(&tangents))
        };
        let normal_matrix = crate::scene::transform::normal_matrix(matrix);
        let tangent_matrix = Mat3::from_mat4(matrix);
        let uniforms = TransformUniforms {
            model: matrix.to_cols_array_2d(),
            normal: mat3_packed(normal_matrix),
            tangent: mat3_packed(tangent_matrix),
            flags: [
                count as f32,
                if matrix.determinant() < 0.0 { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        };
        let uniform_buffer = self.uniform_named("transform-uniforms", &uniforms);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transform-bind-group"),
            layout: &self.transform_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: normal_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tangent_buffer.as_entire_binding(),
                },
            ],
        });
        self.run(
            &self.transform_pipeline,
            &bind_group,
            (count as u32).div_ceil(64),
        );
        let Some(positions_out): Option<Vec<[f32; 4]>> = self.readback(&position_buffer, count)
        else {
            return false;
        };
        let normals_out: Option<Vec<[f32; 4]>> = if normals.is_empty() {
            None
        } else {
            match self.readback(&normal_buffer, count) {
                Some(values) => Some(values),
                None => return false,
            }
        };
        let tangents_out: Option<Vec<[f32; 4]>> = if tangents.is_empty() {
            None
        } else {
            match self.readback(&tangent_buffer, count) {
                Some(values) => Some(values),
                None => return false,
            }
        };

        for (position, out) in primitive.positions.iter_mut().zip(positions_out) {
            *position = [out[0], out[1], out[2]];
        }
        if let (Some(target), Some(values)) = (&mut primitive.normals, normals_out) {
            for (normal, out) in target.iter_mut().zip(values) {
                *normal = [out[0], out[1], out[2]];
            }
        }
        if let (Some(target), Some(values)) = (&mut primitive.tangents, tangents_out) {
            *target = values;
        }
        true
    }

    pub fn smooth_normals(&self, primitive: &mut Primitive) -> bool {
        let count = primitive.positions.len();
        let triangle_count = primitive.indices.len() / 3;
        if count == 0
            || triangle_count == 0
            || !self.fits_dispatch(count)
            || !self.fits_dispatch(triangle_count)
            || !self.fits_storage(count * 16)
            || !self.fits_storage(triangle_count * 12)
            || !self.fits_storage(count * 4)
            || !self.fits_storage(count * 12)
        {
            return false;
        }
        let positions: Vec<[f32; 4]> = primitive
            .positions
            .iter()
            .map(|p| [p[0], p[1], p[2], 1.0])
            .collect();
        let weld = crate::ops::normals::weld_map(&primitive.positions)
            .into_iter()
            .map(|index| index as u32)
            .collect::<Vec<u32>>();
        let accum_count = count * 3;
        let accum_zeros = vec![0i32; accum_count];
        let position_buffer =
            self.storage_named("smooth-positions", bytemuck::cast_slice(&positions));
        let index_buffer =
            self.storage_named("smooth-indices", bytemuck::cast_slice(&primitive.indices));
        let weld_buffer = self.storage_named("smooth-weld", bytemuck::cast_slice(&weld));
        let accum_buffer = self.storage_named("smooth-accum", bytemuck::cast_slice(&accum_zeros));
        let counts = Counts {
            triangles: triangle_count as u32,
            vertices: count as u32,
            pad0: 0,
            pad1: 0,
        };
        let counts_buffer = self.uniform_named("smooth-counts", &counts);
        let accum_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smooth-accum-bind-group"),
            layout: &self.normals_accum_layout,
            entries: &[
                entry(0, &position_buffer),
                entry(1, &index_buffer),
                entry(2, &weld_buffer),
                entry(3, &accum_buffer),
                entry(4, &counts_buffer),
            ],
        });
        self.run(
            &self.normals_accum_pipeline,
            &accum_bind,
            (triangle_count as u32).div_ceil(64),
        );

        let normal_zeros = vec![[0.0f32; 4]; count];
        let normal_buffer =
            self.storage_named("smooth-normals", bytemuck::cast_slice(&normal_zeros));
        let final_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smooth-final-bind-group"),
            layout: &self.normals_final_layout,
            entries: &[
                entry(0, &accum_buffer),
                entry(1, &normal_buffer),
                entry(2, &counts_buffer),
            ],
        });
        self.run(
            &self.normals_final_pipeline,
            &final_bind,
            (count as u32).div_ceil(64),
        );
        let Some(normals): Option<Vec<[f32; 4]>> = self.readback(&normal_buffer, count) else {
            return false;
        };
        primitive.normals = Some(normals.iter().map(|n| [n[0], n[1], n[2]]).collect());
        primitive.tangents = None;
        true
    }

    pub fn generate_tangents(&self, primitive: &mut Primitive) -> bool {
        let normals = match &primitive.normals {
            Some(normals) if normals.len() == primitive.positions.len() => normals.clone(),
            _ => return false,
        };
        let uvs = match primitive.uv0() {
            Some(uvs) if uvs.len() == primitive.positions.len() => uvs.to_vec(),
            _ => return false,
        };
        let count = primitive.positions.len();
        let triangle_count = primitive.indices.len() / 3;
        if count == 0
            || triangle_count == 0
            || !self.fits_dispatch(count)
            || !self.fits_dispatch(triangle_count)
            || !self.fits_storage(count * 16)
            || !self.fits_storage(count * 8)
            || !self.fits_storage(triangle_count * 12)
            || !self.fits_storage(count * 24)
            || !self.fits_storage(count * 16)
        {
            return false;
        }
        let positions: Vec<[f32; 4]> = primitive
            .positions
            .iter()
            .map(|p| [p[0], p[1], p[2], 1.0])
            .collect();
        let normals4: Vec<[f32; 4]> = normals.iter().map(|n| [n[0], n[1], n[2], 0.0]).collect();
        let accum_zeros = vec![0i32; count * 6];
        let position_buffer =
            self.storage_named("tangent-positions", bytemuck::cast_slice(&positions));
        let uv_buffer = self.storage_named("tangent-uvs", bytemuck::cast_slice(&uvs));
        let index_buffer =
            self.storage_named("tangent-indices", bytemuck::cast_slice(&primitive.indices));
        let accum_buffer = self.storage_named("tangent-accum", bytemuck::cast_slice(&accum_zeros));
        let normals_buffer = self.storage_named("tangent-normals", bytemuck::cast_slice(&normals4));
        let counts = Counts {
            triangles: triangle_count as u32,
            vertices: count as u32,
            pad0: 0,
            pad1: 0,
        };
        let counts_buffer = self.uniform_named("tangent-counts", &counts);
        let accum_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tangent-accum-bind-group"),
            layout: &self.tangents_accum_layout,
            entries: &[
                entry(0, &position_buffer),
                entry(1, &uv_buffer),
                entry(2, &index_buffer),
                entry(3, &accum_buffer),
                entry(4, &counts_buffer),
            ],
        });
        self.run(
            &self.tangents_accum_pipeline,
            &accum_bind,
            (triangle_count as u32).div_ceil(64),
        );
        let tangent_zeros = vec![[0.0f32; 4]; count];
        let tangent_buffer =
            self.storage_named("tangent-output", bytemuck::cast_slice(&tangent_zeros));
        let final_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tangent-final-bind-group"),
            layout: &self.tangents_final_layout,
            entries: &[
                entry(0, &accum_buffer),
                entry(1, &normals_buffer),
                entry(2, &tangent_buffer),
                entry(3, &counts_buffer),
            ],
        });
        self.run(
            &self.tangents_final_pipeline,
            &final_bind,
            (count as u32).div_ceil(64),
        );
        let Some(tangents): Option<Vec<[f32; 4]>> = self.readback(&tangent_buffer, count) else {
            return false;
        };
        primitive.tangents = Some(tangents);
        true
    }

    fn fits_storage(&self, bytes: usize) -> bool {
        bytes as u64 <= self.max_storage
    }

    fn fits_dispatch(&self, invocations: usize) -> bool {
        invocations.div_ceil(64) as u64 <= self.max_workgroups as u64
    }

    fn dummy(&self) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn storage_named(&self, label: &str, contents: &[u8]) -> wgpu::Buffer {
        let bytes = contents.to_vec();
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn uniform_named<T: Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(value),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn run(&self, pipeline: &wgpu::ComputePipeline, bind_group: &wgpu::BindGroup, groups: u32) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn readback<T: Pod>(&self, buffer: &wgpu::Buffer, count: usize) -> Option<Vec<T>> {
        let bytes = std::mem::size_of::<T>() * count;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes as u64);
        self.queue.submit(Some(encoder.finish()));
        let done = Arc::new(Mutex::new(false));
        let flag = done.clone();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if let Ok(mut guard) = flag.lock() {
                    *guard = result.is_ok();
                }
            });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if !*done.lock().ok()? {
            return None;
        }
        let data = staging.slice(..).get_mapped_range().ok()?;
        let values = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Some(values)
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn mat3_packed(matrix: Mat3) -> [[f32; 4]; 3] {
    [
        [matrix.x_axis.x, matrix.x_axis.y, matrix.x_axis.z, 0.0],
        [matrix.y_axis.x, matrix.y_axis.y, matrix.y_axis.z, 0.0],
        [matrix.z_axis.x, matrix.z_axis.y, matrix.z_axis.z, 0.0],
    ]
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn create(
    mode: GpuMode,
    verbose: bool,
    warnings: &mut WarningSink,
) -> Result<Option<GpuEngine>> {
    if mode == GpuMode::Off {
        return Ok(None);
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    })) {
        Ok(adapter) => adapter,
        Err(error) => {
            if mode == GpuMode::On {
                return Err(Error::conversion(format!(
                    "--gpu on: no GPU adapter available: {error}"
                )));
            }
            warnings.add(
                WarningKind::GpuFallback,
                "gpu-unavailable",
                "no GPU adapter available; using the CPU path",
            );
            return Ok(None);
        }
    };
    let (device, queue) =
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("trine"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })) {
            Ok(pair) => pair,
            Err(error) => {
                if mode == GpuMode::On {
                    return Err(Error::conversion(format!(
                        "--gpu on: cannot create GPU device: {error}"
                    )));
                }
                warnings.add(
                    WarningKind::GpuFallback,
                    "gpu-device",
                    "cannot create a GPU device; using the CPU path",
                );
                return Ok(None);
            }
        };
    if verbose {
        let info = adapter.get_info();
        eprintln!("gpu: {} ({:?})", info.name, info.backend);
    }

    let transform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("transform-layout"),
        entries: &[
            uniform_entry(0),
            storage_entry(1, false),
            storage_entry(2, false),
            storage_entry(3, false),
        ],
    });
    let normals_accum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("normals-accum-layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
        ],
    });
    let normals_final_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("normals-final-layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
    });
    let tangents_accum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tangents-accum-layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
        ],
    });
    let tangents_final_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tangents-final-layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });

    let pipeline =
        |label: &str, layout: &wgpu::BindGroupLayout, source: &str, entry_point: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

    let transform_pipeline = pipeline(
        "transform",
        &transform_layout,
        include_str!("shaders/transform.wgsl"),
        "main",
    );
    let normals_accum_pipeline = pipeline(
        "normals-accum",
        &normals_accum_layout,
        include_str!("shaders/normals_accum.wgsl"),
        "main",
    );
    let normals_final_pipeline = pipeline(
        "normals-final",
        &normals_final_layout,
        include_str!("shaders/normals_final.wgsl"),
        "main",
    );
    let tangents_accum_pipeline = pipeline(
        "tangents-accum",
        &tangents_accum_layout,
        include_str!("shaders/tangents_accum.wgsl"),
        "main",
    );
    let tangents_final_pipeline = pipeline(
        "tangents-final",
        &tangents_final_layout,
        include_str!("shaders/tangents_final.wgsl"),
        "main",
    );
    let limits = device.limits();
    let max_storage = limits.max_storage_buffer_binding_size;
    let max_workgroups = limits.max_compute_workgroups_per_dimension;
    Ok(Some(GpuEngine {
        mode,
        device,
        queue,
        transform_pipeline,
        transform_layout,
        normals_accum_pipeline,
        normals_accum_layout,
        normals_final_pipeline,
        normals_final_layout,
        tangents_accum_pipeline,
        tangents_accum_layout,
        tangents_final_pipeline,
        tangents_final_layout,
        max_storage,
        max_workgroups,
    }))
}
