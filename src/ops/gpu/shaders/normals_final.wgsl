struct Counts {
    triangles: u32,
    vertices: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> summed: array<i32>;
@group(0) @binding(1) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> counts: Counts;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= counts.vertices) {
        return;
    }
    let base = index * 3u;
    let value = vec3<f32>(f32(summed[base]), f32(summed[base + 1u]), f32(summed[base + 2u]));
    var normal = vec3<f32>(0.0, 0.0, 0.0);
    if (length(value) > 0.0) {
        normal = normalize(value);
    }
    normals[index] = vec4<f32>(normal, 0.0);
}
