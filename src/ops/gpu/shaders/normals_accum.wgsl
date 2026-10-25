struct Counts {
    triangles: u32,
    vertices: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> weld: array<u32>;
@group(0) @binding(3) var<storage, read_write> accum: array<atomic<i32>>;
@group(0) @binding(4) var<uniform> counts: Counts;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let triangle = global_id.x;
    if (triangle >= counts.triangles) {
        return;
    }
    let i0 = indices[triangle * 3u];
    let i1 = indices[triangle * 3u + 1u];
    let i2 = indices[triangle * 3u + 2u];
    let p0 = positions[i0].xyz;
    let p1 = positions[i1].xyz;
    let p2 = positions[i2].xyz;
    let normal = cross(p1 - p0, p2 - p0);
    let scale = 1000000.0;
    let q = vec3<i32>(clamp(normal * scale, vec3<f32>(-2.0e9), vec3<f32>(2.0e9)));
    for (var corner = 0u; corner < 3u; corner += 1u) {
        var vertex = i0;
        if (corner == 1u) { vertex = i1; }
        if (corner == 2u) { vertex = i2; }
        let base = weld[vertex] * 3u;
        atomicAdd(&accum[base], q.x);
        atomicAdd(&accum[base + 1u], q.y);
        atomicAdd(&accum[base + 2u], q.z);
    }
}
