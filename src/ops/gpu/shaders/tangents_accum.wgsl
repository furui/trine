struct Counts {
    triangles: u32,
    vertices: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> uvs: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> accumulated: array<atomic<i32>>;
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
    let uv0 = uvs[i0];
    let uv1 = uvs[i1];
    let uv2 = uvs[i2];
    let edge1 = p1 - p0;
    let edge2 = p2 - p0;
    let delta1 = uv1 - uv0;
    let delta2 = uv2 - uv0;
    let determinant = delta1.x * delta2.y - delta2.x * delta1.y;
    if (abs(determinant) < 1.0e-12) {
        return;
    }
    let reciprocal = 1.0 / determinant;
    let tangent = (edge1 * delta2.y - edge2 * delta1.y) * reciprocal;
    let bitangent = (edge2 * delta1.x - edge1 * delta2.x) * reciprocal;
    let scale = 1000000.0;
    let qt = vec3<i32>(clamp(tangent * scale, vec3<f32>(-2.0e9), vec3<f32>(2.0e9)));
    let qb = vec3<i32>(clamp(bitangent * scale, vec3<f32>(-2.0e9), vec3<f32>(2.0e9)));
    for (var corner = 0u; corner < 3u; corner += 1u) {
        var vertex = i0;
        if (corner == 1u) { vertex = i1; }
        if (corner == 2u) { vertex = i2; }
        let base = vertex * 6u;
        atomicAdd(&accumulated[base], qt.x);
        atomicAdd(&accumulated[base + 1u], qt.y);
        atomicAdd(&accumulated[base + 2u], qt.z);
        atomicAdd(&accumulated[base + 3u], qb.x);
        atomicAdd(&accumulated[base + 4u], qb.y);
        atomicAdd(&accumulated[base + 5u], qb.z);
    }
}
