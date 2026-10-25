struct Counts {
    triangles: u32,
    vertices: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> sums: array<i32>;
@group(0) @binding(1) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> tangents: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> counts: Counts;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= counts.vertices) {
        return;
    }
    let base = index * 6u;
    let normal_raw = normals[index].xyz;
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    if (length(normal_raw) > 0.0) {
        normal = normalize(normal_raw);
    }
    var tangent = vec3<f32>(f32(sums[base]), f32(sums[base + 1u]), f32(sums[base + 2u]));
    tangent = tangent - normal * dot(normal, tangent);
    if (length(tangent) <= 0.0) {
        let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.x) > 0.9);
        tangent = cross(normal, helper);
    }
    tangent = normalize(tangent);
    let bitangent = vec3<f32>(f32(sums[base + 3u]), f32(sums[base + 4u]), f32(sums[base + 5u]));
    var handedness = 1.0;
    if (dot(cross(normal, tangent), bitangent) < 0.0) {
        handedness = -1.0;
    }
    tangents[index] = vec4<f32>(tangent, handedness);
}
