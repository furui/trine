struct Params {
    model: mat4x4<f32>,
    normal: mat3x3<f32>,
    tangent: mat3x3<f32>,
    flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> tangents: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let count = u32(params.flags.x);
    if (index >= count) {
        return;
    }
    let position = positions[index];
    positions[index] = params.model * vec4<f32>(position.xyz, 1.0);

    let normal = normals[index];
    let transformed_normal = params.normal * normal.xyz;
    if (length(transformed_normal) > 0.0) {
        normals[index] = vec4<f32>(normalize(transformed_normal), 0.0);
    } else {
        normals[index] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    }

    let tangent = tangents[index];
    let transformed = params.tangent * tangent.xyz;
    if (length(transformed) > 0.0) {
        let sign = select(1.0, -1.0, params.flags.y > 0.5);
        tangents[index] = vec4<f32>(normalize(transformed), tangent.w * sign);
    }
}
