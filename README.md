# trine

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org) [![AI Assisted](https://www.aihonestybadge.com/badges/ai-assisted-small.svg)](https://www.aihonestybadge.com)

`trine` is a Rust CLI that converts static triangle-mesh scenes between FBX,
3MF, STL, OBJ, OFF, glTF, GLB, DAE (COLLADA), PLY, and STEP.

## Behavior

- Resolves units, scale, and up-axis, with explicit overrides.
- Keeps the node hierarchy where the target supports it and bakes transforms
  where it does not.
- Copies encoded texture bytes without decoding them.
- Writes one output per group with `--split-groups`.
- Reports unsupported or approximated data. `--strict` fails on data loss.
- Uses wgpu for vertex transforms, smooth normals, and tangent generation on
  large meshes, with a CPU fallback.

## Install

```sh
cargo install --path .
# CPU-only build:
cargo install --path . --no-default-features
```

Requirements: Rust 1.87 or newer (edition 2024), a C compiler for the vendored
`ufbx` sources, and on Linux the usual Vulkan/GL headers if you build with the
`gpu` feature (enabled by default).

## Usage

```
trine <INPUT> <OUTPUT> [OPTIONS]
```

| Option | Meaning |
| --- | --- |
| `-f, --from <FORMAT>` | Input format; `auto` (default) uses the extension |
| `-t, --to <FORMAT>` | Output format; defaults to the output extension |
| `--split-groups` | Write one file per source group/node/object |
| `--scale <FACTOR>` | Uniform scale applied after unit conversion |
| `--scale-axes X,Y,Z` | Per-axis scale multipliers |
| `--unit-in <UNIT>` | Source units: `um`, `mm`, `cm`, `m`, `in`, `ft` |
| `--unit-out <UNIT>` | Target units |
| `--no-unit-conversion` | Preserve raw coordinates; disable unit defaults |
| `--up-axis <AXIS>` | Target up axis: `y` or `z` |
| `--source-up-axis <AXIS>` | Override the detected source up axis |
| `--keep-axes` | Disable up-axis conversion |
| `--normals <MODE>` | `preserve` (default), `smooth`, `flat`, `none` |
| `--gpu <MODE>` | `auto` (default), `on`, `off` |
| `--tessellation-tolerance <V>` | STEP tessellation tolerance in source units |
| `--force` | Overwrite existing outputs |
| `--strict` | Exit 1 if the conversion would lose input data |
| `-q, --quiet` | Suppress the summary and warnings |
| `-v, --verbose` | Print format, unit, and axis diagnostics |

Exit codes: `0` success, `1` conversion, validation, or strict failure, `2`
command line usage error. Outputs are staged in a temporary directory and moved
into place after the conversion succeeds.

### Units

| Formats | Behavior |
| --- | --- |
| OBJ, STL, PLY, OFF | No unit metadata. Input is assumed to be millimeters. |
| FBX | Reads `UnitScaleFactor`; assumes centimeters only if absent. |
| 3MF | Reads and writes the 3MF unit attribute (default millimeters). |
| DAE | Reads and writes `<unit meter="...">` (default meters). |
| glTF / GLB | Always meters. `--unit-out` other than `m` is rejected. |
| STEP | Reads SI and conversion-based length units; writes the resolved target unit. |

`--no-unit-conversion` keeps raw numbers. With glTF/GLB the file still declares
meters, so a note is printed. Unit assumptions are warnings and do not fail
`--strict`. Loss of unit metadata when writing a unitless target does.

### Axes

Format defaults are Y-up for FBX, DAE, glTF, GLB, and OBJ; Z-up for 3MF, STEP,
STL, PLY, and OFF. Geometry is rotated only when the source and target
conventions differ. STEP and 3MF metadata describe their conventional Z-up
coordinates.

### Strict mode

`--strict` fails when a conversion would drop or approximate input data:
materials, textures, UVs, vertex colors, normals, tangents, hierarchy, animation,
skinning, morph targets, cameras, lights, constraints, or unsupported shader
graphs. Unit assumptions, nonstandard but valid choices, and GPU fallbacks are
reported but do not fail strict mode.

### Splitting

`--split-groups` writes one file per non-empty node/object. With an output
pattern containing `{group}`, each piece replaces it with the sanitized node
name. Without a pattern, names are `<output_stem>_<group>.<ext>`, or
`<input_stem>_<group>.<ext>` when the output is a directory. Empty groups are
skipped; duplicate names get `_2`, `_3`; empty names become `group_0001`.

STL, PLY, and OFF have no group concept and ignore splitting: they write one
merged file and warn.

## Format support

| | Read | Write | Normals | Tangents | UVs | Vertex colors | Materials | Textures | Units | Hierarchy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FBX | yes | yes (7.4 ASCII) | yes | yes | yes | yes | PBR flattened to Phong | copied | yes | yes |
| 3MF | yes | yes | – | – | – | – | base color | – | yes | components |
| STL | yes | yes (binary) | facet normals only | – | – | – | – | – | – | – |
| OBJ | yes | yes + MTL | yes | – | yes | – | Phong | copied | – | – |
| OFF | yes (COFF) | yes (COFF) | – | – | – | yes | – | – | – | – |
| glTF | yes | yes | yes | yes | yes | yes | PBR metallic-roughness | copied/embedded | m only | yes |
| GLB | yes | yes | yes | yes | yes | yes | PBR metallic-roughness | embedded | m only | yes |
| DAE | yes | yes (1.4.1) | yes | – | yes | yes | Phong | base color | yes | yes |
| PLY | yes | yes (binary LE) | yes | – | yes | yes | – | – | – | – |
| STEP | yes (tessellated) | yes (AP214 faceted) | – | – | – | – | – | – | yes | flattened |

Notes:

- **FBX** import uses the vendored `ufbx` C library and supports binary and
  ASCII FBX from version 3000 onward. Export writes FBX 7.4 ASCII with models,
  geometry layers, Phong materials, and external textures. Shader graphs are
  flattened; skinning, animation, morphs, cameras, lights, and constraints are
  reported and skipped.
- **STEP** import tessellates B-rep shells with `truck`; the tolerance is
  `--tessellation-tolerance`, or 0.1% of the shell diagonal by default. STEP
  output is an AP214 `AUTOMOTIVE_DESIGN` file containing a faceted B-rep
  (`MANIFOLD_SOLID_BREP` + `CLOSED_SHELL` for closed primitives,
  `SHELL_BASED_SURFACE_MODEL` + `OPEN_SHELL` otherwise). Only geometry and units
  are written; materials, UVs, and vertex normals are reported as dropped.
- **DAE** import covers `<triangles>`, `<polylist>`, `<polygons>`, `<trifans>`,
  and `<tristrips>`, node matrices/TRS, materials, and diffuse textures. Export
  writes Phong materials and base-color textures. Tangents are not part of
  COLLADA core and are dropped.
- **3MF** core only: no textures, UVs, normals, or vertex colors. Base colors
  are preserved. Objects with several materials are split into component
  objects so each face keeps its color.
- **STL** import does not create vertex normals; the writer emits the geometric
  facet normal.
- **PLY** writes binary little-endian with `x/y/z`, optional `nx/ny/nz`, `s/t`,
  and `red/green/blue/alpha`.
- **OBJ** export writes a sibling `.mtl`. PBR metallic values are approximated
  with the Phong model and reported.
- **GLB** is used for embedded buffers and images; `.gltf` writes a sibling
  `.bin` and a texture directory. Image bytes are copied, never decoded or
  re-encoded.

## GPU acceleration

The default build enables the `gpu` feature. With `--gpu auto` (the default),
the GPU is used for meshes with at least 50,000 vertices when an adapter is
available; otherwise the CPU path runs. `--gpu on` requires an adapter and fails
if none exists. `--gpu off`, and builds with `--no-default-features`, do not
reference wgpu symbols.

Compute work:

- vertex positions, normals, and tangents during transform baking;
- area-weighted smooth normals using fixed-point atomic accumulation;
- per-vertex tangent/bitangent accumulation and Gram-Schmidt orthonormalization.

CPU and GPU results are compared in `tests/gpu.rs`; run that gated test with
`TRINE_TEST_GPU=1 cargo test --test gpu` on a machine with an adapter.
Oversized buffers, mapping failures, or a missing adapter fall back to the CPU
instead of producing partial results.

## Architecture

```
src/
  cli.rs             argument parsing and validation
  convert.rs         end-to-end pipeline
  error.rs           exit-code-aware errors
  report.rs          warning taxonomy, de-duplication, strict accounting
  scene/             canonical scene, units, transforms, triangulation,
                     validation, topology
  io/                one module per format; reads/writes the canonical scene
  ops/               capability filtering, flattening, splitting, normals,
                     tangents, GPU acceleration
  output/            staged writes and texture planning
```

The canonical `Scene` is format-neutral: nodes with local transforms, meshes
with one or more primitives, core PBR materials, encoded images, and
unit/up-axis metadata. Readers triangulate and do not apply unit defaults;
defaults are resolved once in `convert.rs`. Writers receive a scene filtered for
their capabilities, so warnings are reported once per feature category rather
than per triangle.

Generated group and texture file names are sanitized (`[A-Za-z0-9._-]`), and
texture references are relative to the output file.

## Testing

```sh
cargo test                      # default (GPU) build
cargo test --all-features
cargo test --no-default-features --lib
TRINE_TEST_GPU=1 cargo test --test gpu
cargo build --release --no-default-features
```

Tests generate fixtures at runtime and validate writers by reading their output
back with native readers (`ufbx`, the `trine` glTF reader, the custom 3MF and
PLY readers, `dae-parser`, `tobj`, `stl_io`, and `ruststep`/`truck`). There are
no checked-in binary fixtures. Unit tests cover unit and axis math, warning
de-duplication, strict accounting, triangulation, topology, and PLY/OBJ details.
CLI tests cover exit codes, `--strict`, `--quiet`, `--force`, unit conversion,
axis conversion, and group splitting.

## Limitations

- Static triangle meshes only. Lines, points, NURBS/Bezier surfaces, skinning,
  morph targets, animation, cameras, lights, and constraints are not converted.
- One UV set, one vertex color set, and core PBR materials. Format-specific
  material graphs are flattened and reported.
- STEP import is a tessellating importer, not a B-rep kernel. STEP export is
  faceted and does not carry analytic surfaces.
- DAE import keeps the diffuse texture; other texture slots are not imported.
  DAE export writes only base-color textures.
- Texture images are copied byte-for-byte and never validated or transcoded.
- FBX export is ASCII 7.4 only; binary FBX output is not implemented.

## License

MIT. See [LICENSE](LICENSE).
