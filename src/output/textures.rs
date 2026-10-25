//! Texture staging: sanitized filenames, de-duplication, and byte copies.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::output::staging::Staging;
use crate::report::WarningSink;
use crate::scene::{Image, ImageId, Scene};

/// Assigns stable, sanitized file names to images and copies their encoded
/// bytes into the staged texture directory exactly once.
#[derive(Debug, Default)]
pub struct TexturePlan {
    assignments: HashMap<(PathBuf, String, u64), String>,
    used_names: HashMap<PathBuf, HashSet<String>>,
    staged: HashSet<PathBuf>,
}

impl TexturePlan {
    pub fn new() -> Self {
        Self::default()
    }

    /// Copy the referenced images for one output that lives in
    /// `output_path.parent()/dir_name` and return relative references.
    pub fn stage_for(
        &mut self,
        staging: &mut Staging,
        warnings: &mut WarningSink,
        scene: &Scene,
        output_path: &Path,
        dir_name: &str,
        image_ids: &[ImageId],
    ) -> Result<HashMap<ImageId, String>> {
        let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
        let safe_dir = sanitize_path_component(dir_name, "textures");
        let final_dir = parent.join(&safe_dir);
        let mut result = HashMap::new();
        for image_id in image_ids {
            let Some(image) = scene.images.get(*image_id) else {
                warnings.add(
                    crate::report::WarningKind::MissingAsset,
                    "texture-index",
                    format!("texture index {image_id} is not present in the scene"),
                );
                continue;
            };
            let key = (final_dir.clone(), image.name.clone(), content_hash(image));
            if let Some(existing) = self.assignments.get(&key) {
                result.insert(*image_id, existing.clone());
                continue;
            }
            let file_name = self.file_name_for(&final_dir, image);
            let relative = format!("{safe_dir}/{file_name}");
            let final_path = final_dir.join(&file_name);
            if self.staged.insert(final_path.clone()) {
                staging.stage_bytes(&final_path, &image.bytes)?;
            }
            self.assignments.insert(key, relative.clone());
            result.insert(*image_id, relative);
        }
        Ok(result)
    }

    fn file_name_for(&mut self, dir: &Path, image: &Image) -> String {
        let stem_source = Path::new(&image.name)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.is_empty())
            .unwrap_or("texture");
        let stem = sanitize_path_component(stem_source, "texture");
        let stem = if stem.is_empty() {
            "texture".to_string()
        } else {
            stem
        };
        let ext = image.extension();
        let used = self.used_names.entry(dir.to_path_buf()).or_default();
        let base = format!("{stem}.{ext}");
        if used.insert(base.clone()) {
            return base;
        }
        let mut index = 2usize;
        loop {
            let candidate = format!("{stem}_{index}.{ext}");
            if used.insert(candidate.clone()) {
                return candidate;
            }
            index += 1;
        }
    }
}

fn content_hash(image: &Image) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    image.name.hash(&mut hasher);
    image.mime.hash(&mut hasher);
    image.bytes.hash(&mut hasher);
    hasher.finish()
}

/// Restrict a name to `[A-Za-z0-9._-]`, collapse repeats, trim, and cap length.
pub fn sanitize_path_component(name: &str, fallback: &str) -> String {
    let mut out = String::new();
    let mut last_repeat = false;
    for ch in name.chars() {
        let allowed = ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-');
        if allowed {
            if matches!(ch, '.' | '_' | '-') {
                if last_repeat {
                    continue;
                }
                last_repeat = true;
            } else {
                last_repeat = false;
            }
            out.push(ch);
            if out.len() >= 80 {
                break;
            }
        }
    }
    let trimmed = out
        .trim_matches(|ch| matches!(ch, '.' | '_' | '-'))
        .to_string();
    if trimmed.is_empty() {
        fallback.to_string()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Format;
    use crate::scene::{Image, Scene, Unit};

    fn scene_with_images(names: &[(&str, &[u8])]) -> Scene {
        let mut scene = Scene::new(Format::Obj);
        scene.unit = crate::scene::UnitResolution::Known(Unit::Millimeter);
        for (name, bytes) in names {
            scene.add_image(Image {
                name: name.to_string(),
                bytes: bytes.to_vec(),
                mime: Some("image/png".into()),
                source_path: None,
            });
        }
        scene
    }

    #[test]
    fn sanitizes_traversal_names() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("out.glb");
        let scene = scene_with_images(&[("../../etc/passwd", b"a")]);
        let mut staging = Staging::new(dir.path()).unwrap();
        let mut warnings = WarningSink::new(true);
        let mut plan = TexturePlan::new();
        let paths = plan
            .stage_for(
                &mut staging,
                &mut warnings,
                &scene,
                &output,
                "textures",
                &[0],
            )
            .unwrap();
        let path = &paths[&0];
        assert!(!path.contains(".."), "{path}");
        assert!(path.starts_with("textures/"));
        assert!(!Path::new(path).is_absolute());
    }

    #[test]
    fn distinct_images_with_the_same_name_get_suffixes() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("out.glb");
        let scene = scene_with_images(&[("same.png", b"a"), ("same.png", b"b")]);
        let mut staging = Staging::new(dir.path()).unwrap();
        let mut warnings = WarningSink::new(true);
        let mut plan = TexturePlan::new();
        let paths = plan
            .stage_for(
                &mut staging,
                &mut warnings,
                &scene,
                &output,
                "textures",
                &[0, 1],
            )
            .unwrap();
        assert_ne!(paths[&0], paths[&1]);
        assert_eq!(paths[&0], "textures/same.png");
        assert_eq!(paths[&1], "textures/same_2.png");
    }
}
