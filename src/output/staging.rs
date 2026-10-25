//! Temporary-directory staging with overwrite checks and near-atomic finalize.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use tempfile::TempDir;

use crate::error::{Error, Result};

/// Stage files in a temporary directory that lives next to the output.
///
/// Writers receive temporary paths and the staging area records the final
/// path for each one. Nothing is moved into place until [`Staging::finalize`]
/// succeeds, so a failed conversion does not leave half-written outputs.
pub struct Staging {
    temp: TempDir,
    entries: Vec<(PathBuf, PathBuf)>,
}

impl Staging {
    pub fn new(base_parent: &Path) -> Result<Self> {
        let base_parent = if base_parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            base_parent
        };
        fs::create_dir_all(base_parent).map_err(|error| Error::write(base_parent, error))?;
        let temp = tempfile::Builder::new()
            .prefix(".trine-tmp")
            .tempdir_in(base_parent)
            .map_err(|error| Error::write(base_parent, error))?;
        Ok(Self {
            temp,
            entries: Vec::new(),
        })
    }

    pub fn temp_root(&self) -> &Path {
        self.temp.path()
    }

    /// Register a final path and return the temporary path to write.
    pub fn stage(&mut self, final_path: &Path) -> Result<PathBuf> {
        let temp_path = self
            .temp
            .path()
            .join(format!("staged_{:04}", self.entries.len()));
        self.entries
            .push((temp_path.clone(), final_path.to_path_buf()));
        Ok(temp_path)
    }

    pub fn stage_bytes(&mut self, final_path: &Path, bytes: &[u8]) -> Result<()> {
        let temp_path = self.stage(final_path)?;
        fs::write(&temp_path, bytes).map_err(|error| Error::write(&temp_path, error))?;
        Ok(())
    }

    /// Fail early if any final path already exists and `--force` was not given.
    pub fn check_overwrite(&self, force: bool) -> Result<()> {
        if force {
            return Ok(());
        }
        let mut seen = HashSet::new();
        for (_, final_path) in &self.entries {
            if !seen.insert(final_path.clone()) {
                continue;
            }
            if final_path.exists() {
                return Err(Error::conversion(format!(
                    "output {} already exists; pass --force to overwrite",
                    final_path.display()
                )));
            }
        }
        Ok(())
    }

    /// Move every staged file into place.
    pub fn finalize(self, force: bool) -> Result<()> {
        self.check_overwrite(force)?;
        let mut moved: Vec<PathBuf> = Vec::new();
        for (temp_path, final_path) in &self.entries {
            if let Some(parent) = final_path.parent() {
                if let Err(error) = fs::create_dir_all(parent) {
                    rollback(&moved);
                    return Err(Error::write(parent, error));
                }
            }
            let result = if final_path.exists() {
                if force {
                    fs::remove_file(final_path).and_then(|()| fs::rename(temp_path, final_path))
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::AlreadyExists,
                        "output exists",
                    ))
                }
            } else {
                fs::rename(temp_path, final_path)
            };
            match result {
                Ok(()) => moved.push(final_path.clone()),
                Err(error) => {
                    rollback(&moved);
                    return Err(Error::write(final_path, error));
                }
            }
        }
        Ok(())
    }
}

fn rollback(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overwrite_check_and_force() {
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("out.txt");
        let mut staging = Staging::new(dir.path()).unwrap();
        staging.stage_bytes(&final_path, b"one").unwrap();
        staging.finalize(false).unwrap();
        assert_eq!(fs::read(&final_path).unwrap(), b"one");

        let mut staging = Staging::new(dir.path()).unwrap();
        staging.stage_bytes(&final_path, b"two").unwrap();
        assert!(staging.check_overwrite(false).is_err());
        staging.finalize(true).unwrap();
        assert_eq!(fs::read(&final_path).unwrap(), b"two");
    }

    #[test]
    fn failure_does_not_move_files() {
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("out.txt");
        fs::write(&final_path, b"old").unwrap();
        let mut staging = Staging::new(dir.path()).unwrap();
        staging.stage_bytes(&final_path, b"new").unwrap();
        assert!(staging.finalize(false).is_err());
        assert_eq!(fs::read(&final_path).unwrap(), b"old");
    }
}
