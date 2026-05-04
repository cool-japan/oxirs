//! Local filesystem model registry.
//!
//! Scans a base directory for `*.gguf` files and exposes each as a
//! [`ModelEntry`] of type [`ModelType::Unknown`] (GGUF files are typically
//! chat or completion models, but the type cannot be inferred from the filename
//! alone without parsing the file header).

use std::path::PathBuf;

use crate::marketplace::{MarketplaceError, ModelEntry, ModelRegistry, ModelSource, ModelType};

/// A registry that discovers GGUF models from the local filesystem.
pub struct LocalFileRegistry {
    /// Directory to scan for `*.gguf` files.
    base_path: PathBuf,
}

impl LocalFileRegistry {
    /// Create a registry that will scan `base_path` for GGUF files.
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    /// Scan `base_path` for `*.gguf` files and return a [`ModelEntry`] for each.
    ///
    /// Returns an empty list (not an error) when the directory exists but
    /// contains no GGUF files.
    pub fn scan(&self) -> Result<Vec<ModelEntry>, MarketplaceError> {
        let read_dir = std::fs::read_dir(&self.base_path).map_err(|e| {
            MarketplaceError::Io(format!(
                "cannot read directory '{}': {e}",
                self.base_path.display()
            ))
        })?;

        let mut entries = Vec::new();

        for dir_entry in read_dir {
            let dir_entry = dir_entry.map_err(|e| {
                MarketplaceError::Io(format!(
                    "error reading directory entry in '{}': {e}",
                    self.base_path.display()
                ))
            })?;

            let path = dir_entry.path();

            // Only process regular files with the `.gguf` extension.
            if !path.is_file() {
                continue;
            }

            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default()
                .to_lowercase();

            if ext != "gguf" {
                continue;
            }

            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| {
                    MarketplaceError::InvalidPath(format!(
                        "non-UTF-8 filename in '{}'",
                        self.base_path.display()
                    ))
                })?
                .to_string();

            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&file_name)
                .to_string();

            // Use the file path as a unique, stable identifier.
            let id = format!("local::{}", path.display());

            let size_bytes = dir_entry.metadata().ok().map(|m| m.len());

            entries.push(ModelEntry {
                id,
                name: stem.clone(),
                source: ModelSource::LocalFile { path: path.clone() },
                model_type: ModelType::Unknown,
                size_bytes,
                tags: vec!["gguf".to_string(), "local".to_string()],
                description: format!("Local GGUF model file: {file_name}"),
                download_url: None,
            });
        }

        // Sort for deterministic output (useful in tests).
        entries.sort_by(|a, b| a.id.cmp(&b.id));

        Ok(entries)
    }

    /// Base path configured for this registry.
    pub fn base_path(&self) -> &std::path::Path {
        &self.base_path
    }
}

impl ModelRegistry for LocalFileRegistry {
    fn list_models(&self) -> Result<Vec<ModelEntry>, MarketplaceError> {
        self.scan()
    }

    fn search(&self, query: &str) -> Result<Vec<ModelEntry>, MarketplaceError> {
        let q = query.to_lowercase();
        let all = self.scan()?;
        let results = all
            .into_iter()
            .filter(|entry| {
                entry.id.to_lowercase().contains(&q)
                    || entry.name.to_lowercase().contains(&q)
                    || entry.tags.iter().any(|t| t.to_lowercase().contains(&q))
                    || entry.description.to_lowercase().contains(&q)
            })
            .collect();
        Ok(results)
    }

    fn get_model(&self, id: &str) -> Result<Option<ModelEntry>, MarketplaceError> {
        Ok(self.scan()?.into_iter().find(|e| e.id == id))
    }

    fn source_name(&self) -> &'static str {
        "Local Filesystem"
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;

    #[test]
    fn test_local_registry_scan_empty_dir() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let registry = LocalFileRegistry::new(dir.path());
        let models = registry.scan().expect("scan should succeed on empty dir");
        assert!(
            models.is_empty(),
            "empty directory should yield zero models"
        );
    }

    #[test]
    fn test_local_registry_scan_gguf_files() {
        let dir = tempfile::tempdir().expect("tempdir should be created");

        // Create fake .gguf files (content doesn't matter for discovery).
        File::create(dir.path().join("llama-2-7b.Q4_K_M.gguf")).expect("create first gguf");
        File::create(dir.path().join("mistral-7b.Q4_K_M.gguf")).expect("create second gguf");
        // This should NOT be picked up (wrong extension).
        File::create(dir.path().join("README.txt")).expect("create non-gguf file");

        let registry = LocalFileRegistry::new(dir.path());
        let models = registry.scan().expect("scan should succeed");

        assert_eq!(models.len(), 2, "only .gguf files should be registered");

        // Verify that none of the entries correspond to the txt file.
        for entry in &models {
            assert!(
                entry.description.ends_with(".gguf"),
                "entry description should reference a .gguf file"
            );
        }
    }

    #[test]
    fn test_local_registry_nonexistent_dir_errors() {
        let registry = LocalFileRegistry::new("/this/path/does/not/exist/at/all");
        let result = registry.scan();
        assert!(result.is_err(), "scanning nonexistent dir should error");
        if let Err(MarketplaceError::Io(msg)) = result {
            assert!(
                msg.contains("cannot read directory"),
                "error message mismatch: {msg}"
            );
        } else {
            panic!("expected MarketplaceError::Io");
        }
    }

    #[test]
    fn test_local_registry_source_name() {
        let registry = LocalFileRegistry::new("/tmp");
        assert_eq!(registry.source_name(), "Local Filesystem");
    }
}
