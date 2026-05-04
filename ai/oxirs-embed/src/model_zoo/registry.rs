//! Global model zoo registry populated from embedded TOML manifests.
//!
//! The registry ships five built-in catalog entries (synthetic seed weights for
//! testing).  Users can also supply a directory of additional `.toml` files at
//! runtime via [`ModelZoo::with_manifest_dir`].

use std::collections::HashMap;
use std::path::Path;

use once_cell::sync::Lazy;

use crate::model_zoo::loader::ModelZooError;
use crate::model_zoo::manifest::ModelManifest;

// ---------------------------------------------------------------------------
// Embedded manifests (compile-time)
// ---------------------------------------------------------------------------

const TRANSE_FB15K237: &str = include_str!("manifests/transe-fb15k237.toml");
const TRANSE_WN18RR: &str = include_str!("manifests/transe-wn18rr.toml");
const ROTATE_FB15K237: &str = include_str!("manifests/rotate-fb15k237.toml");
const COMPLEX_WN18RR: &str = include_str!("manifests/complex-wn18rr.toml");
const DISTMULT_FB15K237: &str = include_str!("manifests/distmult-fb15k237.toml");

const BUILTIN_MANIFESTS: &[&str] = &[
    TRANSE_FB15K237,
    TRANSE_WN18RR,
    ROTATE_FB15K237,
    COMPLEX_WN18RR,
    DISTMULT_FB15K237,
];

// ---------------------------------------------------------------------------
// ModelZoo
// ---------------------------------------------------------------------------

/// Registry of pretrained (or synthetic-seed) embedding model manifests.
///
/// The global default registry is lazily initialised once from the five
/// manifests embedded at compile time.  Call [`ModelZoo::registry()`] to
/// access it.  For custom manifests, use [`ModelZoo::with_manifest_dir`].
pub struct ModelZoo {
    entries: HashMap<String, ModelManifest>,
}

static GLOBAL_ZOO: Lazy<ModelZoo> = Lazy::new(|| {
    ModelZoo::from_embedded()
        // Embedded manifests are always valid TOML (compile-time guarantee);
        // panicking here is intentional: a broken built-in manifest is a
        // programming error, not a runtime error.
        .unwrap_or_else(|e| panic!("Failed to parse built-in model zoo manifests: {e}"))
});

impl ModelZoo {
    /// Parse all embedded manifests into a new registry.
    fn from_embedded() -> Result<Self, ModelZooError> {
        let mut entries = HashMap::new();
        for raw in BUILTIN_MANIFESTS {
            let manifest: ModelManifest =
                toml::from_str(raw).map_err(|e| ModelZooError::ManifestParse(e.to_string()))?;
            entries.insert(manifest.name.clone(), manifest);
        }
        Ok(Self { entries })
    }

    /// Return a reference to the global default registry.
    ///
    /// The registry is initialised exactly once (lazily, on first call) and is
    /// safe to call from multiple threads simultaneously.
    pub fn registry() -> &'static ModelZoo {
        &GLOBAL_ZOO
    }

    /// Build a registry from all `.toml` files found in `dir`.
    ///
    /// Files that fail to parse are skipped with a warning rather than causing
    /// the whole registry construction to fail, so that a single malformed
    /// manifest does not prevent access to valid ones.
    pub fn with_manifest_dir(dir: &Path) -> Result<ModelZoo, ModelZooError> {
        let mut entries = HashMap::new();

        let read_dir = std::fs::read_dir(dir)
            .map_err(|e| ModelZooError::Io(std::io::Error::new(e.kind(), e.to_string())))?;

        for entry in read_dir {
            let entry = entry
                .map_err(|e| ModelZooError::Io(std::io::Error::new(e.kind(), e.to_string())))?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) != Some("toml") {
                continue;
            }

            let raw = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("Skipping unreadable manifest {:?}: {}", path, e);
                    continue;
                }
            };

            match toml::from_str::<ModelManifest>(&raw) {
                Ok(manifest) => {
                    entries.insert(manifest.name.clone(), manifest);
                }
                Err(e) => {
                    tracing::warn!("Skipping invalid manifest {:?}: {}", path, e);
                }
            }
        }

        Ok(Self { entries })
    }

    /// Look up a manifest by exact name.
    pub fn get(&self, name: &str) -> Option<&ModelManifest> {
        self.entries.get(name)
    }

    /// List all manifests in the registry (order is unspecified).
    pub fn list(&self) -> Vec<&ModelManifest> {
        self.entries.values().collect()
    }

    /// Case-insensitive substring search across `name`, `dataset`, and
    /// `model_type` fields.
    pub fn search(&self, query: &str) -> Vec<&ModelManifest> {
        let q = query.to_lowercase();
        self.entries
            .values()
            .filter(|m| {
                m.name.to_lowercase().contains(&q)
                    || m.dataset.to_lowercase().contains(&q)
                    || m.model_type.to_lowercase().contains(&q)
            })
            .collect()
    }

    /// Number of entries in the registry.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the registry contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_registry_has_five_entries() {
        let zoo = ModelZoo::registry();
        assert_eq!(zoo.len(), 5, "Expected 5 built-in entries");
    }

    #[test]
    fn test_registry_get_existing() {
        let zoo = ModelZoo::registry();
        let m = zoo.get("transe-fb15k237");
        assert!(m.is_some());
        let m = m.expect("entry should exist");
        assert_eq!(m.model_type, "TransE");
        assert_eq!(m.dimensions, 200);
    }

    #[test]
    fn test_registry_get_missing() {
        let zoo = ModelZoo::registry();
        assert!(zoo.get("nonexistent-model").is_none());
    }

    #[test]
    fn test_registry_list() {
        let zoo = ModelZoo::registry();
        let list = zoo.list();
        assert_eq!(list.len(), 5);
    }

    #[test]
    fn test_registry_search_by_dataset() {
        let zoo = ModelZoo::registry();
        let fb = zoo.search("FB15k");
        // Three models use FB15k-237
        assert_eq!(fb.len(), 3, "FB15k-237 should have 3 entries");
    }

    #[test]
    fn test_registry_search_by_model_type() {
        let zoo = ModelZoo::registry();
        let transe_models = zoo.search("transe");
        // Two TransE models (fb15k237 + wn18rr)
        assert_eq!(transe_models.len(), 2);
    }

    #[test]
    fn test_registry_search_case_insensitive() {
        let zoo = ModelZoo::registry();
        let upper = zoo.search("TRANSE");
        let lower = zoo.search("transe");
        assert_eq!(upper.len(), lower.len());
    }

    #[test]
    fn test_registry_search_no_results() {
        let zoo = ModelZoo::registry();
        let results = zoo.search("zzzznotexist");
        assert!(results.is_empty());
    }

    #[test]
    fn test_with_manifest_dir_empty() {
        let tmp = std::env::temp_dir().join("oxirs_zoo_test_empty");
        std::fs::create_dir_all(&tmp).expect("create temp dir");
        let zoo = ModelZoo::with_manifest_dir(&tmp).expect("empty dir ok");
        assert!(zoo.is_empty());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_with_manifest_dir_single_manifest() {
        let tmp = std::env::temp_dir().join("oxirs_zoo_test_single");
        std::fs::create_dir_all(&tmp).expect("create temp dir");

        let manifest = crate::model_zoo::manifest::ModelManifest {
            name: "custom-model".to_string(),
            model_type: "TransE".to_string(),
            dataset: "CustomDS".to_string(),
            dimensions: 64,
            entities: 100,
            relations: 5,
            sha256: "PLACEHOLDER".to_string(),
            source: "file:///tmp/custom.ckpt".to_string(),
            license: "Apache-2.0".to_string(),
            citation: "Custom 2026".to_string(),
            version: "1.0.0".to_string(),
            created: "2026-05-01".to_string(),
            notes: None,
        };

        let toml_str = toml::to_string(&manifest).expect("serialize");
        std::fs::write(tmp.join("custom-model.toml"), toml_str).expect("write");

        let zoo = ModelZoo::with_manifest_dir(&tmp).expect("parse manifest dir");
        assert_eq!(zoo.len(), 1);
        let m = zoo.get("custom-model").expect("should be present");
        assert_eq!(m.model_type, "TransE");

        std::fs::remove_dir_all(&tmp).ok();
    }
}
