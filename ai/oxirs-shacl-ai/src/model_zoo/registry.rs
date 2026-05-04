//! Global registry of pretrained SHACL shape-learning model manifests.
//!
//! [`ShapeModelZoo`] holds an in-memory catalogue of [`ShapeModelManifest`]
//! entries keyed by name.  The global static instance is pre-populated from
//! four TOML manifests that are *compiled into the binary* via `include_str!`.
//!
//! Additional manifests can be loaded at runtime from a directory on disk
//! using [`ShapeModelZoo::with_manifest_dir`].

use std::collections::HashMap;
use std::path::Path;

use once_cell::sync::Lazy;

use crate::model_zoo::{loader::ShapeModelZooError, manifest::ShapeModelManifest};

// ---------------------------------------------------------------------------
// ShapeModelZoo
// ---------------------------------------------------------------------------

/// In-memory catalogue of pretrained SHACL shape-learning model manifests.
pub struct ShapeModelZoo {
    entries: HashMap<String, ShapeModelManifest>,
}

impl ShapeModelZoo {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Return a reference to the global static registry that is initialised
    /// from the four manifests bundled into the binary.
    pub fn registry() -> &'static ShapeModelZoo {
        &REGISTRY
    }

    /// Clone the global static registry into an owned value.
    ///
    /// Used by [`crate::model_zoo::loader::ShapeModelZooLoader::new`] so that
    /// the loader can own its zoo without holding a `&'static` reference,
    /// which would make testing impossible.
    pub fn clone_registry() -> ShapeModelZoo {
        ShapeModelZoo {
            entries: REGISTRY.entries.clone(),
        }
    }

    /// Build a [`ShapeModelZoo`] by scanning `dir` for `*.toml` files and
    /// parsing each as a [`ShapeModelManifest`].
    ///
    /// Entries from disk **override** entries with the same name that are
    /// already present in the zoo (useful for pinning local checkpoints during
    /// development).
    pub fn with_manifest_dir(dir: &Path) -> Result<ShapeModelZoo, ShapeModelZooError> {
        let mut zoo = ShapeModelZoo {
            entries: HashMap::new(),
        };

        let rd = std::fs::read_dir(dir)?;
        for entry in rd {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let src = std::fs::read_to_string(&path)?;
            let manifest: ShapeModelManifest = toml::from_str(&src)?;
            zoo.entries.insert(manifest.name.clone(), manifest);
        }

        Ok(zoo)
    }

    /// Build a zoo containing a single entry.  Primarily useful in tests.
    pub fn with_single_entry(manifest: ShapeModelManifest) -> ShapeModelZoo {
        let mut entries = HashMap::new();
        entries.insert(manifest.name.clone(), manifest);
        ShapeModelZoo { entries }
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Look up a manifest by its unique registry name.
    pub fn get(&self, name: &str) -> Option<&ShapeModelManifest> {
        self.entries.get(name)
    }

    /// Return all manifests, sorted deterministically by name.
    pub fn list(&self) -> Vec<&ShapeModelManifest> {
        let mut v: Vec<&ShapeModelManifest> = self.entries.values().collect();
        v.sort_by_key(|m| m.name.as_str());
        v
    }

    /// Return all manifests whose `name`, `model_type`, `dataset`, or
    /// `citation` fields contain `query` as a case-insensitive substring.
    pub fn search(&self, query: &str) -> Vec<&ShapeModelManifest> {
        let q = query.to_lowercase();
        let mut v: Vec<&ShapeModelManifest> = self
            .entries
            .values()
            .filter(|m| {
                m.name.to_lowercase().contains(&q)
                    || m.model_type.to_lowercase().contains(&q)
                    || m.dataset.to_lowercase().contains(&q)
                    || m.citation.to_lowercase().contains(&q)
            })
            .collect();
        v.sort_by_key(|m| m.name.as_str());
        v
    }

    /// Return all manifests whose `model_type` field matches `model_type`
    /// (case-insensitive).
    pub fn by_model_type(&self, model_type: &str) -> Vec<&ShapeModelManifest> {
        let mt = model_type.to_lowercase();
        let mut v: Vec<&ShapeModelManifest> = self
            .entries
            .values()
            .filter(|m| m.model_type.to_lowercase() == mt)
            .collect();
        v.sort_by_key(|m| m.name.as_str());
        v
    }

    /// Return the total number of entries in this zoo.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Global static registry
// ---------------------------------------------------------------------------

/// The built-in TOML manifests, compiled into the binary at build time.
static REGISTRY: Lazy<ShapeModelZoo> = Lazy::new(|| {
    let sources = [
        include_str!("manifests/gat-shacl-base.toml"),
        include_str!("manifests/graphsage-shacl-base.toml"),
        include_str!("manifests/graphormer-shacl-base.toml"),
        include_str!("manifests/gt-shacl-base.toml"),
    ];

    let mut zoo = ShapeModelZoo {
        entries: HashMap::new(),
    };

    for src in &sources {
        match toml::from_str::<ShapeModelManifest>(src) {
            Ok(m) => {
                zoo.entries.insert(m.name.clone(), m);
            }
            Err(e) => {
                // At this stage the tracing subscriber may not yet be
                // initialised, so we fall back to stderr.
                tracing::warn!("failed to parse built-in manifest: {e}");
            }
        }
    }

    zoo
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_four_entries() {
        assert_eq!(ShapeModelZoo::registry().list().len(), 4);
    }

    #[test]
    fn test_registry_get_existing() {
        let m = ShapeModelZoo::registry().get("gat-shacl-base");
        assert!(m.is_some());
        assert_eq!(m.expect("should be Some").model_type, "GAT");
    }

    #[test]
    fn test_registry_get_missing() {
        assert!(ShapeModelZoo::registry().get("nonexistent").is_none());
    }

    #[test]
    fn test_search_by_substring() {
        let results = ShapeModelZoo::registry().search("shacl");
        assert!(
            !results.is_empty(),
            "search('shacl') must return ≥ 1 result"
        );
    }

    #[test]
    fn test_by_model_type_gat() {
        let results = ShapeModelZoo::registry().by_model_type("GAT");
        assert!(
            !results.is_empty(),
            "by_model_type('GAT') must return ≥ 1 result"
        );
    }

    #[test]
    fn test_by_model_type_case_insensitive() {
        let a = ShapeModelZoo::registry().by_model_type("gat");
        let b = ShapeModelZoo::registry().by_model_type("GAT");
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn test_with_manifest_dir() {
        let tmp = std::env::temp_dir().join("shacl_ai_zoo_manifest_dir");
        std::fs::create_dir_all(&tmp).expect("create dir must succeed");

        let manifest = ShapeModelManifest {
            name: "tmp-test-model".to_string(),
            model_type: "GAT".to_string(),
            dataset: "test".to_string(),
            input_dim: 8,
            hidden_dim: 32,
            output_dim: 4,
            num_heads: 2,
            num_layers: 2,
            sha256: "deadbeef".to_string(),
            source: "file:///tmp/test.ckpt".to_string(),
            license: "Apache-2.0".to_string(),
            citation: "test".to_string(),
            version: "0.1.0".to_string(),
            created: "2026-05-01".to_string(),
            notes: None,
        };

        let toml_str = toml::to_string(&manifest).expect("serialization must succeed");
        std::fs::write(tmp.join("tmp-test-model.toml"), &toml_str).expect("write must succeed");

        let zoo =
            ShapeModelZoo::with_manifest_dir(&tmp).expect("loading manifest dir must succeed");
        let got = zoo.get("tmp-test-model").expect("entry must be present");
        assert_eq!(got.model_type, "GAT");
    }

    #[test]
    fn test_registry_list_sorted() {
        let list = ShapeModelZoo::registry().list();
        // Verify that the list is sorted by name
        let names: Vec<&str> = list.iter().map(|m| m.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }
}
