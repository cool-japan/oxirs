//! SHA-256-verified checkpoint loading via the existing `ModelRepository`.
//!
//! # Load flow (Approach A — materialize-then-load)
//!
//! 1. Look up the manifest in the registry.
//! 2. Optionally enforce the license gate.
//! 3. Validate the `model_type` string against the set that
//!    `persistence::ModelRepository` can dispatch.
//! 4. Resolve the checkpoint source path (`file:///...` only in default
//!    features).
//! 5. Read the raw bytes from disk.
//! 6. Verify SHA-256 (unless the manifest carries the sentinel `"PLACEHOLDER"`
//!    which marks catalog entries whose checkpoint does not ship with the crate).
//! 7. Write the materialised checkpoint into a temp subdirectory that looks
//!    exactly like what `ModelRepository::new + scan_models` expects:
//!    ```text
//!    <base_dir>/<name>/
//!        model.bin          ← the checkpoint bytes
//!        model_type.json    ← JSON-encoded model_type string
//!        metadata.json      ← minimal ModelMetadata
//!    ```
//! 8. Construct a `ModelRepository` rooted at `<base_dir>` and call
//!    `load_model(&manifest.name)`.

use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::model_zoo::registry::ModelZoo;
use crate::EmbeddingModel;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during model zoo operations.
#[derive(Debug, Error)]
pub enum ModelZooError {
    /// The requested model name is not present in the registry.
    #[error("model '{0}' not found in registry")]
    NotFound(String),

    /// The model's license requires explicit acceptance.
    #[error("license '{license}' requires acceptance — set accept_license=true")]
    LicenseNotAccepted { license: String },

    /// The downloaded/read file's SHA-256 does not match the manifest.
    #[error("SHA256 mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    /// The manifest declares a model type that the persistence layer cannot
    /// handle.
    #[error(
        "unsupported model type '{0}' — supported: TransE, DistMult, ComplEx, RotatE, HoLE, GNNEmbedding"
    )]
    UnsupportedModelType(String),

    /// Failed to parse a TOML manifest.
    #[error("manifest parse error: {0}")]
    ManifestParse(String),

    /// I/O error (file read, temp dir creation, etc.).
    #[error(transparent)]
    Io(#[from] io::Error),

    /// Error propagated from the underlying persistence layer.
    #[error(transparent)]
    Persistence(#[from] anyhow::Error),
}

// ---------------------------------------------------------------------------
// Supported model types (mirrors persistence.rs dispatch table)
// ---------------------------------------------------------------------------

/// Model type strings that `persistence::ModelRepository::load_model` accepts.
const SUPPORTED_MODEL_TYPES: &[&str] = &[
    "TransE",
    "DistMult",
    "ComplEx",
    "RotatE",
    "HoLE",
    "GNN",
    "GNNEmbedding",
];

fn is_supported_model_type(model_type: &str) -> bool {
    SUPPORTED_MODEL_TYPES.contains(&model_type)
}

// ---------------------------------------------------------------------------
// Permissive license set
// ---------------------------------------------------------------------------

/// Licenses that are considered permissive (no acceptance gate required).
const PERMISSIVE_LICENSES: &[&str] = &[
    "Apache-2.0",
    "MIT",
    "MIT OR Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "CC0-1.0",
    "Unlicense",
    "WTFPL",
];

fn is_permissive_license(license: &str) -> bool {
    PERMISSIVE_LICENSES
        .iter()
        .any(|&l| license.eq_ignore_ascii_case(l))
}

// ---------------------------------------------------------------------------
// ModelZooLoader
// ---------------------------------------------------------------------------

/// Loads pretrained (or synthetic-seed) models from the [`ModelZoo`] registry.
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_embed::model_zoo::{ModelZoo, ModelZooLoader};
///
/// let loader = ModelZooLoader::new(std::env::temp_dir()).accept_license();
/// // Would load a real checkpoint when source resolves to a real file:
/// // let model = loader.load("transe-fb15k237")?;
/// ```
pub struct ModelZooLoader {
    zoo: &'static ModelZoo,
    base_dir: PathBuf,
    accept_license: bool,
}

impl ModelZooLoader {
    /// Create a loader that stores materialised checkpoints under `base_dir`.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            zoo: ModelZoo::registry(),
            base_dir: base_dir.into(),
            accept_license: false,
        }
    }

    /// Create a loader backed by a custom (non-global) `ModelZoo`.
    pub fn with_zoo(zoo: &'static ModelZoo, base_dir: impl Into<PathBuf>) -> Self {
        Self {
            zoo,
            base_dir: base_dir.into(),
            accept_license: false,
        }
    }

    /// Signal that the caller accepts all licenses (including restrictive ones
    /// such as `CC-BY-NC-4.0`).
    pub fn accept_license(mut self) -> Self {
        self.accept_license = true;
        self
    }

    /// Load the model identified by `name` from the registry.
    ///
    /// Returns a heap-allocated, type-erased [`EmbeddingModel`].
    pub fn load(&self, name: &str) -> Result<Box<dyn EmbeddingModel>, ModelZooError> {
        // 1. Look up manifest
        let manifest = self
            .zoo
            .get(name)
            .ok_or_else(|| ModelZooError::NotFound(name.to_string()))?;

        // 2. License check
        if !self.accept_license && !is_permissive_license(&manifest.license) {
            return Err(ModelZooError::LicenseNotAccepted {
                license: manifest.license.clone(),
            });
        }

        // 3. Model type validation
        if !is_supported_model_type(&manifest.model_type) {
            return Err(ModelZooError::UnsupportedModelType(
                manifest.model_type.clone(),
            ));
        }

        // 4. Resolve source path
        let source_path = resolve_source_path(&manifest.source, &self.base_dir)?;

        // 5. Read bytes
        let bytes = std::fs::read(&source_path)?;

        // 6. SHA-256 verification (skip sentinel "PLACEHOLDER")
        if manifest.sha256 != "PLACEHOLDER" {
            Self::verify_sha256(&bytes, &manifest.sha256)?;
        }

        // 7. Materialise the repository structure in <base_dir>/<name>/
        let model_dir = self.base_dir.join(&manifest.name);
        materialise_checkpoint(&model_dir, &bytes, &manifest.model_type)?;

        // 8. Construct ModelRepository and load
        let repo = crate::persistence::ModelRepository::new(&self.base_dir)?;
        let model = repo.load_model(&manifest.name)?;
        Ok(model)
    }

    /// Verify that `data` hashes to `expected` (hex-encoded SHA-256).
    fn verify_sha256(data: &[u8], expected: &str) -> Result<(), ModelZooError> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let digest = hasher.finalize();
        let actual = hex::encode(digest);
        if actual != expected.to_lowercase() {
            return Err(ModelZooError::ChecksumMismatch {
                expected: expected.to_string(),
                actual,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the hex-encoded SHA-256 digest of `data`.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Resolve a `file:///` URL into a [`PathBuf`].
///
/// When the literal path under `file:///` does not exist on the filesystem the
/// function tries to resolve it relative to `base_dir` (stripping the
/// `file:///` prefix).
fn resolve_source_path(source: &str, base_dir: &Path) -> Result<PathBuf, ModelZooError> {
    if let Some(rest) = source.strip_prefix("file:///") {
        let absolute = Path::new("/").join(rest);
        if absolute.exists() {
            return Ok(absolute);
        }
        // Try relative to base_dir (e.g. "seeds/transe-fb15k237.ckpt")
        let relative = base_dir.join(rest);
        if relative.exists() {
            return Ok(relative);
        }
        // Return the absolute path even if it doesn't exist yet — callers that
        // want to test the path-not-found code path can catch the IO error.
        return Ok(absolute);
    }

    if source.starts_with("https://") || source.starts_with("http://") {
        return Err(ModelZooError::Io(io::Error::new(
            io::ErrorKind::Unsupported,
            "HTTP download requires the 'download' feature (not enabled in default build). \
             Use a file:/// source or enable the feature.",
        )));
    }

    Err(ModelZooError::Io(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("unrecognised source scheme: {source}"),
    )))
}

/// Write the three files that `ModelRepository::scan_models` expects.
///
/// ```text
/// <model_dir>/
///     model.bin          ← checkpoint bytes
///     model_type.json    ← JSON-encoded type string
///     metadata.json      ← minimal ModelMetadata
/// ```
fn materialise_checkpoint(
    model_dir: &Path,
    bytes: &[u8],
    model_type: &str,
) -> Result<(), ModelZooError> {
    std::fs::create_dir_all(model_dir)?;

    // model.bin
    let mut f = std::fs::File::create(model_dir.join("model.bin"))?;
    f.write_all(bytes)?;

    // model_type.json
    let type_json = serde_json::to_string(model_type)
        .map_err(|e| ModelZooError::Io(io::Error::new(io::ErrorKind::Other, e.to_string())))?;
    std::fs::write(model_dir.join("model_type.json"), &type_json)?;

    // metadata.json — use the same struct that ModelRepository expects
    let metadata = crate::persistence::ModelMetadata::default();
    let meta_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| ModelZooError::Io(io::Error::new(io::ErrorKind::Other, e.to_string())))?;
    std::fs::write(model_dir.join("metadata.json"), &meta_json)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_hex_deterministic() {
        let data = b"hello world";
        let h1 = sha256_hex(data);
        let h2 = sha256_hex(data);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // 32 bytes → 64 hex chars
    }

    #[test]
    fn test_verify_sha256_ok() {
        let data = b"test data for hashing";
        let expected = sha256_hex(data);
        // Should not error
        ModelZooLoader::verify_sha256(data, &expected).expect("verification should pass");
    }

    #[test]
    fn test_verify_sha256_mismatch() {
        let data = b"test data for hashing";
        let wrong_hash = "0".repeat(64);
        let result = ModelZooLoader::verify_sha256(data, &wrong_hash);
        assert!(result.is_err());
        match result {
            Err(ModelZooError::ChecksumMismatch { expected, actual }) => {
                assert_eq!(expected, wrong_hash);
                assert_ne!(actual, wrong_hash);
            }
            other => panic!("Expected ChecksumMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_is_supported_model_type() {
        for ty in SUPPORTED_MODEL_TYPES {
            assert!(is_supported_model_type(ty), "{ty} should be supported");
        }
        assert!(!is_supported_model_type("Bogus"));
        assert!(!is_supported_model_type("TransE2"));
    }

    #[test]
    fn test_is_permissive_license() {
        assert!(is_permissive_license("Apache-2.0"));
        assert!(is_permissive_license("MIT"));
        assert!(is_permissive_license("MIT OR Apache-2.0"));
        assert!(!is_permissive_license("CC-BY-NC-4.0"));
        assert!(!is_permissive_license("Proprietary"));
    }

    #[test]
    fn test_resolve_source_path_file_scheme() {
        // A path that definitely exists
        let base = std::env::temp_dir();
        let existing = base.to_string_lossy().to_string();
        let source = format!("file://{existing}");
        let result = resolve_source_path(&source, &base);
        // Should not error (path exists)
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_source_path_http_error() {
        let base = std::env::temp_dir();
        let result = resolve_source_path("https://example.com/model.ckpt", &base);
        assert!(result.is_err());
        let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(msg.contains("download") || msg.contains("HTTP"));
    }

    #[test]
    fn test_resolve_source_path_unknown_scheme() {
        let base = std::env::temp_dir();
        let result = resolve_source_path("s3://bucket/model.ckpt", &base);
        assert!(result.is_err());
    }

    #[test]
    fn test_materialise_checkpoint_creates_files() {
        let tmp = std::env::temp_dir().join("oxirs_materialise_test");
        let model_dir = tmp.join("test_model");
        let bytes = b"fake checkpoint bytes";

        materialise_checkpoint(&model_dir, bytes, "TransE").expect("materialise ok");

        assert!(model_dir.join("model.bin").exists());
        assert!(model_dir.join("model_type.json").exists());
        assert!(model_dir.join("metadata.json").exists());

        // Verify model_type.json content
        let raw = std::fs::read_to_string(model_dir.join("model_type.json")).expect("read");
        let ty: String = serde_json::from_str(&raw).expect("parse");
        assert_eq!(ty, "TransE");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_load_not_found() {
        let loader = ModelZooLoader::new(std::env::temp_dir()).accept_license();
        let result = loader.load("definitely-does-not-exist");
        assert!(matches!(result, Err(ModelZooError::NotFound(_))));
    }

    #[test]
    fn test_loader_license_refused() {
        use crate::model_zoo::manifest::ModelManifest;
        use crate::model_zoo::registry::ModelZoo;

        // Build a custom zoo with a non-permissive license entry
        let tmp_dir = std::env::temp_dir().join("oxirs_zoo_license_test");
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");

        let manifest = ModelManifest {
            name: "restricted-model".to_string(),
            model_type: "TransE".to_string(),
            dataset: "TestDS".to_string(),
            dimensions: 10,
            entities: 5,
            relations: 2,
            sha256: "PLACEHOLDER".to_string(),
            source: "file:///nonexistent.ckpt".to_string(),
            license: "CC-BY-NC-4.0".to_string(),
            citation: "Test".to_string(),
            version: "1.0.0".to_string(),
            created: "2026-05-01".to_string(),
            notes: None,
        };

        let toml_str = toml::to_string(&manifest).expect("serialize");
        std::fs::write(tmp_dir.join("restricted-model.toml"), toml_str).expect("write");

        let zoo = ModelZoo::with_manifest_dir(&tmp_dir).expect("build zoo");
        // Leak the zoo to get a 'static reference for testing
        let zoo_ref: &'static ModelZoo = Box::leak(Box::new(zoo));

        let loader = ModelZooLoader::with_zoo(zoo_ref, std::env::temp_dir());
        let result = loader.load("restricted-model");

        assert!(matches!(
            result,
            Err(ModelZooError::LicenseNotAccepted { .. })
        ));

        std::fs::remove_dir_all(&tmp_dir).ok();
    }
}
