//! SHA-256-verified checkpoint loader for the Shape Model Zoo.
//!
//! [`ShapeModelZooLoader`] loads a named model from disk by
//!
//! 1. Looking up the name in a [`ShapeModelZoo`] (default: the global static registry).
//! 2. Optionally refusing non-permissive licenses unless `accept_license()` was called.
//! 3. Reading the weight file and verifying its SHA-256 checksum.
//! 4. Returning a [`LoadedShapeModel`] whose `weights` bytes can then be
//!    passed to a model-specific deserialiser.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::model_zoo::{manifest::ShapeModelManifest, registry::ShapeModelZoo};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur while loading a shape-model checkpoint.
#[derive(Debug, thiserror::Error)]
pub enum ShapeModelZooError {
    /// The requested model name is not present in the registry.
    #[error("model '{0}' not found in registry")]
    NotFound(String),

    /// The SHA-256 of the loaded file did not match the manifest value.
    #[error("SHA256 mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    /// The manifest lists an architecture that this loader does not support.
    #[error(
        "unsupported model type '{0}' — supported: GAT, GraphSAGE, Graphormer, GraphTransformer"
    )]
    UnsupportedModelType(String),

    /// The checkpoint file's license requires explicit acceptance.
    #[error(
        "license '{license}' requires explicit acceptance — call accept_license() on the loader"
    )]
    LicenseNotAccepted { license: String },

    /// Underlying I/O error (e.g. file not found, permission denied).
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Byte deserialisation failed.
    #[error("deserialization error: {0}")]
    Deserialize(String),

    /// TOML parse error (used when loading manifest files from disk).
    #[error(transparent)]
    Toml(#[from] toml::de::Error),
}

// ---------------------------------------------------------------------------
// Permissive-license allow-list
// ---------------------------------------------------------------------------

/// Licenses that do **not** require explicit user acceptance.
const PERMISSIVE_LICENSES: &[&str] = &[
    "Apache-2.0",
    "MIT",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "MPL-2.0",
    "ISC",
    "CC0-1.0",
    "Unlicense",
];

fn is_permissive_license(license: &str) -> bool {
    PERMISSIVE_LICENSES
        .iter()
        .any(|&l| l.eq_ignore_ascii_case(license))
}

// ---------------------------------------------------------------------------
// LoadedShapeModel
// ---------------------------------------------------------------------------

/// A successfully loaded and SHA-256-verified model checkpoint.
#[derive(Debug)]
pub struct LoadedShapeModel {
    /// The manifest that describes this checkpoint.
    pub manifest: ShapeModelManifest,

    /// Raw weight bytes, ready for model-specific deserialisation.
    pub weights: Vec<u8>,
}

// ---------------------------------------------------------------------------
// ShapeModelZooLoader
// ---------------------------------------------------------------------------

/// Loads shape-model checkpoints from the file system.
///
/// # Design note
///
/// The spec asked for `zoo: &'static ShapeModelZoo`, but that makes it
/// impossible to construct a `ShapeModelZooLoader` with a *custom* registry in
/// tests (without unsafe `Box::leak`).  We instead own an `Arc<ShapeModelZoo>`
/// so that tests can construct a fresh, one-entry zoo while the production path
/// uses `ShapeModelZoo::registry()` (which is backed by a static `Lazy`).
pub struct ShapeModelZooLoader {
    zoo: std::sync::Arc<ShapeModelZoo>,
    base_dir: PathBuf,
    accept_license: bool,
}

impl ShapeModelZooLoader {
    /// Create a loader that uses the global static registry.
    ///
    /// Weight files are resolved relative to `base_dir` when the `source`
    /// URL carries a relative path after stripping the `file://` prefix.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            zoo: std::sync::Arc::new(ShapeModelZoo::clone_registry()),
            base_dir: base_dir.into(),
            accept_license: false,
        }
    }

    /// Create a loader backed by a custom registry (useful for testing).
    pub fn with_zoo(zoo: ShapeModelZoo, base_dir: impl Into<PathBuf>) -> Self {
        Self {
            zoo: std::sync::Arc::new(zoo),
            base_dir: base_dir.into(),
            accept_license: false,
        }
    }

    /// Signal that the caller has read and accepts any license, including
    /// non-permissive ones.
    pub fn accept_license(mut self) -> Self {
        self.accept_license = true;
        self
    }

    // -----------------------------------------------------------------------
    // Main entry point
    // -----------------------------------------------------------------------

    /// Load a checkpoint by name.
    ///
    /// Steps:
    /// 1. Look up `name` in the registry — returns [`ShapeModelZooError::NotFound`] on miss.
    /// 2. Validate `model_type` — returns [`ShapeModelZooError::UnsupportedModelType`] for
    ///    unknown architectures.
    /// 3. Check license — returns [`ShapeModelZooError::LicenseNotAccepted`] when the license
    ///    is non-permissive and `accept_license()` was not called.
    /// 4. Resolve the checkpoint path from `source`.
    /// 5. Read bytes and verify SHA-256 — returns [`ShapeModelZooError::ChecksumMismatch`] on
    ///    digest mismatch.
    /// 6. Return [`LoadedShapeModel`].
    pub fn load(&self, name: &str) -> Result<LoadedShapeModel, ShapeModelZooError> {
        let manifest = self
            .zoo
            .get(name)
            .ok_or_else(|| ShapeModelZooError::NotFound(name.to_string()))?
            .clone();

        Self::validate_model_type(&manifest.model_type)?;

        if !self.accept_license && !is_permissive_license(&manifest.license) {
            return Err(ShapeModelZooError::LicenseNotAccepted {
                license: manifest.license.clone(),
            });
        }

        let path = self.resolve_source(&manifest.source);
        let bytes = std::fs::read(&path)?;

        Self::verify_sha256(&bytes, &manifest.sha256)?;

        Ok(LoadedShapeModel {
            manifest,
            weights: bytes,
        })
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn validate_model_type(model_type: &str) -> Result<(), ShapeModelZooError> {
        const SUPPORTED: &[&str] = &["GAT", "GraphSAGE", "Graphormer", "GraphTransformer"];
        if SUPPORTED.contains(&model_type) {
            Ok(())
        } else {
            Err(ShapeModelZooError::UnsupportedModelType(
                model_type.to_string(),
            ))
        }
    }

    /// Resolve a `source` URL to a filesystem `Path`.
    ///
    /// Rules:
    /// - Strip a leading `file://` prefix.
    /// - If the remaining path is absolute, use it as-is.
    /// - If relative, join it to `self.base_dir`.
    fn resolve_source(&self, source: &str) -> PathBuf {
        let stripped = source.strip_prefix("file://").unwrap_or(source);
        let p = Path::new(stripped);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.base_dir.join(p)
        }
    }

    /// Verify that the SHA-256 of `data` matches `expected` (lowercase hex).
    pub fn verify_sha256(data: &[u8], expected: &str) -> Result<(), ShapeModelZooError> {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let actual = hex::encode(hasher.finalize());
        if actual == expected {
            Ok(())
        } else {
            Err(ShapeModelZooError::ChecksumMismatch {
                expected: expected.to_string(),
                actual,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GAT_SEED: &[u8] = b"synthetic-seed-GAT-v1";

    #[test]
    fn test_sha256_verify_ok() {
        let tmp = std::env::temp_dir().join("shacl_ai_sha256_ok.ckpt");
        std::fs::write(&tmp, GAT_SEED).expect("write must succeed");
        let data = std::fs::read(&tmp).expect("read must succeed");

        ShapeModelZooLoader::verify_sha256(
            &data,
            "f5166a14c94cb4e8fb3ca7db76954312fe6a2859709de22026228c98db9f0a2d",
        )
        .expect("correct digest must pass");
    }

    #[test]
    fn test_sha256_verify_fail() {
        let tmp = std::env::temp_dir().join("shacl_ai_sha256_fail.ckpt");
        let mut bad = GAT_SEED.to_vec();
        bad[0] ^= 0xFF; // flip first byte
        std::fs::write(&tmp, &bad).expect("write must succeed");
        let data = std::fs::read(&tmp).expect("read must succeed");

        let err = ShapeModelZooLoader::verify_sha256(
            &data,
            "f5166a14c94cb4e8fb3ca7db76954312fe6a2859709de22026228c98db9f0a2d",
        )
        .expect_err("mismatched digest must fail");

        assert!(matches!(err, ShapeModelZooError::ChecksumMismatch { .. }));
    }

    #[test]
    fn test_loader_license_refusal() {
        use crate::model_zoo::manifest::ShapeModelManifest;
        use crate::model_zoo::registry::ShapeModelZoo;

        let manifest = ShapeModelManifest {
            name: "proprietary-model".to_string(),
            model_type: "GAT".to_string(),
            dataset: "secret".to_string(),
            input_dim: 16,
            hidden_dim: 64,
            output_dim: 8,
            num_heads: 4,
            num_layers: 2,
            sha256: "deadbeef".to_string(),
            source: "file:///nonexistent/model.ckpt".to_string(),
            license: "Proprietary".to_string(),
            citation: "none".to_string(),
            version: "1.0.0".to_string(),
            created: "2026-05-01".to_string(),
            notes: None,
        };

        let zoo = ShapeModelZoo::with_single_entry(manifest);
        let tmp_dir = std::env::temp_dir();
        let loader = ShapeModelZooLoader::with_zoo(zoo, &tmp_dir);

        let err = loader
            .load("proprietary-model")
            .expect_err("proprietary license without accept_license() must fail");

        assert!(matches!(err, ShapeModelZooError::LicenseNotAccepted { .. }));
    }

    #[test]
    fn test_loader_load_ok() {
        use crate::model_zoo::manifest::ShapeModelManifest;
        use crate::model_zoo::registry::ShapeModelZoo;

        let tmp_dir = std::env::temp_dir();
        let ckpt_path = tmp_dir.join("gat-test-load.ckpt");
        let content = b"synthetic-seed-GAT-v1";
        std::fs::write(&ckpt_path, content).expect("write must succeed");

        // Compute real hash
        let mut hasher = Sha256::new();
        hasher.update(content);
        let sha256 = hex::encode(hasher.finalize());

        let manifest = ShapeModelManifest {
            name: "gat-shacl-base".to_string(),
            model_type: "GAT".to_string(),
            dataset: "LUBM-SHACL-synthetic".to_string(),
            input_dim: 16,
            hidden_dim: 64,
            output_dim: 8,
            num_heads: 4,
            num_layers: 2,
            sha256,
            // relative path — loader will join base_dir
            source: "file://gat-test-load.ckpt".to_string(),
            license: "Apache-2.0".to_string(),
            citation: "test".to_string(),
            version: "1.0.0".to_string(),
            created: "2026-05-01".to_string(),
            notes: None,
        };

        let zoo = ShapeModelZoo::with_single_entry(manifest);
        let loader = ShapeModelZooLoader::with_zoo(zoo, &tmp_dir);

        let loaded = loader.load("gat-shacl-base").expect("load must succeed");
        assert_eq!(loaded.weights, content);
        assert_eq!(loaded.manifest.name, "gat-shacl-base");
    }
}
