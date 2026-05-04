//! Backup destination configuration.
//!
//! `DestinationConfig` is an enum that captures where backup artefacts
//! should be written.  Additional backends (S3, Azure, GCS) are
//! feature-gated to keep default builds 100 % Pure Rust with no C/Fortran.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Backup artefact destination.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DestinationConfig {
    /// Write backup artefacts to the local filesystem at `path`.
    Filesystem {
        /// Absolute or relative directory path.
        path: PathBuf,
    },

    /// S3-compatible object storage (feature-gated: `s3-backup`).
    ///
    /// Requires the `s3-backup` Cargo feature.  Without it, any attempt to
    /// construct this variant will cause a compile error, keeping the default
    /// build dependency-free.
    #[cfg(feature = "s3-backup")]
    S3 {
        /// S3 bucket name.
        bucket: String,
        /// Key prefix inside the bucket (no leading slash).
        prefix: String,
        /// Override endpoint URL for S3-compatible stores (e.g. MinIO).
        endpoint: String,
    },
}

impl Default for DestinationConfig {
    fn default() -> Self {
        DestinationConfig::Filesystem {
            path: std::env::temp_dir().join("oxirs_backups"),
        }
    }
}

impl DestinationConfig {
    /// Return the local filesystem path, if this is a `Filesystem` variant.
    pub fn local_path(&self) -> Option<&PathBuf> {
        match self {
            DestinationConfig::Filesystem { path } => Some(path),
            #[cfg(feature = "s3-backup")]
            DestinationConfig::S3 { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn default_destination_is_filesystem() {
        let d = DestinationConfig::default();
        assert!(d.local_path().is_some());
    }

    #[test]
    fn filesystem_destination_local_path() {
        let path = env::temp_dir().join("test_backup");
        let d = DestinationConfig::Filesystem { path: path.clone() };
        assert_eq!(d.local_path(), Some(&path));
    }

    #[test]
    fn filesystem_destination_serialises() {
        let path = env::temp_dir().join("backup_ser_test");
        let d = DestinationConfig::Filesystem { path: path.clone() };
        let json = serde_json::to_string(&d).unwrap();
        let back: DestinationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.local_path(), Some(&path));
    }
}
