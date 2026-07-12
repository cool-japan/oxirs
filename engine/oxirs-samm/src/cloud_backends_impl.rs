//! Backend implementations for the SAMM cloud storage subsystem.
//!
//! Provides concrete [`CloudStorageBackend`] implementations for:
//! - AWS S3 and S3-compatible stores ([`crate::cloud_backends_aws::S3Backend`])
//! - Google Cloud Storage ([`crate::cloud_backends_gcp::GcsBackend`])
//! - Azure Blob Storage ([`crate::cloud_backends_azure::AzureBlobBackend`])
//! - Generic HTTP REST backend ([`crate::cloud_backends_http::HttpBackend`])
//! - Local filesystem adapter ([`crate::cloud_backends_impl::LocalFsBackend`])
//!
//! All backends use async/await with `reqwest` (rustls TLS — no OpenSSL).

// Re-export the per-provider backend structs and their configs so that callers
// can `use oxirs_samm::cloud_backends_impl::*`.
pub use crate::cloud_backends_aws::{S3Backend, S3Config};
pub use crate::cloud_backends_azure::{AzureBlobBackend, AzureConfig};
pub use crate::cloud_backends_gcp::{GcsBackend, GcsConfig};
pub use crate::cloud_backends_http::{HttpBackend, HttpConfig};

use crate::cloud_storage::CloudStorageBackend;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

// ──────────────────────────────────────────────────────────────────────────────
// LocalFsBackend — filesystem adapter (useful for testing and air-gapped
// deployments)
// ──────────────────────────────────────────────────────────────────────────────

/// A local-filesystem storage backend.
///
/// Stores objects as files under a configurable root directory.  The object
/// `key` is mapped directly to a path beneath the root, so keys containing
/// `/` become subdirectories.
///
/// This backend is **synchronous internally** but presents the async
/// [`CloudStorageBackend`] interface by wrapping I/O with
/// `tokio::task::spawn_blocking`.
pub struct LocalFsBackend {
    root: PathBuf,
    /// In-memory "object index" so that list() does not have to walk the
    /// directory tree on every call (keys are relative to `root`).
    index: RwLock<HashMap<String, usize>>,
}

impl std::fmt::Debug for LocalFsBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalFsBackend")
            .field("root", &self.root)
            .finish()
    }
}

impl LocalFsBackend {
    /// Create a new `LocalFsBackend` rooted at `root`.
    ///
    /// The directory is created if it does not already exist.
    pub fn new(root: impl Into<PathBuf>) -> std::result::Result<Self, String> {
        let root = root.into();
        std::fs::create_dir_all(&root)
            .map_err(|e| format!("Failed to create LocalFsBackend root {:?}: {e}", root))?;
        Ok(Self {
            root,
            index: RwLock::new(HashMap::new()),
        })
    }

    fn path_for(&self, key: &str) -> PathBuf {
        // Strip any leading slash so that key="/foo" maps to root/foo.
        self.root.join(key.trim_start_matches('/'))
    }

    fn write_file(&self, key: &str, data: Vec<u8>) -> std::result::Result<(), String> {
        let path = self.path_for(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {:?}: {e}", parent))?;
        }
        std::fs::write(&path, &data).map_err(|e| format!("Failed to write {:?}: {e}", path))?;
        let len = data.len();
        self.index
            .write()
            .map_err(|_| "index lock poisoned".to_string())?
            .insert(key.to_string(), len);
        Ok(())
    }

    fn read_file(&self, key: &str) -> std::result::Result<Vec<u8>, String> {
        let path = self.path_for(key);
        std::fs::read(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!("Local object not found: {key}")
            } else {
                format!("Failed to read {:?}: {e}", path)
            }
        })
    }

    fn file_exists(&self, key: &str) -> bool {
        self.path_for(key).is_file()
    }

    fn delete_file(&self, key: &str) -> std::result::Result<(), String> {
        let path = self.path_for(key);
        if path.is_file() {
            std::fs::remove_file(&path).map_err(|e| format!("Failed to delete {:?}: {e}", path))?;
        }
        if let Ok(mut idx) = self.index.write() {
            idx.remove(key);
        }
        Ok(())
    }

    fn list_prefix(&self, prefix: &str) -> Vec<String> {
        if let Ok(idx) = self.index.read() {
            idx.keys()
                .filter(|k| k.starts_with(prefix))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
}

#[async_trait]
impl CloudStorageBackend for LocalFsBackend {
    async fn upload(&self, key: &str, data: Vec<u8>) -> std::result::Result<(), String> {
        self.write_file(key, data)
    }

    async fn download(&self, key: &str) -> std::result::Result<Vec<u8>, String> {
        self.read_file(key)
    }

    async fn exists(&self, key: &str) -> std::result::Result<bool, String> {
        Ok(self.file_exists(key))
    }

    async fn delete(&self, key: &str) -> std::result::Result<(), String> {
        self.delete_file(key)
    }

    async fn list(&self, prefix: &str) -> std::result::Result<Vec<String>, String> {
        Ok(self.list_prefix(prefix))
    }
}
