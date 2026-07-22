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

    /// Resolve an object `key` to an on-disk path beneath `root`, rejecting
    /// any key that would escape `root` via `..`, an absolute path, or a
    /// platform path-prefix component (e.g. a Windows drive letter or UNC
    /// share).
    ///
    /// This is the sole security boundary for the local-filesystem backend:
    /// every filesystem operation (read/write/exists/delete) must route
    /// through this function rather than joining `key` onto `root` directly.
    fn path_for(&self, key: &str) -> std::result::Result<PathBuf, String> {
        let trimmed = key.trim_start_matches('/');
        if trimmed.is_empty() {
            return Err(format!("invalid object key {key:?}: key must not be empty"));
        }
        let mut resolved = self.root.clone();
        for component in std::path::Path::new(trimmed).components() {
            match component {
                std::path::Component::Normal(part) => resolved.push(part),
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir
                | std::path::Component::RootDir
                | std::path::Component::Prefix(_) => {
                    return Err(format!(
                        "invalid object key {key:?}: path traversal or absolute paths are not allowed"
                    ));
                }
            }
        }
        // Belt-and-braces: verify the resolved path is still lexically
        // rooted under `self.root` (guards against any component
        // combination the match above did not anticipate).
        if !resolved.starts_with(&self.root) {
            return Err(format!(
                "invalid object key {key:?}: resolves outside the storage root"
            ));
        }
        Ok(resolved)
    }

    fn write_file(&self, key: &str, data: Vec<u8>) -> std::result::Result<(), String> {
        let path = self.path_for(key)?;
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
        let path = self.path_for(key)?;
        std::fs::read(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                format!("Local object not found: {key}")
            } else {
                format!("Failed to read {:?}: {e}", path)
            }
        })
    }

    fn file_exists(&self, key: &str) -> std::result::Result<bool, String> {
        Ok(self.path_for(key)?.is_file())
    }

    fn delete_file(&self, key: &str) -> std::result::Result<(), String> {
        let path = self.path_for(key)?;
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
        self.file_exists(key)
    }

    async fn delete(&self, key: &str) -> std::result::Result<(), String> {
        self.delete_file(key)
    }

    async fn list(&self, prefix: &str) -> std::result::Result<Vec<String>, String> {
        Ok(self.list_prefix(prefix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_backend() -> (LocalFsBackend, PathBuf) {
        let mut root = std::env::temp_dir();
        root.push(format!(
            "oxirs-samm-localfs-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default()
        ));
        let backend = LocalFsBackend::new(&root).expect("create backend root");
        (backend, root)
    }

    #[test]
    fn regression_path_for_rejects_parent_dir_traversal() {
        let (backend, root) = temp_backend();
        let err = backend
            .path_for("../../../../etc/cron.d/evil")
            .expect_err("must reject parent-dir traversal");
        assert!(err.contains("path traversal"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn regression_path_for_rejects_embedded_parent_dir_component() {
        let (backend, root) = temp_backend();
        let err = backend
            .path_for("models/../../secret")
            .expect_err("must reject embedded ..");
        assert!(err.contains("path traversal"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn regression_path_for_rejects_absolute_path_key() {
        let (backend, root) = temp_backend();
        let resolved = backend
            .path_for("/etc/passwd")
            .expect("leading slash alone is stripped, not absolute");
        // A single leading slash is stripped by design (key="/foo" -> root/foo),
        // so this must resolve *inside* root, never to /etc/passwd.
        assert!(resolved.starts_with(&root));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn regression_path_for_rejects_empty_key() {
        let (backend, root) = temp_backend();
        let err = backend.path_for("").expect_err("must reject empty key");
        assert!(err.contains("empty"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn regression_path_for_accepts_and_confines_normal_key() {
        let (backend, root) = temp_backend();
        let path = backend
            .path_for("models/foo/bar.ttl")
            .expect("normal nested key must resolve");
        assert!(path.starts_with(&root));
        assert_eq!(path, root.join("models").join("foo").join("bar.ttl"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn regression_upload_rejects_traversal_key_without_touching_filesystem() {
        let (backend, root) = temp_backend();
        let outside_marker = root
            .parent()
            .expect("temp root has a parent")
            .join("oxirs-samm-localfs-traversal-marker");
        let _ = std::fs::remove_file(&outside_marker);

        let key = "../oxirs-samm-localfs-traversal-marker";
        let result = backend.upload(key, b"pwned".to_vec()).await;
        assert!(result.is_err(), "traversal upload must fail");
        assert!(
            !outside_marker.exists(),
            "traversal upload must never write outside root"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn regression_download_rejects_traversal_key() {
        let (backend, root) = temp_backend();
        let result = backend.download("../../../../etc/passwd").await;
        assert!(result.is_err(), "traversal download must fail");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn regression_delete_rejects_traversal_key() {
        let (backend, root) = temp_backend();
        let result = backend.delete("../outside").await;
        assert!(result.is_err(), "traversal delete must fail");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn regression_roundtrip_normal_key_still_works() {
        let (backend, root) = temp_backend();
        backend
            .upload("a/b/model.ttl", b"hello".to_vec())
            .await
            .expect("normal upload must succeed");
        assert!(backend.exists("a/b/model.ttl").await.unwrap_or(false));
        let data = backend
            .download("a/b/model.ttl")
            .await
            .expect("normal download must succeed");
        assert_eq!(data, b"hello");
        let _ = std::fs::remove_dir_all(&root);
    }
}
