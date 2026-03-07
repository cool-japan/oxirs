//! Cloud Storage Client Abstraction Layer
//!
//! This module provides a sync, trait-based cloud storage abstraction with:
//!
//! - [`CloudStorageClient`] — core CRUD trait
//! - [`CloudStorageError`] — structured error enum
//! - [`MockCloudStorage`] — in-memory implementation for testing with error injection
//! - [`RetryableCloudClient`] — wraps any client with configurable retry logic
//!
//! # Example
//!
//! ```rust
//! use oxirs_samm::cloud_client::{CloudStorageClient, MockCloudStorage, CloudStorageError};
//!
//! let store = MockCloudStorage::new();
//! store.upload("models/test.ttl", b"@prefix ...").unwrap();
//! assert!(store.exists("models/test.ttl").unwrap());
//! let data = store.download("models/test.ttl").unwrap();
//! assert_eq!(data, b"@prefix ...");
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during cloud storage operations.
#[derive(Debug, thiserror::Error)]
pub enum CloudStorageError {
    /// The requested key was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// The operation was denied due to insufficient permissions.
    #[error("permission denied: {0}")]
    PermissionDenied(String),

    /// The storage quota has been exceeded.
    #[error("quota exceeded: {0}")]
    QuotaExceeded(String),

    /// A transient network error occurred (retryable).
    #[error("network error: {0}")]
    NetworkError(String),

    /// An internal server error occurred (retryable).
    #[error("internal error: {0}")]
    InternalError(String),
}

impl CloudStorageError {
    /// Returns `true` if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            CloudStorageError::NetworkError(_) | CloudStorageError::InternalError(_)
        )
    }
}

/// Clonable kind mirror of [`CloudStorageError`] used internally in [`MockCloudStorage`].
#[derive(Debug, Clone)]
enum CloudStorageErrorKind {
    NotFound(String),
    PermissionDenied(String),
    QuotaExceeded(String),
    NetworkError(String),
    InternalError(String),
}

impl CloudStorageErrorKind {
    /// Convert into a [`CloudStorageError`].
    fn into_error(self) -> CloudStorageError {
        match self {
            CloudStorageErrorKind::NotFound(m) => CloudStorageError::NotFound(m),
            CloudStorageErrorKind::PermissionDenied(m) => CloudStorageError::PermissionDenied(m),
            CloudStorageErrorKind::QuotaExceeded(m) => CloudStorageError::QuotaExceeded(m),
            CloudStorageErrorKind::NetworkError(m) => CloudStorageError::NetworkError(m),
            CloudStorageErrorKind::InternalError(m) => CloudStorageError::InternalError(m),
        }
    }
}

impl From<CloudStorageError> for CloudStorageErrorKind {
    fn from(e: CloudStorageError) -> Self {
        match e {
            CloudStorageError::NotFound(m) => CloudStorageErrorKind::NotFound(m),
            CloudStorageError::PermissionDenied(m) => CloudStorageErrorKind::PermissionDenied(m),
            CloudStorageError::QuotaExceeded(m) => CloudStorageErrorKind::QuotaExceeded(m),
            CloudStorageError::NetworkError(m) => CloudStorageErrorKind::NetworkError(m),
            CloudStorageError::InternalError(m) => CloudStorageErrorKind::InternalError(m),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Sync cloud storage CRUD trait.
///
/// Implement this trait to provide different storage backends.
/// The trait is object-safe and can be used as `Box<dyn CloudStorageClient>`.
pub trait CloudStorageClient: Send + Sync {
    /// Upload `data` bytes under the given `key`.
    fn upload(&self, key: &str, data: &[u8]) -> Result<(), CloudStorageError>;

    /// Download and return the bytes stored under `key`.
    fn download(&self, key: &str) -> Result<Vec<u8>, CloudStorageError>;

    /// Delete the object stored under `key`.
    fn delete(&self, key: &str) -> Result<(), CloudStorageError>;

    /// List all keys that start with `prefix`.
    fn list(&self, prefix: &str) -> Result<Vec<String>, CloudStorageError>;

    /// Check whether an object exists under `key`.
    fn exists(&self, key: &str) -> Result<bool, CloudStorageError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// MockCloudStorage
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory cloud storage mock for testing.
///
/// Supports error injection via [`MockCloudStorage::set_error_on_key`] so that
/// individual test cases can simulate failure modes without hitting a real endpoint.
#[derive(Debug, Clone)]
pub struct MockCloudStorage {
    storage: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    error_keys: Arc<Mutex<HashMap<String, CloudStorageErrorKind>>>,
}

impl MockCloudStorage {
    /// Create a new, empty in-memory store.
    pub fn new() -> Self {
        Self {
            storage: Arc::new(Mutex::new(HashMap::new())),
            error_keys: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Return a sorted list of all keys currently stored.
    pub fn uploaded_keys(&self) -> Vec<String> {
        let guard = match self.storage.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut keys: Vec<String> = guard.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Inject an error that will be returned for any operation on `key`.
    ///
    /// Call [`MockCloudStorage::clear_errors`] to remove injected errors.
    pub fn set_error_on_key(&self, key: &str, error: CloudStorageError) {
        let mut guard = match self.error_keys.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.insert(key.to_string(), CloudStorageErrorKind::from(error));
    }

    /// Remove all injected errors.
    pub fn clear_errors(&self) {
        let mut guard = match self.error_keys.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.clear();
    }

    /// Check for an injected error on `key`, returning it if present.
    fn check_error(&self, key: &str) -> Option<CloudStorageError> {
        let guard = match self.error_keys.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.get(key).cloned().map(|k| k.into_error())
    }
}

impl Default for MockCloudStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudStorageClient for MockCloudStorage {
    fn upload(&self, key: &str, data: &[u8]) -> Result<(), CloudStorageError> {
        if let Some(err) = self.check_error(key) {
            return Err(err);
        }
        let mut guard = match self.storage.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.insert(key.to_string(), data.to_vec());
        Ok(())
    }

    fn download(&self, key: &str) -> Result<Vec<u8>, CloudStorageError> {
        if let Some(err) = self.check_error(key) {
            return Err(err);
        }
        let guard = match self.storage.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard
            .get(key)
            .cloned()
            .ok_or_else(|| CloudStorageError::NotFound(key.to_string()))
    }

    fn delete(&self, key: &str) -> Result<(), CloudStorageError> {
        if let Some(err) = self.check_error(key) {
            return Err(err);
        }
        let mut guard = match self.storage.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.remove(key);
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, CloudStorageError> {
        let guard = match self.storage.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut keys: Vec<String> = guard
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        keys.sort();
        Ok(keys)
    }

    fn exists(&self, key: &str) -> Result<bool, CloudStorageError> {
        if let Some(err) = self.check_error(key) {
            return Err(err);
        }
        let guard = match self.storage.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        Ok(guard.contains_key(key))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RetryableCloudClient
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps any [`CloudStorageClient`] with configurable retry logic.
///
/// Only [`CloudStorageError::NetworkError`] and [`CloudStorageError::InternalError`]
/// are retried. `NotFound`, `PermissionDenied`, and `QuotaExceeded` are returned
/// immediately without retry.
///
/// # Example
///
/// ```rust
/// use oxirs_samm::cloud_client::{MockCloudStorage, RetryableCloudClient, CloudStorageClient};
///
/// let inner = MockCloudStorage::new();
/// let client = RetryableCloudClient::new(inner, 3, 0, true);
/// client.upload("key", b"data").unwrap();
/// ```
#[derive(Debug)]
pub struct RetryableCloudClient<C: CloudStorageClient> {
    inner: C,
    /// Maximum number of retry attempts after the first failure.
    pub max_retries: u32,
    /// Base delay in milliseconds between retries.
    pub base_delay_ms: u64,
    /// Whether to use exponential backoff (`delay = base_delay_ms * 2^attempt`).
    pub exponential_backoff: bool,
}

impl<C: CloudStorageClient> RetryableCloudClient<C> {
    /// Create a new retryable client.
    ///
    /// # Arguments
    ///
    /// * `inner` — the underlying client to wrap
    /// * `max_retries` — how many additional attempts to make after the first failure
    /// * `base_delay_ms` — delay in milliseconds before retrying (use 0 in tests)
    /// * `exponential_backoff` — if `true`, doubles the delay on each attempt
    pub fn new(inner: C, max_retries: u32, base_delay_ms: u64, exponential_backoff: bool) -> Self {
        Self {
            inner,
            max_retries,
            base_delay_ms,
            exponential_backoff,
        }
    }

    /// Compute the sleep duration for a given attempt index (0-based).
    fn sleep_duration(&self, attempt: u32) -> Duration {
        if self.base_delay_ms == 0 {
            return Duration::from_millis(0);
        }
        let multiplier = if self.exponential_backoff {
            // 2^attempt, capped to avoid overflow
            1u64.checked_shl(attempt).unwrap_or(u64::MAX)
        } else {
            1u64
        };
        Duration::from_millis(self.base_delay_ms.saturating_mul(multiplier))
    }

    /// Execute `op` with retry logic for transient errors.
    fn with_retry<F, T>(&self, op: F) -> Result<T, CloudStorageError>
    where
        F: Fn() -> Result<T, CloudStorageError>,
    {
        let mut last_err: Option<CloudStorageError> = None;
        for attempt in 0..=self.max_retries {
            match op() {
                Ok(val) => return Ok(val),
                Err(e) => {
                    if e.is_retryable() {
                        if attempt < self.max_retries {
                            let delay = self.sleep_duration(attempt);
                            if !delay.is_zero() {
                                std::thread::sleep(delay);
                            }
                        }
                        last_err = Some(e);
                    } else {
                        // Non-retryable — return immediately
                        return Err(e);
                    }
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| CloudStorageError::InternalError("retry exhausted".to_string())))
    }
}

impl<C: CloudStorageClient> CloudStorageClient for RetryableCloudClient<C> {
    fn upload(&self, key: &str, data: &[u8]) -> Result<(), CloudStorageError> {
        // Capture data as owned to allow repeated calls inside closure
        let data_owned = data.to_vec();
        self.with_retry(|| self.inner.upload(key, &data_owned))
    }

    fn download(&self, key: &str) -> Result<Vec<u8>, CloudStorageError> {
        self.with_retry(|| self.inner.download(key))
    }

    fn delete(&self, key: &str) -> Result<(), CloudStorageError> {
        self.with_retry(|| self.inner.delete(key))
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>, CloudStorageError> {
        self.with_retry(|| self.inner.list(prefix))
    }

    fn exists(&self, key: &str) -> Result<bool, CloudStorageError> {
        self.with_retry(|| self.inner.exists(key))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Error display ──────────────────────────────────────────────────────

    #[test]
    fn test_cloud_storage_error_not_found_display() {
        let e = CloudStorageError::NotFound("key.ttl".to_string());
        assert!(e.to_string().contains("not found"));
        assert!(e.to_string().contains("key.ttl"));
    }

    #[test]
    fn test_cloud_storage_error_network_error_display() {
        let e = CloudStorageError::NetworkError("timeout".to_string());
        assert!(e.to_string().contains("network error"));
        assert!(e.to_string().contains("timeout"));
    }

    #[test]
    fn test_cloud_storage_error_internal_error_display() {
        let e = CloudStorageError::InternalError("503".to_string());
        assert!(e.to_string().contains("internal error"));
        assert!(e.to_string().contains("503"));
    }

    #[test]
    fn test_cloud_storage_error_permission_denied_display() {
        let e = CloudStorageError::PermissionDenied("read".to_string());
        assert!(e.to_string().contains("permission denied"));
    }

    #[test]
    fn test_cloud_storage_error_quota_exceeded_display() {
        let e = CloudStorageError::QuotaExceeded("5 GB".to_string());
        assert!(e.to_string().contains("quota exceeded"));
    }

    #[test]
    fn test_cloud_storage_error_retryable_network() {
        assert!(CloudStorageError::NetworkError("x".to_string()).is_retryable());
    }

    #[test]
    fn test_cloud_storage_error_retryable_internal() {
        assert!(CloudStorageError::InternalError("x".to_string()).is_retryable());
    }

    #[test]
    fn test_cloud_storage_error_not_retryable_not_found() {
        assert!(!CloudStorageError::NotFound("x".to_string()).is_retryable());
    }

    #[test]
    fn test_cloud_storage_error_not_retryable_permission_denied() {
        assert!(!CloudStorageError::PermissionDenied("x".to_string()).is_retryable());
    }

    #[test]
    fn test_cloud_storage_error_not_retryable_quota_exceeded() {
        assert!(!CloudStorageError::QuotaExceeded("x".to_string()).is_retryable());
    }

    // ── MockCloudStorage basic operations ─────────────────────────────────

    #[test]
    fn test_mock_upload_and_download() {
        let store = MockCloudStorage::new();
        store
            .upload("a.ttl", b"hello")
            .expect("upload should succeed");
        let data = store.download("a.ttl").expect("download should succeed");
        assert_eq!(data, b"hello");
    }

    #[test]
    fn test_mock_download_not_found() {
        let store = MockCloudStorage::new();
        let result = store.download("missing.ttl");
        assert!(matches!(result, Err(CloudStorageError::NotFound(_))));
    }

    #[test]
    fn test_mock_exists_true() {
        let store = MockCloudStorage::new();
        store.upload("x.ttl", b"").expect("upload should succeed");
        assert!(store.exists("x.ttl").expect("exists should succeed"));
    }

    #[test]
    fn test_mock_exists_false() {
        let store = MockCloudStorage::new();
        assert!(!store.exists("absent.ttl").expect("exists should succeed"));
    }

    #[test]
    fn test_mock_delete() {
        let store = MockCloudStorage::new();
        store
            .upload("del.ttl", b"data")
            .expect("upload should succeed");
        store.delete("del.ttl").expect("delete should succeed");
        assert!(!store.exists("del.ttl").expect("exists should succeed"));
    }

    #[test]
    fn test_mock_list_prefix_filter() {
        let store = MockCloudStorage::new();
        store
            .upload("models/a.ttl", b"")
            .expect("upload should succeed");
        store
            .upload("models/b.ttl", b"")
            .expect("upload should succeed");
        store
            .upload("other/c.ttl", b"")
            .expect("upload should succeed");
        let keys = store.list("models/").expect("list should succeed");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"models/a.ttl".to_string()));
        assert!(keys.contains(&"models/b.ttl".to_string()));
    }

    #[test]
    fn test_mock_list_empty_prefix() {
        let store = MockCloudStorage::new();
        store.upload("x.ttl", b"").expect("upload should succeed");
        store.upload("y.ttl", b"").expect("upload should succeed");
        let keys = store.list("").expect("list should succeed");
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_mock_uploaded_keys_sorted() {
        let store = MockCloudStorage::new();
        store.upload("c.ttl", b"").expect("upload should succeed");
        store.upload("a.ttl", b"").expect("upload should succeed");
        store.upload("b.ttl", b"").expect("upload should succeed");
        let keys = store.uploaded_keys();
        assert_eq!(keys, vec!["a.ttl", "b.ttl", "c.ttl"]);
    }

    // ── Error injection ────────────────────────────────────────────────────

    #[test]
    fn test_mock_set_error_on_key_upload() {
        let store = MockCloudStorage::new();
        store.set_error_on_key(
            "bad.ttl",
            CloudStorageError::PermissionDenied("write".to_string()),
        );
        let result = store.upload("bad.ttl", b"data");
        assert!(matches!(
            result,
            Err(CloudStorageError::PermissionDenied(_))
        ));
    }

    #[test]
    fn test_mock_set_error_on_key_download() {
        let store = MockCloudStorage::new();
        store
            .upload("e.ttl", b"data")
            .expect("upload should succeed");
        store.set_error_on_key(
            "e.ttl",
            CloudStorageError::NetworkError("flaky".to_string()),
        );
        let result = store.download("e.ttl");
        assert!(matches!(result, Err(CloudStorageError::NetworkError(_))));
    }

    #[test]
    fn test_mock_set_error_on_key_delete() {
        let store = MockCloudStorage::new();
        store.set_error_on_key(
            "d.ttl",
            CloudStorageError::InternalError("disk".to_string()),
        );
        let result = store.delete("d.ttl");
        assert!(matches!(result, Err(CloudStorageError::InternalError(_))));
    }

    #[test]
    fn test_mock_set_error_on_key_exists() {
        let store = MockCloudStorage::new();
        store.set_error_on_key("e2.ttl", CloudStorageError::NetworkError("net".to_string()));
        let result = store.exists("e2.ttl");
        assert!(matches!(result, Err(CloudStorageError::NetworkError(_))));
    }

    #[test]
    fn test_mock_clear_errors() {
        let store = MockCloudStorage::new();
        store.set_error_on_key(
            "f.ttl",
            CloudStorageError::PermissionDenied("x".to_string()),
        );
        store.clear_errors();
        // After clearing, upload should work normally
        store
            .upload("f.ttl", b"ok")
            .expect("upload should succeed after clearing errors");
        assert!(store.exists("f.ttl").expect("exists should succeed"));
    }

    #[test]
    fn test_mock_error_not_found_on_key() {
        let store = MockCloudStorage::new();
        store.set_error_on_key("nf.ttl", CloudStorageError::NotFound("nf.ttl".to_string()));
        let result = store.download("nf.ttl");
        assert!(matches!(result, Err(CloudStorageError::NotFound(_))));
    }

    #[test]
    fn test_mock_error_permission_denied_on_key() {
        let store = MockCloudStorage::new();
        store.set_error_on_key(
            "pd.ttl",
            CloudStorageError::PermissionDenied("denied".to_string()),
        );
        let result = store.upload("pd.ttl", b"x");
        assert!(matches!(
            result,
            Err(CloudStorageError::PermissionDenied(_))
        ));
    }

    #[test]
    fn test_mock_error_quota_exceeded_on_key() {
        let store = MockCloudStorage::new();
        store.set_error_on_key(
            "qe.ttl",
            CloudStorageError::QuotaExceeded("limit".to_string()),
        );
        let result = store.upload("qe.ttl", b"x");
        assert!(matches!(result, Err(CloudStorageError::QuotaExceeded(_))));
    }

    // ── RetryableCloudClient ───────────────────────────────────────────────

    #[test]
    fn test_retryable_client_succeeds_no_retries() {
        let inner = MockCloudStorage::new();
        let client = RetryableCloudClient::new(inner, 3, 0, false);
        client
            .upload("ok.ttl", b"data")
            .expect("upload should succeed");
        assert!(client.exists("ok.ttl").expect("exists should succeed"));
    }

    #[test]
    fn test_retryable_client_retries_network_error() {
        // Upload to a key, then inject network error AFTER it's stored so retries pass.
        // We verify that a network error is returned after exhausting retries when it persists.
        let inner = MockCloudStorage::new();
        inner.set_error_on_key(
            "net.ttl",
            CloudStorageError::NetworkError("timeout".to_string()),
        );
        let client = RetryableCloudClient::new(inner, 2, 0, false);
        let result = client.upload("net.ttl", b"x");
        // All 3 attempts fail (initial + 2 retries) — should return NetworkError
        assert!(matches!(result, Err(CloudStorageError::NetworkError(_))));
    }

    #[test]
    fn test_retryable_client_does_not_retry_not_found() {
        let inner = MockCloudStorage::new();
        // No data uploaded — download should return NotFound immediately
        let client = RetryableCloudClient::new(inner, 5, 0, false);
        let result = client.download("absent.ttl");
        assert!(matches!(result, Err(CloudStorageError::NotFound(_))));
    }

    #[test]
    fn test_retryable_client_does_not_retry_permission_denied() {
        let inner = MockCloudStorage::new();
        inner.set_error_on_key(
            "pd.ttl",
            CloudStorageError::PermissionDenied("x".to_string()),
        );
        let client = RetryableCloudClient::new(inner, 5, 0, false);
        let result = client.upload("pd.ttl", b"x");
        assert!(matches!(
            result,
            Err(CloudStorageError::PermissionDenied(_))
        ));
    }

    #[test]
    fn test_retryable_client_does_not_retry_quota_exceeded() {
        let inner = MockCloudStorage::new();
        inner.set_error_on_key("qe.ttl", CloudStorageError::QuotaExceeded("x".to_string()));
        let client = RetryableCloudClient::new(inner, 5, 0, false);
        let result = client.upload("qe.ttl", b"x");
        assert!(matches!(result, Err(CloudStorageError::QuotaExceeded(_))));
    }

    #[test]
    fn test_retryable_client_exponential_backoff_no_panic() {
        let inner = MockCloudStorage::new();
        // Zero base_delay_ms so test is fast; just checks no panic
        let client = RetryableCloudClient::new(inner, 3, 0, true);
        client
            .upload("exp.ttl", b"data")
            .expect("upload should succeed");
        let data = client.download("exp.ttl").expect("download should succeed");
        assert_eq!(data, b"data");
    }

    #[test]
    fn test_retryable_client_max_retries_exhausted() {
        let inner = MockCloudStorage::new();
        inner.set_error_on_key(
            "err.ttl",
            CloudStorageError::InternalError("500".to_string()),
        );
        // max_retries = 1 means 2 total attempts
        let client = RetryableCloudClient::new(inner, 1, 0, false);
        let result = client.download("err.ttl");
        assert!(matches!(result, Err(CloudStorageError::InternalError(_))));
    }

    #[test]
    fn test_retryable_client_list_works() {
        let inner = MockCloudStorage::new();
        inner
            .upload("ns/a.ttl", b"")
            .expect("upload should succeed");
        inner
            .upload("ns/b.ttl", b"")
            .expect("upload should succeed");
        let client = RetryableCloudClient::new(inner, 2, 0, false);
        let keys = client.list("ns/").expect("list should succeed");
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_retryable_client_delete_works() {
        let inner = MockCloudStorage::new();
        inner.upload("rm.ttl", b"x").expect("upload should succeed");
        let client = RetryableCloudClient::new(inner, 2, 0, false);
        client.delete("rm.ttl").expect("delete should succeed");
        assert!(!client.exists("rm.ttl").expect("exists should succeed"));
    }
}
