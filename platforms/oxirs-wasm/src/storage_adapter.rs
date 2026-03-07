//! # Storage Adapter
//!
//! WASM storage abstraction layer providing a key-value store with namespace
//! isolation and optional TTL (time-to-live) expiry.
//!
//! Three backend variants are defined (`InMemory`, `LocalStorage`,
//! `SessionStorage`); in a native / test context all three are simulated with
//! an in-process `HashMap`.  In a real WASM build the `LocalStorage` and
//! `SessionStorage` variants would delegate to the respective Web Storage APIs.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_wasm::storage_adapter::{StorageAdapter, StorageBackend};
//!
//! let mut adapter = StorageAdapter::new(StorageBackend::InMemory, "ns");
//! adapter.set("key", b"value".to_vec(), 0).expect("set");
//! let v = adapter.get("key", 0).expect("get");
//! assert_eq!(v, b"value");
//! ```

use std::collections::HashMap;

// ─── Backend ──────────────────────────────────────────────────────────────────

/// The underlying storage mechanism.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageBackend {
    /// Pure in-process hash map (always available).
    InMemory,
    /// Browser `localStorage` (simulated in tests).
    LocalStorage,
    /// Browser `sessionStorage` (simulated in tests).
    SessionStorage,
}

// ─── Entry ────────────────────────────────────────────────────────────────────

/// A single stored entry.
#[derive(Debug, Clone)]
pub struct StorageEntry {
    /// The logical key (without namespace prefix).
    pub key: String,
    /// The stored bytes.
    pub value: Vec<u8>,
    /// Creation time in milliseconds since Unix epoch.
    pub created_at: u64,
    /// Optional TTL in milliseconds; `None` means the entry never expires.
    pub ttl_ms: Option<u64>,
}

impl StorageEntry {
    /// Returns the expiry time in milliseconds since Unix epoch, or `None` if
    /// the entry has no TTL.
    pub fn expires_at(&self) -> Option<u64> {
        self.ttl_ms.map(|t| self.created_at + t)
    }

    /// Returns `true` when the entry has expired relative to `now_ms`.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expires_at().is_some_and(|exp| now_ms > exp)
    }
}

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by [`StorageAdapter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    /// The requested key was not found in this namespace.
    KeyNotFound(String),
    /// The requested namespace does not exist (future use).
    NamespaceNotFound(String),
    /// The entry exists but has expired.
    Expired(String),
    /// Serialization / deserialization failure.
    SerializeError(String),
    /// Storage quota would be exceeded.
    QuotaExceeded,
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::KeyNotFound(k) => write!(f, "Key not found: {k}"),
            StorageError::NamespaceNotFound(ns) => write!(f, "Namespace not found: {ns}"),
            StorageError::Expired(k) => write!(f, "Entry expired: {k}"),
            StorageError::SerializeError(msg) => write!(f, "Serialize error: {msg}"),
            StorageError::QuotaExceeded => write!(f, "Storage quota exceeded"),
        }
    }
}

impl std::error::Error for StorageError {}

// ─── Adapter ──────────────────────────────────────────────────────────────────

/// A namespaced key-value store backed by one of the [`StorageBackend`]s.
///
/// Keys stored in the adapter are logically scoped to the namespace; callers
/// use bare keys and the adapter handles prefixing internally.
pub struct StorageAdapter {
    /// The backend variant.
    backend: StorageBackend,
    /// Namespace prefix used to isolate this adapter's keys.
    namespace: String,
    /// Internal storage: bare key → entry.
    store: HashMap<String, StorageEntry>,
}

impl StorageAdapter {
    /// Create a new adapter with the given backend and namespace.
    pub fn new(backend: StorageBackend, namespace: &str) -> Self {
        Self {
            backend,
            namespace: namespace.to_string(),
            store: HashMap::new(),
        }
    }

    /// Return the namespace this adapter operates in.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Return the backend variant.
    pub fn backend(&self) -> &StorageBackend {
        &self.backend
    }

    // ── Write operations ──────────────────────────────────────────────────────

    /// Store `value` under `key` with no expiry.
    pub fn set(&mut self, key: &str, value: Vec<u8>, now_ms: u64) -> Result<(), StorageError> {
        let entry = StorageEntry {
            key: key.to_string(),
            value,
            created_at: now_ms,
            ttl_ms: None,
        };
        self.store.insert(key.to_string(), entry);
        Ok(())
    }

    /// Store `value` under `key` with an expiry of `ttl_ms` milliseconds from
    /// `now_ms`.
    pub fn set_with_ttl(
        &mut self,
        key: &str,
        value: Vec<u8>,
        now_ms: u64,
        ttl_ms: u64,
    ) -> Result<(), StorageError> {
        let entry = StorageEntry {
            key: key.to_string(),
            value,
            created_at: now_ms,
            ttl_ms: Some(ttl_ms),
        };
        self.store.insert(key.to_string(), entry);
        Ok(())
    }

    // ── Read operations ───────────────────────────────────────────────────────

    /// Retrieve the value stored under `key`.
    ///
    /// Returns [`StorageError::Expired`] if the entry's TTL has elapsed, and
    /// [`StorageError::KeyNotFound`] if the key does not exist.
    pub fn get(&self, key: &str, now_ms: u64) -> Result<Vec<u8>, StorageError> {
        match self.store.get(key) {
            None => Err(StorageError::KeyNotFound(key.to_string())),
            Some(entry) => {
                if entry.is_expired(now_ms) {
                    Err(StorageError::Expired(key.to_string()))
                } else {
                    Ok(entry.value.clone())
                }
            }
        }
    }

    /// Return a reference to the [`StorageEntry`] for `key`, or `None`.
    ///
    /// Does **not** perform expiry checking — use [`Self::get`] for normal access.
    pub fn get_entry(&self, key: &str) -> Option<&StorageEntry> {
        self.store.get(key)
    }

    // ── Delete operations ─────────────────────────────────────────────────────

    /// Delete the entry for `key`.  Returns `true` if the key existed.
    pub fn delete(&mut self, key: &str) -> bool {
        self.store.remove(key).is_some()
    }

    /// Remove all entries in this namespace.
    pub fn clear_namespace(&mut self) {
        self.store.clear();
    }

    // ── Query operations ──────────────────────────────────────────────────────

    /// Return all keys in this namespace whose bare key starts with `prefix`.
    ///
    /// The returned keys are the bare keys (without the namespace prefix).
    /// The order is unspecified.
    pub fn list_keys(&self, prefix: &str) -> Vec<String> {
        let mut keys: Vec<String> = self
            .store
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        keys.sort();
        keys
    }

    /// Purge all expired entries (those whose TTL has elapsed as of `now_ms`).
    ///
    /// Returns the number of entries removed.
    pub fn purge_expired(&mut self, now_ms: u64) -> usize {
        let before = self.store.len();
        self.store.retain(|_k, entry| !entry.is_expired(now_ms));
        before - self.store.len()
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    /// Return the number of entries currently stored (including expired ones
    /// that have not yet been purged).
    pub fn key_count(&self) -> usize {
        self.store.len()
    }

    /// Return the total size in bytes of all stored values (including expired
    /// entries that have not been purged).
    pub fn storage_size_bytes(&self) -> usize {
        self.store.values().map(|e| e.value.len()).sum()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_adapter(ns: &str) -> StorageAdapter {
        StorageAdapter::new(StorageBackend::InMemory, ns)
    }

    // ── StorageEntry ──────────────────────────────────────────────────────────

    #[test]
    fn test_entry_no_ttl_never_expires() {
        let e = StorageEntry {
            key: "k".to_string(),
            value: b"v".to_vec(),
            created_at: 0,
            ttl_ms: None,
        };
        assert!(!e.is_expired(u64::MAX));
        assert!(e.expires_at().is_none());
    }

    #[test]
    fn test_entry_expires_at_correct_time() {
        let e = StorageEntry {
            key: "k".to_string(),
            value: b"v".to_vec(),
            created_at: 1_000,
            ttl_ms: Some(500),
        };
        assert_eq!(e.expires_at(), Some(1_500));
        assert!(!e.is_expired(1_499));
        assert!(!e.is_expired(1_500)); // exact boundary — not yet expired
        assert!(e.is_expired(1_501));
    }

    #[test]
    fn test_entry_clone() {
        let e = StorageEntry {
            key: "x".to_string(),
            value: vec![1, 2, 3],
            created_at: 10,
            ttl_ms: Some(20),
        };
        let c = e.clone();
        assert_eq!(c.key, e.key);
        assert_eq!(c.value, e.value);
    }

    // ── StorageError ──────────────────────────────────────────────────────────

    #[test]
    fn test_error_key_not_found_display() {
        let e = StorageError::KeyNotFound("foo".to_string());
        assert!(e.to_string().contains("foo"));
    }

    #[test]
    fn test_error_expired_display() {
        let e = StorageError::Expired("bar".to_string());
        assert!(e.to_string().contains("bar"));
    }

    #[test]
    fn test_error_quota_exceeded_display() {
        let e = StorageError::QuotaExceeded;
        assert!(e.to_string().contains("quota"));
    }

    #[test]
    fn test_error_serialize_display() {
        let e = StorageError::SerializeError("bad bytes".to_string());
        assert!(e.to_string().contains("bad bytes"));
    }

    // ── set / get ─────────────────────────────────────────────────────────────

    #[test]
    fn test_set_and_get() {
        let mut a = mem_adapter("ns");
        a.set("hello", b"world".to_vec(), 0).expect("set");
        let v = a.get("hello", 0).expect("get");
        assert_eq!(v, b"world");
    }

    #[test]
    fn test_get_missing_key_returns_not_found() {
        let a = mem_adapter("ns");
        let err = a.get("missing", 0).expect_err("should fail");
        assert_eq!(err, StorageError::KeyNotFound("missing".to_string()));
    }

    #[test]
    fn test_set_overwrites_existing() {
        let mut a = mem_adapter("ns");
        a.set("k", b"v1".to_vec(), 0).expect("first set");
        a.set("k", b"v2".to_vec(), 1).expect("second set");
        let v = a.get("k", 2).expect("get");
        assert_eq!(v, b"v2");
    }

    #[test]
    fn test_set_multiple_keys() {
        let mut a = mem_adapter("ns");
        a.set("a", b"1".to_vec(), 0).expect("a");
        a.set("b", b"2".to_vec(), 0).expect("b");
        a.set("c", b"3".to_vec(), 0).expect("c");
        assert_eq!(a.key_count(), 3);
    }

    // ── set_with_ttl ──────────────────────────────────────────────────────────

    #[test]
    fn test_set_with_ttl_before_expiry() {
        let mut a = mem_adapter("ns");
        a.set_with_ttl("k", b"v".to_vec(), 0, 1_000)
            .expect("set_with_ttl");
        let v = a.get("k", 999).expect("get before expiry");
        assert_eq!(v, b"v");
    }

    #[test]
    fn test_set_with_ttl_at_boundary_not_expired() {
        let mut a = mem_adapter("ns");
        a.set_with_ttl("k", b"v".to_vec(), 0, 1_000).expect("set");
        let v = a.get("k", 1_000).expect("get at boundary");
        assert_eq!(v, b"v");
    }

    #[test]
    fn test_set_with_ttl_after_expiry() {
        let mut a = mem_adapter("ns");
        a.set_with_ttl("k", b"v".to_vec(), 0, 500).expect("set");
        let err = a.get("k", 501).expect_err("should be expired");
        assert_eq!(err, StorageError::Expired("k".to_string()));
    }

    #[test]
    fn test_set_with_ttl_zero_expires_immediately() {
        let mut a = mem_adapter("ns");
        a.set_with_ttl("k", b"v".to_vec(), 0, 0).expect("set");
        // created_at=0, ttl=0 → expires_at=0 → now=1 is after → expired
        let err = a.get("k", 1).expect_err("zero-ttl should expire");
        assert_eq!(err, StorageError::Expired("k".to_string()));
    }

    // ── delete ────────────────────────────────────────────────────────────────

    #[test]
    fn test_delete_existing_key_returns_true() {
        let mut a = mem_adapter("ns");
        a.set("k", b"v".to_vec(), 0).expect("set");
        assert!(a.delete("k"));
    }

    #[test]
    fn test_delete_missing_key_returns_false() {
        let mut a = mem_adapter("ns");
        assert!(!a.delete("nope"));
    }

    #[test]
    fn test_delete_removes_key_from_store() {
        let mut a = mem_adapter("ns");
        a.set("k", b"v".to_vec(), 0).expect("set");
        a.delete("k");
        let err = a.get("k", 0).expect_err("key should be gone");
        assert_eq!(err, StorageError::KeyNotFound("k".to_string()));
    }

    // ── clear_namespace ───────────────────────────────────────────────────────

    #[test]
    fn test_clear_namespace_removes_all_keys() {
        let mut a = mem_adapter("ns");
        a.set("a", b"1".to_vec(), 0).expect("a");
        a.set("b", b"2".to_vec(), 0).expect("b");
        a.clear_namespace();
        assert_eq!(a.key_count(), 0);
    }

    #[test]
    fn test_clear_namespace_idempotent_on_empty() {
        let mut a = mem_adapter("ns");
        a.clear_namespace();
        a.clear_namespace(); // second call should not panic
        assert_eq!(a.key_count(), 0);
    }

    // ── list_keys ─────────────────────────────────────────────────────────────

    #[test]
    fn test_list_keys_empty_prefix_returns_all() {
        let mut a = mem_adapter("ns");
        a.set("foo", b"1".to_vec(), 0).expect("foo");
        a.set("bar", b"2".to_vec(), 0).expect("bar");
        let keys = a.list_keys("");
        assert_eq!(keys.len(), 2);
        // sorted
        assert_eq!(keys[0], "bar");
        assert_eq!(keys[1], "foo");
    }

    #[test]
    fn test_list_keys_prefix_filter() {
        let mut a = mem_adapter("ns");
        a.set("user:alice", b"1".to_vec(), 0).expect("a");
        a.set("user:bob", b"2".to_vec(), 0).expect("b");
        a.set("config:x", b"3".to_vec(), 0).expect("c");
        let keys = a.list_keys("user:");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"user:alice".to_string()));
        assert!(keys.contains(&"user:bob".to_string()));
    }

    #[test]
    fn test_list_keys_no_match_returns_empty() {
        let mut a = mem_adapter("ns");
        a.set("alpha", b"1".to_vec(), 0).expect("set");
        let keys = a.list_keys("beta");
        assert!(keys.is_empty());
    }

    #[test]
    fn test_list_keys_sorted_alphabetically() {
        let mut a = mem_adapter("ns");
        for ch in ['z', 'a', 'm', 'b'] {
            a.set(&ch.to_string(), b"v".to_vec(), 0).expect("set");
        }
        let keys = a.list_keys("");
        assert_eq!(keys, vec!["a", "b", "m", "z"]);
    }

    // ── purge_expired ─────────────────────────────────────────────────────────

    #[test]
    fn test_purge_expired_removes_only_stale_entries() {
        let mut a = mem_adapter("ns");
        a.set_with_ttl("stale", b"s".to_vec(), 0, 100)
            .expect("stale");
        a.set("fresh", b"f".to_vec(), 0).expect("fresh");
        let removed = a.purge_expired(200);
        assert_eq!(removed, 1);
        assert_eq!(a.key_count(), 1);
        assert!(a.get("fresh", 200).is_ok());
    }

    #[test]
    fn test_purge_expired_returns_zero_when_none_expired() {
        let mut a = mem_adapter("ns");
        a.set("a", b"1".to_vec(), 0).expect("a");
        a.set("b", b"2".to_vec(), 0).expect("b");
        let removed = a.purge_expired(999);
        assert_eq!(removed, 0);
        assert_eq!(a.key_count(), 2);
    }

    #[test]
    fn test_purge_expired_removes_all_when_all_stale() {
        let mut a = mem_adapter("ns");
        a.set_with_ttl("x", b"1".to_vec(), 0, 10).expect("x");
        a.set_with_ttl("y", b"2".to_vec(), 0, 10).expect("y");
        let removed = a.purge_expired(100);
        assert_eq!(removed, 2);
        assert_eq!(a.key_count(), 0);
    }

    #[test]
    fn test_purge_expired_idempotent() {
        let mut a = mem_adapter("ns");
        a.set_with_ttl("k", b"v".to_vec(), 0, 5).expect("set");
        a.purge_expired(100);
        let removed2 = a.purge_expired(200);
        assert_eq!(removed2, 0);
    }

    // ── key_count ─────────────────────────────────────────────────────────────

    #[test]
    fn test_key_count_empty() {
        let a = mem_adapter("ns");
        assert_eq!(a.key_count(), 0);
    }

    #[test]
    fn test_key_count_after_operations() {
        let mut a = mem_adapter("ns");
        a.set("a", b"1".to_vec(), 0).expect("a");
        a.set("b", b"2".to_vec(), 0).expect("b");
        assert_eq!(a.key_count(), 2);
        a.delete("a");
        assert_eq!(a.key_count(), 1);
    }

    // ── storage_size_bytes ────────────────────────────────────────────────────

    #[test]
    fn test_storage_size_bytes_empty() {
        let a = mem_adapter("ns");
        assert_eq!(a.storage_size_bytes(), 0);
    }

    #[test]
    fn test_storage_size_bytes_accumulates() {
        let mut a = mem_adapter("ns");
        a.set("a", vec![0u8; 10], 0).expect("a");
        a.set("b", vec![0u8; 20], 0).expect("b");
        assert_eq!(a.storage_size_bytes(), 30);
    }

    #[test]
    fn test_storage_size_bytes_decreases_after_delete() {
        let mut a = mem_adapter("ns");
        a.set("a", vec![0u8; 10], 0).expect("a");
        a.set("b", vec![0u8; 5], 0).expect("b");
        a.delete("a");
        assert_eq!(a.storage_size_bytes(), 5);
    }

    // ── namespace isolation ───────────────────────────────────────────────────

    #[test]
    fn test_namespace_isolation_same_key_different_adapters() {
        let mut a1 = StorageAdapter::new(StorageBackend::InMemory, "ns1");
        let mut a2 = StorageAdapter::new(StorageBackend::InMemory, "ns2");

        a1.set("key", b"val1".to_vec(), 0).expect("a1 set");
        a2.set("key", b"val2".to_vec(), 0).expect("a2 set");

        assert_eq!(a1.get("key", 0).expect("a1 get"), b"val1");
        assert_eq!(a2.get("key", 0).expect("a2 get"), b"val2");

        a1.delete("key");
        // a2 is unaffected
        assert_eq!(a2.get("key", 0).expect("a2 still has key"), b"val2");
    }

    #[test]
    fn test_clear_namespace_does_not_affect_other_adapter() {
        let mut a1 = StorageAdapter::new(StorageBackend::InMemory, "ns1");
        let mut a2 = StorageAdapter::new(StorageBackend::InMemory, "ns2");

        a1.set("x", b"1".to_vec(), 0).expect("a1");
        a2.set("x", b"2".to_vec(), 0).expect("a2");

        a1.clear_namespace();
        assert_eq!(a1.key_count(), 0);
        assert_eq!(a2.key_count(), 1);
    }

    // ── backend variants ──────────────────────────────────────────────────────

    #[test]
    fn test_backend_local_storage_simulated() {
        let mut a = StorageAdapter::new(StorageBackend::LocalStorage, "ns");
        a.set("k", b"v".to_vec(), 0).expect("set");
        assert_eq!(a.get("k", 0).expect("get"), b"v");
    }

    #[test]
    fn test_backend_session_storage_simulated() {
        let mut a = StorageAdapter::new(StorageBackend::SessionStorage, "ns");
        a.set("k", b"v".to_vec(), 0).expect("set");
        assert_eq!(a.get("k", 0).expect("get"), b"v");
    }

    #[test]
    fn test_backend_accessor() {
        let a = StorageAdapter::new(StorageBackend::InMemory, "ns");
        assert_eq!(a.backend(), &StorageBackend::InMemory);
    }

    #[test]
    fn test_namespace_accessor() {
        let a = StorageAdapter::new(StorageBackend::InMemory, "myns");
        assert_eq!(a.namespace(), "myns");
    }

    // ── get_entry ─────────────────────────────────────────────────────────────

    #[test]
    fn test_get_entry_returns_none_for_missing() {
        let a = mem_adapter("ns");
        assert!(a.get_entry("missing").is_none());
    }

    #[test]
    fn test_get_entry_returns_entry_with_ttl() {
        let mut a = mem_adapter("ns");
        a.set_with_ttl("k", b"v".to_vec(), 100, 200).expect("set");
        let entry = a.get_entry("k").expect("entry should exist");
        assert_eq!(entry.created_at, 100);
        assert_eq!(entry.ttl_ms, Some(200));
        assert_eq!(entry.expires_at(), Some(300));
    }
}
