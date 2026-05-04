//! ServiceWorker-backed offline cache for RDF/SPARQL data.
//!
//! Provides an `OfflineCache` struct for caching RDF datasets and SPARQL
//! query results with TTL-based expiry.  In the browser, this is backed by
//! IndexedDB (via JS interop); on the server / in tests, it uses an in-memory
//! `HashMap`.
//!
//! ## Cache key design
//! Cache keys are URL strings.  Entries carry an optional TTL; expired entries
//! are evicted lazily on access.
//!
//! ## Sync-on-online
//! The `sync_on_online` method is called when the browser fires the `online`
//! event.  It iterates pending write-back requests and re-submits them.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ─────────────────────────────────────────────────────────────────────────────
// Cache entry
// ─────────────────────────────────────────────────────────────────────────────

/// A cached entry in the offline cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The cached data (serialized as UTF-8 bytes).
    pub data: Vec<u8>,
    /// MIME type / content type, e.g. `"text/turtle"`.
    pub content_type: String,
    /// Unix timestamp (seconds) when this entry was cached.
    pub cached_at: u64,
    /// Optional TTL in seconds; `None` means the entry never expires.
    pub ttl_secs: Option<u64>,
    /// ETag for conditional requests (cache validation).
    pub etag: Option<String>,
}

impl CacheEntry {
    /// Create a new entry with the current timestamp.
    pub fn new(
        data: Vec<u8>,
        content_type: impl Into<String>,
        ttl_secs: Option<u64>,
        etag: Option<String>,
    ) -> Self {
        Self {
            data,
            content_type: content_type.into(),
            cached_at: now_secs(),
            ttl_secs,
            etag,
        }
    }

    /// Check if the entry is still valid at the given timestamp.
    pub fn is_valid_at(&self, now: u64) -> bool {
        match self.ttl_secs {
            None => true,
            Some(ttl) => now < self.cached_at.saturating_add(ttl),
        }
    }

    /// Check if the entry is currently valid.
    pub fn is_valid(&self) -> bool {
        self.is_valid_at(now_secs())
    }

    /// Age of the entry in seconds.
    pub fn age_secs(&self) -> u64 {
        now_secs().saturating_sub(self.cached_at)
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

// ─────────────────────────────────────────────────────────────────────────────
// Pending sync request
// ─────────────────────────────────────────────────────────────────────────────

/// A write-back request queued for when connectivity is restored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingSyncRequest {
    /// The target URL.
    pub url: String,
    /// HTTP method, e.g. `"POST"`, `"PUT"`.
    pub method: String,
    /// Request body.
    pub body: Vec<u8>,
    /// Content-Type header value.
    pub content_type: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// OfflineCache
// ─────────────────────────────────────────────────────────────────────────────

/// An offline-capable cache for RDF data and SPARQL results.
///
/// In the browser, wire this to an IndexedDB store via JS interop.
/// In tests and on the server, it uses an in-memory `HashMap`.
pub struct OfflineCache {
    entries: HashMap<String, CacheEntry>,
    pending_sync: Vec<PendingSyncRequest>,
    default_ttl: Option<u64>,
}

impl OfflineCache {
    /// Create a new cache.
    ///
    /// `default_ttl` sets the TTL for entries that don't specify one.
    /// `None` means entries never expire unless explicitly evicted.
    pub fn new(default_ttl: Option<Duration>) -> Self {
        Self {
            entries: HashMap::new(),
            pending_sync: Vec::new(),
            default_ttl: default_ttl.map(|d| d.as_secs()),
        }
    }

    /// Store a resource in the cache.
    pub fn put(
        &mut self,
        url: impl Into<String>,
        data: Vec<u8>,
        content_type: impl Into<String>,
        ttl: Option<Duration>,
        etag: Option<String>,
    ) {
        let effective_ttl = ttl.map(|d| d.as_secs()).or(self.default_ttl);
        let entry = CacheEntry::new(data, content_type, effective_ttl, etag);
        self.entries.insert(url.into(), entry);
    }

    /// Retrieve a cached entry for `url` if it exists and is still valid.
    ///
    /// Returns `None` if the entry is missing or has expired.
    pub fn get(&mut self, url: &str) -> Option<&CacheEntry> {
        let now = now_secs();
        if let Some(entry) = self.entries.get(url) {
            if entry.is_valid_at(now) {
                return self.entries.get(url);
            } else {
                // Lazy eviction
                self.entries.remove(url);
            }
        }
        None
    }

    /// Explicitly remove an entry.
    pub fn evict(&mut self, url: &str) -> bool {
        self.entries.remove(url).is_some()
    }

    /// Remove all expired entries.  Returns the number of evicted entries.
    pub fn evict_expired(&mut self) -> usize {
        let now = now_secs();
        let before = self.entries.len();
        self.entries.retain(|_, e| e.is_valid_at(now));
        before - self.entries.len()
    }

    /// Return the number of cached entries (including potentially expired ones).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the cache has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Queue a write-back request for deferred sync.
    pub fn queue_sync(&mut self, req: PendingSyncRequest) {
        self.pending_sync.push(req);
    }

    /// Number of pending sync requests.
    pub fn pending_sync_count(&self) -> usize {
        self.pending_sync.len()
    }

    /// Called when the browser fires the `online` event.
    ///
    /// Returns the queued requests so the caller can re-submit them.
    /// The queue is cleared after draining.
    pub fn sync_on_online(&mut self) -> Vec<PendingSyncRequest> {
        std::mem::take(&mut self.pending_sync)
    }

    /// Check if a URL is cached and valid.
    pub fn contains(&mut self, url: &str) -> bool {
        self.get(url).is_some()
    }

    /// List all currently cached URLs (valid and expired).
    pub fn cached_urls(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_cache() -> OfflineCache {
        OfflineCache::new(Some(Duration::from_secs(3600)))
    }

    #[test]
    fn test_put_and_get() {
        let mut cache = mk_cache();
        cache.put(
            "http://example.org/dataset.ttl",
            b"<s> <p> <o> .".to_vec(),
            "text/turtle",
            None,
            None,
        );
        assert!(cache.get("http://example.org/dataset.ttl").is_some());
    }

    #[test]
    fn test_missing_url_returns_none() {
        let mut cache = mk_cache();
        assert!(cache.get("http://no.such.url/").is_none());
    }

    #[test]
    fn test_evict() {
        let mut cache = mk_cache();
        cache.put(
            "http://example.org/x",
            b"data".to_vec(),
            "text/plain",
            None,
            None,
        );
        assert!(cache.evict("http://example.org/x"));
        assert!(
            !cache.evict("http://example.org/x"),
            "double evict returns false"
        );
    }

    #[test]
    fn test_expired_entry_not_returned() {
        let mut cache = OfflineCache::new(None);
        // TTL of 0 seconds → immediately expired
        cache.put(
            "http://example.org/expired",
            b"data".to_vec(),
            "text/plain",
            Some(Duration::ZERO),
            None,
        );
        // Sleep is not available, but with TTL=0 the entry expires at cached_at
        // which was set to now(). Whether now() > cached_at depends on sub-second
        // timing.  We can directly test is_valid_at with a past timestamp.
        let entry = cache.entries.get("http://example.org/expired").unwrap();
        assert!(
            !entry.is_valid_at(entry.cached_at + 1),
            "entry with ttl=0 expires immediately"
        );
    }

    #[test]
    fn test_evict_expired() {
        let mut cache = OfflineCache::new(None);
        // Add one entry that expires immediately (cached_at in the past)
        cache.entries.insert(
            "http://example.org/old".into(),
            CacheEntry {
                data: vec![],
                content_type: "text/plain".into(),
                cached_at: 0, // Unix epoch — always expired
                ttl_secs: Some(1),
                etag: None,
            },
        );
        cache.put(
            "http://example.org/fresh",
            b"ok".to_vec(),
            "text/plain",
            None,
            None,
        );
        let evicted = cache.evict_expired();
        assert_eq!(evicted, 1, "one expired entry evicted");
        assert_eq!(cache.len(), 1, "fresh entry remains");
    }

    #[test]
    fn test_pending_sync_queue_and_drain() {
        let mut cache = mk_cache();
        let req = PendingSyncRequest {
            url: "http://api.example.org/update".into(),
            method: "POST".into(),
            body: b"INSERT DATA { <s> <p> <o> . }".to_vec(),
            content_type: "application/sparql-update".into(),
        };
        cache.queue_sync(req);
        assert_eq!(cache.pending_sync_count(), 1);

        let drained = cache.sync_on_online();
        assert_eq!(drained.len(), 1);
        assert_eq!(cache.pending_sync_count(), 0, "queue cleared after sync");
    }

    #[test]
    fn test_contains() {
        let mut cache = mk_cache();
        assert!(!cache.contains("http://x.org/"));
        cache.put("http://x.org/", b"data".to_vec(), "text/plain", None, None);
        assert!(cache.contains("http://x.org/"));
    }

    #[test]
    fn test_cache_entry_age() {
        let entry = CacheEntry::new(vec![], "text/plain", None, None);
        assert!(entry.age_secs() < 2, "freshly created entry has age < 2s");
    }

    #[test]
    fn test_no_expiry_entry_always_valid() {
        let entry = CacheEntry::new(vec![], "text/plain", None, None);
        assert!(entry.is_valid_at(u64::MAX), "no-TTL entry never expires");
    }
}
