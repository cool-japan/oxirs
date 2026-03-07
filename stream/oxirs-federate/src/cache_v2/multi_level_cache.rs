//! Multi-level cache: L1 (in-memory LRU) + L2 (disk-backed) for federated
//! query results.
//!
//! # Architecture
//!
//! ```text
//!   caller
//!     │
//!     ▼
//!   ┌─────────────────────────────────────────────────┐
//!   │  MultiLevelCache                                │
//!   │  ┌───────────────────┐  ┌─────────────────────┐│
//!   │  │  L1: MemoryCache  │  │  L2: DiskCache      ││
//!   │  │  (LRU, in-process)│  │  (files, tempdir)   ││
//!   │  └───────────────────┘  └─────────────────────┘│
//!   └─────────────────────────────────────────────────┘
//! ```
//!
//! Lookup order: L1 → L2 → miss.
//! Write order: write to L1 first, then asynchronously promote to L2.
//! Eviction: L1 uses capacity-based LRU; L2 uses TTL-based sweep.
//!
//! # Thread safety
//!
//! All public methods take `&self` and use internal `Mutex`/`RwLock`
//! synchronisation.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{self, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ─── CacheKey ─────────────────────────────────────────────────────────────────

/// Opaque key for a cached entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey(pub String);

impl CacheKey {
    /// Create a key from an arbitrary string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Compute a key from a query text and a sorted list of endpoint IDs.
    pub fn from_query(query: &str, mut endpoints: Vec<String>) -> Self {
        endpoints.sort();
        let mut h = DefaultHasher::new();
        query.hash(&mut h);
        endpoints.hash(&mut h);
        Self(format!("{:016x}", h.finish()))
    }

    /// Return the string representation of this key.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ─── CacheEntry ───────────────────────────────────────────────────────────────

/// A cached federated query result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The cache key.
    pub key: CacheKey,
    /// Result bindings: each row is a map from variable name to value string.
    pub result_bindings: Vec<HashMap<String, String>>,
    /// Endpoint IDs whose data contributed to this result (used for invalidation).
    pub contributing_endpoints: Vec<String>,
    /// Unix timestamp (seconds) when this entry was created.
    pub created_at_secs: u64,
    /// Maximum time-to-live for this entry.
    pub ttl_secs: u64,
    /// Number of times this entry has been returned as a cache hit.
    pub hit_count: u64,
}

impl CacheEntry {
    /// Create a new cache entry with the given TTL.
    pub fn new(
        key: CacheKey,
        result_bindings: Vec<HashMap<String, String>>,
        contributing_endpoints: Vec<String>,
        ttl: Duration,
    ) -> Self {
        let created_at_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            key,
            result_bindings,
            contributing_endpoints,
            created_at_secs,
            ttl_secs: ttl.as_secs(),
            hit_count: 0,
        }
    }

    /// Check whether this entry has expired.
    pub fn is_expired(&self) -> bool {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now_secs.saturating_sub(self.created_at_secs) >= self.ttl_secs
    }

    /// Remaining TTL (0 if already expired).
    pub fn remaining_ttl(&self) -> Duration {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let elapsed = now_secs.saturating_sub(self.created_at_secs);
        if elapsed >= self.ttl_secs {
            Duration::ZERO
        } else {
            Duration::from_secs(self.ttl_secs - elapsed)
        }
    }
}

// ─── L1: MemoryCache (LRU) ────────────────────────────────────────────────────

/// Node in the LRU doubly-linked list.
struct LruNode {
    key: CacheKey,
    entry: CacheEntry,
    prev: Option<CacheKey>,
    next: Option<CacheKey>,
}

/// In-memory LRU cache backed by a HashMap + doubly-linked-list for O(1) ops.
///
/// # Implementation notes
///
/// Rust ownership prevents a traditional pointer-based DLL, so we use HashMap
/// keyed by `CacheKey` and track prev/next explicitly.
pub struct MemoryCache {
    capacity: usize,
    map: HashMap<CacheKey, LruNode>,
    head: Option<CacheKey>, // most-recently-used
    tail: Option<CacheKey>, // least-recently-used
    hits: u64,
    misses: u64,
}

impl MemoryCache {
    /// Create a new `MemoryCache` with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            head: None,
            tail: None,
            hits: 0,
            misses: 0,
        }
    }

    /// Get an entry by key, promoting it to MRU position.
    pub fn get(&mut self, key: &CacheKey) -> Option<&CacheEntry> {
        if !self.map.contains_key(key) {
            self.misses += 1;
            return None;
        }
        // Promote to head
        self.detach(key.clone());
        self.attach_head(key.clone());
        if let Some(node) = self.map.get_mut(key) {
            if node.entry.is_expired() {
                // Expire: remove and report miss
                let remove_key = key.clone();
                self.detach(remove_key.clone());
                self.map.remove(&remove_key);
                self.misses += 1;
                return None;
            }
            node.entry.hit_count += 1;
            self.hits += 1;
            // Return via map re-lookup to satisfy borrow checker
        }
        self.map.get(key).map(|n| &n.entry)
    }

    /// Insert or update an entry.  Evicts the LRU entry if at capacity.
    pub fn put(&mut self, entry: CacheEntry) {
        let key = entry.key.clone();
        if self.map.contains_key(&key) {
            self.detach(key.clone());
        } else if self.map.len() >= self.capacity {
            if let Some(lru_key) = self.tail.clone() {
                self.detach(lru_key.clone());
                self.map.remove(&lru_key);
            }
        }
        self.map.insert(
            key.clone(),
            LruNode {
                key: key.clone(),
                entry,
                prev: None,
                next: None,
            },
        );
        self.attach_head(key);
    }

    /// Remove an entry by key.
    pub fn remove(&mut self, key: &CacheKey) {
        if self.map.contains_key(key) {
            self.detach(key.clone());
            self.map.remove(key);
        }
    }

    /// Remove all entries whose `contributing_endpoints` contains `endpoint_id`.
    pub fn invalidate_endpoint(&mut self, endpoint_id: &str) {
        let to_remove: Vec<CacheKey> = self
            .map
            .values()
            .filter(|n| {
                n.entry
                    .contributing_endpoints
                    .iter()
                    .any(|e| e == endpoint_id)
            })
            .map(|n| n.key.clone())
            .collect();
        for key in to_remove {
            self.detach(key.clone());
            self.map.remove(&key);
        }
    }

    /// Remove all expired entries.
    pub fn evict_expired(&mut self) {
        let expired: Vec<CacheKey> = self
            .map
            .values()
            .filter(|n| n.entry.is_expired())
            .map(|n| n.key.clone())
            .collect();
        for key in expired {
            self.detach(key.clone());
            self.map.remove(&key);
        }
    }

    /// Number of entries currently in cache.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Cache hit count since creation.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count since creation.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit rate in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    // ── Private DLL helpers ────────────────────────────────────────────────

    fn detach(&mut self, key: CacheKey) {
        let (prev, next) = match self.map.get(&key) {
            None => return,
            Some(node) => (node.prev.clone(), node.next.clone()),
        };
        if let Some(ref p) = prev {
            if let Some(pn) = self.map.get_mut(p) {
                pn.next = next.clone();
            }
        } else {
            self.head = next.clone();
        }
        if let Some(ref n) = next {
            if let Some(nn) = self.map.get_mut(n) {
                nn.prev = prev.clone();
            }
        } else {
            self.tail = prev.clone();
        }
        if let Some(node) = self.map.get_mut(&key) {
            node.prev = None;
            node.next = None;
        }
    }

    fn attach_head(&mut self, key: CacheKey) {
        let old_head = self.head.clone();
        if let Some(node) = self.map.get_mut(&key) {
            node.prev = None;
            node.next = old_head.clone();
        }
        if let Some(ref h) = old_head {
            if let Some(hn) = self.map.get_mut(h) {
                hn.prev = Some(key.clone());
            }
        } else {
            self.tail = Some(key.clone());
        }
        self.head = Some(key);
    }
}

// ─── L2: DiskCache ────────────────────────────────────────────────────────────

/// File-backed disk cache.  Each entry is serialised as JSON and stored in a
/// file named `<hex-key>.json` under the given directory.
pub struct DiskCache {
    dir: PathBuf,
    hits: u64,
    misses: u64,
}

impl DiskCache {
    /// Open (or create) a disk cache in the given directory.
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self {
            dir,
            hits: 0,
            misses: 0,
        })
    }

    /// Open a disk cache in a new temporary directory (useful for tests).
    pub fn open_temp() -> io::Result<Self> {
        let dir = std::env::temp_dir().join(format!(
            "oxirs-federate-l2-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        Self::open(dir)
    }

    /// Get an entry by key.  Returns `None` for misses and expired entries.
    pub fn get(&mut self, key: &CacheKey) -> Option<CacheEntry> {
        let path = self.entry_path(key);
        let bytes = fs::read(&path).ok()?;
        let entry: CacheEntry = serde_json::from_slice(&bytes).ok()?;
        if entry.is_expired() {
            let _ = fs::remove_file(&path);
            self.misses += 1;
            return None;
        }
        self.hits += 1;
        Some(entry)
    }

    /// Store an entry on disk.
    pub fn put(&mut self, entry: &CacheEntry) -> io::Result<()> {
        let path = self.entry_path(&entry.key);
        let json = serde_json::to_vec(entry).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("serialize error: {}", e),
            )
        })?;
        let mut file = fs::File::create(&path)?;
        file.write_all(&json)?;
        Ok(())
    }

    /// Remove an entry from disk.
    pub fn remove(&mut self, key: &CacheKey) {
        let path = self.entry_path(key);
        let _ = fs::remove_file(path);
    }

    /// Remove all entries whose `contributing_endpoints` contains `endpoint_id`.
    pub fn invalidate_endpoint(&mut self, endpoint_id: &str) {
        if let Ok(read_dir) = fs::read_dir(&self.dir) {
            for entry_result in read_dir.flatten() {
                let path = entry_result.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let should_remove = fs::read(&path)
                    .ok()
                    .and_then(|b| serde_json::from_slice::<CacheEntry>(&b).ok())
                    .map(|e| e.contributing_endpoints.iter().any(|ep| ep == endpoint_id))
                    .unwrap_or(false);
                if should_remove {
                    let _ = fs::remove_file(path);
                }
            }
        }
    }

    /// Sweep the directory and remove all expired entries.
    pub fn evict_expired(&mut self) {
        if let Ok(read_dir) = fs::read_dir(&self.dir) {
            for entry_result in read_dir.flatten() {
                let path = entry_result.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let expired = fs::read(&path)
                    .ok()
                    .and_then(|b| serde_json::from_slice::<CacheEntry>(&b).ok())
                    .map(|e| e.is_expired())
                    .unwrap_or(true); // remove unreadable files
                if expired {
                    let _ = fs::remove_file(path);
                }
            }
        }
    }

    /// Number of entries currently on disk (including potentially expired ones).
    pub fn len(&self) -> usize {
        fs::read_dir(&self.dir)
            .map(|d| {
                d.flatten()
                    .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
                    .count()
            })
            .unwrap_or(0)
    }

    /// Whether the disk cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Disk cache hit count since creation.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Disk cache miss count since creation.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Remove all entries from disk.
    pub fn clear(&self) {
        if let Ok(read_dir) = fs::read_dir(&self.dir) {
            for entry_result in read_dir.flatten() {
                let path = entry_result.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    let _ = fs::remove_file(path);
                }
            }
        }
    }

    fn entry_path(&self, key: &CacheKey) -> PathBuf {
        // Sanitise key for use as a filename
        let filename = key.0.replace(['/', '\\', ':'], "_");
        self.dir.join(format!("{}.json", filename))
    }
}

// ─── MultiLevelCache ──────────────────────────────────────────────────────────

/// Combined L1 (in-memory LRU) + L2 (disk) multi-level cache.
///
/// Reads check L1 first; on L1 miss, L2 is consulted and the result is
/// promoted back into L1.  Writes always populate both levels.
pub struct MultiLevelCache {
    l1: Arc<Mutex<MemoryCache>>,
    l2: Arc<Mutex<DiskCache>>,
    default_ttl: Duration,
}

impl MultiLevelCache {
    /// Create a multi-level cache with the given L1 capacity and default TTL.
    ///
    /// An ephemeral temporary directory is used for L2.
    pub fn new(l1_capacity: usize, default_ttl: Duration) -> io::Result<Self> {
        let l2 = DiskCache::open_temp()?;
        Ok(Self {
            l1: Arc::new(Mutex::new(MemoryCache::new(l1_capacity))),
            l2: Arc::new(Mutex::new(l2)),
            default_ttl,
        })
    }

    /// Create a multi-level cache with a specific directory for L2.
    pub fn with_dir(
        l1_capacity: usize,
        default_ttl: Duration,
        l2_dir: impl AsRef<Path>,
    ) -> io::Result<Self> {
        let l2 = DiskCache::open(l2_dir)?;
        Ok(Self {
            l1: Arc::new(Mutex::new(MemoryCache::new(l1_capacity))),
            l2: Arc::new(Mutex::new(l2)),
            default_ttl,
        })
    }

    /// Look up a key.  Promotes L2 hits to L1.
    pub fn get(&self, key: &CacheKey) -> Option<CacheEntry> {
        // L1 lookup
        {
            let mut l1 = self.l1.lock().expect("l1 lock poisoned");
            if let Some(entry) = l1.get(key) {
                return Some(entry.clone());
            }
        }
        // L2 lookup + promote to L1
        let mut l2 = self.l2.lock().expect("l2 lock poisoned");
        if let Some(entry) = l2.get(key) {
            // Promote to L1
            let mut l1 = self.l1.lock().expect("l1 lock poisoned");
            l1.put(entry.clone());
            return Some(entry);
        }
        None
    }

    /// Store an entry in both L1 and L2.
    pub fn put(&self, entry: CacheEntry) {
        let mut l1 = self.l1.lock().expect("l1 lock poisoned");
        l1.put(entry.clone());
        drop(l1);
        let mut l2 = self.l2.lock().expect("l2 lock poisoned");
        let _ = l2.put(&entry); // disk errors are non-fatal
    }

    /// Store result bindings with the default TTL.
    pub fn put_result(
        &self,
        key: CacheKey,
        result_bindings: Vec<HashMap<String, String>>,
        contributing_endpoints: Vec<String>,
    ) {
        let entry = CacheEntry::new(
            key,
            result_bindings,
            contributing_endpoints,
            self.default_ttl,
        );
        self.put(entry);
    }

    /// Store result bindings with a specific TTL.
    pub fn put_result_with_ttl(
        &self,
        key: CacheKey,
        result_bindings: Vec<HashMap<String, String>>,
        contributing_endpoints: Vec<String>,
        ttl: Duration,
    ) {
        let entry = CacheEntry::new(key, result_bindings, contributing_endpoints, ttl);
        self.put(entry);
    }

    /// Remove an entry from both levels.
    pub fn remove(&self, key: &CacheKey) {
        let mut l1 = self.l1.lock().expect("l1 lock poisoned");
        l1.remove(key);
        drop(l1);
        let mut l2 = self.l2.lock().expect("l2 lock poisoned");
        l2.remove(key);
    }

    /// Invalidate all entries that used the given endpoint in both levels.
    pub fn invalidate_endpoint(&self, endpoint_id: &str) {
        let mut l1 = self.l1.lock().expect("l1 lock poisoned");
        l1.invalidate_endpoint(endpoint_id);
        drop(l1);
        let mut l2 = self.l2.lock().expect("l2 lock poisoned");
        l2.invalidate_endpoint(endpoint_id);
    }

    /// Sweep both levels and remove all expired entries.
    pub fn evict_expired(&self) {
        let mut l1 = self.l1.lock().expect("l1 lock poisoned");
        l1.evict_expired();
        drop(l1);
        let mut l2 = self.l2.lock().expect("l2 lock poisoned");
        l2.evict_expired();
    }

    /// Cache statistics snapshot.
    pub fn stats(&self) -> MultiLevelCacheStats {
        let l1 = self.l1.lock().expect("l1 lock poisoned");
        let l2 = self.l2.lock().expect("l2 lock poisoned");
        MultiLevelCacheStats {
            l1_entries: l1.len(),
            l1_hits: l1.hits(),
            l1_misses: l1.misses(),
            l1_hit_rate: l1.hit_rate(),
            l2_entries: l2.len(),
            l2_hits: l2.hits(),
            l2_misses: l2.misses(),
        }
    }

    /// L1 entry count.
    pub fn l1_len(&self) -> usize {
        self.l1.lock().expect("l1 lock poisoned").len()
    }

    /// L2 entry count.
    pub fn l2_len(&self) -> usize {
        self.l2.lock().expect("l2 lock poisoned").len()
    }
}

/// Statistics snapshot for a `MultiLevelCache`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLevelCacheStats {
    pub l1_entries: usize,
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l1_hit_rate: f64,
    pub l2_entries: usize,
    pub l2_hits: u64,
    pub l2_misses: u64,
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(key: &str, ttl_secs: u64, endpoints: Vec<&str>) -> CacheEntry {
        CacheEntry {
            key: CacheKey::from_str(key),
            result_bindings: vec![[("?s".to_string(), "http://example.org/Alice".to_string())]
                .into_iter()
                .collect()],
            contributing_endpoints: endpoints.into_iter().map(String::from).collect(),
            created_at_secs: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            ttl_secs,
            hit_count: 0,
        }
    }

    // ── CacheKey ──────────────────────────────────────────────────────────

    #[test]
    fn test_cache_key_from_str() {
        let k = CacheKey::from_str("hello");
        assert_eq!(k.as_str(), "hello");
        assert_eq!(format!("{}", k), "hello");
    }

    #[test]
    fn test_cache_key_from_query_deterministic() {
        let k1 = CacheKey::from_query(
            "SELECT * WHERE {}",
            vec!["ep-a".to_string(), "ep-b".to_string()],
        );
        let k2 = CacheKey::from_query(
            "SELECT * WHERE {}",
            vec!["ep-b".to_string(), "ep-a".to_string()],
        );
        // Endpoint ordering should not matter
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_for_different_queries() {
        let k1 = CacheKey::from_query("SELECT ?s WHERE { ?s a <A> }", vec![]);
        let k2 = CacheKey::from_query("SELECT ?s WHERE { ?s a <B> }", vec![]);
        assert_ne!(k1, k2);
    }

    // ── CacheEntry ────────────────────────────────────────────────────────

    #[test]
    fn test_cache_entry_not_expired() {
        let e = sample_entry("k1", 3600, vec!["ep"]);
        assert!(!e.is_expired());
        assert!(e.remaining_ttl() > Duration::from_secs(3599));
    }

    #[test]
    fn test_cache_entry_expired() {
        let mut e = sample_entry("k2", 3600, vec!["ep"]);
        e.created_at_secs = 0; // epoch start → definitely expired
        assert!(e.is_expired());
        assert_eq!(e.remaining_ttl(), Duration::ZERO);
    }

    // ── MemoryCache (L1) ──────────────────────────────────────────────────

    #[test]
    fn test_memory_cache_basic_put_get() {
        let mut cache = MemoryCache::new(10);
        let e = sample_entry("q1", 3600, vec!["ep"]);
        cache.put(e.clone());
        let got = cache.get(&CacheKey::from_str("q1"));
        assert!(got.is_some());
        assert_eq!(got.unwrap().key.as_str(), "q1");
    }

    #[test]
    fn test_memory_cache_miss() {
        let mut cache = MemoryCache::new(10);
        assert!(cache.get(&CacheKey::from_str("no-such")).is_none());
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_memory_cache_hit_count() {
        let mut cache = MemoryCache::new(10);
        cache.put(sample_entry("k", 3600, vec![]));
        cache.get(&CacheKey::from_str("k"));
        cache.get(&CacheKey::from_str("k"));
        assert_eq!(cache.hits(), 2);
    }

    #[test]
    fn test_memory_cache_eviction_on_capacity() {
        let mut cache = MemoryCache::new(2);
        cache.put(sample_entry("a", 3600, vec![]));
        cache.put(sample_entry("b", 3600, vec![]));
        // Access "a" to make it MRU
        cache.get(&CacheKey::from_str("a"));
        // "b" is now LRU
        cache.put(sample_entry("c", 3600, vec![]));
        // "b" should be evicted
        assert!(cache.get(&CacheKey::from_str("b")).is_none());
        assert!(cache.get(&CacheKey::from_str("a")).is_some());
        assert!(cache.get(&CacheKey::from_str("c")).is_some());
    }

    #[test]
    fn test_memory_cache_remove() {
        let mut cache = MemoryCache::new(10);
        cache.put(sample_entry("r", 3600, vec![]));
        cache.remove(&CacheKey::from_str("r"));
        assert!(cache.get(&CacheKey::from_str("r")).is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_memory_cache_invalidate_endpoint() {
        let mut cache = MemoryCache::new(10);
        cache.put(sample_entry("k1", 3600, vec!["ep-a", "ep-b"]));
        cache.put(sample_entry("k2", 3600, vec!["ep-b", "ep-c"]));
        cache.put(sample_entry("k3", 3600, vec!["ep-c"]));

        cache.invalidate_endpoint("ep-a");
        // k1 should be gone (contains ep-a), k2 and k3 remain
        assert!(cache.get(&CacheKey::from_str("k1")).is_none());
        assert!(cache.get(&CacheKey::from_str("k2")).is_some());
        assert!(cache.get(&CacheKey::from_str("k3")).is_some());
    }

    #[test]
    fn test_memory_cache_evict_expired() {
        let mut cache = MemoryCache::new(10);
        let mut expired = sample_entry("exp", 3600, vec![]);
        expired.created_at_secs = 0; // force expired
        cache.put(expired);
        cache.put(sample_entry("live", 3600, vec![]));
        cache.evict_expired();
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&CacheKey::from_str("live")).is_some());
    }

    #[test]
    fn test_memory_cache_hit_rate() {
        let mut cache = MemoryCache::new(10);
        cache.put(sample_entry("x", 3600, vec![]));
        cache.get(&CacheKey::from_str("x")); // hit
        cache.get(&CacheKey::from_str("y")); // miss
        let rate = cache.hit_rate();
        assert!((rate - 0.5).abs() < 1e-9);
    }

    // ── DiskCache (L2) ────────────────────────────────────────────────────

    #[test]
    fn test_disk_cache_open_temp() {
        let mut cache = DiskCache::open_temp().expect("open_temp should succeed");
        assert!(cache.is_empty());
        // write + read
        let e = sample_entry("disk-k1", 3600, vec!["ep"]);
        cache.put(&e).expect("put should succeed");
        let got = cache.get(&CacheKey::from_str("disk-k1"));
        assert!(got.is_some());
        assert_eq!(got.unwrap().key.as_str(), "disk-k1");
        cache.clear();
    }

    #[test]
    fn test_disk_cache_miss_for_expired() {
        let mut cache = DiskCache::open_temp().expect("open_temp should succeed");
        let mut e = sample_entry("old", 3600, vec![]);
        e.created_at_secs = 0; // expired
        cache.put(&e).expect("put should succeed");
        let got = cache.get(&CacheKey::from_str("old"));
        assert!(got.is_none());
        cache.clear();
    }

    #[test]
    fn test_disk_cache_remove() {
        let mut cache = DiskCache::open_temp().expect("open_temp should succeed");
        let e = sample_entry("rm", 3600, vec![]);
        cache.put(&e).expect("put should succeed");
        cache.remove(&CacheKey::from_str("rm"));
        assert!(cache.get(&CacheKey::from_str("rm")).is_none());
        cache.clear();
    }

    #[test]
    fn test_disk_cache_invalidate_endpoint() {
        let mut cache = DiskCache::open_temp().expect("open_temp should succeed");
        cache
            .put(&sample_entry("k1", 3600, vec!["ep-alpha"]))
            .expect("put should succeed");
        cache
            .put(&sample_entry("k2", 3600, vec!["ep-beta"]))
            .expect("put should succeed");
        cache.invalidate_endpoint("ep-alpha");
        assert!(cache.get(&CacheKey::from_str("k1")).is_none());
        assert!(cache.get(&CacheKey::from_str("k2")).is_some());
        cache.clear();
    }

    #[test]
    fn test_disk_cache_evict_expired() {
        let mut cache = DiskCache::open_temp().expect("open_temp should succeed");
        let mut expired = sample_entry("exp2", 3600, vec![]);
        expired.created_at_secs = 0;
        cache.put(&expired).expect("put should succeed");
        cache
            .put(&sample_entry("live2", 3600, vec![]))
            .expect("put should succeed");
        cache.evict_expired();
        assert_eq!(cache.len(), 1);
        cache.clear();
    }

    // ── MultiLevelCache ───────────────────────────────────────────────────

    #[test]
    fn test_multilevel_basic_put_get() {
        let cache = MultiLevelCache::new(10, Duration::from_secs(3600)).expect("create cache");
        cache.put_result(CacheKey::from_str("ml-k1"), vec![], vec!["ep".to_string()]);
        let got = cache.get(&CacheKey::from_str("ml-k1"));
        assert!(got.is_some());
    }

    #[test]
    fn test_multilevel_miss() {
        let cache = MultiLevelCache::new(10, Duration::from_secs(3600)).expect("create cache");
        assert!(cache.get(&CacheKey::from_str("not-there")).is_none());
    }

    #[test]
    fn test_multilevel_l2_promotion_on_l1_miss() {
        // Write directly to L2, L1 miss → should promote to L1
        let cache = MultiLevelCache::new(2, Duration::from_secs(3600)).expect("create cache");
        // Fill L1 to capacity with two entries
        cache.put_result(CacheKey::from_str("a"), vec![], vec![]);
        cache.put_result(CacheKey::from_str("b"), vec![], vec![]);

        // Manually put a third entry to L2 only
        let entry = CacheEntry::new(
            CacheKey::from_str("l2-only"),
            vec![],
            vec!["ep-x".to_string()],
            Duration::from_secs(3600),
        );
        {
            let mut l2 = cache.l2.lock().expect("l2 lock");
            l2.put(&entry).expect("l2 put");
        }
        // Now access "l2-only" – L1 should miss, L2 should hit and promote
        let got = cache.get(&CacheKey::from_str("l2-only"));
        assert!(got.is_some(), "L2 hit should return entry");
    }

    #[test]
    fn test_multilevel_remove_from_both_levels() {
        let cache = MultiLevelCache::new(10, Duration::from_secs(3600)).expect("create cache");
        cache.put_result(CacheKey::from_str("del"), vec![], vec![]);
        cache.remove(&CacheKey::from_str("del"));
        assert!(cache.get(&CacheKey::from_str("del")).is_none());
    }

    #[test]
    fn test_multilevel_invalidate_endpoint() {
        let cache = MultiLevelCache::new(10, Duration::from_secs(3600)).expect("create cache");
        cache.put_result(CacheKey::from_str("i1"), vec![], vec!["ep-z".to_string()]);
        cache.put_result(CacheKey::from_str("i2"), vec![], vec!["ep-q".to_string()]);
        cache.invalidate_endpoint("ep-z");
        assert!(cache.get(&CacheKey::from_str("i1")).is_none());
        assert!(cache.get(&CacheKey::from_str("i2")).is_some());
    }

    #[test]
    fn test_multilevel_evict_expired() {
        let cache = MultiLevelCache::new(10, Duration::from_secs(3600)).expect("create cache");
        let mut e = CacheEntry::new(
            CacheKey::from_str("exp3"),
            vec![],
            vec![],
            Duration::from_secs(3600),
        );
        e.created_at_secs = 0; // force expired
        cache.put(e);
        cache.put_result(CacheKey::from_str("live3"), vec![], vec![]);
        cache.evict_expired();
        assert_eq!(cache.l1_len(), 1);
    }

    #[test]
    fn test_multilevel_stats() {
        let cache = MultiLevelCache::new(10, Duration::from_secs(3600)).expect("create cache");
        cache.put_result(CacheKey::from_str("s1"), vec![], vec![]);
        cache.get(&CacheKey::from_str("s1")); // hit
        cache.get(&CacheKey::from_str("missing")); // miss
        let stats = cache.stats();
        assert_eq!(stats.l1_hits, 1);
        assert_eq!(stats.l1_misses, 1);
        assert_eq!(stats.l1_entries, 1);
    }

    #[test]
    fn test_multilevel_put_with_custom_ttl() {
        let cache = MultiLevelCache::new(10, Duration::from_secs(3600)).expect("create cache");
        cache.put_result_with_ttl(
            CacheKey::from_str("ttl"),
            vec![],
            vec![],
            Duration::from_secs(60),
        );
        let got = cache.get(&CacheKey::from_str("ttl"));
        assert!(got.is_some());
        assert_eq!(got.unwrap().ttl_secs, 60);
    }
}
