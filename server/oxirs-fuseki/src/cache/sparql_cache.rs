//! Intelligent SPARQL Query Result Cache
//!
//! Features:
//! - Query normalization (variable renaming, whitespace stripping)
//! - Dataset-level semantic invalidation on SPARQL UPDATE
//! - Per-dataset or global cache with LRU eviction
//! - Size-aware eviction (respects max_size_bytes)
//! - Cache warm-up support
//! - Named-graph level invalidation

use crate::error::{FusekiError, FusekiResult};
use lru::LruCache;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info};

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// Query type for cache strategy differentiation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparqlQueryType {
    Select,
    Ask,
    Construct,
    Describe,
}

impl SparqlQueryType {
    /// Parse the query type from a SPARQL query string
    pub fn detect(query: &str) -> Self {
        let upper = query.trim_start().to_ascii_uppercase();
        // Strip leading comments / PREFIX / BASE declarations
        let stripped = strip_prologue(&upper);
        if stripped.starts_with("ASK") {
            SparqlQueryType::Ask
        } else if stripped.starts_with("CONSTRUCT") {
            SparqlQueryType::Construct
        } else if stripped.starts_with("DESCRIBE") {
            SparqlQueryType::Describe
        } else {
            SparqlQueryType::Select
        }
    }
}

/// Normalized query key – stable identifier for a particular SPARQL query
/// over a particular dataset.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryCacheKey {
    /// Normalized SPARQL query string (whitespace-collapsed, variable-renamed)
    pub normalized_query: String,
    /// Dataset name
    pub dataset: String,
    /// Query type
    pub query_type: SparqlQueryType,
}

impl QueryCacheKey {
    /// Build a cache key from the raw SPARQL query string and dataset name.
    pub fn new(raw_query: &str, dataset: impl Into<String>) -> Self {
        let dataset = dataset.into();
        let normalized_query = SparqlQueryCache::normalize_query(raw_query);
        let query_type = SparqlQueryType::detect(raw_query);
        QueryCacheKey {
            normalized_query,
            dataset,
            query_type,
        }
    }
}

/// A single cached query result entry.
#[derive(Debug, Clone)]
pub struct CachedQueryResult {
    /// Cache key (for reference in eviction logic)
    pub key: QueryCacheKey,
    /// Serialized result (e.g. SPARQL JSON)
    pub result_json: String,
    /// Content-Type header value for the cached response
    pub content_type: String,
    /// When this entry was created
    pub created_at: Instant,
    /// When this entry was last accessed
    pub last_accessed: Instant,
    /// Time-to-live
    pub ttl: Duration,
    /// Number of cache hits for this entry
    pub hit_count: u64,
    /// Byte size of result_json
    pub size_bytes: usize,
    /// Named graph URIs accessed during the query (used for fine-grained invalidation)
    pub accessed_graphs: Vec<String>,
}

impl CachedQueryResult {
    /// Returns `true` if the TTL has been exceeded.
    pub fn is_stale(&self) -> bool {
        self.created_at.elapsed() >= self.ttl
    }
}

/// Cache-level statistics snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_bytes: usize,
    pub hit_rate: f64,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub invalidation_count: u64,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal state
// ──────────────────────────────────────────────────────────────────────────────

struct CacheInner {
    /// LRU-ordered entries
    lru: LruCache<QueryCacheKey, CachedQueryResult>,
    /// Reverse mapping: dataset → set of cache keys (for dataset invalidation)
    dataset_index: HashMap<String, HashSet<QueryCacheKey>>,
    /// Reverse mapping: graph URI → set of cache keys (for graph invalidation)
    graph_index: HashMap<String, HashSet<QueryCacheKey>>,
    /// Running total of result bytes held in the cache
    current_bytes: usize,
    /// Number of entries evicted due to size pressure
    eviction_count: u64,
    /// Number of entries removed via invalidation
    invalidation_count: u64,
}

impl CacheInner {
    fn new(capacity: usize) -> FusekiResult<Self> {
        let cap = NonZeroUsize::new(capacity).ok_or_else(|| FusekiError::Configuration {
            message: "Cache capacity must be > 0".to_string(),
        })?;
        Ok(CacheInner {
            lru: LruCache::new(cap),
            dataset_index: HashMap::new(),
            graph_index: HashMap::new(),
            current_bytes: 0,
            eviction_count: 0,
            invalidation_count: 0,
        })
    }

    /// Remove one entry (by key) from all secondary indices and the LRU, updating
    /// the byte counter.  Returns the removed entry if it existed.
    fn remove_entry(&mut self, key: &QueryCacheKey) -> Option<CachedQueryResult> {
        let removed = self.lru.pop(key)?;
        self.current_bytes = self.current_bytes.saturating_sub(removed.size_bytes);
        // Clean up dataset index
        if let Some(set) = self.dataset_index.get_mut(&removed.key.dataset) {
            set.remove(&removed.key);
        }
        // Clean up graph index
        for graph in &removed.accessed_graphs {
            if let Some(set) = self.graph_index.get_mut(graph) {
                set.remove(&removed.key);
            }
        }
        Some(removed)
    }

    /// Insert a new entry, updating all secondary indices and the byte counter.
    fn insert_entry(&mut self, key: QueryCacheKey, entry: CachedQueryResult) {
        self.current_bytes += entry.size_bytes;
        // Register in dataset index
        self.dataset_index
            .entry(key.dataset.clone())
            .or_default()
            .insert(key.clone());
        // Register in graph index
        for graph in &entry.accessed_graphs {
            self.graph_index
                .entry(graph.clone())
                .or_default()
                .insert(key.clone());
        }
        // LRU insert (may evict LRU entry)
        if let Some((evicted_key, evicted_val)) = self.lru.push(key, entry) {
            self.current_bytes = self.current_bytes.saturating_sub(evicted_val.size_bytes);
            // Clean up indices for the evicted entry
            if let Some(set) = self.dataset_index.get_mut(&evicted_key.dataset) {
                set.remove(&evicted_key);
            }
            for graph in &evicted_val.accessed_graphs {
                if let Some(set) = self.graph_index.get_mut(graph) {
                    set.remove(&evicted_key);
                }
            }
            self.eviction_count += 1;
        }
    }

    /// Evict LRU entries until `current_bytes < max_bytes`.
    fn evict_by_size(&mut self, max_bytes: usize) {
        while self.current_bytes > max_bytes {
            if let Some((evicted_key, evicted_val)) = self.lru.pop_lru() {
                self.current_bytes = self.current_bytes.saturating_sub(evicted_val.size_bytes);
                if let Some(set) = self.dataset_index.get_mut(&evicted_key.dataset) {
                    set.remove(&evicted_key);
                }
                for graph in &evicted_val.accessed_graphs {
                    if let Some(set) = self.graph_index.get_mut(graph) {
                        set.remove(&evicted_key);
                    }
                }
                self.eviction_count += 1;
            } else {
                break; // Cache is empty
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Intelligent SPARQL query result cache with semantic invalidation.
pub struct SparqlQueryCache {
    inner: Arc<Mutex<CacheInner>>,
    max_size_bytes: usize,
    default_ttl: Duration,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl SparqlQueryCache {
    /// Create a new cache.
    ///
    /// * `capacity`       – maximum number of entries (LRU eviction after this)
    /// * `max_size_bytes` – evict by size when total held bytes exceeds this
    /// * `default_ttl`    – time-to-live for each entry
    pub fn new(
        capacity: usize,
        max_size_bytes: usize,
        default_ttl: Duration,
    ) -> FusekiResult<Self> {
        Ok(SparqlQueryCache {
            inner: Arc::new(Mutex::new(CacheInner::new(capacity)?)),
            max_size_bytes,
            default_ttl,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Look up a query result.  Returns `None` on miss or stale entry.
    pub fn get(&self, key: &QueryCacheKey) -> Option<CachedQueryResult> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| {
                tracing::error!("Cache lock poisoned on get: {}", e);
                e
            })
            .ok()?;

        // Peek first to check staleness without promoting in LRU
        if let Some(entry) = inner.lru.peek(key) {
            if entry.is_stale() {
                let key_clone = key.clone();
                inner.remove_entry(&key_clone);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }

        // Now promote (LRU update) and clone
        if let Some(entry) = inner.lru.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.hit_count += 1;
            let cloned = entry.clone();
            self.hits.fetch_add(1, Ordering::Relaxed);
            debug!(
                query_type = ?cloned.key.query_type,
                dataset = %cloned.key.dataset,
                "Cache hit"
            );
            Some(cloned)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Store a query result.
    ///
    /// * `key`          – cache key (use `QueryCacheKey::new`)
    /// * `result`       – serialized response body (e.g. SPARQL JSON)
    /// * `content_type` – MIME type of the response
    /// * `graphs`       – named graph URIs accessed during the query
    ///
    /// Returns `true` if the entry was inserted, `false` if it was too large to cache.
    pub fn put(
        &self,
        key: QueryCacheKey,
        result: String,
        content_type: &str,
        graphs: Vec<String>,
    ) -> bool {
        let size = result.len();
        if size > self.max_size_bytes {
            debug!(size, "Result too large to cache; skipping");
            return false;
        }

        let now = Instant::now();
        let entry = CachedQueryResult {
            key: key.clone(),
            size_bytes: size,
            result_json: result,
            content_type: content_type.to_string(),
            created_at: now,
            last_accessed: now,
            ttl: self.default_ttl,
            hit_count: 0,
            accessed_graphs: graphs,
        };

        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("Cache lock poisoned on put: {}", e);
                return false;
            }
        };

        inner.insert_entry(key, entry);
        inner.evict_by_size(self.max_size_bytes);
        true
    }

    /// Invalidate ALL cached entries belonging to the given dataset.
    ///
    /// Called when a SPARQL UPDATE targets a dataset (conservative invalidation).
    /// Returns the number of invalidated entries.
    pub fn invalidate_dataset(&self, dataset: &str) -> usize {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("Cache lock poisoned on invalidate_dataset: {}", e);
                return 0;
            }
        };

        let keys: Vec<QueryCacheKey> = inner
            .dataset_index
            .get(dataset)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        let count = keys.len();
        for key in &keys {
            inner.remove_entry(key);
            inner.invalidation_count += 1;
        }
        inner.dataset_index.remove(dataset);

        info!(dataset, count, "Invalidated cache entries for dataset");
        count
    }

    /// Invalidate entries that accessed a specific named graph URI.
    ///
    /// Finer-grained than `invalidate_dataset`; useful when only one named
    /// graph within a dataset was modified.
    /// Returns the number of invalidated entries.
    pub fn invalidate_graph(&self, graph_uri: &str) -> usize {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("Cache lock poisoned on invalidate_graph: {}", e);
                return 0;
            }
        };

        let keys: Vec<QueryCacheKey> = inner
            .graph_index
            .get(graph_uri)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        let count = keys.len();
        for key in &keys {
            inner.remove_entry(key);
            inner.invalidation_count += 1;
        }
        inner.graph_index.remove(graph_uri);

        info!(graph_uri, count, "Invalidated cache entries for graph");
        count
    }

    /// Normalize a SPARQL query string for stable cache-key generation.
    ///
    /// Steps:
    /// 1. Collapse all whitespace runs to a single space
    /// 2. Trim leading / trailing whitespace
    /// 3. Rename all variable names to positional placeholders (`?_v0`, `?_v1`, …)
    ///    so that `SELECT ?s WHERE { ?s ?p ?o }` and
    ///    `SELECT ?x WHERE { ?x ?y ?z }` map to the same cache key.
    pub fn normalize_query(query: &str) -> String {
        // Step 1: collapse whitespace
        let collapsed: String = query.split_whitespace().collect::<Vec<_>>().join(" ");

        // Step 2: rename variables
        let normalized = rename_variables(&collapsed);
        normalized
    }

    /// Return the hit rate (0.0 – 1.0).
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Number of entries currently held.
    pub fn size(&self) -> usize {
        match self.inner.lock() {
            Ok(inner) => inner.lru.len(),
            Err(_) => 0,
        }
    }

    /// Total bytes of cached results currently held.
    pub fn total_bytes(&self) -> usize {
        match self.inner.lock() {
            Ok(inner) => inner.current_bytes,
            Err(_) => 0,
        }
    }

    /// Snapshot of cache statistics for monitoring endpoints.
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            total_entries: inner.lru.len(),
            total_bytes: inner.current_bytes,
            hit_rate: if total == 0 {
                0.0
            } else {
                hits as f64 / total as f64
            },
            hit_count: hits,
            miss_count: misses,
            eviction_count: inner.eviction_count,
            invalidation_count: inner.invalidation_count,
        }
    }

    /// Pre-warm the cache by inserting a batch of query results.
    ///
    /// Useful for startup warm-up from persistent storage.
    pub fn warm_up(&self, entries: Vec<(QueryCacheKey, String, String, Vec<String>)>) -> usize {
        let mut inserted = 0usize;
        for (key, result, content_type, graphs) in entries {
            if self.put(key, result, &content_type, graphs) {
                inserted += 1;
            }
        }
        info!(inserted, "Cache warm-up complete");
        inserted
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Strip SPARQL prologue (PREFIX / BASE declarations) to reach the first keyword.
fn strip_prologue(query: &str) -> &str {
    let mut rest = query.trim();
    loop {
        let upper = rest.trim_start();
        if upper.starts_with("PREFIX") || upper.starts_with("BASE") {
            // Skip to end of line
            if let Some(nl) = upper.find('\n') {
                rest = &upper[nl + 1..];
            } else if let Some(pos) = upper.find('>') {
                rest = &upper[pos + 1..];
            } else {
                break;
            }
        } else {
            return upper;
        }
    }
    rest.trim()
}

/// Rename all SPARQL query variables (`?name` / `$name`) to canonical
/// positional names (`?_v0`, `?_v1`, …) so that structurally equivalent
/// queries with different variable names produce the same cache key.
fn rename_variables(query: &str) -> String {
    let mut result = String::with_capacity(query.len());
    let mut var_map: HashMap<&str, String> = HashMap::new();
    let mut counter: usize = 0;

    let bytes = query.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Check for string literals – don't rename variables inside them
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            let delim = bytes[i];
            result.push(bytes[i] as char);
            i += 1;
            while i < bytes.len() && bytes[i] != delim {
                result.push(bytes[i] as char);
                i += 1;
            }
            if i < bytes.len() {
                result.push(bytes[i] as char);
                i += 1;
            }
            continue;
        }

        // Check for IRI references
        if bytes[i] == b'<' {
            result.push('<');
            i += 1;
            while i < bytes.len() && bytes[i] != b'>' {
                result.push(bytes[i] as char);
                i += 1;
            }
            if i < bytes.len() {
                result.push('>');
                i += 1;
            }
            continue;
        }

        // Check for variable markers
        if bytes[i] == b'?' || bytes[i] == b'$' {
            i += 1; // skip marker
            let start = i;
            // Collect the variable name (alphanumeric + underscore)
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            if i > start {
                let var_name = &query[start..i];
                let canonical = var_map.entry(var_name).or_insert_with(|| {
                    let name = format!("?_v{}", counter);
                    counter += 1;
                    name
                });
                result.push_str(canonical);
            } else {
                // Lone `?` or `$` – pass through
                result.push(bytes[i.saturating_sub(1)] as char);
            }
            continue;
        }

        result.push(bytes[i] as char);
        i += 1;
    }

    result
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_cache() -> SparqlQueryCache {
        SparqlQueryCache::new(100, 10 * 1024 * 1024, Duration::from_secs(60)).unwrap()
    }

    #[test]
    fn test_cache_hit() {
        let cache = make_cache();
        let key = QueryCacheKey::new("SELECT * WHERE { ?s ?p ?o }", "default");

        assert!(cache.get(&key).is_none());
        cache.put(
            key.clone(),
            r#"{"results":[]}"#.to_string(),
            "application/json",
            vec![],
        );
        let hit = cache.get(&key);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().result_json, r#"{"results":[]}"#);
    }

    #[test]
    fn test_cache_miss() {
        let cache = make_cache();
        let key = QueryCacheKey::new("SELECT * WHERE { ?s ?p ?o }", "default");
        assert!(cache.get(&key).is_none());
        assert!((cache.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dataset_invalidation() {
        let cache = make_cache();
        let key1 = QueryCacheKey::new("SELECT * WHERE { ?s ?p ?o }", "ds1");
        let key2 = QueryCacheKey::new("SELECT ?s WHERE { ?s a <C> }", "ds1");
        let key3 = QueryCacheKey::new("SELECT * WHERE { ?s ?p ?o }", "ds2");

        cache.put(key1.clone(), "r1".to_string(), "application/json", vec![]);
        cache.put(key2.clone(), "r2".to_string(), "application/json", vec![]);
        cache.put(key3.clone(), "r3".to_string(), "application/json", vec![]);

        let count = cache.invalidate_dataset("ds1");
        assert_eq!(count, 2);

        assert!(cache.get(&key1).is_none(), "ds1 key1 should be invalidated");
        assert!(cache.get(&key2).is_none(), "ds1 key2 should be invalidated");
        assert!(cache.get(&key3).is_some(), "ds2 key3 should be untouched");
    }

    #[test]
    fn test_graph_invalidation() {
        let cache = make_cache();
        let graph_a = "http://example.org/graphA".to_string();
        let graph_b = "http://example.org/graphB".to_string();

        let key1 = QueryCacheKey::new(
            "SELECT ?s WHERE { GRAPH <http://example.org/graphA> { ?s ?p ?o } }",
            "ds",
        );
        let key2 = QueryCacheKey::new(
            "SELECT ?s WHERE { GRAPH <http://example.org/graphB> { ?s ?p ?o } }",
            "ds",
        );

        cache.put(
            key1.clone(),
            "r1".to_string(),
            "application/json",
            vec![graph_a.clone()],
        );
        cache.put(
            key2.clone(),
            "r2".to_string(),
            "application/json",
            vec![graph_b.clone()],
        );

        let count = cache.invalidate_graph(&graph_a);
        assert_eq!(count, 1);

        assert!(cache.get(&key1).is_none(), "key1 should be invalidated");
        assert!(cache.get(&key2).is_some(), "key2 should be untouched");
    }

    #[test]
    fn test_query_normalization_whitespace() {
        let q1 = "SELECT * WHERE { ?s ?p ?o }";
        let q2 = "SELECT   *   WHERE   {   ?s   ?p   ?o   }";
        // After normalization the variable names differ (?s→?_v0 etc.) but the
        // structure should be identical, yielding the same normalized string.
        assert_eq!(
            SparqlQueryCache::normalize_query(q1),
            SparqlQueryCache::normalize_query(q2)
        );
    }

    #[test]
    fn test_query_normalization_variable_rename() {
        let q1 = "SELECT ?s ?p WHERE { ?s ?p ?o }";
        let q2 = "SELECT ?x ?y WHERE { ?x ?y ?z }";
        assert_eq!(
            SparqlQueryCache::normalize_query(q1),
            SparqlQueryCache::normalize_query(q2),
            "Structurally equivalent queries with different var names must normalize the same"
        );
    }

    #[test]
    fn test_hit_rate() {
        let cache = make_cache();
        let key = QueryCacheKey::new("SELECT * WHERE { ?s ?p ?o }", "default");
        cache.put(key.clone(), "{}".to_string(), "application/json", vec![]);

        cache.get(&key); // hit
        cache.get(&QueryCacheKey::new(
            "SELECT ?x WHERE { ?x a <C> }",
            "default",
        )); // miss

        let rate = cache.hit_rate();
        assert!(
            (rate - 0.5).abs() < f64::EPSILON,
            "hit rate should be 0.5, got {}",
            rate
        );
    }

    #[test]
    fn test_size_eviction() {
        // Create a cache with a very small byte limit
        let cache = SparqlQueryCache::new(1000, 20, Duration::from_secs(60)).unwrap();

        // Each result is > 10 bytes
        let key1 = QueryCacheKey::new("SELECT ?a WHERE { ?a ?b ?c }", "ds");
        let key2 = QueryCacheKey::new("SELECT ?x WHERE { ?x ?y ?z }", "ds");

        cache.put(
            key1.clone(),
            "0123456789abcde".to_string(),
            "application/json",
            vec![],
        );
        cache.put(
            key2.clone(),
            "0123456789ABCDE".to_string(),
            "application/json",
            vec![],
        );

        // Total bytes must be <= 20
        assert!(
            cache.total_bytes() <= 20,
            "size should be <= 20 after eviction, got {}",
            cache.total_bytes()
        );
    }

    #[test]
    fn test_query_type_detection() {
        assert_eq!(
            SparqlQueryType::detect("SELECT * WHERE { ?s ?p ?o }"),
            SparqlQueryType::Select
        );
        assert_eq!(
            SparqlQueryType::detect("ASK { ?s ?p ?o }"),
            SparqlQueryType::Ask
        );
        assert_eq!(
            SparqlQueryType::detect("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"),
            SparqlQueryType::Construct
        );
        assert_eq!(
            SparqlQueryType::detect("DESCRIBE ?s WHERE { ?s ?p ?o }"),
            SparqlQueryType::Describe
        );
    }

    #[test]
    fn test_stale_entry_eviction() {
        let cache = SparqlQueryCache::new(100, 10 * 1024 * 1024, Duration::from_millis(1)).unwrap();
        let key = QueryCacheKey::new("SELECT * WHERE { ?s ?p ?o }", "default");
        cache.put(key.clone(), "{}".to_string(), "application/json", vec![]);
        // Sleep longer than TTL
        std::thread::sleep(Duration::from_millis(10));
        assert!(
            cache.get(&key).is_none(),
            "Stale entry should not be returned"
        );
    }

    #[test]
    fn test_warm_up() {
        let cache = make_cache();
        let entries = vec![
            (
                QueryCacheKey::new("SELECT * WHERE { ?s ?p ?o }", "ds"),
                r#"{"results":{"bindings":[]}}"#.to_string(),
                "application/sparql-results+json".to_string(),
                vec![],
            ),
            (
                QueryCacheKey::new("ASK { ?s ?p ?o }", "ds"),
                r#"{"boolean":false}"#.to_string(),
                "application/sparql-results+json".to_string(),
                vec!["http://example.org/g".to_string()],
            ),
        ];
        let inserted = cache.warm_up(entries);
        assert_eq!(inserted, 2);
        assert_eq!(cache.size(), 2);
    }
}
