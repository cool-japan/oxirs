//! DataLoader pattern for GraphQL N+1 resolution.
//!
//! Implements the classic DataLoader pattern that batches individual key lookups
//! into a single batch call, dramatically reducing N+1 query problems in GraphQL
//! resolvers. Includes an LRU-style per-request cache.

use std::collections::HashMap;
use std::sync::Arc;

/// Errors that can occur during batch loading.
#[derive(Debug)]
pub enum BatchError {
    LoadFailed(String),
    KeyNotFound(String),
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::LoadFailed(msg) => write!(f, "Batch load failed: {}", msg),
            BatchError::KeyNotFound(key) => write!(f, "Key not found: {}", key),
        }
    }
}

impl std::error::Error for BatchError {}

/// A trait for types that can batch-load items by string key.
pub trait BatchLoader: Send + Sync {
    type Item: Clone + Send;

    /// Load multiple items at once. Returns a map from key to item.
    /// Keys not present in the returned map are treated as missing.
    fn load_batch(&self, keys: &[String]) -> Result<HashMap<String, Self::Item>, BatchError>;
}

/// A DataLoader that batches and caches item lookups.
///
/// Items are queued individually via `queue()`, then fetched all at once
/// with `resolve_all()`. Subsequent lookups of cached keys skip the loader.
pub struct BatchResolver<T: Clone + Send> {
    loader: Arc<dyn BatchLoader<Item = T>>,
    max_batch_size: usize,
    cache: HashMap<String, T>,
    queue: Vec<String>,
    batches_executed: usize,
}

impl<T: Clone + Send + 'static> BatchResolver<T> {
    /// Create a new `BatchResolver` with the given loader and maximum batch size.
    ///
    /// `max_batch_size` controls how many keys are sent per `load_batch` call.
    /// A value of 0 is treated as unlimited (all queued keys in one batch).
    pub fn new(loader: Arc<dyn BatchLoader<Item = T>>, max_batch_size: usize) -> Self {
        BatchResolver {
            loader,
            max_batch_size,
            cache: HashMap::new(),
            queue: Vec::new(),
            batches_executed: 0,
        }
    }

    /// Add a key to the pending queue.
    ///
    /// Duplicates are ignored: a key already in the cache or already queued
    /// will not be queued again.
    pub fn queue(&mut self, key: impl Into<String>) {
        let key = key.into();
        if !self.cache.contains_key(&key) && !self.queue.contains(&key) {
            self.queue.push(key);
        }
    }

    /// Execute all pending batches, populating the internal cache.
    ///
    /// Keys are split into batches of at most `max_batch_size` and passed to
    /// the loader. After this call `pending_count()` returns 0.
    pub fn resolve_all(&mut self) -> Result<(), BatchError> {
        if self.queue.is_empty() {
            return Ok(());
        }

        let keys: Vec<String> = self.queue.drain(..).collect();

        // Split into batches
        let batch_size = if self.max_batch_size == 0 {
            keys.len()
        } else {
            self.max_batch_size
        };

        for chunk in keys.chunks(batch_size) {
            let loaded = self.loader.load_batch(chunk)?;
            for (k, v) in loaded {
                self.cache.insert(k, v);
            }
            self.batches_executed += 1;
        }

        Ok(())
    }

    /// Retrieve a cached item by key.
    ///
    /// Returns `None` if the key was never resolved or was missing from the loader.
    pub fn get(&self, key: &str) -> Option<&T> {
        self.cache.get(key)
    }

    /// Number of items in the cache.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the internal cache (does not clear the queue).
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Number of keys currently pending resolution.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Total number of `load_batch` calls made so far.
    pub fn batches_executed(&self) -> usize {
        self.batches_executed
    }
}

// ── Concrete test loader ──────────────────────────────────────────────────────

/// A test loader that uppercases keys.
///
/// - If a key starts with `"error:"` the whole batch fails.
/// - If a key starts with `"missing:"` the key is omitted from the result.
pub struct UppercaseLoader;

impl BatchLoader for UppercaseLoader {
    type Item = String;

    fn load_batch(&self, keys: &[String]) -> Result<HashMap<String, String>, BatchError> {
        for key in keys {
            if key.starts_with("error:") {
                return Err(BatchError::LoadFailed(key.clone()));
            }
        }
        let mut map = HashMap::new();
        for key in keys {
            if !key.starts_with("missing:") {
                map.insert(key.clone(), key.to_uppercase());
            }
        }
        Ok(map)
    }
}

/// A loader that counts how many times `load_batch` is called.
pub struct CountingLoader {
    pub call_count: std::sync::atomic::AtomicUsize,
}

impl Default for CountingLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl CountingLoader {
    pub fn new() -> Self {
        CountingLoader {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn calls(&self) -> usize {
        self.call_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl BatchLoader for CountingLoader {
    type Item = String;

    fn load_batch(&self, keys: &[String]) -> Result<HashMap<String, String>, BatchError> {
        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(keys.iter().map(|k| (k.clone(), k.to_uppercase())).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uppercase_resolver(max_batch: usize) -> BatchResolver<String> {
        BatchResolver::new(Arc::new(UppercaseLoader), max_batch)
    }

    // ── Single key ────────────────────────────────────────────────────────────

    #[test]
    fn test_single_key_resolve() {
        let mut r = uppercase_resolver(10);
        r.queue("hello");
        r.resolve_all().unwrap();
        assert_eq!(r.get("hello"), Some(&"HELLO".to_string()));
    }

    #[test]
    fn test_single_key_before_resolve_returns_none() {
        let r = uppercase_resolver(10);
        assert_eq!(r.get("anything"), None);
    }

    // ── Multiple keys ─────────────────────────────────────────────────────────

    #[test]
    fn test_multiple_keys_single_batch() {
        let mut r = uppercase_resolver(100);
        r.queue("a");
        r.queue("b");
        r.queue("c");
        r.resolve_all().unwrap();
        assert_eq!(r.get("a"), Some(&"A".to_string()));
        assert_eq!(r.get("b"), Some(&"B".to_string()));
        assert_eq!(r.get("c"), Some(&"C".to_string()));
    }

    // ── Batch splitting ───────────────────────────────────────────────────────

    #[test]
    fn test_batch_splitting_into_two() {
        let mut r = uppercase_resolver(2);
        r.queue("a");
        r.queue("b");
        r.queue("c");
        r.queue("d");
        r.resolve_all().unwrap();
        // 4 keys / 2 per batch = 2 batches
        assert_eq!(r.batches_executed(), 2);
        assert_eq!(r.get("a"), Some(&"A".to_string()));
        assert_eq!(r.get("d"), Some(&"D".to_string()));
    }

    #[test]
    fn test_batch_splitting_exact_multiple() {
        let mut r = uppercase_resolver(3);
        for k in ["x1", "x2", "x3", "x4", "x5", "x6"] {
            r.queue(k);
        }
        r.resolve_all().unwrap();
        assert_eq!(r.batches_executed(), 2);
    }

    #[test]
    fn test_batch_splitting_uneven() {
        let mut r = uppercase_resolver(3);
        for k in ["x1", "x2", "x3", "x4", "x5"] {
            r.queue(k);
        }
        r.resolve_all().unwrap();
        assert_eq!(r.batches_executed(), 2); // 3 + 2
    }

    // ── Cache hits ────────────────────────────────────────────────────────────

    #[test]
    fn test_cache_hit_avoids_re_load() {
        let loader = Arc::new(CountingLoader::new());
        let loader_dyn: Arc<dyn BatchLoader<Item = String>> =
            Arc::clone(&loader) as Arc<dyn BatchLoader<Item = String>>;
        let mut r: BatchResolver<String> = BatchResolver::new(loader_dyn, 10);

        r.queue("key");
        r.resolve_all().unwrap();
        assert_eq!(loader.calls(), 1);

        // Queue same key again — should be skipped (already cached)
        r.queue("key");
        r.resolve_all().unwrap();
        assert_eq!(loader.calls(), 1); // no new call
    }

    #[test]
    fn test_cache_size_after_resolve() {
        let mut r = uppercase_resolver(10);
        r.queue("a");
        r.queue("b");
        r.resolve_all().unwrap();
        assert_eq!(r.cache_size(), 2);
    }

    // ── clear_cache ───────────────────────────────────────────────────────────

    #[test]
    fn test_clear_cache() {
        let mut r = uppercase_resolver(10);
        r.queue("a");
        r.resolve_all().unwrap();
        assert_eq!(r.cache_size(), 1);
        r.clear_cache();
        assert_eq!(r.cache_size(), 0);
        assert_eq!(r.get("a"), None);
    }

    #[test]
    fn test_clear_cache_allows_reload() {
        let loader = Arc::new(CountingLoader::new());
        let loader_dyn: Arc<dyn BatchLoader<Item = String>> =
            Arc::clone(&loader) as Arc<dyn BatchLoader<Item = String>>;
        let mut r: BatchResolver<String> = BatchResolver::new(loader_dyn, 10);

        r.queue("key");
        r.resolve_all().unwrap();
        assert_eq!(loader.calls(), 1);

        r.clear_cache();

        r.queue("key");
        r.resolve_all().unwrap();
        assert_eq!(loader.calls(), 2); // reloaded after cache clear
    }

    // ── pending_count ─────────────────────────────────────────────────────────

    #[test]
    fn test_pending_count_before_resolve() {
        let mut r = uppercase_resolver(10);
        assert_eq!(r.pending_count(), 0);
        r.queue("a");
        r.queue("b");
        assert_eq!(r.pending_count(), 2);
    }

    #[test]
    fn test_pending_count_after_resolve() {
        let mut r = uppercase_resolver(10);
        r.queue("a");
        r.resolve_all().unwrap();
        assert_eq!(r.pending_count(), 0);
    }

    // ── batches_executed ──────────────────────────────────────────────────────

    #[test]
    fn test_batches_executed_zero_initially() {
        let r = uppercase_resolver(10);
        assert_eq!(r.batches_executed(), 0);
    }

    #[test]
    fn test_batches_executed_increments() {
        let mut r = uppercase_resolver(1);
        r.queue("a");
        r.queue("b");
        r.resolve_all().unwrap();
        assert_eq!(r.batches_executed(), 2);
    }

    #[test]
    fn test_batches_executed_accumulates() {
        let mut r = uppercase_resolver(10);
        r.queue("a");
        r.resolve_all().unwrap();
        r.queue("b");
        r.resolve_all().unwrap();
        assert_eq!(r.batches_executed(), 2);
    }

    // ── Error propagation ─────────────────────────────────────────────────────

    #[test]
    fn test_error_propagation() {
        let mut r = uppercase_resolver(10);
        r.queue("error:bad");
        let result = r.resolve_all();
        assert!(result.is_err());
        match result.unwrap_err() {
            BatchError::LoadFailed(msg) => assert!(msg.contains("error:bad")),
            e => panic!("wrong error: {:?}", e),
        }
    }

    #[test]
    fn test_error_in_second_batch() {
        let mut r = uppercase_resolver(2);
        r.queue("ok1");
        r.queue("ok2");
        r.queue("error:fail");
        let result = r.resolve_all();
        assert!(result.is_err());
    }

    // ── Deduplication ─────────────────────────────────────────────────────────

    #[test]
    fn test_dedup_queued_keys() {
        let mut r = uppercase_resolver(10);
        r.queue("a");
        r.queue("a");
        r.queue("a");
        assert_eq!(r.pending_count(), 1);
    }

    #[test]
    fn test_dedup_does_not_re_queue_cached() {
        let mut r = uppercase_resolver(10);
        r.queue("a");
        r.resolve_all().unwrap();
        r.queue("a"); // already cached
        assert_eq!(r.pending_count(), 0);
    }

    // ── Missing keys ──────────────────────────────────────────────────────────

    #[test]
    fn test_missing_key_returns_none() {
        let mut r = uppercase_resolver(10);
        r.queue("missing:nothere");
        r.resolve_all().unwrap();
        assert_eq!(r.get("missing:nothere"), None);
    }

    #[test]
    fn test_missing_key_does_not_affect_cache_size() {
        let mut r = uppercase_resolver(10);
        r.queue("ok");
        r.queue("missing:nope");
        r.resolve_all().unwrap();
        assert_eq!(r.cache_size(), 1);
    }

    // ── resolve_all with empty queue ──────────────────────────────────────────

    #[test]
    fn test_resolve_all_empty_queue_ok() {
        let mut r = uppercase_resolver(10);
        let result = r.resolve_all();
        assert!(result.is_ok());
        assert_eq!(r.batches_executed(), 0);
    }

    // ── max_batch_size = 0 (unlimited) ────────────────────────────────────────

    #[test]
    fn test_unlimited_batch_size() {
        let mut r = uppercase_resolver(0);
        for i in 0..100 {
            r.queue(format!("key{}", i));
        }
        r.resolve_all().unwrap();
        // All in one batch
        assert_eq!(r.batches_executed(), 1);
        assert_eq!(r.cache_size(), 100);
    }

    // ── Correct values returned ───────────────────────────────────────────────

    #[test]
    fn test_correct_uppercase_values() {
        let mut r = uppercase_resolver(10);
        r.queue("foo");
        r.queue("bar");
        r.queue("baz");
        r.resolve_all().unwrap();
        assert_eq!(r.get("foo"), Some(&"FOO".to_string()));
        assert_eq!(r.get("bar"), Some(&"BAR".to_string()));
        assert_eq!(r.get("baz"), Some(&"BAZ".to_string()));
    }

    // ── Two resolvers share same loader ───────────────────────────────────────

    #[test]
    fn test_two_resolvers_independent_cache() {
        let loader1: Arc<dyn BatchLoader<Item = String>> = Arc::new(UppercaseLoader);
        let loader2: Arc<dyn BatchLoader<Item = String>> = Arc::new(UppercaseLoader);
        let mut r1: BatchResolver<String> = BatchResolver::new(loader1, 10);
        let mut r2: BatchResolver<String> = BatchResolver::new(loader2, 10);

        r1.queue("hello");
        r1.resolve_all().unwrap();

        // r2 has its own independent cache
        assert_eq!(r2.get("hello"), None);
        r2.queue("hello");
        r2.resolve_all().unwrap();
        assert_eq!(r2.get("hello"), Some(&"HELLO".to_string()));
    }
}
