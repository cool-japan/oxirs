//! Field resolver result cache for GraphQL fields.
//!
//! Provides TTL-based caching with LRU eviction for GraphQL resolver results.

use std::collections::{HashMap, VecDeque};

/// Cache key for a resolver result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub type_name: String,
    pub field_name: String,
    pub args_hash: u64,
    pub parent_id: Option<String>,
}

impl CacheKey {
    /// Create a new cache key with a type and field name.
    pub fn new(type_name: impl Into<String>, field_name: impl Into<String>) -> Self {
        Self {
            type_name: type_name.into(),
            field_name: field_name.into(),
            args_hash: 0,
            parent_id: None,
        }
    }

    /// Set the args hash for this cache key.
    pub fn with_args(mut self, args_hash: u64) -> Self {
        self.args_hash = args_hash;
        self
    }

    /// Set the parent ID for this cache key.
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    /// Serialize the cache key to a string for use as a map key.
    pub fn cache_key_str(&self) -> String {
        match &self.parent_id {
            Some(parent) => format!(
                "{}:{}:{}:{}",
                self.type_name, self.field_name, self.args_hash, parent
            ),
            None => format!("{}:{}:{}", self.type_name, self.field_name, self.args_hash),
        }
    }
}

/// A cached entry with metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    pub value: T,
    pub created_at: u64,
    pub ttl_ms: u64,
    pub hit_count: u64,
}

impl<T> CacheEntry<T> {
    fn new(value: T, created_at: u64, ttl_ms: u64) -> Self {
        Self {
            value,
            created_at,
            ttl_ms,
            hit_count: 0,
        }
    }

    fn is_expired(&self, current_time_ms: u64) -> bool {
        current_time_ms.saturating_sub(self.created_at) > self.ttl_ms
    }
}

/// Configuration for the field resolver cache.
#[derive(Debug, Clone)]
pub struct ResolverCacheConfig {
    pub max_entries: usize,
    pub default_ttl_ms: u64,
    pub enabled: bool,
}

impl Default for ResolverCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            default_ttl_ms: 60_000,
            enabled: true,
        }
    }
}

/// LRU TTL cache for GraphQL field resolver results.
pub struct FieldResolverCache<T: Clone> {
    config: ResolverCacheConfig,
    entries: HashMap<String, CacheEntry<T>>,
    /// Tracks insertion order for LRU eviction (front = oldest, back = newest).
    access_order: VecDeque<String>,
    hits: u64,
    misses: u64,
}

impl<T: Clone> FieldResolverCache<T> {
    /// Create a new cache with the given configuration.
    pub fn new(config: ResolverCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a value in the cache. Returns `None` if not found or expired.
    pub fn get(&mut self, key: &CacheKey, current_time_ms: u64) -> Option<T> {
        if !self.config.enabled {
            self.misses += 1;
            return None;
        }

        let key_str = key.cache_key_str();

        let is_expired = self
            .entries
            .get(&key_str)
            .map(|e| e.is_expired(current_time_ms))
            .unwrap_or(true);

        if is_expired {
            if self.entries.contains_key(&key_str) {
                self.entries.remove(&key_str);
                self.access_order.retain(|k| k != &key_str);
            }
            self.misses += 1;
            return None;
        }

        // Move to back of access_order (most recently used)
        self.access_order.retain(|k| k != &key_str);
        self.access_order.push_back(key_str.clone());

        if let Some(entry) = self.entries.get_mut(&key_str) {
            entry.hit_count += 1;
            self.hits += 1;
            Some(entry.value.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a value into the cache with the default TTL.
    pub fn insert(&mut self, key: CacheKey, value: T, current_time_ms: u64) {
        if !self.config.enabled {
            return;
        }

        let key_str = key.cache_key_str();
        let ttl = self.config.default_ttl_ms;

        // Evict LRU entries if at capacity
        while self.entries.len() >= self.config.max_entries && !self.access_order.is_empty() {
            if let Some(oldest) = self.access_order.pop_front() {
                self.entries.remove(&oldest);
            }
        }

        // Remove from access_order if already present
        self.access_order.retain(|k| k != &key_str);
        self.access_order.push_back(key_str.clone());

        self.entries
            .insert(key_str, CacheEntry::new(value, current_time_ms, ttl));
    }

    /// Invalidate all entries for a given type and field name.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate(&mut self, type_name: &str, field_name: &str) -> usize {
        let prefix = format!("{type_name}:{field_name}:");
        let to_remove: Vec<String> = self
            .entries
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .cloned()
            .collect();

        let count = to_remove.len();
        for key in &to_remove {
            self.entries.remove(key);
            self.access_order.retain(|k| k != key);
        }

        count
    }

    /// Invalidate a specific cache entry.
    ///
    /// Returns `true` if the entry was found and removed.
    pub fn invalidate_key(&mut self, key: &CacheKey) -> bool {
        let key_str = key.cache_key_str();
        if self.entries.remove(&key_str).is_some() {
            self.access_order.retain(|k| k != &key_str);
            true
        } else {
            false
        }
    }

    /// Remove all expired entries.
    ///
    /// Returns the number of entries expired.
    pub fn expire(&mut self, current_time_ms: u64) -> usize {
        let to_remove: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(current_time_ms))
            .map(|(k, _)| k.clone())
            .collect();

        let count = to_remove.len();
        for key in &to_remove {
            self.entries.remove(key);
            self.access_order.retain(|k| k != key);
        }

        count
    }

    /// Clear all entries from the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }

    /// Get the number of entries in the cache.
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// Calculate the cache hit rate (hits / (hits + misses)).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Get total cache hits.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Get total cache misses.
    pub fn misses(&self) -> u64 {
        self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(max: usize, ttl: u64) -> FieldResolverCache<String> {
        FieldResolverCache::new(ResolverCacheConfig {
            max_entries: max,
            default_ttl_ms: ttl,
            enabled: true,
        })
    }

    fn key(t: &str, f: &str) -> CacheKey {
        CacheKey::new(t, f)
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache = make_cache(100, 60_000);
        let k = key("User", "name");
        cache.insert(k.clone(), "Alice".to_string(), 1000);
        let result = cache.get(&k, 2000);
        assert_eq!(result, Some("Alice".to_string()));
    }

    #[test]
    fn test_get_miss_returns_none() {
        let mut cache = make_cache(100, 60_000);
        let k = key("User", "name");
        let result = cache.get(&k, 1000);
        assert_eq!(result, None);
    }

    #[test]
    fn test_ttl_expiry() {
        let mut cache = make_cache(100, 1000);
        let k = key("User", "name");
        cache.insert(k.clone(), "Alice".to_string(), 0);
        // Before expiry
        assert!(cache.get(&k, 500).is_some());
        // After expiry
        assert!(cache.get(&k, 2000).is_none());
    }

    #[test]
    fn test_lru_eviction_at_max_entries() {
        let mut cache = make_cache(3, 60_000);
        let k1 = CacheKey::new("T", "f1");
        let k2 = CacheKey::new("T", "f2");
        let k3 = CacheKey::new("T", "f3");
        let k4 = CacheKey::new("T", "f4");

        cache.insert(k1.clone(), "v1".to_string(), 100);
        cache.insert(k2.clone(), "v2".to_string(), 200);
        cache.insert(k3.clone(), "v3".to_string(), 300);
        // k1 is LRU; inserting k4 should evict k1
        cache.insert(k4.clone(), "v4".to_string(), 400);

        assert_eq!(cache.size(), 3);
        // k1 should be evicted
        assert!(cache.get(&k1, 500).is_none());
        // k2, k3, k4 should remain
        assert!(cache.get(&k2, 500).is_some());
        assert!(cache.get(&k3, 500).is_some());
        assert!(cache.get(&k4, 500).is_some());
    }

    #[test]
    fn test_invalidate_by_type_and_field() {
        let mut cache = make_cache(100, 60_000);
        let k1 = CacheKey::new("User", "name").with_args(1);
        let k2 = CacheKey::new("User", "name").with_args(2);
        let k3 = CacheKey::new("User", "email").with_args(1);

        cache.insert(k1.clone(), "v1".to_string(), 100);
        cache.insert(k2.clone(), "v2".to_string(), 100);
        cache.insert(k3.clone(), "v3".to_string(), 100);

        let removed = cache.invalidate("User", "name");
        assert_eq!(removed, 2);
        assert!(cache.get(&k1, 200).is_none());
        assert!(cache.get(&k2, 200).is_none());
        assert!(cache.get(&k3, 200).is_some());
    }

    #[test]
    fn test_invalidate_key_specific() {
        let mut cache = make_cache(100, 60_000);
        let k = key("User", "name");
        cache.insert(k.clone(), "Alice".to_string(), 100);
        assert!(cache.invalidate_key(&k));
        assert!(cache.get(&k, 200).is_none());
    }

    #[test]
    fn test_invalidate_key_not_found() {
        let mut cache: FieldResolverCache<String> = make_cache(100, 60_000);
        let k = key("User", "name");
        assert!(!cache.invalidate_key(&k));
    }

    #[test]
    fn test_expire_removes_expired_entries() {
        let mut cache = make_cache(100, 500);
        let k1 = key("A", "f1");
        let k2 = key("B", "f2");
        cache.insert(k1.clone(), "v1".to_string(), 0);
        cache.insert(k2.clone(), "v2".to_string(), 1000);

        // Expire at time 1000 — k1 (created at 0, ttl 500) should be expired
        let count = cache.expire(1000);
        assert_eq!(count, 1);
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_clear_empties_cache() {
        let mut cache = make_cache(100, 60_000);
        cache.insert(key("A", "f"), "v".to_string(), 100);
        cache.insert(key("B", "f"), "v".to_string(), 100);
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let mut cache = make_cache(100, 60_000);
        let k = key("U", "f");
        cache.insert(k.clone(), "v".to_string(), 0);
        cache.get(&k, 100);
        cache.get(&k, 200);
        cache.get(&CacheKey::new("U", "other"), 300);
        // 2 hits, 1 miss
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_hit_rate_zero_when_no_accesses() {
        let cache: FieldResolverCache<String> = make_cache(100, 60_000);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_disabled_cache_always_misses() {
        let mut cache: FieldResolverCache<String> = FieldResolverCache::new(ResolverCacheConfig {
            max_entries: 100,
            default_ttl_ms: 60_000,
            enabled: false,
        });
        let k = key("U", "f");
        // Insert should be no-op when disabled
        cache.insert(k.clone(), "v".to_string(), 0);
        // Get should always miss when disabled
        assert_eq!(cache.get(&k, 100), None);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_hit_count_increments() {
        let mut cache = make_cache(100, 60_000);
        let k = key("U", "f");
        cache.insert(k.clone(), "v".to_string(), 0);
        cache.get(&k, 100);
        cache.get(&k, 200);
        let key_str = k.cache_key_str();
        let entry = cache.entries.get(&key_str).expect("should succeed");
        assert_eq!(entry.hit_count, 2);
    }

    #[test]
    fn test_cache_key_serialization_no_parent() {
        let k = CacheKey::new("User", "name").with_args(42);
        assert_eq!(k.cache_key_str(), "User:name:42");
    }

    #[test]
    fn test_cache_key_serialization_with_parent() {
        let k = CacheKey::new("User", "posts")
            .with_args(7)
            .with_parent("user-123");
        assert_eq!(k.cache_key_str(), "User:posts:7:user-123");
    }

    #[test]
    fn test_cache_key_no_args_no_parent() {
        let k = CacheKey::new("Query", "allUsers");
        assert_eq!(k.cache_key_str(), "Query:allUsers:0");
    }

    #[test]
    fn test_hits_and_misses_counters() {
        let mut cache = make_cache(100, 60_000);
        let k = key("T", "f");
        cache.get(&k, 100);
        cache.insert(k.clone(), "v".to_string(), 100);
        cache.get(&k, 200);
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_size_after_operations() {
        let mut cache = make_cache(100, 60_000);
        assert_eq!(cache.size(), 0);
        cache.insert(key("A", "f"), "v".to_string(), 0);
        assert_eq!(cache.size(), 1);
        cache.insert(key("B", "f"), "v".to_string(), 0);
        assert_eq!(cache.size(), 2);
        cache.invalidate_key(&key("A", "f"));
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_insert_overwrites_existing() {
        let mut cache = make_cache(100, 60_000);
        let k = key("U", "f");
        cache.insert(k.clone(), "old".to_string(), 0);
        cache.insert(k.clone(), "new".to_string(), 100);
        let result = cache.get(&k, 200);
        assert_eq!(result, Some("new".to_string()));
    }

    #[test]
    fn test_cache_key_with_args_differs_from_without() {
        let k1 = CacheKey::new("T", "f").with_args(1);
        let k2 = CacheKey::new("T", "f").with_args(2);
        assert_ne!(k1.cache_key_str(), k2.cache_key_str());
    }

    #[test]
    fn test_expire_no_entries() {
        let mut cache: FieldResolverCache<String> = make_cache(100, 60_000);
        let count = cache.expire(1000);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_invalidate_no_matching_entries() {
        let mut cache = make_cache(100, 60_000);
        cache.insert(key("A", "f"), "v".to_string(), 0);
        let removed = cache.invalidate("B", "f");
        assert_eq!(removed, 0);
        assert_eq!(cache.size(), 1);
    }
}
