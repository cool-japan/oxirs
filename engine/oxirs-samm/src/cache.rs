//! LRU Cache for SAMM Model Elements
//!
//! This module provides an efficient Least Recently Used (LRU) cache for SAMM model elements.
//! The cache automatically evicts the least recently used items when it reaches capacity.
//!
//! # Use Cases
//!
//! - **Large model repositories**: Cache frequently accessed aspects
//! - **Web applications**: Reduce parsing overhead for repeated requests
//! - **Batch processing**: Reuse parsed models across multiple operations
//! - **Memory constraints**: Limit cache size while maintaining performance
//!
//! # Example
//!
//! ```rust
//! use oxirs_samm::cache::LruModelCache;
//! use oxirs_samm::metamodel::Aspect;
//! use std::sync::Arc;
//!
//! let mut cache = LruModelCache::new(100); // Cache up to 100 aspects
//!
//! let aspect = Aspect::new("urn:samm:org.example:1.0.0#MyAspect".to_string());
//! cache.put("my-aspect".to_string(), Arc::new(aspect));
//!
//! // Later retrieval
//! if let Some(cached_aspect) = cache.get("my-aspect") {
//!     println!("Cache hit!");
//! }
//!
//! // Check cache statistics
//! println!("Cache size: {}/{}", cache.len(), cache.capacity());
//! println!("Hit rate: {:.2}%", cache.hit_rate() * 100.0);
//! ```

use crate::metamodel::{Aspect, Characteristic, Entity, Operation, Property};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Entry in the LRU cache with access tracking
#[derive(Clone)]
struct CacheEntry<T> {
    value: Arc<T>,
    access_count: usize,
    last_accessed: std::time::Instant,
}

/// LRU (Least Recently Used) cache for model elements
///
/// Thread-safe cache with automatic eviction of least recently used items.
pub struct LruModelCache<T> {
    capacity: usize,
    entries: Arc<RwLock<HashMap<String, CacheEntry<T>>>>,
    access_order: Arc<RwLock<Vec<String>>>,
    hits: Arc<RwLock<usize>>,
    misses: Arc<RwLock<usize>>,
}

impl<T> LruModelCache<T>
where
    T: Clone,
{
    /// Create a new LRU cache with specified capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of items to cache
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_samm::cache::LruModelCache;
    /// use oxirs_samm::metamodel::Aspect;
    ///
    /// let cache: LruModelCache<Aspect> = LruModelCache::new(50);
    /// assert_eq!(cache.capacity(), 50);
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1), // Minimum capacity of 1
            entries: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(Vec::new())),
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Get an item from the cache
    ///
    /// Updates access statistics and LRU ordering.
    pub fn get(&self, key: &str) -> Option<Arc<T>> {
        let mut entries = self.entries.write().unwrap();

        if let Some(entry) = entries.get_mut(key) {
            // Update access statistics
            entry.access_count += 1;
            entry.last_accessed = std::time::Instant::now();

            // Update access order
            let mut access_order = self.access_order.write().unwrap();
            if let Some(pos) = access_order.iter().position(|k| k == key) {
                access_order.remove(pos);
            }
            access_order.push(key.to_string());

            // Record hit
            *self.hits.write().unwrap() += 1;

            Some(Arc::clone(&entry.value))
        } else {
            // Record miss
            *self.misses.write().unwrap() += 1;
            None
        }
    }

    /// Put an item into the cache
    ///
    /// If the cache is at capacity, evicts the least recently used item.
    pub fn put(&mut self, key: String, value: Arc<T>) {
        let mut entries = self.entries.write().unwrap();
        let mut access_order = self.access_order.write().unwrap();

        // Remove existing entry if present
        if entries.contains_key(&key) {
            if let Some(pos) = access_order.iter().position(|k| k == &key) {
                access_order.remove(pos);
            }
        }

        // Evict LRU entry if at capacity
        if entries.len() >= self.capacity && !entries.contains_key(&key) {
            if let Some(lru_key) = access_order.first() {
                let lru_key = lru_key.clone();
                entries.remove(&lru_key);
                access_order.remove(0);
            }
        }

        // Insert new entry
        let entry = CacheEntry {
            value,
            access_count: 0,
            last_accessed: std::time::Instant::now(),
        };

        entries.insert(key.clone(), entry);
        access_order.push(key);
    }

    /// Check if the cache contains a key
    pub fn contains(&self, key: &str) -> bool {
        let entries = self.entries.read().unwrap();
        entries.contains_key(key)
    }

    /// Remove an item from the cache
    pub fn remove(&mut self, key: &str) -> Option<Arc<T>> {
        let mut entries = self.entries.write().unwrap();
        let mut access_order = self.access_order.write().unwrap();

        if let Some(pos) = access_order.iter().position(|k| k == key) {
            access_order.remove(pos);
        }

        entries.remove(key).map(|entry| entry.value)
    }

    /// Clear all items from the cache
    pub fn clear(&mut self) {
        let mut entries = self.entries.write().unwrap();
        let mut access_order = self.access_order.write().unwrap();

        entries.clear();
        access_order.clear();
    }

    /// Get the number of items in the cache
    pub fn len(&self) -> usize {
        let entries = self.entries.read().unwrap();
        entries.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the cache capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Resize the cache capacity
    ///
    /// If the new capacity is smaller than current size, evicts LRU items.
    pub fn resize(&mut self, new_capacity: usize) {
        self.capacity = new_capacity.max(1);

        let mut entries = self.entries.write().unwrap();
        let mut access_order = self.access_order.write().unwrap();

        // Evict items if over capacity
        while entries.len() > self.capacity {
            if let Some(lru_key) = access_order.first() {
                let lru_key = lru_key.clone();
                entries.remove(&lru_key);
                access_order.remove(0);
            } else {
                break;
            }
        }
    }

    /// Get cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let hits = *self.hits.read().unwrap();
        let misses = *self.misses.read().unwrap();
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Get total number of cache hits
    pub fn hits(&self) -> usize {
        *self.hits.read().unwrap()
    }

    /// Get total number of cache misses
    pub fn misses(&self) -> usize {
        *self.misses.read().unwrap()
    }

    /// Reset cache statistics
    pub fn reset_statistics(&mut self) {
        *self.hits.write().unwrap() = 0;
        *self.misses.write().unwrap() = 0;
    }

    /// Get all keys in the cache (in LRU order)
    pub fn keys(&self) -> Vec<String> {
        let access_order = self.access_order.read().unwrap();
        access_order.clone()
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        CacheStatistics {
            size: self.len(),
            capacity: self.capacity,
            hits: self.hits(),
            misses: self.misses(),
            hit_rate: self.hit_rate(),
        }
    }
}

impl<T> Clone for LruModelCache<T> {
    fn clone(&self) -> Self {
        Self {
            capacity: self.capacity,
            entries: Arc::clone(&self.entries),
            access_order: Arc::clone(&self.access_order),
            hits: Arc::clone(&self.hits),
            misses: Arc::clone(&self.misses),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Current number of items
    pub size: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Total hits
    pub hits: usize,
    /// Total misses
    pub misses: usize,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

impl CacheStatistics {
    /// Get the fill percentage (0.0 to 100.0)
    pub fn fill_percentage(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.size as f64 / self.capacity as f64) * 100.0
        }
    }

    /// Get total accesses
    pub fn total_accesses(&self) -> usize {
        self.hits + self.misses
    }
}

// Type aliases for common cache types
/// LRU cache for Aspect models
pub type AspectCache = LruModelCache<Aspect>;

/// LRU cache for Property elements
pub type PropertyCache = LruModelCache<Property>;

/// LRU cache for Characteristic elements
pub type CharacteristicCache = LruModelCache<Characteristic>;

/// LRU cache for Entity elements
pub type EntityCache = LruModelCache<Entity>;

/// LRU cache for Operation elements
pub type OperationCache = LruModelCache<Operation>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::ElementMetadata;

    #[test]
    fn test_cache_creation() {
        let cache: LruModelCache<Aspect> = LruModelCache::new(10);
        assert_eq!(cache.capacity(), 10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_put_and_get() {
        let mut cache = LruModelCache::new(5);
        let aspect = Arc::new(Aspect::new("urn:test:1.0.0#Test".to_string()));

        cache.put("test".to_string(), Arc::clone(&aspect));
        assert_eq!(cache.len(), 1);

        let retrieved = cache.get("test");
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = LruModelCache::new(3);

        // Fill cache to capacity
        cache.put(
            "a".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#A".to_string())),
        );
        cache.put(
            "b".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#B".to_string())),
        );
        cache.put(
            "c".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#C".to_string())),
        );

        assert_eq!(cache.len(), 3);

        // Add one more - should evict "a" (least recently used)
        cache.put(
            "d".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#D".to_string())),
        );

        assert_eq!(cache.len(), 3);
        assert!(!cache.contains("a")); // "a" was evicted
        assert!(cache.contains("b"));
        assert!(cache.contains("c"));
        assert!(cache.contains("d"));
    }

    #[test]
    fn test_lru_access_updates() {
        let mut cache = LruModelCache::new(3);

        cache.put(
            "a".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#A".to_string())),
        );
        cache.put(
            "b".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#B".to_string())),
        );
        cache.put(
            "c".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#C".to_string())),
        );

        // Access "a" to make it recently used
        cache.get("a");

        // Add "d" - should evict "b" (now LRU)
        cache.put(
            "d".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#D".to_string())),
        );

        assert!(cache.contains("a")); // "a" was accessed, not evicted
        assert!(!cache.contains("b")); // "b" was evicted
        assert!(cache.contains("c"));
        assert!(cache.contains("d"));
    }

    #[test]
    fn test_remove() {
        let mut cache = LruModelCache::new(5);
        cache.put(
            "test".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#Test".to_string())),
        );

        assert_eq!(cache.len(), 1);
        let removed = cache.remove("test");
        assert!(removed.is_some());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut cache = LruModelCache::new(5);
        cache.put(
            "a".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#A".to_string())),
        );
        cache.put(
            "b".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#B".to_string())),
        );

        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = LruModelCache::new(5);
        cache.put(
            "test".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#Test".to_string())),
        );

        // 2 hits
        cache.get("test");
        cache.get("test");

        // 1 miss
        cache.get("nonexistent");

        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
        assert!((cache.hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_resize() {
        let mut cache = LruModelCache::new(5);

        // Fill with 5 items
        for i in 0..5 {
            cache.put(
                format!("item{}", i),
                Arc::new(Aspect::new(format!("urn:test:1.0.0#Item{}", i))),
            );
        }

        assert_eq!(cache.len(), 5);

        // Resize to 3 - should evict 2 LRU items
        cache.resize(3);
        assert_eq!(cache.capacity(), 3);
        assert_eq!(cache.len(), 3);

        // Oldest items should be evicted
        assert!(!cache.contains("item0"));
        assert!(!cache.contains("item1"));
        assert!(cache.contains("item2"));
        assert!(cache.contains("item3"));
        assert!(cache.contains("item4"));
    }

    #[test]
    fn test_statistics() {
        let mut cache = LruModelCache::new(10);
        cache.put(
            "test".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#Test".to_string())),
        );

        cache.get("test");
        cache.get("test");
        cache.get("nonexistent");

        let stats = cache.statistics();
        assert_eq!(stats.size, 1);
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_accesses(), 3);
        assert_eq!(stats.fill_percentage(), 10.0);
    }

    #[test]
    fn test_keys() {
        let mut cache = LruModelCache::new(5);
        cache.put(
            "a".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#A".to_string())),
        );
        cache.put(
            "b".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#B".to_string())),
        );
        cache.put(
            "c".to_string(),
            Arc::new(Aspect::new("urn:test:1.0.0#C".to_string())),
        );

        let keys = cache.keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
        assert!(keys.contains(&"c".to_string()));
    }
}
