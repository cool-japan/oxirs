//! Query Result Caching
//!
//! This module provides caching for expensive query operations on SAMM models.
//! It wraps `ModelQuery` and caches results to avoid repeated computations.
//!
//! # Features
//!
//! - **Automatic Caching**: Transparently caches query results
//! - **LRU Eviction**: Least recently used results are evicted when cache is full
//! - **Cache Statistics**: Track hit rates and performance
//! - **Configurable Size**: Control memory usage
//!
//! # Examples
//!
//! ```rust
//! use oxirs_samm::query_cache::CachedModelQuery;
//! use oxirs_samm::metamodel::Aspect;
//!
//! # fn example(aspect: &Aspect) {
//! let mut query = CachedModelQuery::new(aspect, 100); // Cache up to 100 results
//!
//! // First call computes result
//! let metrics1 = query.complexity_metrics();
//!
//! // Second call returns cached result (much faster)
//! let metrics2 = query.complexity_metrics();
//!
//! // Check cache performance
//! let stats = query.cache_statistics();
//! println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);
//! # }
//! ```

use crate::metamodel::{Aspect, ModelElement, Property};
use crate::query::{ComplexityMetrics, Dependency, ModelQuery};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Cache key for query results
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
enum CacheKey {
    ComplexityMetrics,
    OptionalProperties,
    RequiredProperties,
    CollectionProperties,
    DependencyGraph,
    CircularDependencies,
    FindByType(String),
    FindByNamespace(String),
}

/// Cached model query wrapper
pub struct CachedModelQuery<'a> {
    query: ModelQuery<'a>,
    cache: Arc<RwLock<HashMap<CacheKey, CachedValue>>>,
    hits: Arc<RwLock<usize>>,
    misses: Arc<RwLock<usize>>,
    max_cache_size: usize,
}

/// Cached value with access tracking
#[derive(Clone)]
struct CachedValue {
    value: CachedResult,
    access_count: usize,
    last_accessed: std::time::Instant,
}

/// Union type for different cached results
#[derive(Clone)]
enum CachedResult {
    ComplexityMetrics(ComplexityMetrics),
    Properties(Vec<String>), // Store URNs instead of references
    Dependencies(Vec<Dependency>),
}

impl<'a> CachedModelQuery<'a> {
    /// Create a new cached query wrapper
    ///
    /// # Arguments
    ///
    /// * `aspect` - The aspect to query
    /// * `max_cache_size` - Maximum number of cached results
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxirs_samm::query_cache::CachedModelQuery;
    /// # use oxirs_samm::metamodel::Aspect;
    /// # fn example(aspect: &Aspect) {
    /// let query = CachedModelQuery::new(aspect, 50);
    /// # }
    /// ```
    pub fn new(aspect: &'a Aspect, max_cache_size: usize) -> Self {
        Self {
            query: ModelQuery::new(aspect),
            cache: Arc::new(RwLock::new(HashMap::new())),
            hits: Arc::new(RwLock::new(0)),
            misses: Arc::new(RwLock::new(0)),
            max_cache_size: max_cache_size.max(1),
        }
    }

    /// Get complexity metrics (cached)
    pub fn complexity_metrics(&mut self) -> ComplexityMetrics {
        let key = CacheKey::ComplexityMetrics;

        if let Some(CachedResult::ComplexityMetrics(metrics)) = self.get_cached(&key) {
            return metrics;
        }

        // Compute and cache
        let metrics = self.query.complexity_metrics();
        self.cache_result(key, CachedResult::ComplexityMetrics(metrics.clone()));
        metrics
    }

    /// Find optional properties (cached)
    pub fn find_optional_properties(&mut self) -> Vec<&Property> {
        let key = CacheKey::OptionalProperties;

        if let Some(CachedResult::Properties(urns)) = self.get_cached(&key) {
            // Convert URNs back to references
            return self
                .query
                .aspect()
                .properties()
                .iter()
                .filter(|p| urns.contains(&p.urn().to_string()))
                .collect();
        }

        // Compute and cache
        let props = self.query.find_optional_properties();
        let urns: Vec<String> = props.iter().map(|p| p.urn().to_string()).collect();
        self.cache_result(key, CachedResult::Properties(urns));
        props
    }

    /// Find required properties (cached)
    pub fn find_required_properties(&mut self) -> Vec<&Property> {
        let key = CacheKey::RequiredProperties;

        if let Some(CachedResult::Properties(urns)) = self.get_cached(&key) {
            return self
                .query
                .aspect()
                .properties()
                .iter()
                .filter(|p| urns.contains(&p.urn().to_string()))
                .collect();
        }

        let props = self.query.find_required_properties();
        let urns: Vec<String> = props.iter().map(|p| p.urn().to_string()).collect();
        self.cache_result(key, CachedResult::Properties(urns));
        props
    }

    /// Find collection properties (cached)
    pub fn find_properties_with_collection_characteristic(&mut self) -> Vec<&Property> {
        let key = CacheKey::CollectionProperties;

        if let Some(CachedResult::Properties(urns)) = self.get_cached(&key) {
            return self
                .query
                .aspect()
                .properties()
                .iter()
                .filter(|p| urns.contains(&p.urn().to_string()))
                .collect();
        }

        let props = self.query.find_properties_with_collection_characteristic();
        let urns: Vec<String> = props.iter().map(|p| p.urn().to_string()).collect();
        self.cache_result(key, CachedResult::Properties(urns));
        props
    }

    /// Build dependency graph (cached)
    pub fn build_dependency_graph(&mut self) -> Vec<Dependency> {
        let key = CacheKey::DependencyGraph;

        if let Some(CachedResult::Dependencies(deps)) = self.get_cached(&key) {
            return deps;
        }

        let deps = self.query.build_dependency_graph();
        self.cache_result(key, CachedResult::Dependencies(deps.clone()));
        deps
    }

    /// Detect circular dependencies (cached)
    pub fn detect_circular_dependencies(&mut self) -> Vec<Vec<String>> {
        let key = CacheKey::CircularDependencies;

        // For now, compute without caching (complex return type)
        self.query.detect_circular_dependencies()
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        let mut cache = self
            .cache
            .write()
            .expect("cache mutex should not be poisoned");
        cache.clear();
    }

    /// Get cache statistics
    pub fn cache_statistics(&self) -> CacheStatistics {
        let hits = *self.hits.read().expect("hits mutex should not be poisoned");
        let misses = *self
            .misses
            .read()
            .expect("misses mutex should not be poisoned");
        let total = hits + misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        };

        let cache = self
            .cache
            .read()
            .expect("cache mutex should not be poisoned");

        CacheStatistics {
            size: cache.len(),
            capacity: self.max_cache_size,
            hits,
            misses,
            hit_rate,
        }
    }

    /// Get cached value if it exists
    fn get_cached(&self, key: &CacheKey) -> Option<CachedResult> {
        let mut cache = self
            .cache
            .write()
            .expect("cache mutex should not be poisoned");

        if let Some(entry) = cache.get_mut(key) {
            entry.access_count += 1;
            entry.last_accessed = std::time::Instant::now();
            *self.hits.write().unwrap() += 1;
            Some(entry.value.clone())
        } else {
            *self.misses.write().unwrap() += 1;
            None
        }
    }

    /// Cache a result
    fn cache_result(&self, key: CacheKey, value: CachedResult) {
        let mut cache = self
            .cache
            .write()
            .expect("cache mutex should not be poisoned");

        // Evict LRU if at capacity
        if cache.len() >= self.max_cache_size && !cache.contains_key(&key) {
            if let Some((lru_key, _)) = cache
                .iter()
                .min_by_key(|(_, v)| v.last_accessed)
                .map(|(k, v)| (k.clone(), v.clone()))
            {
                cache.remove(&lru_key);
            }
        }

        cache.insert(
            key,
            CachedValue {
                value,
                access_count: 0,
                last_accessed: std::time::Instant::now(),
            },
        );
    }

    /// Get the underlying query (bypasses cache)
    pub fn query(&self) -> &ModelQuery<'a> {
        &self.query
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Current cache size
    pub size: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Total cache hits
    pub hits: usize,
    /// Total cache misses
    pub misses: usize,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

impl CacheStatistics {
    /// Get total accesses
    pub fn total_accesses(&self) -> usize {
        self.hits + self.misses
    }

    /// Get fill percentage
    pub fn fill_percentage(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.size as f64 / self.capacity as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Characteristic, CharacteristicKind};

    fn create_test_aspect() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let mut prop1 = Property::new("urn:samm:test:1.0.0#prop1".to_string());
        prop1.optional = false;

        let mut prop2 = Property::new("urn:samm:test:1.0.0#prop2".to_string());
        prop2.optional = true;

        let mut prop3 = Property::new("urn:samm:test:1.0.0#prop3".to_string());
        prop3.is_collection = true;
        // Add a Collection characteristic so it can be found by find_properties_with_collection_characteristic
        let collection_char = Characteristic::new(
            "urn:samm:test:1.0.0#CollectionChar".to_string(),
            CharacteristicKind::Collection {
                element_characteristic: None,
            },
        );
        prop3.characteristic = Some(collection_char);

        aspect.add_property(prop1);
        aspect.add_property(prop2);
        aspect.add_property(prop3);

        aspect
    }

    #[test]
    fn test_cached_complexity_metrics() {
        let aspect = create_test_aspect();
        let mut query = CachedModelQuery::new(&aspect, 10);

        // First call - cache miss
        let metrics1 = query.complexity_metrics();
        assert_eq!(metrics1.total_properties, 3);

        let stats1 = query.cache_statistics();
        assert_eq!(stats1.misses, 1);
        assert_eq!(stats1.hits, 0);

        // Second call - cache hit
        let metrics2 = query.complexity_metrics();
        assert_eq!(metrics2.total_properties, 3);

        let stats2 = query.cache_statistics();
        assert_eq!(stats2.misses, 1);
        assert_eq!(stats2.hits, 1);
        assert!((stats2.hit_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cached_optional_properties() {
        let aspect = create_test_aspect();
        let mut query = CachedModelQuery::new(&aspect, 10);

        let props1 = query.find_optional_properties();
        assert_eq!(props1.len(), 1);

        let stats1 = query.cache_statistics();
        assert_eq!(stats1.misses, 1);

        let props2 = query.find_optional_properties();
        assert_eq!(props2.len(), 1);

        let stats2 = query.cache_statistics();
        assert_eq!(stats2.hits, 1);
    }

    #[test]
    fn test_cached_required_properties() {
        let aspect = create_test_aspect();
        let mut query = CachedModelQuery::new(&aspect, 10);

        let props1 = query.find_required_properties();
        assert_eq!(props1.len(), 2);

        let props2 = query.find_required_properties();
        assert_eq!(props2.len(), 2);

        let stats = query.cache_statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_clear() {
        let aspect = create_test_aspect();
        let mut query = CachedModelQuery::new(&aspect, 10);

        query.complexity_metrics();
        query.complexity_metrics(); // Hit

        let stats_before = query.cache_statistics();
        assert_eq!(stats_before.size, 1);

        query.clear_cache();

        let stats_after = query.cache_statistics();
        assert_eq!(stats_after.size, 0);
        assert_eq!(stats_after.hits, 1); // Stats not cleared
    }

    #[test]
    fn test_cache_lru_eviction() {
        let aspect = create_test_aspect();
        let mut query = CachedModelQuery::new(&aspect, 2); // Small cache

        // Fill cache
        query.complexity_metrics();
        query.find_optional_properties();

        let stats1 = query.cache_statistics();
        assert_eq!(stats1.size, 2);

        // This should evict LRU
        query.find_required_properties();

        let stats2 = query.cache_statistics();
        assert_eq!(stats2.size, 2); // Still at capacity
    }

    #[test]
    fn test_cache_statistics() {
        let aspect = create_test_aspect();
        let mut query = CachedModelQuery::new(&aspect, 10);

        query.complexity_metrics();
        query.complexity_metrics();
        query.find_optional_properties();

        let stats = query.cache_statistics();
        assert_eq!(stats.total_accesses(), 3);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert!((stats.hit_rate - 0.333).abs() < 0.01);
        assert_eq!(stats.size, 2);
    }

    #[test]
    fn test_collection_properties_caching() {
        let aspect = create_test_aspect();
        let mut query = CachedModelQuery::new(&aspect, 10);

        let props1 = query.find_properties_with_collection_characteristic();
        assert_eq!(props1.len(), 1);

        let props2 = query.find_properties_with_collection_characteristic();
        assert_eq!(props2.len(), 1);

        let stats = query.cache_statistics();
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_dependency_graph_caching() {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        let mut prop = Property::new("urn:samm:test:1.0.0#prop1".to_string());
        let char = Characteristic::new(
            "urn:samm:test:1.0.0#Char1".to_string(),
            CharacteristicKind::Trait,
        );
        prop.characteristic = Some(char);
        aspect.add_property(prop);

        let mut query = CachedModelQuery::new(&aspect, 10);

        let deps1 = query.build_dependency_graph();
        assert!(!deps1.is_empty());

        let deps2 = query.build_dependency_graph();
        assert!(!deps2.is_empty());

        let stats = query.cache_statistics();
        assert_eq!(stats.hits, 1);
    }
}
