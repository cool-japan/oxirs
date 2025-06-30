//! Caching functionality for validation operations

use indexmap::IndexMap;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::{Constraint, ConstraintComponentId, PropertyPath, ShapeId};
use oxirs_core::model::Term;

use super::{ConstraintCacheKey, ConstraintEvaluationResult};

/// Cache for constraint evaluation results
#[derive(Debug, Default)]
pub struct ConstraintCache {
    cache: RefCell<HashMap<ConstraintCacheKey, ConstraintEvaluationResult>>,
    hits: RefCell<usize>,
    misses: RefCell<usize>,
}

impl ConstraintCache {
    /// Create a new constraint cache
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a cached result if available
    pub fn get(&self, key: &ConstraintCacheKey) -> Option<ConstraintEvaluationResult> {
        let cache = self.cache.borrow();
        if let Some(result) = cache.get(key) {
            *self.hits.borrow_mut() += 1;
            Some(result.clone())
        } else {
            *self.misses.borrow_mut() += 1;
            None
        }
    }

    /// Insert a result into the cache
    pub fn insert(&self, key: ConstraintCacheKey, result: ConstraintEvaluationResult) {
        let mut cache = self.cache.borrow_mut();
        cache.insert(key, result);
    }

    /// Clear all cached results
    pub fn clear(&self) {
        let mut cache = self.cache.borrow_mut();
        cache.clear();
        *self.hits.borrow_mut() = 0;
        *self.misses.borrow_mut() = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize) {
        (*self.hits.borrow(), *self.misses.borrow())
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = *self.hits.borrow();
        let misses = *self.misses.borrow();
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Get the number of cached entries
    pub fn size(&self) -> usize {
        self.cache.borrow().len()
    }

    /// Check if the cache contains a specific key
    pub fn contains(&self, key: &ConstraintCacheKey) -> bool {
        self.cache.borrow().contains_key(key)
    }

    /// Remove a specific entry from the cache
    pub fn remove(&self, key: &ConstraintCacheKey) -> Option<ConstraintEvaluationResult> {
        self.cache.borrow_mut().remove(key)
    }
}

/// Cache for resolved inherited constraints
#[derive(Debug, Default)]
pub struct InheritanceCache {
    cache: RefCell<HashMap<ShapeId, IndexMap<ConstraintComponentId, Constraint>>>,
}

impl InheritanceCache {
    /// Create a new inheritance cache
    pub fn new() -> Self {
        Self::default()
    }

    /// Get cached inherited constraints for a shape
    pub fn get(&self, shape_id: &ShapeId) -> Option<IndexMap<ConstraintComponentId, Constraint>> {
        let cache = self.cache.borrow();
        cache.get(shape_id).cloned()
    }

    /// Cache inherited constraints for a shape
    pub fn insert(
        &self,
        shape_id: ShapeId,
        constraints: IndexMap<ConstraintComponentId, Constraint>,
    ) {
        let mut cache = self.cache.borrow_mut();
        cache.insert(shape_id, constraints);
    }

    /// Clear all cached inherited constraints
    pub fn clear(&self) {
        let mut cache = self.cache.borrow_mut();
        cache.clear();
    }

    /// Get the number of cached entries
    pub fn size(&self) -> usize {
        self.cache.borrow().len()
    }

    /// Check if the cache contains a specific shape
    pub fn contains(&self, shape_id: &ShapeId) -> bool {
        self.cache.borrow().contains_key(shape_id)
    }

    /// Remove a specific entry from the cache
    pub fn remove(
        &self,
        shape_id: &ShapeId,
    ) -> Option<IndexMap<ConstraintComponentId, Constraint>> {
        self.cache.borrow_mut().remove(shape_id)
    }
}

/// Configuration for caching behavior
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of constraint results to cache
    pub max_constraint_cache_size: usize,
    /// Maximum number of inheritance results to cache
    pub max_inheritance_cache_size: usize,
    /// Whether to enable constraint result caching
    pub enable_constraint_cache: bool,
    /// Whether to enable inheritance caching
    pub enable_inheritance_cache: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_constraint_cache_size: 10000,
            max_inheritance_cache_size: 1000,
            enable_constraint_cache: true,
            enable_inheritance_cache: true,
        }
    }
}

/// Manages all caching for the validation engine
#[derive(Debug)]
pub struct CacheManager {
    constraint_cache: ConstraintCache,
    inheritance_cache: InheritanceCache,
    config: CacheConfig,
}

impl CacheManager {
    /// Create a new cache manager with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new cache manager with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            constraint_cache: ConstraintCache::new(),
            inheritance_cache: InheritanceCache::new(),
            config,
        }
    }

    /// Get the constraint cache
    pub fn constraint_cache(&self) -> &ConstraintCache {
        &self.constraint_cache
    }

    /// Get the inheritance cache
    pub fn inheritance_cache(&self) -> &InheritanceCache {
        &self.inheritance_cache
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        if self.config.enable_constraint_cache {
            self.constraint_cache.clear();
        }
        if self.config.enable_inheritance_cache {
            self.inheritance_cache.clear();
        }
    }

    /// Get overall cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        let (constraint_hits, constraint_misses) = self.constraint_cache.stats();

        CacheStatistics {
            constraint_cache_size: self.constraint_cache.size(),
            inheritance_cache_size: self.inheritance_cache.size(),
            constraint_cache_hits: constraint_hits,
            constraint_cache_misses: constraint_misses,
            constraint_cache_hit_rate: self.constraint_cache.hit_rate(),
        }
    }

    /// Get the cache configuration
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about cache performance
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub constraint_cache_size: usize,
    pub inheritance_cache_size: usize,
    pub constraint_cache_hits: usize,
    pub constraint_cache_misses: usize,
    pub constraint_cache_hit_rate: f64,
}

impl CacheStatistics {
    /// Format cache statistics as a human-readable string
    pub fn summary(&self) -> String {
        format!(
            "Cache Stats: constraint_size={}, inheritance_size={}, hit_rate={:.1}% ({}/{} hits)",
            self.constraint_cache_size,
            self.inheritance_cache_size,
            self.constraint_cache_hit_rate * 100.0,
            self.constraint_cache_hits,
            self.constraint_cache_hits + self.constraint_cache_misses
        )
    }
}
