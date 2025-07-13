//! Validation result caching for SHACL validation
//!
//! This module provides intelligent caching capabilities for validation results,
//! including LRU eviction, TTL support, and cache statistics.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::types::{CacheStatistics, CachedValidationResult, ValidationResult};

/// Validation result cache with LRU eviction and TTL support
#[derive(Debug)]
pub struct ValidationCache {
    cache: HashMap<String, CachedValidationResult>,
    size_limit: usize,
    access_order: VecDeque<String>,
    hit_count: u64,
    miss_count: u64,
    eviction_count: u64,
}

impl ValidationCache {
    pub fn new(size_limit: usize) -> Self {
        Self {
            cache: HashMap::new(),
            size_limit,
            access_order: VecDeque::new(),
            hit_count: 0,
            miss_count: 0,
            eviction_count: 0,
        }
    }

    /// Get a cached validation result
    pub fn get(&mut self, key: &str) -> Option<&ValidationResult> {
        // Check if key exists first
        if !self.cache.contains_key(key) {
            self.miss_count += 1;
            return None;
        }

        // Check TTL by cloning the timestamp (to avoid borrowing conflicts)
        let is_expired = {
            let cached = self.cache.get(key)?;
            cached.created_at.elapsed() > cached.ttl
        };

        // Remove expired entry
        if is_expired {
            self.cache.remove(key);
            self.miss_count += 1;
            return None;
        }

        // Update access order
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        self.access_order.push_back(key.to_string());

        self.hit_count += 1;

        // Return the cached value
        self.cache.get(key).map(|cached| &cached.result)
    }

    /// Put a validation result in the cache
    pub fn put(&mut self, key: String, result: CachedValidationResult) {
        // Evict if at capacity
        while self.cache.len() >= self.size_limit && !self.cache.is_empty() {
            if let Some(old_key) = self.access_order.pop_front() {
                self.cache.remove(&old_key);
                self.eviction_count += 1;
            } else {
                break; // Safety: avoid infinite loop
            }
        }

        // Remove existing entry if it exists
        if self.cache.contains_key(&key) {
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                self.access_order.remove(pos);
            }
        }

        self.cache.insert(key.clone(), result);
        self.access_order.push_back(key);
    }

    /// Get cache hit rate
    pub fn get_hit_rate(&self) -> f64 {
        let total_requests = self.hit_count + self.miss_count;
        if total_requests > 0 {
            self.hit_count as f64 / total_requests as f64
        } else {
            0.0
        }
    }

    /// Get detailed cache statistics
    pub fn get_cache_stats(&self) -> CacheStatistics {
        let total_entries = self.cache.len();
        let total_requests = self.hit_count + self.miss_count;
        let hit_rate = self.get_hit_rate();
        let miss_rate = 1.0 - hit_rate;

        // Calculate memory usage estimation
        let estimated_memory_mb = total_entries as f64 * 0.1; // Estimate 0.1MB per cache entry

        CacheStatistics {
            total_requests,
            cache_hits: self.hit_count,
            cache_misses: self.miss_count,
            hit_rate,
            memory_usage_mb: estimated_memory_mb,
            eviction_count: self.eviction_count,
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        self.hit_count = 0;
        self.miss_count = 0;
        self.eviction_count = 0;
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Remove expired entries
    pub fn cleanup_expired(&mut self) {
        let mut expired_keys = Vec::new();

        for (key, cached) in &self.cache {
            if cached.created_at.elapsed() > cached.ttl {
                expired_keys.push(key.clone());
            }
        }

        for key in expired_keys {
            self.cache.remove(&key);
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                self.access_order.remove(pos);
            }
        }
    }

    /// Check if a key exists and is not expired
    pub fn contains_key(&mut self, key: &str) -> bool {
        if let Some(cached) = self.cache.get(key) {
            if cached.created_at.elapsed() <= cached.ttl {
                return true;
            } else {
                // Remove expired entry
                self.cache.remove(key);
                if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                    self.access_order.remove(pos);
                }
            }
        }
        false
    }

    /// Get all cache keys (excluding expired ones)
    pub fn keys(&mut self) -> Vec<String> {
        self.cleanup_expired();
        self.cache.keys().cloned().collect()
    }

    /// Get cache efficiency metrics
    pub fn get_efficiency_metrics(&self) -> CacheEfficiencyMetrics {
        let total_requests = self.hit_count + self.miss_count;
        let memory_efficiency = if self.size_limit > 0 {
            self.cache.len() as f64 / self.size_limit as f64
        } else {
            0.0
        };

        CacheEfficiencyMetrics {
            hit_rate: self.get_hit_rate(),
            memory_efficiency,
            eviction_rate: if total_requests > 0 {
                self.eviction_count as f64 / total_requests as f64
            } else {
                0.0
            },
            average_ttl_utilization: self.calculate_average_ttl_utilization(),
        }
    }

    /// Calculate average TTL utilization
    fn calculate_average_ttl_utilization(&self) -> f64 {
        if self.cache.is_empty() {
            return 0.0;
        }

        let mut total_utilization = 0.0;
        for cached in self.cache.values() {
            let elapsed = cached.created_at.elapsed();
            let utilization = if cached.ttl.as_secs() > 0 {
                (elapsed.as_secs_f64() / cached.ttl.as_secs_f64()).min(1.0)
            } else {
                0.0
            };
            total_utilization += utilization;
        }

        total_utilization / self.cache.len() as f64
    }

    /// Set cache size limit
    pub fn set_size_limit(&mut self, new_limit: usize) {
        self.size_limit = new_limit;

        // Evict entries if we're now over the limit
        while self.cache.len() > self.size_limit && !self.cache.is_empty() {
            if let Some(old_key) = self.access_order.pop_front() {
                self.cache.remove(&old_key);
                self.eviction_count += 1;
            } else {
                break;
            }
        }
    }
}

/// Cache efficiency metrics
#[derive(Debug, Clone)]
pub struct CacheEfficiencyMetrics {
    pub hit_rate: f64,
    pub memory_efficiency: f64,
    pub eviction_rate: f64,
    pub average_ttl_utilization: f64,
}

impl Default for ValidationCache {
    fn default() -> Self {
        Self::new(10000) // Default size limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_cache_basic_operations() {
        let mut cache = ValidationCache::new(2);

        let result = ValidationResult {
            is_valid: true,
            violations: vec![],
            execution_time: Duration::from_millis(100),
            memory_usage_mb: 10.0,
            constraint_results: HashMap::new(),
        };

        let cached_result = CachedValidationResult {
            key: "test_key".to_string(),
            result: result.clone(),
            created_at: Instant::now(),
            ttl: Duration::from_secs(60),
        };

        // Test put and get
        cache.put("test_key".to_string(), cached_result);
        assert!(cache.contains_key("test_key"));

        let retrieved = cache.get("test_key");
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().is_valid);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = ValidationCache::new(2);

        let result = ValidationResult {
            is_valid: true,
            violations: vec![],
            execution_time: Duration::from_millis(100),
            memory_usage_mb: 10.0,
            constraint_results: HashMap::new(),
        };

        // Add three items to a cache with limit 2
        for i in 0..3 {
            let cached_result = CachedValidationResult {
                key: format!("key_{}", i),
                result: result.clone(),
                created_at: Instant::now(),
                ttl: Duration::from_secs(60),
            };
            cache.put(format!("key_{}", i), cached_result);
        }

        // Cache should only have 2 items
        assert_eq!(cache.size(), 2);

        // First item should be evicted
        assert!(!cache.contains_key("key_0"));
        assert!(cache.contains_key("key_1"));
        assert!(cache.contains_key("key_2"));
    }
}
