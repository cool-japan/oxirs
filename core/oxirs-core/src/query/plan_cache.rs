//! Query plan caching for improved performance
//!
//! This module provides an LRU cache for compiled query plans with persistence support.

use crate::query::plan::ExecutionPlan;
use crate::OxirsError;
use lru::LruCache;
use scirs2_core::metrics::Counter;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Query plan cache with LRU eviction
///
/// Caches compiled execution plans to avoid repeated query compilation
/// and optimization overhead.
pub struct QueryPlanCache {
    /// LRU cache for execution plans
    cache: Arc<RwLock<LruCache<u64, CachedPlan>>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStatistics>>,
    /// Metrics counters
    hit_counter: Counter,
    miss_counter: Counter,
    eviction_counter: Counter,
    /// Cache configuration
    config: CacheConfig,
}

/// Cached execution plan with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPlan {
    /// The compiled execution plan
    pub plan: SerializablePlan,
    /// Query signature (hash)
    pub signature: u64,
    /// When the plan was cached
    pub cached_at_ms: u128,
    /// How many times this plan was accessed
    pub access_count: u64,
    /// Last access time
    pub last_accessed_ms: u128,
    /// Estimated execution cost
    pub estimated_cost: f64,
    /// Average actual execution time (milliseconds)
    pub avg_execution_time_ms: f64,
}

/// Serializable representation of execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializablePlan {
    /// Triple scan operation
    TripleScan { pattern_desc: String },
    /// Hash join operation
    HashJoin {
        left: Box<SerializablePlan>,
        right: Box<SerializablePlan>,
        join_vars: Vec<String>,
    },
    /// Filter operation
    Filter {
        input: Box<SerializablePlan>,
        expr_desc: String,
    },
    /// Projection operation
    Project {
        input: Box<SerializablePlan>,
        variables: Vec<String>,
    },
    /// Union operation
    Union {
        left: Box<SerializablePlan>,
        right: Box<SerializablePlan>,
    },
    /// Empty plan
    Empty,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of cached plans
    pub max_size: usize,
    /// Enable persistence to disk
    pub enable_persistence: bool,
    /// Path for persisted cache
    pub persistence_path: Option<String>,
    /// Time-to-live for cached plans (milliseconds)
    pub ttl_ms: Option<u128>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 1000,
            enable_persistence: false,
            persistence_path: None,
            ttl_ms: Some(3_600_000), // 1 hour
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Current cache size
    pub current_size: usize,
    /// Total plans cached
    pub total_cached: u64,
}

impl QueryPlanCache {
    /// Create a new query plan cache
    pub fn new(config: CacheConfig) -> Self {
        let capacity =
            NonZeroUsize::new(config.max_size).unwrap_or(NonZeroUsize::new(1000).unwrap());

        Self {
            cache: Arc::new(RwLock::new(LruCache::new(capacity))),
            stats: Arc::new(RwLock::new(CacheStatistics::default())),
            hit_counter: Counter::new("plan_cache.hits".to_string()),
            miss_counter: Counter::new("plan_cache.misses".to_string()),
            eviction_counter: Counter::new("plan_cache.evictions".to_string()),
            config,
        }
    }

    /// Get a cached plan for a query string
    pub fn get(&self, query: &str) -> Option<CachedPlan> {
        let signature = Self::compute_signature(query);

        let start = Instant::now();

        let result = {
            let mut cache = self.cache.write().ok()?;
            cache.get_mut(&signature).cloned()
        };

        let _elapsed = start.elapsed(); // Track elapsed time for future use

        if let Some(mut plan) = result {
            // Update access metrics
            plan.access_count += 1;
            plan.last_accessed_ms = Instant::now().elapsed().as_millis();

            // Check TTL
            if let Some(ttl) = self.config.ttl_ms {
                let age = plan.last_accessed_ms - plan.cached_at_ms;
                if age > ttl {
                    // Expired, remove from cache
                    self.remove(query);
                    self.record_miss();
                    return None;
                }
            }

            // Update cache with new access info
            if let Ok(mut cache) = self.cache.write() {
                cache.put(signature, plan.clone());
            }

            self.record_hit();
            Some(plan)
        } else {
            self.record_miss();
            None
        }
    }

    /// Put a compiled plan into the cache
    pub fn put(
        &self,
        query: &str,
        plan: ExecutionPlan,
        estimated_cost: f64,
    ) -> Result<(), OxirsError> {
        let signature = Self::compute_signature(query);
        let serializable = Self::convert_to_serializable(&plan);

        let cached_plan = CachedPlan {
            plan: serializable,
            signature,
            cached_at_ms: Instant::now().elapsed().as_millis(),
            access_count: 0,
            last_accessed_ms: Instant::now().elapsed().as_millis(),
            estimated_cost,
            avg_execution_time_ms: 0.0,
        };

        let mut cache = self
            .cache
            .write()
            .map_err(|e| OxirsError::Query(format!("Failed to write cache: {}", e)))?;

        // Check if we're evicting an entry
        let will_evict = cache.len() >= cache.cap().get();
        if will_evict {
            self.record_eviction();
        }

        cache.put(signature, cached_plan);

        // Update statistics
        let mut stats = self
            .stats
            .write()
            .map_err(|e| OxirsError::Query(format!("Failed to write stats: {}", e)))?;
        stats.current_size = cache.len();
        stats.total_cached += 1;

        Ok(())
    }

    /// Remove a cached plan
    pub fn remove(&self, query: &str) -> Option<CachedPlan> {
        let signature = Self::compute_signature(query);

        self.cache.write().ok()?.pop(&signature)
    }

    /// Clear all cached plans
    pub fn clear(&self) -> Result<(), OxirsError> {
        let mut cache = self
            .cache
            .write()
            .map_err(|e| OxirsError::Query(format!("Failed to write cache: {}", e)))?;
        cache.clear();

        let mut stats = self
            .stats
            .write()
            .map_err(|e| OxirsError::Query(format!("Failed to write stats: {}", e)))?;
        stats.current_size = 0;

        Ok(())
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        self.stats
            .read()
            .ok()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let stats = self.statistics();
        let total = stats.hits + stats.misses;
        if total == 0 {
            return 0.0;
        }
        stats.hits as f64 / total as f64
    }

    /// Persist cache to disk
    pub fn persist(&self) -> Result<(), OxirsError> {
        if !self.config.enable_persistence {
            return Ok(());
        }

        let path = self
            .config
            .persistence_path
            .as_ref()
            .ok_or_else(|| OxirsError::Io("No persistence path configured".to_string()))?;

        let cache = self
            .cache
            .read()
            .map_err(|e| OxirsError::Query(format!("Failed to read cache: {}", e)))?;

        // Convert cache to serializable format
        let entries: Vec<(u64, CachedPlan)> = cache.iter().map(|(k, v)| (*k, v.clone())).collect();

        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| OxirsError::Serialize(e.to_string()))?;

        std::fs::write(path, json).map_err(|e| OxirsError::Io(e.to_string()))?;

        tracing::info!("Persisted {} cached plans to {}", entries.len(), path);

        Ok(())
    }

    /// Load cache from disk
    pub fn load(&self) -> Result<(), OxirsError> {
        if !self.config.enable_persistence {
            return Ok(());
        }

        let path = self
            .config
            .persistence_path
            .as_ref()
            .ok_or_else(|| OxirsError::Io("No persistence path configured".to_string()))?;

        if !Path::new(path).exists() {
            return Ok(()); // No cached data to load
        }

        let json = std::fs::read_to_string(path).map_err(|e| OxirsError::Io(e.to_string()))?;

        let entries: Vec<(u64, CachedPlan)> =
            serde_json::from_str(&json).map_err(|e| OxirsError::Parse(e.to_string()))?;

        let mut cache = self
            .cache
            .write()
            .map_err(|e| OxirsError::Query(format!("Failed to write cache: {}", e)))?;

        for (sig, plan) in entries {
            cache.put(sig, plan);
        }

        tracing::info!("Loaded {} cached plans from {}", cache.len(), path);

        Ok(())
    }

    /// Update execution time for a cached plan
    pub fn update_execution_time(
        &self,
        query: &str,
        execution_time_ms: f64,
    ) -> Result<(), OxirsError> {
        let signature = Self::compute_signature(query);

        let mut cache = self
            .cache
            .write()
            .map_err(|e| OxirsError::Query(format!("Failed to write cache: {}", e)))?;

        if let Some(plan) = cache.get_mut(&signature) {
            // Update average execution time with exponential moving average
            let alpha = 0.3; // Weight for new measurement
            if plan.avg_execution_time_ms == 0.0 {
                plan.avg_execution_time_ms = execution_time_ms;
            } else {
                plan.avg_execution_time_ms =
                    alpha * execution_time_ms + (1.0 - alpha) * plan.avg_execution_time_ms;
            }
        }

        Ok(())
    }

    // Private helper methods

    fn compute_signature(query: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        hasher.finish()
    }

    fn convert_to_serializable(plan: &ExecutionPlan) -> SerializablePlan {
        match plan {
            ExecutionPlan::TripleScan { pattern } => SerializablePlan::TripleScan {
                pattern_desc: format!("{:?}", pattern),
            },
            ExecutionPlan::HashJoin {
                left,
                right,
                join_vars,
            } => SerializablePlan::HashJoin {
                left: Box::new(Self::convert_to_serializable(left)),
                right: Box::new(Self::convert_to_serializable(right)),
                join_vars: join_vars.iter().map(|v| format!("{:?}", v)).collect(),
            },
            ExecutionPlan::Filter { input, condition } => SerializablePlan::Filter {
                input: Box::new(Self::convert_to_serializable(input)),
                expr_desc: format!("{:?}", condition),
            },
            ExecutionPlan::Project { input, vars } => SerializablePlan::Project {
                input: Box::new(Self::convert_to_serializable(input)),
                variables: vars.iter().map(|v| format!("{:?}", v)).collect(),
            },
            ExecutionPlan::Union { left, right } => SerializablePlan::Union {
                left: Box::new(Self::convert_to_serializable(left)),
                right: Box::new(Self::convert_to_serializable(right)),
            },
            _ => SerializablePlan::Empty,
        }
    }

    fn record_hit(&self) {
        self.hit_counter.add(1);
        if let Ok(mut stats) = self.stats.write() {
            stats.hits += 1;
        }
    }

    fn record_miss(&self) {
        self.miss_counter.add(1);
        if let Ok(mut stats) = self.stats.write() {
            stats.misses += 1;
        }
    }

    fn record_eviction(&self) {
        self.eviction_counter.add(1);
        if let Ok(mut stats) = self.stats.write() {
            stats.evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let config = CacheConfig::default();
        let cache = QueryPlanCache::new(config);

        let stats = cache.statistics();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_put_get() {
        let config = CacheConfig::default();
        let cache = QueryPlanCache::new(config);

        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        let plan = ExecutionPlan::TripleScan {
            pattern: crate::model::pattern::TriplePattern::new(None, None, None),
        };

        cache.put(query, plan, 100.0).unwrap();

        let cached = cache.get(query);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().estimated_cost, 100.0);
    }

    #[test]
    fn test_cache_miss() {
        let config = CacheConfig::default();
        let cache = QueryPlanCache::new(config);

        let result = cache.get("SELECT ?s WHERE { ?s ?p ?o }");
        assert!(result.is_none());

        let stats = cache.statistics();
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_remove() {
        let config = CacheConfig::default();
        let cache = QueryPlanCache::new(config);

        let query = "SELECT ?s WHERE { ?s ?p ?o }";
        let plan = ExecutionPlan::TripleScan {
            pattern: crate::model::pattern::TriplePattern::new(None, None, None),
        };

        cache.put(query, plan, 50.0).unwrap();
        assert!(cache.get(query).is_some());

        cache.remove(query);
        assert!(cache.get(query).is_none());
    }

    #[test]
    fn test_cache_clear() {
        let config = CacheConfig::default();
        let cache = QueryPlanCache::new(config);

        let plan = ExecutionPlan::TripleScan {
            pattern: crate::model::pattern::TriplePattern::new(None, None, None),
        };

        cache.put("query1", plan.clone(), 50.0).unwrap();
        cache.put("query2", plan, 75.0).unwrap();

        cache.clear().unwrap();

        let stats = cache.statistics();
        assert_eq!(stats.current_size, 0);
    }

    #[test]
    fn test_hit_rate() {
        let config = CacheConfig::default();
        let cache = QueryPlanCache::new(config);

        let plan = ExecutionPlan::TripleScan {
            pattern: crate::model::pattern::TriplePattern::new(None, None, None),
        };

        let query = "SELECT * WHERE { ?s ?p ?o }";
        cache.put(query, plan, 100.0).unwrap();

        // One hit
        cache.get(query);
        // One miss
        cache.get("SELECT * WHERE { ?x ?y ?z }");

        let hit_rate = cache.hit_rate();
        assert!((hit_rate - 0.5).abs() < 0.01); // 50% hit rate
    }

    #[test]
    fn test_lru_eviction() {
        let config = CacheConfig {
            max_size: 2, // Small cache for testing
            ..Default::default()
        };

        let cache = QueryPlanCache::new(config);

        let plan = ExecutionPlan::TripleScan {
            pattern: crate::model::pattern::TriplePattern::new(None, None, None),
        };

        cache.put("query1", plan.clone(), 10.0).unwrap();
        cache.put("query2", plan.clone(), 20.0).unwrap();
        cache.put("query3", plan, 30.0).unwrap(); // Should evict query1

        assert!(cache.get("query1").is_none()); // Evicted
        assert!(cache.get("query2").is_some());
        assert!(cache.get("query3").is_some());
    }

    #[test]
    fn test_execution_time_update() {
        let config = CacheConfig::default();
        let cache = QueryPlanCache::new(config);

        let query = "SELECT ?s WHERE { ?s ?p ?o }";
        let plan = ExecutionPlan::TripleScan {
            pattern: crate::model::pattern::TriplePattern::new(None, None, None),
        };

        cache.put(query, plan, 100.0).unwrap();

        cache.update_execution_time(query, 50.0).unwrap();

        let cached = cache.get(query).unwrap();
        assert_eq!(cached.avg_execution_time_ms, 50.0);

        cache.update_execution_time(query, 70.0).unwrap();

        let cached2 = cache.get(query).unwrap();
        // Should be exponential moving average
        assert!(cached2.avg_execution_time_ms > 50.0 && cached2.avg_execution_time_ms < 70.0);
    }
}
